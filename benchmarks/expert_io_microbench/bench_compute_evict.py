from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportPrivateLocalImportUsage=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false, reportUnusedParameter=false, reportAttributeAccessIssue=false, reportImplicitStringConcatenation=false
import argparse
import json
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Callable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.expert_io_microbench.stats import TimingCollector

DEFAULT_WARMUP = 10
DEFAULT_ITERS = 100
DEFAULT_MAX_NEW_TOKENS = 1
DEFAULT_DEVICE_MEMORY_RATIO = 0.3
TARGET_STAGES = ("expert_compute", "eviction", "queue_coordination")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark expert compute, eviction, and queue coordination "
            "via IOProfiler events."
        )
    )
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument(
        "--offload-dir",
        required=True,
        help="Directory used for MoE expert offload storage",
    )
    parser.add_argument(
        "--device-memory-ratio",
        type=float,
        default=DEFAULT_DEVICE_MEMORY_RATIO,
        help=(
            "Fraction of GPU memory used for expert cache. "
            "Default 0.3 to naturally increase eviction pressure."
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help="Warmup generate iterations (not measured)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=DEFAULT_ITERS,
        help="Measured generate iterations",
    )
    parser.add_argument(
        "--output-json",
        default="compute_evict_microbench_results.json",
        help="Path to write benchmark JSON",
    )
    parser.add_argument(
        "--host-only",
        action="store_true",
        help=(
            "Copy offload weights to /dev/shm (tmpfs) to eliminate disk I/O. "
            "Measures pure PCIe+sync overhead without disk reads. "
            "Requires sufficient /dev/shm space (use Docker --shm-size=32g)."
        ),
    )
    return parser.parse_args()


def _directory_size_bytes(root: str) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                total += os.path.getsize(file_path)
            except OSError:
                continue
    return total


def _is_under_dev_shm(path: str) -> bool:
    try:
        resolved = os.path.realpath(path)
        return os.path.commonpath([resolved, "/dev/shm"]) == "/dev/shm"
    except (OSError, ValueError):
        return False


def setup_offload_dir(
    args: argparse.Namespace,
) -> tuple[str, str, Callable[[], None]]:
    if not args.host_only:
        return args.offload_dir, "disk", lambda: None

    if not os.path.exists(args.offload_dir):
        print(
            f"WARNING: offload dir does not exist: {args.offload_dir}",
            file=sys.stderr,
        )
        print(
            "Hint: provide a valid --offload-dir before enabling --host-only",
            file=sys.stderr,
        )
        raise RuntimeError(f"offload dir does not exist: {args.offload_dir}")

    if _is_under_dev_shm(args.offload_dir):
        print(
            f"Using existing tmpfs offload dir: {args.offload_dir}",
            flush=True,
        )
        return args.offload_dir, "host-only", lambda: None

    src_size = _directory_size_bytes(args.offload_dir)
    shm_stat = shutil.disk_usage("/dev/shm")
    required_bytes = int(src_size * 1.1)
    if shm_stat.free < required_bytes:
        print(
            "WARNING: /dev/shm has "
            f"{shm_stat.free / 1e9:.1f}GB free but offload needs {src_size / 1e9:.1f}GB",
            file=sys.stderr,
        )
        print(
            "Hint: docker run --shm-size=<size>g or increase host shm",
            file=sys.stderr,
        )
        raise RuntimeError("insufficient /dev/shm space for host-only mode")

    dst = tempfile.mkdtemp(dir="/dev/shm", prefix="moe_hostonly_")
    print(
        f"Copying {src_size / 1e9:.1f}GB offload -> {dst} (tmpfs)...",
        flush=True,
    )
    shutil.copytree(args.offload_dir, dst, dirs_exist_ok=True)
    print("Copy done. Running in host-only (RAM) mode.", flush=True)

    def cleanup() -> None:
        shutil.rmtree(dst, ignore_errors=True)

    return dst, "host-only", cleanup


def environment_info() -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count()
    info: dict[str, Any] = {
        "torch_version": getattr(torch, "__version__", "unknown"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
    }
    if cuda_available and cuda_device_count > 0:
        info["cuda_device_names"] = [
            torch.cuda.get_device_name(idx) for idx in range(cuda_device_count)
        ]
    return info


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _repeat_to_length(token_ids: list[int], target_length: int) -> list[int]:
    if not token_ids:
        return [0] * target_length
    output: list[int] = []
    while len(output) < target_length:
        output.extend(token_ids)
    return output[:target_length]


def build_prompt_input_ids(
    tokenizer: Any, target_length: int = 96
) -> torch.Tensor:
    base_text = (
        "MoE-Infinity expert compute + eviction microbenchmark prompt. "
        "Use deterministic text for repeatable IOProfiler timing."
    )
    encoded = tokenizer.encode(base_text, add_special_tokens=False)
    prompt_ids = _repeat_to_length(encoded, target_length)
    return torch.tensor([prompt_ids], dtype=torch.long, device="cuda")


def ensure_model_available_locally(model_name: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub import failed while checking local model cache"
        ) from exc

    try:
        _ = snapshot_download(repo_id=model_name, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            "Model is not available in local HuggingFace cache. "
            "Please pre-download it before running this benchmark "
            f"(model={model_name}). Original error: {type(exc).__name__}: {exc}"
        ) from exc


def load_model_and_tokenizer(
    model_name: str,
    offload_dir: str,
    device_memory_ratio: float,
) -> tuple[Any, Any]:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError(f"transformers import failed: {exc}") from exc

    try:
        from moe_infinity import MoE
    except Exception as exc:
        raise RuntimeError(f"moe_infinity import failed: {exc}") from exc

    ensure_model_available_locally(model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = {
        "offload_path": offload_dir,
        "device_memory_ratio": device_memory_ratio,
    }
    model = MoE(model_name, config)
    return model, tokenizer


def setup_profiler() -> Any:
    os.environ["MOE_INFINITY_PROFILE_IO"] = "1"
    os.environ["MOE_INFINITY_PROFILE_IO_SAMPLE"] = "1.0"
    from moe_infinity.profiling.io_profiler import IOProfiler

    profiler = IOProfiler.instance()
    profiler.enabled = True
    sample = getattr(profiler, "_sample", None)
    if isinstance(sample, (int, float)):
        setattr(profiler, "_sample", 1.0)
    profiler.reset()
    return profiler


def _snapshot_profiler_events(profiler: Any) -> list[dict[str, Any]]:
    lock = getattr(profiler, "_lock", None)
    events = getattr(profiler, "_events", None)
    if events is None:
        return []
    if lock is None:
        return [dict(event) for event in events if isinstance(event, dict)]

    with lock:
        return [dict(event) for event in events if isinstance(event, dict)]


def _safe_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _empty_stage_stats() -> dict[str, int | float | None]:
    return {
        "min_ns": None,
        "max_ns": None,
        "mean_ns": None,
        "p50_ns": None,
        "p95_ns": None,
        "p99_ns": None,
        "count": 0,
        "total_bytes": 0,
    }


def _summarize_components(
    events: list[dict[str, Any]],
) -> dict[str, dict[str, int | float | None]]:
    collector = TimingCollector()
    bytes_by_stage: dict[str, int] = {stage: 0 for stage in TARGET_STAGES}

    for event in events:
        stage_obj = event.get("stage")
        if not isinstance(stage_obj, str) or stage_obj not in TARGET_STAGES:
            continue

        dur_ns = _safe_int(event.get("dur_ns"))
        if dur_ns < 0:
            dur_ns = 0
        collector.record(stage_obj, dur_ns)
        bytes_by_stage[stage_obj] += max(_safe_int(event.get("bytes")), 0)

    raw_stats = collector.to_dict()
    stage_stats: dict[str, dict[str, int | float | None]] = {}
    for stage in TARGET_STAGES:
        base = dict(raw_stats.get(stage, _empty_stage_stats()))
        if "total_bytes" not in base:
            base["total_bytes"] = bytes_by_stage.get(stage, 0)
        else:
            base["total_bytes"] = bytes_by_stage.get(stage, 0)
        stage_stats[stage] = base
    return stage_stats


def run_benchmark(
    model: Any,
    tokenizer: Any,
    profiler: Any,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    input_ids = build_prompt_input_ids(tokenizer)

    for _ in range(warmup):
        _ = model.generate(
            input_ids,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    profiler.reset()

    for _ in range(iters):
        _ = model.generate(
            input_ids,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    events = _snapshot_profiler_events(profiler)
    components = _summarize_components(events)
    observed_stages = sorted(
        {
            str(event.get("stage"))
            for event in events
            if isinstance(event.get("stage"), str)
        }
    )
    non_empty_components = sum(
        1 for stage in TARGET_STAGES if int(components[stage]["count"] or 0) > 0
    )

    return {
        **components,
        "io_profiler_events": len(events),
        "observed_stages": observed_stages,
        "non_empty_components": non_empty_components,
    }


def _blocked_payload(
    *,
    reason: str,
    env: dict[str, Any],
    model: str,
    offload_dir: str,
    device_memory_ratio: float,
    warmup: int,
    iters: int,
    mode: str,
) -> dict[str, Any]:
    return {
        "status": "BLOCKED",
        "reason": reason,
        "environment": env,
        "requested_model": model,
        "offload_dir": offload_dir,
        "mode": mode,
        "io_mode": mode,
        "device_memory_ratio": device_memory_ratio,
        "warmup": warmup,
        "iters": iters,
        "expert_compute": _empty_stage_stats(),
        "eviction": _empty_stage_stats(),
        "queue_coordination": _empty_stage_stats(),
        "io_profiler_events": 0,
        "observed_stages": [],
        "non_empty_components": 0,
    }


def main() -> int:
    args = parse_args()
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.iters <= 0:
        raise ValueError("--iters must be > 0")
    if not (0.0 < args.device_memory_ratio <= 1.0):
        raise ValueError("--device-memory-ratio must be in (0, 1]")

    env = environment_info()
    output_path = Path(args.output_json)

    print("=== MoE-Infinity Expert Compute + Eviction Microbenchmark ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"CUDA available: {env['cuda_available']}")

    mode = "host-only" if args.host_only else "disk"
    actual_offload_dir = args.offload_dir
    cleanup: Callable[[], None] = lambda: None

    if not env["cuda_available"]:
        payload = _blocked_payload(
            reason="No CUDA device",
            env=env,
            model=args.model,
            offload_dir=args.offload_dir,
            device_memory_ratio=args.device_memory_ratio,
            warmup=args.warmup,
            iters=args.iters,
            mode=mode,
        )
        write_json(output_path, payload)
        return 0

    try:
        profiler = setup_profiler()
    except Exception as exc:
        payload = _blocked_payload(
            reason=f"IOProfiler init failed: {type(exc).__name__}: {exc}",
            env=env,
            model=args.model,
            offload_dir=args.offload_dir,
            device_memory_ratio=args.device_memory_ratio,
            warmup=args.warmup,
            iters=args.iters,
            mode=mode,
        )
        write_json(output_path, payload)
        return 0

    try:
        actual_offload_dir, mode, cleanup = setup_offload_dir(args)
        model, tokenizer = load_model_and_tokenizer(
            model_name=args.model,
            offload_dir=actual_offload_dir,
            device_memory_ratio=args.device_memory_ratio,
        )
    except Exception as exc:
        cleanup()
        payload = _blocked_payload(
            reason=f"{type(exc).__name__}: {exc}",
            env=env,
            model=args.model,
            offload_dir=args.offload_dir,
            device_memory_ratio=args.device_memory_ratio,
            warmup=args.warmup,
            iters=args.iters,
            mode=mode,
        )
        write_json(output_path, payload)
        return 0

    try:
        results = run_benchmark(
            model=model,
            tokenizer=tokenizer,
            profiler=profiler,
            warmup=args.warmup,
            iters=args.iters,
        )

        payload = {
            "status": "PASS",
            "environment": env,
            "requested_model": args.model,
            "offload_dir": actual_offload_dir,
            "source_offload_dir": args.offload_dir,
            "mode": mode,
            "io_mode": mode,
            "device_memory_ratio": args.device_memory_ratio,
            "warmup": args.warmup,
            "iters": args.iters,
            **results,
        }
        write_json(output_path, payload)
        return 0
    finally:
        cleanup()


if __name__ == "__main__":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    raise SystemExit(main())
