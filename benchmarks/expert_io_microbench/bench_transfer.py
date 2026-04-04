from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportPrivateLocalImportUsage=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false, reportUnusedParameter=false, reportAttributeAccessIssue=false, reportImplicitStringConcatenation=false
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.expert_io_microbench.stats import TimingCollector

DEFAULT_WARMUP = 2
DEFAULT_ITERS = 10
TRANSFER_STAGES = ("disk_to_cpu", "cpu_to_gpu")
SYNC_STAGE = "sync_wait"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure transfer timing and sync overhead via real MoE inference."
    )
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument(
        "--offload-dir",
        required=True,
        help="Directory used for MoE expert offload storage",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help="Warmup iterations before measurement",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=DEFAULT_ITERS,
        help="Measured iterations",
    )
    parser.add_argument(
        "--output-json",
        default="transfer_results.json",
        help="Path to write benchmark output JSON",
    )
    return parser.parse_args()


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
    tokenizer: Any, target_length: int = 128
) -> torch.Tensor:
    base_text = (
        "MoE-Infinity transfer timing benchmark prompt. "
        "Use deterministic content for stable measurements."
    )
    encoded = tokenizer.encode(base_text, add_special_tokens=False)
    prompt_ids = _repeat_to_length(encoded, target_length)
    return torch.tensor([prompt_ids], dtype=torch.long, device="cuda")


def load_model_and_tokenizer(
    model_name: str, offload_dir: str
) -> tuple[Any, Any]:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError(f"transformers import failed: {exc}") from exc

    try:
        from moe_infinity import MoE
    except Exception as exc:
        raise RuntimeError(f"moe_infinity import failed: {exc}") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = {
        "offload_path": offload_dir,
        "device_memory_ratio": 0.75,
    }
    model = MoE(model_name, config)
    return model, tokenizer


def setup_profiler() -> Any:
    os.environ["MOE_INFINITY_PROFILE_IO"] = "1"
    from moe_infinity.profiling.io_profiler import IOProfiler

    profiler = IOProfiler.instance()
    profiler.enabled = True
    profiler.reset()
    return profiler


def _snapshot_events(profiler: Any) -> list[dict[str, object]]:
    lock = getattr(profiler, "_lock", None)
    events = getattr(profiler, "_events", None)
    if lock is None or events is None:
        return []
    with lock:
        return list(events)


def run_one_iteration(
    model: Any, tokenizer: Any, profiler: Any
) -> tuple[int, list[dict[str, object]]]:
    input_ids = build_prompt_input_ids(tokenizer)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    profiler.reset()
    start_ns = time.perf_counter_ns()
    _ = model.generate(
        input_ids,
        max_new_tokens=1,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    step_ns = time.perf_counter_ns() - start_ns
    events = _snapshot_events(profiler)
    return step_ns, events


def _safe_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def summarize(
    events: list[dict[str, object]],
    total_step_ns: int,
) -> dict[str, Any]:
    collector = TimingCollector()
    sync_total_ns = 0
    sync_event_count = 0

    for event in events:
        stage = str(event.get("stage", ""))
        duration_ns = max(_safe_int(event.get("dur_ns")), 0)
        bytes_transferred = max(_safe_int(event.get("bytes")), 0)

        if stage in TRANSFER_STAGES:
            collector.record_transfer(stage, duration_ns, bytes_transferred)
        if stage == SYNC_STAGE:
            sync_total_ns += duration_ns
            sync_event_count += 1

    disk_stats = dict(collector.get_transfer_stats("disk_to_cpu"))
    cpu_stats = dict(collector.get_transfer_stats("cpu_to_gpu"))

    sync_pct = (
        0.0
        if total_step_ns <= 0
        else (float(sync_total_ns) / float(total_step_ns)) * 100.0
    )
    return {
        "disk_to_cpu": disk_stats,
        "cpu_to_gpu": cpu_stats,
        "sync_overhead": {
            "total_sync_ms": sync_total_ns / 1_000_000.0,
            "sync_pct_of_step": sync_pct,
            "sync_event_count": sync_event_count,
        },
    }


def main() -> int:
    args = parse_args()
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.iters <= 0:
        raise ValueError("--iters must be > 0")

    env = environment_info()
    output_path = Path(args.output_json)
    print("=== MoE-Infinity Transfer Timing Microbenchmark ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"CUDA available: {env['cuda_available']}")

    if not env["cuda_available"]:
        payload = {
            "disk_to_cpu": dict(
                TimingCollector().get_transfer_stats("disk_to_cpu")
            ),
            "cpu_to_gpu": dict(
                TimingCollector().get_transfer_stats("cpu_to_gpu")
            ),
            "sync_overhead": {
                "total_sync_ms": 0.0,
                "sync_pct_of_step": 0.0,
                "sync_event_count": 0,
            },
            "status": "BLOCKED",
            "reason": "No CUDA device",
        }
        write_json(output_path, payload)
        return 0

    try:
        model, tokenizer = load_model_and_tokenizer(
            args.model, args.offload_dir
        )
        profiler = setup_profiler()
    except Exception as exc:
        payload = {
            "disk_to_cpu": dict(
                TimingCollector().get_transfer_stats("disk_to_cpu")
            ),
            "cpu_to_gpu": dict(
                TimingCollector().get_transfer_stats("cpu_to_gpu")
            ),
            "sync_overhead": {
                "total_sync_ms": 0.0,
                "sync_pct_of_step": 0.0,
                "sync_event_count": 0,
            },
            "status": "BLOCKED",
            "reason": f"{type(exc).__name__}: {exc}",
        }
        write_json(output_path, payload)
        return 0

    for _ in range(args.warmup):
        _ = run_one_iteration(model, tokenizer, profiler)

    all_events: list[dict[str, object]] = []
    total_step_ns = 0
    for _ in range(args.iters):
        step_ns, events = run_one_iteration(model, tokenizer, profiler)
        total_step_ns += step_ns
        all_events.extend(events)

    payload = summarize(all_events, total_step_ns)
    write_json(output_path, payload)
    return 0


if __name__ == "__main__":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    raise SystemExit(main())
