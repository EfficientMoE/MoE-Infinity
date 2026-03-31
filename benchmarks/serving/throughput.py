from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportPrivateLocalImportUsage=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false, reportUnusedParameter=false, reportAttributeAccessIssue=false, reportImplicitStringConcatenation=false
import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16, 32)
DEFAULT_MAX_NEW_TOKENS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure continuous-batching throughput (tokens/s)."
    )
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument(
        "--offload-dir",
        required=True,
        help="Directory used for MoE expert offload storage",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=50,
        help="Total requests to run per batch-size setting",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_BATCH_SIZES),
        help="Batch sizes to sweep",
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=128,
        help="Prompt length in tokens",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Generated tokens per request",
    )
    parser.add_argument(
        "--baseline-json",
        default="baseline_results.json",
        help="Optional baseline_performance.py output JSON for comparison",
    )
    parser.add_argument(
        "--output-json",
        default="throughput_results.json",
        help="Path to write the benchmark summary JSON",
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
    if target_length <= 0:
        raise ValueError(f"target_length must be > 0, got {target_length}")
    if not token_ids:
        return [0] * target_length

    output: list[int] = []
    while len(output) < target_length:
        output.extend(token_ids)
    return output[:target_length]


def build_prompt_input_ids(
    tokenizer: Any, target_length: int, batch_size: int
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    base_text = (
        "MoE-Infinity continuous batching throughput benchmark prompt. "
        "Keep this text deterministic for stable measurements."
    )
    encoded = tokenizer.encode(base_text, add_special_tokens=False)
    prompt_ids = _repeat_to_length(encoded, target_length)
    batch_input = [list(prompt_ids) for _ in range(batch_size)]
    return torch.tensor(batch_input, dtype=torch.long, device="cuda")


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


def run_one_batch(
    model: Any,
    tokenizer: Any,
    *,
    batch_size: int,
    prompt_length: int,
    max_new_tokens: int,
) -> tuple[int, float]:
    input_ids = build_prompt_input_ids(tokenizer, prompt_length, batch_size)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - start

    generated_tokens = max(int(output_ids.numel() - input_ids.numel()), 0)
    return generated_tokens, elapsed_s


def run_sweep(
    model: Any,
    tokenizer: Any,
    *,
    num_requests: int,
    batch_sizes: list[int],
    prompt_length: int,
    max_new_tokens: int,
) -> dict[str, float]:
    measurements: dict[str, float] = {}
    for batch_size in batch_sizes:
        remaining = num_requests
        total_generated_tokens = 0
        total_elapsed_s = 0.0

        while remaining > 0:
            current_batch_size = min(batch_size, remaining)
            generated_tokens, elapsed_s = run_one_batch(
                model,
                tokenizer,
                batch_size=current_batch_size,
                prompt_length=prompt_length,
                max_new_tokens=max_new_tokens,
            )
            total_generated_tokens += generated_tokens
            total_elapsed_s += elapsed_s
            remaining -= current_batch_size

        tokens_per_second = (
            0.0
            if total_elapsed_s <= 0
            else total_generated_tokens / total_elapsed_s
        )
        measurements[str(batch_size)] = tokens_per_second
    return measurements


def load_baseline(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _to_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def estimate_baseline_tokens_per_second(
    baseline_payload: dict[str, Any] | None,
) -> float | None:
    if baseline_payload is None:
        return None
    measurement = baseline_payload.get("measurement")
    if not isinstance(measurement, dict):
        return None
    per_token_ms = _to_float(measurement.get("per_token_latency_ms"))
    if per_token_ms is None or per_token_ms <= 0:
        return None
    return 1000.0 / per_token_ms


def main() -> int:
    args = parse_args()
    if args.num_requests <= 0:
        raise ValueError("--num-requests must be > 0")
    if args.prompt_length <= 0:
        raise ValueError("--prompt-length must be > 0")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be > 0")

    batch_sizes = [size for size in args.batch_sizes if size > 0]
    if not batch_sizes:
        raise ValueError(
            "--batch-sizes must include at least one positive value"
        )

    env = environment_info()
    output_path = Path(args.output_json)
    baseline_payload = load_baseline(Path(args.baseline_json))
    baseline_tokens_per_second = estimate_baseline_tokens_per_second(
        baseline_payload
    )

    print("=== MoE-Infinity Continuous Batching Throughput ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"CUDA available: {env['cuda_available']}")

    if not env["cuda_available"]:
        print("BLOCKED: No CUDA. Run on GPU hardware.")
        payload = {
            "status": "BLOCKED",
            "reason": "No CUDA device",
            "environment": env,
            "measurement": {str(size): None for size in batch_sizes},
            "baseline_tokens_per_second": baseline_tokens_per_second,
            "requested_model": args.model,
            "offload_dir": args.offload_dir,
            "num_requests": args.num_requests,
        }
        write_json(output_path, payload)
        return 0

    try:
        model, tokenizer = load_model_and_tokenizer(
            args.model, args.offload_dir
        )
    except Exception as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}")
        payload = {
            "status": "BLOCKED",
            "reason": f"{type(exc).__name__}: {exc}",
            "environment": env,
            "measurement": {str(size): None for size in batch_sizes},
            "baseline_tokens_per_second": baseline_tokens_per_second,
            "requested_model": args.model,
            "offload_dir": args.offload_dir,
            "num_requests": args.num_requests,
        }
        write_json(output_path, payload)
        return 0

    measurements = run_sweep(
        model,
        tokenizer,
        num_requests=args.num_requests,
        batch_sizes=batch_sizes,
        prompt_length=args.prompt_length,
        max_new_tokens=args.max_new_tokens,
    )
    speedup_vs_baseline: dict[str, float | None] = {}
    for batch_size, tokens_per_second in measurements.items():
        if (
            baseline_tokens_per_second is None
            or baseline_tokens_per_second <= 0.0
        ):
            speedup_vs_baseline[batch_size] = None
        else:
            speedup_vs_baseline[batch_size] = (
                tokens_per_second / baseline_tokens_per_second
            )

    payload = {
        "status": "PASS",
        "environment": env,
        "measurement": measurements,
        "comparison": {
            "baseline_tokens_per_second": baseline_tokens_per_second,
            "speedup_vs_baseline": speedup_vs_baseline,
        },
        "requested_model": args.model,
        "offload_dir": args.offload_dir,
        "num_requests": args.num_requests,
        "prompt_length": args.prompt_length,
        "max_new_tokens": args.max_new_tokens,
    }
    write_json(output_path, payload)
    return 0


if __name__ == "__main__":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    raise SystemExit(main())
