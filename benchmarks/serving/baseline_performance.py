from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportPrivateLocalImportUsage=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false, reportUnusedParameter=false, reportAttributeAccessIssue=false, reportImplicitStringConcatenation=false
import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_LENGTHS = (32, 128, 512)
DEFAULT_MAX_NEW_TOKENS = 16
MEGABYTE = 1024 * 1024


@dataclass
class RequestResult:
    prompt_tokens: int
    generated_tokens: int | None
    ttft_ms: float | None
    per_token_latency_ms: float | None
    total_time_s: float | None
    peak_gpu_memory_mb: float | None
    status: str
    detail: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure MoE-Infinity single-request baseline performance."
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
        default=10,
        help="Number of requests to measure",
    )
    parser.add_argument(
        "--output-json",
        default="baseline_results.json",
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
    if info["cuda_available"] and info["cuda_device_count"] > 0:
        info["cuda_device_names"] = [
            torch.cuda.get_device_name(idx)
            for idx in range(torch.cuda.device_count())
        ]
    return info


def _repeat_to_length(token_ids: list[int], target_length: int) -> list[int]:
    if not token_ids:
        return [0] * target_length
    output: list[int] = []
    while len(output) < target_length:
        output.extend(token_ids)
    return output[:target_length]


def build_prompt_input_ids(tokenizer: Any, target_length: int) -> torch.Tensor:
    base_text = (
        "MoE-Infinity baseline benchmark prompt. "
        "Measure single-request latency and output timing. "
        "Keep the text stable across requests."
    )
    encoded = tokenizer.encode(base_text, add_special_tokens=False)
    prompt_ids = _repeat_to_length(encoded, target_length)
    return torch.tensor([prompt_ids], dtype=torch.long, device="cuda")


class TimingStreamer:
    def __init__(self) -> None:
        self.first_token_at: float | None = None
        self._seen_prompt = False

    def put(self, value: Any) -> None:
        if not self._seen_prompt:
            self._seen_prompt = True
            return
        if self.first_token_at is None:
            self.first_token_at = time.perf_counter()

    def end(self) -> None:
        return None


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
        model_name, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = {
        "offload_path": offload_dir,
        "device_memory_ratio": 0.75,
    }
    model = MoE(model_name, config)
    return model, tokenizer


def run_one_request(
    model: Any,
    tokenizer: Any,
    prompt_tokens: int,
    max_new_tokens: int,
) -> RequestResult:
    input_ids = build_prompt_input_ids(tokenizer, prompt_tokens)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    streamer = TimingStreamer()
    start = time.perf_counter()
    try:
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
    except Exception as exc:
        return RequestResult(
            prompt_tokens=prompt_tokens,
            generated_tokens=None,
            ttft_ms=None,
            per_token_latency_ms=None,
            total_time_s=None,
            peak_gpu_memory_mb=(
                float(torch.cuda.max_memory_allocated() / MEGABYTE)
                if torch.cuda.is_available()
                else None
            ),
            status="ERROR",
            detail=f"{type(exc).__name__}: {exc}",
        )

    generated_tokens = max(int(output_ids.shape[-1] - input_ids.shape[-1]), 0)
    generated_tokens = generated_tokens or max_new_tokens
    ttft_ms = None
    if streamer.first_token_at is not None:
        ttft_ms = (streamer.first_token_at - start) * 1000.0
    total_time_s = end - start
    per_token_latency_ms = (total_time_s * 1000.0) / max(generated_tokens, 1)
    peak_gpu_memory_mb = (
        float(torch.cuda.max_memory_allocated() / MEGABYTE)
        if torch.cuda.is_available()
        else None
    )
    return RequestResult(
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        ttft_ms=ttft_ms,
        per_token_latency_ms=per_token_latency_ms,
        total_time_s=total_time_s,
        peak_gpu_memory_mb=peak_gpu_memory_mb,
        status="PASS",
    )


def summarize(results: list[RequestResult]) -> dict[str, Any]:
    successful = [r for r in results if r.status == "PASS"]
    if not successful:
        return {
            "ttft_ms": None,
            "per_token_latency_ms": None,
            "total_time_s": None,
            "peak_gpu_memory_mb": None,
            "num_requests": len(results),
        }

    ttft_values = [r.ttft_ms for r in successful if r.ttft_ms is not None]
    per_token_values = [
        r.per_token_latency_ms
        for r in successful
        if r.per_token_latency_ms is not None
    ]
    total_time_values = [
        r.total_time_s for r in successful if r.total_time_s is not None
    ]
    peak_mem_values = [
        r.peak_gpu_memory_mb
        for r in successful
        if r.peak_gpu_memory_mb is not None
    ]
    return {
        "ttft_ms": sum(ttft_values) / len(ttft_values) if ttft_values else None,
        "per_token_latency_ms": (
            sum(per_token_values) / len(per_token_values)
            if per_token_values
            else None
        ),
        "total_time_s": sum(total_time_values) / len(total_time_values)
        if total_time_values
        else None,
        "peak_gpu_memory_mb": max(peak_mem_values) if peak_mem_values else None,
        "num_requests": len(results),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> int:
    args = parse_args()
    env = environment_info()
    print("=== MoE-Infinity Baseline Performance ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"torch version: {env['torch_version']}")
    print(f"torch CUDA version: {env['torch_cuda_version']}")
    print(f"CUDA available: {env['cuda_available']}")
    print(f"CUDA device count: {env['cuda_device_count']}")

    output_path = Path(args.output_json)
    if not env["cuda_available"]:
        print(
            "BLOCKED: No CUDA device. Measurements will be collected on GPU hardware."
        )
        payload = {
            "status": "BLOCKED",
            "reason": "No CUDA device",
            "environment": env,
            "measurement": {
                "ttft_ms": None,
                "per_token_latency_ms": None,
                "total_time_s": None,
                "peak_gpu_memory_mb": None,
                "num_requests": args.num_requests,
            },
            "requested_model": args.model,
            "offload_dir": args.offload_dir,
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
            "measurement": {
                "ttft_ms": None,
                "per_token_latency_ms": None,
                "total_time_s": None,
                "peak_gpu_memory_mb": None,
                "num_requests": args.num_requests,
            },
            "requested_model": args.model,
            "offload_dir": args.offload_dir,
        }
        write_json(output_path, payload)
        return 0

    results: list[RequestResult] = []
    for idx in range(args.num_requests):
        prompt_tokens = PROMPT_LENGTHS[idx % len(PROMPT_LENGTHS)]
        result = run_one_request(
            model, tokenizer, prompt_tokens, DEFAULT_MAX_NEW_TOKENS
        )
        results.append(result)
        print(
            f"request={idx + 1} prompt_tokens={prompt_tokens} status={result.status} "
            f"ttft_ms={result.ttft_ms} per_token_latency_ms={result.per_token_latency_ms} "
            f"total_time_s={result.total_time_s} peak_gpu_memory_mb={result.peak_gpu_memory_mb}"
        )

    summary = summarize(results)
    payload = {
        "status": "PASS"
        if all(r.status == "PASS" for r in results)
        else "PARTIAL",
        "environment": env,
        "measurement": summary,
        "requested_model": args.model,
        "offload_dir": args.offload_dir,
        "results": [asdict(r) for r in results],
    }
    write_json(output_path, payload)
    return 0


if __name__ == "__main__":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    raise SystemExit(main())
