#!/usr/bin/env python3
from __future__ import annotations

# pyright: reportMissingImports=false, reportImplicitRelativeImport=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Optional

import torch

sys.path.insert(0, "/workspace")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import (
    MODEL_CONFIGS,
    PROMPT_DATASET,
    BenchmarkResult,
    get_gpu_name,
    save_result,
)

SUPPORTED_MODELS = [
    "deepseek-v2-lite",
    "mixtral-8x7b",
    "qwen3-30b",
    "gpt-oss-20b",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run vLLM v0.18.1 benchmark for comparison suite."
    )
    _ = parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS + ["all"],
        default="all",
    )
    _ = parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.75,
    )
    _ = parser.add_argument(
        "--output-dir",
        default="/results",
    )
    _ = parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
    )
    return parser.parse_args()


def _make_result(
    model_name: str,
    precision: str,
    per_token_latency_s: Optional[float],
    ttft_s: Optional[float],
    peak_gpu_mb: Optional[float],
    num_iterations: int,
    notes: str,
) -> BenchmarkResult:
    return BenchmarkResult(
        model=model_name,
        framework="vllm",
        precision=precision,
        per_token_latency_s=per_token_latency_s,
        ttft_s=ttft_s,
        peak_gpu_mb=peak_gpu_mb,
        num_iterations=num_iterations,
        timestamp=datetime.now().isoformat(),
        gpu_name=get_gpu_name(),
        notes=notes,
    )


def _is_oom_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "cuda out of memory" in text or "out of memory" in text


def _is_not_supported_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return (
        "not supported" in text
        or "unknown model" in text
        or "valueerror" in text
        or isinstance(exc, ValueError)
    )


def _quantization_attempts(
    model_name: str,
) -> list[tuple[Optional[str], str, str]]:
    if model_name in {"mixtral-8x7b", "qwen3-30b"}:
        return [("fp8", "FP8", "")]
    if model_name == "deepseek-v2-lite":
        return [
            (None, "FP16", ""),
            ("fp8", "FP8", "Fell back to FP8 after FP16 OOM."),
        ]
    return [(None, "FP16", "")]


def _extract_token_count(output_obj: object) -> int:
    outputs = getattr(output_obj, "outputs", None)
    if not outputs:
        return 0
    first_output = outputs[0]
    token_ids = getattr(first_output, "token_ids", None)
    if token_ids is None:
        return 0
    return len(token_ids)


def _extract_timings(
    output_obj: object,
    elapsed_s: float,
    token_count: int,
) -> tuple[float, Optional[float]]:
    ttft_s: Optional[float] = None
    per_token_s: Optional[float] = None
    metrics = getattr(output_obj, "metrics", None)
    first_token_time = None
    finished_time = None
    time_in_queue = 0.0

    if metrics is not None:
        first_token_time = getattr(metrics, "first_token_time", None)
        finished_time = getattr(metrics, "finished_time", None)
        time_in_queue = float(getattr(metrics, "time_in_queue", 0.0) or 0.0)

    if first_token_time is not None:
        ttft_s = max(0.0, float(first_token_time) - time_in_queue)

    if (
        first_token_time is not None
        and finished_time is not None
        and token_count > 0
    ):
        decode_s = max(0.0, float(finished_time) - float(first_token_time))
        per_token_s = decode_s / float(token_count)

    if ttft_s is None:
        ttft_s = elapsed_s

    if per_token_s is None and token_count > 0:
        decode_s = max(0.0, elapsed_s - ttft_s)
        if decode_s == 0.0:
            decode_s = elapsed_s
        per_token_s = decode_s / float(token_count)

    return ttft_s, per_token_s


def run_single_model(
    model_name: str, args: argparse.Namespace
) -> BenchmarkResult:
    from vllm import LLM, SamplingParams

    model_id = MODEL_CONFIGS[model_name]
    attempts = _quantization_attempts(model_name)

    llm = None
    precision_used = "FP16"
    fallback_note = ""

    for index, (quantization, precision, note) in enumerate(attempts):
        try:
            llm = LLM(
                model=model_id,
                quantization=quantization,
                gpu_memory_utilization=args.gpu_memory_utilization,
                dtype="auto",
                max_model_len=2048,
            )
            precision_used = precision
            fallback_note = note
            break
        except Exception as exc:
            if _is_not_supported_error(exc):
                return _make_result(
                    model_name=model_name,
                    precision=precision,
                    per_token_latency_s=None,
                    ttft_s=None,
                    peak_gpu_mb=None,
                    num_iterations=len(PROMPT_DATASET),
                    notes=f"Model not supported by vLLM v0.18.1: {exc}",
                )

            if _is_oom_error(exc):
                has_next = index < (len(attempts) - 1)
                if has_next and model_name == "deepseek-v2-lite":
                    continue
                return _make_result(
                    model_name=model_name,
                    precision=precision,
                    per_token_latency_s=None,
                    ttft_s=None,
                    peak_gpu_mb=None,
                    num_iterations=len(PROMPT_DATASET),
                    notes="OOM on 24GB",
                )

            return _make_result(
                model_name=model_name,
                precision=precision,
                per_token_latency_s=None,
                ttft_s=None,
                peak_gpu_mb=None,
                num_iterations=len(PROMPT_DATASET),
                notes=f"Initialization failed: {type(exc).__name__}: {exc}",
            )

    if llm is None:
        return _make_result(
            model_name=model_name,
            precision="FP16",
            per_token_latency_s=None,
            ttft_s=None,
            peak_gpu_mb=None,
            num_iterations=len(PROMPT_DATASET),
            notes="Initialization failed: unknown error",
        )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_new_tokens,
    )

    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        _ = llm.generate(["Hello"] * 5, sampling_params)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        ttfts: list[float] = []
        itls: list[float] = []

        for prompt in PROMPT_DATASET:
            start_ts = time.time()
            outputs = llm.generate([prompt], sampling_params)
            elapsed_s = time.time() - start_ts
            output_obj = outputs[0]
            token_count = _extract_token_count(output_obj)
            ttft_s, per_token_s = _extract_timings(
                output_obj=output_obj,
                elapsed_s=elapsed_s,
                token_count=token_count,
            )
            ttfts.append(ttft_s)
            if per_token_s is not None:
                itls.append(per_token_s)

        peak_gpu_mb: Optional[float] = None
        if torch.cuda.is_available():
            peak_gpu_mb = float(torch.cuda.max_memory_allocated()) / float(
                1024**2
            )

        notes = fallback_note
        if not itls:
            if notes:
                notes = notes + " "
            notes = (
                notes
                + "Decode token metrics unavailable; per-token latency missing."
            )

        return _make_result(
            model_name=model_name,
            precision=precision_used,
            per_token_latency_s=mean(itls) if itls else None,
            ttft_s=mean(ttfts) if ttfts else None,
            peak_gpu_mb=peak_gpu_mb,
            num_iterations=len(ttfts),
            notes=notes,
        )
    except Exception as exc:
        if _is_oom_error(exc):
            return _make_result(
                model_name=model_name,
                precision=precision_used,
                per_token_latency_s=None,
                ttft_s=None,
                peak_gpu_mb=None,
                num_iterations=len(PROMPT_DATASET),
                notes="OOM on 24GB",
            )
        return _make_result(
            model_name=model_name,
            precision=precision_used,
            per_token_latency_s=None,
            ttft_s=None,
            peak_gpu_mb=None,
            num_iterations=len(PROMPT_DATASET),
            notes=f"Runtime failed: {type(exc).__name__}: {exc}",
        )
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _failure_code_hint(result: BenchmarkResult) -> Optional[int]:
    if result.notes.startswith("Model not supported by vLLM v0.18.1"):
        return 2
    if "OOM on 24GB" in result.notes:
        return 3
    return None


def main() -> None:
    args = parse_args()
    model_names = SUPPORTED_MODELS if args.model == "all" else [args.model]

    results: list[BenchmarkResult] = []
    for model_name in model_names:
        result = run_single_model(model_name, args)
        output_path = save_result(result, args.output_dir)
        results.append(result)
        print(
            f"[{model_name}] framework={result.framework} precision={result.precision} "
            + f"ttft_s={result.ttft_s} per_token_latency_s={result.per_token_latency_s} "
            + f"peak_gpu_mb={result.peak_gpu_mb} notes={result.notes} saved={output_path}"
        )

    failure_codes = [_failure_code_hint(result) for result in results]
    all_failed = all(code is not None for code in failure_codes)
    if all_failed and all(code == 2 for code in failure_codes):
        sys.exit(2)
    if all_failed and all(code == 3 for code in failure_codes):
        sys.exit(3)
    sys.exit(0)


if __name__ == "__main__":
    main()
