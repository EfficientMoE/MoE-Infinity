#!/usr/bin/env python3
"""MoE-Infinity benchmark script for the comparison suite."""

from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportPrivateLocalImportUsage=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false, reportUnusedParameter=false, reportAttributeAccessIssue=false, reportImplicitStringConcatenation=false, reportImplicitOverride=false, reportDeprecated=false, reportUnknownParameterType=false, reportMissingParameterType=false
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoTokenizer, TextStreamer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.comparison.common import (
    MODEL_CONFIGS,
    PROMPT_DATASET,
    BenchmarkResult,
    get_gpu_name,
    save_result,
)


class StopWatch(TextStreamer):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, skip_prompt=True, **kwargs)
        self.start_prefilling = None
        self.prefilling_time = None
        self.start_decoding = None
        self.decoding_time = None
        self.decoding_iterations = 0

    def put(self, value):
        if self.start_prefilling is None:
            self.start_prefilling = time.time()
            return
        elif self.prefilling_time is None:
            self.prefilling_time = time.time() - self.start_prefilling
            self.start_decoding = time.time()
        self.decoding_iterations += 1
        return super().put(value)

    def end(self):
        if self.decoding_time is None and self.start_decoding is not None:
            self.decoding_time = time.time() - self.start_decoding
        return super().end()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MoE-Infinity benchmarks for comparison suite."
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()) + ["all"],
        default="all",
        help="Model key to benchmark, or 'all' for all models.",
    )
    parser.add_argument(
        "--offload-dir",
        required=True,
        help="Directory for MoE expert offload storage.",
    )
    parser.add_argument(
        "--device-memory-ratio",
        type=float,
        default=0.75,
        help="Fraction of GPU memory for expert caching.",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/comparison/results/",
        help="Directory to write JSON benchmark results.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum new tokens generated per measured prompt.",
    )
    return parser.parse_args()


def _oom_result(model_name: str, note: str) -> BenchmarkResult:
    return BenchmarkResult(
        model=model_name,
        framework="moe_infinity",
        precision="FP16",
        per_token_latency_s=None,
        ttft_s=None,
        peak_gpu_mb=None,
        num_iterations=20,
        timestamp=datetime.now().isoformat(),
        gpu_name=get_gpu_name(),
        notes=note,
    )


def run_benchmark(model_name: str, args: argparse.Namespace) -> BenchmarkResult:
    model_id = MODEL_CONFIGS[model_name]

    try:
        from moe_infinity import MoE
    except Exception as exc:
        return _oom_result(
            model_name, f"ImportError: {type(exc).__name__}: {exc}"
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = MoE(
            model_id,
            {
                "offload_path": os.path.join(args.offload_dir, model_name),
                "device_memory_ratio": args.device_memory_ratio,
            },
        )

        warmup_ids = tokenizer("Hello", return_tensors="pt").input_ids.to(
            "cuda:0"
        )
        with torch.no_grad():
            for _ in range(5):
                _ = model.generate(
                    warmup_ids,
                    max_new_tokens=8,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

        ttfts: list[float] = []
        per_token_latencies: list[float] = []
        peak_gpu_mbs: list[float] = []

        with torch.no_grad():
            for prompt in PROMPT_DATASET:
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
                    "cuda:0"
                )
                stopwatch = StopWatch(tokenizer, skip_special_tokens=True)

                torch.cuda.reset_peak_memory_stats()
                _ = model.generate(
                    input_ids,
                    streamer=stopwatch,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

                if stopwatch.prefilling_time is None:
                    raise RuntimeError(
                        "StopWatch did not capture prefill time."
                    )
                if (
                    stopwatch.decoding_time is None
                    or stopwatch.decoding_iterations <= 0
                ):
                    raise RuntimeError(
                        "StopWatch did not capture decoding latency correctly."
                    )

                ttfts.append(float(stopwatch.prefilling_time))
                per_token_latencies.append(
                    float(stopwatch.decoding_time)
                    / float(stopwatch.decoding_iterations)
                )
                peak_gpu_mbs.append(
                    float(torch.cuda.max_memory_allocated()) / float(1024**2)
                )

        return BenchmarkResult(
            model=model_name,
            framework="moe_infinity",
            precision="FP16",
            per_token_latency_s=mean(per_token_latencies),
            ttft_s=mean(ttfts),
            peak_gpu_mb=max(peak_gpu_mbs),
            num_iterations=20,
            timestamp=datetime.now().isoformat(),
            gpu_name=get_gpu_name(),
            notes="",
        )
    except torch.cuda.OutOfMemoryError:
        return _oom_result(model_name, "OOM")
    except RuntimeError as exc:
        if "CUDA out of memory" in str(exc):
            return _oom_result(model_name, "OOM")
        return _oom_result(model_name, f"RuntimeError: {exc}")
    except Exception as exc:
        return _oom_result(model_name, f"{type(exc).__name__}: {exc}")


def main() -> int:
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    model_names = (
        list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]
    )

    results: list[BenchmarkResult] = []
    for model_name in model_names:
        result = run_benchmark(model_name, args)
        output_path = save_result(result, args.output_dir)
        results.append(result)
        print(
            f"[{model_name}] framework={result.framework} "
            f"ttft_s={result.ttft_s} "
            f"per_token_latency_s={result.per_token_latency_s} "
            f"peak_gpu_mb={result.peak_gpu_mb} notes={result.notes} "
            f"saved={output_path}"
        )

    all_oom = all(result.notes == "OOM" for result in results)
    if all_oom:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
