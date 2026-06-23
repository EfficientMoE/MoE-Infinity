#!/usr/bin/env python3
"""A/B benchmark: fused kernels ON vs OFF.

Usage:
    python benchmarks/ab_fused_kernels.py --model deepseek-ai/DeepSeek-V2-Lite-Chat --offload-dir /offload
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A/B test fused kernels")
    parser.add_argument("--model", required=True)
    parser.add_argument("--offload-dir", required=True)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--output-json", default="/results/ab_results.json")
    return parser.parse_args()


def run_latency_bench(
    env_overrides: dict[str, str],
    args: argparse.Namespace,
    label: str,
) -> dict:
    env = os.environ.copy()
    env.update(env_overrides)

    cmd = [
        sys.executable,
        "benchmarks/serving/latency.py",
        "--model",
        args.model,
        "--offload-dir",
        args.offload_dir,
        "--concurrency",
        *[str(c) for c in args.concurrency],
        "--num-rounds",
        str(args.num_rounds),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--prompt-length",
        str(args.prompt_length),
    ]

    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"  Env: {env_overrides}")
    print(f"{'='*60}\n")

    start = time.perf_counter()
    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parents[1]),
    )
    elapsed = time.perf_counter() - start

    print(result.stdout)
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr}", file=sys.stderr)
        return {
            "label": label,
            "error": result.stderr,
            "returncode": result.returncode,
        }

    lines = result.stdout.strip().split("\n")
    metrics: dict = {
        "label": label,
        "elapsed_s": round(elapsed, 2),
        "raw_output": result.stdout,
    }

    for line in lines:
        if "p50" in line.lower() or "median" in line.lower():
            metrics["summary_line"] = line.strip()
        if "itl" in line.lower():
            metrics.setdefault("itl_lines", []).append(line.strip())

    return metrics


def main() -> None:
    args = parse_args()

    results_a = run_latency_bench(
        {"MOE_DISABLE_FUSED_KERNELS": "1", "MOE_DISABLE_CUDA_GRAPHS": "1"},
        args,
        label="BASELINE (fused kernels OFF)",
    )

    results_b = run_latency_bench(
        {"MOE_DISABLE_FUSED_KERNELS": "0", "MOE_DISABLE_CUDA_GRAPHS": "1"},
        args,
        label="FUSED KERNELS ON",
    )

    report = {
        "model": args.model,
        "concurrency": args.concurrency,
        "num_rounds": args.num_rounds,
        "max_new_tokens": args.max_new_tokens,
        "prompt_length": args.prompt_length,
        "baseline": results_a,
        "fused": results_b,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))

    print(f"\n{'='*60}")
    print("  A/B COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline: {results_a.get('elapsed_s', 'N/A')}s total")
    print(f"  Fused:    {results_b.get('elapsed_s', 'N/A')}s total")
    if results_a.get("summary_line"):
        print(f"  Baseline metric: {results_a['summary_line']}")
    if results_b.get("summary_line"):
        print(f"  Fused metric:    {results_b['summary_line']}")
    print(f"\n  Full results: {args.output_json}")


if __name__ == "__main__":
    main()
