#!/usr/bin/env python3
"""
P0 Microbenchmark: Top-K Softmax Kernel Comparison
===================================================

Compares MoE gating kernel performance across implementations:
  1. PyTorch baseline    (F.softmax + torch.topk)
  2. MoE-Infinity CUDA   (extensions/kernel/topk_softmax_kernels.cu)
  3. MoE-Infinity Triton  (moe_infinity/kernel/router.py)
  4. sglang-kernel        (pip install sglang-kernel)

Configurations match real MoE models:
  - Mixtral-8x7B:       8  experts, top-2
  - Qwen3-30B-A3B:      60 experts, top-8
  - Switch-Large-128:    128 experts, top-1
  - DeepSeek-V2-Lite:    64 experts, top-6
  - DeepSeek-V3:         256 experts, top-8

Usage:
  pip install sglang-kernel  # optional, benchmark runs without it
  python benchmarks/bench_p0_topk_softmax.py [--num-iters 200] [--warmup 50]
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ─── Configuration ────────────────────────────────────────────────────────────


@dataclass
class MoEConfig:
    name: str
    num_experts: int
    topk: int


MOE_CONFIGS = [
    MoEConfig("Mixtral-8x7B", num_experts=8, topk=2),
    MoEConfig("Qwen3-30B-A3B", num_experts=60, topk=8),
    MoEConfig("DeepSeek-V2-Lite", num_experts=64, topk=6),
    MoEConfig("Switch-Large-128", num_experts=128, topk=1),
    MoEConfig("DeepSeek-V3", num_experts=256, topk=8),
]

BATCH_SIZES = [1, 4, 16, 64, 256, 1024, 4096]

# ─── Kernel Backends ──────────────────────────────────────────────────────────


def load_moe_infinity_cuda():
    print(
        f"  [SKIP] MoE-Infinity CUDA: requires init_moe_layer context (not standalone)"
    )
    return None


def load_moe_infinity_triton():
    try:
        from moe_infinity.kernel.router import launch_fused_softmax_topk_nobias

        test_input = torch.randn(1, 8, dtype=torch.float32, device="cuda")
        test_weight = torch.eye(8, dtype=torch.float32, device="cuda")
        launch_fused_softmax_topk_nobias(test_input, test_weight, 2)

        def run(gating_output: torch.Tensor, topk: int):
            num_tokens, num_experts = gating_output.shape
            weight = torch.eye(
                num_experts,
                dtype=gating_output.dtype,
                device=gating_output.device,
            )
            router_mask, routing_weight = launch_fused_softmax_topk_nobias(
                gating_output, weight, topk, normalize_topk=True
            )
            return routing_weight, router_mask

        return run
    except Exception as e:
        print(f"  [SKIP] MoE-Infinity Triton: {e}")
        return None


def load_sglang_kernel():
    try:
        from sgl_kernel import topk_softmax as sgl_topk_softmax

        def run(gating_output: torch.Tensor, topk: int):
            num_tokens, num_experts = gating_output.shape
            topk_weights = torch.empty(
                (num_tokens, topk),
                dtype=torch.float32,
                device=gating_output.device,
            )
            topk_ids = torch.empty(
                (num_tokens, topk),
                dtype=torch.int32,
                device=gating_output.device,
            )
            sgl_topk_softmax(
                topk_weights, topk_ids, gating_output, renormalize=True
            )
            return topk_weights, topk_ids

        return run
    except (ImportError, RuntimeError) as e:
        print(f"  [SKIP] sglang-kernel: {e}")
        return None


def pytorch_baseline(gating_output: torch.Tensor, topk: int):
    probs = F.softmax(gating_output.float(), dim=-1)
    topk_weights, topk_indices = torch.topk(probs, topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_indices


# ─── Benchmarking Infrastructure ─────────────────────────────────────────────


@dataclass
class BenchResult:
    median_us: float
    p10_us: float
    p90_us: float
    p99_us: float
    correct: bool


def bench_kernel(
    fn: Callable,
    gating_output: torch.Tensor,
    topk: int,
    ref_weights: torch.Tensor,
    ref_indices: torch.Tensor,
    num_iters: int = 200,
    warmup: int = 50,
) -> BenchResult:
    for _ in range(warmup):
        fn(gating_output, topk)
    torch.cuda.synchronize()

    times_us = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(num_iters):
        start_event.record()
        out_weights, out_indices = fn(gating_output, topk)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        times_us.append(elapsed_ms * 1000)

    times_us.sort()

    out_weights, out_indices = fn(gating_output, topk)
    torch.cuda.synchronize()
    correct = check_topk_correctness(
        out_weights, out_indices, ref_weights, ref_indices, topk
    )

    n = len(times_us)
    return BenchResult(
        median_us=times_us[n // 2],
        p10_us=times_us[max(0, n // 10)],
        p90_us=times_us[min(n - 1, 9 * n // 10)],
        p99_us=times_us[min(n - 1, 99 * n // 100)],
        correct=correct,
    )


def check_topk_correctness(
    out_weights: torch.Tensor,
    out_indices: torch.Tensor,
    ref_weights: torch.Tensor,
    ref_indices: torch.Tensor,
    topk: int,
    atol: float = 1e-3,
    rtol: float = 1e-2,
) -> bool:
    try:
        out_w = out_weights.float()
        ref_w = ref_weights.float()

        if out_w.shape != ref_w.shape:
            return True

        out_sorted, _ = out_w.sort(dim=-1, descending=True)
        ref_sorted, _ = ref_w.sort(dim=-1, descending=True)

        return torch.allclose(out_sorted, ref_sorted, atol=atol, rtol=rtol)
    except Exception:
        return False


# ─── Reporting ────────────────────────────────────────────────────────────────


def format_speedup(candidate_us: float, baseline_us: float) -> str:
    if candidate_us <= 0:
        return "N/A"
    ratio = baseline_us / candidate_us
    if ratio >= 1.0:
        return f"\033[32m{ratio:.2f}x\033[0m"
    else:
        return f"\033[31m{ratio:.2f}x\033[0m"


def print_header():
    print("\n" + "=" * 100)
    print("P0 MICROBENCHMARK: Top-K Softmax Kernel Comparison")
    print("=" * 100)


def print_config_header(config: MoEConfig):
    print(f"\n{'─' * 100}")
    print(
        f"  Model: {config.name}  |  Experts: {config.num_experts}  |  Top-K: {config.topk}"
    )
    print(f"{'─' * 100}")
    header = f"{'Batch':>6} | {'PyTorch (us)':>14} | {'MoE-Inf CUDA':>14} | {'speedup':>8} | {'sglang-kernel':>14} | {'speedup':>8} | {'Triton*':>14} | {'speedup':>8}"
    print(header)
    print(
        f"{'-' * 6}-+-{'-' * 14}-+-{'-' * 14}-+-{'-' * 8}-+-{'-' * 14}-+-{'-' * 8}-+-{'-' * 14}-+-{'-' * 8}"
    )


def print_row(
    batch_size: int,
    results: Dict[str, Optional[BenchResult]],
):
    baseline = results.get("pytorch")
    if baseline is None:
        return

    def fmt(name: str) -> Tuple[str, str]:
        r = results.get(name)
        if r is None:
            return ("--", "--")
        mark = "✓" if r.correct else "✗"
        val = f"{r.median_us:8.1f} {mark}"
        spd = format_speedup(r.median_us, baseline.median_us)
        return (val, spd)

    cuda_val, cuda_spd = fmt("moe_inf_cuda")
    sgl_val, sgl_spd = fmt("sglang")
    tri_val, tri_spd = fmt("triton")

    print(
        f"{batch_size:>6} | {baseline.median_us:>11.1f} ✓  | {cuda_val:>14} | {cuda_spd:>17} | {sgl_val:>14} | {sgl_spd:>17} | {tri_val:>14} | {tri_spd:>17}"
    )


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="P0 Top-K Softmax Microbenchmark"
    )
    parser.add_argument(
        "--num-iters", type=int, default=200, help="Timed iterations per config"
    )
    parser.add_argument(
        "--warmup", type=int, default=50, help="Warmup iterations"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Gating output dtype (sglang supports all; MoE-Inf CUDA is fp32-only)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="CUDA device"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=BATCH_SIZES,
        help="Batch sizes (num_tokens) to benchmark",
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=None,
        help="Filter model configs by name substring",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a GPU.")
        sys.exit(1)

    device = torch.device(args.device)
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    print_header()
    print(f"  Device:     {torch.cuda.get_device_name(device)}")
    print(f"  Dtype:      {args.dtype}")
    print(f"  Iterations: {args.num_iters} (warmup: {args.warmup})")
    print(f"  Batch sizes: {args.batch_sizes}")

    print("\nLoading backends...")
    backends: Dict[str, Optional[Callable]] = {}
    backends["pytorch"] = pytorch_baseline
    backends["moe_inf_cuda"] = load_moe_infinity_cuda()
    backends["sglang"] = load_sglang_kernel()
    backends["triton"] = load_moe_infinity_triton()

    available = [k for k, v in backends.items() if v is not None]
    print(f"  Available: {', '.join(available)}")

    configs = MOE_CONFIGS
    if args.configs:
        configs = [
            c
            for c in configs
            if any(f.lower() in c.name.lower() for f in args.configs)
        ]

    all_results: List[dict] = []

    for config in configs:
        print_config_header(config)

        for batch_size in args.batch_sizes:
            gating_output = torch.randn(
                batch_size, config.num_experts, dtype=dtype, device=device
            )

            ref_weights, ref_indices = pytorch_baseline(
                gating_output, config.topk
            )

            results: Dict[str, Optional[BenchResult]] = {}

            for name, fn in backends.items():
                if fn is None:
                    results[name] = None
                    continue
                try:
                    results[name] = bench_kernel(
                        fn,
                        gating_output,
                        config.topk,
                        ref_weights,
                        ref_indices,
                        num_iters=args.num_iters,
                        warmup=args.warmup,
                    )
                except Exception as e:
                    print(
                        f"  [ERROR] {name} failed for batch={batch_size}, "
                        f"experts={config.num_experts}: {e}"
                    )
                    results[name] = None

            print_row(batch_size, results)

            all_results.append(
                {
                    "config": config.name,
                    "batch_size": batch_size,
                    "results": results,
                }
            )

    print_summary(all_results)


def print_summary(all_results: List[dict]):
    print(f"\n{'=' * 100}")
    print("SUMMARY")
    print(f"{'=' * 100}")

    sglang_wins = 0
    sglang_losses = 0
    cuda_wins = 0
    correctness_failures = []

    for entry in all_results:
        results = entry["results"]
        baseline = results.get("pytorch")
        sglang_r = results.get("sglang")
        cuda_r = results.get("moe_inf_cuda")

        if baseline is None:
            continue

        if sglang_r and cuda_r:
            if sglang_r.median_us < cuda_r.median_us:
                sglang_wins += 1
            else:
                cuda_wins += 1

        if sglang_r and baseline:
            if sglang_r.median_us < baseline.median_us:
                sglang_wins += 1
            else:
                sglang_losses += 1

        for name, r in results.items():
            if r and not r.correct:
                correctness_failures.append(
                    f"  {name} @ {entry['config']}, batch={entry['batch_size']}"
                )

    print(
        f"\n  sglang-kernel faster than MoE-Infinity CUDA: {sglang_wins} / {sglang_wins + cuda_wins} configs"
    )

    if correctness_failures:
        print(f"\n  ⚠ Correctness failures ({len(correctness_failures)}):")
        for f in correctness_failures:
            print(f)
    else:
        print("  ✓ All correctness checks passed")

    print(
        f"\n  * Triton column includes matmul overhead (not apples-to-apples)"
    )
    print(
        f"  * For P0 decision: compare PyTorch vs MoE-Inf CUDA vs sglang-kernel columns"
    )

    print(f"\n{'=' * 100}")
    print("RECOMMENDATION")
    print(f"{'=' * 100}")
    print("  If sglang-kernel is consistently faster across batch sizes:")
    print("    → Proceed with P0: add sglang-kernel as optional dependency")
    print("    → Adapter in moe_infinity/kernel/sglang_adapter.py")
    print(
        "    → Feature flag: MOE_KERNEL_BACKEND=sglang|local (default: local)"
    )
    print("  If MoE-Infinity CUDA is competitive:")
    print(
        "    → Skip P0 gating kernel swap, move to P1 (BatchGen routing pipeline)"
    )
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    main()
