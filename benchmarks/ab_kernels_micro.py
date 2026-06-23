#!/usr/bin/env python3
"""Kernel-level A/B microbenchmark: fused vs unfused.

Measures pure kernel performance (no model overhead). This is the cleanest
A/B test of the new fused kernels.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch


def bench(fn, n_warmup: int = 10, n_iters: int = 100) -> dict:
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "total_s": elapsed,
        "per_iter_ms": elapsed / n_iters * 1000,
        "throughput_per_s": n_iters / elapsed,
    }


def bench_fused_qkv(
    M: int = 32,
    hidden_dim: int = 4096,
    num_q_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
) -> dict:
    print(
        f"\n--- Fused QKV: M={M} hidden={hidden_dim} q_heads={num_q_heads} kv_heads={num_kv_heads} head_dim={head_dim}"
    )

    from moe_infinity.kernel.fused_qkv import fused_qkv_proj

    total_dim = (num_q_heads + 2 * num_kv_heads) * head_dim
    x = torch.randn(M, hidden_dim, dtype=torch.bfloat16, device="cuda")
    w_qkv = torch.randn(
        hidden_dim, total_dim, dtype=torch.bfloat16, device="cuda"
    )

    q_dim = num_q_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    w_q = w_qkv[:, :q_dim].T.contiguous()
    w_k = w_qkv[:, q_dim : q_dim + kv_dim].T.contiguous()
    w_v = w_qkv[:, q_dim + kv_dim :].T.contiguous()

    def fused():
        return fused_qkv_proj(x, w_qkv, num_q_heads, num_kv_heads, head_dim)

    def unfused():
        q = (x @ w_q.T).reshape(M, num_q_heads, head_dim)
        k = (x @ w_k.T).reshape(M, num_kv_heads, head_dim)
        v = (x @ w_v.T).reshape(M, num_kv_heads, head_dim)
        return q, k, v

    qf, kf, vf = fused()
    qu, ku, vu = unfused()
    max_err = max(
        (qf - qu).abs().max().item(),
        (kf - ku).abs().max().item(),
        (vf - vu).abs().max().item(),
    )

    res_unfused = bench(unfused)
    res_fused = bench(fused)
    speedup = res_unfused["per_iter_ms"] / res_fused["per_iter_ms"]

    print(f"  Unfused (3x matmul):  {res_unfused['per_iter_ms']:.3f} ms/iter")
    print(f"  Fused (1 kernel):     {res_fused['per_iter_ms']:.3f} ms/iter")
    print(f"  Speedup:              {speedup:.2f}x")
    print(f"  Max numerical error:  {max_err:.4f}")

    return {
        "kernel": "fused_qkv",
        "shape": {
            "M": M,
            "hidden": hidden_dim,
            "q_heads": num_q_heads,
            "kv_heads": num_kv_heads,
            "head_dim": head_dim,
        },
        "unfused_ms": res_unfused["per_iter_ms"],
        "fused_ms": res_fused["per_iter_ms"],
        "speedup": speedup,
        "max_error": max_err,
    }


def bench_fused_ffn(
    M: int = 32,
    hidden_dim: int = 4096,
    intermediate_size: int = 11008,
) -> dict:
    print(
        f"\n--- Fused FFN: M={M} hidden={hidden_dim} intermediate={intermediate_size}"
    )

    from moe_infinity.kernel.fused_ffn import fused_ffn

    x = torch.randn(M, hidden_dim, dtype=torch.bfloat16, device="cuda")
    gate_w = torch.randn(
        intermediate_size, hidden_dim, dtype=torch.bfloat16, device="cuda"
    )
    up_w = torch.randn(
        intermediate_size, hidden_dim, dtype=torch.bfloat16, device="cuda"
    )
    down_w = torch.randn(
        hidden_dim, intermediate_size, dtype=torch.bfloat16, device="cuda"
    )

    def fused():
        return fused_ffn(x, gate_w, up_w, down_w)

    def unfused():
        gate = x @ gate_w.T
        up = x @ up_w.T
        intermediate = torch.nn.functional.silu(gate) * up
        return intermediate @ down_w.T

    out_f = fused()
    out_u = unfused()
    max_err = (out_f - out_u).abs().max().item()

    res_unfused = bench(unfused)
    res_fused = bench(fused)
    speedup = res_unfused["per_iter_ms"] / res_fused["per_iter_ms"]

    print(f"  Unfused (4 ops):      {res_unfused['per_iter_ms']:.3f} ms/iter")
    print(f"  Fused (Triton):       {res_fused['per_iter_ms']:.3f} ms/iter")
    print(f"  Speedup:              {speedup:.2f}x")
    print(f"  Max numerical error:  {max_err:.4f}")

    return {
        "kernel": "fused_ffn",
        "shape": {
            "M": M,
            "hidden": hidden_dim,
            "intermediate": intermediate_size,
        },
        "unfused_ms": res_unfused["per_iter_ms"],
        "fused_ms": res_fused["per_iter_ms"],
        "speedup": speedup,
        "max_error": max_err,
    }


def bench_marlin(
    M: int = 32,
    K: int = 4096,
    N: int = 4096,
    groupsize: int = 128,
) -> dict:
    print(f"\n--- Marlin INT4: M={M} K={K} N={N} groupsize={groupsize}")

    try:
        import moe_infinity._marlin
    except ImportError:
        print("  SKIP: moe_infinity._marlin not available")
        return {
            "kernel": "marlin_gemm",
            "skipped": True,
            "reason": "not compiled",
        }

    cap = torch.cuda.get_device_capability(0)
    if cap[0] >= 9:
        print(
            f"  SKIP: Marlin kernel requires sm_80/86/89 (Ampere/Ada), got sm_{cap[0]}{cap[1]} (Hopper/Blackwell)"
        )
        print(
            f"  Note: Memory reduction is still 4x, just kernel performance is unoptimized for sm_>=90"
        )
        from moe_infinity.kernel.marlin_gemm import marlin_quantize

        weight_fp16 = torch.randn(K, N, dtype=torch.float16, device="cuda")
        packed, scales = marlin_quantize(weight_fp16, groupsize)
        memory_reduction = (K * N * 2) / (
            packed.numel() * 4 + scales.numel() * 2
        )
        print(
            f"  Weight memory: FP16={K*N*2/1e6:.1f} MB  INT4={packed.numel()*4/1e6:.1f} MB  reduction={memory_reduction:.2f}x"
        )
        return {
            "kernel": "marlin_gemm",
            "shape": {"M": M, "K": K, "N": N, "groupsize": groupsize},
            "skipped": True,
            "reason": f"sm_{cap[0]}{cap[1]} not supported by Marlin (needs Ampere/Ada)",
            "memory_reduction_x": memory_reduction,
        }

    from moe_infinity.kernel.marlin_gemm import (
        marlin_gemm,
        marlin_quantize,
        prepare_workspace,
    )

    weight_fp16 = torch.randn(K, N, dtype=torch.float16, device="cuda")
    packed, scales = marlin_quantize(weight_fp16, groupsize)
    workspace = prepare_workspace(N, torch.device("cuda"))
    x = torch.randn(M, K, dtype=torch.float16, device="cuda")

    def fp16_baseline():
        return x @ weight_fp16

    def marlin():
        return marlin_gemm(x, packed, scales, workspace)

    res_fp16 = bench(fp16_baseline)
    res_marlin = bench(marlin)
    speedup = res_fp16["per_iter_ms"] / res_marlin["per_iter_ms"]
    memory_reduction = (K * N * 2) / (packed.numel() * 4 + scales.numel() * 2)

    print(f"  FP16 baseline:        {res_fp16['per_iter_ms']:.3f} ms/iter")
    print(f"  Marlin INT4:          {res_marlin['per_iter_ms']:.3f} ms/iter")
    print(f"  Speedup:              {speedup:.2f}x")
    print(f"  Weight memory reduction: {memory_reduction:.2f}x")

    return {
        "kernel": "marlin_gemm",
        "shape": {"M": M, "K": K, "N": N, "groupsize": groupsize},
        "fp16_ms": res_fp16["per_iter_ms"],
        "marlin_ms": res_marlin["per_iter_ms"],
        "speedup": speedup,
        "memory_reduction_x": memory_reduction,
    }


def main() -> None:
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(
        f"CUDA capability: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}"
    )

    results: list[dict] = []

    print("\n" + "=" * 70)
    print("  FUSED QKV PROJECTION (decode shapes)")
    print("=" * 70)
    for batch in [1, 4, 16, 32]:
        results.append(
            bench_fused_qkv(
                M=batch,
                hidden_dim=4096,
                num_q_heads=32,
                num_kv_heads=8,
                head_dim=128,
            )
        )

    print("\n" + "=" * 70)
    print("  FUSED FFN (Llama-3 8B sized)")
    print("=" * 70)
    for batch in [1, 4, 16, 32]:
        results.append(
            bench_fused_ffn(M=batch, hidden_dim=4096, intermediate_size=14336)
        )

    print("\n" + "=" * 70)
    print("  MARLIN W4A16 GEMM")
    print("=" * 70)
    for batch in [1, 4, 16, 32]:
        results.append(bench_marlin(M=batch, K=4096, N=4096, groupsize=128))

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(
        f"{'Kernel':<15} {'Shape':<35} {'Unfused':>10} {'Fused':>10} {'Speedup':>10}"
    )
    print("-" * 80)
    for r in results:
        if r.get("skipped"):
            continue
        kernel = r["kernel"]
        shape = r["shape"]
        if kernel == "fused_qkv":
            shape_str = f"M={shape['M']} hidden={shape['hidden']}"
            unfused = r["unfused_ms"]
            fused = r["fused_ms"]
        elif kernel == "fused_ffn":
            shape_str = f"M={shape['M']} hidden={shape['hidden']} int={shape['intermediate']}"
            unfused = r["unfused_ms"]
            fused = r["fused_ms"]
        else:
            shape_str = f"M={shape['M']} K={shape['K']} N={shape['N']}"
            unfused = r["fp16_ms"]
            fused = r["marlin_ms"]
        print(
            f"{kernel:<15} {shape_str:<35} {unfused:>9.3f}ms {fused:>9.3f}ms {r['speedup']:>9.2f}x"
        )

    output_path = Path("/results/kernel_ab_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
