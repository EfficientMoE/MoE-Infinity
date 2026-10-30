#!/usr/bin/env python3
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Hy3-FP8 dequant/kernel microbenchmark (no checkpoint required).

Measures, for both hy_v3 routed-expert shapes ``[1536, 4096]`` (gate/up)
and ``[4096, 1536]`` (down) plus the batch-of-3-tensors expert bundle
(~18.9 MB FP8):

1. ``fp8_dequant_blockwise`` (native kernel) with the uniform expanded
   128x128 scale grid (``[12, 32]`` / ``[32, 12]`` fp32) -- the exact
   hy_v3 production dequant-on-copy path.
2. Pure-torch per-tensor dequant upper bounds for a hypothetical
   dedicated scalar kernel: ``(w.float() * s).to(bfloat16)`` and
   ``w.to(bfloat16) * s_bf16``.
3. Pinned-host -> GPU H2D copy of the same FP8 bytes, i.e. the transfer
   the dequant overlaps/follows in dequant-on-copy.

Timing uses ``torch.cuda.Event`` with >=20 warmup and >=100 measured
iterations; reports p50/p90 microseconds, effective GB/s over the FP8
input bytes (so dequant and H2D share a denominator), and the
dequant/H2D time ratio. Decision question: is a dedicated per-tensor
dequant kernel worth it, or is the uniform-grid blockwise kernel already
negligible vs PCIe?

Usage:
    CUDA_VISIBLE_DEVICES=<gpu> python benchmarks/hy3/bench_hy3_kernel.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch

GATE_UP_SHAPE = (1536, 4096)
DOWN_SHAPE = (4096, 1536)
BLOCK = 128
GIB = 1024.0**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--out", default=None, help="optional JSON output")
    return parser.parse_args()


def time_op(
    fn: Callable[[], object], warmup: int, iters: int
) -> dict[str, float]:
    """Return p50/p90/mean microseconds via torch.cuda.Event timing."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples_ms: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        torch.cuda.synchronize()
        samples_ms.append(start.elapsed_time(stop))
    samples_us = sorted(s * 1000.0 for s in samples_ms)

    def pct(p: float) -> float:
        idx = min(len(samples_us) - 1, int(round(p * (len(samples_us) - 1))))
        return samples_us[idx]

    return {
        "p50_us": pct(0.50),
        "p90_us": pct(0.90),
        "mean_us": statistics.fmean(samples_us),
    }


def uniform_grid(shape: tuple[int, int], scalar: float) -> torch.Tensor:
    """Expand a per-tensor scalar into the uniform 128x128 scale grid."""
    # Ceil division to match the production
    # _expand_pertensor_scale_to_blocks grid (identical for the 128-multiple
    # shapes benched here, but keeps the two in lockstep).
    rows = -(-shape[0] // BLOCK)
    cols = -(-shape[1] // BLOCK)
    return torch.full((rows, cols), scalar, dtype=torch.float32, device="cuda")


def make_fp8(shape: tuple[int, int]) -> torch.Tensor:
    return (
        torch.randn(shape, device="cuda", dtype=torch.float32)
        .clamp(-4, 4)
        .to(torch.float8_e4m3fn)
    )


def bench_shape(
    label: str, shape: tuple[int, int], warmup: int, iters: int
) -> list[dict[str, object]]:
    from moe_infinity._v4_fp4 import fp8_dequant_blockwise

    scalar = 0.0125
    w = make_fp8(shape)
    grid = uniform_grid(shape, scalar)
    s_f32 = torch.tensor(scalar, device="cuda", dtype=torch.float32)
    s_bf16 = torch.tensor(scalar, device="cuda", dtype=torch.bfloat16)
    fp8_bytes = w.numel() * w.element_size()

    host = torch.empty(shape, dtype=torch.uint8, pin_memory=True)
    dev = torch.empty(shape, dtype=torch.uint8, device="cuda")

    ops: list[tuple[str, Callable[[], object]]] = [
        (
            "native fp8_dequant_blockwise (uniform grid)",
            lambda: fp8_dequant_blockwise(w, grid),
        ),
        (
            "torch (w.float() * s).to(bf16)",
            lambda: (w.to(torch.float32) * s_f32).to(torch.bfloat16),
        ),
        (
            "torch w.to(bf16) * s_bf16",
            lambda: w.to(torch.bfloat16) * s_bf16,
        ),
        (
            "H2D pinned copy (same FP8 bytes)",
            lambda: dev.copy_(host, non_blocking=True),
        ),
    ]

    rows: list[dict[str, object]] = []
    for name, fn in ops:
        stats = time_op(fn, warmup, iters)
        rows.append(
            {
                "tensor": label,
                "op": name,
                "bytes": fp8_bytes,
                **stats,
                "gbps_p50": fp8_bytes / (stats["p50_us"] * 1e-6) / GIB,
            }
        )
    return rows


def bench_expert_bundle(warmup: int, iters: int) -> list[dict[str, object]]:
    """One full expert = gate + up ([1536,4096]) + down ([4096,1536])."""
    from moe_infinity._v4_fp4 import fp8_dequant_blockwise

    scalar = 0.0125
    tensors = [
        make_fp8(GATE_UP_SHAPE),
        make_fp8(GATE_UP_SHAPE),
        make_fp8(DOWN_SHAPE),
    ]
    grids = [uniform_grid(tuple(t.shape), scalar) for t in tensors]
    total_bytes = sum(t.numel() * t.element_size() for t in tensors)

    hosts = [
        torch.empty(tuple(t.shape), dtype=torch.uint8, pin_memory=True)
        for t in tensors
    ]
    devs = [
        torch.empty(tuple(t.shape), dtype=torch.uint8, device="cuda")
        for t in tensors
    ]

    def dequant_all() -> None:
        for t, g in zip(tensors, grids):
            fp8_dequant_blockwise(t, g)

    def h2d_all() -> None:
        for h, d in zip(hosts, devs):
            d.copy_(h, non_blocking=True)

    rows: list[dict[str, object]] = []
    for name, fn in (
        ("native dequant x3 (one expert)", dequant_all),
        ("H2D pinned copy x3 (one expert)", h2d_all),
    ):
        stats = time_op(fn, warmup, iters)
        rows.append(
            {
                "tensor": f"expert bundle ({total_bytes / 2**20:.1f} MB)",
                "op": name,
                "bytes": total_bytes,
                **stats,
                "gbps_p50": total_bytes / (stats["p50_us"] * 1e-6) / GIB,
            }
        )
    return rows


def print_table(rows: list[dict[str, object]]) -> None:
    print("| Tensor | Op | FP8 MB | p50 µs | p90 µs | eff GB/s (p50) |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for r in rows:
        print(
            f"| {r['tensor']} | {r['op']} | {r['bytes'] / 2**20:.2f} "
            f"| {r['p50_us']:.1f} | {r['p90_us']:.1f} "
            f"| {r['gbps_p50']:.1f} |"
        )


def print_ratios(rows: list[dict[str, object]]) -> dict[str, float]:
    ratios: dict[str, float] = {}
    by_tensor: dict[str, dict[str, float]] = {}
    for r in rows:
        bt = by_tensor.setdefault(str(r["tensor"]), {})
        op = str(r["op"])
        if op.startswith("native"):
            bt["dequant"] = float(r["p50_us"])
        elif op.startswith("H2D"):
            bt["h2d"] = float(r["p50_us"])
    print("\n| Tensor | dequant p50 / H2D p50 |")
    print("| --- | ---: |")
    for tensor, vals in by_tensor.items():
        if "dequant" in vals and "h2d" in vals:
            ratio = vals["dequant"] / vals["h2d"]
            ratios[tensor] = ratio
            print(f"| {tensor} | {ratio:.3f} |")
    return ratios


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("BLOCKED: CUDA is not available")
        return 1
    try:
        from moe_infinity._v4_fp4 import (  # noqa: F401
            fp8_dequant_blockwise,
        )
    except Exception as exc:
        print(f"BLOCKED: moe_infinity._v4_fp4 import failed: {exc}")
        return 1

    device_name = torch.cuda.get_device_name(0)
    print(f"GPU: {device_name}")
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"warmup={args.warmup} iters={args.iters}\n")

    rows: list[dict[str, object]] = []
    rows += bench_shape(
        "gate/up [1536,4096]", GATE_UP_SHAPE, args.warmup, args.iters
    )
    rows += bench_shape("down [4096,1536]", DOWN_SHAPE, args.warmup, args.iters)
    rows += bench_expert_bundle(args.warmup, args.iters)

    print_table(rows)
    ratios = print_ratios(rows)

    if args.out:
        payload = {
            "gpu": device_name,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "warmup": args.warmup,
            "iters": args.iters,
            "rows": rows,
            "dequant_over_h2d_p50": ratios,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
