"""Measure the empirical roofline ceilings of the visible sm_120 GPU.

Ceilings: dense tensor-core throughput (BF16 / FP8 / FP4 via cuBLAS),
DRAM read / copy bandwidth, L2 read bandwidth, and kernel-launch overhead.

Usage:
  CUDA_VISIBLE_DEVICES=5 python -m benchmarks.kernel_audit.ceilings
"""

from __future__ import annotations

import time

import torch
import triton
import triton.language as tl

from benchmarks.kernel_audit.common import (
    RESULTS_DIR,
    GpuMonitor,
    dump_json,
    gpu_identity,
    other_gpu_procs,
)


@triton.jit
def _read_kernel(
    x_ptr, out_ptr, n_elem, LOOPS: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for _ in range(LOOPS):
        for start in range(pid * BLOCK, n_elem, nprog * BLOCK):
            offs = start + tl.arange(0, BLOCK)
            # .cg: cache in L2 only, so repeated passes measure L2, not L1.
            v = tl.load(
                x_ptr + offs,
                mask=offs < n_elem,
                other=0.0,
                cache_modifier=".cg",
            )
            acc += v
    tl.store(out_ptr + pid * BLOCK + tl.arange(0, BLOCK), acc)


def _to_blocked(m: torch.Tensor) -> torch.Tensor:
    """cuBLAS block-scale swizzle (128x4 tiles), as in torch's test utils."""
    rows, cols = m.shape
    rb, cb = -(-rows // 128), -(-cols // 4)
    padded = torch.zeros(rb * 128, cb * 4, device=m.device, dtype=m.dtype)
    padded[:rows, :cols] = m
    blocks = padded.view(rb, 128, cb, 4).permute(0, 2, 1, 3)
    return (
        blocks.reshape(-1, 4, 32, 4)
        .transpose(1, 2)
        .reshape(-1, 32, 16)
        .flatten()
    )


def _sustained(fn, seconds: float = 3.0) -> tuple[float, dict]:
    """Run fn back-to-back for `seconds`; returns median per-call ms + telemetry."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    per_call = []
    with GpuMonitor(50) as mon:
        t0 = time.time()
        while time.time() - t0 < seconds:
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(5):
                fn()
            e.record()
            e.synchronize()
            per_call.append(s.elapsed_time(e) / 5)
    per_call.sort()
    tail = per_call[len(per_call) // 2 :]  # steady (power-throttled) half
    return sorted(tail)[len(tail) // 2], mon.summary()


def gemm_bf16(m: int, n: int, k: int) -> dict:
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    ms, tel = _sustained(lambda: a @ b.T)
    return {
        "m": m,
        "n": n,
        "k": k,
        "ms": ms,
        "tflops": 2 * m * n * k / ms / 1e9,
        **tel,
    }


def gemm_fp8(m: int, n: int, k: int) -> dict:
    a = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
    one = torch.ones((), device="cuda")
    fn = lambda: torch._scaled_mm(a, b.T, one, one, out_dtype=torch.bfloat16)  # noqa: E731
    fn()
    ms, tel = _sustained(fn)
    return {
        "m": m,
        "n": n,
        "k": k,
        "ms": ms,
        "tflops": 2 * m * n * k / ms / 1e9,
        **tel,
    }


def gemm_fp4(m: int, n: int, k: int) -> dict:
    """NVFP4 (e2m1 x2, e4m3 scale per 16) through torch._scaled_mm/cuBLASLt."""
    to_blocked = _to_blocked
    a = torch.randint(
        0, 255, (m, k // 2), device="cuda", dtype=torch.uint8
    ).view(torch.float4_e2m1fn_x2)
    b = torch.randint(
        0, 255, (n, k // 2), device="cuda", dtype=torch.uint8
    ).view(torch.float4_e2m1fn_x2)
    sa = to_blocked(
        torch.ones(m, k // 16, device="cuda", dtype=torch.float8_e4m3fn)
    )
    sb = to_blocked(
        torch.ones(n, k // 16, device="cuda", dtype=torch.float8_e4m3fn)
    )
    fn = lambda: torch._scaled_mm(a, b.T, sa, sb, out_dtype=torch.bfloat16)  # noqa: E731
    fn()
    ms, tel = _sustained(fn)
    return {
        "m": m,
        "n": n,
        "k": k,
        "ms": ms,
        "tflops": 2 * m * n * k / ms / 1e9,
        **tel,
    }


def dram_read(gib: float = 4.0) -> dict:
    n = int(gib * 2**30 // 4)
    x = torch.ones(n, device="cuda", dtype=torch.float32)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    best = None
    for per_sm in (4, 8, 16):
        grid = sms * per_sm
        out = torch.empty(grid * 4096, device="cuda", dtype=torch.float32)
        fn = lambda: _read_kernel[(grid,)](
            x, out, n, LOOPS=1, BLOCK=4096, num_warps=8
        )  # noqa: E731
        ms, tel = _sustained(fn, 2.0)
        gbs = n * 4 / ms / 1e6
        if best is None or gbs > best["gbs"]:
            best = {
                "bytes": n * 4,
                "ms": ms,
                "gbs": gbs,
                "ctas_per_sm": per_sm,
                **tel,
            }
    return best


def dram_copy(gib: float = 2.0) -> dict:
    n = int(gib * 2**30 // 2)
    src = torch.ones(n, device="cuda", dtype=torch.bfloat16)
    dst = torch.empty_like(src)
    ms, tel = _sustained(lambda: dst.copy_(src), 2.0)
    return {
        "bytes_moved": 2 * n * 2,
        "ms": ms,
        "gbs": 2 * n * 2 / ms / 1e6,
        **tel,
    }


def l2_read(mib: int = 32, loops: int = 40) -> dict:
    n = mib * 2**20 // 4
    x = torch.ones(n, device="cuda", dtype=torch.float32)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    grid = sms * 8
    out = torch.empty(grid * 2048, device="cuda", dtype=torch.float32)
    fn = lambda: _read_kernel[(grid,)](
        x, out, n, LOOPS=loops, BLOCK=2048, num_warps=8
    )  # noqa: E731
    ms, tel = _sustained(fn, 2.0)
    return {
        "working_set_mib": mib,
        "ms": ms,
        "gbs": n * 4 * loops / ms / 1e6,
        **tel,
    }


def launch_overhead() -> dict:
    x = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        x.add_(1)
    torch.cuda.synchronize()
    eager_us = (time.perf_counter() - t0) / n * 1e6
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(200):
            x.add_(1)
    g.replay()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(10):
        g.replay()
    e.record()
    e.synchronize()
    graph_us = s.elapsed_time(e) * 1e3 / 2000
    return {"eager_launch_us": eager_us, "graph_kernel_us": graph_us}


def main() -> None:
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    res: dict = {"gpu": gpu_identity(), "other_procs": other_gpu_procs()}
    res["launch"] = launch_overhead()
    res["dram_read"] = dram_read()
    res["dram_copy"] = dram_copy()
    res["l2_read"] = l2_read()
    res["bf16"] = [gemm_bf16(8192, 8192, 8192), gemm_bf16(16384, 8192, 8192)]
    try:
        res["fp8"] = [gemm_fp8(8192, 8192, 8192), gemm_fp8(16384, 8192, 8192)]
    except Exception as exc:  # pragma: no cover - capability probe
        res["fp8"] = {"error": repr(exc)[:300]}
    try:
        res["fp4"] = [gemm_fp4(8192, 8192, 8192), gemm_fp4(16384, 8192, 16384)]
    except Exception as exc:  # pragma: no cover - capability probe
        res["fp4"] = {"error": repr(exc)[:300]}
    res["other_procs_after"] = other_gpu_procs()
    out = RESULTS_DIR / "ceilings.json"
    dump_json(out, res)
    import json

    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
