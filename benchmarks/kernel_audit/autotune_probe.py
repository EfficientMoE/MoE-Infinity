"""First-call cost of the Triton MXFP4 GEMM for a never-seen token count M.

The original `_fused_mxfp4_gemm_kernel` was autotuned with key=["M", "N", "K"],
so every new per-expert token count re-ran the autotuner (prefill produces a
different M for almost every expert). The kernels now use a static per-M-bucket
config table and `warmup_mxfp4_gemm` compiles every bucket at load time; this
probe times that warm-up once, then the first call for each unseen M.
"""

from __future__ import annotations

import json
import time

import torch

from benchmarks.kernel_audit.common import RESULTS_DIR, dump_json


def main() -> None:
    from moe_infinity.kernel import mxfp4_gemm as mg

    H, N = 2880, 5760
    w = torch.randint(0, 255, (N, H // 2), device="cuda", dtype=torch.uint8)
    s = torch.randint(118, 124, (N, H // 32), device="cuda", dtype=torch.uint8)
    b = torch.zeros(N, device="cuda", dtype=torch.bfloat16)
    out = []
    warmup = getattr(mg, "warmup_mxfp4_gemm", None)
    if warmup is not None:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        warmup(N, H, w.device)
        torch.cuda.synchronize()
        out.append({"warmup_ms": (time.perf_counter() - t0) * 1e3})
        print(json.dumps(out[-1]))
    for M in (1, 37, 41, 53, 97, 211, 1009, 3001, 7777):
        x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        mg.fused_mxfp4_gemm(x, w, s, b)
        torch.cuda.synchronize()
        first = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        for _ in range(10):
            mg.fused_mxfp4_gemm(x, w, s, b)
        torch.cuda.synchronize()
        steady = (time.perf_counter() - t0) * 1e3 / 10
        out.append({"M": M, "first_call_ms": first, "steady_ms": steady})
        print(json.dumps(out[-1]))
    dump_json(RESULTS_DIR / "autotune_probe.json", out)


if __name__ == "__main__":
    main()
