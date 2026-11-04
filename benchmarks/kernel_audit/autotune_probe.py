"""First-call cost of the Triton MXFP4 GEMM for a never-seen token count M.

`_fused_mxfp4_gemm_kernel` is autotuned with key=["M", "N", "K"], so every new
per-expert token count re-runs the autotuner (prefill produces a different M
for almost every expert).
"""

from __future__ import annotations

import json
import time

import torch

from benchmarks.kernel_audit.common import RESULTS_DIR, dump_json


def main() -> None:
    from moe_infinity.kernel.mxfp4_gemm import fused_mxfp4_gemm

    H, N = 2880, 5760
    w = torch.randint(0, 255, (N, H // 2), device="cuda", dtype=torch.uint8)
    s = torch.randint(118, 124, (N, H // 32), device="cuda", dtype=torch.uint8)
    b = torch.zeros(N, device="cuda", dtype=torch.bfloat16)
    out = []
    for M in (1, 37, 41, 53, 97, 211, 1009):
        x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fused_mxfp4_gemm(x, w, s, b)
        torch.cuda.synchronize()
        first = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        for _ in range(10):
            fused_mxfp4_gemm(x, w, s, b)
        torch.cuda.synchronize()
        steady = (time.perf_counter() - t0) * 1e3 / 10
        out.append({"M": M, "first_call_ms": first, "steady_ms": steady})
        print(json.dumps(out[-1]))
    dump_json(RESULTS_DIR / "autotune_probe.json", out)


if __name__ == "__main__":
    main()
