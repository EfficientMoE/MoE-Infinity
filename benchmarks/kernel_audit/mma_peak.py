"""Sustained peak of the sm_120 block-scaled MMA instructions (gate Rule C).

DeepSeek-V4's official W4A8 recipe (e4m3 activations x e2m1 weights, ue8m0
scales) needs `mma.sync ... kind::mxf8f6f4.block_scale`; FP4 x FP4 can use
`kind::mxf4nvf4` (twice the K per instruction). This measures both, plus the
cuBLAS FP8 GEMM that defines the audit's FP8 roof, under the same sustained
load (3 s back to back, so the 300 W cap sets the clock), with SM clock and
power from nvidia-smi.

Each warp keeps 4 independent accumulators in flight; operands are register
resident, so the number is the instruction issue limit, not a GEMM.

  CUDA_VISIBLE_DEVICES=5 python -m benchmarks.kernel_audit.mma_peak
"""

from __future__ import annotations

import json
import time

import torch
from torch.utils.cpp_extension import load_inline

from benchmarks.kernel_audit.common import GpuMonitor, gpu_identity

_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

template <int KIND>
__device__ __forceinline__ void mma(float* d, const uint32_t* a,
                                    const uint32_t* b, uint32_t sf) {
  if (KIND == 0)
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale."
        "scale_vec::1X.f32.e2m1.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},%10,{%11,%12},"
        "%10,{%11,%12};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "r"(sf), "h"((uint16_t)0), "h"((uint16_t)0));
  else
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale."
        "scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},%10,{%11,%12},"
        "%10,{%11,%12};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "r"(sf), "h"((uint16_t)0), "h"((uint16_t)0));
}

template <int KIND>
__global__ void bench(float* out, int iters, uint32_t seed) {
  uint32_t a[4], b[2];
  for (int i = 0; i < 4; ++i) a[i] = (seed * (threadIdx.x + i)) & 0x3c3c3c3cu;
  for (int i = 0; i < 2; ++i) b[i] = (seed * (threadIdx.x + 7 * i)) & 0x3f3f3f3fu;
  float d[4][4] = {};
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int j = 0; j < 4; ++j) mma<KIND>(d[j], a, b, 0x7f7f7f7fu);
  }
  float s = 0.f;
  for (int j = 0; j < 4; ++j)
    for (int i = 0; i < 4; ++i) s += d[j][i];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

void run(int kind, int blocks, int iters, torch::Tensor out) {
  auto s = at::cuda::getCurrentCUDAStream();
  if (kind == 0)
    bench<0><<<blocks, 128, 0, s>>>(out.data_ptr<float>(), iters, 3);
  else
    bench<1><<<blocks, 128, 0, s>>>(out.data_ptr<float>(), iters, 3);
}
"""

_ITERS = 4096


def _sustained(fn, seconds: float = 3.0):
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
    tail = sorted(per_call[len(per_call) // 2 :])
    return tail[len(tail) // 2], mon.summary()


def main() -> None:
    ext = load_inline(
        "kaudit_mma_peak",
        cpp_sources="void run(int kind, int blocks, int iters, torch::Tensor out);",
        cuda_sources=_SRC,
        functions=["run"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a"],
    )
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    blocks = sms * 8  # 32 warps per SM
    out = torch.empty(blocks * 128, device="cuda")
    print(json.dumps({"gpu": gpu_identity()}))
    for kind, name, k in (
        (0, "mxf8f6f4 m16n8k32 e2m1 x e4m3 (W4A8)", 32),
        (1, "mxf4nvf4 m16n8k64 e2m1 x e2m1 (W4A4)", 64),
    ):
        ms, tel = _sustained(lambda: ext.run(kind, blocks, _ITERS, out))
        flops = 2.0 * 16 * 8 * k * 4 * _ITERS * blocks * 4
        print(
            json.dumps(
                {"instr": name, "tflops": flops / ms / 1e9, "ms": ms, **tel}
            )
        )
    m = n = k = 8192
    a = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
    one = torch.ones((), device="cuda")
    ms, tel = _sustained(
        lambda: torch._scaled_mm(a, b.T, one, one, out_dtype=torch.bfloat16)
    )
    print(
        json.dumps(
            {
                "instr": "cuBLAS FP8 GEMM 8192^3 (audit FP8 roof)",
                "tflops": 2 * m * n * k / ms / 1e9,
                "ms": ms,
                **tel,
            }
        )
    )


if __name__ == "__main__":
    main()
