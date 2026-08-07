// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team
//
// FP8 (E4M3) block-scale weight -> BF16 dequant for GLM-5.2-FP8 routed experts.
// Weight: [N, K] float8_e4m3fn; Scale: [ceil(N/128), ceil(K/128)] float32.
// Each 128x128 block of weights shares one scale (weight_scale_inv =
// multiplier). Output: [N, K] bfloat16.
//
// Reference: moe_infinity/utils/fp8.py::dequant_fp8_blockwise
//   w_f32 = weight.float()
//   s_full = scale.repeat_interleave(128, dim=0).repeat_interleave(128,
//   dim=1)[:N, :K] out = (w_f32 * s_full).to(bfloat16)

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

// One thread per output element (row, col).
// Reads fp8 byte, interprets as float8_e4m3fn, multiplies by block scale.
// fp8_e4m3fn: sign=1, exp=4, mantissa=3, bias=7, no inf (NaN=0x7F).
__device__ __forceinline__ float fp8_e4m3fn_to_float(uint8_t bits) {
  // Decode float8_e4m3fn (bias=7, no inf, NaN=0x7F/0xFF)
  int sign = (bits >> 7) & 1;
  int exp = (bits >> 3) & 0xF;
  int mant = bits & 0x7;

  float val;
  if (exp == 0) {
    // subnormal: value = (-1)^sign * 2^(-6) * (mant/8)
    val = (float)mant / 8.0f * 0.015625f;  // 2^(-6) = 0.015625
  } else if (exp == 15 && mant == 7) {
    // NaN (0x7F or 0xFF)
    val = 0.0f;  // treat NaN as 0 for dequant
  } else {
    // normal: value = (-1)^sign * 2^(exp-7) * (1 + mant/8)
    val = ldexpf(1.0f + (float)mant / 8.0f, exp - 7);
  }
  return sign ? -val : val;
}

__global__ void fp8_dequant_blockwise_kernel(
    const uint8_t* __restrict__ weight,  // [N, K] fp8 bytes
    const float* __restrict__ scale,     // [SN, SK] float32, SN=ceil(N/128),
                                         // SK=ceil(K/128)
    __nv_bfloat16* __restrict__ out,     // [N, K] bf16
    int N, int K, int SK                 // SK = ceil(K/128)
) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long total = (long)N * K;
  if (idx >= total) return;

  int row = (int)(idx / K);
  int col = (int)(idx % K);

  // Block indices
  int br = row / 128;
  int bc = col / 128;
  float s = scale[(long)br * SK + bc];

  float w = fp8_e4m3fn_to_float(weight[idx]);
  out[idx] = __float2bfloat16(w * s);
}

}  // namespace

void fp8_dequant_blockwise_cuda(const void* weight, const void* scale,
                                void* out, int N, int K, cudaStream_t stream) {
  int SK = (K + 127) / 128;
  long total = (long)N * K;
  int threads = 256;
  long blocks = (total + threads - 1) / threads;
  fp8_dequant_blockwise_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(weight),
      reinterpret_cast<const float*>(scale),
      reinterpret_cast<__nv_bfloat16*>(out), N, K, SK);
}
