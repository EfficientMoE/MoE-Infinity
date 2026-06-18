// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team
//
// FP4 (E2M1) packed weight -> BF16 dequant for DeepSeek-V4-Flash routed
// experts. Weight: [N, K/2] uint8, two E2M1 values per byte packed along K (low
// nibble is the even-K element). Scale: [N, K/32] ue8m0 (one scale per 32
// K-elements). Matches the reference dequant_fp4_e2m1 (E2M1 lookup table x
// 2^(e-127)).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {
// E2M1 value table (index = 4-bit code).
__constant__ float kE2M1[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,
                                4.0f,  6.0f,  -0.0f, -0.5f, -1.0f, -1.5f,
                                -2.0f, -3.0f, -4.0f, -6.0f};

__global__ void fp4_dequant_kernel(const uint8_t* __restrict__ packed,
                                   const uint8_t* __restrict__ scale_e8m0,
                                   __nv_bfloat16* __restrict__ out, int N,
                                   int K) {
  // One thread per packed byte => 2 output elements.
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long total_bytes = (long)N * (K / 2);
  if (idx >= total_bytes) return;
  int row = idx / (K / 2);
  int byte_col = idx % (K / 2);
  int k0 = byte_col * 2;

  uint8_t b = packed[idx];
  int lo = b & 0x0F;
  int hi = (b >> 4) & 0x0F;

  int sblocks = K / 32;
  // scale for element k = scale_e8m0[row, k/32]; both k0 and k0+1 share a block
  // unless k0 is the last in a block (k0 and k0+1 always in same 32-block since
  // 32 is even and k0 is even).
  int sidx0 = row * sblocks + (k0 / 32);
  float s = exp2f((float)scale_e8m0[sidx0] - 127.0f);

  long out0 = (long)row * K + k0;
  out[out0] = __float2bfloat16(kE2M1[lo] * s);
  out[out0 + 1] = __float2bfloat16(kE2M1[hi] * s);
}
}  // namespace

void fp4_dequant_to_bf16(const void* packed, const void* scale_e8m0, void* out,
                         int N, int K, cudaStream_t stream) {
  long total_bytes = (long)N * (K / 2);
  int threads = 256;
  long blocks = (total_bytes + threads - 1) / threads;
  fp4_dequant_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(packed),
      reinterpret_cast<const uint8_t*>(scale_e8m0),
      reinterpret_cast<__nv_bfloat16*>(out), N, K);
}
