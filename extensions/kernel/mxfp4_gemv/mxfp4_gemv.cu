// Small-M (M <= 8) MXFP4 x BF16 GEMM for decode on sm_120a
// (EfficientMoE/MoE-Infinity#312).
//
//   out[M, N] = x[M, K] @ dequant(w[N, K/2], s[N, K/32])^T (+ bias[N])
//
// One CTA owns a full-K strip of 16 weight rows, so there is no split-K
// fix-up across CTAs: the CTA's warps take disjoint K slices and reduce their
// accumulators in shared memory. Each warp step is one
// mma.sync.m16n8k16 (weights = A, 16 rows; tokens = B, 8 columns, M <= 8).
//
// Fragment mapping. The four lanes of an A row group (lane / 4) each load
// one whole 16-byte scale group (32 E2M1 values) of rows r and r + 8 with a
// single 16-byte load. Because a dot product does not depend on the order of
// k, logical k slots 2q, 2q+1 (lane q = lane % 4) of mma step `st` are bound to
// physical k = 32 * group + 4 * st + {0, 1} and slots 2q+8, 2q+9 to + {2, 3};
// the B fragment of the same lane reads the activations at exactly those k.
// Every A register is one packed byte (lo, hi nibble = two consecutive k),
// decoded with cvt.rn.f16x2.e2m1x2 and scaled once per group.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

__device__ __forceinline__ void e2m1x4_to_f16x2(uint32_t w, uint32_t (&h)[4]) {
  asm("{\n"
      ".reg .b8 b0, b1, b2, b3;\n"
      "mov.b32 {b0, b1, b2, b3}, %4;\n"
      "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
      "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
      "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
      "cvt.rn.f16x2.e2m1x2 %3, b3;\n"
      "}\n"
      : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
      : "r"(w));
}

// f16x2 holding exact E2M1 values -> bf16x2, multiplied by the group scale.
__device__ __forceinline__ uint32_t f16x2_to_scaled_bf16x2(uint32_t h,
                                                           __nv_bfloat162 sc) {
  __half2 hv = *reinterpret_cast<__half2*>(&h);
  float2 f = __half22float2(hv);
  __nv_bfloat162 b = __floats2bfloat162_rn(f.x, f.y);
  b = __hmul2(b, sc);
  return *reinterpret_cast<uint32_t*>(&b);
}

__device__ __forceinline__ __nv_bfloat162 e8m0_scale(uint32_t s) {
  // bf16 bits (s << 7) == 2^(s - 127); s == 0 maps to +0.
  uint32_t bits = (s << 7) | (s << 23);
  return *reinterpret_cast<__nv_bfloat162*>(&bits);
}

__device__ __forceinline__ void mma_bf16_16816(float (&c)[4],
                                               const uint32_t (&a)[4],
                                               uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int WARPS, int MAXU>
__global__ void __launch_bounds__(WARPS * 32)
    mxfp4_gemv_mma_kernel(const __nv_bfloat16* __restrict__ x,
                          const uint8_t* __restrict__ w,
                          const uint8_t* __restrict__ s,
                          const __nv_bfloat16* __restrict__ bias,
                          __nv_bfloat16* __restrict__ out, int M, int N, int K,
                          int64_t stride_xm, int64_t stride_wn,
                          int64_t stride_sn, int64_t stride_om) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int r = lane >> 2;  // A row group / B token
  const int q = lane & 3;   // A k-pair / B k-pair
  const int n0 = blockIdx.x * 16;
  const int row0 = n0 + r;
  const int row1 = n0 + r + 8;
  const bool v0 = row0 < N;
  const bool v1 = row1 < N;
  const bool tv = r < M;
  const int G = K >> 5;            // scale groups per row
  const int units = (G + 3) >> 2;  // 128-k units (4 groups, one per lane q)

  float c[4] = {0.f, 0.f, 0.f, 0.f};

  for (int base = warp; base < units; base += WARPS * MAXU) {
    uint4 w0[MAXU], w1[MAXU], xv[MAXU][4];
    uint32_t s0[MAXU], s1[MAXU];
    // Issue every load of this batch before any math.
#pragma unroll
    for (int i = 0; i < MAXU; ++i) {
      const int u = base + i * WARPS;
      const int g = u * 4 + q;
      const bool gv = u < units && g < G;
      w0[i] = (gv && v0) ? __ldg(reinterpret_cast<const uint4*>(
                               w + row0 * stride_wn + (int64_t)g * 16))
                         : make_uint4(0, 0, 0, 0);
      w1[i] = (gv && v1) ? __ldg(reinterpret_cast<const uint4*>(
                               w + row1 * stride_wn + (int64_t)g * 16))
                         : make_uint4(0, 0, 0, 0);
      s0[i] = (gv && v0) ? __ldg(s + row0 * stride_sn + g) : 0u;
      s1[i] = (gv && v1) ? __ldg(s + row1 * stride_sn + g) : 0u;
    }
#pragma unroll
    for (int i = 0; i < MAXU; ++i) {
      const int u = base + i * WARPS;
      const int g = u * 4 + q;
      const bool gv = u < units && g < G && tv;
      const uint4* xp =
          reinterpret_cast<const uint4*>(x + r * stride_xm + (int64_t)g * 32);
#pragma unroll
      for (int j = 0; j < 4; ++j)
        xv[i][j] = gv ? __ldg(xp + j) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int i = 0; i < MAXU; ++i) {
      if (base + i * WARPS >= units) break;
      const __nv_bfloat162 f0 = e8m0_scale(s0[i]);
      const __nv_bfloat162 f1 = e8m0_scale(s1[i]);
      const uint32_t wa[4] = {w0[i].x, w0[i].y, w0[i].z, w0[i].w};
      const uint32_t wb[4] = {w1[i].x, w1[i].y, w1[i].z, w1[i].w};
      const uint32_t xw[16] = {xv[i][0].x, xv[i][0].y, xv[i][0].z, xv[i][0].w,
                               xv[i][1].x, xv[i][1].y, xv[i][1].z, xv[i][1].w,
                               xv[i][2].x, xv[i][2].y, xv[i][2].z, xv[i][2].w,
                               xv[i][3].x, xv[i][3].y, xv[i][3].z, xv[i][3].w};
#pragma unroll
      for (int wd = 0; wd < 4; ++wd) {  // 4 bytes = 2 mma steps
        uint32_t h0[4], h1[4];
        e2m1x4_to_f16x2(wa[wd], h0);
        e2m1x4_to_f16x2(wb[wd], h1);
#pragma unroll
        for (int hs = 0; hs < 2; ++hs) {
          const int st = wd * 2 + hs;  // bytes 2*st and 2*st+1 of the group
          uint32_t a[4];
          a[0] = f16x2_to_scaled_bf16x2(h0[2 * hs], f0);
          a[1] = f16x2_to_scaled_bf16x2(h1[2 * hs], f1);
          a[2] = f16x2_to_scaled_bf16x2(h0[2 * hs + 1], f0);
          a[3] = f16x2_to_scaled_bf16x2(h1[2 * hs + 1], f1);
          mma_bf16_16816(c, a, xw[2 * st], xw[2 * st + 1]);
        }
      }
    }
  }

  // c[0], c[1]: weight row r, tokens 2q, 2q+1; c[2], c[3]: row r + 8.
  __shared__ float red[WARPS][32][4];
#pragma unroll
  for (int j = 0; j < 4; ++j) red[warp][lane][j] = c[j];
  __syncthreads();
  if (warp == 0) {
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int wi = 0; wi < WARPS; ++wi)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[j] += red[wi][lane][j];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int row = (j < 2) ? row0 : row1;
      const int tok = 2 * q + (j & 1);
      if (row < N && tok < M) {
        float v = acc[j];
        if (bias != nullptr) v += __bfloat162float(bias[row]);
        out[tok * stride_om + row] = __float2bfloat16_rn(v);
      }
    }
  }
}

}  // namespace

// Returns false (and launches nothing) when the shape / alignment is outside
// what the kernel supports; the caller then uses the Triton path.
bool mxfp4_gemv_small_launch(const void* x, const void* w, const void* s,
                             const void* bias, void* out, int M, int N, int K,
                             int64_t stride_xm, int64_t stride_wn,
                             int64_t stride_sn, int64_t stride_om, int warps,
                             cudaStream_t stream) {
  if (M < 1 || M > 8 || K % 32 != 0) return false;
  if (stride_wn % 16 != 0 || stride_xm % 8 != 0) return false;
  if ((reinterpret_cast<uintptr_t>(w) | reinterpret_cast<uintptr_t>(x)) % 16)
    return false;
  const dim3 grid((N + 15) / 16);
  const auto* xb = static_cast<const __nv_bfloat16*>(x);
  const auto* wb = static_cast<const uint8_t*>(w);
  const auto* sb = static_cast<const uint8_t*>(s);
  const auto* bb = static_cast<const __nv_bfloat16*>(bias);
  auto* ob = static_cast<__nv_bfloat16*>(out);
  switch (warps) {
    case 4:
      mxfp4_gemv_mma_kernel<4, 6>
          <<<grid, 128, 0, stream>>>(xb, wb, sb, bb, ob, M, N, K, stride_xm,
                                     stride_wn, stride_sn, stride_om);
      break;
    case 16:
      mxfp4_gemv_mma_kernel<16, 2>
          <<<grid, 512, 0, stream>>>(xb, wb, sb, bb, ob, M, N, K, stride_xm,
                                     stride_wn, stride_sn, stride_om);
      break;
    default:
      mxfp4_gemv_mma_kernel<8, 3>
          <<<grid, 256, 0, stream>>>(xb, wb, sb, bb, ob, M, N, K, stride_xm,
                                     stride_wn, stride_sn, stride_om);
      break;
  }
  return cudaPeekAtLastError() == cudaSuccess;
}
