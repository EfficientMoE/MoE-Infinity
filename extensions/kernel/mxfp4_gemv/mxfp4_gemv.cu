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

template <int WARPS, int MAXU, int MT>
__global__ void __launch_bounds__(WARPS * 32)
    mxfp4_gemv_sm120_kernel(const __nv_bfloat16* __restrict__ x,
                            const uint8_t* __restrict__ w,
                            const uint8_t* __restrict__ s,
                            const __nv_bfloat16* __restrict__ bias,
                            __nv_bfloat16* __restrict__ out, int M, int N,
                            int K, int64_t stride_xm, int64_t stride_wn,
                            int64_t stride_sn, int64_t stride_om) {
  // MT m16 tiles (16 * MT weight rows) per CTA share each B fragment.
  constexpr int RH = 2 * MT;  // row halves: rows r + 8 * h, h < RH
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int r = lane >> 2;  // A row group / B token
  const int q = lane & 3;   // A k-pair / B k-pair
  const int n0 = blockIdx.x * 16 * MT;
  const bool tv = r < M;
  const int G = K >> 5;            // scale groups per row
  const int units = (G + 3) >> 2;  // 128-k units (4 groups, one per lane q)

  float c[MT][4];
#pragma unroll
  for (int t = 0; t < MT; ++t)
#pragma unroll
    for (int j = 0; j < 4; ++j) c[t][j] = 0.f;

  for (int base = warp; base < units; base += WARPS * MAXU) {
    uint4 wv[MAXU][RH], xv[MAXU][4];
    uint32_t sv[MAXU][RH];
    // Issue every load of this batch before any math.
#pragma unroll
    for (int i = 0; i < MAXU; ++i) {
      const int u = base + i * WARPS;
      const int g = u * 4 + q;
      const bool gv = u < units && g < G;
#pragma unroll
      for (int h = 0; h < RH; ++h) {
        const int row = n0 + r + 8 * h;
        const bool ok = gv && row < N;
        wv[i][h] = ok ? __ldg(reinterpret_cast<const uint4*>(
                            w + row * stride_wn + (int64_t)g * 16))
                      : make_uint4(0, 0, 0, 0);
        sv[i][h] = ok ? __ldg(s + row * stride_sn + g) : 0u;
      }
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
      const uint32_t xw[16] = {xv[i][0].x, xv[i][0].y, xv[i][0].z, xv[i][0].w,
                               xv[i][1].x, xv[i][1].y, xv[i][1].z, xv[i][1].w,
                               xv[i][2].x, xv[i][2].y, xv[i][2].z, xv[i][2].w,
                               xv[i][3].x, xv[i][3].y, xv[i][3].z, xv[i][3].w};
#pragma unroll
      for (int t = 0; t < MT; ++t) {
        const __nv_bfloat162 f0 = e8m0_scale(sv[i][2 * t]);
        const __nv_bfloat162 f1 = e8m0_scale(sv[i][2 * t + 1]);
        const uint4 wa4 = wv[i][2 * t], wb4 = wv[i][2 * t + 1];
        const uint32_t wa[4] = {wa4.x, wa4.y, wa4.z, wa4.w};
        const uint32_t wb[4] = {wb4.x, wb4.y, wb4.z, wb4.w};
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
            mma_bf16_16816(c[t], a, xw[2 * st], xw[2 * st + 1]);
          }
        }
      }
    }
  }

  // c[t][0..1]: weight row n0 + 16t + r, tokens 2q, 2q+1; c[t][2..3]: + 8.
  __shared__ float red[WARPS][MT][32][4];
#pragma unroll
  for (int t = 0; t < MT; ++t)
#pragma unroll
    for (int j = 0; j < 4; ++j) red[warp][t][lane][j] = c[t][j];
  __syncthreads();
  for (int idx = threadIdx.x; idx < MT * 32 * 4; idx += WARPS * 32) {
    const int t = idx / 128, ln = (idx / 4) % 32, j = idx % 4;
    float v = 0.f;
#pragma unroll
    for (int wi = 0; wi < WARPS; ++wi) v += red[wi][t][ln][j];
    const int row = n0 + 16 * t + (ln >> 2) + (j < 2 ? 0 : 8);
    const int tok = 2 * (ln & 3) + (j & 1);
    if (row < N && tok < M) {
      if (bias != nullptr) v += __bfloat162float(bias[row]);
      out[tok * stride_om + row] = __float2bfloat16_rn(v);
    }
  }
}

}  // namespace

// Returns false (and launches nothing) when the shape / alignment is outside
// what the kernel supports; the caller then uses the Triton path.
bool mxfp4_gemv_sm120_launch(const void* x, const void* w, const void* s,
                             const void* bias, void* out, int M, int N, int K,
                             int64_t stride_xm, int64_t stride_wn,
                             int64_t stride_sn, int64_t stride_om, int variant,
                             cudaStream_t stream) {
  if (M < 1 || M > 8 || K % 32 != 0) return false;
  if (stride_wn % 16 != 0 || stride_xm % 8 != 0) return false;
  if ((reinterpret_cast<uintptr_t>(w) | reinterpret_cast<uintptr_t>(x)) % 16)
    return false;
  const auto* xb = static_cast<const __nv_bfloat16*>(x);
  const auto* wb = static_cast<const uint8_t*>(w);
  const auto* sb = static_cast<const uint8_t*>(s);
  const auto* bb = static_cast<const __nv_bfloat16*>(bias);
  auto* ob = static_cast<__nv_bfloat16*>(out);
#define MXFP4_GEMV_LAUNCH(W, U, T)                                      \
  mxfp4_gemv_sm120_kernel<W, U, T>                                      \
      <<<(N + 16 * T - 1) / (16 * T), W * 32, 0, stream>>>(             \
          xb, wb, sb, bb, ob, M, N, K, stride_xm, stride_wn, stride_sn, \
          stride_om)
  switch (variant) {
    case 0:
      MXFP4_GEMV_LAUNCH(4, 6, 1);
      break;
    case 1:
      MXFP4_GEMV_LAUNCH(8, 3, 1);
      break;
    case 2:
      MXFP4_GEMV_LAUNCH(16, 2, 1);
      break;
    case 3:
      MXFP4_GEMV_LAUNCH(4, 6, 2);
      break;
    case 4:
      MXFP4_GEMV_LAUNCH(8, 3, 2);
      break;
    case 5:
      MXFP4_GEMV_LAUNCH(16, 2, 2);
      break;
    default:
      return false;
  }
#undef MXFP4_GEMV_LAUNCH
  return cudaPeekAtLastError() == cudaSuccess;
}
