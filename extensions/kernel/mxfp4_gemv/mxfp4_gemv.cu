// Small-M (M <= 8) MXFP4 x BF16 GEMM for decode on sm_120a
// (EfficientMoE/MoE-Infinity#312).
//
//   out[M, N] = x[M, K] @ dequant(w[N, K/2], s[N, K/32])^T (+ bias[N])
//
// One CTA owns a full-K strip of weight rows, so there is no split-K fix-up
// across CTAs: the CTA's warps take disjoint K slices and reduce their
// accumulators in shared memory.
//
// * M == 2, wide N (variant 6): register-streaming FMA kernel
//   (mxfp4_gemv_m2_kernel). Each 2-warp CTA owns 2 weight rows (2880 CTAs
//   across 188 SMs, 16 CTAs/SM = 32 warps/SM resident in one wave), loads
//   128-byte coalesced weight strips and 512-byte coalesced activation strips
//   per warp step, decodes E2M1 once with cvt.rn.f16x2.e2m1x2 and accumulates
//   both token rows in registers before a warp shuffle + 32B shared-memory
//   reduction.
// * M = 3..8, all N (variant 8): mma.sync.m16n8k16 kernel
//   (mxfp4_gemv_fast_kernel) with single-batch K unrolling for K <= 4096,
//   activation loads issued ahead of cold-DRAM weight loads so shared x hits L2
//   without queueing behind weight sector misses, 2-op f16x2 -> scaled bf16x2
//   conversion ((h >> 3) & 0x0fff0fff | (h & 0x80008000)) * 2^112 * 2^(s-127),
//   16 warps x 16 rows @ 64 regs = 2 CTAs/SM, and an early 9-warp partial
//   reduction + bias prefetch (bar.sync 1, 288) overlapped with warps 0..6's
//   second K unit; narrow N at M <= 2 keeps variant 2 (16 warps x 16 rows).

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

// Direct bit-shift f16x2 (exact E2M1) -> bf16x2 (value * 2^-112), then two
// bf16x2 multiplies apply 2^112 (0x7780) and 2^(s - 127). Avoids widening to
// f32 and repacking to bf16x2 on the F2FP pipe.
__device__ __forceinline__ uint32_t f16x2_to_scaled_bf16x2_fast(uint32_t h,
                                                                uint32_t sc) {
  uint32_t b = ((h >> 3) & 0x0fff0fffu) | (h & 0x80008000u);
  uint32_t r;
  asm("mul.rn.bf16x2 %0, %1, %3;\n"
      "mul.rn.bf16x2 %0, %0, %2;\n"
      : "=r"(r)
      : "r"(b), "r"(sc), "r"(0x77807780u));
  return r;
}

__device__ __forceinline__ uint32_t e8m0_scale_u32(uint32_t s) {
  return (s << 7) | (s << 23);
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

__device__ __forceinline__ uint4 ldg128_nc_rt(const void* p, bool ok) {
  uint4 v;
  asm volatile(
      "{\n"
      ".reg .pred pr;\n"
      "setp.ne.b32 pr, %5, 0;\n"
      "mov.b64 {%0, %1}, 0;\n"
      "mov.b64 {%2, %3}, 0;\n"
      "@pr ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];\n"
      "}\n"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "l"(p), "r"((int)ok));
  return v;
}

__device__ __forceinline__ uint32_t ldg8_nc_rt(const void* p, bool ok) {
  uint32_t v;
  asm volatile(
      "{\n"
      ".reg .pred pr;\n"
      "setp.ne.b32 pr, %2, 0;\n"
      "mov.b32 %0, 0;\n"
      "@pr ld.global.nc.u8 %0, [%1];\n"
      "}\n"
      : "=r"(v)
      : "l"(p), "r"((int)ok));
  return v;
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

template <int WARPS, int MAXU, int RH, int MIN_BLOCKS>
__global__ void __launch_bounds__(WARPS * 32, MIN_BLOCKS)
    mxfp4_gemv_fast_kernel(const __nv_bfloat16* __restrict__ x,
                           const uint8_t* __restrict__ w,
                           const uint8_t* __restrict__ s,
                           const __nv_bfloat16* __restrict__ bias,
                           __nv_bfloat16* __restrict__ out, int M, int N, int K,
                           int64_t stride_xm, int64_t stride_wn,
                           int64_t stride_sn, int64_t stride_om) {
  constexpr int MT = (RH + 1) / 2;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int r = lane >> 2;
  const int q = lane & 3;
  const int n0 = blockIdx.x * 8 * RH;
  const bool tv = r < M;
  const int G = K >> 5;
  const int units = (G + 3) >> 2;

  float c[MT][4];

  auto run_batch = [&](int base, bool first_batch)
      __attribute__((always_inline)) {
    // Issue shared activation loads first so L2 hits on x do not queue behind
    // thousands of cold DRAM sector misses for w and s.
    uint4 xv[MAXU][4];
    {
      const uint4* xp_base =
          reinterpret_cast<const uint4*>(x + (int64_t)r * stride_xm) +
          base * 16 + q * 4;
#pragma unroll
      for (int i = 0; i < MAXU; ++i) {
        const int u = base + i * WARPS;
        const int g = u * 4 + q;
        const bool gv = u < units && g < G && tv;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          xv[i][j] = ldg128_nc_rt(xp_base + i * WARPS * 16 + j, gv);
        }
      }
    }
    uint4 wv[MAXU][RH];
    uint32_t sv[MAXU][RH];
    {
      const uint8_t* wp[RH];
      const uint8_t* sp[RH];
      bool row_ok[RH];
#pragma unroll
      for (int h = 0; h < RH; ++h) {
        const int row = n0 + r + 8 * h;
        row_ok[h] = row < N;
        wp[h] = w + (int64_t)row * stride_wn + (base * 4 + q) * 16;
        sp[h] = s + (int64_t)row * stride_sn + base * 4 + q;
      }
#pragma unroll
      for (int i = 0; i < MAXU; ++i) {
        const int u = base + i * WARPS;
        const int g = u * 4 + q;
        const bool gv = u < units && g < G;
#pragma unroll
        for (int h = 0; h < RH; ++h) {
          const bool ok = gv && row_ok[h];
          wv[i][h] = ldg128_nc_rt(wp[h] + i * WARPS * 64, ok);
          sv[i][h] = ldg8_nc_rt(sp[h] + i * WARPS * 4, ok);
        }
      }
    }
    if (first_batch) {
#pragma unroll
      for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int j = 0; j < 4; ++j) c[t][j] = 0.f;
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
        const uint32_t f0 = e8m0_scale_u32(sv[i][2 * t]);
        const uint4 wa4 = wv[i][2 * t];
        const uint32_t wa[4] = {wa4.x, wa4.y, wa4.z, wa4.w};
        uint32_t f1 = 0;
        uint32_t wb[4] = {0, 0, 0, 0};
        if constexpr (RH > 1) {
          f1 = e8m0_scale_u32(sv[i][2 * t + (RH > 1 ? 1 : 0)]);
          const uint4 wb4 = wv[i][2 * t + (RH > 1 ? 1 : 0)];
          wb[0] = wb4.x;
          wb[1] = wb4.y;
          wb[2] = wb4.z;
          wb[3] = wb4.w;
        }
#pragma unroll
        for (int wd = 0; wd < 4; ++wd) {
          uint32_t h0[4], h1[4];
          e2m1x4_to_f16x2(wa[wd], h0);
          if constexpr (RH > 1) e2m1x4_to_f16x2(wb[wd], h1);
#pragma unroll
          for (int hs = 0; hs < 2; ++hs) {
            const int st = wd * 2 + hs;
            uint32_t a[4];
            a[0] = f16x2_to_scaled_bf16x2_fast(h0[2 * hs], f0);
            a[1] = (RH > 1) ? f16x2_to_scaled_bf16x2_fast(h1[2 * hs], f1) : 0u;
            a[2] = f16x2_to_scaled_bf16x2_fast(h0[2 * hs + 1], f0);
            a[3] =
                (RH > 1) ? f16x2_to_scaled_bf16x2_fast(h1[2 * hs + 1], f1) : 0u;
            mma_bf16_16816(c[t], a, xw[2 * st], xw[2 * st + 1]);
          }
        }
      }
    }
  };

  if (units <= WARPS * MAXU) {
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
      for (int j = 0; j < 4; ++j) c[t][j] = 0.f;
    if (warp < units) run_batch(warp, true);
  } else {
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
      for (int j = 0; j < 4; ++j) c[t][j] = 0.f;
    for (int base = warp; base < units; base += WARPS * MAXU)
      run_batch(base, false);
  }

  if constexpr (WARPS == 16 && RH == 2) {
    // At K = 2880 (23 units), warps 7..15 only execute 1 unit and finish
    // halfway through the kernel while warps 0..6 execute 2 units. Have warps
    // 12..15 prefetch bias from DRAM and sum warps 7..15 behind a 9-warp named
    // barrier (bar.sync 1, 288) while warps 0..6 are still computing unit 2.
    __shared__ float red[4][WARPS][32];
#pragma unroll
    for (int j = 0; j < 4; ++j) red[j][warp][lane] = c[0][j];

    float v_hi = 0.f;
    const int k = warp - 12;
    const int out_row = n0 + r + ((k >= 2) ? 8 : 0);
    const int out_tok = 2 * q + (k & 1);
    if (warp >= 7) {
      float b_val = 0.f;
      if (k >= 0 && bias != nullptr && out_row < N && out_tok < M) {
        b_val = __bfloat162float(__ldg(bias + out_row));
      }
      asm volatile("bar.sync 1, 288;\n" ::: "memory");
      if (k >= 0) {
        v_hi = b_val;
#pragma unroll
        for (int wi = 7; wi < WARPS; ++wi) v_hi += red[k][wi][lane];
      }
    }
    __syncthreads();
    if (k >= 0) {
      float v = v_hi;
#pragma unroll
      for (int wi = 0; wi < 7; ++wi) v += red[k][wi][lane];
      if (out_row < N && out_tok < M) {
        out[out_tok * stride_om + out_row] = __float2bfloat16_rn(v);
      }
    }
  } else {
    constexpr int C_PER_T = (RH == 1) ? 2 : 4;
    __shared__ float red[WARPS][MT][32][C_PER_T];
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
      for (int j = 0; j < C_PER_T; ++j) red[warp][t][lane][j] = c[t][j];
    __syncthreads();
    for (int idx = threadIdx.x; idx < MT * 32 * C_PER_T; idx += WARPS * 32) {
      const int t = idx / (32 * C_PER_T);
      const int ln = (idx / C_PER_T) % 32;
      const int j = idx % C_PER_T;
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
}

template <int RN, int WARPS, int U, int MIN_BLOCKS>
__global__ void __launch_bounds__(WARPS * 32, MIN_BLOCKS)
    mxfp4_gemv_m2_kernel(const __nv_bfloat16* __restrict__ x,
                         const uint8_t* __restrict__ w,
                         const uint8_t* __restrict__ s,
                         const __nv_bfloat16* __restrict__ bias,
                         __nv_bfloat16* __restrict__ out, int M, int N, int K,
                         int64_t stride_xm, int64_t stride_wn,
                         int64_t stride_sn, int64_t stride_om) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int gl = lane >> 2;
  const int n0 = blockIdx.x * RN;
  const int G = K >> 5;

  const uint8_t* w_row[RN];
  const uint8_t* s_row[RN];
  bool row_ok[RN];
#pragma unroll
  for (int r = 0; r < RN; ++r) {
    const int row = n0 + r;
    row_ok[r] = row < N;
    w_row[r] = w + (int64_t)row * stride_wn + lane * 4;
    s_row[r] = s + (int64_t)row * stride_sn + gl;
  }
  const uint4* x0_ptr = reinterpret_cast<const uint4*>(x + lane * 8);
  const uint4* x1_ptr =
      reinterpret_cast<const uint4*>(x + (M > 1 ? stride_xm : 0) + lane * 8);

  float acc0[RN], acc1[RN];
#pragma unroll
  for (int r = 0; r < RN; ++r) {
    acc0[r] = 0.f;
    acc1[r] = 0.f;
  }

  for (int g_base = warp * 8; g_base < G; g_base += WARPS * U * 8) {
    uint32_t wv[U][RN];
    uint32_t sv[U][RN];
    uint4 xv0[U], xv1[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int g0 = g_base + u * WARPS * 8;
      const int g = g0 + gl;
      const bool gv = g < G;
#pragma unroll
      for (int r = 0; r < RN; ++r) {
        const bool ok = gv && row_ok[r];
        wv[u][r] =
            ok ? __ldg(reinterpret_cast<const uint32_t*>(w_row[r] + g0 * 16))
               : 0u;
        sv[u][r] = ok ? (uint32_t)__ldg(s_row[r] + g0) : 0u;
      }
      xv0[u] = gv ? __ldg(x0_ptr + g0 * 4) : make_uint4(0, 0, 0, 0);
      xv1[u] = gv ? __ldg(x1_ptr + g0 * 4) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      if (g_base + u * WARPS * 8 >= G) break;
      const uint32_t x0w[4] = {xv0[u].x, xv0[u].y, xv0[u].z, xv0[u].w};
      const uint32_t x1w[4] = {xv1[u].x, xv1[u].y, xv1[u].z, xv1[u].w};
      float x0e[4], x0o[4], x1e[4], x1o[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        x0e[k] = __int_as_float(x0w[k] << 16);
        x0o[k] = __int_as_float(x0w[k] & 0xffff0000u);
        x1e[k] = __int_as_float(x1w[k] << 16);
        x1o[k] = __int_as_float(x1w[k] & 0xffff0000u);
      }
#pragma unroll
      for (int r = 0; r < RN; ++r) {
        uint32_t h[4];
        e2m1x4_to_f16x2(wv[u][r], h);
        const float sc = __int_as_float(sv[u][r] << 23);
        float d0 = 0.f, d1 = 0.f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          __half2 hv = *reinterpret_cast<__half2*>(&h[k]);
          float2 wf = __half22float2(hv);
          d0 = fmaf(wf.x, x0e[k], fmaf(wf.y, x0o[k], d0));
          d1 = fmaf(wf.x, x1e[k], fmaf(wf.y, x1o[k], d1));
        }
        acc0[r] = fmaf(d0, sc, acc0[r]);
        acc1[r] = fmaf(d1, sc, acc1[r]);
      }
    }
  }

#pragma unroll
  for (int r = 0; r < RN; ++r) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      acc0[r] += __shfl_down_sync(0xffffffffu, acc0[r], off);
      acc1[r] += __shfl_down_sync(0xffffffffu, acc1[r], off);
    }
  }
  __shared__ float red[WARPS][2][RN];
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < RN; ++r) {
      red[warp][0][r] = acc0[r];
      red[warp][1][r] = acc1[r];
    }
  }
  __syncthreads();
  if (warp == 0 && lane < RN) {
    const int row = n0 + lane;
    if (row < N) {
      float b = bias ? __bfloat162float(bias[row]) : 0.f;
      float v0 = b, v1 = b;
#pragma unroll
      for (int w = 0; w < WARPS; ++w) {
        v0 += red[w][0][lane];
        v1 += red[w][1][lane];
      }
      out[row] = __float2bfloat16_rn(v0);
      if (M > 1) out[stride_om + row] = __float2bfloat16_rn(v1);
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
#define MXFP4_FAST_LAUNCH(W, U, RH, MB)                                 \
  mxfp4_gemv_fast_kernel<W, U, RH, MB>                                  \
      <<<(N + 8 * RH - 1) / (8 * RH), W * 32, 0, stream>>>(             \
          xb, wb, sb, bb, ob, M, N, K, stride_xm, stride_wn, stride_sn, \
          stride_om)
#define MXFP4_M2_LAUNCH(RN, W, U, MB)                                         \
  mxfp4_gemv_m2_kernel<RN, W, U, MB>                                          \
      <<<(N + RN - 1) / RN, W * 32, 0, stream>>>(xb, wb, sb, bb, ob, M, N, K, \
                                                 stride_xm, stride_wn,        \
                                                 stride_sn, stride_om)
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
    case 6:
      if (M > 2) return false;
      MXFP4_M2_LAUNCH(2, 2, 2, 16);
      break;
    case 8:
      MXFP4_FAST_LAUNCH(16, 2, 2, 2);
      break;
    default:
      return false;
  }
#undef MXFP4_GEMV_LAUNCH
#undef MXFP4_FAST_LAUNCH
#undef MXFP4_M2_LAUNCH
  return cudaPeekAtLastError() == cudaSuccess;
}
