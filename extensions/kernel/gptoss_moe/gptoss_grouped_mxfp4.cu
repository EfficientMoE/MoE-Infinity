// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// Grouped decode MoE for GPT-OSS resident MXFP4 experts (#313): every routed
// expert of a layer in two launches, no host sync, CUDA-graph capturable.
//
// Derived from the BF16 grouped decode kernel of #306 / #332
// (extensions/kernel/grouped_moe_decode.cu): mma.sync.m16n8k16 "swap-AB"
// (16 weight rows are the A operand, up to 8 tokens the B operand), fragments
// loaded straight from global memory with U-step register double buffering,
// warps split K and reduce through shared memory, the routing-weight multiply
// and scatter-add are fused into the down epilogue. Shared with #332 and to be
// deduplicated once both merge: ld_stream / mma_bf16 / stash_partials /
// sum_partials / pick and the launch-config scheme.
//
// Differences:
//  * Weights are MXFP4: per 64-wide K step a lane loads 8 bytes (16 E2M1
//    values, half of one 32-value scale group) of each of its two rows plus
//    the group's E8M0 byte, and decodes them in registers. Nibble bits go to
//    BF16 bits 6-8 / 15 (value * 2^-126, the bit trick of #312's
//    _e2m1_to_bf16), then one packed BF16 multiply by 2^(s-1) applies both
//    2^126 and the block scale; exact for E8M0 scales s >= 1.
//    The decode pairs nibbles (k0, k2) / (k1, k3); the token fragment is
//    permuted the same way with prmt, which leaves the dot product unchanged.
//  * Routing comes from top-k ids [T, k]: CTA row y of the grid serves the
//    y-th distinct routed expert (found in shared memory), so only
//    min(E, T*k) expert rows are launched and none is read twice.
//  * GPT-OSS gate/up rows are interleaved (gate = 2i, up = 2i + 1) and carry
//    biases; the epilogue applies the clamped SwiGLU-OAI
//    (up.clamp(-l, l) + 1) * g * sigmoid(alpha * g), g = min(gate, l), with
//    the same bf16 rounding points as the per-expert reference loop.
//  * Down: out[t] += bf16(w * bf16(acc + bias)) with bf16 atomics into the
//    bf16 output, zeroed by launch 1 (same rounding as the reference loop's
//    final_hidden[mask] += w * y).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace {

using bf16 = __nv_bfloat16;

constexpr int kMaxAssign = 256;   // T * top_k per call
constexpr int kMaxExperts = 256;  // experts per layer

struct Params {
  const bf16* x;  // [T, H]
  int64_t ldx;
  const int64_t* ids;   // [T, k]
  const bf16* rw;       // [T, k]
  const uint8_t* gu_w;  // [E, 2I, H/2]
  const uint8_t* gu_s;  // [E, 2I, H/32]
  const bf16* gu_b;     // [E, 2I]
  const uint8_t* d_w;   // [E, H, I/2]
  const uint8_t* d_s;   // [E, H, I/32]
  const bf16* d_b;      // [E, H]
  int E;
  int T;
  int K;  // top_k
  int H;
  int I;
  float alpha;
  float limit;
  bf16* act;      // [T * k, I]
  int* tok_list;  // [Y, T*k]: token | slot << 16
  int* tok_cnt;   // [Y]
  int* y_expert;  // [Y]
  bf16* out;      // [T, H]
  int pdl;
};

__device__ __forceinline__ uint2 ld_stream8(const uint8_t* p) {
  uint2 r;
  asm("ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%2];"
      : "=r"(r.x), "=r"(r.y)
      : "l"(p));
  return r;
}

__device__ __forceinline__ uint32_t ld_scale(const uint8_t* p) {
  uint16_t r;
  asm("ld.global.nc.u8 %0, [%1];" : "=h"(r) : "l"(p));
  return r;
}

__device__ __forceinline__ uint4 ld_act(const bf16* p) {
  return __ldg(reinterpret_cast<const uint4*>(p));
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], uint32_t a0,
                                         uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
  return r;
}

__device__ __forceinline__ uint32_t bf16x2_mul(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

// 4 packed bytes (8 E2M1 values, k0..k7) scaled by 2^(s-127) ->
// r[0] = (k0, k2), r[1] = (k1, k3), r[2] = (k4, k6), r[3] = (k5, k7).
// The bit trick yields value * 2^-126; one packed multiply by 2^(s-1)
// (exponent field s + 126) restores it and applies the scale for s <= 128;
// larger scales take a second multiply.
__device__ __forceinline__ void decode4(uint32_t w, uint32_t s,
                                        uint32_t (&r)[4]) {
  const uint32_t f = (s <= 128u ? s + 126u : 254u) << 7;
  const uint32_t m1 = f | (f << 16);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const uint32_t t = prmt(w, 0u, h ? 0x4342u : 0x4140u);
    const uint32_t lo = ((t << 6) & 0x01C001C0u) | ((t << 12) & 0x80008000u);
    const uint32_t hi = ((t << 2) & 0x01C001C0u) | ((t << 8) & 0x80008000u);
    r[2 * h] = bf16x2_mul(lo, m1);
    r[2 * h + 1] = bf16x2_mul(hi, m1);
  }
  if (s > 128u) {  // remaining 2^(s - 128)
    const uint32_t f2 = (s - 1u) << 7;
    const uint32_t m2 = f2 | (f2 << 16);
#pragma unroll
    for (int i = 0; i < 4; ++i) r[i] = bf16x2_mul(r[i], m2);
  }
}

// One warp: acc[mt][nt] += A_mt (16 x K) * B_nt (K x 8) over the 64-wide K
// steps step0, step0 + stride, ... Rows g / g+8 of tile mt: ra / rb (packed
// bytes) with scale rows sa / sb (nullptr = zero row); token g of tile nt is
// xb[nt] (nullptr = padding).
template <int MT, int NT, int U>
__device__ __forceinline__ void warp_tile(
    float (&acc)[MT][NT][4], const uint8_t* const (&ra)[MT],
    const uint8_t* const (&rb)[MT], const uint8_t* const (&sa)[MT],
    const uint8_t* const (&sb)[MT], const bf16* const (&xb)[NT], int nt_active,
    int nsteps, int step0, int stride, int t4) {
  const uint2 zero2 = make_uint2(0, 0);
  const uint4 zero = make_uint4(0, 0, 0, 0);
  for (int s = step0; s < nsteps; s += stride * U) {
    uint2 wa[U][MT];
    uint2 wb[U][MT];
    uint32_t qa[U][MT];
    uint32_t qb[U][MT];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int ss = s + u * stride;
      const int kb = ss * 32 + t4 * 8;    // byte offset in the row
      const int gi = ss * 2 + (t4 >> 1);  // scale group
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const bool va = ss < nsteps && ra[mt];
        const bool vb = ss < nsteps && rb[mt];
        wa[u][mt] = va ? ld_stream8(ra[mt] + kb) : zero2;
        wb[u][mt] = vb ? ld_stream8(rb[mt] + kb) : zero2;
        qa[u][mt] = va ? ld_scale(sa[mt] + gi) : 0u;
        qb[u][mt] = vb ? ld_scale(sb[mt] + gi) : 0u;
      }
    }
    uint4 xv[U][NT][2];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int ss = s + u * stride;
      const int k = ss * 64 + t4 * 16;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const bool v = ss < nsteps && nt < nt_active && xb[nt];
        xv[u][nt][0] = v ? ld_act(xb[nt] + k) : zero;
        xv[u][nt][1] = v ? ld_act(xb[nt] + k + 8) : zero;
      }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int ss = s + u * stride;
      if (ss >= nsteps) break;
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        // a / a2: row g, k0..7 / k8..15; b / b2: row g + 8.
        uint32_t a[1][4], a2[1][4], b[1][4], b2[1][4];
        decode4(wa[u][mt].x, qa[u][mt], a[0]);
        decode4(wa[u][mt].y, qa[u][mt], a2[0]);
        decode4(wb[u][mt].x, qb[u][mt], b[0]);
        decode4(wb[u][mt].y, qb[u][mt], b2[0]);
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          if (nt >= nt_active) break;
          const uint4 x0 = xv[u][nt][0];  // k0..k7  as (k0,k1) (k2,k3) ...
          const uint4 x1 = xv[u][nt][1];  // k8..k15
          // word 0 of the lane's 8 bytes holds k0..k7, word 1 k8..k15.
          mma_bf16(acc[mt][nt], a[0][0], b[0][0], a[0][1], b[0][1],
                   prmt(x0.x, x0.y, 0x5410u), prmt(x0.x, x0.y, 0x7632u));
          mma_bf16(acc[mt][nt], a[0][2], b[0][2], a[0][3], b[0][3],
                   prmt(x0.z, x0.w, 0x5410u), prmt(x0.z, x0.w, 0x7632u));
          mma_bf16(acc[mt][nt], a2[0][0], b2[0][0], a2[0][1], b2[0][1],
                   prmt(x1.x, x1.y, 0x5410u), prmt(x1.x, x1.y, 0x7632u));
          mma_bf16(acc[mt][nt], a2[0][2], b2[0][2], a2[0][3], b2[0][3],
                   prmt(x1.z, x1.w, 0x5410u), prmt(x1.z, x1.w, 0x7632u));
        }
      }
    }
  }
}

// Split-K partials of the CTA's warps in shared memory (as #332).
template <int MT, int NT>
__device__ __forceinline__ void stash_partials(float* red,
                                               const float (&acc)[MT][NT][4],
                                               int warp, int lane,
                                               int nt_active) {
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      if (nt >= nt_active) break;
#pragma unroll
      for (int c = 0; c < 4; ++c)
        red[(((warp * MT + mt) * NT + nt) * 4 + c) * 32 + lane] =
            acc[mt][nt][c];
    }
}

template <int MT, int NT, int WR, int WK>
__device__ __forceinline__ float sum_partials(const float* red, int wr, int mt,
                                              int nt, int c, int lane) {
  float v = 0.f;
#pragma unroll
  for (int wk = 0; wk < WK; ++wk)
    v += red[((((wk * WR + wr) * MT + mt) * NT + nt) * 4 + c) * 32 + lane];
  return v;
}

template <int MT, int NT>
__device__ __forceinline__ float pick(const float (&acc)[MT][NT][4], int mt,
                                      int nt, int c) {
  float v = 0.f;
#pragma unroll
  for (int m2 = 0; m2 < MT; ++m2)
#pragma unroll
    for (int n2 = 0; n2 < NT; ++n2)
#pragma unroll
      for (int c2 = 0; c2 < 4; ++c2)
        if (m2 == mt && n2 == nt && c2 == c) v = acc[m2][n2][c2];
  return v;
}

// Expert of grid row y (the y-th distinct routed expert, -1 if none) and the
// (token | slot << 16) list of its assignments.
template <int NTHREADS>
__device__ int route(const Params& p, int y, int* s_tok, int* s_cnt,
                     int* s_present, int* s_wcnt, int* s_e) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int TK = p.T * p.K;
  for (int i = tid; i < kMaxExperts; i += NTHREADS) s_present[i] = 0;
  if (tid == 0) {
    *s_cnt = 0;
    *s_e = -1;
  }
  __syncthreads();
  for (int i = tid; i < TK; i += NTHREADS) s_present[p.ids[i]] = 1;
  __syncthreads();
  // Exclusive rank of each present expert; NTHREADS >= kMaxExperts.
  const bool f = tid < p.E && s_present[tid] != 0;
  const unsigned ballot = __ballot_sync(0xffffffffu, f);
  if (lane == 0) s_wcnt[warp] = __popc(ballot);
  __syncthreads();
  int before = __popc(ballot & ((1u << lane) - 1u));
  for (int w = 0; w < warp; ++w) before += s_wcnt[w];
  if (f && before == y) *s_e = tid;
  __syncthreads();
  const int e = *s_e;
  if (e < 0) return -1;
  for (int i = tid; i < TK; i += NTHREADS) {
    if (p.ids[i] == e) {
      const int pos = atomicAdd(s_cnt, 1);
      s_tok[pos] = (i / p.K) | ((i % p.K) << 16);
    }
  }
  __syncthreads();
  return e;
}

__device__ __forceinline__ float swiglu_oai(float gate, float up, float alpha,
                                            float limit) {
  const float g = fminf(gate, limit);
  const float u = fminf(fmaxf(up, -limit), limit);
  return (u + 1.0f) * (g / (1.0f + __expf(-alpha * g)));
}

template <int MT, int NT, int WR, int WK, int U>
__global__ void __launch_bounds__(WR* WK * 32) gate_up_kernel(const Params p) {
  constexpr int NTHREADS = WR * WK * 32;
  static_assert(NTHREADS >= kMaxExperts, "routing scan needs 256 threads");
  constexpr int ROWS = WR * MT * 8;
  extern __shared__ float red[];
  __shared__ int s_tok[kMaxAssign];
  __shared__ int s_present[kMaxExperts];
  __shared__ int s_cnt, s_e;
  __shared__ int s_wcnt[NTHREADS / 32];

  const int y = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wr = warp % WR;
  const int wk = warp / WR;
  const int g = lane >> 2;
  const int t4 = lane & 3;
#if __CUDA_ARCH__ >= 900
  // Let launch 2 get scheduled while this grid drains (it waits for our
  // completion with griddepcontrol.wait before touching anything).
  asm volatile("griddepcontrol.launch_dependents;");
#endif

  {  // zero the bf16 output for launch 2's scatter-add
    const int64_t nblocks = int64_t(gridDim.x) * gridDim.y;
    const int64_t b = int64_t(blockIdx.y) * gridDim.x + blockIdx.x;
    const int64_t total = int64_t(p.T) * p.H;
    for (int64_t i = b * NTHREADS + tid; i < total; i += nblocks * NTHREADS)
      p.out[i] = __float2bfloat16(0.f);
  }

  const int e = route<NTHREADS>(p, y, s_tok, &s_cnt, s_present, s_wcnt, &s_e);
  const int cnt = e < 0 ? 0 : s_cnt;
  if (blockIdx.x == 0) {
    for (int i = tid; i < cnt; i += NTHREADS)
      p.tok_list[y * (p.T * p.K) + i] = s_tok[i];
    if (tid == 0) {
      p.tok_cnt[y] = cnt;
      p.y_expert[y] = e;
    }
  }
  if (cnt == 0) return;

  const int64_t HB = p.H / 2;
  const int64_t HS = p.H / 32;
  const uint8_t* gw = p.gu_w + int64_t(e) * 2 * p.I * HB;
  const uint8_t* gs = p.gu_s + int64_t(e) * 2 * p.I * HS;
  const bf16* gb = p.gu_b + int64_t(e) * 2 * p.I;

  const int row0 = blockIdx.x * ROWS;
  const uint8_t* ra[MT];
  const uint8_t* rb[MT];
  const uint8_t* sa[MT];
  const uint8_t* sb[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    const int r = row0 + (wr * MT + mt) * 8 + g;
    const bool ok = r < p.I;
    ra[mt] = ok ? gw + int64_t(2 * r) * HB : nullptr;      // gate row
    rb[mt] = ok ? gw + int64_t(2 * r + 1) * HB : nullptr;  // up row
    sa[mt] = ok ? gs + int64_t(2 * r) * HS : nullptr;
    sb[mt] = ok ? gs + int64_t(2 * r + 1) * HS : nullptr;
  }

  for (int chunk = 0; chunk < cnt; chunk += 8 * NT) {
    const int nt_active = min(NT, (cnt - chunk + 7) / 8);
    const bf16* xb[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int i = chunk + nt * 8 + g;
      xb[nt] = i < cnt ? p.x + int64_t(s_tok[i] & 0xffff) * p.ldx : nullptr;
    }
    float acc[MT][NT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[mt][nt][c] = 0.f;

    warp_tile<MT, NT, U>(acc, ra, rb, sa, sb, xb, nt_active, p.H / 64, wk, WK,
                         t4);

    if constexpr (WK > 1) {
      stash_partials<MT, NT>(red, acc, warp, lane, nt_active);
      __syncthreads();
    }
    // Lane l of an m16 tile holds gate (c = pair) and up (c = 2 + pair) of
    // intermediate row l / 4 for token 2 * (l % 4) + pair.
    const int per_wr = MT * nt_active * 2 * 32;
    const int items = WK > 1 ? WR * per_wr : per_wr;
    for (int it = WK > 1 ? tid : lane; it < items;
         it += WK > 1 ? NTHREADS : 32) {
      const int l = it & 31;
      const int pair = (it >> 5) & 1;
      int rest = it >> 6;
      const int nt = rest % nt_active;
      rest /= nt_active;
      const int mt = rest % MT;
      const int w = WK > 1 ? rest / MT : wr;
      float gate_acc, up_acc;
      if constexpr (WK > 1) {
        gate_acc = sum_partials<MT, NT, WR, WK>(red, w, mt, nt, pair, l);
        up_acc = sum_partials<MT, NT, WR, WK>(red, w, mt, nt, 2 + pair, l);
      } else {
        gate_acc = pick(acc, mt, nt, pair);
        up_acc = pick(acc, mt, nt, 2 + pair);
      }
      const int r = row0 + (w * MT + mt) * 8 + (l >> 2);
      const int i = chunk + nt * 8 + 2 * (l & 3) + pair;
      if (r < p.I && i < cnt) {
        // Reference rounding: bf16(x @ W^T + b), then SwiGLU, then bf16.
        const float gt = __bfloat162float(
            __float2bfloat16(gate_acc + __bfloat162float(gb[2 * r])));
        const float ut = __bfloat162float(
            __float2bfloat16(up_acc + __bfloat162float(gb[2 * r + 1])));
        const int tok = s_tok[i];
        const int64_t slot = int64_t(tok & 0xffff) * p.K + (tok >> 16);
        p.act[slot * p.I + r] =
            __float2bfloat16(swiglu_oai(gt, ut, p.alpha, p.limit));
      }
    }
    if constexpr (WK > 1) __syncthreads();
  }
}

template <int MT, int NT, int WR, int WK, int U>
__global__ void __launch_bounds__(WR* WK * 32) down_kernel(const Params p) {
  constexpr int NTHREADS = WR * WK * 32;
  constexpr int ROWS = WR * MT * 16;
  extern __shared__ float red[];
  __shared__ int s_tok[kMaxAssign];
  __shared__ float s_w[kMaxAssign];

  const int y = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wr = warp % WR;
  const int wk = warp / WR;
  const int g = lane >> 2;
  const int t4 = lane & 3;

#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  const int cnt = p.tok_cnt[y];
  if (cnt == 0) return;
  const int e = p.y_expert[y];
  for (int i = tid; i < cnt; i += NTHREADS) {
    const int tok = p.tok_list[y * (p.T * p.K) + i];
    s_tok[i] = tok;
    s_w[i] = __bfloat162float(p.rw[(tok & 0xffff) * p.K + (tok >> 16)]);
  }
  __syncthreads();

  const int64_t IB = p.I / 2;
  const int64_t IS = p.I / 32;
  const uint8_t* dw = p.d_w + int64_t(e) * p.H * IB;
  const uint8_t* ds = p.d_s + int64_t(e) * p.H * IS;
  const bf16* db = p.d_b + int64_t(e) * p.H;

  const int row0 = blockIdx.x * ROWS;
  const uint8_t* ra[MT];
  const uint8_t* rb[MT];
  const uint8_t* sa[MT];
  const uint8_t* sb[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    const int h = row0 + (wr * MT + mt) * 16 + g;
    ra[mt] = h < p.H ? dw + int64_t(h) * IB : nullptr;
    rb[mt] = h + 8 < p.H ? dw + int64_t(h + 8) * IB : nullptr;
    sa[mt] = h < p.H ? ds + int64_t(h) * IS : nullptr;
    sb[mt] = h + 8 < p.H ? ds + int64_t(h + 8) * IS : nullptr;
  }

  for (int chunk = 0; chunk < cnt; chunk += 8 * NT) {
    const int nt_active = min(NT, (cnt - chunk + 7) / 8);
    const bf16* xb[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int i = chunk + nt * 8 + g;
      if (i < cnt) {
        const int tok = s_tok[i];
        const int64_t slot = int64_t(tok & 0xffff) * p.K + (tok >> 16);
        xb[nt] = p.act + slot * p.I;
      } else {
        xb[nt] = nullptr;
      }
    }
    float acc[MT][NT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[mt][nt][c] = 0.f;

    warp_tile<MT, NT, U>(acc, ra, rb, sa, sb, xb, nt_active, p.I / 64, wk, WK,
                         t4);

    if constexpr (WK > 1) {
      stash_partials<MT, NT>(red, acc, warp, lane, nt_active);
      __syncthreads();
    }
    const int per_wr = MT * nt_active * 4 * 32;
    const int items = WK > 1 ? WR * per_wr : per_wr;
    for (int it = WK > 1 ? tid : lane; it < items;
         it += WK > 1 ? NTHREADS : 32) {
      const int l = it & 31;
      const int c = (it >> 5) & 3;
      int rest = it >> 7;
      const int nt = rest % nt_active;
      rest /= nt_active;
      const int mt = rest % MT;
      const int w = WK > 1 ? rest / MT : wr;
      float v;
      if constexpr (WK > 1) {
        v = sum_partials<MT, NT, WR, WK>(red, w, mt, nt, c, l);
      } else {
        v = pick(acc, mt, nt, c);
      }
      const int h = row0 + (w * MT + mt) * 16 + (l >> 2) + (c >= 2 ? 8 : 0);
      const int i = chunk + nt * 8 + 2 * (l & 3) + (c & 1);
      if (h < p.H && i < cnt) {
        // Reference: final_hidden[mask] += bf16(w) * bf16(x @ Wd^T + b).
        const float o =
            __bfloat162float(__float2bfloat16(v + __bfloat162float(db[h])));
        atomicAdd(p.out + int64_t(s_tok[i] & 0xffff) * p.H + h,
                  __float2bfloat16(o * s_w[i]));
      }
    }
    if constexpr (WK > 1) __syncthreads();
  }
}

struct LaunchCfg {
  int mt;
  int wr;
  int wk;
  int u;
};

template <int MT, int NT, int WR, int WK, int U, bool DOWN>
void launch_one(const Params& p, int Y, cudaStream_t stream) {
  constexpr int kSmem =
      WK > 1 ? WR * WK * MT * NT * 4 * 32 * int(sizeof(float)) : 0;
  auto kernel =
      DOWN ? down_kernel<MT, NT, WR, WK, U> : gate_up_kernel<MT, NT, WR, WK, U>;
  if constexpr (kSmem > 48 * 1024) {
    static std::once_flag once;
    std::call_once(once, [&] {
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           kSmem);
    });
  }
  const int rows = DOWN ? WR * MT * 16 : WR * MT * 8;
  const int dim = DOWN ? p.H : p.I;
  dim3 grid((dim + rows - 1) / rows, Y);
  if (!DOWN) {
    kernel<<<grid, WR * WK * 32, kSmem, stream>>>(p);
    return;
  }
  // Launch 2 as a programmatic dependent of launch 1 (sm_90+): its CTAs are
  // scheduled during launch 1's tail and wait in griddepcontrol.wait.
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(WR * WK * 32);
  cfg.dynamicSmemBytes = kSmem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = p.pdl;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, kernel, p);
}

// Instantiated (MT, WR, WK, U); 8 warps each (the routing scan needs 256
// threads).
#define GMX_CONFIGS(X) \
  X(1, 1, 8, 2)        \
  X(1, 1, 8, 4)        \
  X(1, 2, 4, 2)        \
  X(1, 2, 4, 4)        \
  X(1, 4, 2, 2)        \
  X(1, 4, 2, 4)        \
  X(1, 8, 1, 2)        \
  X(1, 8, 1, 4)        \
  X(1, 2, 4, 8)        \
  X(1, 4, 2, 8)        \
  X(1, 8, 1, 8)        \
  X(2, 4, 2, 2)        \
  X(2, 8, 1, 2)

template <int NT, bool DOWN>
void launch_nt(const LaunchCfg& c, const Params& p, int Y,
               cudaStream_t stream) {
#define GMX_CASE(MT_, WR_, WK_, U_)                           \
  if (c.mt == MT_ && c.wr == WR_ && c.wk == WK_ && c.u == U_) \
    return launch_one<MT_, NT, WR_, WK_, U_, DOWN>(p, Y, stream);
  GMX_CONFIGS(GMX_CASE)
#undef GMX_CASE
  TORCH_CHECK(false, "gptoss_grouped_mxfp4: unsupported config mt=", c.mt,
              " wr=", c.wr, " wk=", c.wk, " u=", c.u);
}

template <bool DOWN>
void launch(int nt, const LaunchCfg& c, const Params& p, int Y,
            cudaStream_t stream) {
  switch (nt) {
    case 1:
      return launch_nt<1, DOWN>(c, p, Y, stream);
    case 2:
      return launch_nt<2, DOWN>(c, p, Y, stream);
    case 4:
      return launch_nt<4, DOWN>(c, p, Y, stream);
    default:
      return launch_nt<8, DOWN>(c, p, Y, stream);
  }
}

// Tuning override: MOE_GPTOSS_MXFP4_CFG="nt_up,nt_down;mt,wr,wk,u;mt,wr,wk,u".
bool env_cfg(int& nt_up, int& nt_down, LaunchCfg& up, LaunchCfg& down) {
  const char* env = std::getenv("MOE_GPTOSS_MXFP4_CFG");
  if (env == nullptr || *env == '\0') return false;
  int v[10];
  if (std::sscanf(env, "%d,%d;%d,%d,%d,%d;%d,%d,%d,%d", &v[0], &v[1], &v[2],
                  &v[3], &v[4], &v[5], &v[6], &v[7], &v[8], &v[9]) != 10)
    return false;
  nt_up = v[0];
  nt_down = v[1];
  up = LaunchCfg{v[2], v[3], v[4], v[5]};
  down = LaunchCfg{v[6], v[7], v[8], v[9]};
  return true;
}

}  // namespace

bool gptoss_grouped_mxfp4_supported(int64_t tokens, int64_t top_k,
                                    int64_t experts, int64_t hidden,
                                    int64_t inter) {
  return tokens > 0 && tokens * top_k <= kMaxAssign && tokens < (1 << 16) &&
         experts <= kMaxExperts && hidden % 64 == 0 && inter % 64 == 0;
}

torch::Tensor gptoss_grouped_mxfp4(
    const torch::Tensor& x, const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights, const torch::Tensor& gu_blocks,
    const torch::Tensor& gu_scales, const torch::Tensor& gu_bias,
    const torch::Tensor& d_blocks, const torch::Tensor& d_scales,
    const torch::Tensor& d_bias, double alpha, double limit) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == at::kBFloat16 &&
                  x.stride(1) == 1 && x.stride(0) % 8 == 0,
              "gptoss_grouped_mxfp4: x must be a CUDA bf16 [T, H] matrix");
  const int64_t T = x.size(0), H = x.size(1);
  const int64_t E = gu_blocks.size(0);
  const int64_t I = gu_blocks.size(1) / 2;
  TORCH_CHECK(topk_ids.scalar_type() == at::kLong && topk_ids.dim() == 2 &&
                  topk_ids.size(0) == T && topk_ids.is_contiguous(),
              "gptoss_grouped_mxfp4: topk_ids must be int64 [T, k]");
  const int64_t K = topk_ids.size(1);
  TORCH_CHECK(topk_weights.scalar_type() == at::kBFloat16 &&
                  topk_weights.sizes() == topk_ids.sizes() &&
                  topk_weights.is_contiguous(),
              "gptoss_grouped_mxfp4: topk_weights must be bf16 [T, k]");
  TORCH_CHECK(gptoss_grouped_mxfp4_supported(T, K, E, H, I),
              "gptoss_grouped_mxfp4: unsupported shape T=", T, " k=", K,
              " E=", E, " H=", H, " I=", I);
  auto check_w = [&](const torch::Tensor& t, int64_t rows, int64_t cols,
                     const char* name) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kByte &&
                    t.is_contiguous() && t.numel() == E * rows * cols,
                "gptoss_grouped_mxfp4: ", name, " must be contiguous uint8 [",
                E, ", ", rows, ", ", cols, "]");
  };
  check_w(gu_blocks, 2 * I, H / 2, "gate_up blocks");
  check_w(gu_scales, 2 * I, H / 32, "gate_up scales");
  check_w(d_blocks, H, I / 2, "down blocks");
  check_w(d_scales, H, I / 32, "down scales");
  TORCH_CHECK(gu_bias.scalar_type() == at::kBFloat16 &&
                  gu_bias.is_contiguous() && gu_bias.numel() == E * 2 * I &&
                  d_bias.scalar_type() == at::kBFloat16 &&
                  d_bias.is_contiguous() && d_bias.numel() == E * H,
              "gptoss_grouped_mxfp4: biases must be contiguous bf16");

  c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
  auto out = torch::empty({T, H}, x.options());
  const int64_t TK = T * K;
  const int64_t Y = std::min(E, TK);
  // Workspace: act [T*k, I] bf16 | tok_list [Y, T*k] | tok_cnt [Y] | y_expert.
  const int64_t act_bytes = (TK * I * 2 + 255) / 256 * 256;
  const int64_t list_bytes = (Y * TK * 4 + 255) / 256 * 256;
  auto ws = torch::empty({act_bytes + list_bytes + 2 * Y * 4},
                         x.options().dtype(at::kByte));
  uint8_t* base = ws.data_ptr<uint8_t>();

  Params p{};
  p.x = reinterpret_cast<const bf16*>(x.data_ptr());
  p.ldx = x.stride(0);
  p.ids = topk_ids.data_ptr<int64_t>();
  p.rw = reinterpret_cast<const bf16*>(topk_weights.data_ptr());
  p.gu_w = gu_blocks.data_ptr<uint8_t>();
  p.gu_s = gu_scales.data_ptr<uint8_t>();
  p.gu_b = reinterpret_cast<const bf16*>(gu_bias.data_ptr());
  p.d_w = d_blocks.data_ptr<uint8_t>();
  p.d_s = d_scales.data_ptr<uint8_t>();
  p.d_b = reinterpret_cast<const bf16*>(d_bias.data_ptr());
  p.E = static_cast<int>(E);
  p.T = static_cast<int>(T);
  p.K = static_cast<int>(K);
  p.H = static_cast<int>(H);
  p.I = static_cast<int>(I);
  p.alpha = static_cast<float>(alpha);
  p.limit = static_cast<float>(limit);
  p.act = reinterpret_cast<bf16*>(base);
  p.tok_list = reinterpret_cast<int*>(base + act_bytes);
  p.tok_cnt = reinterpret_cast<int*>(base + act_bytes + list_bytes);
  p.y_expert = p.tok_cnt + Y;
  p.out = reinterpret_cast<bf16*>(out.data_ptr());
  p.pdl = at::cuda::getCurrentDeviceProperties()->major >= 9 ? 1 : 0;

  // Launch configs measured on sm_120 (RTX PRO 6000 Max-Q, 300 W) for the
  // GPT-OSS shapes (H = I = 2880): up to ~4 tokens per expert, one n8 token
  // tile per pass (extra tokens take a second, L2-resident pass) and K-split
  // warps; ~8 tokens per expert, two n8 tiles and row-split warps.
  const int64_t m_avg = (TK + Y - 1) / Y;
  int nt = 1, nt_down = 1;
  LaunchCfg up{1, 2, 4, 4};
  LaunchCfg down{1, 1, 8, 2};
  if (m_avg > 4) {
    nt = nt_down = m_avg <= 16 ? 2 : m_avg <= 32 ? 4 : 8;
    up = LaunchCfg{1, 8, 1, 8};
    down = LaunchCfg{1, 2, 4, 2};
  }
  env_cfg(nt, nt_down, up, down);
  launch<false>(nt, up, p, static_cast<int>(Y), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  launch<true>(nt_down, down, p, static_cast<int>(Y), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
