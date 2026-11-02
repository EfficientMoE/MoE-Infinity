// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// Grouped decode MoE: all resident experts of a layer in two launches.
//
// Decode feeds 1-64 tokens to each expert, so every projection is a skinny
// GEMM that must stream its weights at DRAM speed. The kernels use
// mma.sync.m16n8k16 (BF16, sm_80+) "swap-AB": 16 weight rows are the MMA A
// operand and up to 8 tokens are the B operand, so no MMA row is padding.
//
// Fragments are loaded straight from global memory. A lane (g = lane / 4,
// t = lane % 4) loads 16 contiguous bytes of its weight rows per 32-wide K
// step, and the same 8 K positions of its token: the K order inside the MMA
// is permuted identically for A and B, which leaves the dot product unchanged
// while every 32-byte sector is fully used by one warp instruction.
//
// Launch 1 (gate/up): an m16 tile holds 8 gate rows (A rows 0-7) and the same
// 8 up rows (A rows 8-15), so each lane owns gate and up of one intermediate
// row and computes silu(bf16(gate)) * up in registers. Warps of a CTA split K
// and reduce through shared memory. The activations land in a [T * kcap, I]
// workspace at row t * kcap + rank(t, e).
// Launch 2 (down): out[t] += float(bf16(x_e(t) @ down^T)) * w[t, e] with fp32
// atomics, i.e. the routing-weight multiply and the scatter-add are fused into
// the epilogue.
//
// Routing is resolved on the device: each launch-1 CTA scans
// router_mask[:, e] (T <= 256) and the first CTA of every expert publishes the
// token list for launch 2. No host sync, no sort / align launch.

#include "kernel/grouped_moe_decode.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

namespace {

using bf16 = __nv_bfloat16;

struct ExpertEntry {
  const bf16* gate;
  const bf16* up;
  const bf16* down;
  int64_t expert_id;
};
static_assert(sizeof(ExpertEntry) == 32, "expert table row is int64[4]");

struct Params {
  const bf16* x;
  int64_t ldx;
  const uint8_t* mask;
  int64_t mask_ld;
  int64_t mask_cs;
  const void* rw;
  int64_t rw_ld;
  int64_t rw_cs;
  int rw_dtype;  // 0 = fp32, 1 = bf16, 2 = fp16
  const ExpertEntry* experts;
  int n;
  int T;
  int H;
  int I;
  int kcap;
  bf16* act;      // [T * kcap, I]
  int* tok_list;  // [n, T]: token | rank << 16
  int* tok_cnt;   // [n]
  float* out;
  int64_t ldo;
  int zero_out;
};

__device__ __forceinline__ uint4 ld_stream(const bf16* p) {
  uint4 r;
  asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
      : "l"(p));
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

__device__ __forceinline__ float router_weight(const Params& p, int t, int e) {
  const int64_t off = t * p.rw_ld + e * p.rw_cs;
  if (p.rw_dtype == 0) return reinterpret_cast<const float*>(p.rw)[off];
  if (p.rw_dtype == 1)
    return __bfloat162float(reinterpret_cast<const bf16*>(p.rw)[off]);
  return __half2float(reinterpret_cast<const __half*>(p.rw)[off]);
}

// One warp: acc[mt][nt] += A_mt (16 x K) * B_nt (K x 8) over the K steps
// step0, step0 + stride, ... A rows g / g+8 of tile mt are ra[mt] / rb[mt]
// (nullptr = zero row); token g of tile nt is xb[nt] (nullptr = padding).
template <int MT, int NT, int U>
__device__ __forceinline__ void warp_tile(float (&acc)[MT][NT][4],
                                          const bf16* const (&ra)[MT],
                                          const bf16* const (&rb)[MT],
                                          const bf16* const (&xb)[NT],
                                          int nt_active, int nsteps, int step0,
                                          int stride, int t4) {
  const uint4 zero = make_uint4(0, 0, 0, 0);
  for (int s = step0; s < nsteps; s += stride * U) {
    uint4 wa[U][MT];
    uint4 wb[U][MT];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int ss = s + u * stride;
      const int k = ss * 32 + t4 * 8;
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        wa[u][mt] = (ss < nsteps && ra[mt]) ? ld_stream(ra[mt] + k) : zero;
        wb[u][mt] = (ss < nsteps && rb[mt]) ? ld_stream(rb[mt] + k) : zero;
      }
    }
    // Token fragments of all U steps are issued before the first MMA so their
    // L1/L2 latency overlaps with the weight stream.
    uint4 xv[U][NT];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int ss = s + u * stride;
      const int k = ss * 32 + t4 * 8;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
        xv[u][nt] = (ss < nsteps && nt < nt_active && xb[nt])
                        ? ld_act(xb[nt] + k)
                        : zero;
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int ss = s + u * stride;
      if (ss >= nsteps) break;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        if (nt >= nt_active) break;
        const uint4 b = xv[u][nt];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          mma_bf16(acc[mt][nt], wa[u][mt].x, wb[u][mt].x, wa[u][mt].y,
                   wb[u][mt].y, b.x, b.y);
          mma_bf16(acc[mt][nt], wa[u][mt].z, wb[u][mt].z, wa[u][mt].w,
                   wb[u][mt].w, b.z, b.w);
        }
      }
    }
  }
}

// Split-K partials of the CTA's warps in shared memory.
// Layout: red[warp][mt][nt][c][lane], warp = wk * WR + wr.
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

// Register value acc[mt][nt][c] for runtime (mt, nt, c) without local memory.
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

// Ordered list of the tokens routed to expert e, with each token's rank among
// its routed experts (activation workspace row = t * kcap + rank).
template <int NTHREADS>
__device__ void build_token_list(const Params& p, int e, int* s_tok, int* s_cnt,
                                 int* s_wcnt) {
  constexpr int NW = NTHREADS / 32;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (tid == 0) *s_cnt = 0;
  __syncthreads();
  for (int base = 0; base < p.T; base += NTHREADS) {
    const int t = base + tid;
    const bool f = t < p.T && p.mask[t * p.mask_ld + e * p.mask_cs] != 0;
    const unsigned ballot = __ballot_sync(0xffffffffu, f);
    if (lane == 0) s_wcnt[warp] = __popc(ballot);
    __syncthreads();
    int before = *s_cnt;
    int total = 0;
#pragma unroll
    for (int w = 0; w < NW; ++w) {
      if (w < warp) before += s_wcnt[w];
      total += s_wcnt[w];
    }
    if (f) s_tok[before + __popc(ballot & ((1u << lane) - 1u))] = t;
    __syncthreads();
    if (tid == 0) *s_cnt += total;
    __syncthreads();
  }
  const int cnt = *s_cnt;
  // rank(t, e) = number of experts e' < e routed for token t; one warp/token.
  for (int i = warp; i < cnt; i += NW) {
    const int t = s_tok[i];
    int r = 0;
    for (int c = lane; c < e; c += 32)
      r += p.mask[t * p.mask_ld + c * p.mask_cs] != 0;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) r += __shfl_xor_sync(0xffffffffu, r, o);
    if (lane == 0) s_tok[i] = t | (r << 16);
  }
  __syncthreads();
}

// CTA = WR x WK warps: WR row groups (MT m16 tiles each) x WK K-splits.
// Row-split warps walk K together, so the token fragments they share mostly
// hit L1; K-split warps keep enough CTAs in flight when there are few rows.
template <int MT, int NT, int WR, int WK, int U>
__global__ void __launch_bounds__(WR* WK * 32) gate_up_kernel(const Params p) {
  constexpr int NTHREADS = WR * WK * 32;
  constexpr int ROWS = WR * MT * 8;
  extern __shared__ float red[];
  __shared__ int s_tok[kGroupedMoeMaxTokens];
  __shared__ int s_cnt;
  __shared__ int s_wcnt[WR * WK];

  const int j = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wr = warp % WR;
  const int wk = warp / WR;
  const int g = lane >> 2;
  const int t4 = lane & 3;

  if (p.zero_out) {
    const int64_t nblocks = int64_t(gridDim.x) * gridDim.y;
    const int64_t b = int64_t(blockIdx.y) * gridDim.x + blockIdx.x;
    const int64_t total = int64_t(p.T) * p.H;
    for (int64_t i = b * NTHREADS + tid; i < total; i += nblocks * NTHREADS)
      p.out[(i / p.H) * p.ldo + (i % p.H)] = 0.f;
  }

  const ExpertEntry ex = p.experts[j];
  const int e = static_cast<int>(ex.expert_id);
  build_token_list<NTHREADS>(p, e, s_tok, &s_cnt, s_wcnt);
  const int cnt = s_cnt;
  if (blockIdx.x == 0) {
    for (int i = tid; i < cnt; i += NTHREADS)
      p.tok_list[j * p.T + i] = s_tok[i];
    if (tid == 0) p.tok_cnt[j] = cnt;
  }
  if (cnt == 0) return;

  const int row0 = blockIdx.x * ROWS;
  const bf16* ra[MT];
  const bf16* rb[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    const int r = row0 + (wr * MT + mt) * 8 + g;
    ra[mt] = r < p.I ? ex.gate + int64_t(r) * p.H : nullptr;
    rb[mt] = r < p.I ? ex.up + int64_t(r) * p.H : nullptr;
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

    warp_tile<MT, NT, U>(acc, ra, rb, xb, nt_active, p.H / 32, wk, WK, t4);

    if constexpr (WK > 1) {
      stash_partials<MT, NT>(red, acc, warp, lane, nt_active);
      __syncthreads();
    }
    // Epilogue items (wr, mt, nt, pair, lane); pair picks token 2 * t4 + pair.
    // Lane l of an m16 tile holds gate (c = pair) and up (c = 2 + pair) of
    // intermediate row l / 4.
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
        // Same rounding points as GEMM0 (bf16 gate_buf) + SiLUAndMulEpilogue.
        const float x = __bfloat162float(__float2bfloat16(gate_acc));
        const float silu_x = x * (1.0f / (1.0f + expf(-x)));
        const int tok = s_tok[i];
        const int64_t slot = int64_t(tok & 0xffff) * p.kcap + (tok >> 16);
        p.act[slot * p.I + r] = __float2bfloat16(silu_x * up_acc);
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
  __shared__ int s_tok[kGroupedMoeMaxTokens];
  __shared__ float s_w[kGroupedMoeMaxTokens];

  const int j = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wr = warp % WR;
  const int wk = warp / WR;
  const int g = lane >> 2;
  const int t4 = lane & 3;

  const ExpertEntry ex = p.experts[j];
  const int e = static_cast<int>(ex.expert_id);
  const int cnt = p.tok_cnt[j];
  if (cnt == 0) return;
  for (int i = tid; i < cnt; i += NTHREADS) {
    const int tok = p.tok_list[j * p.T + i];
    s_tok[i] = tok;
    s_w[i] = router_weight(p, tok & 0xffff, e);
  }
  __syncthreads();

  const int row0 = blockIdx.x * ROWS;
  const bf16* ra[MT];
  const bf16* rb[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    const int h = row0 + (wr * MT + mt) * 16 + g;
    ra[mt] = h < p.H ? ex.down + int64_t(h) * p.I : nullptr;
    rb[mt] = h + 8 < p.H ? ex.down + int64_t(h + 8) * p.I : nullptr;
  }

  for (int chunk = 0; chunk < cnt; chunk += 8 * NT) {
    const int nt_active = min(NT, (cnt - chunk + 7) / 8);
    const bf16* xb[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int i = chunk + nt * 8 + g;
      if (i < cnt) {
        const int tok = s_tok[i];
        const int64_t slot = int64_t(tok & 0xffff) * p.kcap + (tok >> 16);
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

    warp_tile<MT, NT, U>(acc, ra, rb, xb, nt_active, p.I / 32, wk, WK, t4);

    if constexpr (WK > 1) {
      stash_partials<MT, NT>(red, acc, warp, lane, nt_active);
      __syncthreads();
    }
    // Epilogue items (wr, mt, nt, c, lane): row l / 4 + 8 * (c >= 2),
    // token 2 * (l % 4) + c % 2.
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
        // Same as GEMM2 (bf16 output) + OutputFunc (fp32 weighted index_add).
        const float o = __bfloat162float(__float2bfloat16(v));
        atomicAdd(p.out + int64_t(s_tok[i] & 0xffff) * p.ldo + h, o * s_w[i]);
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
void launch_one(const Params& p, cudaStream_t stream) {
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
  dim3 grid((dim + rows - 1) / rows, p.n);
  kernel<<<grid, WR * WK * 32, kSmem, stream>>>(p);
}

// Instantiated (MT, WR, WK, U) configurations.
#define GMOE_CONFIGS(X) \
  X(1, 1, 8, 2)         \
  X(1, 1, 8, 4)         \
  X(1, 1, 8, 8)         \
  X(1, 2, 4, 2)         \
  X(1, 2, 4, 4)         \
  X(1, 2, 4, 8)         \
  X(1, 8, 1, 2)         \
  X(2, 8, 1, 2)

template <int NT, bool DOWN>
void launch_nt(const LaunchCfg& c, const Params& p, cudaStream_t stream) {
#define GMOE_CASE(MT_, WR_, WK_, U_)                          \
  if (c.mt == MT_ && c.wr == WR_ && c.wk == WK_ && c.u == U_) \
    return launch_one<MT_, NT, WR_, WK_, U_, DOWN>(p, stream);
  GMOE_CONFIGS(GMOE_CASE)
#undef GMOE_CASE
  TORCH_CHECK(false, "grouped_moe_decode: unsupported launch config mt=", c.mt,
              " wr=", c.wr, " wk=", c.wk, " u=", c.u);
}

template <bool DOWN>
void launch(int nt, const LaunchCfg& c, const Params& p, cudaStream_t stream) {
  switch (nt) {
    case 1:
      return launch_nt<1, DOWN>(c, p, stream);
    case 2:
      return launch_nt<2, DOWN>(c, p, stream);
    case 4:
      return launch_nt<4, DOWN>(c, p, stream);
    default:
      return launch_nt<8, DOWN>(c, p, stream);
  }
}

// Optional override for tuning: MOE_GROUPED_MOE_CFG="mt,wr,wk,u;mt,wr,wk,u".
bool env_cfg(LaunchCfg& up, LaunchCfg& down) {
  static const char* env = std::getenv("MOE_GROUPED_MOE_CFG");
  if (env == nullptr || *env == '\0') return false;
  return std::sscanf(env, "%d,%d,%d,%d;%d,%d,%d,%d", &up.mt, &up.wr, &up.wk,
                     &up.u, &down.mt, &down.wr, &down.wk, &down.u) == 8;
}

}  // namespace

bool grouped_moe_decode_supported(int64_t tokens, int64_t hidden,
                                  int64_t inter) {
  return tokens > 0 && tokens <= kGroupedMoeMaxTokens && hidden % 32 == 0 &&
         inter % 32 == 0 && hidden > 0 && inter > 0;
}

void grouped_moe_decode(const torch::Tensor& hidden,
                        const torch::Tensor& router_mask,
                        const torch::Tensor& router_weight,
                        const torch::Tensor& expert_table, int64_t inter,
                        int64_t kcap, torch::Tensor& out, bool zero_out,
                        cudaStream_t stream) {
  TORCH_CHECK(hidden.is_cuda() && hidden.dim() == 2 &&
                  hidden.scalar_type() == at::kBFloat16 &&
                  hidden.stride(1) == 1,
              "grouped_moe_decode: hidden must be a CUDA bf16 [T, H] matrix");
  const int64_t T = hidden.size(0);
  const int64_t H = hidden.size(1);
  TORCH_CHECK(grouped_moe_decode_supported(T, H, inter),
              "grouped_moe_decode: unsupported shape T=", T, " H=", H,
              " I=", inter);
  TORCH_CHECK(hidden.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(hidden.data_ptr()) % 16 == 0,
              "grouped_moe_decode: hidden rows must be 16-byte aligned");
  TORCH_CHECK(router_mask.dim() == 2 && router_mask.size(0) == T &&
                  (router_mask.scalar_type() == at::kBool ||
                   router_mask.scalar_type() == at::kByte),
              "grouped_moe_decode: router_mask must be bool/uint8 [T, E]");
  TORCH_CHECK(router_weight.dim() == 2 && router_weight.size(0) == T &&
                  router_weight.size(1) == router_mask.size(1),
              "grouped_moe_decode: router_weight must be [T, E]");
  int rw_dtype;
  switch (router_weight.scalar_type()) {
    case at::kFloat:
      rw_dtype = 0;
      break;
    case at::kBFloat16:
      rw_dtype = 1;
      break;
    case at::kHalf:
      rw_dtype = 2;
      break;
    default:
      TORCH_CHECK(false, "grouped_moe_decode: router_weight dtype ",
                  router_weight.scalar_type(), " not supported");
  }
  TORCH_CHECK(expert_table.is_cuda() && expert_table.dim() == 2 &&
                  expert_table.size(1) == 4 &&
                  expert_table.scalar_type() == at::kLong &&
                  expert_table.is_contiguous(),
              "grouped_moe_decode: expert_table must be int64 [n, 4] on CUDA");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kFloat &&
                  out.dim() == 2 && out.size(0) == T && out.size(1) == H &&
                  out.stride(1) == 1,
              "grouped_moe_decode: out must be float32 [T, H]");
  TORCH_CHECK(kcap > 0 && kcap * T < (1 << 30), "grouped_moe_decode: kcap");
  const int64_t n = expert_table.size(0);
  if (n == 0) {
    if (zero_out) out.zero_();
    return;
  }

  c10::cuda::CUDAGuard guard(hidden.device());
  // One workspace: act [T * kcap, I] bf16 | tok_list [n, T] | tok_cnt [n].
  const int64_t act_bytes = (T * kcap * inter * 2 + 255) / 256 * 256;
  const int64_t list_bytes = (n * T * 4 + 255) / 256 * 256;
  auto ws = torch::empty({act_bytes + list_bytes + n * 4},
                         hidden.options().dtype(at::kByte));
  uint8_t* base = ws.data_ptr<uint8_t>();

  Params p{};
  p.x = reinterpret_cast<const bf16*>(hidden.data_ptr());
  p.ldx = hidden.stride(0);
  p.mask = reinterpret_cast<const uint8_t*>(router_mask.data_ptr());
  p.mask_ld = router_mask.stride(0);
  p.mask_cs = router_mask.stride(1);
  p.rw = router_weight.data_ptr();
  p.rw_ld = router_weight.stride(0);
  p.rw_cs = router_weight.stride(1);
  p.rw_dtype = rw_dtype;
  p.experts = reinterpret_cast<const ExpertEntry*>(expert_table.data_ptr());
  p.n = static_cast<int>(n);
  p.T = static_cast<int>(T);
  p.H = static_cast<int>(H);
  p.I = static_cast<int>(inter);
  p.kcap = static_cast<int>(kcap);
  p.act = reinterpret_cast<bf16*>(base);
  p.tok_list = reinterpret_cast<int*>(base + act_bytes);
  p.tok_cnt = reinterpret_cast<int*>(base + act_bytes + list_bytes);
  p.out = out.data_ptr<float>();
  p.ldo = out.stride(0);
  p.zero_out = zero_out ? 1 : 0;

  // Token tile: enough n8 tiles for the largest possible per-expert batch
  // (all T tokens), capped at 64 tokens per pass.
  const int nt = T <= 8 ? 1 : T <= 16 ? 2 : T <= 32 ? 4 : 8;
  // Few tokens per expert: 8 K-split warps per 8-16 rows keep enough CTAs and
  // bytes in flight. Many tokens per expert: 8 row-split warps share the token
  // fragments through L1, and two m16 tiles per warp halve their L2 traffic.
  // Long K rows (wide experts) want more bytes in flight per thread (U = 8);
  // short ones want more CTAs. Measured on sm_120 (RTX PRO 6000 Max-Q) over
  // the audit's decode shapes.
  const int64_t m_avg = (T * kcap + n - 1) / n;
  const int u_max = nt <= 2 ? 8 : 4;
  LaunchCfg up{1, 1, 8, H >= 4096 ? u_max : 2};
  LaunchCfg down = inter >= 3072   ? LaunchCfg{1, 1, 8, u_max}
                   : inter >= 1536 ? LaunchCfg{1, 2, 4, u_max}
                                   : LaunchCfg{1, 2, 4, nt <= 2 ? 4 : 2};
  if (nt == 8) {
    up = m_avg >= 32 ? LaunchCfg{2, 8, 1, 2} : LaunchCfg{1, 8, 1, 2};
    down = up;
  }
  env_cfg(up, down);
  launch<false>(nt, up, p, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  launch<true>(nt, down, p, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
