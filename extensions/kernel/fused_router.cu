// Fused MoE routing (#315): gate logits on tensor cores + scoring + (grouped)
// top-k + renorm, written straight to compact ids / weights, the dense [T, E]
// mask / weights the expert dispatcher consumes, and (optionally) the per-layer
// active-expert flags in pinned host memory -- so the dispatcher never has to
// reduce the mask or copy it back, and nothing on the path syncs the host.
//
//   T <= kFusedMaxTokens: ONE launch.
//     phase 1  split-K x expert-tile x token-tile work items (WMMA bf16/fp16,
//              fp32 accumulate) write fp32 partials;
//     hand-off the last CTA (one tile) or the last min(T, grid) CTAs to
//              finish phase 1 route (no grid barrier);
//     phase 2  each router reduces the partials of its tokens in a fixed
//              order (deterministic) and routes them.
//   larger T: cuBLAS tensor-core logits + one routing launch.
//
// Selection math mirrors the torch routers (expf / log1pf / sqrtf / IEEE
// division, lowest index wins ties), so expert selection is identical
// whenever the logits are.
//
// Hardware: RTX_PRO_6000 gate (moe_infinity::hw, kernel/hw_profile.h). Split
// counts, tile sizes, the 99 KiB shared-memory budgets and the one-wave
// thresholds were tuned on the RTX PRO 6000 Blackwell (188 SMs, 128 MiB L2).
// fused_route() returns {} on any other device and the Python side runs the
// previous torch routing (moe_infinity/kernel/router.py, _route_reference).

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CachingHostAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>

#include "kernel/hw_profile.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace moe_router {

namespace wmma = nvcuda::wmma;

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kPad = 8;  // smem row padding (bank conflicts)
constexpr int kStages = 3;
constexpr int kMaxE = 512;
constexpr int kMaxK = 32;
constexpr int kMaxGroups = 32;
#ifndef ROUTER_MAXSPLIT
  #define ROUTER_MAXSPLIT 16
#endif
constexpr int kMaxSplit = ROUTER_MAXSPLIT;
#ifndef ROUTER_BE16
  #define ROUTER_BE16 32
#endif
#ifndef ROUTER_BT64_MIN
  #define ROUTER_BT64_MIN 17
#endif
constexpr int kBT64MinTokens = ROUTER_BT64_MIN;
#ifndef ROUTER_BE64
  #define ROUTER_BE64 64
#endif
constexpr int kRows = 8;  // phase-2 rows reduced per smem pass (1 per warp)
constexpr int kFusedMaxTokens = 1024;
#ifndef ROUTER_WIDE_MIN_T
  #define ROUTER_WIDE_MIN_T 4096
#endif
// From this many tokens wide routers use the fused CUTLASS GEMM + top-k.
constexpr int kWideMinTokens = ROUTER_WIDE_MIN_T;
#ifndef ROUTER_WIDE_BIG_WAVES
  #define ROUTER_WIDE_BIG_WAVES 4
#endif
constexpr int kWideBigWaves = ROUTER_WIDE_BIG_WAVES;
constexpr int64_t kFusedMaxElems = 196608;  // T * E, see fused_route
// One CTA reduces all partials when T * E is this small (decode b1).
#ifndef ROUTER_SINGLE_ELEMS
  #define ROUTER_SINGLE_ELEMS 512
#endif
constexpr int kSingleTileMaxElems = ROUTER_SINGLE_ELEMS;

enum Scoring : int { kSoftmax = 0, kSigmoid = 1, kSqrtSoftplus = 2 };

// Persistent per-(device, stream) state: hand-off counters, the publish
// counter and a zeroed device copy of the active-expert flags. Every field
// is restored to zero by the kernel that used it.
struct alignas(16) RouteState {
  unsigned int bar_count;
  unsigned int bar_gen;  // unused
  unsigned int publish_count;
  unsigned int tile_count;
  int flags[kMaxE];
  unsigned int bar_leave;
  // Multi-router decode: launch tag (seq + 1) of the last launch that wrote
  // each expert's pinned flag. Never needs re-zeroing.
  int claimed[kMaxE];
};

struct EpilogueParams {
  int T, E, K;
  const float* bias;  // [E] or nullptr
  int n_group, topk_group;
  bool renorm;
  float scale, eps;
  bool round_logits;
  int32_t* ids;      // [T, K]
  float* tw;         // [T, K]
  bool* mask;        // [T, E] or nullptr
  void* dw;          // [T, E] or nullptr
  int dw_dtype;      // 0 fp32, 1 bf16, 2 fp16
  void* logits_out;  // [T, E] or nullptr
  int logits_dtype;
  RouteState* state;
  bool* flags_host;    // [E] pinned host memory or nullptr
  int64_t* done_word;  // pinned; receives seq when a CTA finishes, or null
  int64_t seq;
};

__device__ __forceinline__ void store_typed(void* base, int64_t i, float v,
                                            int dtype) {
  if (dtype == 0) {
    static_cast<float*>(base)[i] = v;
  } else if (dtype == 1) {
    static_cast<__nv_bfloat16*>(base)[i] = __float2bfloat16_rn(v);
  } else {
    static_cast<__half*>(base)[i] = __float2half_rn(v);
  }
}

__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }

// Monotone float -> uint32 map; 0 never occurs for a real value, so 0 marks
// "no candidate".
__device__ __forceinline__ uint32_t ord32(float v) {
  const uint32_t u = __float_as_uint(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float unord32(uint32_t k) {
  return __uint_as_float((k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k);
}
// Warp argmax, lowest index among equal values. Lanes pass their local best.
__device__ __forceinline__ int warp_argmax(uint32_t lv, int li,
                                           uint32_t* wmax_out) {
  const uint32_t wmax = __reduce_max_sync(0xffffffffu, lv);
  const uint32_t cand = lv == wmax ? static_cast<uint32_t>(li) : 0xFFFFFFFFu;
  *wmax_out = wmax;
  return static_cast<int>(__reduce_min_sync(0xffffffffu, cand));
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Routing of `rows` (<= kWarps) consecutive tokens by one CTA, in three
// stages so a single decode token is spread over all 256 threads:
//   1. every thread scores (token, expert) elements -> smem scores / keys;
//   2. warp r runs token r's top-k (its keys in registers, lane e % 32 owns
//      expert e), writes ids / weights / flags, marks chosen keys with 0 and
//      leaves (top logit, 1 / sum, denominator) in smem;
//   3. every thread writes the dense mask / weights.
// logit(i) returns the logit of element i = r * E + e.
template <int VPL, int SC, typename LogitFn>
__device__ __forceinline__ void route_rows(LogitFn logit, int t0, int rows,
                                           const EpilogueParams& p, float* st,
                                           uint32_t* kt, bool* chosen,
                                           int* flag_sink) {
  __shared__ float rowinfo[kWarps][3];
  const int E = p.E;
  const int n = rows * E;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

  // Two elements per thread per pass, loads first: both elements' partial
  // reads are in flight together (one L2 round trip instead of two at
  // E = 512 decode).
  constexpr int kLB = 2;
  for (int i0 = tid; i0 < n; i0 += kLB * blockDim.x) {
    float lv[kLB];
#pragma unroll
    for (int b = 0; b < kLB; ++b) {
      const int i = i0 + b * blockDim.x;
      lv[b] = i < n ? logit(i) : 0.f;
    }
#pragma unroll
    for (int b = 0; b < kLB; ++b) {
      const int i = i0 + b * blockDim.x;
      if (i >= n) break;
      float v = lv[b];
      if (p.round_logits) v = __bfloat162float(__float2bfloat16_rn(v));
      if (p.logits_out != nullptr)
        store_typed(p.logits_out, static_cast<int64_t>(t0) * E + i, v,
                    p.logits_dtype);
      float sc;
      uint32_t key;
      if (SC == kSoftmax) {
        // softmax is strictly increasing: select on the (rounded) logits; the
        // probabilities of the selected experts are formed below.
        sc = v;
        key = ord32(v);
      } else {
        if (SC == kSigmoid) {
          sc = __frcp_rn(1.0f + expf(-v));  // == 1.0f / (...), IEEE
        } else {
          sc = sqrtf(v > 20.f ? v : log1pf(expf(v)));
        }
        const int e = i - (i / E) * E;
        key = ord32(p.bias != nullptr ? sc + p.bias[e] : sc);
      }
      st[i] = sc;
      kt[i] = key;
    }
  }
  __syncthreads();

  if (warp < rows && p.n_group > 1) {
    uint32_t* krow = kt + warp * E;
    {
      // noaux_tc: group score = sum of the group's top-2 selection values;
      // experts outside the topk_group best groups are masked to -inf.
      const int gs = E / p.n_group;
      uint32_t keep = 0;
#pragma unroll 1
      for (int j = 0; j < p.topk_group; ++j) {
        int bi = -1;
        float best = 0.f;
#pragma unroll 1
        for (int g = 0; g < p.n_group; ++g) {
          if ((keep >> g) & 1u) continue;
          uint32_t lv = 0;
          int li = 0x7FFFFFFF;
#pragma unroll 1
          for (int e = g * gs + lane; e < (g + 1) * gs; e += 32) {
            if (krow[e] > lv) {
              lv = krow[e];
              li = e;
            }
          }
          uint32_t m1;
          const int i1 = warp_argmax(lv, li, &m1);
          uint32_t lv2 = 0;
#pragma unroll 1
          for (int e = g * gs + lane; e < (g + 1) * gs; e += 32)
            if (e != i1) lv2 = max(lv2, krow[e]);
          const uint32_t m2 = __reduce_max_sync(0xffffffffu, lv2);
          const float score = unord32(m1) + unord32(m2);
          if (bi < 0 || score > best) {
            bi = g;
            best = score;
          }
        }
        keep |= 1u << bi;
      }
      const uint32_t masked = ord32(-INFINITY);
#pragma unroll 1
      for (int e = lane; e < E; e += 32)
        if (!((keep >> (e / gs)) & 1u)) krow[e] = masked;
      __syncwarp();
    }
  }
  __syncthreads();

  int my_id = 0;
  // Top-k: warp r scans row r. Lane l owns experts l, l + 32, ... in
  // registers; each step takes the warp argmax (larger key, then lower
  // index -- torch's tie order) and removes it from the owner lane.
  if (warp < rows) {
    const int t = t0 + warp;
    float* srow = st + warp * E;
    const uint32_t* krow = kt + warp * E;
    bool* crow = chosen + warp * E;
    uint32_t key[VPL];  // 0 = not a candidate (no real value maps to 0)
    {
#pragma unroll
      for (int i = 0; i < VPL; ++i) {
        const int e = lane + 32 * i;
        key[i] = e < E ? krow[e] : 0u;
        if (e < E) crow[e] = false;
      }
      // Fast path: no K serial argmax rounds. Keys made unique with the
      // index (larger key, then lower index -- torch's tie order). The K-th
      // largest lane maximum tau bounds the K-th largest key from below
      // (the K best lane maxima are K distinct elements), so the top K are
      // among the elements >= tau; those (usually ~K..2K) are ranked all
      // pairs at once. More than 64 candidates: the rounds below.
      bool done_sel = false;
      {
        __shared__ uint64_t sbuf[kWarps][64];
        __shared__ int ssel[kWarps][kMaxK];
        const int K = p.K;
        uint64_t ck[VPL];
#pragma unroll
        for (int i = 0; i < VPL; ++i) {
          const int e = lane + 32 * i;
          ck[i] = e < E ? (static_cast<uint64_t>(key[i]) << 32) |
                              (0xFFFFFFFFu - static_cast<uint32_t>(e))
                        : 0ull;
        }
        uint64_t mt[VPL];
#pragma unroll
        for (int i = 0; i < VPL; ++i) mt[i] = ck[i];
#pragma unroll
        for (int w2 = VPL / 2; w2 > 0; w2 >>= 1)
#pragma unroll
          for (int i = 0; i < w2; ++i)
            mt[i] = mt[i + w2] > mt[i] ? mt[i + w2] : mt[i];
        const uint64_t m = mt[0];

        sbuf[warp][lane] = m;
        __syncwarp();
        int r = 0;
#pragma unroll 8
        for (int j = 0; j < 32; ++j) r += sbuf[warp][j] > m;
        const unsigned bal = __ballot_sync(0xffffffffu, r == K - 1);
        const uint64_t tau =
            bal ? __shfl_sync(0xffffffffu, m, __ffs(bal) - 1) : 0ull;

        // Compaction offsets from one ballot per slot (no scan chain).
        unsigned bal_i[VPL];
#pragma unroll
        for (int i = 0; i < VPL; ++i)
          bal_i[i] = __ballot_sync(0xffffffffu, ck[i] != 0ull && ck[i] >= tau);
        const unsigned lt = (1u << lane) - 1u;
        int total = 0, slot_pos[VPL];
#pragma unroll
        for (int i = 0; i < VPL; ++i) {
          slot_pos[i] = total + __popc(bal_i[i] & lt);
          total += __popc(bal_i[i]);
        }

        __syncwarp();
        if (total <= 64) {
#pragma unroll
          for (int i = 0; i < VPL; ++i)
            if ((bal_i[i] >> lane) & 1u) sbuf[warp][slot_pos[i]] = ck[i];
          __syncwarp();
          for (int c = lane; c < total; c += 32) {
            const uint64_t v = sbuf[warp][c];
            int rr = 0;
#pragma unroll 8
            for (int j = 0; j < total; ++j) rr += sbuf[warp][j] > v;
            if (rr < K)
              ssel[warp][rr] =
                  static_cast<int>(0xFFFFFFFFu - static_cast<uint32_t>(v));
          }
          __syncwarp();
          if (lane < K) my_id = ssel[warp][lane];
          done_sel = true;
        }
      }
      if (!done_sel) {
#pragma unroll 1
        for (int k = 0; k < p.K; ++k) {
          // Lane-local argmax as a tree (depth log2 VPL instead of a VPL-long
          // dependency chain: ~4x shorter per round at VPL = 16). Adjacent
          // pairs keep the lower slot on ties.
          uint32_t tk[VPL];
          int ti[VPL];
#pragma unroll
          for (int i = 0; i < VPL; ++i) {
            tk[i] = key[i];
            ti[i] = i;
          }
#pragma unroll
          for (int st2 = 1; st2 < VPL; st2 <<= 1)
#pragma unroll
            for (int i = 0; i + st2 < VPL; i += 2 * st2)
              if (tk[i + st2] > tk[i]) {
                tk[i] = tk[i + st2];
                ti[i] = ti[i + st2];
              }
          const uint32_t lv = tk[0];
          const int li = lv != 0u ? lane + 32 * ti[0] : 0x7FFFFFFF;
          uint32_t wmax;
          const int idx = warp_argmax(lv, li, &wmax);
          if ((idx & 31) == lane) {
#pragma unroll
            for (int i = 0; i < VPL; ++i)
              if (i == (idx >> 5)) key[i] = 0u;
          }
          if (lane == k) my_id = idx;
        }
      }
    }
    __syncwarp();
    if (lane < p.K) crow[my_id] = true;
    float my_w = lane < p.K ? srow[my_id] : 0.f;
    const float top_logit = __shfl_sync(0xffffffffu, my_w, 0);  // rank 0
    // Softmax: p_k = exp(l_k - m) / sum(exp(l - m)); with renorm the
    // full-row sum cancels, so only the selected experts are exponentiated.
    float inv_all = 1.f;
    if (SC == kSoftmax) {
      if (!p.renorm) {
        float sum = 0.f;
#pragma unroll 1
        for (int e = lane; e < E; e += 32) sum += expf(srow[e] - top_logit);
        inv_all = 1.f / warp_sum(sum);
      }
      my_w = lane < p.K ? expf(my_w - top_logit) * inv_all : 0.f;
    }
    float den = 1.f;
    if (p.renorm) den = warp_sum(lane < p.K ? my_w : 0.f) + p.eps;
    if (lane < p.K) {
      const float v = (p.renorm ? my_w / den : my_w) * p.scale;
      if (p.ids != nullptr) {
        p.ids[static_cast<int64_t>(t) * p.K + lane] = my_id;
        p.tw[static_cast<int64_t>(t) * p.K + lane] = v;
      }
      if (flag_sink != nullptr) flag_sink[my_id] = 1;
    }
    if (lane == 0) {
      rowinfo[warp][0] = top_logit;
      rowinfo[warp][1] = inv_all;
      rowinfo[warp][2] = den;
    }
  }
  __syncthreads();

  if (p.mask != nullptr) {
    for (int i = tid; i < n; i += blockDim.x) {
      const int r = rows == 1 ? 0 : i / E;
      const bool on = chosen[i];
      p.mask[static_cast<int64_t>(t0) * E + i] = on;
      float v = 0.f;
      if (on) {
        const float raw = SC == kSoftmax
                              ? expf(st[i] - rowinfo[r][0]) * rowinfo[r][1]
                              : st[i];
        v = (p.renorm ? raw / rowinfo[r][2] : raw) * p.scale;
      }
      store_typed(p.dw, static_cast<int64_t>(t0) * E + i, v, p.dw_dtype);
    }
  }
  __syncthreads();
}

// Tell the host ring this launch's slot writes are done (see FlagRing).
// No fence needed: the host only trusts done > S, i.e. a CTA of a LATER
// launch finished, which implies launch S drained.
__device__ __forceinline__ void signal_done(const EpilogueParams& p) {
  if (p.done_word == nullptr) return;
  if (threadIdx.x == 0)
    *reinterpret_cast<volatile int64_t*>(p.done_word) = p.seq;
}

// After a CTA has routed its tokens: the last CTA to finish publishes the
// device flags to pinned host memory and re-zeroes them (self-cleaning).
__device__ void publish_flags(const EpilogueParams& p, unsigned int n_ctas) {
  if (p.flags_host == nullptr) return;
  __shared__ bool last;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0)
    last = atomicAdd(&p.state->publish_count, 1u) == n_ctas - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  for (int e = threadIdx.x; e < p.E; e += blockDim.x) {
    volatile int* f = p.state->flags;
    p.flags_host[e] = f[e] != 0;
    f[e] = 0;
  }
  if (threadIdx.x == 0) p.state->publish_count = 0;
  signal_done(p);
}

// Hand-off for multi-row decode: instead of a grid barrier, the LAST R CTAs
// to finish phase 1 route (R = min(T, grid)); earlier CTAs exit at once.
// The last arrival starts routing without waiting; the other R - 1 poll the
// arrival count, which reaches the grid size moments later. Returns this
// CTA's rank among the routers, or -1. Routers call leave_routers() when
// done; the last resets the counters (the next launch on the stream starts
// after this one ends).
__device__ int last_arrivals(RouteState* st, int R) {
  __shared__ int s_rank;
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    const unsigned int a = atomicAdd(&st->bar_count, 1u);
    const int r = static_cast<int>(a) - static_cast<int>(gridDim.x - R);
    s_rank = r;
    if (r >= 0) {
      if (a != gridDim.x - 1) {
        volatile unsigned int* c = &st->bar_count;
        while (*c < gridDim.x) {
        }
      }
      __threadfence();
    }
  }
  __syncthreads();
  return s_rank;
}

// Called by each router after routing (off the critical path): the last one
// to leave resets the hand-off counters.
__device__ void leave_routers(RouteState* st, int R) {
  if (threadIdx.x == 0 &&
      atomicAdd(&st->bar_leave, 1u) == static_cast<unsigned int>(R - 1)) {
    st->bar_count = 0;
    st->bar_leave = 0;
  }
}

struct DecodeArgs {
  const void* x;
  int64_t ldx;
  const void* w;
  int64_t ldw;
  int H;
  int n_t, n_e, n_split, h_per_split;
  float* partial;  // [n_split][T][E]
};

// Work-item tile: BT tokens x kBE experts over one H split.
//   BT=16 : 8 warps = (kBE / 16) expert columns x (8 / columns) slices of
//           each H stage; slices are summed in a fixed order.
//   BT=64 : 8 warps = 4 token rows (16) x 2 expert column halves
//           (kBE / 2 experts each, kBE / 32 fragments per warp).
//   (A 64x128 tile with 2x2 warp fragments measured slower at T=256: fewer
//   items -> fewer SMs streaming the gate weight.)
#ifndef ROUTER_STAGES16
  #define ROUTER_STAGES16 5
#endif
constexpr int kBE16 = ROUTER_BE16;
constexpr int kBE64 = ROUTER_BE64;
template <int BT>
struct Tile {
  static constexpr int kBE = BT == 16 ? kBE16 : kBE64;
  static constexpr int kFr64 = kBE / 32;  // BT=64: 16x16 fragments per warp
  static constexpr int kCols = kBE / 16;  // BT=16 column groups
  static constexpr int kParts = kWarps / kCols;    // BT=16 H slices
  static constexpr int kHC = BT == 16 ? 128 : 64;  // H per stage
  static constexpr int kLd = kHC + kPad;
  static constexpr int kXElems = BT * kLd;
  static constexpr int kWElems = kBE * kLd;
  static constexpr int kSt = BT == 16 ? ROUTER_STAGES16 : kStages;
};

template <typename Tin, int BT>
constexpr int decode_smem_bytes() {
  using TL = Tile<BT>;
  constexpr int op = TL::kSt * (TL::kXElems + TL::kWElems) * sizeof(Tin);
  constexpr int acc = (BT == 16 ? TL::kParts * 16 : BT) * TL::kBE * 4;
  constexpr int red =
      2 * kRows * kMaxE * 4 + kRows * kMaxE;  // scores, keys, chosen
  constexpr int m = op > acc ? op : acc;
  return m > red ? m : red;
}

// Phase 1 for one work item.
template <typename Tin, int BT>
__device__ void logits_item(const DecodeArgs& a, const EpilogueParams& p,
                            int item, unsigned char* smem) {
  using TL = Tile<BT>;
  constexpr int kHC = TL::kHC, kLd = TL::kLd, kBE = TL::kBE;
  const int tile_t = item % a.n_t;
  const int rest = item / a.n_t;
  const int tile_e = rest % a.n_e, split = rest / a.n_e;
  const int t0 = tile_t * BT, e0 = tile_e * kBE;
  const int h_lo = split * a.h_per_split;
  const int h_hi = min(a.H, h_lo + a.h_per_split);
  const int T = p.T, E = p.E;
  const int tid = threadIdx.x, warp = tid >> 5;
  const Tin* x = static_cast<const Tin*>(a.x);
  const Tin* w = static_cast<const Tin*>(a.w);
  Tin* xs = reinterpret_cast<Tin*>(smem);
  Tin* ws = xs + TL::kSt * TL::kXElems;

  auto load_stage = [&](int stage, int h0) {
    constexpr int kVec = 8;  // 16 B per cp.async
    constexpr int kRowChunks = kHC / kVec;
    Tin* xst = xs + stage * TL::kXElems;
    Tin* wst = ws + stage * TL::kWElems;
    for (int c = tid; c < BT * kRowChunks; c += kThreads) {
      const int r = c / kRowChunks, col = (c % kRowChunks) * kVec;
      const int t = t0 + r, h = h0 + col;
      const bool ok = t < T && h < h_hi;
      const Tin* src = ok ? x + t * a.ldx + h : x;
      const uint32_t dst =
          static_cast<uint32_t>(__cvta_generic_to_shared(xst + r * kLd + col));
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst),
                   "l"(src), "r"(ok ? 16 : 0));
    }
    for (int c = tid; c < kBE * kRowChunks; c += kThreads) {
      const int r = c / kRowChunks, col = (c % kRowChunks) * kVec;
      const int e = e0 + r, h = h0 + col;
      const bool ok = e < E && h < h_hi;
      const Tin* src = ok ? w + e * a.ldw + h : w;
      const uint32_t dst =
          static_cast<uint32_t>(__cvta_generic_to_shared(wst + r * kLd + col));
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst),
                   "l"(src), "r"(ok ? 16 : 0));
    }
  };

  using FragA =
      wmma::fragment<wmma::matrix_a, 16, 16, 16, Tin, wmma::row_major>;
  using FragB =
      wmma::fragment<wmma::matrix_b, 16, 16, 16, Tin, wmma::col_major>;
  using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
  FragC acc[1][2];
  wmma::fill_fragment(acc[0][0], 0.f);
  wmma::fill_fragment(acc[0][1], 0.f);

  const int n_chunks = h_hi > h_lo ? (h_hi - h_lo + kHC - 1) / kHC : 0;
#pragma unroll
  for (int s = 0; s < TL::kSt - 1; ++s) {
    if (s < n_chunks) load_stage(s, h_lo + s * kHC);
    asm volatile("cp.async.commit_group;\n" ::);
  }
  for (int c = 0; c < n_chunks; ++c) {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(TL::kSt - 2));
    __syncthreads();
    if (c + TL::kSt - 1 < n_chunks)
      load_stage((c + TL::kSt - 1) % TL::kSt, h_lo + (c + TL::kSt - 1) * kHC);
    asm volatile("cp.async.commit_group;\n" ::);
    const Tin* xa = xs + (c % TL::kSt) * TL::kXElems;
    const Tin* wb = ws + (c % TL::kSt) * TL::kWElems;
    if constexpr (BT == 16) {
      constexpr int kSlice = kHC / TL::kParts;
      const int col = (warp % TL::kCols) * 16;
      const int h0 = (warp / TL::kCols) * kSlice;
#pragma unroll
      for (int kk = 0; kk < kSlice; kk += 16) {
        FragA fa;
        FragB fb;
        wmma::load_matrix_sync(fa, xa + h0 + kk, kLd);
        wmma::load_matrix_sync(fb, wb + col * kLd + h0 + kk, kLd);
        wmma::mma_sync(acc[0][0], fa, fb, acc[0][0]);
      }
    } else {
      const int row = (warp & 3) * 16, col = (warp >> 2) * (kBE / 2);
#pragma unroll
      for (int kk = 0; kk < kHC; kk += 16) {
        FragA fa;
        FragB fb[2];
        wmma::load_matrix_sync(fa, xa + row * kLd + kk, kLd);
#pragma unroll
        for (int j = 0; j < TL::kFr64; ++j) {
          wmma::load_matrix_sync(fb[j], wb + (col + 16 * j) * kLd + kk, kLd);
          wmma::mma_sync(acc[0][j], fa, fb[j], acc[0][j]);
        }
      }
    }
  }
  asm volatile("cp.async.wait_group 0;\n" ::);
  __syncthreads();
  float* cs = reinterpret_cast<float*>(smem);
  if constexpr (BT == 16) {
    const int col = (warp % TL::kCols) * 16, part = warp / TL::kCols;
    wmma::store_matrix_sync(cs + part * 16 * kBE + col, acc[0][0], kBE,
                            wmma::mem_row_major);
  } else {
    const int row = (warp & 3) * 16, col = (warp >> 2) * (kBE / 2);
#pragma unroll
    for (int j = 0; j < TL::kFr64; ++j)
      wmma::store_matrix_sync(cs + row * kBE + col + 16 * j, acc[0][j], kBE,
                              wmma::mem_row_major);
  }
  __syncthreads();
  for (int i = tid; i < BT * kBE; i += kThreads) {
    const int r = i / kBE, ce = i % kBE;
    const int t = t0 + r, e = e0 + ce;
    if (t < T && e < E) {
      float v = cs[i];
      if constexpr (BT == 16) {
#pragma unroll
        for (int q = 1; q < TL::kParts; ++q) v += cs[q * 16 * kBE + i];
      }
      a.partial[(static_cast<int64_t>(split) * T + t) * E + e] = v;
    }
  }
  __syncthreads();  // smem is reused by the next item
}

// Phase 2 for tokens [t_begin, t_end): reduce the split partials in a fixed
// order (deterministic) into smem, one warp routes each token.
template <int VPL, int SC>
__device__ void reduce_and_route(const DecodeArgs& a, const EpilogueParams& p,
                                 int t_begin, int t_end, unsigned char* smem,
                                 int n_routers) {
  const int T = p.T, E = p.E;
  float* st = reinterpret_cast<float*>(smem);                  // scores
  uint32_t* kt = reinterpret_cast<uint32_t*>(st + kRows * E);  // keys
  bool* chosen = reinterpret_cast<bool*>(kt + kRows * E);
  __shared__ int sflags[kMaxE];
  const bool want_flags = p.flags_host != nullptr;
  if (want_flags)
    for (int e = threadIdx.x; e < E; e += kThreads) sflags[e] = 0;
  const int64_t plane = static_cast<int64_t>(T) * E;
  const int n_split = a.n_split;
  for (int t0 = t_begin; t0 < t_end; t0 += kRows) {
    const float* base = a.partial + static_cast<int64_t>(t0) * E;
    // Fixed split order s = 0, 1, 2, ... (deterministic). Every load of a
    // thread's elements is issued before any is summed: one L2 round trip.
    // The sums go to st[i] for the thread that owns i in route_rows (same
    // i = tid + 256 j mapping), which reads them back before overwriting.
    const int n = min(kRows, t_end - t0) * E;
    for (int i0 = threadIdx.x; i0 < n; i0 += 2 * kThreads) {
      const int i1 = i0 + kThreads;
      float v0[kMaxSplit], v1[kMaxSplit];
#pragma unroll
      for (int sp = 0; sp < kMaxSplit; ++sp) {
        if (sp < n_split) {
          v0[sp] = __ldcg(base + i0 + sp * plane);
          v1[sp] = i1 < n ? __ldcg(base + i1 + sp * plane) : 0.f;
        }
      }
      float a0 = 0.f, a1 = 0.f;
#pragma unroll
      for (int sp = 0; sp < kMaxSplit; ++sp) {
        if (sp < n_split) {
          a0 += v0[sp];
          a1 += v1[sp];
        }
      }
      st[i0] = a0;
      if (i1 < n) st[i1] = a1;
    }
    auto logit = [&](int i) { return st[i]; };
    route_rows<VPL, SC>(logit, t0, min(kRows, t_end - t0), p, st, kt, chosen,
                        want_flags ? sflags : nullptr);
  }
  // Host pre-zeroed the pinned flags; set the experts this CTA routed to.
  // With many routers only the first to claim an expert in this launch
  // (tagged seq + 1, so the claims never need re-zeroing) writes it to
  // pinned memory: <= E PCIe writes instead of up to T * K.
  // The claim costs one L2 round trip before the write, so it is only used
  // when the picks repeat a lot (T * K >= 4 E: measured +1 us at b16 / b64,
  // -1 to -2.8 us at b256).
  const int tag = static_cast<int>(p.seq) + 1;
  const bool claim = n_routers > 1 && p.T * p.K >= 4 * E;
  if (want_flags && t_begin < t_end)
    for (int e = threadIdx.x; e < E; e += kThreads)
      if (sflags[e] && (!claim || atomicExch(&p.state->claimed[e], tag) != tag))
        p.flags_host[e] = true;
  signal_done(p);
}

// T <= BT (one token tile): the last CTA to finish its phase-1 items routes
// every token. Larger T: cooperative launch (co-residency for the hand-off
// wait), the last min(T, grid) CTAs to arrive route one token range each.
template <typename Tin, int BT, int VPL, int SC, bool kSingleTile>
__global__ void __launch_bounds__(kThreads)
    fused_route_pro6000_kernel(DecodeArgs a, EpilogueParams p) {
  extern __shared__ __align__(128) unsigned char smem[];
  // Warm L2 with what phase 2 touches (counter / barrier words, bias) while
  // phase 1 waits on the gate weight: after an L2 flush each of these would
  // otherwise add a serial DRAM round trip to the critical path.
  if (threadIdx.x < 32) {
    if (threadIdx.x == 0)
      asm volatile("prefetch.global.L2 [%0];" ::"l"(p.state));
    if (p.bias != nullptr)
      for (int e = threadIdx.x * 32; e < p.E; e += 32 * 32)
        asm volatile("prefetch.global.L2 [%0];" ::"l"(p.bias + e));
  }
  // Phase 2 reads the bias from shared memory (loaded under phase 1).
  __shared__ float sbias[kMaxE];
  EpilogueParams q = p;
  if (p.bias != nullptr) {
    // Asynchronous (its own cp.async group, older than the mainloop's): the
    // bias arrives alongside the first weight chunk instead of stalling
    // every CTA for a DRAM round trip before its first load.
    for (int e = threadIdx.x; e < p.E; e += kThreads) {
      const uint32_t dst =
          static_cast<uint32_t>(__cvta_generic_to_shared(sbias + e));
      asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dst),
                   "l"(p.bias + e));
    }
    asm volatile("cp.async.commit_group;\n" ::);
    q.bias = sbias;
  }
  const int n_items = a.n_t * a.n_e * a.n_split;
  for (int item = blockIdx.x; item < n_items; item += gridDim.x)
    logits_item<Tin, BT>(a, p, item, smem);
  __threadfence();
  if constexpr (kSingleTile) {
    __shared__ bool last;
    __syncthreads();
    if (threadIdx.x == 0)
      last = atomicAdd(&p.state->tile_count, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    reduce_and_route<VPL, SC>(a, q, 0, p.T, smem, 1);
    if (threadIdx.x == 0) p.state->tile_count = 0;
  } else {
    const int R = min(p.T, static_cast<int>(gridDim.x));
    const int rank = last_arrivals(p.state, R);
    if (rank < 0) return;
    const int tpc = (p.T + R - 1) / R;
    const int t_begin = rank * tpc;
    reduce_and_route<VPL, SC>(a, q, t_begin, min(p.T, t_begin + tpc), smem, R);
    leave_routers(p.state, R);
  }
}

// Large-T routing over precomputed logits: one warp per token.
template <typename Tl, int VPL, int SC>
__global__ void __launch_bounds__(256)
    route_logits_pro6000_kernel(const Tl* __restrict__ logits,
                                EpilogueParams p) {
  extern __shared__ __align__(16) unsigned char smem[];
  float* st = reinterpret_cast<float*>(smem);
  uint32_t* kt = reinterpret_cast<uint32_t*>(st + kWarps * p.E);
  bool* chosen = reinterpret_cast<bool*>(kt + kWarps * p.E);
  const int t0 = blockIdx.x * kWarps;
  const Tl* base = logits + static_cast<int64_t>(t0) * p.E;
  if (t0 < p.T)
    route_rows<VPL, SC>([&](int i) { return to_float(base[i]); }, t0,
                        min(kWarps, p.T - t0), p, st, kt, chosen,
                        p.flags_host != nullptr ? p.state->flags : nullptr);
  publish_flags(p, gridDim.x);
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

struct StateKey {
  int device;
  cudaStream_t stream;
  bool operator==(const StateKey& o) const {
    return device == o.device && stream == o.stream;
  }
};
struct StateKeyHash {
  size_t operator()(const StateKey& k) const {
    return std::hash<void*>()(k.stream) ^ (static_cast<size_t>(k.device) << 1);
  }
};

RouteState* state_for(int device, cudaStream_t stream) {
  static std::mutex mu;
  static std::unordered_map<StateKey, at::Tensor, StateKeyHash> bufs;
  std::lock_guard<std::mutex> lock(mu);
  at::Tensor& buf = bufs[StateKey{device, stream}];
  if (!buf.defined()) {
    buf = at::zeros(
        {static_cast<int64_t>(sizeof(RouteState))},
        at::TensorOptions().dtype(at::kByte).device(at::kCUDA, device));
  }
  return reinterpret_cast<RouteState*>(buf.data_ptr());
}

// Pinned active-flag slots, one ring per (device, stream). Every routing
// launch stores its sequence number into a pinned word when its CTAs finish
// (zero-copy), so the host can tell that the launch which last used a slot
// has completed with a plain memory read -- no allocation, no CUDA event, no
// API call per MoE layer. A slot is reused kFlagSlots calls later; the
// returned flags stay valid until then (the dispatcher consumes them within
// the same layer).
constexpr int kFlagSlots = 256;
struct FlagRing {
  at::Tensor buf;  // pinned: [kFlagSlots * kMaxE] flags + 8 B done word
  int64_t slot_seq[kFlagSlots] = {0};
  int64_t issued = 0;
  int next = 0;
  volatile int64_t* done() {
    return reinterpret_cast<volatile int64_t*>(
        static_cast<uint8_t*>(buf.data_ptr()) +
        static_cast<int64_t>(kFlagSlots) * kMaxE);
  }
  // Returns (slot, seq). Launch S is complete once done > S: a CTA of a
  // later launch on the same stream has finished, so S has drained.
  std::pair<int, int64_t> acquire() {
    const int s = next;
    next = (next + 1) % kFlagSlots;
    while (slot_seq[s] != 0 && *done() <= slot_seq[s]) {
      // The GPU is >= kFlagSlots routing launches behind: wait.
    }
    slot_seq[s] = ++issued;
    return {s, issued};
  }
  // Per-slot views, created once per expert count (no per-call op).
  int64_t views_e = -1;
  std::vector<at::Tensor> views;
  const at::Tensor& view(int s, int64_t E) {
    if (views_e != E) {
      views.clear();
      for (int i = 0; i < kFlagSlots; ++i)
        views.push_back(buf.narrow(0, static_cast<int64_t>(i) * kMaxE, E));
      views_e = E;
    }
    return views[s];
  }
};

FlagRing* flag_ring_for(int device, cudaStream_t stream) {
  static std::mutex mu;
  static std::unordered_map<StateKey, std::unique_ptr<FlagRing>, StateKeyHash>
      rings;
  std::lock_guard<std::mutex> lock(mu);
  auto& r = rings[StateKey{device, stream}];
  if (!r) {
    r = std::make_unique<FlagRing>();
    r->buf =
        at::zeros({static_cast<int64_t>(kFlagSlots) * kMaxE + 64},
                  at::TensorOptions().dtype(at::kBool).pinned_memory(true));
  }
  return r.get();
}

// Split-K partials: a grow-only fp32 buffer per (device, stream). Stream
// order makes reuse safe; it is fully overwritten before it is read.
float* scratch_for(int device, cudaStream_t stream, int64_t numel) {
  static std::mutex mu;
  static std::unordered_map<StateKey, at::Tensor, StateKeyHash> bufs;
  std::lock_guard<std::mutex> lock(mu);
  at::Tensor& buf = bufs[StateKey{device, stream}];
  if (!buf.defined() || buf.numel() < numel) {
    buf = at::empty(
        {std::max<int64_t>(numel, 1 << 20)},
        at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, device));
  }
  return buf.data_ptr<float>();
}

// logits[T, E] = x[T, H] @ w[E, H]^T on tensor cores (fp32 accumulate) with
// a cuBLASLt algorithm that does not split K: at::mm may pick split-K,
// which adds a reduction kernel (a third launch per MoE layer).
cudaDataType_t lt_type(at::ScalarType t) {
  return t == at::kFloat      ? CUDA_R_32F
         : t == at::kBFloat16 ? CUDA_R_16BF
                              : CUDA_R_16F;
}

at::Tensor lt_logits(const at::Tensor& x, const at::Tensor& w,
                     at::ScalarType out_dtype) {
  const int64_t T = x.size(0), H = x.size(1), E = w.size(0);
  at::Tensor out = at::empty({T, E}, x.options().dtype(out_dtype));
  cublasLtHandle_t handle = at::cuda::getCurrentCUDABlasLtHandle();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  struct Key {
    int64_t T, H, E, ldx, ldw;
    int in, out;
    bool operator==(const Key& o) const {
      return T == o.T && H == o.H && E == o.E && ldx == o.ldx && ldw == o.ldw &&
             in == o.in && out == o.out;
    }
  };
  struct KeyHash {
    size_t operator()(const Key& k) const {
      return std::hash<int64_t>()(k.T * 1000003 + k.H * 101 + k.E) ^
             (k.in * 7 + k.out);
    }
  };
  struct Plan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t algo;
    size_t ws = 0;
  };
  static std::mutex mu;
  static std::unordered_map<Key, Plan, KeyHash> plans;
  const Key key{T,
                H,
                E,
                x.stride(0),
                w.stride(0),
                static_cast<int>(x.scalar_type()),
                static_cast<int>(out_dtype)};
  Plan plan;
  {
    std::lock_guard<std::mutex> lock(mu);
    auto it = plans.find(key);
    if (it != plans.end()) plan = it->second;
  }
  const cudaDataType_t tin = lt_type(x.scalar_type());
  const cudaDataType_t tout = lt_type(out_dtype);
  if (plan.op == nullptr) {
    // Column-major view: C^T[E, T] = W[E, H] (op T on the [H, E] buffer)
    // times X^T[H, T].
    TORCH_CUDABLAS_CHECK(
        cublasLtMatmulDescCreate(&plan.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
        plan.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
        plan.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    TORCH_CUDABLAS_CHECK(
        cublasLtMatrixLayoutCreate(&plan.a, tin, H, E, w.stride(0)));
    TORCH_CUDABLAS_CHECK(
        cublasLtMatrixLayoutCreate(&plan.b, tin, H, T, x.stride(0)));
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.c, tout, E, T, E));
    cublasLtMatmulPreference_t pref;
    TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    const size_t max_ws = 32u << 20;
    TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws,
        sizeof(max_ws)));
    cublasLtMatmulHeuristicResult_t res[16];
    int n = 0;
    TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        handle, plan.op, plan.a, plan.b, plan.c, plan.c, pref, 16, res, &n));
    cublasLtMatmulPreferenceDestroy(pref);
    // Take the best candidate that runs unsplit: force split-K = 1 and no
    // reduction scheme, keep it if cuBLASLt accepts that configuration.
    int pick = -1;
    for (int i = 0; i < n && pick < 0; ++i) {
      if (res[i].state != CUBLAS_STATUS_SUCCESS) continue;
      const uint32_t one = 1, none = CUBLASLT_REDUCTION_SCHEME_NONE;
      cublasLtMatmulAlgoConfigSetAttribute(
          &res[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &one, sizeof(one));
      cublasLtMatmulAlgoConfigSetAttribute(
          &res[i].algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &none,
          sizeof(none));
      cublasLtMatmulHeuristicResult_t check;
      if (cublasLtMatmulAlgoCheck(handle, plan.op, plan.a, plan.b, plan.c,
                                  plan.c, &res[i].algo,
                                  &check) == CUBLAS_STATUS_SUCCESS &&
          check.workspaceSize <= max_ws) {
        res[i].workspaceSize = check.workspaceSize;
        pick = i;
      }
    }
    TORCH_CHECK(pick >= 0, "fused_route: no cuBLASLt algorithm for logits");
    plan.algo = res[pick].algo;
    plan.ws = res[pick].workspaceSize;
    std::lock_guard<std::mutex> lock(mu);
    plans.emplace(key, plan);
  }
  at::Tensor ws;
  if (plan.ws > 0)
    ws = at::empty({static_cast<int64_t>(plan.ws)},
                   x.options().dtype(at::kByte));
  const float alpha = 1.f, beta = 0.f;
  TORCH_CUDABLAS_CHECK(cublasLtMatmul(
      handle, plan.op, &alpha, w.data_ptr(), plan.a, x.data_ptr(), plan.b,
      &beta, out.data_ptr(), plan.c, out.data_ptr(), plan.c, &plan.algo,
      plan.ws > 0 ? ws.data_ptr() : nullptr, plan.ws, stream));
  return out;
}

int dtype_code(at::ScalarType t) {
  switch (t) {
    case at::kFloat:
      return 0;
    case at::kBFloat16:
      return 1;
    case at::kHalf:
      return 2;
    default:
      TORCH_CHECK(false, "fused_route: unsupported dtype ", t);
  }
}

template <int VPL, int SC>
void launch_logits_route(const at::Tensor& logits, const EpilogueParams& p,
                         cudaStream_t stream) {
  constexpr int warps = 8;
  dim3 grid((p.T + warps - 1) / warps);
  const size_t smem =
      static_cast<size_t>(warps) * p.E * (2 * sizeof(float) + 1);
  switch (logits.scalar_type()) {
    case at::kFloat:
      route_logits_pro6000_kernel<float, VPL, SC>
          <<<grid, warps * 32, smem, stream>>>(logits.data_ptr<float>(), p);
      break;
    case at::kBFloat16:
      route_logits_pro6000_kernel<__nv_bfloat16, VPL, SC>
          <<<grid, warps * 32, smem, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()), p);
      break;
    case at::kHalf:
      route_logits_pro6000_kernel<__half, VPL, SC>
          <<<grid, warps * 32, smem, stream>>>(
              reinterpret_cast<const __half*>(logits.data_ptr()), p);
      break;
    default:
      TORCH_CHECK(false, "fused_route: unsupported logits dtype");
  }
}

template <typename Tin, int BT, int VPL, int SC, bool kSingleTile>
void launch_fused(const at::Tensor& x, const at::Tensor& w, int H,
                  EpilogueParams p, cudaStream_t stream, int device) {
  auto kernel = fused_route_pro6000_kernel<Tin, BT, VPL, SC, kSingleTile>;
  constexpr int smem = decode_smem_bytes<Tin, BT>();
  static int max_ctas[16] = {0};  // co-resident CTAs, per device
  int& cap = max_ctas[device & 15];
  const int sms = at::cuda::getDeviceProperties(device)->multiProcessorCount;
  if (cap == 0) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    int per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_sm, kernel, kThreads, smem));
    cap = std::max(1, std::min(per_sm, 2) * sms);
  }
  const int T = p.T, E = p.E;
  const int n_t = (T + BT - 1) / BT;
  const int n_e = (E + Tile<BT>::kBE - 1) / Tile<BT>::kBE;
  constexpr int kHC = Tile<BT>::kHC;
  const int max_split = std::max(1, H / kHC);
  // Wide rows (E > 320): fewer partial planes keep the single-token
  // reduction short; measured ~1 us faster for E = 384 / 512 decode.
  // Wide rows (E > 320) in the single-CTA hand-off (b1): fewer partial planes
  // keep its reduction short. Multi-row decode spreads the reduction over
  // R CTAs, so it takes more splits (more CTAs streaming the weight).
  const int split_cap = E > 320 && kSingleTile ? kMaxSplit / 2 : kMaxSplit;
  int split = std::max(1, std::min({max_split, split_cap, sms / (n_t * n_e)}));
  int h_per_split = ((H + split - 1) / split + kHC - 1) / kHC * kHC;
  split = (H + h_per_split - 1) / h_per_split;
  const int n_items = n_t * n_e * split;
  const int grid = std::min(n_items, cap);

  float* partial =
      scratch_for(device, stream, static_cast<int64_t>(split) * T * E);
  DecodeArgs a{x.data_ptr(), x.stride(0), w.data_ptr(), w.stride(0), H,
               n_t,          n_e,         split,        h_per_split, partial};
  if constexpr (kSingleTile) {
    kernel<<<grid, kThreads, smem, stream>>>(a, p);
  } else {
    void* args[] = {&a, &p};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(kernel),
                                               dim3(grid), dim3(kThreads), args,
                                               smem, stream));
  }
}

template <int VPL, int SC>
void launch_fused_dispatch(const at::Tensor& x, const at::Tensor& w, int H,
                           const EpilogueParams& p, cudaStream_t stream,
                           int device) {
  const int T = p.T;
  if (x.scalar_type() == at::kBFloat16) {
    if (T <= kRows && T * p.E <= kSingleTileMaxElems)
      launch_fused<__nv_bfloat16, 16, VPL, SC, true>(x, w, H, p, stream,
                                                     device);
    else if (T < kBT64MinTokens)
      launch_fused<__nv_bfloat16, 16, VPL, SC, false>(x, w, H, p, stream,
                                                      device);
    else
      launch_fused<__nv_bfloat16, 64, VPL, SC, false>(x, w, H, p, stream,
                                                      device);
  } else {
    if (T <= kRows && T * p.E <= kSingleTileMaxElems)
      launch_fused<__half, 16, VPL, SC, true>(x, w, H, p, stream, device);
    else if (T < kBT64MinTokens)
      launch_fused<__half, 16, VPL, SC, false>(x, w, H, p, stream, device);
    else
      launch_fused<__half, 64, VPL, SC, false>(x, w, H, p, stream, device);
  }
}

template <template <int, int> class L, typename... Args>
void dispatch_vpl_sc(int E, int sc, Args&&... args) {
  auto by_sc = [&](auto vpl) {
    constexpr int V = decltype(vpl)::value;
    if (sc == kSoftmax)
      L<V, kSoftmax>::run(std::forward<Args>(args)...);
    else if (sc == kSigmoid)
      L<V, kSigmoid>::run(std::forward<Args>(args)...);
    else
      L<V, kSqrtSoftplus>::run(std::forward<Args>(args)...);
  };
  if (E <= 128)
    by_sc(std::integral_constant<int, 4>{});
  else if (E <= 256)
    by_sc(std::integral_constant<int, 8>{});
  else
    by_sc(std::integral_constant<int, 16>{});
}

template <int VPL, int SC>
struct FusedL {
  static void run(const at::Tensor& x, const at::Tensor& w, int H,
                  const EpilogueParams& p, cudaStream_t s, int dev) {
    launch_fused_dispatch<VPL, SC>(x, w, H, p, s, dev);
  }
};
template <int VPL, int SC>
struct LogitsL {
  static void run(const at::Tensor& logits, const EpilogueParams& p,
                  cudaStream_t s) {
    launch_logits_route<VPL, SC>(logits, p, s);
  }
};

}  // namespace moe_router
#include "kernel/fused_router_gemm.cuh"
namespace moe_router {

// Flags produced together with a mask on this thread, for the executor
// (take_active_flags): identity of the mask, held weakly.
thread_local c10::weak_intrusive_ptr<c10::TensorImpl> t_last_mask{
    c10::intrusive_ptr<c10::TensorImpl>()};
thread_local at::Tensor t_last_flags;

void remember_flags(const at::Tensor& mask, const at::Tensor& flags) {
  t_last_mask =
      c10::weak_intrusive_ptr<c10::TensorImpl>(mask.getIntrusivePtr());
  t_last_flags = flags;
}

c10::optional<at::Tensor> take_active_flags(const at::Tensor& mask) {
  if (!t_last_flags.defined()) return c10::nullopt;
  auto held = t_last_mask.lock();
  at::Tensor flags = std::move(t_last_flags);
  t_last_flags = at::Tensor();
  t_last_mask = c10::weak_intrusive_ptr<c10::TensorImpl>(
      c10::intrusive_ptr<c10::TensorImpl>());
  if (held && mask.defined() && held.get() == mask.unsafeGetTensorImpl())
    return flags;
  return c10::nullopt;
}

// Returns [topk_ids, topk_weights, router_mask, routing_weights, logits,
// active_flags] (unrequested outputs are None), or an empty list when the
// shape / dtype / alignment is outside what this kernel supports.
std::vector<at::Tensor> fused_route(
    const at::Tensor& x_in, const at::Tensor& w_in, int64_t top_k,
    int64_t scoring, const c10::optional<at::Tensor>& bias_in, int64_t n_group,
    int64_t topk_group, bool renormalize, double scaling, double eps,
    bool round_logits, at::ScalarType weights_dtype,
    at::ScalarType logits_dtype, bool return_logits, bool dense,
    bool active_flags, bool compact, int64_t gemm_min_tokens) {
  TORCH_CHECK(x_in.is_cuda() && w_in.is_cuda(), "fused_route: CUDA tensors");
  TORCH_CHECK(x_in.dim() == 2 && w_in.dim() == 2, "fused_route: 2-D inputs");
  const int64_t T = x_in.size(0), H = x_in.size(1), E = w_in.size(0);
  TORCH_CHECK(w_in.size(1) == H, "fused_route: hidden size mismatch");
  TORCH_CHECK(top_k >= 1 && top_k <= E, "fused_route: 1 <= top_k <= E");
  TORCH_CHECK(n_group >= 1 && E % n_group == 0 && topk_group >= 1 &&
                  topk_group <= n_group,
              "fused_route: invalid n_group / topk_group");
  TORCH_CHECK(scoring >= 0 && scoring <= 2, "fused_route: bad scoring");
  const at::Tensor x = x_in.stride(1) == 1 ? x_in : x_in.contiguous();
  const at::Tensor w = w_in.stride(1) == 1 ? w_in : w_in.contiguous();
  // Valid but outside this kernel's envelope: the caller falls back.
  const bool supported =
      x.scalar_type() == w.scalar_type() &&
      (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf) &&
      E <= kMaxE && top_k <= kMaxK && n_group <= kMaxGroups && H % 8 == 0 &&
      x.stride(0) % 8 == 0 && w.stride(0) % 8 == 0 &&
      reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0;
  const int device = x.get_device();
  if (!supported ||
      !moe_infinity::hw::Supports(moe_infinity::hw::Gate::kRtxPro6000, device))
    return {};
  c10::cuda::OptionalCUDAGuard guard;
  if (c10::cuda::current_device() != device) guard.set_index(device);
  const auto cstream = at::cuda::getCurrentCUDAStream(device);
  cudaStream_t stream = cstream.stream();

  // One device allocation for every output (this runs once per MoE layer
  // per decode step; each allocator round trip is host latency).
  const auto opts = x.options();
  const int64_t wsz = c10::elementSize(weights_dtype);
  const int64_t lsz = c10::elementSize(logits_dtype);
  auto align = [](int64_t b) { return (b + 255) / 256 * 256; };
  const int64_t b_dw = dense ? align(T * E * wsz) : 0;
  const int64_t b_lg = return_logits ? align(T * E * lsz) : 0;
  const int64_t b_ids = compact ? align(T * top_k * 4) : 0;
  const int64_t b_tw = compact ? align(T * top_k * 4) : 0;
  const int64_t b_mask = dense ? align(T * E) : 0;
  const int64_t total = b_dw + b_lg + b_ids + b_tw + b_mask;
  at::Tensor ids, tw, mask, dw, logits_out, bias, flags;
  if (total > 0) {
    const at::Tensor buf = at::empty({total}, opts.dtype(at::kByte));
    auto* base = static_cast<uint8_t*>(buf.data_ptr());
    const c10::Storage keep = buf.storage();
    auto view = [&](int64_t off, at::IntArrayRef sizes, at::ScalarType dt) {
      return at::from_blob(base + off, sizes, [keep](void*) {}, opts.dtype(dt));
    };
    int64_t off = 0;
    if (dense) dw = view(off, {T, E}, weights_dtype), off += b_dw;
    if (return_logits)
      logits_out = view(off, {T, E}, logits_dtype), off += b_lg;
    if (compact) {
      ids = view(off, {T, top_k}, at::kInt), off += b_ids;
      tw = view(off, {T, top_k}, at::kFloat), off += b_tw;
    }
    if (dense) mask = view(off, {T, E}, at::kBool);
  }
  if (bias_in.has_value() && bias_in->defined()) {
    const at::Tensor& b = *bias_in;
    bias = (b.is_cuda() && b.get_device() == device &&
            b.scalar_type() == at::kFloat && b.is_contiguous())
               ? b
               : b.to(x.device(), at::kFloat).contiguous();
    TORCH_CHECK(bias.numel() == E, "fused_route: bias must have E elements");
  }
  volatile int64_t* done_word = nullptr;
  int64_t seq = 0;
  if (active_flags) {
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture == cudaStreamCaptureStatusNone) {
      FlagRing* ring = flag_ring_for(device, stream);
      const auto slot = ring->acquire();
      flags = ring->view(slot.first, E);
      done_word = ring->done();
      seq = slot.second;
    } else {
      flags = at::empty(
          {E}, at::TensorOptions().dtype(at::kBool).pinned_memory(true));
      // The kernel writes this pinned block: keep it from being recycled
      // until the stream has passed this point.
      at::getHostAllocator(at::kCUDA)->record_event(
          flags.data_ptr(), flags.storage().data_ptr().get_context(),
          cstream.unwrap());
    }
    // Zeroed on the host; the routing kernel only sets flags.
    std::memset(flags.data_ptr(), 0, E);
  }
  if (T == 0) return {ids, tw, mask, dw, logits_out, flags};

  EpilogueParams p;
  p.T = static_cast<int>(T);
  p.E = static_cast<int>(E);
  p.K = static_cast<int>(top_k);
  p.bias = bias.defined() ? bias.data_ptr<float>() : nullptr;
  p.n_group = static_cast<int>(n_group);
  p.topk_group = static_cast<int>(topk_group);
  p.renorm = renormalize;
  p.scale = static_cast<float>(scaling);
  p.eps = static_cast<float>(eps);
  p.round_logits = round_logits;
  p.ids = compact ? ids.data_ptr<int32_t>() : nullptr;
  p.tw = compact ? tw.data_ptr<float>() : nullptr;
  p.mask = dense ? mask.data_ptr<bool>() : nullptr;
  p.dw = dense ? dw.data_ptr() : nullptr;
  p.dw_dtype = dense ? dtype_code(weights_dtype) : 0;
  p.logits_out = return_logits ? logits_out.data_ptr() : nullptr;
  p.logits_dtype = return_logits ? dtype_code(logits_dtype) : 0;
  p.state = state_for(device, stream);
  p.flags_host = active_flags ? flags.data_ptr<bool>() : nullptr;
  p.done_word = const_cast<int64_t*>(done_word);
  p.seq = seq;

  const int sc = static_cast<int>(scoring);
  // gemm_min_tokens <= 0: automatic. The single fused launch wins while its
  // split-K GEMM keeps every SM busy (decode, and small prefill chunks);
  // beyond that cuBLASLt's GEMM plus the routing launch is faster.
  const bool use_gemm =
      T > kFusedMaxTokens ||
      (gemm_min_tokens > 0 ? T >= gemm_min_tokens
                           : (T > 256 && T * E > kFusedMaxElems));
  // Fused CUTLASS GEMM + top-k, E in (128, 256] (pro6000 tile table):
  //  - 64-token tiles while they fit one wave of SMs (4K-12K tokens: 1.1-1.4x
  //    faster than cuBLASLt + routing);
  //  - 128-token tiles from kWideBigWaves waves of them (200K: 0.80-0.87 of
  //    the tensor-core roof vs 0.61-0.79 for cuBLASLt + routing);
  //  - cuBLASLt + routing in between (a 2-3 wave tail costs the fused
  //    kernel as much as its fusion saves).
  const int sms = at::cuda::getDeviceProperties(device)->multiProcessorCount;
  const bool wide_one_wave = (T + 63) / 64 <= sms;
  const bool wide_big = (T + 127) / 128 >= kWideBigWaves * sms;
  if (use_gemm && T >= kWideMinTokens && (wide_one_wave || wide_big) &&
      try_launch_wide(x, w, static_cast<int>(H), p, sc, wide_big, stream,
                      device)) {
    // fused CUTLASS GEMM + top-k (one launch), E in (128, 512]
  } else if (use_gemm) {
    // Tensor-core logits (cuBLASLt, no split-K so the layer stays at two
    // launches); fp32 out unless the router rounds to the model dtype.
    at::Tensor logits =
        lt_logits(x, w, round_logits ? x.scalar_type() : at::kFloat);
    dispatch_vpl_sc<LogitsL>(static_cast<int>(E), sc, logits, p, stream);
  } else {
    dispatch_vpl_sc<FusedL>(static_cast<int>(E), sc, x, w, static_cast<int>(H),
                            p, stream, device);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (active_flags && dense) remember_flags(mask, flags);
  return {ids, tw, mask, dw, logits_out, flags};
}

// A router bound to one gate: per-call work is a single-tensor call (the
// static arguments are converted once). One per MoE layer in the wrappers.
class Router {
 public:
  Router(at::Tensor weight, int64_t top_k, int64_t scoring,
         c10::optional<at::Tensor> bias, int64_t n_group, int64_t topk_group,
         bool renormalize, double scaling, double eps, bool round_logits,
         at::ScalarType weights_dtype, at::ScalarType logits_dtype,
         bool return_logits, bool dense, bool active_flags, bool compact,
         int64_t gemm_min_tokens)
      : w_(weight.stride(1) == 1 ? weight : weight.contiguous()),
        top_k_(top_k),
        scoring_(scoring),
        n_group_(n_group),
        topk_group_(topk_group),
        renormalize_(renormalize),
        scaling_(scaling),
        eps_(eps),
        round_logits_(round_logits),
        weights_dtype_(weights_dtype),
        logits_dtype_(logits_dtype),
        return_logits_(return_logits),
        dense_(dense),
        active_flags_(active_flags),
        compact_(compact),
        gemm_min_tokens_(gemm_min_tokens) {
    if (bias.has_value() && bias->defined())
      bias_ = bias->to(w_.device(), at::kFloat).contiguous();
  }
  std::vector<at::Tensor> operator()(const at::Tensor& x) const {
    return fused_route(
        x, w_, top_k_, scoring_,
        bias_.defined() ? c10::optional<at::Tensor>(bias_) : c10::nullopt,
        n_group_, topk_group_, renormalize_, scaling_, eps_, round_logits_,
        weights_dtype_, logits_dtype_, return_logits_, dense_, active_flags_,
        compact_, gemm_min_tokens_);
  }
  const at::Tensor& weight() const { return w_; }

 private:
  at::Tensor w_, bias_;
  int64_t top_k_, scoring_, n_group_, topk_group_;
  bool renormalize_;
  double scaling_, eps_;
  bool round_logits_;
  at::ScalarType weights_dtype_, logits_dtype_;
  bool return_logits_, dense_, active_flags_, compact_;
  int64_t gemm_min_tokens_;
};

}  // namespace moe_router

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_route", &moe_router::fused_route,
        "Fused MoE routing: logits + scoring + (grouped) top-k + renorm",
        py::arg("hidden_states"), py::arg("weight"), py::arg("top_k"),
        py::arg("scoring"), py::arg("bias"), py::arg("n_group"),
        py::arg("topk_group"), py::arg("renormalize"), py::arg("scaling"),
        py::arg("eps"), py::arg("round_logits"), py::arg("weights_dtype"),
        py::arg("logits_dtype"), py::arg("return_logits"), py::arg("dense"),
        py::arg("active_flags"), py::arg("compact"),
        py::arg("gemm_min_tokens"));
  m.def(
      "acquire_flags",
      [](int64_t device, int64_t E) {
        // For kernels launched from Python (Triton prefill path): a zeroed
        // pinned flag slot, the ring's completion word and this launch's
        // sequence number (see FlagRing).
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(static_cast<int>(device)).stream();
        auto* ring =
            moe_router::flag_ring_for(static_cast<int>(device), stream);
        const auto slot = ring->acquire();
        at::Tensor flags = ring->view(slot.first, E);
        std::memset(flags.data_ptr(), 0, E);
        at::Tensor done =
            ring->buf
                .narrow(0,
                        static_cast<int64_t>(moe_router::kFlagSlots) *
                            moe_router::kMaxE,
                        8)
                .view(at::kLong);
        return py::make_tuple(flags, done, slot.second);
      },
      py::arg("device"), py::arg("E"));
  m.def("remember_flags", &moe_router::remember_flags);
  m.def("take_active_flags", &moe_router::take_active_flags,
        "Pinned active-expert flags produced with this mask by the last "
        "routing call on this thread (consumed), or None");
  py::class_<moe_router::Router>(m, "Router")
      .def(
          py::init<at::Tensor, int64_t, int64_t, c10::optional<at::Tensor>,
                   int64_t, int64_t, bool, double, double, bool, at::ScalarType,
                   at::ScalarType, bool, bool, bool, bool, int64_t>(),
          py::arg("weight"), py::arg("top_k"), py::arg("scoring"),
          py::arg("bias"), py::arg("n_group"), py::arg("topk_group"),
          py::arg("renormalize"), py::arg("scaling"), py::arg("eps"),
          py::arg("round_logits"), py::arg("weights_dtype"),
          py::arg("logits_dtype"), py::arg("return_logits"), py::arg("dense"),
          py::arg("active_flags"), py::arg("compact"),
          py::arg("gemm_min_tokens"))
      .def("__call__", &moe_router::Router::operator())
      .def_property_readonly("weight", &moe_router::Router::weight);
}
