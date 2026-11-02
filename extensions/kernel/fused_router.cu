// Fused MoE routing (#315): gate logits on tensor cores + scoring + (grouped)
// top-k + renorm, written straight to compact ids / weights, the dense [T, E]
// mask / weights the expert dispatcher consumes, and (optionally) the per-layer
// active-expert flags in pinned host memory -- so the dispatcher never has to
// reduce the mask or copy it back, and nothing on the path syncs the host.
//
//   T <= kFusedMaxTokens: ONE cooperative launch.
//     phase 1  split-K x expert-tile x token-tile work items (WMMA bf16/fp16,
//              fp32 accumulate) write fp32 partials;
//     grid barrier;
//     phase 2  every CTA reduces the partials of its tokens in a fixed order
//              (deterministic) and routes them, one warp per token.
//   larger T: cuBLAS tensor-core logits + one routing launch.
//
// Selection math mirrors the torch routers (expf / log1pf / sqrtf / IEEE
// division, lowest index wins ties), so expert selection is identical
// whenever the logits are.

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CachingHostAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace moe_router {

namespace wmma = nvcuda::wmma;

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kPad = 8;  // smem row padding (bank conflicts)
constexpr int kStages = 3;
constexpr int kMaxE = 512;
constexpr int kMaxK = 32;
constexpr int kMaxGroups = 32;
constexpr int kMaxSplit = 16;
constexpr int kRows = 8;  // phase-2 rows reduced per smem pass (1 per warp)
constexpr int kFusedMaxTokens = 1024;
// One CTA reduces all partials when T * E is this small (decode b1).
constexpr int kSingleTileMaxElems = 512;

enum Scoring : int { kSoftmax = 0, kSigmoid = 1, kSqrtSoftplus = 2 };

// Persistent per-(device, stream) state: grid barrier words, the publish
// counter and a zeroed device copy of the active-expert flags. Every field
// is restored to zero by the kernel that used it.
struct alignas(16) RouteState {
  unsigned int bar_count;
  unsigned int bar_gen;
  unsigned int publish_count;
  unsigned int tile_count;
  int flags[kMaxE];
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
  bool* flags_host;  // [E] pinned host memory or nullptr
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

// One warp routes one token (expert e is owned by lane e % 32).
//
// A decode step runs this once per SM, cold: its latency is set by
// instruction-cache misses on straight-line code, not by ALU work. So the
// per-expert work runs in non-unrolled lane-strided loops over shared memory
// (scores / keys), and only the top-k scan keeps its VPL candidate keys in
// registers (that loop body is re-executed K times, i.e. cached).
//   lrow: the token's logits (smem or global); srow / krow: this warp's smem
//   scratch (E floats / E keys).
template <int VPL, int SC, typename Tl>
__device__ __forceinline__ void route_token(const Tl* lrow, float* srow,
                                            uint32_t* krow, int t,
                                            const EpilogueParams& p,
                                            int* flag_sink) {
  const int lane = threadIdx.x & 31;
  const int E = p.E;
  const int64_t row_off = static_cast<int64_t>(t) * E;

#pragma unroll 4
  for (int e = lane; e < E; e += 32) {
    float v = to_float(lrow[e]);
    if (p.round_logits) v = __bfloat162float(__float2bfloat16_rn(v));
    if (p.logits_out != nullptr)
      store_typed(p.logits_out, row_off + e, v, p.logits_dtype);
    float s;
    uint32_t key;
    if (SC == kSoftmax) {
      // softmax is strictly increasing: select on the (rounded) logits; the
      // probabilities of the selected experts are formed below.
      s = v;
      key = ord32(v);
    } else {
      if (SC == kSigmoid) {
        s = __frcp_rn(1.0f + expf(-v));  // == 1.0f / (...), IEEE
      } else {
        s = sqrtf(v > 20.f ? v : log1pf(expf(v)));
      }
      key = ord32(p.bias != nullptr ? s + p.bias[e] : s);
    }
    srow[e] = s;
    krow[e] = key;
  }
  __syncwarp();

  if (p.n_group > 1) {
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

  uint32_t key[VPL];  // 0 = not a candidate (no real value maps to 0)
#pragma unroll
  for (int i = 0; i < VPL; ++i)
    key[i] = lane + 32 * i < E ? krow[lane + 32 * i] : 0u;

  int my_id = 0;
  float my_w = 0.f;
#pragma unroll 1
  for (int k = 0; k < p.K; ++k) {
    uint32_t lv = 0;
    int li = 0x7FFFFFFF;
#pragma unroll
    for (int i = 0; i < VPL; ++i) {
      if (key[i] > lv) {
        lv = key[i];
        li = lane + 32 * i;
      }
    }
    uint32_t wmax;
    const int idx = warp_argmax(lv, li, &wmax);
    if ((idx & 31) == lane) {
#pragma unroll
      for (int i = 0; i < VPL; ++i)
        if (i == (idx >> 5)) key[i] = 0u;
      krow[idx] = 0u;  // marks "chosen" for the dense pass
    }
    if (lane == k) my_id = idx;
  }
  __syncwarp();
  if (lane < p.K) my_w = srow[my_id];
  const float top_logit = __shfl_sync(0xffffffffu, my_w, 0);

  // Weights. Softmax: p_k = exp(l_k - m) / sum(exp(l - m)); with renorm the
  // full-row sum cancels, so only the selected experts are exponentiated.
  float inv_all = 1.f;
  if (SC == kSoftmax) {
    if (!p.renorm) {
      float s = 0.f;
#pragma unroll 1
      for (int e = lane; e < E; e += 32) s += expf(srow[e] - top_logit);
      inv_all = 1.f / warp_sum(s);
    }
    my_w = lane < p.K ? expf(my_w - top_logit) * inv_all : 0.f;
  }
  float den = 1.f;
  if (p.renorm) den = warp_sum(lane < p.K ? my_w : 0.f) + p.eps;
  if (lane < p.K) {
    const float v = (p.renorm ? my_w / den : my_w) * p.scale;
    p.ids[static_cast<int64_t>(t) * p.K + lane] = my_id;
    p.tw[static_cast<int64_t>(t) * p.K + lane] = v;
    if (flag_sink != nullptr) flag_sink[my_id] = 1;
  }
  if (p.mask != nullptr) {
#pragma unroll 4
    for (int e = lane; e < E; e += 32) {
      const bool on = krow[e] == 0u;
      p.mask[row_off + e] = on;
      float v = 0.f;
      if (on) {
        const float raw =
            SC == kSoftmax ? expf(srow[e] - top_logit) * inv_all : srow[e];
        v = (p.renorm ? raw / den : raw) * p.scale;
      }
      store_typed(p.dw, row_off + e, v, p.dw_dtype);
    }
  }
  __syncwarp();
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
}

__device__ void grid_barrier(RouteState* st) {
  __syncthreads();
  if (threadIdx.x == 0) {
    volatile unsigned int* gen = &st->bar_gen;
    const unsigned int g = *gen;
    __threadfence();
    if (atomicAdd(&st->bar_count, 1u) == gridDim.x - 1) {
      st->bar_count = 0;
      __threadfence();
      atomicAdd(&st->bar_gen, 1u);
    } else {
      while (*gen == g) __nanosleep(20);
    }
    __threadfence();
  }
  __syncthreads();
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
//   BT=16 : 8 warps = 4 expert columns (16) x 2 halves of each H stage.
//   BT=64 : 8 warps = 4 token rows (16) x 2 expert columns (32).
//   (A 64x128 tile with 2x2 warp fragments measured slower at T=256: fewer
//   items -> fewer SMs streaming the gate weight.)
template <int BT>
struct Tile {
  static constexpr int kBE = 64;
  static constexpr int kHC = BT == 16 ? 128 : 64;  // H per stage
  static constexpr int kLd = kHC + kPad;
  static constexpr int kXElems = BT * kLd;
  static constexpr int kWElems = kBE * kLd;
};

template <typename Tin, int BT>
constexpr int decode_smem_bytes() {
  using TL = Tile<BT>;
  constexpr int op = kStages * (TL::kXElems + TL::kWElems) * sizeof(Tin);
  constexpr int acc = (BT == 16 ? 2 * 16 : BT) * TL::kBE * 4;
  constexpr int red = 3 * kRows * kMaxE * 4;  // logits, scores, keys
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
  Tin* ws = xs + kStages * TL::kXElems;

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
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < n_chunks) load_stage(s, h_lo + s * kHC);
    asm volatile("cp.async.commit_group;\n" ::);
  }
  for (int c = 0; c < n_chunks; ++c) {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(kStages - 2));
    __syncthreads();
    if (c + kStages - 1 < n_chunks)
      load_stage((c + kStages - 1) % kStages, h_lo + (c + kStages - 1) * kHC);
    asm volatile("cp.async.commit_group;\n" ::);
    const Tin* xa = xs + (c % kStages) * TL::kXElems;
    const Tin* wb = ws + (c % kStages) * TL::kWElems;
    if constexpr (BT == 16) {
      const int col = (warp & 3) * 16, half = warp >> 2;
#pragma unroll
      for (int kk = 0; kk < kHC / 2; kk += 16) {
        FragA fa;
        FragB fb;
        wmma::load_matrix_sync(fa, xa + half * (kHC / 2) + kk, kLd);
        wmma::load_matrix_sync(fb, wb + col * kLd + half * (kHC / 2) + kk, kLd);
        wmma::mma_sync(acc[0][0], fa, fb, acc[0][0]);
      }
    } else {
      const int row = (warp & 3) * 16, col = (warp >> 2) * 32;
#pragma unroll
      for (int kk = 0; kk < kHC; kk += 16) {
        FragA fa;
        FragB fb[2];
        wmma::load_matrix_sync(fa, xa + row * kLd + kk, kLd);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
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
    const int col = (warp & 3) * 16, half = warp >> 2;
    wmma::store_matrix_sync(cs + half * 16 * kBE + col, acc[0][0], kBE,
                            wmma::mem_row_major);
  } else {
    const int row = (warp & 3) * 16, col = (warp >> 2) * 32;
#pragma unroll
    for (int j = 0; j < 2; ++j)
      wmma::store_matrix_sync(cs + row * kBE + col + 16 * j, acc[0][j], kBE,
                              wmma::mem_row_major);
  }
  __syncthreads();
  for (int i = tid; i < BT * kBE; i += kThreads) {
    const int r = i / kBE, ce = i % kBE;
    const int t = t0 + r, e = e0 + ce;
    if (t < T && e < E) {
      float v = cs[i];
      if constexpr (BT == 16) v += cs[16 * kBE + i];
      a.partial[(static_cast<int64_t>(split) * T + t) * E + e] = v;
    }
  }
  __syncthreads();  // smem is reused by the next item
}

// Phase 2 for tokens [t_begin, t_end): reduce the split partials in a fixed
// order (deterministic) into smem, one warp routes each token.
template <int VPL, int SC>
__device__ void reduce_and_route(const DecodeArgs& a, const EpilogueParams& p,
                                 int t_begin, int t_end, unsigned char* smem) {
  const int T = p.T, E = p.E;
  float* lt = reinterpret_cast<float*>(smem);  // [kRows][E] logits
  float* st = lt + kRows * E;                  // [kRows][E] scores
  uint32_t* kt = reinterpret_cast<uint32_t*>(st + kRows * E);  // keys
  __shared__ int sflags[kMaxE];
  const bool want_flags = p.flags_host != nullptr;
  if (want_flags)
    for (int e = threadIdx.x; e < E; e += kThreads) sflags[e] = 0;
  const int64_t plane = static_cast<int64_t>(T) * E;
  const int n_split = a.n_split;
  for (int t0 = t_begin; t0 < t_end; t0 += kRows) {
    const int rows = min(kRows, t_end - t0);
    for (int i = threadIdx.x; i < rows * E; i += kThreads) {
      const float* src = a.partial + static_cast<int64_t>(t0) * E + i;
      float v[kMaxSplit];
#pragma unroll
      for (int s = 0; s < kMaxSplit; ++s)
        v[s] = s < n_split ? __ldcg(src + s * plane) : 0.f;
      float acc = 0.f;
#pragma unroll
      for (int s = 0; s < kMaxSplit; ++s) acc += v[s];
      lt[i] = acc;
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5;
    if (warp < rows)
      route_token<VPL, SC>(lt + warp * E, st + warp * E, kt + warp * E,
                           t0 + warp, p, want_flags ? sflags : nullptr);
    __syncthreads();
  }
  // Host pre-zeroed the pinned flags; set the experts this CTA routed to.
  if (want_flags && t_begin < t_end)
    for (int e = threadIdx.x; e < E; e += kThreads)
      if (sflags[e]) p.flags_host[e] = true;
}

// T <= BT (one token tile): no grid barrier -- the last CTA to finish its
// phase-1 items routes every token. Larger T: cooperative launch, grid
// barrier, phase 2 spread over all CTAs.
template <typename Tin, int BT, int VPL, int SC, bool kSingleTile>
__global__ void __launch_bounds__(kThreads)
    fused_route_kernel(DecodeArgs a, EpilogueParams p) {
  extern __shared__ __align__(128) unsigned char smem[];
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
    reduce_and_route<VPL, SC>(a, p, 0, p.T, smem);
    if (threadIdx.x == 0) p.state->tile_count = 0;
  } else {
    grid_barrier(p.state);
    const int tpc = (p.T + gridDim.x - 1) / gridDim.x;
    const int t_begin = blockIdx.x * tpc;
    reduce_and_route<VPL, SC>(a, p, t_begin, min(p.T, t_begin + tpc), smem);
  }
}

// Large-T routing over precomputed logits: one warp per token.
template <typename Tl, int VPL, int SC>
__global__ void __launch_bounds__(256)
    route_logits_kernel(const Tl* __restrict__ logits, EpilogueParams p) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int warp = threadIdx.x >> 5;
  const int t = blockIdx.x * (blockDim.x / 32) + warp;
  float* st = reinterpret_cast<float*>(smem) + warp * 2 * p.E;
  if (t < p.T)
    route_token<VPL, SC>(logits + static_cast<int64_t>(t) * p.E, st,
                         reinterpret_cast<uint32_t*>(st + p.E), t, p,
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
  const size_t smem = static_cast<size_t>(warps) * 2 * p.E * sizeof(float);
  switch (logits.scalar_type()) {
    case at::kFloat:
      route_logits_kernel<float, VPL, SC>
          <<<grid, warps * 32, smem, stream>>>(logits.data_ptr<float>(), p);
      break;
    case at::kBFloat16:
      route_logits_kernel<__nv_bfloat16, VPL, SC>
          <<<grid, warps * 32, smem, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()), p);
      break;
    case at::kHalf:
      route_logits_kernel<__half, VPL, SC><<<grid, warps * 32, smem, stream>>>(
          reinterpret_cast<const __half*>(logits.data_ptr()), p);
      break;
    default:
      TORCH_CHECK(false, "fused_route: unsupported logits dtype");
  }
}

template <typename Tin, int BT, int VPL, int SC, bool kSingleTile>
void launch_fused(const at::Tensor& x, const at::Tensor& w, int H,
                  EpilogueParams p, cudaStream_t stream, int device) {
  auto kernel = fused_route_kernel<Tin, BT, VPL, SC, kSingleTile>;
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
  int split = std::max(1, std::min({max_split, kMaxSplit, sms / (n_t * n_e)}));
  int h_per_split = ((H + split - 1) / split + kHC - 1) / kHC * kHC;
  split = (H + h_per_split - 1) / h_per_split;
  const int n_items = n_t * n_e * split;
  const int grid = std::min(n_items, cap);

  auto partial = at::empty({static_cast<int64_t>(split) * T * E},
                           x.options().dtype(at::kFloat));
  DecodeArgs a{x.data_ptr(),
               x.stride(0),
               w.data_ptr(),
               w.stride(0),
               H,
               n_t,
               n_e,
               split,
               h_per_split,
               partial.data_ptr<float>()};
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
    if (T * p.E <= kSingleTileMaxElems)
      launch_fused<__nv_bfloat16, 16, VPL, SC, true>(x, w, H, p, stream,
                                                     device);
    else if (T <= 16)
      launch_fused<__nv_bfloat16, 16, VPL, SC, false>(x, w, H, p, stream,
                                                      device);
    else if (T <= 64)
      launch_fused<__nv_bfloat16, 16, VPL, SC, false>(x, w, H, p, stream,
                                                      device);
    else
      launch_fused<__nv_bfloat16, 64, VPL, SC, false>(x, w, H, p, stream,
                                                      device);
  } else {
    if (T * p.E <= kSingleTileMaxElems)
      launch_fused<__half, 16, VPL, SC, true>(x, w, H, p, stream, device);
    else if (T <= 16)
      launch_fused<__half, 16, VPL, SC, false>(x, w, H, p, stream, device);
    else if (T <= 64)
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

// Returns [topk_ids, topk_weights, router_mask, routing_weights, logits,
// active_flags] (unrequested outputs are None), or an empty list when the
// shape / dtype / alignment is outside what this kernel supports.
std::vector<at::Tensor> fused_route(
    const at::Tensor& x_in, const at::Tensor& w_in, int64_t top_k,
    int64_t scoring, const c10::optional<at::Tensor>& bias_in, int64_t n_group,
    int64_t topk_group, bool renormalize, double scaling, double eps,
    bool round_logits, at::ScalarType weights_dtype,
    at::ScalarType logits_dtype, bool return_logits, bool dense,
    bool active_flags, int64_t gemm_min_tokens) {
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
  if (!supported) return {};
  const at::cuda::CUDAGuard guard(x.device());

  const auto opts = x.options();
  auto ids = at::empty({T, top_k}, opts.dtype(at::kInt));
  auto tw = at::empty({T, top_k}, opts.dtype(at::kFloat));
  at::Tensor mask, dw, logits_out, bias, flags;
  if (dense) {
    mask = at::empty({T, E}, opts.dtype(at::kBool));
    dw = at::empty({T, E}, opts.dtype(weights_dtype));
  }
  if (return_logits) logits_out = at::empty({T, E}, opts.dtype(logits_dtype));
  if (bias_in.has_value() && bias_in->defined()) {
    bias = bias_in->to(x.device(), at::kFloat).contiguous();
    TORCH_CHECK(bias.numel() == E, "fused_route: bias must have E elements");
  }
  const int device = x.get_device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
  if (active_flags) {
    flags = at::empty({E},
                      at::TensorOptions().dtype(at::kBool).pinned_memory(true));
    // The kernel writes this pinned block: keep it from being recycled
    // until the stream has passed this point.
    at::getHostAllocator(at::kCUDA)->record_event(
        flags.data_ptr(), flags.storage().data_ptr().get_context(),
        at::cuda::getCurrentCUDAStream(device).unwrap());
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
  p.ids = ids.data_ptr<int32_t>();
  p.tw = tw.data_ptr<float>();
  p.mask = dense ? mask.data_ptr<bool>() : nullptr;
  p.dw = dense ? dw.data_ptr() : nullptr;
  p.dw_dtype = dense ? dtype_code(weights_dtype) : 0;
  p.logits_out = return_logits ? logits_out.data_ptr() : nullptr;
  p.logits_dtype = return_logits ? dtype_code(logits_dtype) : 0;
  p.state = state_for(device, stream);
  p.flags_host = active_flags ? flags.data_ptr<bool>() : nullptr;

  const int sc = static_cast<int>(scoring);
  if (T > std::min<int64_t>(gemm_min_tokens - 1, kFusedMaxTokens)) {
    // Tensor-core logits (cuBLAS); fp32 out unless the router rounds.
    at::Tensor logits =
        round_logits ? at::mm(x, w.t()) : at::mm(x, w.t(), at::kFloat);
    dispatch_vpl_sc<LogitsL>(static_cast<int>(E), sc, logits, p, stream);
  } else {
    dispatch_vpl_sc<FusedL>(static_cast<int>(E), sc, x, w, static_cast<int>(H),
                            p, stream, device);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {ids, tw, mask, dw, logits_out, flags};
}

}  // namespace moe_router

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_route", &moe_router::fused_route,
        "Fused MoE routing: logits + scoring + (grouped) top-k + renorm",
        py::arg("hidden_states"), py::arg("weight"), py::arg("top_k"),
        py::arg("scoring"), py::arg("bias"), py::arg("n_group"),
        py::arg("topk_group"), py::arg("renormalize"), py::arg("scaling"),
        py::arg("eps"), py::arg("round_logits"), py::arg("weights_dtype"),
        py::arg("logits_dtype"), py::arg("return_logits"), py::arg("dense"),
        py::arg("active_flags"), py::arg("gemm_min_tokens"));
}
