// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// Grouped prefill MoE layer (see grouped_moe_prefill.h and moe_grouped_gemm.h).
//
// RTX PRO 6000 only (gate RTX_PRO_6000): tuned on the RTX PRO 6000 Blackwell
// (188 SMs, 128 MiB L2, 99 KiB shared memory per block). Other GPUs run
// grouped_moe_prefill_per_expert at the bottom of this file.
//
// Launch 1 (routing, cooperative): every block counts the routed (token,
// expert) pairs of a token range, the grid agrees on per-expert offsets
// (grid.sync), and every block writes its pairs into the sorted layout:
// token id and routing weight per row, rows of one expert contiguous. It also
// publishes the per-expert tile prefix of both GEMMs and zeroes `out` when
// asked. No host sync: the GEMM grids are persistent and read the counts on
// the device.
// Launch 2: grouped gate/up dual GEMM -> act [rows, I] (bf16).
// Launch 3: grouped down GEMM, out[t] += w · bf16(act · Wdᵀ) (fp32 reductions);
// for outputs far larger than L2 (combine mode) it writes y[r] = bf16(act ·
// Wdᵀ) per sorted row instead, and the same launch sums every token's rows in
// a fixed order (moe_sum), interleaved with the next pass's down tiles.

#include "kernel/grouped_moe_prefill.h"
#include "kernel/fused_moe_mlp.h"
#include "kernel/hw_profile.h"
#include "kernel/moe_grouped_gemm.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <mutex>
#include <tuple>
#include <utility>

namespace cg = cooperative_groups;

namespace {

using Elem = cutlass::bfloat16_t;
template <int M, int N, int K>
using Shape = cutlass::gemm::GemmShape<M, N, K>;
using moe_gemm::GroupedExpert;
using moe_gemm::GroupedExpertGemm;
using moe_gemm::GroupedRouting;

// gate/up (dual, gathered A) and down (scatter-add epilogue) configs. The
// mixed ones run full tiles of one height and finish every expert with tiles
// of half that height: an expert with 288 rows pads to 320 instead of 384
// (128 + 64). Configs 5 (gate/up) and 7 (down) finish with 16-row tiles
// (direct fragment stores, no CUTLASS epilogue): 144 rows = 128 + 16, no
// padding. Their small-tile rates keep them to remainders of <= 32 rows.
template <int M, int N, int K, int WM, int WN, int S, bool Dual>
using G = GroupedExpertGemm<Elem, Shape<M, N, K>, Shape<WM, WN, K>, S, Dual>;
using moe_gemm::GroupedTiledGemm;
using DualConfigs = std::tuple<
    GroupedTiledGemm<G<128, 128, 32, 64, 32, 3, true>>,  // 0: 8 warps
    GroupedTiledGemm<G<128, 64, 32, 64, 32, 3, true>>,   // 1: 4 warps
    GroupedTiledGemm<G<64, 128, 32, 32, 64, 3, true>>,   // 2: 4 warps
    GroupedTiledGemm<G<128, 128, 32, 64, 32, 3, true>,
                     G<64, 128, 32, 32, 32, 3, true>>,  // 3: 8 warps, mixed
    GroupedTiledGemm<G<64, 128, 32, 32, 64, 3, true>,
                     G<32, 128, 32, 32, 32, 3, true>>,  // 4: 4 warps, mixed
    GroupedTiledGemm<G<128, 64, 32, 64, 32, 3, true>,
                     G<16, 64, 64, 16, 16, 3, true>>>;  // 5: 4 warps, mixed
using DownConfigs = std::tuple<
    GroupedTiledGemm<G<128, 256, 32, 64, 64, 3, false>>,  // 0: 8 warps
    GroupedTiledGemm<G<128, 128, 32, 64, 64, 3, false>>,  // 1: 4 warps
    GroupedTiledGemm<G<64, 128, 32, 32, 64, 4, false>>,   // 2: 4 warps
    GroupedTiledGemm<G<128, 256, 32, 64, 64, 3, false>,
                     G<64, 256, 32, 32, 64, 3, false>>,  // 3: 8 warps, mixed
    GroupedTiledGemm<G<128, 128, 32, 64, 64, 3, false>,
                     G<64, 128, 32, 32, 64, 4, false>>,  // 4: 4 warps, mixed
    GroupedTiledGemm<G<64, 128, 32, 32, 64, 4, false>,
                     G<32, 128, 32, 32, 32, 4, false>>,  // 5: 4 warps, mixed
    GroupedTiledGemm<G<64, 128, 64, 32, 64, 3, false>,
                     G<32, 128, 64, 32, 32, 3, false>>,  // 6: 4 warps, mixed
    GroupedTiledGemm<G<128, 128, 32, 64, 64, 3, false>,
                     G<16, 128, 64, 16, 32, 3, false>>>;  // 7: 4 warps, mixed
constexpr int kNumDual = std::tuple_size_v<DualConfigs>;
constexpr int kNumDown = std::tuple_size_v<DownConfigs>;

template <typename Configs>
struct Heights;
template <typename... Gs>
struct Heights<std::tuple<Gs...>> {
  // nvcc drops the bound of a dependent constexpr array inside a generic
  // lambda (the type arrives as `const int[]`), so std::size cannot deduce
  // it. The pack length is the bound.
  static constexpr int count = int(sizeof...(Gs));
  static constexpr int big[] = {Gs::kBigM...};
  static constexpr int small[] = {(Gs::kMixed ? Gs::kSmallM : 0)...};
};

template <typename Configs, typename F, size_t... Is>
void visit(int cfg, F&& f, std::index_sequence<Is...>) {
  bool done =
      ((cfg == int(Is)
            ? (f.template operator()<std::tuple_element_t<Is, Configs>>(), true)
            : false) ||
       ...);
  TORCH_CHECK(done, "grouped_moe_prefill: invalid tile config ", cfg);
}
template <typename Configs, typename F>
void visit(int cfg, F&& f) {
  visit<Configs>(cfg, std::forward<F>(f),
                 std::make_index_sequence<std::tuple_size_v<Configs>>{});
}

// ---------------------------------------------------------------------------
// Workspace: one growing device buffer per (device, stream), stream-ordered.
// ---------------------------------------------------------------------------

std::mutex g_mutex;
std::map<std::pair<int, cudaStream_t>, std::pair<char*, size_t>> g_ws;

char* workspace(int device, cudaStream_t stream, size_t bytes) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto& ws = g_ws[{device, stream}];
  if (ws.second < bytes) {
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &capture);
    TORCH_CHECK(capture == cudaStreamCaptureStatusNone,
                "grouped_moe_prefill: run once outside graph capture first "
                "to size the workspace");
    if (ws.first) cudaFreeAsync(ws.first, stream);
    ws = {nullptr, 0};
    size_t grow = std::max(bytes, size_t(64) << 20);
    TORCH_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&ws.first), grow,
                                stream) == cudaSuccess,
                "grouped_moe_prefill: workspace allocation failed");
    ws.second = grow;
  }
  return ws.first;
}

template <typename Gemm>
int occupancy(int device) {
  static std::array<int, 64> occ{};
  static std::mutex mutex;
  if (occ[device] > 0) return occ[device];
  std::lock_guard<std::mutex> lock(mutex);
  if (occ[device] > 0) return occ[device];
  auto kernel = moe_gemm::grouped_expert_gemm_pro6000_kernel<Gemm>;
  if (Gemm::kSmemBytes >= (48 << 10)) {
    TORCH_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    Gemm::kSmemBytes) == cudaSuccess,
                "grouped_moe_prefill: cannot reserve shared memory");
  }
  int blocks = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, Gemm::kThreads,
                                                Gemm::kSmemBytes);
  occ[device] = std::max(blocks, 1);
  return occ[device];
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

// Down passes (N panel groups) at most; see panels[1] below.
constexpr int kMaxPasses = 8;

struct RouteParams {
  uint8_t const* mask;
  int64_t mask_ld, mask_cs;
  void const* rw;
  int64_t rw_ld, rw_cs;
  int rw_dtype;  // 0 fp32, 1 bf16, 2 fp16
  GroupedExpert const* experts;
  int n, T, E, H;
  int chunk;                     // tokens per block
  int* block_count;              // [blocks, n]
  int* block_base;               // [blocks, n]
  int* total;                    // [n]
  int* offsets;                  // [n + 1]
  int* tokens;                   // [rows]
  float* weights;                // [rows]
  int* tile_start[2];            // [n + 1] per GEMM
  moe_gemm::TileSplit split[2];  // M tiles of an expert per GEMM
  int tiles_n[2];                // N panels per pass
  float* out;
  int64_t ldo;
  int zero_out;
  // Combine mode: sorted rows of every token, ascending, count in tcount.
  int* next_tile;  // [2 + kMaxPasses] tile counters of the GEMM launches,
                   // then finished down tiles per pass (combine mode)
  int* tcount;     // [T] or nullptr
  int* rowmap;     // [T, kcap]
  int kcap;
};

__device__ __forceinline__ float route_weight(RouteParams const& p, int t,
                                              int e) {
  int64_t off = t * p.rw_ld + e * p.rw_cs;
  if (p.rw_dtype == 0) return static_cast<float const*>(p.rw)[off];
  if (p.rw_dtype == 1)
    return __bfloat162float(static_cast<__nv_bfloat16 const*>(p.rw)[off]);
  return __half2float(static_cast<__half const*>(p.rw)[off]);
}

// Exclusive prefix over n values by one warp; returns the total in lane 31.
template <typename F, typename G>
__device__ void warp_scan(int n, F&& value, G&& store) {
  int const lane = threadIdx.x % 32;
  int const per = (n + 31) / 32;
  int sum = 0;
  for (int i = 0; i < per; ++i) {
    int e = lane * per + i;
    if (e < n) sum += value(e);
  }
  int incl = sum;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    int v = __shfl_up_sync(0xffffffff, incl, d);
    if (lane >= d) incl += v;
  }
  int run = incl - sum;
  for (int i = 0; i < per; ++i) {
    int e = lane * per + i;
    if (e < n) {
      store(e, run);
      run += value(e);
    }
  }
  if (lane == 31) store(n, incl);
}

// Calls f(t, e) for every set mask entry of tokens [t0, t1). Contiguous
// masks are read 16 bytes at a time.
template <typename F>
__device__ void for_each_routed(RouteParams const& p, int t0, int t1, F&& f) {
  if (t1 <= t0) return;
  if (p.mask_cs == 1 && p.E % 16 == 0 && p.mask_ld % 16 == 0 &&
      reinterpret_cast<uintptr_t>(p.mask) % 16 == 0) {
    int const per_row = p.E / 16;
    int const vecs = (t1 - t0) * per_row;
    for (int v = threadIdx.x; v < vecs; v += blockDim.x) {
      int const t = t0 + v / per_row;
      int const e0 = (v % per_row) * 16;
      uint4 const bits =
          *reinterpret_cast<uint4 const*>(p.mask + int64_t(t) * p.mask_ld + e0);
      uint32_t const w[4] = {bits.x, bits.y, bits.z, bits.w};
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        uint32_t x = w[k];
        while (x) {
          int const byte = (__ffs(x) - 1) / 8;
          f(t, e0 + k * 4 + byte);
          x &= ~(0xffu << (8 * byte));
        }
      }
    }
    return;
  }
  int const pairs = (t1 - t0) * p.E;
  for (int i = threadIdx.x; i < pairs; i += blockDim.x) {
    int const t = t0 + i / p.E;
    int const e = i % p.E;
    if (p.mask[int64_t(t) * p.mask_ld + int64_t(e) * p.mask_cs]) f(t, e);
  }
}

__global__ void __launch_bounds__(256)
    grouped_route_pro6000_kernel(RouteParams p) {
  cg::grid_group grid = cg::this_grid();
  extern __shared__ int sm[];
  int* map = sm;          // [E] expert id -> table row
  int* cnt = map + p.E;   // [n]
  int* offs = cnt + p.n;  // [n + 1]
  int const tid = threadIdx.x;
  int const b = blockIdx.x;
  int const t0 = b * p.chunk;
  int const t1 = min(p.T, t0 + p.chunk);

  for (int e = tid; e < p.E; e += blockDim.x) map[e] = -1;
  __syncthreads();
  for (int j = tid; j < p.n; j += blockDim.x) {
    int64_t id = p.experts[j].expert_id;
    if (id >= 0 && id < p.E) map[id] = j;
    cnt[j] = 0;
  }
  __syncthreads();

  if (p.zero_out && t1 > t0) {
    int const vec = p.H / 4;
    for (int64_t i = tid; i < int64_t(t1 - t0) * vec; i += blockDim.x) {
      int t = t0 + int(i / vec);
      int c = int(i % vec) * 4;
      *reinterpret_cast<float4*>(p.out + t * p.ldo + c) =
          make_float4(0.f, 0.f, 0.f, 0.f);
    }
  }

  if (p.tcount)
    for (int t = t0 + tid; t < t1; t += blockDim.x) p.tcount[t] = 0;
  for_each_routed(p, t0, t1, [&](int t, int e) {
    int j = map[e];
    if (j >= 0) atomicAdd(&cnt[j], 1);
  });
  __syncthreads();
  for (int j = tid; j < p.n; j += blockDim.x)
    p.block_count[int64_t(b) * p.n + j] = cnt[j];
  grid.sync();

  // Per expert: exclusive scan over blocks.
  for (int j = b * blockDim.x + tid; j < p.n; j += gridDim.x * blockDim.x) {
    int s = 0;
    for (int bb = 0; bb < gridDim.x; ++bb) {
      p.block_base[int64_t(bb) * p.n + j] = s;
      s += p.block_count[int64_t(bb) * p.n + j];
    }
    p.total[j] = s;
  }
  grid.sync();

  if (tid < 32) {
    warp_scan(
        p.n, [&](int e) { return p.total[e]; },
        [&](int e, int v) { offs[e] = v; });
  }
  if (b == 0 && tid >= 32 && tid < 96) {
    int const g = (tid - 32) / 32;  // tile prefix of GEMM g
    warp_scan(
        p.n, [&](int e) { return p.split[g].tiles(p.total[e]) * p.tiles_n[g]; },
        [&](int e, int v) { p.tile_start[g][e] = v; });
  }
  __syncthreads();
  if (b == 0) {
    for (int j = tid; j <= p.n; j += blockDim.x) p.offsets[j] = offs[j];
    if (tid < 2 + kMaxPasses) p.next_tile[tid] = 0;
  }
  for (int j = tid; j < p.n; j += blockDim.x) cnt[j] = 0;
  __syncthreads();

  for_each_routed(p, t0, t1, [&](int t, int e) {
    int j = map[e];
    if (j < 0) return;
    int r = atomicAdd(&cnt[j], 1);
    int pos = offs[j] + p.block_base[int64_t(b) * p.n + j] + r;
    p.tokens[pos] = t;
    p.weights[pos] = route_weight(p, t, e);
    if (p.tcount) {
      int const slot = atomicAdd(&p.tcount[t], 1);
      if (slot < p.kcap) p.rowmap[int64_t(t) * p.kcap + slot] = pos;
    }
  });
  if (!p.tcount) return;
  // This block owns tokens [t0, t1): sort each token's rows so the combine
  // sums them in a fixed (expert) order.
  __syncthreads();
  for (int t = t0 + tid; t < t1; t += blockDim.x) {
    int* r = p.rowmap + int64_t(t) * p.kcap;
    int const c = min(p.tcount[t], p.kcap);
    for (int a = 1; a < c; ++a) {
      int const v = r[a];
      int b = a - 1;
      for (; b >= 0 && r[b] > v; --b) r[b + 1] = r[b];
      r[b + 1] = v;
    }
  }
}

// Output slice per down pass (see panels[1] below).
constexpr size_t kOutPassBytes = size_t(32) << 20;
// Combine mode: down passes (qwen3.5-35b 50K down + combine: 1 pass 5.02 ms,
// 2: 4.75, 4: 4.72, 8: 5.20) and tokens per combine item.
constexpr int kCombinePasses = 4;
constexpr int kCombineTokens = 32;
// Largest combine-mode row buffer (see `combine` below).
constexpr size_t kMaxCombineBytes = size_t(4) << 30;

size_t align_up(size_t x) { return (x + 255) & ~size_t(255); }

int sm_count(int device) {
  static std::array<int, 64> sms{};
  if (sms[device] == 0) {
    int n = 0;
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, device);
    sms[device] = n;
  }
  return sms[device];
}

}  // namespace

bool grouped_moe_prefill_supported(int64_t tokens, int64_t hidden,
                                   int64_t inter, int64_t experts) {
  return tokens > 0 && tokens < (int64_t(1) << 31) / 64 && hidden % 8 == 0 &&
         inter % 8 == 0 && experts > 0 && experts <= 4096;
}

namespace {

// RTX PRO 6000 path: routing + grouped GEMMs (see the top of this file).
void grouped_moe_prefill_pro6000(
    const torch::Tensor& hidden, const torch::Tensor& router_mask,
    const torch::Tensor& router_weight, const torch::Tensor& expert_table,
    int64_t inter, int64_t kcap, torch::Tensor& out, bool zero_out,
    int dual_cfg, int down_cfg, int combine_mode, cudaStream_t stream) {
  TORCH_CHECK(hidden.is_cuda() && hidden.scalar_type() == at::kBFloat16 &&
                  hidden.dim() == 2 && hidden.stride(1) == 1 &&
                  hidden.stride(0) % 8 == 0,
              "grouped_moe_prefill: hidden must be bf16 [T, H] with "
              "16-byte aligned rows");
  TORCH_CHECK(out.scalar_type() == at::kFloat && out.dim() == 2 &&
                  out.stride(1) == 1 && out.stride(0) % 4 == 0,
              "grouped_moe_prefill: out must be fp32 [T, H]");
  TORCH_CHECK(
      router_mask.dim() == 2 && (router_mask.scalar_type() == at::kBool ||
                                 router_mask.scalar_type() == at::kByte),
      "grouped_moe_prefill: router_mask must be bool/uint8 [T, E]");
  TORCH_CHECK(
      router_weight.dim() == 2 && router_weight.sizes() == router_mask.sizes(),
      "grouped_moe_prefill: router_weight must match router_mask");
  TORCH_CHECK(expert_table.scalar_type() == at::kLong &&
                  expert_table.dim() == 2 && expert_table.size(1) == 4 &&
                  expert_table.is_contiguous(),
              "grouped_moe_prefill: expert_table must be int64 [n, 4]");
  int const T = int(hidden.size(0));
  int const H = int(hidden.size(1));
  int const I = int(inter);
  int const E = int(router_mask.size(1));
  int const n = int(expert_table.size(0));
  TORCH_CHECK(router_mask.size(0) == T && out.size(0) == T && out.size(1) == H,
              "grouped_moe_prefill: token / hidden dims mismatch");
  TORCH_CHECK(grouped_moe_prefill_supported(T, H, I, n),
              "grouped_moe_prefill: unsupported shape");
  TORCH_CHECK(kcap >= 1, "grouped_moe_prefill: kcap must be >= 1");
  if (T == 0 || n == 0) return;

  int const device = hidden.get_device();
  c10::cuda::CUDAGuard guard(device);
  int const sms = sm_count(device);
  int const rw_dtype = router_weight.scalar_type() == at::kFloat      ? 0
                       : router_weight.scalar_type() == at::kBFloat16 ? 1
                       : router_weight.scalar_type() == at::kHalf     ? 2
                                                                      : -1;
  TORCH_CHECK(rw_dtype >= 0, "grouped_moe_prefill: router_weight dtype");

  // Tile config: rows per expert are known only on the device; T·kcap / n is
  // their mean when routing is balanced. Relative MMA rates of the tile
  // heights (sm_120 sweep) and the padding of every expert's last tiles
  // decide between them.
  double const mean_rows = double(T) * kcap / n;
  int const rows_e = std::max(1, int(std::lround(mean_rows)));
  struct Rate {
    double big, small;
  };
  static constexpr Rate kDualRate[] = {{1.85, 0},    {1.68, 0},
                                       {1.45, 0},    {1.85, 1.45},
                                       {1.45, 1.30}, {1.68, 0.55}};
  static constexpr Rate kDownRate[] = {{1.70, 0},    {1.64, 0},    {1.40, 0},
                                       {1.70, 1.40}, {1.64, 1.40}, {1.40, 1.05},
                                       {1.40, 1.05}, {1.64, 0.50}};
  static_assert(std::size(kDualRate) == kNumDual &&
                std::size(kDownRate) == kNumDown);
  // A config's split: small tiles finish an expert while they cost less than
  // one more big tile.
  auto split_of = [](Rate r, int big_m, int small_m) {
    if (small_m == 0) return moe_gemm::TileSplit{big_m, big_m, 0};
    return moe_gemm::TileSplit{
        big_m, small_m, int(std::floor(big_m / r.big / (small_m / r.small)))};
  };
  auto cost = [&](Rate r, moe_gemm::TileSplit sp) {
    int nb, ns;
    sp.split(rows_e, nb, ns);
    return nb * sp.big_m / r.big + (ns ? ns * sp.small_m / r.small : 0.0);
  };
  auto pick = [&]<typename Configs>(Rate const* rate, int& cfg) {
    using Hm = Heights<Configs>;
    if (cfg < 0) {
      double best = 0;
      for (int c = 0; c < Hm::count; ++c) {
        double t = cost(rate[c], split_of(rate[c], Hm::big[c], Hm::small[c]));
        if (cfg < 0 || t < best) {
          best = t;
          cfg = c;
        }
      }
    }
    TORCH_CHECK(cfg < Hm::count, "grouped_moe_prefill: invalid tile config ",
                cfg);
    return split_of(rate[cfg], Hm::big[cfg], Hm::small[cfg]);
  };
  moe_gemm::TileSplit const split[2] = {
    pick.template operator()<DualConfigs>(kDualRate, dual_cfg),
    pick.template operator()<DownConfigs>(kDownRate, down_cfg)
  };
  int const ms[2] = {split[0].max_small, split[1].max_small};
  int tiles_n[2], panels[2];
  visit<DualConfigs>(dual_cfg, [&]<typename Gm>() {
    tiles_n[0] = (I + Gm::ThreadblockShape::kN - 1) / Gm::ThreadblockShape::kN;
    panels[0] = tiles_n[0];
  });
  // Combine mode: when the fp32 output is too large for the
  // scatter-add to stay in L2 even in 4 passes, the down GEMM writes every
  // (token, expert) row in bf16 and then sums each token's rows (moe_sum).
  // The bf16 rows take T·kcap·H·2 bytes of workspace; above kMaxCombineBytes
  // the layer keeps the scatter-add.
  bool const combine =
      combine_mode == 1 ||
      (combine_mode < 0 && double(T) * H * 4 > 4.0 * double(kOutPassBytes) &&
       double(T) * kcap * H * 2 <= double(kMaxCombineBytes));
  visit<DownConfigs>(down_cfg, [&]<typename Gm>() {
    tiles_n[1] = (H + Gm::ThreadblockShape::kN - 1) / Gm::ThreadblockShape::kN;
    // The scatter-add output is [T, H] fp32 and every row receives kcap
    // reductions from different experts. When it does not fit in L2 those
    // reductions miss; sweeping it in P passes of N panels keeps one slice
    // (T x H / P x 4 B) resident at the cost of reading the activations P
    // times, which pays off for up to 4 passes.
    panels[1] = tiles_n[1];
    // Combine mode: passes let the combine of one pass overlap the down
    // tiles of the next (each pass re-reads the activations).
    if (combine) {
      int passes = kCombinePasses;
      while (passes > 1 && tiles_n[1] % passes != 0) --passes;
      panels[1] = tiles_n[1] / passes;
    }
    for (int passes = 2; passes <= 4 && !combine; ++passes) {
      if (tiles_n[1] % passes != 0) continue;
      if (double(T) * H * 4 <= double(kOutPassBytes)) break;
      if (double(T) * H * 4 / passes <= double(kOutPassBytes)) {
        panels[1] = tiles_n[1] / passes;
        break;
      }
    }
  });
  // Routing grid: <= one block per SM so a cooperative launch always fits.
  int const blocks = std::max(1, std::min(sms, (T + 63) / 64));
  int const chunk = (T + blocks - 1) / blocks;
  int64_t const rows = int64_t(T) * kcap;

  size_t off = 0;
  auto carve = [&](size_t bytes) {
    size_t at = off;
    off += align_up(bytes);
    return at;
  };
  size_t const o_bc = carve(sizeof(int) * size_t(blocks) * n);
  size_t const o_bb = carve(sizeof(int) * size_t(blocks) * n);
  size_t const o_tot = carve(sizeof(int) * n);
  size_t const o_off = carve(sizeof(int) * (n + 1));
  size_t const o_ts0 = carve(sizeof(int) * (n + 1));
  size_t const o_ts1 = carve(sizeof(int) * (n + 1));
  size_t const o_next = carve(sizeof(int) * (2 + kMaxPasses));
  size_t const o_tok = carve(sizeof(int) * rows);
  size_t const o_w = carve(sizeof(float) * rows);
  size_t const o_act = carve(sizeof(Elem) * rows * I);
  size_t const o_tc = combine ? carve(sizeof(int) * size_t(T)) : 0;
  size_t const o_rm = combine ? carve(sizeof(int) * rows) : 0;
  size_t const o_y = combine ? carve(sizeof(Elem) * rows * H) : 0;
  char* ws = workspace(device, stream, off);

  RouteParams rp;
  rp.mask = static_cast<uint8_t const*>(router_mask.data_ptr());
  rp.mask_ld = router_mask.stride(0);
  rp.mask_cs = router_mask.stride(1);
  rp.rw = router_weight.data_ptr();
  rp.rw_ld = router_weight.stride(0);
  rp.rw_cs = router_weight.stride(1);
  rp.rw_dtype = rw_dtype;
  rp.experts = reinterpret_cast<GroupedExpert const*>(expert_table.data_ptr());
  rp.n = n;
  rp.T = T;
  rp.E = E;
  rp.H = H;
  rp.chunk = chunk;
  rp.block_count = reinterpret_cast<int*>(ws + o_bc);
  rp.block_base = reinterpret_cast<int*>(ws + o_bb);
  rp.total = reinterpret_cast<int*>(ws + o_tot);
  rp.offsets = reinterpret_cast<int*>(ws + o_off);
  rp.tokens = reinterpret_cast<int*>(ws + o_tok);
  rp.weights = reinterpret_cast<float*>(ws + o_w);
  rp.tile_start[0] = reinterpret_cast<int*>(ws + o_ts0);
  rp.tile_start[1] = reinterpret_cast<int*>(ws + o_ts1);
  rp.split[0] = split[0];
  rp.split[1] = split[1];
  rp.tiles_n[0] = panels[0];
  rp.tiles_n[1] = panels[1];
  rp.out = out.data_ptr<float>();
  rp.ldo = out.stride(0);
  rp.zero_out = zero_out && !combine ? 1 : 0;
  rp.next_tile = reinterpret_cast<int*>(ws + o_next);
  rp.tcount = combine ? reinterpret_cast<int*>(ws + o_tc) : nullptr;
  rp.rowmap = combine ? reinterpret_cast<int*>(ws + o_rm) : nullptr;
  rp.kcap = int(kcap);
  size_t const route_smem = sizeof(int) * (E + 2 * size_t(n) + 1);
  {
    void* args[] = {&rp};
    cudaError_t err = cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(grouped_route_pro6000_kernel), dim3(blocks),
        dim3(256), args, route_smem, stream);
    TORCH_CHECK(err == cudaSuccess, "grouped_moe_prefill routing launch: ",
                cudaGetErrorString(err));
  }

  auto* act = reinterpret_cast<Elem*>(ws + o_act);
  auto* x = reinterpret_cast<Elem const*>(hidden.data_ptr());
  auto const* experts =
      reinterpret_cast<GroupedExpert const*>(expert_table.data_ptr());
  GroupedRouting routing{rp.offsets, rp.tokens, rp.weights, nullptr};

  visit<DualConfigs>(dual_cfg, [&]<typename G>() {
    routing.tile_start = rp.tile_start[0];
    auto p = G::make_params(ms[0], n, I, H, int(hidden.stride(0)), experts,
                            routing, x, act, nullptr, 0);
    p.next_tile = rp.next_tile;
    int const grid = sms * occupancy<G>(device);
    moe_gemm::grouped_expert_gemm_pro6000_kernel<G>
        <<<grid, G::kThreads, G::kSmemBytes, stream>>>(p);
  });
  visit<DownConfigs>(down_cfg, [&]<typename G>() {
    routing.tile_start = rp.tile_start[1];
    auto p = G::make_params(ms[1], n, H, I, I, experts, routing, act, nullptr,
                            out.data_ptr<float>(), out.stride(0));
    p.big.panels = panels[1];
    p.small.panels = panels[1];
    p.next_tile = rp.next_tile + 1;
    if (combine) {
      auto* y = reinterpret_cast<Elem*>(ws + o_y);
      p.big.y = p.small.y = y;
      p.combine = {y,
                   rp.weights,
                   rp.tcount,
                   rp.rowmap,
                   out.data_ptr<float>(),
                   out.stride(0),
                   T,
                   H,
                   int(kcap),
                   zero_out ? 1 : 0,
                   rp.next_tile + 2,
                   kCombineTokens};
    }
    int const grid = sms * occupancy<G>(device);
    moe_gemm::grouped_expert_gemm_pro6000_kernel<G>
        <<<grid, G::kThreads, G::kSmemBytes, stream>>>(p);
  });
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "grouped_moe_prefill launch failed: ", cudaGetErrorString(err));
}

}  // namespace

// Previous execution, for every other GPU: one expert at a time through
// fused_moe_ffn_into (the generic 3-GEMM path there) and a weighted fp32
// index_add per expert. Reads the routing on the host, like the per-expert
// dispatch it replaces.
void grouped_moe_prefill_per_expert(const torch::Tensor& hidden,
                                    const torch::Tensor& router_mask,
                                    const torch::Tensor& router_weight,
                                    const torch::Tensor& expert_table,
                                    int64_t inter, torch::Tensor& out,
                                    bool zero_out, cudaStream_t stream) {
  int const device = hidden.get_device();
  c10::cuda::CUDAGuard guard(device);
  c10::cuda::CUDAStreamGuard stream_guard(
      c10::cuda::getStreamFromExternal(stream, device));
  int64_t const H = hidden.size(1);
  int64_t const E = router_mask.size(1);
  if (zero_out) out.zero_();
  auto const table = expert_table.cpu();
  auto const* rows = table.data_ptr<int64_t>();
  auto const wopts = hidden.options().dtype(at::kBFloat16);
  for (int64_t j = 0; j < table.size(0); ++j) {
    int64_t const e = rows[4 * j + 3];
    if (e < 0 || e >= E) continue;
    auto tok = router_mask.select(1, e).nonzero().squeeze(1);
    int64_t const m = tok.numel();
    if (m == 0) continue;
    auto x = hidden.index_select(0, tok).contiguous();
    auto g = torch::from_blob(reinterpret_cast<void*>(rows[4 * j]), {inter, H},
                              wopts);
    auto u = torch::from_blob(reinterpret_cast<void*>(rows[4 * j + 1]),
                              {inter, H}, wopts);
    auto d = torch::from_blob(reinterpret_cast<void*>(rows[4 * j + 2]),
                              {H, inter}, wopts);
    auto gate_buf = torch::empty({m, inter}, wopts);
    auto fused_buf = torch::empty({m, inter}, wopts);
    auto y = torch::empty({m, H}, wopts);
    fused_moe_ffn_into(x, g, u, d, gate_buf, fused_buf, y, stream);
    auto w = router_weight.select(1, e).index_select(0, tok).to(at::kFloat);
    out.index_add_(0, tok, y.to(at::kFloat) * w.unsqueeze(1));
  }
}

bool grouped_moe_prefill_pro6000_enabled(int device) {
  return moe_infinity::hw::Supports(moe_infinity::hw::Gate::kRtxPro6000,
                                    device);
}

void grouped_moe_prefill_cfg(const torch::Tensor& hidden,
                             const torch::Tensor& router_mask,
                             const torch::Tensor& router_weight,
                             const torch::Tensor& expert_table, int64_t inter,
                             int64_t kcap, torch::Tensor& out, bool zero_out,
                             int dual_cfg, int down_cfg, int combine_mode,
                             cudaStream_t stream) {
  if (grouped_moe_prefill_pro6000_enabled(hidden.get_device())) {
    grouped_moe_prefill_pro6000(hidden, router_mask, router_weight,
                                expert_table, inter, kcap, out, zero_out,
                                dual_cfg, down_cfg, combine_mode, stream);
    return;
  }
  TORCH_CHECK(dual_cfg < 0 && down_cfg < 0 && combine_mode < 0,
              "grouped_moe_prefill: tile configs exist only on the RTX PRO "
              "6000 path");
  TORCH_CHECK(hidden.is_cuda() && out.scalar_type() == at::kFloat &&
                  expert_table.scalar_type() == at::kLong &&
                  expert_table.dim() == 2 && expert_table.size(1) == 4,
              "grouped_moe_prefill: bad arguments");
  grouped_moe_prefill_per_expert(hidden, router_mask, router_weight,
                                 expert_table, inter, out, zero_out, stream);
}

void grouped_moe_prefill(const torch::Tensor& hidden,
                         const torch::Tensor& router_mask,
                         const torch::Tensor& router_weight,
                         const torch::Tensor& expert_table, int64_t inter,
                         int64_t kcap, torch::Tensor& out, bool zero_out,
                         cudaStream_t stream) {
  grouped_moe_prefill_cfg(hidden, router_mask, router_weight, expert_table,
                          inter, kcap, out, zero_out, -1, -1, -1, stream);
}
