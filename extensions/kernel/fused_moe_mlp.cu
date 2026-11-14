// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// Fused MoE expert FFN with CUTLASS: two launches per expert.
//
// RTX_PRO_6000 (Server 600W + Max-Q 300W unified) (gate RTX_PRO_6000,
// docs/hardware-gated-kernels.md): the tile/split table, the cost model, the
// L2 persisting window and the L2-fit rasterisation below were tuned on the
// RTX PRO 6000 Blackwell (188 SMs, 128 MiB L2, 99 KiB shared memory per
// block). fused_moe_ffn_into checks the gate once per device and runs
// fused_moe_ffn_into_generic (fused_moe_mlp_generic.cu, the previous 3-GEMM
// path) everywhere else. All three gated activations (SiLU, clamped SiLU, and
// SwiGLU-OAI) are fused directly into the dual GEMM register epilogue.
//
//   1. dual GEMM   fused_buf = act(X · gate_projᵀ, X · up_projᵀ)         [M, I]
//   2. GEMM        output    = fused_buf · down_projᵀ                    [M, H]
//
// Each launch picks a threadblock tile / stage count / split-K factor per
// (M, N, K) from a small table (see select_config), sized for sm_120's
// 99 KB-per-block shared memory and 188-SM wave quantisation, and valid on
// every compiled architecture (sm_80 / sm_86 / sm_90 / sm_120).

#include "kernel/fused_moe_mlp.h"
#include "kernel/fused_moe_mlp_tuned.h"
#include "kernel/hw_profile.h"
#include "kernel/moe_expert_gemm.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <mutex>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using Elem = cutlass::bfloat16_t;
template <int M, int N, int K>
using Shape = cutlass::gemm::GemmShape<M, N, K>;

// ---------------------------------------------------------------------------
// Tile configurations. Shared memory per CTA = stages × (TBm + TBn·(1+dual))
// × TBk × 2 B; all configs fit 99 KB. Dual configs keep warp tiles at 64×32
// or smaller: two accumulators per warp double the register footprint.
// ---------------------------------------------------------------------------

template <int TM, int TN, int TK, int WM, int WN, int WK, int Stages, bool Dual>
using Cfg = moe_gemm::ExpertGemm<Elem, Shape<TM, TN, TK>, Shape<WM, WN, WK>,
                                 Stages, Dual, /*SplitK=*/true>;

// Each config carries cost-model constants fitted on RTX PRO 6000 Blackwell
// (sm_120, 188 SMs, L2 flushed) for the selection model below. On other GPUs
// the SM count and occupancy are queried at run time; only the ratios between
// configs matter for large problems.
struct ConfigCost {
  double tflops_per_sm;  // with every resident CTA slot of the SM busy
  double round_us;       // per round of CTAs on an SM (prologue + epilogue)
  double gamma;          // SM throughput ∝ (resident CTAs / occupancy)^gamma
};

// gate + up (dual)
using DualConfigs =
    std::tuple<Cfg<128, 128, 32, 64, 32, 32, 3, true>,  // 0: 8 warps
               Cfg<128, 64, 32, 64, 32, 32, 3, true>,   // 1: 4 warps
               Cfg<128, 64, 64, 64, 32, 64, 3, true>,   // 2: 4 warps
               Cfg<64, 64, 64, 32, 32, 64, 4, true>,    // 3: 4 warps
               Cfg<64, 64, 32, 32, 32, 32, 4, true>,    // 4: 4 warps
               Cfg<32, 64, 64, 16, 32, 64, 3, true>>;   // 5: 4 warps
constexpr ConfigCost kDualCost[] = {{1.85, 6.59, 1.50}, {1.68, 2.93, 0.00},
                                    {1.66, 1.83, 0.47}, {1.37, 1.02, 0.87},
                                    {1.36, 1.44, 0.10}, {0.81, 0.47, 0.22}};

// down
using DownConfigs =
    std::tuple<Cfg<128, 256, 32, 64, 64, 32, 3, false>,  // 0: 8 warps
               Cfg<128, 128, 32, 64, 64, 32, 4, false>,  // 1: 4 warps
               Cfg<128, 128, 32, 64, 64, 32, 3, false>,  // 2: 4 warps
               Cfg<128, 64, 64, 64, 32, 64, 4, false>,   // 3: 4 warps
               Cfg<128, 128, 64, 64, 64, 64, 3, false>,  // 4: 4 warps
               Cfg<64, 64, 32, 32, 32, 32, 4, false>,    // 5: 4 warps
               Cfg<32, 64, 64, 16, 32, 64, 4, false>>;   // 6: 4 warps
constexpr ConfigCost kDownCost[] = {{1.70, 2.08, 1.00}, {1.52, 0.16, 1.30},
                                    {1.64, 0.78, 0.22}, {1.34, 0.29, 1.22},
                                    {1.55, 0.38, 0.72}, {1.02, 0.00, 0.20},
                                    {0.71, 0.01, 0.73}};

constexpr int kNumDual = std::tuple_size_v<DualConfigs>;
constexpr int kNumDown = std::tuple_size_v<DownConfigs>;

// ---------------------------------------------------------------------------
// Per-(device, stream) split-K workspace. Counters start at zero and are
// reset by the kernels themselves; the buffers only ever grow, through the
// stream-ordered allocator so a replaced buffer is freed after the kernels
// already queued on that stream.
// ---------------------------------------------------------------------------

struct Workspace {
  float* partials = nullptr;
  size_t partial_floats = 0;
  int* counters = nullptr;
  size_t num_counters = 0;
};

constexpr size_t kMaxWorkspaceBytes = size_t(32) << 20;

std::mutex g_ws_mutex;
std::map<std::pair<int, cudaStream_t>, Workspace> g_ws;

// Returns false if the workspace cannot be provided (e.g. during graph
// capture before it was ever allocated): the caller then runs without split-K.
bool get_workspace(int device, cudaStream_t stream, size_t floats,
                   size_t counters, Workspace& out) {
  std::lock_guard<std::mutex> lock(g_ws_mutex);
  Workspace& ws = g_ws[{device, stream}];
  if (ws.partial_floats < floats || ws.num_counters < counters) {
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &capture);
    if (capture != cudaStreamCaptureStatusNone) return false;
    if (ws.partial_floats < floats) {
      if (ws.partials) cudaFreeAsync(ws.partials, stream);
      ws.partials = nullptr;
      ws.partial_floats = 0;
      if (cudaMallocAsync(&ws.partials, floats * sizeof(float), stream) !=
          cudaSuccess) {
        cudaGetLastError();
        return false;
      }
      ws.partial_floats = floats;
    }
    if (ws.num_counters < counters) {
      size_t n = std::max<size_t>(counters, 4096);
      if (ws.counters) cudaFreeAsync(ws.counters, stream);
      ws.counters = nullptr;
      ws.num_counters = 0;
      if (cudaMallocAsync(&ws.counters, n * sizeof(int), stream) !=
          cudaSuccess) {
        cudaGetLastError();
        return false;
      }
      cudaMemsetAsync(ws.counters, 0, n * sizeof(int), stream);
      ws.num_counters = n;
    }
  }
  out = ws;
  return true;
}

// ---------------------------------------------------------------------------
// Per-device kernel properties (SM count, CTAs per SM of every config).
// ---------------------------------------------------------------------------

template <typename Gemm>
int setup_kernel(int device) {
  static std::array<int, 64> occupancy{};  // 0 = not yet initialised
  static std::mutex mutex;
  int& occ = occupancy[device];
  if (occ > 0) return occ;
  std::lock_guard<std::mutex> lock(mutex);
  if (occ > 0) return occ;
  auto kernel = moe_gemm::expert_gemm_pro6000_kernel<Gemm>;
  int smem = Gemm::kSmemBytes;
  if (smem >= (48 << 10)) {
    TORCH_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    smem) == cudaSuccess,
                "fused_moe_ffn: cannot reserve ", smem, " B shared memory");
  }
  int blocks = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, Gemm::kThreads,
                                                smem);
  occ = std::max(blocks, 0) + 1;  // store +1 so 0 means "unknown"
  return occ;
}

int sm_count(int device) {
  static std::array<int, 64> sms{};
  if (sms[device] == 0) {
    int n = 0;
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, device);
    sms[device] = n;
  }
  return sms[device];
}

struct TileInfo {
  int tm, tn, tk;
  int ctas_per_sm;
  bool split_k;
};

template <typename Gemm>
TileInfo tile_info(int device) {
  using TB = typename Gemm::ThreadblockShape;
  return {TB::kM, TB::kN, TB::kK, setup_kernel<Gemm>(device) - 1,
          Gemm::kSplitK};
}

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------

template <typename Gemm>
void launch(int device, int sms, cutlass::gemm::GemmCoord problem, int splits,
            Elem const* A, int lda, Elem const* B0, Elem const* B1, int ldb,
            Elem* D, int ldd, cudaStream_t stream, double l2_bytes,
            moe_gemm::SiluAndMul::Params act = {}) {
  int const slots = sms * (setup_kernel<Gemm>(device) - 1);
  auto params =
      Gemm::make_params(problem, splits, sms, A, lda, B0, B1, ldb, D, ldd, act);
  Gemm::fit_raster(params, problem, l2_bytes, slots);
  if (Gemm::split_tiles(params) > 0) {
    size_t floats = Gemm::workspace_floats(params);
    Workspace ws;
    if (floats * sizeof(float) > kMaxWorkspaceBytes ||
        !get_workspace(device, stream, floats, Gemm::workspace_counters(params),
                       ws)) {
      params =
          Gemm::make_params(problem, 1, sms, A, lda, B0, B1, ldb, D, ldd, act);
      Gemm::fit_raster(params, problem, l2_bytes, slots);
    } else {
      params.partials = ws.partials;
      params.counters = ws.counters;
    }
  }
  moe_gemm::expert_gemm_pro6000_kernel<Gemm>
      <<<Gemm::grid_dim(params), Gemm::kThreads, Gemm::kSmemBytes, stream>>>(
          params);
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "fused_moe_ffn launch failed: ", cudaGetErrorString(err));
}

// Calls f.template operator()<Gemm>() for the cfg-th type of Configs.
template <typename Configs, typename F, size_t... Is>
void visit_config(int cfg, F&& f, std::index_sequence<Is...>) {
  bool done =
      ((cfg == int(Is)
            ? (f.template operator()<std::tuple_element_t<Is, Configs>>(), true)
            : false) ||
       ...);
  TORCH_CHECK(done, "fused_moe_ffn: invalid tile config ", cfg);
}

template <typename Configs, typename F>
void visit_config(int cfg, F&& f) {
  visit_config<Configs>(cfg, std::forward<F>(f),
                        std::make_index_sequence<std::tuple_size_v<Configs>>{});
}

template <typename Configs, size_t... Is>
auto tile_table(int device, std::index_sequence<Is...>) {
  return std::array<TileInfo, sizeof...(Is)>{
      tile_info<std::tuple_element_t<Is, Configs>>(device)...};
}

template <typename Configs>
auto tile_table(int device) {
  return tile_table<Configs>(
      device, std::make_index_sequence<std::tuple_size_v<Configs>>{});
}

// ---------------------------------------------------------------------------
// Config selection: minimise a wave-quantised cost model (µs) whose constants
// were fitted (log-error least squares) to a sweep of every config × split-K
// factor over the expert shapes of all models in
// benchmarks/kernel_audit/models.py, decode to 200K-token prefill.
//
//   q       = CTAs on the busiest SM, occ = resident CTAs per SM
//   compute = ⌊q / occ⌋ · occ · CTA flops / rate
//             + r · CTA flops / (rate · (r / occ)^gamma)    (r = q mod occ)
//             + ⌈q / occ⌉ · per-round overhead
//   memory  = operand bytes / (2 TB/s · min(1, CTAs / SMs))
//   split-K = 1.13 µs · splits + 0.79 µs (fixup round trips)
//
// Split-K candidates keep ≥ 8 k-tiles per slice and stop once the grid fills
// one round of resident CTAs (gate/up) or reaches 96 CTAs (down); these caps
// gave the lowest selection regret on the sweep (≈ 1 % time-weighted).
// ---------------------------------------------------------------------------

struct Choice {
  int cfg;
  int splits;
};

// Modelled time (µs) of one config with `splits` K slices; infinity when the
// split-K workspace would not fit (launch() would fall back to one slice).
double model_cost(TileInfo const& t, ConfigCost const& cost, int sms, int M,
                  int Nd, int K, int accumulators, int splits) {
  long tiles_mn = long((M + t.tm - 1) / t.tm) * ((Nd + t.tn - 1) / t.tn);
  int k_tiles = (K + t.tk - 1) / t.tk;
  int k_per = (k_tiles + splits - 1) / splits;
  splits = (k_tiles + k_per - 1) / k_per;
  long ctas = tiles_mn * splits;
  long q = (ctas + sms - 1) / sms;
  long full = q / t.ctas_per_sm;
  long rem = q % t.ctas_per_sm;
  double rate = cost.tflops_per_sm * 1e6;
  double cta_flops = 2.0 * t.tm * t.tn * t.tk * double(k_per) * accumulators;
  double compute = full * t.ctas_per_sm * cta_flops / rate;
  if (rem > 0) {
    double partial = double(rem) / t.ctas_per_sm;
    compute += rem * cta_flops / (rate * std::pow(partial, cost.gamma));
  }
  compute += (full + (rem > 0)) * cost.round_us;
  double bytes = (double(accumulators) * Nd + M) * K * 2.0;
  double memory = bytes / (2.0e6 * std::min(1.0, double(ctas) / double(sms)));
  double reduce = 0.0;
  if (splits > 1) {
    if (double(ctas) * t.tm * t.tn * accumulators * 4.0 >
        double(kMaxWorkspaceBytes))
      return 1e30;
    reduce = 1.13 * splits + 0.79;
  }
  return std::max(compute, memory) + reduce;
}

template <size_t N>
Choice select_config(std::array<TileInfo, N> const& tiles,
                     ConfigCost const (&cost_of)[N], int sms, int M, int Nd,
                     int K, int accumulators, bool split_to_full_round) {
  Choice best{static_cast<int>(N) - 1, 1};
  double best_cost = 1e30;
  for (size_t c = 0; c < N; ++c) {
    TileInfo const& t = tiles[c];
    if (t.ctas_per_sm <= 0) continue;
    long tiles_mn = long((M + t.tm - 1) / t.tm) * ((Nd + t.tn - 1) / t.tn);
    int k_tiles = (K + t.tk - 1) / t.tk;
    int max_splits = 1;
    if (t.split_k) {
      auto room = [&](int s) {
        return split_to_full_round
                   ? tiles_mn * (s + 1) <= long(sms) * t.ctas_per_sm
                   : tiles_mn * s < 96;
      };
      while (max_splits < 8 && room(max_splits) &&
             k_tiles >= 8 * (max_splits + 1))
        ++max_splits;
    }
    for (int s = 1; s <= max_splits; ++s) {
      double cost = model_cost(t, cost_of[c], sms, M, Nd, K, accumulators, s);
      if (cost < best_cost) {
        best_cost = cost;
        best = {static_cast<int>(c), s};
      }
    }
  }
  return best;
}

// Decode / short prefill (M ≤ 64) is memory-latency bound and poorly
// described by the model above: use the smallest tile (last config) and add
// K slices until the grid reaches `target_ctas`, keeping ≥ 8 k-tiles per
// slice (fitted on the same sweep: within 3 % of the best config on average).
template <size_t N>
Choice small_m_config(std::array<TileInfo, N> const& tiles, int M, int Nd,
                      int K, long target_ctas) {
  constexpr int kSplits[] = {1, 2, 3, 4, 6, 8};
  TileInfo const& t = tiles[N - 1];
  long tiles_mn = long((M + t.tm - 1) / t.tm) * ((Nd + t.tn - 1) / t.tn);
  int k_tiles = (K + t.tk - 1) / t.tk;
  int splits = 1;
  for (int s : kSplits) {
    if (tiles_mn * splits >= target_ctas) break;
    if (k_tiles >= 8 * s) splits = s;
  }
  return {static_cast<int>(N) - 1, splits};
}

// Measured table (fused_moe_mlp_tuned.h) for the expert shapes of the
// supported models on the GPU it was tuned on: the winner at the smallest
// tuned M ≥ M (the largest one beyond the grid). Rounding up keeps the grid
// of CTAs no larger than the one the entry was measured with, so a config
// never gains an extra wave between grid points. Other shapes / GPUs use the
// cost model.
bool tuned_config(bool dual, int sms, int M, int Nd, int K, int num_cfgs,
                  Choice& out) {
  if (sms != kTunedSmsPro6000) return false;
  TunedExpertGemm const* lo = nullptr;
  TunedExpertGemm const* hi = nullptr;
  for (auto const& e : kTunedExpertGemmPro6000) {
    if (e.dual != dual || e.n != Nd || e.k != K || e.cfg >= num_cfgs) continue;
    if (e.m <= M && (!lo || e.m > lo->m)) lo = &e;
    if (e.m >= M && (!hi || e.m < hi->m)) hi = &e;
  }
  TunedExpertGemm const* e = hi ? hi : lo;
  if (!e) return false;
  out = {e->cfg, e->splits};
  return true;
}

}  // namespace

size_t l2_size(int device) {
  static std::array<int, 64> l2{};
  if (l2[device] == 0)
    cudaDeviceGetAttribute(&l2[device], cudaDevAttrL2CacheSize, device);
  return size_t(l2[device]);
}

// Keeps the [M, I] intermediate in L2 between the gate/up launch that writes
// it and the down launch that reads it. Without a hint, the weights and X
// streamed by gate/up evict it (mixtral 9K: 66 MB written back and read
// again on a 390 MB op). The gate/up launch runs with an L2 persisting
// access-policy window on the intermediate (weights and X stream around it);
// the down launch reads it with a streaming window, which releases the lines.
// Applies to intermediates from 8 MB up to twice the persisting carve-out
// (80 MB on sm_120); one larger than the carve-out persists in part
// (hitRatio = carve-out / size).
class IntermediateWindow {
  // Share of the carve-out given to the window (mixtral 12K, 88 MB: DRAM /
  // ideal 1.10 at 1.0, 1.15 at 0.9).
  static constexpr double kPersistShare = 1.0;

 public:
  IntermediateWindow(int device, cudaStream_t stream, void* base, size_t bytes)
      : stream_(stream) {
    size_t const carve = carve_out(device);
    // Below half resident the carve-out costs the operands more L2 than it
    // saves (mixtral 50K: 367 MB intermediate, Rule F).
    if (!applies(carve, bytes)) return;
    if (cudaStreamGetAttribute(stream, cudaStreamAttributeAccessPolicyWindow,
                               &saved_) != cudaSuccess)
      return;
    window_.base_ptr = base;
    window_.num_bytes = std::min(bytes, window_max(device));
    window_.hitRatio = float(std::min(
        1.0, kPersistShare * double(carve) / double(window_.num_bytes)));
    window_.hitProp = cudaAccessPropertyPersisting;
    window_.missProp = cudaAccessPropertyStreaming;
    active_ = set(window_);
  }
  // Bytes of an intermediate of `bytes` that the window keeps in L2.
  static size_t persisting_bytes(int device, size_t bytes) {
    size_t const carve = carve_out(device);
    if (!applies(carve, bytes)) return 0;
    double const n = double(std::min(bytes, window_max(device)));
    return size_t(n * std::min(1.0, kPersistShare * double(carve) / n));
  }
  void last_use() {
    if (!active_) return;
    window_.hitRatio = 1.f;
    window_.hitProp = cudaAccessPropertyStreaming;
    set(window_);
  }
  ~IntermediateWindow() {
    if (active_)
      cudaStreamSetAttribute(stream_, cudaStreamAttributeAccessPolicyWindow,
                             &saved_);
  }

 private:
  static bool applies(size_t carve, size_t bytes) {
    return carve > 0 && bytes >= (size_t(8) << 20) &&
           double(bytes) <= 2 * kPersistShare * double(carve);
  }
  bool set(cudaAccessPolicyWindow const& w) {
    cudaStreamAttrValue v{};
    v.accessPolicyWindow = w;
    bool ok =
        cudaStreamSetAttribute(stream_, cudaStreamAttributeAccessPolicyWindow,
                               &v) == cudaSuccess;
    if (!ok) cudaGetLastError();
    return ok;
  }
  // Raises the persisting L2 limit to the device maximum once per device.
  static size_t carve_out(int device) {
    static std::array<long long, 64> carve{};
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    if (carve[device] == 0) {
      int max_persist = 0;
      cudaDeviceGetAttribute(&max_persist, cudaDevAttrMaxPersistingL2CacheSize,
                             device);
      carve[device] = -1;
      if (max_persist > 0 && cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                                                max_persist) == cudaSuccess)
        carve[device] = max_persist;
      cudaGetLastError();
    }
    return carve[device] > 0 ? size_t(carve[device]) : 0;
  }
  static size_t window_max(int device) {
    int v = 0;
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxAccessPolicyWindowSize, device);
    return size_t(std::max(v, 0));
  }

  cudaStream_t stream_;
  cudaStreamAttrValue saved_{};
  cudaAccessPolicyWindow window_{};
  bool active_ = false;
};

// When a GEMM's operands do not fit in L2 together, its DRAM traffic depends
// on the N tile as much as on the tile order (mixtral 50K down: 128 x 256
// tiles 1.26 GB, 128 x 128 x 64 tiles 0.91 GB, within 2 % of the time).
// Switch to the config with the least modelled traffic when the chosen one
// would read more than 1.25x that.
template <typename Configs>
void l2_lean_config(int device, int sms, cutlass::gemm::GemmCoord problem,
                    double l2, int& cfg, int& splits) {
  constexpr int kNum = int(std::tuple_size_v<Configs>);
  double traffic[kNum];
  for (int c = 0; c < kNum; ++c)
    visit_config<Configs>(c, [&]<typename Gemm>() {
      int const resident = sms * (setup_kernel<Gemm>(device) - 1);
      traffic[c] = moe_gemm::plan_raster<Gemm>(problem, l2, resident).traffic;
    });
  if (traffic[cfg] == 0) return;  // fits: keep the tuned choice
  int best = cfg;
  for (int c = 0; c < kNum; ++c)
    if (traffic[c] > 0 && traffic[c] < traffic[best]) best = c;
  if (traffic[cfg] > 1.25 * traffic[best]) {
    cfg = best;
    splits = 1;
  }
}

void fused_moe_ffn_pro6000(torch::Tensor& hidden, torch::Tensor& gate_proj,
                           torch::Tensor& up_proj, torch::Tensor& down_proj,
                           torch::Tensor& fused_buf, torch::Tensor& output,
                           cudaStream_t stream, int dual_cfg, int dual_splits,
                           int down_cfg, int down_splits,
                           const MoEActivation& activation) {
  check_moe_activation(activation);
  TORCH_CHECK(fused_moe_ffn_pro6000_enabled(hidden.get_device()),
              "fused_moe_ffn_pro6000: tuned for the RTX PRO 6000 only "
              "(MOE_INFINITY_UNVALIDATED_HW=1 to re-tune elsewhere)");
  TORCH_CHECK(hidden.scalar_type() == at::kBFloat16,
              "fused_moe_ffn_into: BF16 only");
  const int M = static_cast<int>(hidden.size(0));
  const int H = static_cast<int>(hidden.size(1));
  const int I = static_cast<int>(gate_proj.size(0));

  TORCH_CHECK(down_proj.size(0) == H,
              "fused_moe_ffn_into: hidden dim mismatch");
  TORCH_CHECK(
      gate_proj.size(1) == H && up_proj.size(1) == H && up_proj.size(0) == I,
      "fused_moe_ffn_into: gate/up proj shape mismatch");
  TORCH_CHECK(down_proj.size(1) == I,
              "fused_moe_ffn_into: down proj intermediate dim mismatch");
  TORCH_CHECK(H % 8 == 0 && I % 8 == 0,
              "fused_moe_ffn_into: H and I must be multiples of 8");
  for (auto* t :
       {&hidden, &gate_proj, &up_proj, &down_proj, &fused_buf, &output}) {
    TORCH_CHECK(t->scalar_type() == at::kBFloat16,
                "fused_moe_ffn_into: BF16 only");
    TORCH_CHECK(t->is_contiguous(),
                "fused_moe_ffn_into: tensors must be row-major contiguous");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0,
                "fused_moe_ffn_into: tensors must be 16-byte aligned");
  }
  TORCH_CHECK(fused_buf.numel() >= int64_t(M) * I,
              "fused_moe_ffn_into: fused_buf too small");
  if (M == 0) return;

  const int device = hidden.get_device();
  TORCH_CHECK(device >= 0 && device < 64, "fused_moe_ffn_into: CUDA tensors");
  c10::cuda::CUDAGuard device_guard(device);
  const int sms = sm_count(device);

  constexpr int kSmallM = 64;
  bool const auto_dual = dual_cfg < 0, auto_down = down_cfg < 0;
  Choice tuned;
  if (dual_cfg < 0 && tuned_config(true, sms, M, I, H, kNumDual, tuned)) {
    dual_cfg = tuned.cfg;
    dual_splits = tuned.splits;
  }
  if (down_cfg < 0 && tuned_config(false, sms, M, H, I, kNumDown, tuned)) {
    down_cfg = tuned.cfg;
    down_splits = tuned.splits;
  }
  if (dual_cfg < 0) {
    auto tiles = tile_table<DualConfigs>(device);
    Choice c = M <= kSmallM
                   ? small_m_config(tiles, M, I, H, 96)
                   : select_config(tiles, kDualCost, sms, M, I, H, 2, true);
    dual_cfg = c.cfg;
    dual_splits = c.splits;
  }
  if (down_cfg < 0) {
    auto tiles = tile_table<DownConfigs>(device);
    Choice c = M <= kSmallM
                   ? small_m_config(tiles, M, H, I, sms)
                   : select_config(tiles, kDownCost, sms, M, H, I, 1, false);
    down_cfg = c.cfg;
    down_splits = c.splits;
  }

  // L2 left to the operands: minus the persisting intermediate (gate/up
  // only).
  size_t const act_bytes = size_t(M) * I * sizeof(Elem);
  double const l2 = double(l2_size(device));
  double const l2_dual =
      l2 - double(IntermediateWindow::persisting_bytes(device, act_bytes));
  if (auto_dual)
    l2_lean_config<DualConfigs>(device, sms, {M, I, H}, l2_dual, dual_cfg,
                                dual_splits);
  if (auto_down)
    l2_lean_config<DownConfigs>(device, sms, {M, H, I}, l2, down_cfg,
                                down_splits);

  auto* x = reinterpret_cast<Elem const*>(hidden.data_ptr());
  auto* wg = reinterpret_cast<Elem const*>(gate_proj.data_ptr());
  auto* wu = reinterpret_cast<Elem const*>(up_proj.data_ptr());
  auto* wd = reinterpret_cast<Elem const*>(down_proj.data_ptr());
  auto* act = reinterpret_cast<Elem*>(fused_buf.data_ptr());
  auto* out = reinterpret_cast<Elem*>(output.data_ptr());

  moe_gemm::SiluAndMul::Params const act_params{
      activation.kind, activation.alpha, activation.limit};
  IntermediateWindow window(device, stream, act, act_bytes);
  visit_config<DualConfigs>(dual_cfg, [&]<typename Gemm>() {
    launch<Gemm>(device, sms, {M, I, H}, dual_splits, x, H, wg, wu, H, act, I,
                 stream, l2_dual, act_params);
  });
  window.last_use();
  visit_config<DownConfigs>(down_cfg, [&]<typename Gemm>() {
    launch<Gemm>(device, sms, {M, H, I}, down_splits, act, I, wd, nullptr, I,
                 out, H, stream, l2);
  });
}

void fused_moe_ffn_pro6000(torch::Tensor& hidden, torch::Tensor& gate_proj,
                           torch::Tensor& up_proj, torch::Tensor& down_proj,
                           torch::Tensor& fused_buf, torch::Tensor& output,
                           cudaStream_t stream, int dual_cfg, int dual_splits,
                           int down_cfg, int down_splits) {
  fused_moe_ffn_pro6000(hidden, gate_proj, up_proj, down_proj, fused_buf,
                        output, stream, dual_cfg, dual_splits, down_cfg,
                        down_splits, MoEActivation{});
}

void check_moe_activation(const MoEActivation& act) {
  switch (act.kind) {
    case MoEActivationKind::kSiLU:
      return;
    case MoEActivationKind::kClampedSiLU:
      TORCH_CHECK(act.limit > 0.0f,
                  "clamped_silu activation requires limit > 0, got ",
                  act.limit);
      return;
    case MoEActivationKind::kSwiGLUOAI:
      TORCH_CHECK(act.limit > 0.0f && act.alpha > 0.0f,
                  "swiglu_oai activation requires alpha > 0 and limit > 0, "
                  "got alpha=",
                  act.alpha, ", limit=", act.limit);
      return;
  }
  TORCH_CHECK(false, "unknown MoE activation kind ",
              static_cast<int>(act.kind));
}

void fused_moe_ffn_into(torch::Tensor& hidden,     // [M, H]
                        torch::Tensor& gate_proj,  // [I, H]
                        torch::Tensor& up_proj,    // [I, H]
                        torch::Tensor& down_proj,  // [H, I]
                        torch::Tensor& gate_buf,   // [M, I]
                        torch::Tensor& fused_buf,  // [M, I]
                        torch::Tensor& output,     // [M, H]
                        cudaStream_t stream) {
  fused_moe_ffn_into(hidden, gate_proj, up_proj, down_proj, gate_buf, fused_buf,
                     output, MoEActivation{}, stream);
}

void fused_moe_ffn_into(torch::Tensor& hidden,     // [M, H]
                        torch::Tensor& gate_proj,  // [I, H]
                        torch::Tensor& up_proj,    // [I, H]
                        torch::Tensor& down_proj,  // [H, I]
                        torch::Tensor& gate_buf,   // [M, I]
                        torch::Tensor& fused_buf,  // [M, I]
                        torch::Tensor& output,     // [M, H]
                        const MoEActivation& activation, cudaStream_t stream) {
  check_moe_activation(activation);
  if (fused_moe_ffn_pro6000_enabled(hidden.get_device())) {
    fused_moe_ffn_pro6000(hidden, gate_proj, up_proj, down_proj, fused_buf,
                          output, stream, -1, 0, -1, 0, activation);
    return;
  }
  fused_moe_ffn_into_generic(hidden, gate_proj, up_proj, down_proj, gate_buf,
                             fused_buf, output, activation, stream);
}

bool fused_moe_ffn_pro6000_enabled(int device) {
  return moe_infinity::hw::Supports(moe_infinity::hw::Gate::kRtxPro6000,
                                    device);
}

std::pair<int, int> fused_moe_ffn_num_configs() { return {kNumDual, kNumDown}; }

namespace {
template <typename Configs, size_t... Is>
std::vector<std::array<int, 3>> tile_shapes(std::index_sequence<Is...>) {
  return {{std::tuple_element_t<Is, Configs>::ThreadblockShape::kM,
           std::tuple_element_t<Is, Configs>::ThreadblockShape::kN,
           std::tuple_element_t<Is, Configs>::ThreadblockShape::kK}...};
}
}  // namespace

std::pair<std::vector<std::array<int, 3>>, std::vector<std::array<int, 3>>>
fused_moe_ffn_tile_shapes() {
  return {tile_shapes<DualConfigs>(
              std::make_index_sequence<std::tuple_size_v<DualConfigs>>{}),
          tile_shapes<DownConfigs>(
              std::make_index_sequence<std::tuple_size_v<DownConfigs>>{})};
}
