// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// Grouped (all experts of one MoE layer per launch) prefill GEMMs.
//
//   GroupedExpertGemm<..., Dual=true>   act[r] = act(x[tok[r]]·Wgᵀ,
//   x[tok[r]]·Wuᵀ) GroupedExpertGemm<..., Dual=false>  out[tok[r]] += w[r] ·
//   bf16(act[r]·Wdᵀ)
//
// Rows r are the routed (token, expert) pairs in the sorted layout produced by
// the routing kernel: expert e owns rows [offsets[e], offsets[e+1]). The
// gate/up launch gathers its X rows through the token list (CUTLASS GatherA
// iterator) and writes the activations contiguously; the down launch reads them
// back contiguously and fuses the routing-weight multiply and the scatter-add
// into its epilogue (fp32 vector reductions straight from the accumulators).
//
// Both launches are persistent: SMs × occupancy CTAs walk one tile space that
// concatenates the tiles of every expert, so a layer with many small experts
// still fills the GPU. Within an expert, consecutive tiles share a weight
// panel (M tiles of one N panel), which keeps that panel in L2.

#pragma once

#include "kernel/moe_expert_gemm.h"

#include <cstdint>
#include <type_traits>

namespace moe_gemm {

struct GroupedExpert {
  void const* gate;  // [I, H]
  void const* up;    // [I, H]
  void const* down;  // [H, I]
  int64_t expert_id;
};
static_assert(sizeof(GroupedExpert) == 32, "expert table row is int64[4]");

struct GroupedRouting {
  int const* offsets;     // [n + 1] first sorted row of every expert
  int const* tokens;      // [rows] token of every sorted row
  float const* weights;   // [rows] routing weight of every sorted row
  int const* tile_start;  // [n + 1] first tile of every expert (this GEMM)
};

template <typename Element_, typename ThreadblockShape_, typename WarpShape_,
          int Stages, bool Dual, typename Act = SiluAndMul>
struct GroupedExpertGemm {
  using Element = Element_;
  using ElementAccumulator = float;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int kAlignment = 128 / cutlass::sizeof_bits<Element>::value;
  static constexpr bool kDual = Dual;

  // gate/up gathers its A rows (tokens); down reads them contiguously.
  using DefaultMma = typename cutlass::gemm::threadblock::DefaultMma<
      Element, LayoutA, kAlignment, Element, LayoutB, kAlignment,
      ElementAccumulator, LayoutC, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, ThreadblockShape, WarpShape, InstructionShape,
      Stages, cutlass::arch::OpMultiplyAdd, false,
      cutlass::gemm::SharedMemoryClearOption::kNone,
      /*GatherA=*/Dual>::ThreadblockMma;
  using MmaSingle = DefaultMma;
  static constexpr auto kCacheOpB = DefaultMma::kCacheOpB;

  using MmaDual = cutlass::gemm::threadblock::DualMmaMultistage<
      typename MmaSingle::Shape, typename MmaSingle::IteratorA,
      typename MmaSingle::SmemIteratorA, DefaultMma::kCacheOpA,
      typename MmaSingle::IteratorB, typename MmaSingle::SmemIteratorB,
      kCacheOpB, typename MmaSingle::IteratorB,
      typename MmaSingle::SmemIteratorB, ElementAccumulator, LayoutC,
      typename MmaSingle::Policy, typename MmaSingle::Policy, Stages,
      cutlass::gemm::SharedMemoryClearOption::kNone>;

  using Mma = std::conditional_t<Dual, MmaDual, MmaSingle>;
  using IteratorA = typename MmaSingle::IteratorA;
  using IteratorB = typename MmaSingle::IteratorB;
  using FragmentC = typename MmaSingle::FragmentC;
  using WarpMma = typename MmaSingle::Operator;
  using MmaIterations = typename WarpMma::MmaIterations;
  using WarpCount = typename MmaSingle::WarpCount;

  static constexpr int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;
  static_assert(kPartitionsK == 1, "no intra-CTA split-K");
  using OutputOp = cutlass::epilogue::thread::LinearCombination<
      Element, kAlignment, ElementAccumulator, ElementAccumulator,
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;
  using Epilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
          ThreadblockShape, WarpMma, kPartitionsK, OutputOp,
          kAlignment>::Epilogue;
  using OutputIterator = typename Epilogue::OutputTileIterator;

  static constexpr int kThreads = 32 * WarpCount::kCount;

  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };
  struct KernelStorage {
    SharedStorage tiles;
  };
  static constexpr int kSmemBytes = int(sizeof(KernelStorage));

  struct Params {
    int n;     // experts
    int N, K;  // per-expert GEMM: [rows_e, K] x [K, N]
    int tiles_n;
    int panels;  // N panels per pass: all experts sweep `panels` N panels
                 // before the next pass, bounding the output working set
    GroupedExpert const* experts;
    GroupedRouting routing;
    typename IteratorA::Params params_A;
    Element const* A;  // gate/up: hidden [T, K]; down: act [rows, K]
    typename IteratorB::Params params_B;
    typename OutputIterator::Params params_D;
    Element* act;  // gate/up output [rows, N]
    float* out;    // down output [T, N], accumulated
    Element* y;    // down output per sorted row [rows, N] (combine mode)
    int64_t ldo;
    typename Act::Params act_params;
  };

  static Params make_params(int n, int N, int K, int ldA,
                            GroupedExpert const* experts,
                            GroupedRouting routing, Element const* A,
                            Element* act, float* out, int64_t ldo,
                            typename Act::Params act_params = {}) {
    Params p;
    p.n = n;
    p.N = N;
    p.K = K;
    p.tiles_n = (N + ThreadblockShape::kN - 1) / ThreadblockShape::kN;
    p.panels = p.tiles_n;
    p.experts = experts;
    p.routing = routing;
    p.params_A = typename IteratorA::Params(LayoutA(ldA));
    p.A = A;
    p.params_B = typename IteratorB::Params(LayoutB(K));
    p.params_D = typename OutputIterator::Params(LayoutC(N));
    p.act = act;
    p.out = out;
    p.y = nullptr;
    p.ldo = ldo;
    p.act_params = act_params;
    return p;
  }

  CUTLASS_DEVICE
  static void red_add4(float* addr, float4 v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(addr),
                 "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
                 : "memory");
#else
    atomicAdd(addr, v.x);
    atomicAdd(addr + 1, v.y);
    atomicAdd(addr + 2, v.z);
    atomicAdd(addr + 3, v.w);
#endif
  }

  // out[tok[r]] += w[r] * bf16(acc) straight from the mma.m16n8 fragments.
  // Chunk c covers mma (c % kRow, c / kRow) of the warp tile; lane (g, q)
  // holds rows g and g+8 at columns 2q, 2q+1. Lane pairs (q, q^1) swap one
  // row each so every lane owns 4 consecutive columns of one row and issues
  // one 16-byte fp32 vector reduction.
  CUTLASS_DEVICE
  static void scatter_epilogue(Params const& p, FragmentC const& acc, int row0,
                               int rows, int tile_m, int tile_n, int warp_idx,
                               int lane_idx) {
    int const warp_mn = warp_idx % (WarpCount::kM * WarpCount::kN);
    int const warp_m = warp_mn % WarpCount::kM;
    int const warp_n = warp_mn / WarpCount::kM;
    int const g = lane_idx / 4;
    int const q = lane_idx % 4;
    bool const odd = q & 1;
    int const base_r = tile_m * ThreadblockShape::kM + warp_m * WarpShape::kM +
                       g + (odd ? 8 : 0);
    int const base_c =
        tile_n * ThreadblockShape::kN + warp_n * WarpShape::kN + 2 * (q & ~1);
    float const* v = reinterpret_cast<float const*>(&acc);
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < MmaIterations::kRow; ++mi) {
      int const r = base_r + mi * 16;
      bool const live = r < rows;
      int tok = 0;
      float w = 0.f;
      if (live) {
        tok = p.routing.tokens[row0 + r];
        w = p.routing.weights[row0 + r];
      }
      float* orow = p.out + int64_t(tok) * p.ldo;
      CUTLASS_PRAGMA_UNROLL
      for (int ni = 0; ni < MmaIterations::kColumn; ++ni) {
        int const chunk = mi + ni * MmaIterations::kRow;
        float const x0 = float(Element(v[chunk * 4 + 0]));
        float const x1 = float(Element(v[chunk * 4 + 1]));
        float const x2 = float(Element(v[chunk * 4 + 2]));
        float const x3 = float(Element(v[chunk * 4 + 3]));
        // even lanes keep row g and send row g+8; odd lanes the opposite.
        float const s0 = odd ? x0 : x2;
        float const s1 = odd ? x1 : x3;
        float const r0 = __shfl_xor_sync(0xffffffff, s0, 1);
        float const r1 = __shfl_xor_sync(0xffffffff, s1, 1);
        int const c = base_c + ni * 8;
        if (!live || c >= p.N) continue;
        float4 out =
            odd ? make_float4(r0, r1, x2, x3) : make_float4(x0, x1, r0, r1);
        out.x *= w;
        out.y *= w;
        out.z *= w;
        out.w *= w;
        red_add4(orow + c, out);
      }
    }
  }

  CUTLASS_DEVICE
  static void run_tile(Params const& p, SharedStorage& smem, int e, int tile_m,
                       int tile_n) {
    int const row0 = p.routing.offsets[e];
    int const rows = p.routing.offsets[e + 1] - row0;
    GroupedExpert const ex = p.experts[e];

    int const thread_idx = threadIdx.x;
    int const warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int const lane_idx = threadIdx.x % 32;
    int const k_iterations =
        (p.K + ThreadblockShape::kK - 1) / ThreadblockShape::kK;

    cutlass::MatrixCoord offset_A{tile_m * ThreadblockShape::kM, 0};
    cutlass::MatrixCoord offset_B{0, tile_n * ThreadblockShape::kN};

    FragmentC acc[Dual ? 2 : 1];
    CUTLASS_PRAGMA_UNROLL
    for (int a = 0; a < (Dual ? 2 : 1); ++a) acc[a].clear();

    Mma mma(smem.main_loop, thread_idx, warp_idx, lane_idx);
    if constexpr (Dual) {
      IteratorA iterator_A(p.params_A, const_cast<Element*>(p.A), {rows, p.K},
                           thread_idx, offset_A, p.routing.tokens + row0);
      IteratorB iterator_B0(p.params_B,
                            static_cast<Element*>(const_cast<void*>(ex.gate)),
                            {p.K, p.N}, thread_idx, offset_B);
      IteratorB iterator_B1(p.params_B,
                            static_cast<Element*>(const_cast<void*>(ex.up)),
                            {p.K, p.N}, thread_idx, offset_B);
      mma(k_iterations, acc[0], acc[1], iterator_A, iterator_B0, iterator_B1,
          acc[0], acc[1]);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentC::kElements; ++i)
        acc[0][i] = Act::apply(acc[0][i], acc[1][i], p.act_params);

      cutlass::arch::cp_async_wait<0>();
      __syncthreads();
      OutputIterator iterator_D(
          p.params_D, p.act + int64_t(row0) * p.N, {rows, p.N}, thread_idx,
          {tile_m * ThreadblockShape::kM, tile_n * ThreadblockShape::kN});
      Epilogue epilogue(smem.epilogue, thread_idx, warp_idx, lane_idx);
      epilogue(OutputOp(typename OutputOp::Params(1.0f)), iterator_D, acc[0]);
    } else {
      IteratorA iterator_A(p.params_A,
                           const_cast<Element*>(p.A) + int64_t(row0) * p.K,
                           {rows, p.K}, thread_idx, offset_A);
      IteratorB iterator_B(p.params_B,
                           static_cast<Element*>(const_cast<void*>(ex.down)),
                           {p.K, p.N}, thread_idx, offset_B);
      mma(k_iterations, acc[0], iterator_A, iterator_B, acc[0]);
      cutlass::arch::cp_async_wait<0>();
      if (p.y) {
        // Combine mode: bf16 rows in sorted order, summed per token later.
        __syncthreads();
        OutputIterator iterator_D(
            p.params_D, p.y + int64_t(row0) * p.N, {rows, p.N}, thread_idx,
            {tile_m * ThreadblockShape::kM, tile_n * ThreadblockShape::kN});
        Epilogue epilogue(smem.epilogue, thread_idx, warp_idx, lane_idx);
        epilogue(OutputOp(typename OutputOp::Params(1.0f)), iterator_D, acc[0]);
      } else {
        scatter_epilogue(p, acc[0], row0, rows, tile_m, tile_n, warp_idx,
                         lane_idx);
      }
    }
    __syncthreads();  // the next tile's prologue reuses shared memory
  }
};

// Rows of one expert split into `nbig` tiles of big_m rows followed by
// `nsmall` tiles of small_m rows: the small tiles cover the remainder when at
// most max_small of them are cheaper than one more big tile. A single-height
// config has small_m = big_m and max_small = 0.
struct TileSplit {
  int big_m, small_m, max_small;
  CUTLASS_HOST_DEVICE void split(int rows, int& nbig, int& nsmall) const {
    nbig = rows / big_m;
    int const rem = rows - nbig * big_m;
    nsmall = 0;
    if (rem > 0) {
      nsmall = (rem + small_m - 1) / small_m;
      if (nsmall > max_small) {
        ++nbig;
        nsmall = 0;
      }
    }
  }
  CUTLASS_HOST_DEVICE int tiles(int rows) const {
    int a, b;
    split(rows, a, b);
    return a + b;
  }
};

// One persistent launch over the tiles of every expert, with two tile heights
// (Big for full tiles, Small for the remainder of each expert; Small = Big for
// a single-height config). Both share the CTA size and the N tile.
template <typename Big_, typename Small_ = Big_>
struct GroupedTiledGemm {
  using Big = Big_;
  using Small = Small_;
  static constexpr bool kMixed = !std::is_same_v<Big, Small>;
  static_assert(Big::kThreads == Small::kThreads, "one CTA size");
  static_assert(Big::ThreadblockShape::kN == Small::ThreadblockShape::kN,
                "one N tile");
  static_assert(Big::ThreadblockShape::kM % Small::ThreadblockShape::kM == 0,
                "small tiles start on big tile boundaries");
  static_assert(Big::kDual == Small::kDual, "same GEMM");
  using ThreadblockShape = typename Big::ThreadblockShape;
  static constexpr int kBigM = Big::ThreadblockShape::kM;
  static constexpr int kSmallM = Small::ThreadblockShape::kM;
  static constexpr int kThreads = Big::kThreads;

  union KernelStorage {
    typename Big::SharedStorage big;
    typename Small::SharedStorage small;
  };
  static constexpr int kSmemBytes = int(sizeof(KernelStorage));

  struct Params {
    typename Big::Params big;
    typename Small::Params small;
    TileSplit split;
    int* next_tile;  // zeroed before the launch: CTAs claim tiles in order
  };

  template <typename... Args>
  static Params make_params(int max_small, Args&&... args) {
    Params p;
    p.big = Big::make_params(args...);
    p.small = Small::make_params(args...);
    p.split = {kBigM, kMixed ? kSmallM : kBigM, kMixed ? max_small : 0};
    p.next_tile = nullptr;
    return p;
  }
  CUTLASS_DEVICE
  static void run(Params const& p, KernelStorage& storage) {
    // Tiles are numbered pass-major (groups of `panels` N panels), then
    // expert, then N panel, then M tile (big tiles first); the routing kernel
    // published the first tile of every expert within one pass.
    auto const& pb = p.big;
    int const* start = pb.routing.tile_start;
    int const per_pass = start[pb.n];
    int const total = per_pass * (pb.tiles_n / pb.panels);
    // Tiles of different heights take different times, so CTAs claim the
    // next tile from a counter instead of a fixed stride.
    __shared__ int claimed;
    for (int t = blockIdx.x; t < total;) {
      int const pass = t / per_pass;
      int const tl = t - pass * per_pass;
      int lo = 0, hi = pb.n;  // largest e with start[e] <= tl
      while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (__ldg(start + mid) <= tl)
          lo = mid;
        else
          hi = mid;
      }
      int const e = lo;
      int const rows = pb.routing.offsets[e + 1] - pb.routing.offsets[e];
      int nbig, nsmall;
      p.split.split(rows, nbig, nsmall);
      int const tiles_m = nbig + nsmall;
      int const local = tl - start[e];
      int const m = local % tiles_m;
      int const n = pass * pb.panels + local / tiles_m;
      if (kMixed && m >= nbig)
        Small::run_tile(p.small, storage.small, e,
                        nbig * (kBigM / kSmallM) + (m - nbig), n);
      else
        Big::run_tile(pb, storage.big, e, m, n);
      // run_tile ends with a barrier, so every thread has read `claimed`.
      if (threadIdx.x == 0)
        claimed = p.next_tile ? atomicAdd(p.next_tile, 1) + int(gridDim.x)
                              : t + int(gridDim.x);
      __syncthreads();
      t = claimed;
    }
  }
};

template <typename Gemm>
__global__ void __launch_bounds__(Gemm::kThreads)
    grouped_expert_gemm_kernel(typename Gemm::Params params) {
  extern __shared__ __align__(128) char smem[];
  Gemm::run(params, *reinterpret_cast<typename Gemm::KernelStorage*>(smem));
}

}  // namespace moe_gemm
