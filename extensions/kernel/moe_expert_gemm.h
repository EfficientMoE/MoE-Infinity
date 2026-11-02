// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// Expert-FFN GEMM kernels built from CUTLASS 2.x threadblock mainloops.
//
//   ExpertGemm<..., Dual=false>  D = A · Bᵀ                       (down proj)
//   ExpertGemm<..., Dual=true>   D = act(A · B0ᵀ, A · B1ᵀ)         (gate + up)
//
// The dual variant shares the A tile in shared memory between the two B
// operands (CUTLASS examples/45_dual_gemm mainloop), combines the two fp32
// accumulators in registers with `Act` and writes only D, so X is read once
// and no [M, I] gate intermediate is materialised.
//
// Both variants support split-K with an in-kernel fixup: every K slice stores
// its fp32 accumulators to a workspace, the last slice to arrive at a tile
// (per-tile arrival counter) sums all slices in a fixed order and runs the
// epilogue. Counters are reset by that last CTA, so a workspace can be reused
// by every launch on the same stream without a memset. Split-K covers either
// every tile or only the tiles of the last partial wave ("stream-K
// remainder"), which removes the wave-quantisation tail of grids just above a
// multiple of the resident-CTA count. Tiles are rasterised in groups so the
// re-read operand stays in L2.

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/threadblock/default_epilogue_tensor_op.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/threadblock/default_mma.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/matrix_coord.h>
#include <cutlass/numeric_types.h>

#include <algorithm>
#include <type_traits>

#include "kernel/dual_gemm/dual_mma_multistage.h"

namespace moe_gemm {

// act(g, u) applied to the fp32 gate / up accumulators of the dual GEMM.
// New activations (e.g. clamped SwiGLU) add a functor with the same shape;
// per-launch constants go in Params.
struct SiluAndMul {
  struct Params {};
  CUTLASS_DEVICE static float apply(float g, float u, Params const&) {
    return g / (1.0f + __expf(-g)) * u;
  }
};

template <typename Element_,           // bf16 / fp16
          typename ThreadblockShape_,  // GemmShape<M, N, K>
          typename WarpShape_,         // GemmShape<M, N, K>
          int Stages, bool Dual, bool SplitK, typename Act = SiluAndMul>
struct ExpertGemm {
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
  static constexpr bool kSplitK = SplitK;
  static constexpr int kStages = Stages;

  using MmaSingle = typename cutlass::gemm::threadblock::DefaultMma<
      Element, LayoutA, kAlignment, Element, LayoutB, kAlignment,
      ElementAccumulator, LayoutC, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, ThreadblockShape, WarpShape, InstructionShape,
      Stages, cutlass::arch::OpMultiplyAdd>::ThreadblockMma;

  using MmaDual = cutlass::gemm::threadblock::DualMmaMultistage<
      typename MmaSingle::Shape, typename MmaSingle::IteratorA,
      typename MmaSingle::SmemIteratorA, MmaSingle::kCacheOpA,
      typename MmaSingle::IteratorB, typename MmaSingle::SmemIteratorB,
      MmaSingle::kCacheOpB, typename MmaSingle::IteratorB,
      typename MmaSingle::SmemIteratorB, ElementAccumulator, LayoutC,
      typename MmaSingle::Policy, typename MmaSingle::Policy, Stages,
      cutlass::gemm::SharedMemoryClearOption::kNone>;

  using Mma = std::conditional_t<Dual, MmaDual, MmaSingle>;
  using IteratorA = typename MmaSingle::IteratorA;
  using IteratorB = typename MmaSingle::IteratorB;
  using FragmentC = typename MmaSingle::FragmentC;

  static constexpr int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;
  using OutputOp = cutlass::epilogue::thread::LinearCombination<
      Element, kAlignment, ElementAccumulator, ElementAccumulator,
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;
  using Epilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
          ThreadblockShape, typename MmaSingle::Operator, kPartitionsK,
          OutputOp, kAlignment>::Epilogue;
  using OutputIterator = typename Epilogue::OutputTileIterator;

  static constexpr int kThreads = 32 * MmaSingle::WarpCount::kCount;
  static constexpr int kAccumulators = Dual ? 2 : 1;
  // fp32 workspace per tile and K slice (all accumulators of the CTA).
  static constexpr int kTileWorkspaceFloats =
      kAccumulators * ThreadblockShape::kM * ThreadblockShape::kN;

  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };
  // All shared memory is dynamic: a static __shared__ variable would shift the
  // dynamic buffer off its 128 B alignment and break the bank-conflict-free
  // swizzled tile layouts.
  struct KernelStorage {
    SharedStorage tiles;
    int is_last_slice;
  };
  static constexpr int kSmemBytes = int(sizeof(KernelStorage));

  struct Params {
    cutlass::gemm::GemmCoord problem;
    int tiles_m, tiles_n, k_tiles;
    // Tiles [0, dp_tiles) run whole, one CTA each; every later tile is split
    // into sk_slices K slices (one CTA each) reduced by the split-K fixup.
    int dp_tiles, sk_slices, sk_k_tiles;  // k-tiles per slice
    int total_ctas;
    // Rasterisation: groups of `group` tiles along M (N sweeps inside a
    // group, the A band stays in L2) or along N (M sweeps, B band stays).
    int group;
    bool group_along_m;
    typename IteratorA::Params params_A;
    Element* ptr_A;
    typename IteratorB::Params params_B;
    Element* ptr_B0;
    Element* ptr_B1;
    typename OutputIterator::Params params_D;
    Element* ptr_D;
    typename Act::Params act;
    float* partials;  // split-K only
    int* counters;    // split-K only, zero-initialised, self-resetting
  };

  // Plans the launch for `problem`.
  //   splits >= 1: every tile is split into `splits` K slices (1 = no split).
  //   splits == 0: stream-K remainder: full waves of `slots` (= SMs x resident
  //                CTAs per SM) tiles run whole, the tiles of the last partial
  //                wave are split so that they fill one more wave.
  static Params make_params(cutlass::gemm::GemmCoord problem, int splits,
                            int slots, Element const* A, int lda,
                            Element const* B0, Element const* B1, int ldb,
                            Element* D, int ldd,
                            typename Act::Params act = {}) {
    Params p;
    p.problem = problem;
    p.tiles_m = (problem.m() + ThreadblockShape::kM - 1) / ThreadblockShape::kM;
    p.tiles_n = (problem.n() + ThreadblockShape::kN - 1) / ThreadblockShape::kN;
    p.k_tiles = (problem.k() + ThreadblockShape::kK - 1) / ThreadblockShape::kK;
    int const tiles = p.tiles_m * p.tiles_n;
    int split_tiles = tiles;  // tiles that are split
    if (splits == 0) {
      int const rem = slots > 0 ? tiles % slots : 0;
      if (tiles <= slots || rem == 0) {
        splits = tiles < slots ? slots / tiles : 1;
      } else {
        split_tiles = rem;
        splits = slots / rem;
      }
      splits = std::min(splits, 8);
    }
    if (!SplitK) splits = 1;
    // Keep >= 2 k-tiles per slice; drop empty slices.
    splits = std::max(1, std::min(splits, p.k_tiles / 2));
    int const k_per = (p.k_tiles + splits - 1) / splits;
    splits = (p.k_tiles + k_per - 1) / k_per;
    if (splits == 1) split_tiles = 0;
    p.dp_tiles = tiles - split_tiles;
    p.sk_slices = splits;
    p.sk_k_tiles = k_per;
    p.total_ctas = p.dp_tiles + split_tiles * splits;
    p.group = 8;
    // Re-read the smaller operand from L2: grouping along M keeps the A rows
    // resident and re-sweeps B (the weights), and vice versa.
    long a_bytes = long(problem.m()) * problem.k();
    long b_bytes = long(problem.n()) * problem.k() * kAccumulators;
    p.group_along_m = b_bytes <= a_bytes;
    p.params_A = typename IteratorA::Params(LayoutA(lda));
    p.ptr_A = const_cast<Element*>(A);
    p.params_B = typename IteratorB::Params(LayoutB(ldb));
    p.ptr_B0 = const_cast<Element*>(B0);
    p.ptr_B1 = const_cast<Element*>(B1);
    p.params_D = typename OutputIterator::Params(LayoutC(ldd));
    p.ptr_D = D;
    p.act = act;
    p.partials = nullptr;
    p.counters = nullptr;
    return p;
  }

  static int split_tiles(Params const& p) {
    return p.sk_slices > 1 ? p.tiles_m * p.tiles_n - p.dp_tiles : 0;
  }
  static size_t workspace_floats(Params const& p) {
    return size_t(split_tiles(p)) * p.sk_slices * kTileWorkspaceFloats;
  }
  static size_t workspace_counters(Params const& p) {
    return size_t(split_tiles(p));
  }
  static dim3 grid_dim(Params const& p) { return dim3(p.total_ctas, 1, 1); }

  // Grouped rasterisation of linear tile index t.
  CUTLASS_DEVICE
  static void tile_coord(Params const& p, int t, int& tm, int& tn) {
    int const major = p.group_along_m ? p.tiles_m : p.tiles_n;
    int const minor = p.group_along_m ? p.tiles_n : p.tiles_m;
    int const per_group = p.group * minor;
    int const first = (t / per_group) * p.group;
    int const size = min(major - first, p.group);
    int const r = t % per_group;
    int const a = first + r % size;  // along the grouped dimension
    int const b = r / size;
    tm = p.group_along_m ? a : b;
    tn = p.group_along_m ? b : a;
  }

  // Split-K fixup. Returns true in the CTA that must run the epilogue, with
  // `acc` holding the sum over all K slices. Only accumulator rows inside the
  // problem are exchanged, so small-M (decode) tiles move M rows, not kM.
  CUTLASS_DEVICE
  static bool reduce_slices(Params const& p, int tile_m, int tile_idx,
                            int slice, FragmentC (&acc)[kAccumulators],
                            int warp_idx, int lane_idx, int& is_last) {
    using WarpMma = typename MmaSingle::Operator;
    using MmaIterations = typename WarpMma::MmaIterations;
    // One float4 chunk = the 4 accumulators of one m16n8 mma: rows r, r + 8
    // with r = lane / 4. Chunks are ordered column-major over the warp's mma
    // grid (MmaTensorOp, AccumulatorsInRowMajor = false).
    static_assert(FragmentC::kElements == MmaIterations::kCount * 4,
                  "unexpected accumulator layout");
    constexpr int kChunks = FragmentC::kElements / 4;

    int const slices = p.sk_slices;
    int const warp_m =
        (warp_idx % (MmaSingle::WarpCount::kM * MmaSingle::WarpCount::kN)) %
        MmaSingle::WarpCount::kM;
    int const rows_left = p.problem.m() - tile_m * ThreadblockShape::kM -
                          warp_m * WarpShape::kM - lane_idx / 4;
    auto chunk_live = [&](int c) {
      return (c % MmaIterations::kRow) * InstructionShape::kM < rows_left;
    };

    // [tile][slice][accumulator][chunk][thread] float4: coalesced per warp.
    float4* base =
        reinterpret_cast<float4*>(p.partials) +
        size_t(tile_idx) * slices * kAccumulators * kChunks * kThreads +
        threadIdx.x;
    auto slot = [&](int slice, int a) {
      return base + size_t(slice * kAccumulators + a) * kChunks * kThreads;
    };

    CUTLASS_PRAGMA_UNROLL
    for (int a = 0; a < kAccumulators; ++a) {
      float4 const* src = reinterpret_cast<float4 const*>(&acc[a]);
      float4* dst = slot(slice, a);
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < kChunks; ++c)
        if (chunk_live(c)) __stcg(dst + c * kThreads, src[c]);
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      int arrived = atomicAdd(p.counters + tile_idx, 1);
      is_last = (arrived == slices - 1);
      if (is_last) p.counters[tile_idx] = 0;  // ready for the next launch
    }
    __syncthreads();
    if (!is_last) return false;
    __threadfence();

    // Deterministic order: slice 0, 1, ..., independent of arrival order
    // (this CTA's own slice is read back from the workspace as well). The
    // chunks of one slice are independent loads, so the fixup costs about one
    // L2 round trip per slice.
    CUTLASS_PRAGMA_UNROLL
    for (int a = 0; a < kAccumulators; ++a) acc[a].clear();
    for (int s = 0; s < slices; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int a = 0; a < kAccumulators; ++a) {
        float4* dst = reinterpret_cast<float4*>(&acc[a]);
        float4 const* src = slot(s, a);
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < kChunks; ++c) {
          if (!chunk_live(c)) continue;
          float4 v = __ldcg(src + c * kThreads);
          dst[c].x += v.x;
          dst[c].y += v.y;
          dst[c].z += v.z;
          dst[c].w += v.w;
        }
      }
    }
    return true;
  }

  CUTLASS_DEVICE
  static void run(Params const& p, KernelStorage& storage) {
    SharedStorage& smem = storage.tiles;
    int const cta = blockIdx.x;
    int tile, slice = 0;
    if (cta < p.dp_tiles) {
      tile = cta;
    } else {
      int const j = cta - p.dp_tiles;
      tile = p.dp_tiles + j / p.sk_slices;
      slice = j % p.sk_slices;
    }
    bool const split = cta >= p.dp_tiles && p.sk_slices > 1;
    int tile_m, tile_n;
    tile_coord(p, tile, tile_m, tile_n);

    int const k_begin = split ? slice * p.sk_k_tiles * ThreadblockShape::kK : 0;
    int const k_end = split ? min(p.problem.k(),
                                  k_begin + p.sk_k_tiles * ThreadblockShape::kK)
                            : p.problem.k();
    int const k_iterations =
        (k_end - k_begin + ThreadblockShape::kK - 1) / ThreadblockShape::kK;

    int const thread_idx = threadIdx.x;
    int const warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int const lane_idx = threadIdx.x % 32;

    cutlass::MatrixCoord offset_A{tile_m * ThreadblockShape::kM, k_begin};
    cutlass::MatrixCoord offset_B{k_begin, tile_n * ThreadblockShape::kN};
    IteratorA iterator_A(p.params_A, p.ptr_A, {p.problem.m(), k_end},
                         thread_idx, offset_A);
    IteratorB iterator_B0(p.params_B, p.ptr_B0, {k_end, p.problem.n()},
                          thread_idx, offset_B);

    FragmentC acc[kAccumulators];
    CUTLASS_PRAGMA_UNROLL
    for (int a = 0; a < kAccumulators; ++a) acc[a].clear();

    Mma mma(smem.main_loop, thread_idx, warp_idx, lane_idx);
    if constexpr (Dual) {
      IteratorB iterator_B1(p.params_B, p.ptr_B1, {k_end, p.problem.n()},
                            thread_idx, offset_B);
      mma(k_iterations, acc[0], acc[1], iterator_A, iterator_B0, iterator_B1,
          acc[0], acc[1]);
    } else {
      mma(k_iterations, acc[0], iterator_A, iterator_B0, acc[0]);
    }

    if constexpr (SplitK) {
      if (split) {
        if (!reduce_slices(p, tile_m, tile - p.dp_tiles, slice, acc, warp_idx,
                           lane_idx, storage.is_last_slice))
          return;
      }
    }

    if constexpr (Dual) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentC::kElements; ++i)
        acc[0][i] = Act::apply(acc[0][i], acc[1][i], p.act);
    }

    // The mainloop shared memory is reused by the epilogue.
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();
    cutlass::MatrixCoord offset_D{tile_m * ThreadblockShape::kM,
                                  tile_n * ThreadblockShape::kN};
    OutputIterator iterator_D(p.params_D, p.ptr_D, p.problem.mn(), thread_idx,
                              offset_D);
    Epilogue epilogue(smem.epilogue, thread_idx, warp_idx, lane_idx);
    epilogue(OutputOp(typename OutputOp::Params(1.0f)), iterator_D, acc[0]);
  }
};

template <typename Gemm>
__global__ void __launch_bounds__(Gemm::kThreads)
    expert_gemm_kernel(typename Gemm::Params params) {
  extern __shared__ __align__(128) char smem[];
  Gemm::run(params, *reinterpret_cast<typename Gemm::KernelStorage*>(smem));
}

}  // namespace moe_gemm
