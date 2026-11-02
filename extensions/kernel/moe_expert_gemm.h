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
// by every launch on the same stream without a memset.

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
          int Stages, int SwizzleN, bool Dual, bool SplitK,
          typename Act = SiluAndMul>
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

  using Swizzle =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleN>;
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
    cutlass::gemm::GemmCoord grid;  // tiles m, n and K slices
    int swizzle_log_tile;
    int gemm_k_size;  // K elements per slice (multiple of ThreadblockShape::kK)
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

  // Plans the launch for `problem` split into `splits` K slices. Empty K
  // slices are dropped, so the returned grid.k() may be smaller.
  static Params make_params(cutlass::gemm::GemmCoord problem, int splits,
                            Element const* A, int lda, Element const* B0,
                            Element const* B1, int ldb, Element* D, int ldd,
                            float* partials, int* counters,
                            typename Act::Params act = {}) {
    Params p;
    p.problem = problem;
    if (!SplitK) splits = 1;
    int k_tiles =
        (problem.k() + ThreadblockShape::kK - 1) / ThreadblockShape::kK;
    splits = splits < 1 ? 1 : (splits > k_tiles ? k_tiles : splits);
    int k_tiles_per_split = (k_tiles + splits - 1) / splits;
    p.gemm_k_size = k_tiles_per_split * ThreadblockShape::kK;
    splits = (k_tiles + k_tiles_per_split - 1) / k_tiles_per_split;
    Swizzle swizzle;
    p.grid = swizzle.get_tiled_shape(
        problem,
        {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
        splits);
    p.swizzle_log_tile = swizzle.get_log_tile(p.grid);
    p.params_A = typename IteratorA::Params(LayoutA(lda));
    p.ptr_A = const_cast<Element*>(A);
    p.params_B = typename IteratorB::Params(LayoutB(ldb));
    p.ptr_B0 = const_cast<Element*>(B0);
    p.ptr_B1 = const_cast<Element*>(B1);
    p.params_D = typename OutputIterator::Params(LayoutC(ldd));
    p.ptr_D = D;
    p.act = act;
    p.partials = partials;
    p.counters = counters;
    return p;
  }

  static size_t workspace_floats(Params const& p) {
    return p.grid.k() > 1 ? size_t(p.grid.m()) * p.grid.n() * p.grid.k() *
                                kTileWorkspaceFloats
                          : 0;
  }
  static size_t workspace_counters(Params const& p) {
    return p.grid.k() > 1 ? size_t(p.grid.m()) * p.grid.n() : 0;
  }
  static dim3 grid_dim(Params const& p) {
    return Swizzle().get_grid_shape(p.grid);
  }

  // Split-K fixup. Returns true in the CTA that must run the epilogue, with
  // `acc` holding the sum over all K slices. Only accumulator rows inside the
  // problem are exchanged, so small-M (decode) tiles move M rows, not kM.
  CUTLASS_DEVICE
  static bool reduce_slices(Params const& p,
                            cutlass::gemm::GemmCoord const& tile,
                            FragmentC (&acc)[kAccumulators], int warp_idx,
                            int lane_idx, int& is_last) {
    using WarpMma = typename MmaSingle::Operator;
    using MmaIterations = typename WarpMma::MmaIterations;
    // One float4 chunk = the 4 accumulators of one m16n8 mma: rows r, r + 8
    // with r = lane / 4. Chunks are ordered column-major over the warp's mma
    // grid (MmaTensorOp, AccumulatorsInRowMajor = false).
    static_assert(FragmentC::kElements == MmaIterations::kCount * 4,
                  "unexpected accumulator layout");
    constexpr int kChunks = FragmentC::kElements / 4;

    int const tile_idx = tile.m() + tile.n() * p.grid.m();
    int const slices = p.grid.k();
    int const warp_m =
        (warp_idx % (MmaSingle::WarpCount::kM * MmaSingle::WarpCount::kN)) %
        MmaSingle::WarpCount::kM;
    int const rows_left = p.problem.m() - tile.m() * ThreadblockShape::kM -
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
      float4* dst = slot(tile.k(), a);
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
    Swizzle swizzle;
    cutlass::gemm::GemmCoord tile = swizzle.get_tile_offset(p.swizzle_log_tile);
    if (tile.m() >= p.grid.m() || tile.n() >= p.grid.n()) return;

    int const k_begin = tile.k() * p.gemm_k_size;
    int const k_end = min(p.problem.k(), k_begin + p.gemm_k_size);
    int const k_iterations =
        (k_end - k_begin + ThreadblockShape::kK - 1) / ThreadblockShape::kK;

    int const thread_idx = threadIdx.x;
    int const warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int const lane_idx = threadIdx.x % 32;

    cutlass::MatrixCoord offset_A{tile.m() * ThreadblockShape::kM, k_begin};
    cutlass::MatrixCoord offset_B{k_begin, tile.n() * ThreadblockShape::kN};
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
      if (p.grid.k() > 1) {
        if (!reduce_slices(p, tile, acc, warp_idx, lane_idx,
                           storage.is_last_slice))
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
    cutlass::MatrixCoord offset_D{tile.m() * ThreadblockShape::kM,
                                  tile.n() * ThreadblockShape::kN};
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
