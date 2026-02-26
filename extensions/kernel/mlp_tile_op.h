#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/transform/threadblock/predicated_tile_iterator.h>
#include <cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h>
#include <cutlass/transform/pitch_linear_thread_map.h>
#include <cuda_runtime.h>
#include <tuple>
#include <type_traits>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/warp/mma_tensor_op.h>

#include <thrust/tuple.h>

#include "common/types.h"
#include "tile_size.h"

// template <typename ArchTag>
// struct MmaOperation {
//   using ElementA = cutlass::bfloat16_t;
//   using ElementB = cutlass::bfloat16_t;
//   using ElementC = float;
//   using ElementAccumulator = float;

//   // Instruction shape for tensor cores (e.g., 16x8x16 for Ampere)
//   using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

//   // Warp-level tile shape
//   using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;

//   // MMA operator
//   using MmaTensorOp = cutlass::gemm::warp::MmaTensorOp<
//       WarpShape, ElementA, cutlass::layout::RowMajor, ElementB,
//       cutlass::layout::ColumnMajor, ElementC, cutlass::layout::RowMajor,
//       cutlass::arch::OpMultiplyAdd, ArchTag>;

//   // Fragment shapes - A and B have different shapes!
//   using FragmentA = typename MmaTensorOp::FragmentA;
//   using FragmentB = typename MmaTensorOp::FragmentB;
//   using FragmentC = typename MmaTensorOp::FragmentC;

//   // Iterator shapes for A and B are different
//   static constexpr int kFragmentARows = WarpShape::kM;
//   static constexpr int kFragmentACols = WarpShape::kK;
//   static constexpr int kFragmentBRows = WarpShape::kK;
//   static constexpr int kFragmentBCols = WarpShape::kN;
// };

template <typename Element_, typename Layout_, typename ThreadblockShape_,
          typename ThreadMap_>
struct TileLoader2D {
  using Element = Element_;
  using Layout = Layout_;
  using ThreadblockShape = ThreadblockShape_;
  using ThreadMap = ThreadMap_;

  using GmemIterator = cutlass::transform::threadblock::PredicatedTileIterator<
      ThreadblockShape, Element, Layout, 1, ThreadMap>;

  static const int ElementSize = cutlass::sizeof_bits<Element>::value;
  static const int Crosswise = 64;

  using SmemLayout = std::conditional_t<
      std::is_same_v<Layout, cutlass::layout::RowMajor>,
      cutlass::layout::RowMajorTensorOpMultiplicandCongruous<ElementSize,
                                                             Crosswise>,
      cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<ElementSize,
                                                                Crosswise>>;

  using SmemIterator = cutlass::transform::threadblock::RegularTileIterator<
      ThreadblockShape, Element, SmemLayout, 1, ThreadMap, 16>;

  Element* smem_ptr;

  __device__ TileLoader2D(Element* smem_ptr_) : smem_ptr(smem_ptr_) {}

  __device__ void operator()(
      cutlass::TensorView<Element, Layout> const& global_view,
      cutlass::MatrixCoord const& tb_offset) {
    int thread_idx = threadIdx.x;

    auto extent = global_view.extent();

    GmemIterator gmem_it(global_view.layout(), global_view.data(),
                         global_view.extent(), thread_idx, tb_offset);

    typename GmemIterator::Fragment frag;
    frag.clear();
    gmem_it.load(frag);

    cutlass::TensorRef<Element, SmemLayout> smem_ref(
        smem_ptr, SmemLayout::packed(
                      {ThreadblockShape::kRow, ThreadblockShape::kColumn}));
    SmemIterator smem_it(smem_ref, thread_idx);
    smem_it.store(frag);

    // __syncthreads();
  }
};

// Flexible multi-view tile loader class
template <int BusWidthBits, typename... ViewConfigs>
class MultiViewTileLoader {
 public:
  static constexpr size_t NumViews = sizeof...(ViewConfigs);

  // Extract types from ViewConfigs
  template <size_t I>
  using Element = typename GetNthType_t<I, ViewConfigs...>::Element;

  template <size_t I>
  using Layout = typename GetNthType_t<I, ViewConfigs...>::Layout;

  template <size_t I>
  using Shape = typename GetNthType_t<I, ViewConfigs...>::Shape;

  template <size_t I>
  using ThreadMap = typename GetNthType_t<I, ViewConfigs...>::ThreadMap;

  template <size_t I>
  using GemmCalc =
      BusAwareGemmTileCalculator<BusWidthBits, Element<I>, Element<I>, float>;

  // Calculate total shared memory size
  static constexpr size_t calculate_smem_size() {
    return calculate_smem_size_impl<0>();  // For pipelining
  }

 private:
  template <size_t I>
  static constexpr size_t calculate_smem_size_impl() {
    if constexpr (I >= NumViews) {
      return 0;
    } else {
      using CurrentShape = Shape<I>;
      using CurrentElement = Element<I>;
      return CurrentShape::kRow * CurrentShape::kColumn *
                 sizeof(CurrentElement) +
             calculate_smem_size_impl<I + 1>();
    }
  }

 public:
  // Helper to load a specific view
  template <size_t ViewIndex>
  __device__ static void load_view(
      cutlass::TensorView<Element<ViewIndex>, Layout<ViewIndex>> const& view,
      char* smem_base, uint32_t* signal, uint32_t stage_idx) {
    // // Only assign warps that are within NumViews
    // if (warp_id >= NumViews) return;  // Add this check
    // int lane_idx = threadIdx.x % 32;
    if (threadIdx.y % GemmCalc<ViewIndex>::PipelineConfig::stages != stage_idx)
      return;

    // Calculate shared memory offset for this view
    size_t smem_offset = calculate_view_offset<ViewIndex>();
    Element<ViewIndex>* smem_ptr =
        reinterpret_cast<Element<ViewIndex>*>(smem_base + smem_offset);

    // Calculate threadblock offset based on view configuration
    cutlass::MatrixCoord tb_offset =
        GetNthType_t<ViewIndex, ViewConfigs...>::calculate_offset();

    // Create and use loader
    using Loader = TileLoader2D<Element<ViewIndex>, Layout<ViewIndex>,
                                Shape<ViewIndex>, ThreadMap<ViewIndex>>;
    Loader loader(smem_ptr);
    loader(view, tb_offset);

    __syncwarp();  // Ensure all threads have loaded their data
    if (signal == nullptr) return;
    if (threadIdx.x == 0) {
      atomicAdd(&signal[stage_idx], 1);
      printf(
          "Block(%d,%d) Warp %d loaded view %zu at offset %zu; signal[%d] = "
          "%u\n",
          blockIdx.x, blockIdx.y, threadIdx.y, ViewIndex, smem_offset,
          stage_idx, signal[stage_idx]);
    }
  }

  // Get the auto-calculated shape for a specific view
  template <size_t ViewIndex>
  static constexpr auto get_shape() {
    return Shape<ViewIndex>{};
  }

  static void print_configuration() {
    printf("=== Auto-Sized Multi-View Loader Configuration ===\n");
    printf("Bus Width: %d bits (%d bytes/cycle)\n", BusWidthBits,
           BusWidthBits / 8);
    printf("Number of Views: %zu\n", NumViews);
    printf("Total Shared Memory: %zu bytes\n", calculate_smem_size());
    print_view_configs<0>();
    printf("================================================\n");
  }

 private:
 private:
  template <size_t I>
  static void print_view_configs() {
    if constexpr (I < NumViews) {
      printf("View %zu: %d-bit elements, %dx%d tile (%zu bytes)\n", I,
             cutlass::sizeof_bits<Element<I>>::value, Shape<I>::kRow,
             Shape<I>::kColumn,
             Shape<I>::kRow * Shape<I>::kColumn * sizeof(Element<I>));
      print_view_configs<I + 1>();
    }
  }

  template <size_t ViewIndex>
  static constexpr size_t calculate_view_offset() {
    if constexpr (ViewIndex == 0) {
      return 0;
    } else {
      return calculate_view_offset<ViewIndex - 1>() +
             Shape<ViewIndex - 1>::kRow * Shape<ViewIndex - 1>::kColumn *
                 sizeof(Element<ViewIndex - 1>);
    }
  }
};

// View configuration helper
template <typename Element_, typename Layout_, typename Shape_,
          typename ThreadMap_>
struct ViewConfig {
  using Element = Element_;
  using Layout = Layout_;
  using Shape = Shape_;
  using ThreadMap = ThreadMap_;

  // Virtual method to be specialized for different matrix positions
  __device__ static cutlass::MatrixCoord calculate_offset() {
    return cutlass::MatrixCoord(0, 0);
  }
};

// Specialized view configurations for A and B matrices
template <typename Element_, typename Layout_, typename Shape_,
          typename ThreadMap_>
struct ViewConfigA : ViewConfig<Element_, Layout_, Shape_, ThreadMap_> {
  __device__ static cutlass::MatrixCoord calculate_offset() {
    return cutlass::MatrixCoord(int(blockIdx.y * Shape_::kRow),
                                int(blockIdx.z * Shape_::kColumn));
  }
};

template <typename Element_, typename Layout_, typename Shape_,
          typename ThreadMap_>
struct ViewConfigB : ViewConfig<Element_, Layout_, Shape_, ThreadMap_> {
  __device__ static cutlass::MatrixCoord calculate_offset() {
    return cutlass::MatrixCoord(int(blockIdx.z * Shape_::kRow),
                                int(blockIdx.x * Shape_::kColumn));
  }
};

// Helper function templates to load each view by index
template <typename MultiLoader, size_t Index, typename FirstView,
          typename... RestViews>
__device__ void load_each(char* smem, uint32_t* signal, uint32_t stage_idx,
                          FirstView& first, RestViews&... rest) {
  if (threadIdx.x == 0) {
    printf("Loading view %zu at stage %u\n", Index, stage_idx);
  }
  MultiLoader::template load_view<Index>(first, smem, signal, stage_idx);
  if constexpr (sizeof...(RestViews) > 0) {
    load_each<MultiLoader, Index + 1>(smem, signal, stage_idx, rest...);
  }
}

// Base case for recursion
template <typename MultiLoader, size_t Index>
__device__ void load_each(char* smem, uint32_t* signal, uint32_t stage_idx) {}

// // Kernel that uses the flexible loader
// template <typename MultiLoader, typename... Views>
// __global__ void flexible_load_kernel(Views... views) {
//   extern __shared__ __align__(16) char smem[];
//   load_each<MultiLoader, 0>(smem, views...);
// }

// Add this helper function before the kernel
template <typename MultiLoader, typename TupleType, size_t... Is>
__device__ void apply_load_each_impl(char* smem, uint32_t* signal,
                                     uint32_t stage_idx, TupleType&& t,
                                     std::index_sequence<Is...>) {
  load_each<MultiLoader, 0>(smem, signal, stage_idx, thrust::get<Is>(t)...);
  // (load_each<MultiLoader, Is>(smem, signal, stage_idx, thrust::get<Is>(t)),
  //  ...);
}

template <typename MultiLoader, typename TupleType>
__device__ void apply_load_each(char* smem, uint32_t* signal,
                                uint32_t stage_idx, TupleType&& t) {
  apply_load_each_impl<MultiLoader>(
      smem, signal, stage_idx, std::forward<TupleType>(t),
      std::make_index_sequence<std::tuple_size_v<std::decay_t<TupleType>>>{});
}
