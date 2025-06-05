#include <cutlass/cutlass.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/transform/threadblock/predicated_tile_iterator.h>
#include <cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h>
#include <cutlass/transform/pitch_linear_thread_map.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cuda_runtime.h>
#include <iostream>
#include <tuple>
#include <type_traits>

#include "common/types.h"

// TileLoader2D remains the same
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

    __syncthreads();
  }
};

// Flexible multi-view tile loader class
template <typename... ViewConfigs>
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

  // Calculate total shared memory size
  static constexpr size_t calculate_smem_size() {
    return calculate_smem_size_impl<0>();
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
      char* smem_base, int warp_id) {
    if (threadIdx.y != warp_id) return;

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

    //     int err_cnt = 0;
    // #pragma unroll
    //     for (int i = 0; i < Shape<ViewIndex>::kRow *
    //     Shape<ViewIndex>::kColumn; ++i) {
    //       if (smem_ptr[i] != Element<ViewIndex>(1.0f) && threadIdx.x == 0 &&
    //       blockIdx.x == 0 &&
    //           blockIdx.y == 0 && blockIdx.z == 0 && err_cnt < 10) {
    //         printf("smem[%d] = %f (expected 1.0)\n", i, float(smem_ptr[i]));
    //         err_cnt++;
    //       }
    //     }
  }

 private:
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

  template <size_t ViewIndex>
  static constexpr size_t calculate_view_size() {
    return Shape<ViewIndex>::kRow * Shape<ViewIndex>::kColumn *
           sizeof(Element<ViewIndex>);
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
__device__ void load_each(char* smem, FirstView& first, RestViews&... rest) {
  MultiLoader::template load_view<Index>(first, smem, Index);
  if constexpr (sizeof...(RestViews) > 0) {
    load_each<MultiLoader, Index + 1>(smem, rest...);
  }
}

// Base case for recursion
template <typename MultiLoader, size_t Index>
__device__ void load_each(char* smem) {}

// Kernel that uses the flexible loader
template <typename MultiLoader, typename... Views>
__global__ void flexible_load_kernel(Views... views) {
  extern __shared__ __align__(16) char smem[];
  load_each<MultiLoader, 0>(smem, views...);
}

// Example usage
int main() {
  using ElementA = cutlass::bfloat16_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::bfloat16_t;
  using LayoutB = cutlass::layout::ColumnMajor;

  using ShapeA = cutlass::MatrixShape<64, 64>;
  using ShapeB = cutlass::MatrixShape<64, 64>;

  using ThreadMapA = cutlass::transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<ShapeA::kRow, ShapeA::kColumn>, 32,
      cutlass::layout::PitchLinearShape<8, 4>,
      128 / cutlass::sizeof_bits<ElementA>::value>;

  using ThreadMapB = cutlass::transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<ShapeB::kRow, ShapeB::kColumn>, 32,
      cutlass::layout::PitchLinearShape<8, 4>,
      128 / cutlass::sizeof_bits<ElementB>::value>;

  // Define view configurations
  using ViewA = ViewConfigA<ElementA, LayoutA, ShapeA, ThreadMapA>;
  using ViewB1 = ViewConfigB<ElementB, LayoutB, ShapeB, ThreadMapB>;
  using ViewB2 = ViewConfigB<ElementB, LayoutB, ShapeB, ThreadMapB>;

  // Create multi-view loader
  using Loader = MultiViewTileLoader<ViewA, ViewB1, ViewB2>;

  constexpr int M = 128, N = 768, K = 2048;
  cutlass::HostTensor<ElementA, LayoutA> A({M, K});
  cutlass::HostTensor<ElementB, LayoutB> B1({K, N});
  cutlass::HostTensor<ElementB, LayoutB> B2({K, N});

  cutlass::reference::host::TensorFill(A.host_view(), ElementA(1.0f));
  cutlass::reference::host::TensorFill(B1.host_view(), ElementB(1.0f));
  cutlass::reference::host::TensorFill(B2.host_view(), ElementB(1.0f));

  A.sync_device();
  B1.sync_device();
  B2.sync_device();

  // Create views
  cutlass::TensorView<ElementA, LayoutA> viewA(A.device_data(), A.layout(),
                                               {M, K});
  cutlass::TensorView<ElementB, LayoutB> viewB1(B1.device_data(), B1.layout(),
                                                {K, N});
  cutlass::TensorView<ElementB, LayoutB> viewB2(B2.device_data(), B2.layout(),
                                                {K, N});

  // Calculate grid and block dimensions
  dim3 grid((N + ShapeB::kColumn - 1) / ShapeB::kColumn,
            (M + ShapeA::kRow - 1) / ShapeA::kRow,
            (K + std::max(ShapeA::kColumn, ShapeB::kRow) - 1) /
                std::max(ShapeA::kColumn, ShapeB::kRow));
  dim3 block(32, Loader::NumViews,
             1);  // Automatically adjust to number of views

  size_t smem_size = Loader::calculate_smem_size();
  printf("Number of views: %zu\n", Loader::NumViews);
  printf("Shared memory size: %zu bytes\n", smem_size);

  cudaFuncSetAttribute(
      flexible_load_kernel<Loader, cutlass::TensorView<ElementA, LayoutA>,
                           cutlass::TensorView<ElementB, LayoutB>,
                           cutlass::TensorView<ElementB, LayoutB>>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  // CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  constexpr int iterations = 2000;

  cudaEventRecord(start, stream);
  for (int i = 0; i < iterations; ++i) {
    flexible_load_kernel<Loader>
        <<<grid, block, smem_size, stream>>>(viewA, viewB1, viewB2);

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
      return -1;
    }
  }
  cudaEventRecord(stop, stream);

  cudaEventSynchronize(stop);
  cudaStreamSynchronize(stream);

  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);

  // Compute throughput
  size_t total_elements_A = size_t(M) * K;
  size_t total_elements_B = size_t(K) * N;
  size_t total_bytes_A = total_elements_A * sizeof(ElementA);
  size_t total_bytes_B = total_elements_B * sizeof(ElementB) * 2;
  size_t total_bytes = total_bytes_A + total_bytes_B;

  float time_sec = (milliseconds / 1000.0f) / iterations;
  float gb_transferred = float(total_bytes) / 1e9f;
  float throughput = gb_transferred / time_sec;

  std::cout << "Flexible multi-view tile load completed.\n";
  std::cout << "Time: " << milliseconds << " ms\n";
  std::cout << "Data transferred: " << gb_transferred << " GB\n";
  std::cout << "Throughput: " << throughput << " GB/s\n";

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  return 0;
}
