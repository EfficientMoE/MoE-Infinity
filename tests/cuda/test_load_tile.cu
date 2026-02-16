#include <cutlass/cutlass.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/transform/threadblock/predicated_tile_iterator.h>
#include <cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h>
#include <cutlass/transform/pitch_linear_thread_map.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cutlass/util/debug.h>
#include <cutlass/util/device_dump.h>

using ShapeA = cutlass::MatrixShape<64, 64>;
using ShapeB = cutlass::MatrixShape<64, 64>;

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
  static const int Crosswise = 64;  // Typical for SM80

  using SmemLayout = std::conditional_t<
      std::is_same_v<Layout, cutlass::layout::RowMajor>,
      cutlass::layout::RowMajorTensorOpMultiplicandCongruous<ElementSize,
                                                             Crosswise>,
      cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<ElementSize,
                                                                Crosswise>>;

  using SmemIterator = cutlass::transform::threadblock::RegularTileIterator<
      ThreadblockShape, Element, SmemLayout, 1, ThreadMap, 16>;

  // Shared memory pointer
  Element* smem_ptr;

  // Constructor
  __device__ TileLoader2D(Element* smem_ptr_) : smem_ptr(smem_ptr_) {}

  // Load a tile from global to shared memory
  __device__ void operator()(
      cutlass::TensorView<Element, Layout> const& global_view,
      cutlass::MatrixCoord const& tb_offset) {
    int thread_idx = threadIdx.x;
    // int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;

    // In loader_A and loader_B
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
    smem_it.store(frag);  // This assumes total number of threads is wrap size

    __syncthreads();  // Ensure all threads have completed their loads
  }
};

template <typename ElementA, typename LayoutA, typename ElementB,
          typename LayoutB>
__global__ void load_3d_kernel(cutlass::TensorView<ElementA, LayoutA> viewA,
                               cutlass::TensorView<ElementB, LayoutB> viewB1,
                               cutlass::TensorView<ElementB, LayoutB> viewB2) {
  size_t M = viewA.extent().row();
  size_t N = viewB1.extent().column();
  size_t K = viewA.extent().column();

  // Calculate threadblock tile offsets
  cutlass::MatrixCoord tb_offset_A(int(blockIdx.y * ShapeA::kRow),
                                   int(blockIdx.z * ShapeA::kColumn));
  cutlass::MatrixCoord tb_offset_B(int(blockIdx.z * ShapeB::kRow),
                                   int(blockIdx.x * ShapeB::kColumn));

  if (tb_offset_A.row() >= M || tb_offset_A.column() >= K ||
      tb_offset_B.row() >= K || tb_offset_B.column() >= N) {
    // This assumes matrix size is multiple of tile size in all dimensions
    return;  // Skip invalid tiles
  }

  extern __shared__ __align__(16) char smem[];
  int smem_offset = 0;

  // Shared Memory Allocation
  ElementA* smem_A = reinterpret_cast<ElementA*>(smem + smem_offset);
  constexpr int size_A = ShapeA::kRow * ShapeA::kColumn * sizeof(ElementA);
  constexpr int size_B = ShapeB::kRow * ShapeB::kColumn * sizeof(ElementB);
  smem_offset += size_A;
  ElementB* smem_B1 = reinterpret_cast<ElementB*>(smem + smem_offset);
  smem_offset += size_B;
  ElementB* smem_B2 = reinterpret_cast<ElementB*>(smem + smem_offset);

  //   KERNEL_LOG_DEBUG("size_A = %d, size_B = %d, smem_A = %p, smem_B = %p\n",
  //                    size_A, size_B, static_cast<void*>(smem_A),
  //                    static_cast<void*>(smem_B));

  using ThreadMapA = cutlass::transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<ShapeA::kRow, ShapeA::kColumn>, 32,
      cutlass::layout::PitchLinearShape<8, 4>,
      128 / cutlass::sizeof_bits<ElementA>::value>;

  using ThreadMapB = cutlass::transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<ShapeB::kRow, ShapeB::kColumn>, 32,
      cutlass::layout::PitchLinearShape<8, 4>,
      128 / cutlass::sizeof_bits<ElementB>::value>;

  // Instantiate loaders
  TileLoader2D<ElementA, LayoutA, ShapeA, ThreadMapA> loader_A(smem_A);
  TileLoader2D<ElementB, LayoutB, ShapeB, ThreadMapB> loader_B1(smem_B1);
  TileLoader2D<ElementB, LayoutB, ShapeB, ThreadMapB> loader_B2(smem_B2);

  // Distinguish between warp 0 and warp 1
  if (threadIdx.y == 0 && tb_offset_A.row() < M && tb_offset_A.column() < K) {
    loader_A(viewA, tb_offset_A);
    //     int error_cnt_A = 0;
    // #pragma unroll
    //     for (int i = 0; i < size_A / sizeof(ElementA); ++i) {
    //       if (smem_A[i] != ElementA(1.0f) && threadIdx.x == 0 && blockIdx.x
    //       == 0 &&
    //           blockIdx.y == 0 && blockIdx.z == 0 && error_cnt_A < 10) {
    //         printf("smem_A[%d] = %f (expected 1.0)\n", i, float(smem_A[i]));
    //         error_cnt_A++;
    //       }
    //     }
  }
  if (threadIdx.y == 1 && tb_offset_B.row() < K && tb_offset_B.column() < N) {
    loader_B1(viewB1, tb_offset_B);
    // int error_cnt_B = 0;
    // #pragma unroll
    //     for (int i = 0; i < size_B / sizeof(ElementB); ++i) {
    //       if (smem_B1[i] != ElementB(1.0f) && threadIdx.x == 0 && blockIdx.x
    //       == 0 &&
    //           blockIdx.y == 0 && blockIdx.z == 0 && error_cnt_B < 10) {
    //         printf("smem_B[%d] = %f (expected 1.0)\n", i, float(smem_B1[i]));
    //         error_cnt_B++;
    //       }
    //     }
  }
  if (threadIdx.y == 2 && tb_offset_B.row() < K && tb_offset_B.column() < N) {
    loader_B2(viewB2, tb_offset_B);
  }
}

int main() {
  using ElementA = cutlass::bfloat16_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::bfloat16_t;
  using LayoutB = cutlass::layout::ColumnMajor;

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

  dim3 grid((N + ShapeB::kColumn - 1) / ShapeB::kColumn,
            (M + ShapeA::kRow - 1) / ShapeA::kRow,
            (K + std::max(ShapeA::kColumn, ShapeB::kRow) - 1) /
                std::max(ShapeA::kColumn, ShapeB::kRow));
  dim3 block(32, 3, 1);
  size_t smem_size = ShapeA::kRow * ShapeA::kColumn * sizeof(ElementA) +
                     ShapeB::kRow * ShapeB::kColumn * sizeof(ElementB) * 2;
  printf("smem_size = %zu bytes\n", smem_size);
  cudaFuncSetAttribute(load_3d_kernel<ElementA, LayoutA, ElementB, LayoutB>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  // CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  constexpr int iterations = 1000;
  // Launch kernel and time
  auto start_chrono = std::chrono::high_resolution_clock::now();
  cudaEventRecord(start, stream);
  for (int i = 0; i < iterations; ++i) {
    load_3d_kernel<ElementA, LayoutA, ElementB, LayoutB>
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
  auto end_chrono = std::chrono::high_resolution_clock::now();

  cudaEventSynchronize(stop);

  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);

  //   milliseconds =
  //       std::chrono::duration<float, std::milli>(end_chrono - start_chrono)
  //           .count();

  // Compute throughput
  size_t total_elements_A = size_t(M) * K;
  size_t total_elements_B = size_t(K) * N;
  size_t total_bytes_A = total_elements_A * sizeof(ElementA);
  size_t total_bytes_B = total_elements_B * sizeof(ElementB) * 2;
  size_t total_bytes = total_bytes_A + total_bytes_B;

  float time_sec = (milliseconds / 1000.0f) / iterations;
  float gb_transferred = float(total_bytes) / 1e9f;
  float throughput = gb_transferred / time_sec;

  std::cout << "3D matrix tile load completed.\n";
  std::cout << "Time: " << milliseconds << " ms\n";
  std::cout << "Data transferred: " << gb_transferred << " GB\n";
  std::cout << "Throughput: " << throughput << " GB/s\n";

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  return 0;
}
