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
#include <cuda/pipeline>
#include <cuda/barrier>

#include "kernel/tile_size.h"
#include "kernel/mlp_tile_op.h"  // Include the header with BusAwareGemmTileCalculator

// GEMM-style kernel with K iteration inside
template <typename MultiLoader, typename... Views>
__global__ void gemm_auto_load_kernel(Views... views) {
  extern __shared__ __align__(16) char smem[];

  // Get problem dimensions from first view (assuming all views have consistent
  // dims)
  auto& view0 = std::get<0>(std::tie(views...));
  auto& view1 = std::get<1>(std::tie(views...));
  const uint32_t M = view0.extent().row();     // Assuming A is MxK
  const uint32_t K = view0.extent().column();  // Assuming A is MxK
  const uint32_t N = view1.extent().row();     // Assuming B is NxK

  using ShapeA = typename MultiLoader::template Shape<0>;
  using ShapeB = typename MultiLoader::template Shape<1>;
  // Bounds check
  if (blockIdx.x * ShapeB::kColumn >= N || blockIdx.y * ShapeA::kRow >= M)
    return;

  // Get tile K dimension from loader
  using ShapeA = typename MultiLoader::template Shape<0>;
  constexpr int TILE_K = ShapeA::kColumn;

  using GemmCalc = typename MultiLoader::GemmCalc<0>;
  constexpr int NUM_STAGES = GemmCalc::PipelineConfig::stages;
  constexpr int WARPS_PER_GROUP = 9;  // Assuming 9 warps per group

  __shared__ __align__(16) uint32_t signal[NUM_STAGES];

  // Multi-stage buffer setup
  const size_t stage_size = MultiLoader::calculate_smem_size();
  char* smem_stages[NUM_STAGES];
  for (int i = 0; i < NUM_STAGES; ++i) {
    smem_stages[i] = smem + i * stage_size;
  }

  // Create pipeline barriers per warp
  __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, NUM_STAGES>
      pipeline_state;

  auto load_stage = [&](uint32_t stage_idx, int k_offset) {
    if (k_offset >= K) return;
    // if (threadIdx.y >= MultiLoader::NumViews) return;

    int tile_k = min(TILE_K, K - k_offset);
    char* stage_ptr = smem_stages[stage_idx % NUM_STAGES];

    auto make_view = [&](auto& view, int view_idx) {
      using ViewElement = typename std::decay_t<decltype(view)>::Element;
      using ViewLayout = typename std::decay_t<decltype(view)>::Layout;

      if (view_idx == 0) {  // A
        size_t offset = blockIdx.y * ShapeA::kRow * K + k_offset;
        return cutlass::TensorView<ViewElement, ViewLayout>(
            view.data() + offset, view.layout(),
            {min(ShapeA::kRow, M - blockIdx.y * ShapeA::kRow), tile_k});
      } else {  // B
        size_t offset = k_offset * N + blockIdx.x * ShapeB::kColumn;
        return cutlass::TensorView<ViewElement, ViewLayout>(
            view.data() + offset, view.layout(),
            {tile_k, min(ShapeB::kColumn, N - blockIdx.x * ShapeB::kColumn)});
      }
    };

    // Apply K-tiling to each view
    int idx = 0;
    auto tiled_views = thrust::make_tuple(make_view(views, idx++)...);

    if (threadIdx.x == 0) {
      printf("Loading stage %d, Wrap %d, K offset %d, tile size %dx%d\n",
             stage_idx, threadIdx.y, k_offset, ShapeA::kRow, ShapeB::kColumn);
    }

    // Load tiles for current K iteration
    apply_load_each<MultiLoader>(smem, signal, stage_idx % NUM_STAGES,
                                 tiled_views);
  };

  // Pipeline state
  int read_stage = 0;
  int write_stage = 0;
  int compute_stage = 0;

  uint32_t stage_idx = threadIdx.y % NUM_STAGES;
  int lane_idx = threadIdx.x % 32;  // Assuming 32 threads per warp

  // // Prologue: fill pipeline
  // for (; read_stage < NUM_STAGES - 1 && read_stage * TILE_K < K;
  // ++read_stage) {
  //   load_stage(read_stage, read_stage * TILE_K);
  // }
  load_stage(read_stage++, 0);

  // Wait for pipeline 0 to be ready
  while (MultiLoader::NumViews > signal[stage_idx] && 0 == stage_idx) {
    // just busy wait
  }

  // Main loop for compute stage
  for (int k_start = 0; k_start < K; k_start += TILE_K) {
    // Start loading future stage
    int next_k = k_start + (NUM_STAGES - 1) * TILE_K;
    if (next_k < K) {
      load_stage(read_stage, next_k);
    }
    int current_stage = k_start % NUM_STAGES;

    // Wait for current read stage to finish
    while (MultiLoader::NumViews > signal[current_stage] &&
           stage_idx == current_stage) {
      // just busy wait
    }

    if (lane_idx == 0 && stage_idx == current_stage) {
      atomicSub(&signal[current_stage], 1);
    }

    // Compute on current stage
    // Actual computation would use smem_stages[compute_stage % NUM_STAGES]

    // Advance pipeline
    __syncthreads();
    read_stage++;
    write_stage++;
    compute_stage++;
  }
}

// Example usage
int main() {
  // Define bus width (384 bits for RTX 4090)
  constexpr int BusWidth = 384;

  using ElementA = cutlass::bfloat16_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::bfloat16_t;
  using LayoutB = cutlass::layout::ColumnMajor;

  // Calculate auto-sized shapes using GEMM calculator
  using GemmCalc =
      BusAwareGemmTileCalculator<BusWidth, ElementA, ElementB, float>;

  printf("Auto-calculated tile sizes for %d-bit bus:\n", BusWidth);
  printf("  GEMM tile (A/B with BF16): %dx%dx%d\n",
         GemmCalc::OptimizedTiles::kM, GemmCalc::OptimizedTiles::kN,
         GemmCalc::OptimizedTiles::kK);
  printf("  Pipeline stages: %d\n", GemmCalc::PipelineConfig::stages);
  printf("  Cluster shape: %dx%dx%d\n", GemmCalc::ClusterShape::kM,
         GemmCalc::ClusterShape::kN, GemmCalc::ClusterShape::kK);

  // Auto-sized thread maps based on GEMM tile dimensions
  using ThreadMapA = cutlass::transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<GemmCalc::OptimizedTiles::kM,
                                        GemmCalc::OptimizedTiles::kK>,
      32, cutlass::layout::PitchLinearShape<8, 4>,
      128 / cutlass::sizeof_bits<ElementA>::value>;

  using ThreadMapB = cutlass::transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<GemmCalc::OptimizedTiles::kN,
                                        GemmCalc::OptimizedTiles::kK>,
      32, cutlass::layout::PitchLinearShape<8, 4>,
      128 / cutlass::sizeof_bits<ElementB>::value>;

  // Define view configurations
  using ViewA = ViewConfigA<ElementA, LayoutA,
                            cutlass::MatrixShape<GemmCalc::OptimizedTiles::kM,
                                                 GemmCalc::OptimizedTiles::kK>,
                            ThreadMapA>;
  using ViewB1 = ViewConfigB<ElementB, LayoutB,
                             cutlass::MatrixShape<GemmCalc::OptimizedTiles::kN,
                                                  GemmCalc::OptimizedTiles::kK>,
                             ThreadMapB>;
  using ViewB2 = ViewConfigB<ElementB, LayoutB,
                             cutlass::MatrixShape<GemmCalc::OptimizedTiles::kN,
                                                  GemmCalc::OptimizedTiles::kK>,
                             ThreadMapB>;

  // Create auto-sized multi-view loader
  using Loader = MultiViewTileLoader<BusWidth, ViewA, ViewB1, ViewB2>;

  // Print configuration
  Loader::print_configuration();

  constexpr int M = 128, N = 768, K = 2048;
  cutlass::HostTensor<ElementA, LayoutA> A({M, K});
  cutlass::HostTensor<ElementB, LayoutB> B1({N, K});
  cutlass::HostTensor<ElementB, LayoutB> B2({N, K});

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
                                                {N, K});
  cutlass::TensorView<ElementB, LayoutB> viewB2(B2.device_data(), B2.layout(),
                                                {N, K});

  // Get auto-sized shapes
  using ShapeA = typename Loader::template Shape<0>;
  using ShapeB = typename Loader::template Shape<1>;

  size_t smem_size =
      Loader::calculate_smem_size() * GemmCalc::PipelineConfig::stages;

  // Calculate 2D grid (no Z dimension for K)
  dim3 grid =
      ThreadBlockAutoTuner<BusWidth, ElementA>::get_optimal_grid_dims(M, N, K);
  dim3 block(32, 9, 1);  // Fixed block size for simplicity
  // Auto-tune thread block configuration
  // dim3 block =
  //     ThreadBlockAutoTuner<BusWidth, ElementA>::get_optimal_block_dims(M, N,
  //     K);
  // block.y = std::max((uint32_t)3, block.y);  // Ensure at least one warp in Y
  printf("\nKernel launch configuration:\n");
  printf("  Problem size: %dx%dx%d\n", M, N, K);
  printf("  Tile size: %dx%dx%d\n", ShapeA::kRow, ShapeB::kColumn,
         ShapeA::kColumn);
  printf("  Grid: (%d, %d) - 2D grid, K iteration in kernel\n", grid.x, grid.y);
  printf("  Block: (%d, %d) - %d warps total\n", block.x, block.y, block.y);
  printf("  Shared memory: %zu bytes\n", smem_size);
  printf("  K iterations per block: %d\n",
         (K + ShapeA::kColumn - 1) / ShapeA::kColumn);

  cudaFuncSetAttribute(
      gemm_auto_load_kernel<Loader, cutlass::TensorView<ElementA, LayoutA>,
                            cutlass::TensorView<ElementB, LayoutB>,
                            cutlass::TensorView<ElementB, LayoutB>>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  // Timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  constexpr int iterations = 1000;

  // Warm-up
  gemm_auto_load_kernel<Loader>
      <<<grid, block, smem_size, stream>>>(viewA, viewB1, viewB2);
  cudaStreamSynchronize(stream);

  cudaEventRecord(start, stream);
  for (int i = 0; i < iterations; ++i) {
    gemm_auto_load_kernel<Loader>
        <<<grid, block, smem_size, stream>>>(viewA, viewB1, viewB2);

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
      return -1;
    }
  }
  cudaEventRecord(stop, stream);

  cudaEventSynchronize(stop);

  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);

  // Compute throughput - corrected for actual tile loading
  int tiles_m = (M + ShapeA::kRow - 1) / ShapeA::kRow;
  int tiles_n = (N + ShapeB::kColumn - 1) / ShapeB::kColumn;
  int tiles_k = (K + ShapeA::kColumn - 1) / ShapeA::kColumn;

  // Each threadblock loads one A tile and two B tiles per K iteration
  size_t bytes_per_tile_a = ShapeA::kRow * ShapeA::kColumn * sizeof(ElementA);
  size_t bytes_per_tile_b = ShapeB::kRow * ShapeB::kColumn * sizeof(ElementB);

  // Total data movement
  size_t total_a_tiles = tiles_m * tiles_k;
  size_t total_b_tiles = tiles_n * tiles_k * 2;  // B1 and B2
  size_t total_bytes =
      total_a_tiles * bytes_per_tile_a + total_b_tiles * bytes_per_tile_b;

  float time_sec = (milliseconds / 1000.0f) / iterations;
  float gb_transferred = float(total_bytes) / 1e9f;
  float throughput = gb_transferred / time_sec;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int memory_clock_khz = prop.memoryClockRate;  // in kHz

  // Calculate efficiency
  float theoretical_bandwidth =
      (BusWidth / 8) * 2.0 * memory_clock_khz / 1e6;  // GB/s
  float efficiency = (throughput / theoretical_bandwidth) * 100.0f;

  printf("\n=== Performance Results ===\n");
  printf("Time: %.3f ms total, %.3f μs per iteration\n", milliseconds,
         milliseconds * 1000.0f / iterations);
  printf("Data transferred: %.3f GB\n", gb_transferred);
  printf("Throughput: %.2f GB/s\n", throughput);
  printf("Theoretical peak: %.2f GB/s\n", theoretical_bandwidth);
  printf("Efficiency: %.1f%%\n", efficiency);

  // Cleanup
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);

  return 0;
}
