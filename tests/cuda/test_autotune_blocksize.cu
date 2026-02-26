#include <cutlass/cutlass.h>
#include <cutlass/layout/matrix.h>
#include <algorithm>
#include <vector>
#include <numeric>
#include <cutlass/transform/threadblock/predicated_tile_iterator.h>
#include <cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h>
#include <cutlass/transform/pitch_linear_thread_map.h>

#include "kernel/tile_size.h"

// Example usage
int main() {
  constexpr int BusWidth = 5120;
  constexpr int M = 2048, N = 2048, K = 2048;

  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::bfloat16_t;

  // Get GEMM tile sizes
  using GemmCalc =
      BusAwareGemmTileCalculator<BusWidth, ElementA, ElementB, float>;

  // Run autotuner
  using Tuner = ThreadBlockAutoTuner<BusWidth, ElementA>;
  Tuner::print_autotuning_result(M, N, K, GemmCalc::OptimizedTiles::kM,
                                 GemmCalc::OptimizedTiles::kN,
                                 GemmCalc::OptimizedTiles::kK);

  // Get optimal block dimensions
  auto block_dims = Tuner::get_optimal_block_dims(M, N, K);

  printf("\nRecommended kernel launch:\n");
  printf("  <<<grid, dim3(%d, %d, 1), smem_size>>>\n", block_dims.x,
         block_dims.y);

  return 0;
}
