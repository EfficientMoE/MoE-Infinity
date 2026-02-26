#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <type_traits>

#include "common/constant.h"
#include "common/types.h"

// Bus width aware GEMM tile calculator
template <int BusWidthBits, typename ElementA, typename ElementB,
          typename ElementC, int TargetCycles = 4>
struct BusAwareGemmTileCalculator {
  // Memory bandwidth parameters
  static constexpr int bytes_per_cycle = BusWidthBits / 8;
  static constexpr int target_cycles = TargetCycles;

  // Element sizes
  static constexpr int size_A = cutlass::sizeof_bits<ElementA>::value / 8;
  static constexpr int size_B = cutlass::sizeof_bits<ElementB>::value / 8;
  static constexpr int size_C = cutlass::sizeof_bits<ElementC>::value / 8;

  // Total bytes we can move in target cycles
  static constexpr int total_bandwidth_bytes = bytes_per_cycle * target_cycles;

  // Scale factor based on bus width (normalized to 256-bit baseline)
  static constexpr int bus_scale_factor = BusWidthBits / 256;
  static constexpr int bus_sqrt_scale = ConstexprSqrt<bus_scale_factor>::value;

  // Base tile dimensions for 256-bit bus
  struct BaseTileSizes {
    static constexpr int kM = 64;
    static constexpr int kN = 64;
    static constexpr int kK = 32;
  };

  // Scale tiles based on bus width
  struct ScaledGemmTile {
    // Scale M and N with square root of bus scaling to maintain aspect ratio
    // Scale K less aggressively to maintain data reuse
    static constexpr int kM_raw = BaseTileSizes::kM * bus_sqrt_scale;
    static constexpr int kN_raw = BaseTileSizes::kN * bus_sqrt_scale;
    static constexpr int kK_raw =
        BaseTileSizes::kK * (bus_sqrt_scale + 1) / 2;  // Scale K by half

    // Round to tensor core friendly sizes
    static constexpr int kM = RoundToMultiple<kM_raw, 16>::value;
    static constexpr int kN = RoundToMultiple<kN_raw, 16>::value;
    static constexpr int kK = RoundToMultiple<kK_raw, 16>::value;

    // Calculate actual memory usage
    static constexpr int elements_A = kM * kK;
    static constexpr int elements_B = kK * kN;
    static constexpr int elements_C = kM * kN;

    static constexpr int bytes_A = elements_A * size_A;
    static constexpr int bytes_B = elements_B * size_B;
    static constexpr int bytes_C = elements_C * size_C;

    // Total bytes for one tile computation (read A, B and write C)
    static constexpr int total_bytes = bytes_A + bytes_B + bytes_C;

    // Cycles needed to transfer this data
    static constexpr int cycles_needed =
        (total_bytes + bytes_per_cycle - 1) / bytes_per_cycle;

    // Check if we fit within bandwidth budget
    static constexpr bool fits_bandwidth = cycles_needed <= target_cycles;
  };

  // Architecture-specific optimized tiles
  struct OptimizedTiles {
    // For narrow bus (consumer GPUs): prioritize square tiles
    struct NarrowBus {
      static constexpr bool is_narrow = BusWidthBits <= 384;
      static constexpr int kM = is_narrow ? 64 : ScaledGemmTile::kM;
      static constexpr int kN = is_narrow ? 64 : ScaledGemmTile::kN;
      static constexpr int kK = is_narrow ? 32 : ScaledGemmTile::kK;
    };

    // For wide bus (HBM GPUs): can afford larger tiles
    struct WideBus {
      static constexpr bool is_wide = BusWidthBits >= 4096;
      static constexpr int kM =
          is_wide ? RoundToMultiple<ScaledGemmTile::kM, 64>::value
                  : ScaledGemmTile::kM;
      static constexpr int kN =
          is_wide ? RoundToMultiple<ScaledGemmTile::kN, 64>::value
                  : ScaledGemmTile::kN;
      static constexpr int kK =
          is_wide ? RoundToMultiple<ScaledGemmTile::kK, 32>::value
                  : ScaledGemmTile::kK;
    };

    // Choose based on bus width
    static constexpr int kM =
        WideBus::is_wide
            ? WideBus::kM
            : (NarrowBus::is_narrow ? NarrowBus::kM : ScaledGemmTile::kM);
    static constexpr int kN =
        WideBus::is_wide
            ? WideBus::kN
            : (NarrowBus::is_narrow ? NarrowBus::kN : ScaledGemmTile::kN);
    static constexpr int kK =
        WideBus::is_wide
            ? WideBus::kK
            : (NarrowBus::is_narrow ? NarrowBus::kK : ScaledGemmTile::kK);
  };

  // Threadblock clusters for CUTLASS 3.x
  struct ClusterShape {
    // Larger clusters for wider memory interfaces
    static constexpr int kM = BusWidthBits >= 4096 ? 2 : 1;
    static constexpr int kN = BusWidthBits >= 4096 ? 2 : 1;
    static constexpr int kK = 1;  // K dimension clustering less beneficial
  };

  // Pipeline stages based on bandwidth
  struct PipelineConfig {
    // More stages for wider interfaces to hide latency
    static constexpr int stages = BusWidthBits >= 4096  ? 4
                                  : BusWidthBits >= 384 ? 3
                                                        : 2;
  };

  // Warp arrangement
  struct WarpArrangement {
    // More warps for larger tiles
    static constexpr int warps_m = OptimizedTiles::kM / 32;
    static constexpr int warps_n = OptimizedTiles::kN / 32;
    static constexpr int total_warps = warps_m * warps_n;

    // Ensure we don't exceed SM warp limits
    static constexpr int max_warps = 16;  // Typical limit
    static constexpr bool valid = total_warps <= max_warps;
  };

  static void print_config() {
    printf("=== Bus-Aware GEMM Tile Configuration ===\n");
    printf("Memory Bus: %d bits (%d bytes/cycle)\n", BusWidthBits,
           bytes_per_cycle);
    printf("Element sizes: A=%d, B=%d, C=%d bytes\n", size_A, size_B, size_C);
    printf("Bus scale factor: %dx (sqrt: %dx)\n", bus_scale_factor,
           bus_sqrt_scale);
    printf("\nScaled tile dimensions:\n");
    printf("  Raw: %dx%dx%d\n", ScaledGemmTile::kM_raw, ScaledGemmTile::kN_raw,
           ScaledGemmTile::kK_raw);
    printf("  Aligned: %dx%dx%d\n", ScaledGemmTile::kM, ScaledGemmTile::kN,
           ScaledGemmTile::kK);
    printf("  Memory: A=%d, B=%d, C=%d bytes (total: %d)\n",
           ScaledGemmTile::bytes_A, ScaledGemmTile::bytes_B,
           ScaledGemmTile::bytes_C, ScaledGemmTile::total_bytes);
    printf("  Cycles needed: %d (budget: %d)\n", ScaledGemmTile::cycles_needed,
           target_cycles);
    printf("\nOptimized tile: %dx%dx%d\n", OptimizedTiles::kM,
           OptimizedTiles::kN, OptimizedTiles::kK);
    printf("Cluster shape: %dx%dx%d\n", ClusterShape::kM, ClusterShape::kN,
           ClusterShape::kK);
    printf("Pipeline stages: %d\n", PipelineConfig::stages);
    printf("Warp arrangement: %dx%d = %d warps\n", WarpArrangement::warps_m,
           WarpArrangement::warps_n, WarpArrangement::total_warps);
    printf("=========================================\n");
  }
};

template <int BusWidthBits, typename Element>
struct ThreadBlockAutoTuner {
  using GemmCalc =
      BusAwareGemmTileCalculator<BusWidthBits, Element, Element, float>;

  // Hardware constraints
  static constexpr int MAX_THREADS_PER_BLOCK = 1024;
  static constexpr int WARP_SIZE = 32;
  static constexpr int MAX_WARPS_PER_SM = 32;
  static constexpr int MAX_SHARED_MEMORY_PER_BLOCK = 49152;  // 48KB typical

  // Memory bandwidth parameters
  static constexpr int bytes_per_cycle = BusWidthBits / 8;
  static constexpr int element_size = cutlass::sizeof_bits<Element>::value / 8;

  struct ThreadConfig {
    int threads_x;  // First dimension (within warp)
    int threads_y;  // Second dimension (warp count)
    int total_threads;
    float score;  // Efficiency score

    bool is_valid() const {
      return threads_x == WARP_SIZE &&  // First dim must be warp size
             total_threads <= MAX_THREADS_PER_BLOCK &&
             total_threads % WARP_SIZE == 0;
    }
  };

  // Calculate optimal thread configuration
  static ThreadConfig autotune(int M, int N, int K, int tile_m, int tile_n,
                               int tile_k) {
    std::vector<ThreadConfig> candidates;

    // First dimension is always 32 (warp size) for coalesced access
    const int threads_x = WARP_SIZE;

    // Calculate workload
    int tiles_m = (M + tile_m - 1) / tile_m;
    int tiles_n = (N + tile_n - 1) / tile_n;
    int tiles_k = (K + tile_k - 1) / tile_k;
    int total_tiles = tiles_m * tiles_n * tiles_k;

    // Memory requirements per block
    size_t smem_per_block = calculate_smem_requirement(tile_m, tile_n, tile_k);

    // Try different warp counts (threads_y)
    for (int warps = 1; warps <= 32; warps++) {
      ThreadConfig config;
      config.threads_x = threads_x;
      config.threads_y = warps;
      config.total_threads = threads_x * warps;

      if (!config.is_valid()) continue;

      // Calculate efficiency score
      config.score = calculate_efficiency_score(config, M, N, K, tile_m, tile_n,
                                                tile_k, tiles_m, tiles_n,
                                                total_tiles, smem_per_block);

      candidates.push_back(config);
    }

    // Select best configuration
    auto best =
        std::max_element(candidates.begin(), candidates.end(),
                         [](const ThreadConfig& a, const ThreadConfig& b) {
                           return a.score < b.score;
                         });

    return *best;
  }

 private:
  static size_t calculate_smem_requirement(int tile_m, int tile_n, int tile_k) {
    // For typical GEMM: A tile + B tile + optional C tile
    size_t size_a = tile_m * tile_k * element_size;
    size_t size_b = tile_k * tile_n * element_size;
    size_t size_c = tile_m * tile_n * sizeof(float);  // Accumulator

    // Add padding for bank conflict avoidance
    size_t padding = 256;  // Conservative padding

    return size_a + size_b + size_c + padding;
  }

  static float calculate_efficiency_score(const ThreadConfig& config, int M,
                                          int N, int K, int tile_m, int tile_n,
                                          int tile_k, int tiles_m, int tiles_n,
                                          int total_tiles,
                                          size_t smem_per_block) {
    float score = 0.0f;

    // 1. Occupancy score (more warps = better, up to a point)
    float occupancy = calculate_occupancy(config, smem_per_block);
    score += occupancy * 30.0f;

    // 2. Memory bandwidth utilization
    float bandwidth_efficiency =
        calculate_bandwidth_efficiency(config, tile_m, tile_n, tile_k);
    score += bandwidth_efficiency * 25.0f;

    // 3. Load balance score
    float load_balance =
        calculate_load_balance(config, tiles_m, tiles_n, total_tiles);
    score += load_balance * 20.0f;

    // 4. Warp efficiency (avoid partial warps)
    float warp_efficiency =
        (config.total_threads % WARP_SIZE == 0) ? 1.0f : 0.5f;
    score += warp_efficiency * 15.0f;

    // 5. Bank conflict avoidance score
    float bank_score = calculate_bank_conflict_score(config, tile_m, tile_n);
    score += bank_score * 10.0f;

    return score;
  }

  static float calculate_occupancy(const ThreadConfig& config,
                                   size_t smem_per_block) {
    // Estimate blocks per SM
    int blocks_limited_by_threads = MAX_THREADS_PER_BLOCK * MAX_WARPS_PER_SM /
                                    (config.total_threads * WARP_SIZE);
    int blocks_limited_by_smem = MAX_SHARED_MEMORY_PER_BLOCK / smem_per_block;

    int blocks_per_sm =
        std::min(blocks_limited_by_threads, blocks_limited_by_smem);
    int warps_per_sm = blocks_per_sm * config.threads_y;

    return float(warps_per_sm) / MAX_WARPS_PER_SM;
  }

  static float calculate_bandwidth_efficiency(const ThreadConfig& config,
                                              int tile_m, int tile_n,
                                              int tile_k) {
    // Bytes moved per block
    size_t bytes_per_block = (tile_m * tile_k + tile_k * tile_n) * element_size;

    // Threads available for loading
    int loading_threads = config.total_threads;

    // Bytes per thread
    float bytes_per_thread = float(bytes_per_block) / loading_threads;

    // Ideal: each thread loads 128 bytes (one cache line)
    float ideal_bytes = 128.0f;

    // Score based on how close we are to ideal
    float ratio = bytes_per_thread / ideal_bytes;
    if (ratio > 1.0f)
      ratio = 2.0f - ratio;  // Penalize too much work per thread

    return std::max(0.0f, ratio);
  }

  static float calculate_load_balance(const ThreadConfig& config, int tiles_m,
                                      int tiles_n, int total_tiles) {
    // For multi-view loading, we want threads_y to evenly divide work
    float balance = 1.0f;

    if (config.threads_y == 2) {
      // 2 warps: ideal for A and B loading
      balance = 1.0f;
    } else if (config.threads_y == 3) {
      // 3 warps: ideal for A, B1, B2 (your case)
      balance = 1.0f;
    } else if (config.threads_y == 4) {
      // 4 warps: good for A, B, C prefetch + compute
      balance = 0.9f;
    } else if (config.threads_y > 4) {
      // More warps: may have idle threads
      balance = 4.0f / config.threads_y;
    }

    return balance;
  }

  static float calculate_bank_conflict_score(const ThreadConfig& config,
                                             int tile_m, int tile_n) {
    // Estimate bank conflicts based on access pattern
    // 32 banks in shared memory
    const int BANK_COUNT = 32;

    // For row-major: threads in a warp access consecutive elements
    int stride = tile_n;  // Assuming row-major
    int conflicts = std::gcd(stride, BANK_COUNT);

    // Perfect score if no conflicts
    float score = (conflicts == 1) ? 1.0f : (1.0f / conflicts);

    return score;
  }

 public:
  // Convenience function to get recommended configuration
  static dim3 get_optimal_block_dims(int M, int N, int K) {
    int tile_m = GemmCalc::OptimizedTiles::kM;
    int tile_n = GemmCalc::OptimizedTiles::kN;
    int tile_k = GemmCalc::OptimizedTiles::kK;
    auto config = autotune(M, N, K, tile_m, tile_n, tile_k);
    return dim3(config.threads_x, config.threads_y, 1);
  }

  static dim3 get_optimal_grid_dims(int M, int N, int K) {
    int tile_m = GemmCalc::OptimizedTiles::kM;
    int tile_n = GemmCalc::OptimizedTiles::kN;
    //     int tile_k = GemmCalc::OptimizedTiles::kK;
    int grid_x = (M + tile_m - 1) / tile_m;
    int grid_y = (N + tile_n - 1) / tile_n;
    return dim3(grid_x, grid_y, 1);
  }

  static constexpr dim3 default_block_dims = dim3(32, 9, 1);

  // Get configuration with rationale
  static void print_autotuning_result(int M, int N, int K, int tile_m,
                                      int tile_n, int tile_k) {
    printf("=== ThreadBlock Autotuning ===\n");
    printf("Problem size: %dx%dx%d\n", M, N, K);
    printf("Tile size: %dx%dx%d\n", tile_m, tile_n, tile_k);
    printf("Bus width: %d bits\n", BusWidthBits);

    auto config = autotune(M, N, K, tile_m, tile_n, tile_k);

    printf("\nOptimal configuration:\n");
    printf("  threads.x = %d (warp size, for coalescing)\n", config.threads_x);
    printf("  threads.y = %d (number of warps)\n", config.threads_y);
    printf("  Total threads = %d\n", config.total_threads);
    printf("  Score = %.2f\n", config.score);

    // Additional analysis
    size_t smem = calculate_smem_requirement(tile_m, tile_n, tile_k);
    printf("\nResource usage:\n");
    printf("  Shared memory per block: %zu bytes\n", smem);
    printf("  Registers per thread: ~64 (estimated)\n");

    float occupancy = calculate_occupancy(config, smem);
    printf("  Theoretical occupancy: %.1f%%\n", occupancy * 100);
  }
};
