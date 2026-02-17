#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <type_traits>

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

// Predefined configurations for common GPUs
template <typename ElementA, typename ElementB, typename ElementC>
struct GPUOptimalGemmTiles {
  // Consumer GPUs
  using RTX4090 = BusAwareGemmTileCalculator<384, ElementA, ElementB, ElementC>;
  using RTX4080 = BusAwareGemmTileCalculator<256, ElementA, ElementB, ElementC>;
  using RTX3090 = BusAwareGemmTileCalculator<384, ElementA, ElementB, ElementC>;

  // Data center GPUs
  using H200 = BusAwareGemmTileCalculator<6144, ElementA, ElementB, ElementC>;
  using H100 = BusAwareGemmTileCalculator<5120, ElementA, ElementB, ElementC>;
  using A100 = BusAwareGemmTileCalculator<5120, ElementA, ElementB, ElementC>;
  using V100 = BusAwareGemmTileCalculator<4096, ElementA, ElementB, ElementC>;

  static void compare_all() {
    printf("=== GEMM Tile Sizes Across GPUs ===\n");
    printf("Data types: A=%d-bit, B=%d-bit, C=%d-bit\n\n",
           cutlass::sizeof_bits<ElementA>::value,
           cutlass::sizeof_bits<ElementB>::value,
           cutlass::sizeof_bits<ElementC>::value);

    printf("GPU        Bus    Tile (MxNxK)    Cluster  Stages\n");
    printf("---------- ------ --------------- -------- ------\n");

    auto print_gpu = [](const char* name, int bus, auto calc) {
      using Calc = decltype(calc);
      printf("%-10s %4d   %3dx%3dx%2d     %dx%dx%d    %d\n", name, bus,
             Calc::OptimizedTiles::kM, Calc::OptimizedTiles::kN,
             Calc::OptimizedTiles::kK, Calc::ClusterShape::kM,
             Calc::ClusterShape::kN, Calc::ClusterShape::kK,
             Calc::PipelineConfig::stages);
    };

    print_gpu("RTX 4080", 256, RTX4080{});
    print_gpu("RTX 3090", 384, RTX3090{});
    print_gpu("RTX 4090", 384, RTX4090{});
    print_gpu("V100", 4096, V100{});
    print_gpu("A100", 5120, A100{});
    print_gpu("H100", 5120, H100{});
    print_gpu("H200", 6144, H200{});
  }
};

// Example usage
int main() {
  // Show how tile sizes scale with bus width
  printf("=== Scaling Analysis ===\n");

  // Same data types, different bus widths
  using BF16_Gemm_256 = BusAwareGemmTileCalculator<256, cutlass::bfloat16_t,
                                                   cutlass::bfloat16_t, float>;
  using BF16_Gemm_512 = BusAwareGemmTileCalculator<512, cutlass::bfloat16_t,
                                                   cutlass::bfloat16_t, float>;
  using BF16_Gemm_1024 = BusAwareGemmTileCalculator<1024, cutlass::bfloat16_t,
                                                    cutlass::bfloat16_t, float>;
  using BF16_Gemm_4096 = BusAwareGemmTileCalculator<4096, cutlass::bfloat16_t,
                                                    cutlass::bfloat16_t, float>;
  using BF16_Gemm_5120 = BusAwareGemmTileCalculator<5120, cutlass::bfloat16_t,
                                                    cutlass::bfloat16_t, float>;

  printf("\nBF16xBF16->FP32 GEMM tiles vs bus width:\n");
  printf("Bus Width  Tile Size      Bandwidth Usage\n");
  printf("--------- -------------- ----------------\n");

  auto print_config = [](int bus, auto calc) {
    using Calc = decltype(calc);
    printf("%4d-bit  %3dx%3dx%2d    %d/%d cycles\n", bus,
           Calc::OptimizedTiles::kM, Calc::OptimizedTiles::kN,
           Calc::OptimizedTiles::kK, Calc::ScaledGemmTile::cycles_needed,
           Calc::target_cycles);
  };

  print_config(256, BF16_Gemm_256{});
  print_config(512, BF16_Gemm_512{});
  print_config(1024, BF16_Gemm_1024{});
  print_config(4096, BF16_Gemm_4096{});
  print_config(5120, BF16_Gemm_5120{});

  // Compare across different GPUs
  printf("\n");
  GPUOptimalGemmTiles<cutlass::bfloat16_t, cutlass::bfloat16_t,
                      float>::compare_all();

  // Detailed config for H100
  printf("\n");
  BF16_Gemm_5120::print_config();

  printf("\n");
  BF16_Gemm_256::print_config();

  return 0;
}
