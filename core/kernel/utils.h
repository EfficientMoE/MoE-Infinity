#pragma once

#include <cutlass/arch/arch.h>
#include <cassert>

#define KERNEL_LOG_DEBUG(msg, ...)                                \
  do {                                                            \
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && \
        blockIdx.z == 0) {                                        \
      printf(msg, ##__VA_ARGS__);                                 \
    }                                                             \
  } while (0)

template <int ArchCode>
using DetectedArchT = ArchTagT(ArchCode >= 900) ? cutlass::arch::Sm90
                      : (ArchCode >= 800)       ? cutlass::arch::Sm80
                                                : void > ;

template <typename T, typename = void>
struct DetectedArch {
  using SM = void;
};

template <typename T>
struct DetectedArch<T, std::enable_if_t<(__CUDA_ARCH__ > 0)>> {
  using SM = DetectedArchT<__CUDA_ARCH__>;
};

template <int BusWidthBits>
struct OptimalTileCalculator {
  // Assuming we want to saturate memory in 4-8 cycles
  static constexpr int bytes_per_cycle = BusWidthBits / 8;
  static constexpr int target_cycles = 4;

  static constexpr int optimal_tile_bytes = bytes_per_cycle * target_cycles;
  static constexpr int optimal_tile_elements_fp16 = optimal_tile_bytes / 2;
  static constexpr int optimal_tile_elements_fp32 = optimal_tile_bytes / 4;
};
