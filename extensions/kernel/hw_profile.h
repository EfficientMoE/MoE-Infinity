// Hardware gates for kernels written or tuned for one GPU. Policy and the list
// of gated kernels: docs/hardware-gated-kernels.md. Python twin:
// moe_infinity/utils/hw_profile.py.
//
//   Gate::kSm120       compute capability 12.0 (sm_120a-only instructions)
//   Gate::kRtxPro6000  capability 12.0, 188 SMs, 128 MiB L2, 99 KiB shared
//                      memory per block (RTX PRO 6000 Blackwell editions)
//
// A gated path runs only when Supports() is true; other devices keep the
// previous implementation. MOE_INFINITY_UNVALIDATED_HW=1 (read once) lifts
// kRtxPro6000 for re-tuning on another card; it never lifts kSm120.
#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <cstdlib>
#include <cstring>

namespace moe_infinity {
namespace hw {

enum class Gate { kSm120, kRtxPro6000 };

namespace detail {

constexpr int kMaxDevices = 64;
constexpr int kQueried = 1;
constexpr int kIsSm120 = 2;
constexpr int kIsRtxPro6000 = 4;

inline std::atomic<int>& Slot(int device) {
  static std::atomic<int> slots[kMaxDevices];
  return slots[device];
}

inline int Query(int device) {
  int major = 0, minor = 0, sms = 0, l2 = 0, smem = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                             device) != cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                             device) != cudaSuccess ||
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) !=
          cudaSuccess ||
      cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, device) !=
          cudaSuccess ||
      cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                             device) != cudaSuccess) {
    cudaGetLastError();
    return kQueried;
  }
  int bits = kQueried;
  if (major == 12 && minor == 0) {
    bits |= kIsSm120;
    if (sms == 188 && l2 == 128 * 1024 * 1024 && smem == 99 * 1024) {
      bits |= kIsRtxPro6000;
    }
  }
  return bits;
}

inline bool UnvalidatedAllowed() {
  static const bool allowed = [] {
    const char* env = std::getenv("MOE_INFINITY_UNVALIDATED_HW");
    return env != nullptr && std::strcmp(env, "1") == 0;
  }();
  return allowed;
}

}  // namespace detail

// True if `device` (default: the current device) passes `gate`.
inline bool Supports(Gate gate, int device = -1) {
  if (device < 0 && cudaGetDevice(&device) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  if (device >= detail::kMaxDevices) return false;
  std::atomic<int>& slot = detail::Slot(device);
  int bits = slot.load(std::memory_order_relaxed);
  if (bits == 0) {
    bits = detail::Query(device);
    slot.store(bits, std::memory_order_relaxed);
  }
  if (gate == Gate::kSm120) return (bits & detail::kIsSm120) != 0;
  if (detail::UnvalidatedAllowed()) return true;
  return (bits & detail::kIsRtxPro6000) != 0;
}

}  // namespace hw
}  // namespace moe_infinity
