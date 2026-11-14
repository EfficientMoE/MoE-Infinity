// Hardware gates for kernels written or tuned for one GPU. Policy and the list
// of gated kernels: docs/hardware-gated-kernels.md. Python twin:
// moe_infinity/utils/hw_profile.py.
//
//   Gate::kSm120             compute capability 12.0 (sm_120a-only instructions)
//   Gate::kRtxPro6000        capability 12.0, 188 SMs, 128 MiB L2, 99 KiB shared
//                            memory per block (unified across RTX PRO 6000
//                            Blackwell Server 600W and Max-Q 300W editions)
//   Gate::kRtxPro6000Server  RTX PRO 6000 Blackwell Server Edition (600 W,
//                            335.5 TF BF16 / 656.9 TF FP8, 1530.5 GB/s DRAM)
//   Gate::kRtxPro6000MaxQ    RTX PRO 6000 Blackwell Max-Q Workstation Edition
//                            (300 W, 268.8 TF BF16 / 530.3 TF FP8, 1641.1 GB/s)
//
// Fusion policy: if an edit tuned on Server Edition (600 W) does not regress
// Max-Q (300 W), both editions share the unified Gate::kRtxPro6000 path. Only
// when a tile/config/threshold diverges between the 600 W and 300 W power/DRAM
// envelopes does a kernel branch on Gate::kRtxPro6000Server vs
// Gate::kRtxPro6000MaxQ (or Edition(device)).
//
// A gated path runs only when Supports() is true; other devices keep the
// previous implementation. MOE_INFINITY_UNVALIDATED_HW=1 (read once) lifts
// kRtxPro6000* tuning gates for re-tuning on another card; it never lifts
// kSm120. MOE_INFINITY_PRO6000_EDITION={server|maxq} overrides edition
// detection on a matching RTX PRO 6000 device.
#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <cstdlib>
#include <cstring>

namespace moe_infinity {
namespace hw {

enum class Gate {
  kSm120,
  kRtxPro6000,
  kRtxPro6000Server,
  kRtxPro6000MaxQ,
};

enum class Pro6000Edition {
  kNone,
  kServer,
  kMaxQ,
};

namespace detail {

constexpr int kMaxDevices = 64;
constexpr int kQueried = 1;
constexpr int kIsSm120 = 2;
constexpr int kIsRtxPro6000 = 4;
constexpr int kIsRtxPro6000Server = 8;
constexpr int kIsRtxPro6000MaxQ = 16;

inline std::atomic<int>& Slot(int device) {
  static std::atomic<int> slots[kMaxDevices];
  return slots[device];
}

inline int DetectPro6000Edition(int device) {
  if (const char* env = std::getenv("MOE_INFINITY_PRO6000_EDITION")) {
    if (std::strcmp(env, "server") == 0) return kIsRtxPro6000Server;
    if (std::strcmp(env, "maxq") == 0 || std::strcmp(env, "max_q") == 0) {
      return kIsRtxPro6000MaxQ;
    }
  }
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
    if (std::strstr(prop.name, "Max-Q") != nullptr ||
        std::strstr(prop.name, "MaxQ") != nullptr) {
      return kIsRtxPro6000MaxQ;
    }
    if (std::strstr(prop.name, "Server") != nullptr) {
      return kIsRtxPro6000Server;
    }
  } else {
    cudaGetLastError();
  }
  int mem_khz = 0;
  if (cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, device) ==
      cudaSuccess) {
    if (mem_khz > 0 && mem_khz <= 13000000) return kIsRtxPro6000Server;
    if (mem_khz >= 13500000) return kIsRtxPro6000MaxQ;
  } else {
    cudaGetLastError();
  }
  return 0;
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
      bits |= DetectPro6000Edition(device);
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

inline int LoadBits(int& device) {
  if (device < 0 && cudaGetDevice(&device) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  if (device < 0 || device >= kMaxDevices) return 0;
  std::atomic<int>& slot = Slot(device);
  int bits = slot.load(std::memory_order_relaxed);
  if (bits == 0) {
    bits = Query(device);
    slot.store(bits, std::memory_order_relaxed);
  }
  return bits;
}

}  // namespace detail

// True if `device` (default: the current device) passes `gate`.
inline bool Supports(Gate gate, int device = -1) {
  int bits = detail::LoadBits(device);
  if (bits == 0) return false;
  if (gate == Gate::kSm120) return (bits & detail::kIsSm120) != 0;
  if (detail::UnvalidatedAllowed()) return true;
  if (gate == Gate::kRtxPro6000) return (bits & detail::kIsRtxPro6000) != 0;
  if (gate == Gate::kRtxPro6000Server) {
    return (bits & detail::kIsRtxPro6000Server) != 0;
  }
  if (gate == Gate::kRtxPro6000MaxQ) {
    return (bits & detail::kIsRtxPro6000MaxQ) != 0;
  }
  return false;
}

// Returns the specific RTX PRO 6000 edition (`kServer`, `kMaxQ`, or `kNone`).
inline Pro6000Edition Edition(int device = -1) {
  int bits = detail::LoadBits(device);
  if ((bits & detail::kIsRtxPro6000Server) != 0) return Pro6000Edition::kServer;
  if ((bits & detail::kIsRtxPro6000MaxQ) != 0) return Pro6000Edition::kMaxQ;
  return Pro6000Edition::kNone;
}

}  // namespace hw
}  // namespace moe_infinity
