#pragma once

#include <cuda_runtime_api.h>

#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "common/types.h"
#include "shared_memory.h"
#include "utils/cuda_utils.h"
#include "utils/logger.h"

#define MEMORY_TYPE_VALUES(X, EnumType) \
  X(SHM, EnumType)                      \
  X(PIN, EnumType)                      \
  X(CUDA, EnumType)                     \
  X(PIN_SHM, EnumType)

DEFINE_ENUM_CLASS(MemoryType, MEMORY_TYPE_VALUES)

class CachingAllocator;
extern std::unique_ptr<CachingAllocator> kCachingAllocator;

struct TorchCtx {
  void* ptr;
  size_t size;
};

extern "C" {
void* TorchAllocate(size_t bytes);
void TorchFree(void* ptr);
void TorchFreeCtx(void* ctx);
void* TorchAllocateDevice(size_t bytes);
void TorchFreeDevice(void* ptr);
void TorchFreeCtxDevice(void* ctx);
}

// the caching allocator that supports CPU and CUDA memory
// work as an offset manager for the memory pool
class CachingAllocator : public base::noncopyable {
 public:
  explicit CachingAllocator(size_t bytes, MemoryType type, int device_id = -1);
  virtual ~CachingAllocator();

  virtual void* Allocate(const size_t bytes);
  virtual void Free(void* ptr);

  bool IsAllocated(void* ptr) {
    std::lock_guard<std::mutex> guard(mutex_);
    return allocation_map_.find(ptr) != allocation_map_.end();
  }

  ShmMeta FindShmMetaByRange(void* ptr);
  void InsertShmMeta(ShmMeta meta);

  MemoryType GetType() const { return type_; }

  size_t GetMaxBytes() const { return max_bytes_; }
  size_t GetAllocatedBytes() const { return allocated_bytes_; }
  size_t GetUsedBytes() const { return used_bytes_; }

 private:
  void* AllocateAndCache(const size_t bytes);
  void FreeCached();

  void* AllocateMemory(size_t bytes);

  void* AllocCudaMemory(size_t bytes);
  void* AllocPinMemory(size_t bytes);
  void* AllocShmMemory(size_t bytes);
  void* AllocPinShmMemory(size_t bytes);

  void FreeMemory(void* ptr);
  void FreeCudaMemory(void* ptr);
  void FreePinMemory(void* ptr);
  void FreeShmMemory(void* ptr);
  void FreePinShmMemory(void* ptr);

 protected:
  int device_id_;
  MemoryType type_;
  const size_t max_bytes_;
  size_t allocated_bytes_;
  size_t used_bytes_;

  std::unordered_map<size_t, std::deque<void*>> available_map_;
  std::unordered_map<void*, size_t> allocation_map_;
  std::mutex mutex_;
  std::unordered_map<void*, ShmMeta> shm_id_map_;
};

extern std::once_flag kInitCachingAllocatorFlag;

static void InitCachingAllocator(MemoryType type, int device_id = -1) {
  std::call_once(kInitCachingAllocatorFlag, [&]() {
    size_t bytes = 0;
    DLOG_DEBUG("InitCachingAllocator: type: {}, device_id: {}", type,
               device_id);
    if (type == MemoryType::CUDA) {
      // DLOG_FATAL_IF(device_id < 0, "Invalid device id");
      if (device_id < 0) {
        CUDA_CHECK(cudaGetDevice(&device_id));
      }
      // Get environment variable MOEINF_SHM_SIZE
      const char* size = std::getenv("MOEINF_GPU_SIZE");
      if (size == nullptr) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
        bytes = prop.totalGlobalMem;
      } else {
        DLOG_FATAL_IF(size == nullptr, "MOEINF_GPU_SIZE is not set");
        bytes = std::stoull(size);
      }
    } else if (type == MemoryType::SHM) {
      // Get environment variable MOEINF_SHM_SIZE
      const char* size = std::getenv("MOEINF_SHM_SIZE");
      DLOG_FATAL_IF(size == nullptr, "MOEINF_SHM_SIZE is not set");
      bytes = std::stoull(size);
    } else if (type == MemoryType::PIN or type == MemoryType::PIN_SHM) {
      const char* size = std::getenv("MOEINF_PIN_SIZE");
      DLOG_FATAL_IF(size == nullptr, "MOEINF_PIN_SIZE is not set");
      bytes = std::stoull(size);
    } else {
      DLOG_FATAL("Unknown memory type");
    }
    DLOG_FATAL_IF(kCachingAllocator != nullptr,
                  "Caching allocator is already initialized");

    kCachingAllocator =
        std::make_unique<CachingAllocator>(bytes, type, device_id);
    DLOG_INFO("Caching allocator initialized with {}GB, type: {}", bytes / GB,
              type);
  });
}
