#include "caching_allocator.h"

#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "common/generator.h"
#include "utils/logger.h"

std::unique_ptr<CachingAllocator> kCachingAllocator = nullptr;
std::once_flag kInitCachingAllocatorFlag;

void* CachingAllocator::Allocate(const size_t bytes) {
  std::lock_guard<std::mutex> guard(mutex_);
  // DLOG_DEBUG("Try Allocate: size: {}, used: {}", bytes, used_bytes_);
  const auto& it = available_map_.find(bytes);
  void* ptr = nullptr;
  if (it == available_map_.end() || it->second.empty()) {
    if (bytes == 0) {
      ptr = malloc(bytes);
      DLOG_WARN("Attempted to allocate 0 bytes, return ", ptr);
    } else {
      ptr = AllocateAndCache(bytes);
    }
  } else {
    ptr = it->second.back();
    it->second.pop_back();
  }
  used_bytes_ += bytes;
  allocation_map_[ptr] = bytes;
  // DLOG_DEBUG("Allocate: {:p}, size: {}, used: {}", ptr, bytes, used_bytes_);
  return ptr;
}

void CachingAllocator::Free(void* ptr) {
  std::lock_guard<std::mutex> guard(mutex_);
  // DLOG_DEBUG("Try Free: {:p}", ptr);
  const auto& it = allocation_map_.find(ptr);
  DLOG_FATAL_IF(it == allocation_map_.end(),
                "Attempted to free unallocated memory ", ptr);
  const size_t alloc_size = it->second;
  available_map_[alloc_size].push_back(ptr);
  used_bytes_ -= alloc_size;
  allocation_map_.erase(it);
  // DLOG_DEBUG("Free: {:p}, size: {}, used: {}", ptr, alloc_size, used_bytes_);
}

void CachingAllocator::InsertShmMeta(ShmMeta meta) {
  std::lock_guard<std::mutex> guard(mutex_);
  shm_id_map_[meta.ptr] = meta;
}

ShmMeta CachingAllocator::FindShmMetaByRange(void* ptr) {
  std::lock_guard<std::mutex> guard(mutex_);
  {
    auto it = shm_id_map_.find(ptr);
    if (it != shm_id_map_.end()) {
      return it->second;
    }
  }
  for (const auto& it : shm_id_map_) {
    if (it.first <= ptr && ptr < (char*)it.first + it.second.size) {
      DLOG_WARN_IF(it.first != ptr, "FindShmMetaByRange not exact: expected ",
                   ptr, ", got ", it.first);
      return it.second;
    }
  }
  DLOG_FATAL("Cannot find shm meta by range ", ptr);
  return {};
}

// int CachingAllocator::GetShmId(void* ptr) {
//   std::lock_guard<std::mutex> guard(mutex_);
//   // const auto& it = shm_id_map_.find(ptr);
//   // DLOG_FATAL_IF(it == shm_id_map_.end(), "Cannot find shm id {:p}", ptr);
//   auto meta = FindShmMetaByRange(ptr);
//   return meta.id;
//   // return it->second.id;
// }

// std::string CachingAllocator::GetShmName(void* ptr) {
//   std::lock_guard<std::mutex> guard(mutex_);
//   auto meta = FindShmMetaByRange(ptr);
//   // const auto& it = shm_id_map_.find(ptr);
//   // DLOG_FATAL_IF(it == shm_id_map_.end(), "Cannot find shm name {:p}",
//   ptr); return meta.name;
// }

// size_t CachingAllocator::GetShmSize(void* ptr) {
//   std::lock_guard<std::mutex> guard(mutex_);
//   auto meta = FindShmMetaByRange(ptr);
//   size_t remain_size = meta.size - (ptr - meta.ptr);
//   // const auto& it = shm_id_map_.find(ptr);
//   // DLOG_FATAL_IF(it == shm_id_map_.end(), "Cannot find shm size {:p}",
//   ptr); return remain_size;
// }

void CachingAllocator::FreeCached() {
  for (const auto& it : available_map_) {
    for (const auto& ptr : it.second) {
      FreeMemory(ptr);
      allocated_bytes_ -= it.first;
      allocation_map_.erase(ptr);
      used_bytes_ -= it.first;
    }
  }
  available_map_.clear();
}

void* CachingAllocator::AllocateAndCache(const size_t bytes) {
  // DLOG_DEBUG("AllocateAndCache: size: {}, used: {}", bytes, used_bytes_);
  if (allocated_bytes_ + bytes > max_bytes_) {
    FreeCached();
    DLOG_FATAL_IF(allocated_bytes_ + bytes > max_bytes_,
                  "Out of memory; attempted to allocate ", bytes / GB,
                  "GB, allocated ", allocated_bytes_ / GB, "GB, ", "but only ",
                  (max_bytes_ - allocated_bytes_) / GB, "GB available");
  }
  void* ptr = AllocateMemory(bytes);
  return ptr;
}

void* CachingAllocator::AllocateMemory(size_t bytes) {
  switch (type_) {
    case MemoryType::SHM:
      return AllocShmMemory(bytes);
    case MemoryType::PIN:
      return AllocPinMemory(bytes);
    case MemoryType::CUDA:
      return AllocCudaMemory(bytes);
    case MemoryType::PIN_SHM:
      return AllocPinShmMemory(bytes);
    default:
      DLOG_FATAL("Unknown memory type");
      return nullptr;
  }
  DLOG_FATAL("Unknown memory type");
  return nullptr;
}
void* CachingAllocator::AllocCudaMemory(size_t bytes) {
  void* ptr;
  cudaSetDevice(device_id_);
  cudaMalloc(&ptr, bytes);
  return ptr;
}
void* CachingAllocator::AllocPinMemory(size_t bytes) {
  void* ptr = aligned_alloc(4096, bytes);
  // int ret = mlock(ptr, bytes);
  // DLOG_FATAL_IF(ret != 0, "mlock failed: errno {}, message {}", errno,
  //              strerror(errno));
  cudaHostRegister(ptr, bytes, cudaHostRegisterDefault);
  // void* ptr;
  // cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
  return ptr;
}
void* CachingAllocator::AllocShmMemory(size_t bytes) {
  int shm_id = shmget(IPC_PRIVATE, bytes, IPC_CREAT | 0666);
  void* ptr = shmat(shm_id, nullptr, 0);
  DLOG_FATAL_IF(ptr == (void*)-1, "shmat failed: errno ", errno, ", message ",
                strerror(errno));
  shm_id_map_[ptr] = {shm_id, ptr, bytes, ""};
  return ptr;
}
void* CachingAllocator::AllocPinShmMemory(size_t bytes) {
  ShmMeta shm_meta;
  shm_meta.name = "/emulator_shm_" + GenUUID();

  // DLOG_DEBUG("shm_meta name: {}", shm_meta.name);

  int shm_fd = shm_open(shm_meta.name.c_str(), O_CREAT | O_RDWR, 0666);
  DLOG_FATAL_IF(shm_fd == -1, "shm_open failed: errno ", errno, ", message ",
                strerror(errno));
  DLOG_FATAL_IF(ftruncate(shm_fd, bytes) == -1, "ftruncate failed: errno ",
                errno, ", message ", strerror(errno));

  size_t page_size = sysconf(_SC_PAGESIZE);
  size_t aligned_bytes = ((bytes + page_size - 1) / page_size) * page_size;

  // Specify the fixed address
  void* buf = AllocPinMemory(aligned_bytes);

  // DLOG_DEBUG("alloc pin shm: bytes: {}, preferred_addr: {:p}, aligned_bytes:
  // {}",
  //           bytes, buf, aligned_bytes);
  void* shm_addr = mmap(buf, aligned_bytes, PROT_READ | PROT_WRITE,
                        MAP_SHARED | MAP_FIXED | MAP_LOCKED, shm_fd, 0);
  DLOG_FATAL_IF(shm_addr == MAP_FAILED, "mmap failed: errno ", errno,
                ", message ", strerror(errno));

  DLOG_FATAL_IF(shm_addr != buf, "mmap failed: expected addr: ", buf, ", got ",
                shm_addr);

  shm_meta.id = shm_fd;
  shm_meta.ptr = shm_addr;
  shm_meta.size = aligned_bytes;
  shm_id_map_[shm_addr] = shm_meta;
  // DLOG_DEBUG("shm_meta: {}", shm_meta);

  return shm_addr;
}

void CachingAllocator::FreeMemory(void* ptr) {
  switch (type_) {
    case MemoryType::SHM:
      FreeShmMemory(ptr);
      break;
    case MemoryType::PIN:
      FreePinMemory(ptr);
      break;
    case MemoryType::CUDA:
      FreeCudaMemory(ptr);
      break;
    case MemoryType::PIN_SHM:
      FreePinShmMemory(ptr);
      break;
    default:
      DLOG_FATAL("Unknown memory type");
      break;
  }
}
void CachingAllocator::FreeCudaMemory(void* ptr) {
  cudaSetDevice(device_id_);
  cudaFree(ptr);
}
void CachingAllocator::FreePinMemory(void* ptr) { cudaFreeHost(ptr); }
void CachingAllocator::FreeShmMemory(void* ptr) {
  shmdt(ptr);
  auto shmid = shm_id_map_[ptr].id;
  shmctl(shmid, IPC_RMID, nullptr);
  shm_id_map_.erase(ptr);
}
void CachingAllocator::FreePinShmMemory(void* ptr) {
  auto meta = FindShmMetaByRange(ptr);
  auto shm_name = meta.name;
  auto size = meta.size;
  // auto id = meta.id;

  // DLOG_DEBUG("FreePinShmMemory: addr: {:p}, name: {}, size: {}", ptr,
  // shm_name, size);
  munmap(ptr, size);
  // shm_unlink(shm_id_map_[ptr].name);
  close(shm_id_map_[ptr].id);
  FreePinMemory(ptr);
  shm_id_map_.erase(ptr);
}

CachingAllocator::CachingAllocator(size_t bytes, MemoryType type, int device_id)
    : max_bytes_(bytes),
      allocated_bytes_(0),
      used_bytes_(0),
      type_(type),
      device_id_(device_id) {}

CachingAllocator::~CachingAllocator() { FreeCached(); }

extern "C" {
void* TorchAllocate(size_t bytes) {
  // DLOG_DEBUG("TorchAllocate: size: {}", bytes);
  InitCachingAllocator(MemoryType::PIN);
  void* ptr = kCachingAllocator->Allocate(bytes);
  return ptr;
}

void TorchFree(void* ptr) {
  // DLOG_DEBUG("TorchFree: {:p}", ptr);
  InitCachingAllocator(MemoryType::PIN);
  kCachingAllocator->Free(ptr);
}

void TorchFreeCtx(void* ctx) {
  InitCachingAllocator(MemoryType::PIN);
  TorchCtx* torch_ctx = static_cast<TorchCtx*>(ctx);
  kCachingAllocator->Free(torch_ctx->ptr);
  delete torch_ctx;
}

void* TorchAllocateDevice(size_t bytes) {
  InitCachingAllocator(MemoryType::CUDA);
  void* ptr = kCachingAllocator->Allocate(bytes);
  return ptr;
}

void TorchFreeDevice(void* ptr) {
  InitCachingAllocator(MemoryType::CUDA);
  kCachingAllocator->Free(ptr);
}

void TorchFreeCtxDevice(void* ctx) {
  InitCachingAllocator(MemoryType::CUDA);
  TorchCtx* torch_ctx = static_cast<TorchCtx*>(ctx);
  kCachingAllocator->Free(torch_ctx->ptr);
  delete torch_ctx;
}
}
