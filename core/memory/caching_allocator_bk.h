#pragma once

#include <cuda_runtime_api.h>
#include <unordered_map>
#include <vector>
#include <stdexcept>

#include "utils/cuda_utils.h"

// Templated CachingAllocator class
template <typename Allocator>
class CachingAllocator {
 public:
  static CachingAllocator<Allocator>* instance(int idx) {
    static std::array<CachingAllocator<Allocator>*, 8> instances;
    if (instances[idx] == nullptr) {
      instances[idx] = new CachingAllocator<Allocator>();
    }
    return instances[idx];
  }

  void* allocate(const size_t bytes) {
    const auto& it = available_map_.find(bytes);
    if (it == available_map_.end() || it->second.empty()) {
      return allocate_and_cache(bytes);
    }
    void* ptr = it->second.back();
    it->second.pop_back();
    return ptr;
  }

  void free(void* ptr) {
    const auto& it = allocation_map_.find(ptr);
    if (it == allocation_map_.end()) {
      Allocator::deallocate(ptr);
      return;
    }
    const size_t alloc_size = it->second;
    available_map_[alloc_size].push_back(ptr);
  }

  void record_free(void* ptr) {
    const auto& it = allocation_map_.find(ptr);
    if (it != allocation_map_.end()) {
      allocation_map_.erase(it);
    }
  }

  void free_cached() {
    for (const auto& it : available_map_) {
      for (const auto ptr : it.second) {
        Allocator::deallocate(ptr);
        allocation_map_.erase(ptr);
      }
    }
    available_map_.clear();
  }

  ~CachingAllocator() { free_cached(); }

 private:
  void* allocate_and_cache(const size_t bytes) {
    void* ptr = Allocator::allocate(bytes);
    allocation_map_[ptr] = bytes;
    return ptr;
  }

  std::unordered_map<size_t, std::vector<void*>> available_map_;
  std::unordered_map<void*, size_t> allocation_map_;
};

// Example Allocator for CUDA
struct CudaDeviceAllocator {
  static void* allocate(size_t bytes) {
    void* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    return ptr;
  }

  static void deallocate(void* ptr) { CUDA_CHECK(cudaFree(ptr)); }
};

// Example Allocator for Unified Memory
struct CudaUnifiedAllocator {
  static void* allocate(size_t bytes) {
    void* ptr;
    CUDA_CHECK(cudaMallocManaged(&ptr, bytes));
    return ptr;
  }

  static void deallocate(void* ptr) { CUDA_CHECK(cudaFree(ptr)); }
};

// Example Allocator for cudaHostAlloc
struct CudaHostAllocator {
  static void* allocate(size_t bytes) {
    void* ptr;
    CUDA_CHECK(cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault));
    return ptr;
  }

  static void deallocate(void* ptr) { CUDA_CHECK(cudaFreeHost(ptr)); }
};

// Template specialization for all types of CachingAllocator
typedef CachingAllocator<CudaDeviceAllocator> CudaDeviceCachingAllocator;
typedef CachingAllocator<CudaUnifiedAllocator> CudaUnifiedCachingAllocator;
typedef CachingAllocator<CudaHostAllocator> CudaHostCachingAllocator;
