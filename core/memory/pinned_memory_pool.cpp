// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "memory/pinned_memory_pool.h"

#include <cuda_runtime_api.h>

#include <cstdlib>

#include "utils/logger.h"

PinnedMemoryPool::PinnedMemoryPool(std::size_t chunk_size, int num_chunks)
    : chunk_size_(chunk_size) {
  DLOG_INFO("Creating pinned memory pool: ", num_chunks, " chunks of ",
            chunk_size / 1024, " KB");
  all_chunks_.reserve(num_chunks);
  for (int i = 0; i < num_chunks; ++i) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 4096, chunk_size) != 0) {
      DLOG_FATAL("Failed to allocate aligned memory for pinned pool chunk ", i);
    }
    cudaError_t err = cudaHostRegister(ptr, chunk_size, cudaHostRegisterDefault);
    if (err != cudaSuccess) {
      DLOG_WARN("cudaHostRegister failed for chunk ", i, ": ",
                cudaGetErrorString(err), "; falling back to unpinned");
    }
    all_chunks_.push_back(ptr);
    free_list_.push(ptr);
  }
}

PinnedMemoryPool::~PinnedMemoryPool() {
  for (auto* ptr : all_chunks_) {
    cudaHostUnregister(ptr);
    free(ptr);
  }
}

void* PinnedMemoryPool::Acquire() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return !free_list_.empty(); });
  void* ptr = free_list_.front();
  free_list_.pop();
  return ptr;
}

void PinnedMemoryPool::Release(void* ptr) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    free_list_.push(ptr);
  }
  cv_.notify_one();
}
