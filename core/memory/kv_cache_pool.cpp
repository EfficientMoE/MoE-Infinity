// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

#include "memory/kv_cache_pool.h"

#include <cstdlib>
#include <stdexcept>

#include "memory/stream_pool.h"
#include "utils/cuda_utils.h"
#include "utils/logger.h"

namespace archer {

namespace {

inline cudaStream_t ResolveStream(int device_id, int stream_idx) {
  DLOG_FATAL_IF(kTorchStreamPool == nullptr,
                "Torch stream pool not initialized");
  auto& streams = (*kTorchStreamPool)(device_id);
  if (stream_idx < 0 || stream_idx >= static_cast<int>(streams.size())) {
    throw std::out_of_range("stream_idx is out of range for TorchStreamPool");
  }
  return streams[stream_idx].stream();
}

inline void ValidateBlockId(int block_id, int num_blocks) {
  if (block_id < 0 || block_id >= num_blocks) {
    throw std::out_of_range("block_id is out of range");
  }
}

}  // namespace

KVCachePool::KVCachePool(size_t block_size_bytes, int num_blocks)
    : block_size_bytes_(block_size_bytes),
      num_blocks_(num_blocks),
      pinned_memory_(nullptr),
      pinned_registered_(false) {
  DLOG_FATAL_IF(block_size_bytes_ == 0, "block_size_bytes must be > 0");
  DLOG_FATAL_IF(num_blocks_ <= 0, "num_blocks must be > 0");

  const size_t total_bytes =
      block_size_bytes_ * static_cast<size_t>(num_blocks_);
  if (posix_memalign(&pinned_memory_, 4096, total_bytes) != 0) {
    DLOG_FATAL("Failed to allocate aligned memory for KV cache pool");
  }
  cudaError_t register_err =
      cudaHostRegister(pinned_memory_, total_bytes, cudaHostRegisterDefault);
  pinned_registered_ = (register_err == cudaSuccess);
  if (!pinned_registered_) {
    DLOG_WARN("cudaHostRegister failed for KV cache pool: ",
              cudaGetErrorString(register_err), "; falling back to unpinned");
  }

  free_list_.reserve(num_blocks_);
  for (int i = 0; i < num_blocks_; ++i) {
    free_list_.push_back(i);
  }
}

KVCachePool::~KVCachePool() {
  if (pinned_memory_ != nullptr) {
    if (pinned_registered_) {
      cudaError_t err = cudaHostUnregister(pinned_memory_);
      if (err != cudaSuccess) {
        DLOG_WARN("cudaHostUnregister failed: ", cudaGetErrorString(err));
      }
    }
    free(pinned_memory_);
    pinned_memory_ = nullptr;
  }
}

int KVCachePool::Allocate() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return !free_list_.empty(); });
  int block_id = free_list_.back();
  free_list_.pop_back();
  return block_id;
}

void KVCachePool::Free(int block_id) {
  ValidateBlockId(block_id, num_blocks_);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    free_list_.push_back(block_id);
  }
  cv_.notify_one();
}

void KVCachePool::AsyncCopyToCPU(torch::Tensor& src, int block_id,
                                 int stream_idx) {
  ValidateBlockId(block_id, num_blocks_);
  if (!src.is_cuda()) {
    throw std::invalid_argument("src tensor must be CUDA tensor");
  }
  if (!src.is_contiguous()) {
    throw std::invalid_argument("src tensor must be contiguous");
  }

  const size_t copy_bytes = src.nbytes();
  if (copy_bytes > block_size_bytes_) {
    throw std::invalid_argument("src tensor bytes exceed block_size_bytes");
  }

  const int device_id = src.device().index();
  cudaStream_t stream = ResolveStream(device_id, stream_idx);
  void* cpu_ptr = static_cast<char*>(pinned_memory_) +
                  static_cast<size_t>(block_id) * block_size_bytes_;

  CUDA_CHECK(cudaMemcpyAsync(cpu_ptr, src.data_ptr(), copy_bytes,
                             cudaMemcpyDeviceToHost, stream));
}

void KVCachePool::AsyncCopyToGPU(int block_id, torch::Tensor& dst,
                                 int stream_idx) {
  ValidateBlockId(block_id, num_blocks_);
  if (!dst.is_cuda()) {
    throw std::invalid_argument("dst tensor must be CUDA tensor");
  }
  if (!dst.is_contiguous()) {
    throw std::invalid_argument("dst tensor must be contiguous");
  }

  const size_t copy_bytes = dst.nbytes();
  if (copy_bytes > block_size_bytes_) {
    throw std::invalid_argument("dst tensor bytes exceed block_size_bytes");
  }

  const int device_id = dst.device().index();
  cudaStream_t stream = ResolveStream(device_id, stream_idx);
  const void* cpu_ptr = static_cast<const char*>(pinned_memory_) +
                        static_cast<size_t>(block_id) * block_size_bytes_;

  CUDA_CHECK(cudaMemcpyAsync(dst.data_ptr(), cpu_ptr, copy_bytes,
                             cudaMemcpyHostToDevice, stream));
}

void KVCachePool::SyncStream(int stream_idx) {
  const int device_id = GetDevice();
  cudaStream_t stream = ResolveStream(device_id, stream_idx);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace archer
