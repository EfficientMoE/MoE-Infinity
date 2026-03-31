// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <vector>

#include <cuda_runtime.h>
#include <torch/extension.h>

namespace archer {

class KVCachePool {
 public:
  KVCachePool(size_t block_size_bytes, int num_blocks);
  ~KVCachePool();

  int Allocate();

  void Free(int block_id);

  void AsyncCopyToCPU(torch::Tensor& src, int block_id, int stream_idx);

  void AsyncCopyToGPU(int block_id, torch::Tensor& dst, int stream_idx);

  void SyncStream(int stream_idx);

  int NumBlocks() const { return num_blocks_; }
  size_t BlockSizeBytes() const { return block_size_bytes_; }

 private:
  size_t block_size_bytes_;
  int num_blocks_;
  void* pinned_memory_;
  bool pinned_registered_;
  std::vector<int> free_list_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
};

}  // namespace archer
