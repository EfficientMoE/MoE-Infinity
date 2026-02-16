// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <queue>
#include <vector>

class PinnedMemoryPool {
 public:
  PinnedMemoryPool(std::size_t chunk_size, int num_chunks);
  ~PinnedMemoryPool();

  void* Acquire();
  void Release(void* ptr);

  std::size_t chunk_size() const { return chunk_size_; }

 private:
  std::size_t chunk_size_;
  std::vector<void*> all_chunks_;
  std::queue<void*> free_list_;
  std::mutex mutex_;
  std::condition_variable cv_;
};
