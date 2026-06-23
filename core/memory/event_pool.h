// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <cuda_runtime_api.h>

#include <memory>
#include <mutex>
#include <vector>

#include "base/noncopyable.h"
#include "utils/cuda_utils.h"

class CudaEventPool : public base::noncopyable {
 public:
  CudaEventPool() {}
  ~CudaEventPool() {
    for (auto& ev : pool_) cudaEventDestroy(ev);
  }

  cudaEvent_t Acquire() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pool_.empty()) {
      auto ev = pool_.back();
      pool_.pop_back();
      return ev;
    }
    cudaEvent_t ev;
    cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    return ev;
  }

  void Release(cudaEvent_t ev) {
    std::lock_guard<std::mutex> lock(mutex_);
    pool_.push_back(ev);
  }

 private:
  std::mutex mutex_;
  std::vector<cudaEvent_t> pool_;
};

extern std::unique_ptr<CudaEventPool> kCudaEventPool;
