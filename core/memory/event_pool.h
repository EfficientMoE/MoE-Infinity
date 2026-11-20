// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <cuda_runtime_api.h>

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/noncopyable.h"
#include "utils/cuda_utils.h"

// Pool of disable-timing CUDA events, kept per device.
//
// A CUDA event belongs to the device that was current when it was created and
// may only be *recorded* on a stream of that device; recording it on a stream
// of another device fails (cudaErrorInvalidResourceHandle) and a later
// cudaStreamWaitEvent on such an event returns success without waiting.
// Waiting on an event from a stream of another device is fine.
// Acquire(device) therefore creates the event with `device` current, and
// Release(ev) returns it to the pool of the device it was created for.
class CudaEventPool : public base::noncopyable {
 public:
  CudaEventPool() {}
  ~CudaEventPool() {
    for (auto& entry : pools_) {
      for (auto& ev : entry.second) cudaEventDestroy(ev);
    }
  }

  // Returns an event that may be recorded on streams of `device`.
  cudaEvent_t Acquire(int device) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto& pool = pools_[device];
      if (!pool.empty()) {
        auto ev = pool.back();
        pool.pop_back();
        return ev;
      }
    }
    int previous = -1;
    CUDA_CHECK(cudaGetDevice(&previous));
    cudaEvent_t ev = nullptr;
    cudaError_t status = cudaSuccess;
    if (previous != device) status = cudaSetDevice(device);
    if (status == cudaSuccess) {
      status = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    }
    if (previous != device) cudaSetDevice(previous);
    CUDA_CHECK(status);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      event_device_[ev] = device;
    }
    return ev;
  }

  // Returns `ev` (obtained from Acquire) to the pool of its own device.
  void Release(cudaEvent_t ev) {
    if (ev == nullptr) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = event_device_.find(ev);
    if (it == event_device_.end()) {
      cudaEventDestroy(ev);  // not ours; never recycle across devices
      return;
    }
    pools_[it->second].push_back(ev);
  }

  // Device `ev` was created for, or -1 if it is not a pooled event.
  int DeviceOf(cudaEvent_t ev) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = event_device_.find(ev);
    return it == event_device_.end() ? -1 : it->second;
  }

 private:
  std::mutex mutex_;
  std::unordered_map<int, std::vector<cudaEvent_t>> pools_;
  std::unordered_map<cudaEvent_t, int> event_device_;
};

extern std::unique_ptr<CudaEventPool> kCudaEventPool;
