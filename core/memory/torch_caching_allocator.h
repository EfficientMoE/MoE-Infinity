#pragma once

#include <torch/torch.h>

#include "caching_allocator.h"

struct TorchCachingAllocator : public torch::Allocator {
  // For Torch Interface
  torch::DataPtr allocate(size_t n) override {
    void* data = TorchAllocate(n);
    return {data, data, &TorchFree, torch::DeviceType::CPU};
  }

  void copy_data(void* dest, const void* src, size_t count) const override {
    LOG_DEBUG("Copy data from {:p} to {:p}, size: {}", src, dest, count);
    memcpy(dest, src, count);
  }

  // // Optional: Handle deallocation (if needed)
  // void deallocate(void* ptr) override {
  //   Free(ptr);  // Custom deallocation logic
  // }
};

// extern std::unique_ptr<TorchCachingAllocator> kTorchCachingAllocator;

class ReplaceTorchAllocatorOnLoad {
 public:
  ReplaceTorchAllocatorOnLoad() {
    std::call_once(flag_, [&]() {
      InitCachingAllocator(MemoryType::PINNED);
      torch_caching_allocator_ = new TorchCachingAllocator();
      LOG_INFO("Replace torch allocator with caching allocator");
      torch::SetAllocator(torch::DeviceType::CPU, torch_caching_allocator_);
      LOG_INFO("Torch allocator replaced");
    });
  }

 private:
  TorchCachingAllocator* torch_caching_allocator_;
  std::once_flag flag_;
};

// Create a static instance of this class
extern ReplaceTorchAllocatorOnLoad kReplaceTorchAllocatorOnLoad;
