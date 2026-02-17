#include "torch_caching_allocator.h"

// std::unique_ptr<TorchCachingAllocator> kTorchCachingAllocator =
// std::make_unique<TorchCachingAllocator>();

ReplaceTorchAllocatorOnLoad kReplaceTorchAllocatorOnLoad;
