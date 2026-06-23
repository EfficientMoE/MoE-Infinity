// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "archer_prio_aio_handle.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <thread>

#include <cstdlib>

#ifndef NVTX_DISABLE
  #include <nvtx3/nvtx3.hpp>
#endif

#include "archer_aio_utils.h"
#include "utils/cuda_utils.h"
#include "utils/logger.h"

static int GetBlockSize() {
  const char* env = std::getenv("MOE_IO_BLOCK_SIZE_KB");
  if (env != nullptr) {
    int kb = std::atoi(env);
    if (kb >= 256 && kb <= 65536) return kb * 1024;
  }
  return 1024 * 1024;
}

const int kBlockSize = GetBlockSize();
const int kQueueDepth = 32;

ArcherPrioAioHandle::ArcherPrioAioHandle(const std::string& prefix,
                                         int num_io_threads)
    : time_to_exit_(false),
      aio_context_(kBlockSize, time_to_exit_, num_io_threads) {
  // InitLogger();
  int effective_threads = (num_io_threads > 0)
                              ? num_io_threads
                              : ArcherPrioAioContext::GetDefaultNumIoThreads();
  pinned_pool_ =
      std::make_shared<PinnedMemoryPool>(kBlockSize, effective_threads * 4);
  thread_ = std::thread(&ArcherPrioAioHandle::Run, this);
}

void ArcherPrioAioHandle::Run() {
  while (!time_to_exit_.load()) {
    aio_context_.Schedule();
  }
}

ArcherPrioAioHandle::~ArcherPrioAioHandle() {
  time_to_exit_.store(true);
  aio_context_.NotifyExit();
  thread_.join();
  for (auto& file : file_set_) {
    close(file.second);
  }
}

std::int64_t ArcherPrioAioHandle::Read(const std::string& filename,
                                       void* buffer, const bool high_prio,
                                       const std::int64_t num_bytes,
                                       const std::int64_t offset) {
#ifndef NVTX_DISABLE
  nvtx3::scoped_range r("aio_read_submit");
#endif

  int fd = -1;
  {
    std::lock_guard<std::mutex> lock(file_set_mutex_);
    auto file = file_set_.find(filename);
    if (file == file_set_.end()) {
      fd = ArcherOpenFile(filename.c_str());
      file_set_.insert(std::make_pair(filename, fd));
    }
    fd = file_set_[filename];
  }

  std::int64_t num_bytes_aligned =
      (num_bytes + kAioAlignment - 1) & ~(kAioAlignment - 1);

  // O_DIRECT requires 4096-byte aligned buffers. The caller's buffer may only
  // be 64-byte aligned (c10 allocator). Use pinned-pool bounce buffers that
  // are guaranteed 4096-aligned, then memcpy to the final destination.
  const auto n_blocks =
      num_bytes_aligned / static_cast<std::int64_t>(kBlockSize);
  const auto last_block_size =
      num_bytes_aligned % static_cast<std::int64_t>(kBlockSize);
  const auto n_iocbs = n_blocks + (last_block_size > 0 ? 1 : 0);

  std::vector<AioCallback> callbacks;
  auto pool = pinned_pool_;

  for (std::int64_t i = 0; i < n_iocbs; ++i) {
    const std::int64_t shift = i * static_cast<std::int64_t>(kBlockSize);
    const std::int64_t xfer_offset = offset + shift;
    auto byte_count = static_cast<std::int64_t>(kBlockSize);
    if ((shift + byte_count) > num_bytes_aligned) {
      byte_count = num_bytes_aligned - shift;
    }
    std::int64_t copy_count =
        (shift + byte_count <= num_bytes)
            ? byte_count
            : std::max(num_bytes - shift, static_cast<std::int64_t>(0));
    void* dst_ptr = static_cast<char*>(buffer) + shift;

    callbacks.push_back(
        [pool, fd, dst_ptr, copy_count, byte_count, xfer_offset]() -> int {
          void* chunk = pool->Acquire();
          int ret = ArcherReadFile(fd, chunk, byte_count, xfer_offset);
          if (copy_count > 0) {
            memcpy(dst_ptr, chunk, copy_count);
          }
          pool->Release(chunk);
          return ret;
        });
  }

  auto io_request = std::make_shared<struct AioRequest>();
  io_request->callbacks = std::move(callbacks);
  io_request->pending_callbacks.store(io_request->callbacks.size());
  aio_context_.AcceptRequest(io_request, high_prio);

  Wait(io_request);

  return num_bytes_aligned;
}

std::int64_t ArcherPrioAioHandle::Write(const std::string& filename,
                                        const void* buffer,
                                        const bool high_prio,
                                        const std::int64_t num_bytes,
                                        const std::int64_t offset) {
#ifndef NVTX_DISABLE
  nvtx3::scoped_range r("aio_write_submit");
#endif

  int fd = -1;
  {
    std::lock_guard<std::mutex> lock(file_set_mutex_);
    auto file = file_set_.find(filename);
    if (file == file_set_.end()) {
      fd = ArcherOpenFile(filename.c_str());
      file_set_.insert(std::make_pair(filename, fd));
    }
    fd = file_set_[filename];
  }

  std::int64_t num_bytes_aligned =
      (num_bytes + kAioAlignment - 1) & ~(kAioAlignment - 1);

  auto mem_type =
      IsDevicePointer(buffer) ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost;

  // Build per-chunk callbacks that acquire from pinned pool, copy, write,
  // release
  const auto n_blocks =
      num_bytes_aligned / static_cast<std::int64_t>(kBlockSize);
  const auto last_block_size =
      num_bytes_aligned % static_cast<std::int64_t>(kBlockSize);
  const auto n_iocbs = n_blocks + (last_block_size > 0 ? 1 : 0);

  std::vector<AioCallback> callbacks;
  auto pool = pinned_pool_;

  for (std::int64_t i = 0; i < n_iocbs; ++i) {
    const std::int64_t shift = i * static_cast<std::int64_t>(kBlockSize);
    const std::int64_t xfer_offset = offset + shift;
    auto byte_count = static_cast<std::int64_t>(kBlockSize);
    if ((shift + byte_count) > num_bytes_aligned) {
      byte_count = num_bytes_aligned - shift;
    }
    // How many bytes of real data (not padding) to copy for this chunk
    std::int64_t copy_count =
        (shift + byte_count <= num_bytes)
            ? byte_count
            : std::max(num_bytes - shift, (std::int64_t)0);
    const void* src_ptr = static_cast<const char*>(buffer) + shift;

    callbacks.push_back([pool, fd, src_ptr, copy_count, byte_count, xfer_offset,
                         mem_type]() -> int {
      void* chunk = pool->Acquire();
      if (copy_count > 0) {
        CudaMemcpy(chunk, src_ptr, copy_count, mem_type);
      }
      // Zero-fill padding beyond real data
      if (copy_count < byte_count) {
        memset(static_cast<char*>(chunk) + copy_count, 0,
               byte_count - copy_count);
      }
      int ret = ArcherWriteFile(fd, chunk, byte_count, xfer_offset);
      pool->Release(chunk);
      return ret;
    });
  }

  auto io_request = std::make_shared<struct AioRequest>();
  io_request->callbacks = std::move(callbacks);
  io_request->pending_callbacks.store(io_request->callbacks.size());
  aio_context_.AcceptRequest(io_request, high_prio);

  Wait(io_request);

  return num_bytes_aligned;
}

void ArcherPrioAioHandle::Wait(const std::shared_ptr<AioRequest>& io_request) {
#ifndef NVTX_DISABLE
  nvtx3::scoped_range r("aio_wait");
#endif

  std::unique_lock<std::mutex> lock(io_request->mutex);
  io_request->cv.wait(lock, [&io_request] {
    return io_request->pending_callbacks.load() == 0;
  });
}

int ArcherPrioAioContext::GetDefaultNumIoThreads() {
  auto hw = std::thread::hardware_concurrency();
  int num = static_cast<int>(hw) / 4;
  return std::max(num, 4);
}

ArcherPrioAioContext::ArcherPrioAioContext(const int block_size,
                                           std::atomic<bool>& time_to_exit,
                                           int num_io_threads)
    : block_size_(block_size), time_to_exit_(time_to_exit) {
  if (num_io_threads <= 0) {
    num_io_threads = GetDefaultNumIoThreads();
  }
  DLOG_INFO("Creating I/O thread pool with ", num_io_threads, " threads");
  thread_pool_ = std::make_unique<ArcherAioThreadPool>(num_io_threads);
  thread_pool_->Start();
}

ArcherPrioAioContext::~ArcherPrioAioContext() {}

void ArcherPrioAioContext::NotifyExit() { schedule_cv_.notify_all(); }

void ArcherPrioAioContext::Schedule() {
  std::shared_ptr<AioRequest> io_request = nullptr;

  // Wait until there is work to do
  {
    std::unique_lock<std::mutex> lock(schedule_mutex_);
    schedule_cv_.wait(lock, [this] {
      std::lock_guard<std::mutex> lh(io_queue_high_mutex_);
      std::lock_guard<std::mutex> ll(io_queue_low_mutex_);
      return !io_queue_high_.empty() || !io_queue_low_.empty() ||
             time_to_exit_.load();
    });
    if (time_to_exit_.load()) {
      return;
    }
  }

  // Prefer high-priority requests (on-demand expert loads).
  {
    std::lock_guard<std::mutex> lock(io_queue_high_mutex_);
    if (!io_queue_high_.empty()) {
      io_request = io_queue_high_.front();
      io_queue_high_.pop_front();
    }
  }

  if (io_request != nullptr) {
    for (auto& cb : io_request->callbacks) {
      thread_pool_->Enqueue(cb);
    }
    thread_pool_->Wait();
    io_request->callbacks.clear();
    io_request->pending_callbacks.store(0);
    io_request->cv.notify_one();
    return;
  }

  // Low-priority requests (prefetch, FetchExec): batch ALL callbacks to the
  // thread pool so every I/O thread works in parallel, matching the high-prio
  // path.  Preemption is maintained between complete requests — if a high-prio
  // request arrives while a low-prio request is in-flight, it will be served
  // on the next Schedule() iteration.
  {
    std::lock_guard<std::mutex> lock(io_queue_low_mutex_);
    if (!io_queue_low_.empty()) {
      io_request = io_queue_low_.front();
      io_queue_low_.pop_front();
    }
  }

  if (io_request == nullptr) {
    return;
  }

  for (auto& cb : io_request->callbacks) {
    thread_pool_->Enqueue(cb);
  }
  thread_pool_->Wait();
  io_request->callbacks.clear();
  io_request->pending_callbacks.store(0);
  io_request->cv.notify_one();
}

void ArcherPrioAioContext::AcceptRequest(
    std::shared_ptr<AioRequest>& io_request, bool high_prio) {
  {
    std::lock_guard<std::mutex> lock(high_prio ? io_queue_high_mutex_
                                               : io_queue_low_mutex_);
    if (high_prio) {
      io_queue_high_.push_back(io_request);
    } else {
      io_queue_low_.push_back(io_request);
    }
  }
  schedule_cv_.notify_one();
}

std::vector<AioCallback> ArcherPrioAioContext::PrepIocbs(
    const bool read_op, void* buffer, const int fd, const int block_size,
    const std::int64_t offset, const std::int64_t total_size) {
  const auto n_blocks = total_size / static_cast<std::int64_t>(block_size);
  const auto last_block_size =
      total_size % static_cast<std::int64_t>(block_size);
  const auto n_iocbs = n_blocks + (last_block_size > 0 ? 1 : 0);

  std::vector<AioCallback> callbacks;

  for (auto i = 0; i < n_iocbs; ++i) {
    const std::int64_t shift = i * static_cast<std::int64_t>(block_size);
    const auto xfer_buffer = static_cast<char*>(buffer) + shift;
    const std::int64_t xfer_offset = offset + shift;
    auto byte_count = static_cast<std::int64_t>(block_size);
    if ((shift + block_size) > total_size) {
      byte_count = total_size - shift;
    }
    if (read_op) {
      auto cb =
          std::bind(ArcherReadFile, fd, xfer_buffer, byte_count, xfer_offset);
      callbacks.push_back(std::move(cb));
    } else {
      auto cb =
          std::bind(ArcherWriteFile, fd, xfer_buffer, byte_count, xfer_offset);
      callbacks.push_back(std::move(cb));
    }
  }

  return callbacks;
}
