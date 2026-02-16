// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <unordered_map>

#include "archer_aio_threadpool.h"
#include "memory/pinned_memory_pool.h"
#include "utils/simple_object_pool.h"

static const std::size_t kAioAlignment = 4096;
using IocbPtr = typename SimpleObjectPool<struct iocb>::Pointer;

struct AioRequest {
  // std::vector<IocbPtr> iocbs;
  std::vector<AioCallback> callbacks;
  std::mutex mutex;
  std::condition_variable cv;
  std::atomic<int> pending_callbacks;
};

class ArcherPrioAioContext {
 public:
  explicit ArcherPrioAioContext(const int block_size,
                                std::atomic<bool>& time_to_exit,
                                int num_io_threads = 0);
  ~ArcherPrioAioContext();

  static int GetDefaultNumIoThreads();

  void AcceptRequest(std::shared_ptr<AioRequest>& io_request, bool high_prio);
  void NotifyExit();

  void Schedule();
  std::vector<AioCallback> PrepIocbs(const bool read_op, void* buffer,
                                     const int fd, const int block_size,
                                     const std::int64_t offset,
                                     const std::int64_t total_size);

 private:
  std::int64_t block_size_;

  std::mutex io_queue_high_mutex_;
  std::mutex io_queue_low_mutex_;

  std::deque<std::shared_ptr<struct AioRequest>> io_queue_high_;
  std::deque<std::shared_ptr<struct AioRequest>> io_queue_low_;

  std::condition_variable schedule_cv_;
  std::mutex schedule_mutex_;
  std::atomic<bool>& time_to_exit_;

  std::unique_ptr<ArcherAioThreadPool> thread_pool_;
};

class ArcherPrioAioHandle {
 public:
  explicit ArcherPrioAioHandle(const std::string& prefix,
                               int num_io_threads = 0);
  ~ArcherPrioAioHandle();

  std::int64_t Read(const std::string& filename, void* buffer,
                    const bool high_prio, const std::int64_t num_bytes,
                    const std::int64_t offset);
  std::int64_t Write(const std::string& filename, const void* buffer,
                     const bool high_prio, const std::int64_t num_bytes,
                     const std::int64_t offset);

 private:
  void Run();  // io submit thread function

 private:
  std::atomic<bool> time_to_exit_;
  std::thread thread_;
  std::mutex file_set_mutex_;
  std::unordered_map<std::string, int> file_set_;
  std::shared_ptr<PinnedMemoryPool> pinned_pool_;
  ArcherPrioAioContext aio_context_;
};
