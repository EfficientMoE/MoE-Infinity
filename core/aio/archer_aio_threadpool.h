// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <atomic>
#include <cstdint>
#include <vector>

#include "archer_aio_thread.h"

class ArcherAioThreadPool {
 public:
  explicit ArcherAioThreadPool(int num_threads);
  ~ArcherAioThreadPool();

  void Start();
  void Stop();

  void Enqueue(AioCallback& callback, int thread_id = -1);
  void Wait();

 private:
  int num_threads_;
  std::vector<std::unique_ptr<ArcherAioThread>> threads_;
  std::atomic<int> round_robin_counter_{0};
};
