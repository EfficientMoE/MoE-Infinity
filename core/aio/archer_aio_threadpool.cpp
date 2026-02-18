// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "archer_aio_threadpool.h"

#include "utils/logger.h"

ArcherAioThreadPool::ArcherAioThreadPool(int num_threads)
    : num_threads_(num_threads) {
  for (auto i = 0; i < num_threads_; ++i) {
    threads_.emplace_back(std::make_unique<ArcherAioThread>(i));
  }
}

ArcherAioThreadPool::~ArcherAioThreadPool() { Stop(); }

void ArcherAioThreadPool::Start() {
  for (auto& thread : threads_) {
    thread->Start();
  }
}

void ArcherAioThreadPool::Stop() {
  for (auto& thread : threads_) {
    thread->Stop();
  }
}

void ArcherAioThreadPool::Enqueue(AioCallback& callback, int thread_id) {
  if (thread_id < 0) {
    const auto idx = round_robin_counter_.fetch_add(1) % num_threads_;
    threads_[idx]->Enqueue(callback);
  } else {
    threads_[thread_id]->Enqueue(callback);
  }
}

void ArcherAioThreadPool::Wait() {
  for (auto& thread : threads_) {
    thread->Wait();
  }
}
