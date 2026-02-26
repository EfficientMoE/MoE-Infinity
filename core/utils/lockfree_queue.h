#pragma once

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "base/noncopyable.h"

template <typename T>
class LockFreeQueue : public base::noncopyable {
 public:
  LockFreeQueue() {
    head_.store(nullptr);
    tail_.store(nullptr);
  }

  ~LockFreeQueue() = default;

  void Push(T value) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(std::move(value));
  }

  bool Pop(T& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return false;
    }
    value = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  bool Empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  bool Full() const {
    return false;  // Queue is unbounded
  }

 protected:
  mutable std::mutex mutex_;
  std::queue<T> queue_;
  std::atomic<void*> head_;
  std::atomic<void*> tail_;
};

template <typename T>
class LockFreeRecyclingQueue : public LockFreeQueue<T> {
 public:
  LockFreeRecyclingQueue() = default;

  void Pop(T& item) override {
    LockFreeQueue<T>::Pop(item);
    Push(item);
  }

  bool TryPop(T& item) override {
    bool success = LockFreeQueue<T>::TryPop(item);
    Push(item);
    return success;
  }
};
