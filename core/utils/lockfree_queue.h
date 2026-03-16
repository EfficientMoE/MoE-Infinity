#pragma once

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "base/noncopyable.h"

template <typename T>
class MutexQueue : public base::noncopyable {
 public:
  MutexQueue() = default;

  ~MutexQueue() = default;

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
};

template <typename T>
class LockFreeRecyclingQueue : public MutexQueue<T> {
 public:
  LockFreeRecyclingQueue() = default;

  bool Pop(T& item) {
    bool success = MutexQueue<T>::Pop(item);
    if (success) {
      this->Push(item);
    }
    return success;
  }

  bool TryPop(T& item) { return Pop(item); }
};
