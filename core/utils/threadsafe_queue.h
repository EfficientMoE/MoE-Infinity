#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>

#include "base/noncopyable.h"

template <typename T>
class ThreadSafeQueue : public base::noncopyable {
 public:
  ThreadSafeQueue() = default;

  // Disable copy constructor and assignment to avoid accidental data races.
  ThreadSafeQueue(const ThreadSafeQueue&) = delete;
  ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;

  // Pushes an item into the queue (thread-safe).
  void Push(T& item) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (closed_) {
        return;
      }
      queue_.push(std::move(item));
    }
    cond_.notify_one();
  }

  // Pops an item from the queue (blocking).
  virtual bool Pop(T& item) {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this] { return closed_ || !queue_.empty(); });

    if (queue_.empty()) {
      return false;
    }

    item = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  // Tries to pop an item without blocking. Returns false if the queue is empty.
  virtual bool TryPop(T& item) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return false;
    }
    item = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  // Returns true if the queue is empty.
  bool Empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  void NotifyAll() {
    // std::lock_guard<std::mutex> lock(mutex_);
    cond_.notify_all();
  }

  void Close() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
    }
    cond_.notify_all();
  }

 protected:
  std::queue<T> queue_;
  mutable std::mutex mutex_;
  std::condition_variable cond_;
  bool closed_ = false;
};

// recycling queue implementation, popped item is pushed back to the queue
template <typename T>
class ThreadSafeRecyclingQueue : public ThreadSafeQueue<T> {
 public:
  ThreadSafeRecyclingQueue() = default;

  bool Pop(T& item) override {
    bool result = ThreadSafeQueue<T>::Pop(item);
    if (result) {
      this->Push(item);
    }
    return result;
  }

  bool TryPop(T& item) override {
    bool success = ThreadSafeQueue<T>::TryPop(item);
    if (success) {
      this->Push(item);
    }
    return success;
  }
};
