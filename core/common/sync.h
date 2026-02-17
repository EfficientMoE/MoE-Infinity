#pragma once

#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <atomic>
#include <cerrno>
#include <stdexcept>

// Templated Futex class for atomic variable
template <typename T>
class Futex {
 public:
  Futex() { value_.store(0); }
  explicit Futex(T initial_value) : value_(initial_value) {}
  explicit Futex(const Futex<T>& other) : value_(other.value_.get()) {}

  void wait(T expected) {
    while (value_.load() != expected) {
      int ret = syscall(SYS_futex, &value_, FUTEX_WAIT, expected, nullptr,
                        nullptr, 0);
      if (ret == -1 && errno != EAGAIN) {
        throw std::runtime_error("Futex wait failed");
      }
    }
  }

  void wake(int count = 1) {
    int ret =
        syscall(SYS_futex, &value_, FUTEX_WAKE, count, nullptr, nullptr, 0);
    if (ret == -1) {
      throw std::runtime_error("Futex wake failed");
    }
  }

  void set(T new_value) { value_.store(new_value); }

  T get() const { return value_.load(); }

  void set_and_wake(T new_value, int count = 1) {
    value_.store(new_value);
    wake(count);
  }

  void wait_and_set(T expected, T new_value) {
    while (true) {
      T current = value_.load();
      if (current != expected) {
        int ret = syscall(SYS_futex, &value_, FUTEX_WAIT, current, nullptr,
                          nullptr, 0);
        if (ret == -1 && errno != EAGAIN) {
          throw std::runtime_error("Futex wait failed");
        }
      } else if (value_.compare_exchange_strong(current, new_value)) {
        // Successfully set the new value atomically
        break;
      }
    }
  }

 private:
  std::atomic<T> value_;
};
