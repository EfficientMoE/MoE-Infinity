#pragma once

#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <atomic>
#include <cerrno>
#include <stdexcept>
#include <type_traits>

// Templated Futex class for atomic variable
template <typename T>
class Futex {
 public:
  static_assert(std::is_integral_v<T>,
                "Futex<T> requires an integral futex-compatible type");
  static_assert(sizeof(T) == sizeof(int),
                "Futex<T> requires a 32-bit futex-compatible storage type");
  static_assert(
      sizeof(std::atomic<T>) == sizeof(int),
      "Futex<T> requires atomic storage compatible with futex 32-bit word");

  Futex() { value_.store(0); }
  explicit Futex(T initial_value) : value_(initial_value) {}
  explicit Futex(const Futex<T>& other) : value_(other.value_.load()) {}

  void wait(T expected) {
    while (value_.load() != expected) {
      int ret = syscall(SYS_futex, reinterpret_cast<int*>(&value_), FUTEX_WAIT,
                        static_cast<int>(expected), nullptr, nullptr, 0);
      if (ret == -1 && errno != EAGAIN) {
        throw std::runtime_error("Futex wait failed");
      }
    }
  }

  void wake(int count = 1) {
    int ret = syscall(SYS_futex, reinterpret_cast<int*>(&value_), FUTEX_WAKE,
                      count, nullptr, nullptr, 0);
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
        int ret =
            syscall(SYS_futex, reinterpret_cast<int*>(&value_), FUTEX_WAIT,
                    static_cast<int>(current), nullptr, nullptr, 0);
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
