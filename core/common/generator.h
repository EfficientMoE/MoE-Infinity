// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <uuid/uuid.h>

#include <bitset>
#include <chrono>
#include <iomanip>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <atomic>

class NumGenerator {
 public:
  // 0是一个特殊的id，必须保证永远不会生成0这个id
  static uint32_t ctx_id() {
    std::lock_guard g(mutex_);
    uint32_t ret = ctx_id_++;
    if (ret == 0) ret = ctx_id_++;
    return ret;
  }
  static uint32_t flowno() {
    static std::atomic<uint32_t> flowno(1024);
    return flowno++;
  }

 private:
  static std::mutex mutex_;
  static uint32_t ctx_id_;  // Start from 1 to avoid 0
};

// Static member definitions
std::mutex NumGenerator::mutex_;
uint32_t NumGenerator::ctx_id_ = 1;

inline std::string GenUUID() {
  uuid_t uuid;
  uuid_generate(uuid);
  char uuid_str[37];
  uuid_unparse(uuid, uuid_str);
  return std::string(uuid_str);
}

inline uint64_t GenUUID64() {
  static std::random_device rd;
  static std::mt19937_64 eng(rd());
  static std::uniform_int_distribution<uint64_t> distr;

  std::bitset<64> uuid;
  uuid = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  uuid ^= distr(eng);

  return uuid.to_ullong();
}

inline std::string CurrentTimeString() {
  // Get current time as time_point
  auto now = std::chrono::system_clock::now();

  // Convert time_point to system time for breaking down into components
  auto now_c = std::chrono::system_clock::to_time_t(now);
  auto now_tm = *std::localtime(&now_c);

  // Get the current time as milliseconds
  auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now.time_since_epoch()) %
                1000;

  // Use stringstream to format the time
  std::ostringstream oss;
  oss << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S");
  oss << '.' << std::setfill('0') << std::setw(3) << now_ms.count();

  return oss.str();
}

// constexpr microseconds since epoch
inline uint64_t CurrentTimeMicros() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}
