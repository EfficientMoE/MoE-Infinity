// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <cstdint>
#include <mutex>
#include <unordered_map>

enum class DataType { BFLOAT16 = 0, FLOAT32 = 1, FLOAT16 = 2, FP8_E4M3FN = 3 };

struct Context {
  // Add any necessary member variables or methods here
  int64_t max_expert_tokens = 128;  // Default maximum expert tokens
  int64_t max_tokens = 4096;        // Default maximum tokens
  int num_experts = 8;              // Default number of experts
  int topk = 2;                     // Default top-k value
  int64_t hidden_dim = 1024;        // Default hidden dimension
  int64_t intermediate_dim =
      4096;  // Default intermediate dimension for experts
  DataType dtype = DataType::FLOAT32;  // Default data type

  void SetFromDict(const std::unordered_map<std::string, int64_t>& dict) {
    if (dict.find("max_expert_tokens") != dict.end()) {
      max_expert_tokens = dict.at("max_expert_tokens");
    }
    if (dict.find("max_tokens") != dict.end()) {
      max_tokens = dict.at("max_tokens");
    }
    if (dict.find("num_experts") != dict.end()) {
      num_experts = dict.at("num_experts");
    }
    if (dict.find("topk") != dict.end()) {
      topk = dict.at("topk");
    }
    if (dict.find("hidden_dim") != dict.end()) {
      hidden_dim = dict.at("hidden_dim");
    }
    if (dict.find("intermediate_dim") != dict.end()) {
      intermediate_dim = dict.at("intermediate_dim");
    }
    if (dict.find("dtype") != dict.end()) {
      int dtype_value = dict.at("dtype");
      switch (dtype_value) {
        case 0:
          dtype = DataType::BFLOAT16;
          break;
        case 1:
          dtype = DataType::FLOAT32;
          break;
        case 2:
          dtype = DataType::FLOAT16;
          break;
        case 3:
          dtype = DataType::FP8_E4M3FN;
          break;
        default:
          throw std::invalid_argument("Invalid dtype value");
      }
    }
  }
};

Context& getContext() {
  static Context instance;
  return instance;
}
