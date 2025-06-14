// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <torch/extension.h>
#include "aio/archer_prio_aio_handle.h"
#include "base/noncopyable.h"

#define CPU_DEVICE torch::Device(torch::kCPU)
#define CUDA_DEVICE(index) torch::Device(torch::kCUDA, index)
#define DISK_DEVICE torch::Device(torch::kMeta)
#define DEFAULT_CUDA_DEVICE torch::Device(torch::kCUDA, 0)

#define TENSOR_OPTIONS(dtype, target) \
  torch::TensorOptions()              \
      .dtype(dtype)                   \
      .device(target)                 \
      .requires_grad(false)           \
      .memory_format(torch::MemoryFormat::Contiguous)

#define FLOAT32_TENSOR_OPTIONS(target) TENSOR_OPTIONS(torch::kFloat32, target)
#define FLOAT16_TENSOR_OPTIONS(target) TENSOR_OPTIONS(torch::kFloat16, target)
#define INT32_TENSOR_OPTIONS(target) TENSOR_OPTIONS(torch::kInt32, target)
#define INT64_TENSOR_OPTIONS(target) TENSOR_OPTIONS(torch::kInt64, target)
#define BFLOAT16_TENSOR_OPTIONS(target) TENSOR_OPTIONS(torch::kBFloat16, target)

#define FAKE_TENSOR_SIZES torch::IntArrayRef({1})

inline std::vector<uint32_t> list_to_vector(py::list list) {
  std::vector<uint32_t> vec;
  for (auto item : list) {
    vec.push_back(item.cast<uint32_t>());
  }
  return vec;
}

inline py::list vector_to_list(std::vector<uint32_t>& vec) {
  py::list list;
  for (auto item : vec) {
    list.append(item);
  }
  return list;
}

#define DTYPE_BFLOAT16 0
#define DTYPE_FLOAT32 1
#define DTYPE_FLOAT16 2
#define DTYPE_FP8_E4M3FN 3

inline torch::ScalarType dtype_to_torch(int dtype) {
  auto tensor_dtype = torch::kFloat32;
  switch (dtype) {
    case DTYPE_BFLOAT16:
      tensor_dtype = torch::kBFloat16;
      break;
    case DTYPE_FLOAT16:
      tensor_dtype = torch::kHalf;
      break;
    case DTYPE_FLOAT32:
      tensor_dtype = torch::kFloat32;
      break;
    case DTYPE_FP8_E4M3FN:
      tensor_dtype = torch::kFloat8_e4m3fn;
      break;
    default:
      assert(false);
  }
  return tensor_dtype;
}

inline int torch_dtype_to_int(torch::ScalarType dtype) {
  auto tensor_dtype = DTYPE_FLOAT32;
  switch (dtype) {
    case torch::kBFloat16:
      tensor_dtype = DTYPE_BFLOAT16;
      break;
    case torch::kHalf:
      tensor_dtype = DTYPE_FLOAT16;
      break;
    case torch::kFloat32:
      tensor_dtype = DTYPE_FLOAT32;
      break;
    case torch::kFloat8_e4m3fn:
      tensor_dtype = DTYPE_FP8_E4M3FN;
      break;
    default:
      assert(false);
  }
  return tensor_dtype;
}

inline size_t torch_dtype_size(int dtype) {
  size_t itemsize = 0;
  switch (dtype) {
    case DTYPE_BFLOAT16:
      itemsize = 2;  // bfloat16 is 2 bytes
      break;
    case DTYPE_FLOAT16:
      itemsize = 2;  // float16 is 2 bytes
      break;
    case DTYPE_FLOAT32:
      itemsize = 4;  // float32 is 4 bytes
      break;
    case DTYPE_FP8_E4M3FN:
      itemsize = 1;  // fp8_e4m3fn is 1 byte
      break;
    default:
      assert(false);  // Invalid dtype
  }
  return itemsize;
}

inline size_t torch_shape_size(const std::vector<int64_t>& shape, int dtype) {
  auto itemsize = torch_dtype_size(dtype);
  size_t size = 1;
  for (auto dim : shape) {
    size *= dim;
  }
  size *= itemsize;
  return size;
}
