#pragma once

#include <cassert>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

enum class ActFunc {
  SiLU,
  ReLU,
  GeLU,
};

// Device activation function implementations
template <typename T>
__device__ __forceinline__ T relu(T x) {
  return fmaxf(x, T(0.0f));
}

template <typename T>
__device__ __forceinline__ T silu(T x) {
  return x / (T(1.0f) + expf(-x));
}

template <typename T>
__device__ __forceinline__ T gelu(T x) {
  // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  const T sqrt_2_over_pi = T(0.7978845608f);
  const T coeff = T(0.044715f);
  T x_cubed = x * x * x;
  T inner = sqrt_2_over_pi * (x + coeff * x_cubed);
  return T(0.5f) * x * (T(1.0f) + tanhf(inner));
}

// Specializations for __nv_bfloat16
template <>
__device__ __forceinline__ __nv_bfloat16 relu(__nv_bfloat16 x) {
  return __hmax(x, __float2bfloat16(0.0f));
}

template <>
__device__ __forceinline__ __nv_bfloat16 silu(__nv_bfloat16 x) {
  float x_f = __bfloat162float(x);
  float result = x_f / (1.0f + expf(-x_f));
  return __float2bfloat16(result);
}

template <>
__device__ __forceinline__ __nv_bfloat16 gelu(__nv_bfloat16 x) {
  float x_f = __bfloat162float(x);
  const float sqrt_2_over_pi = 0.7978845608f;
  const float coeff = 0.044715f;
  float x_cubed = x_f * x_f * x_f;
  float inner = sqrt_2_over_pi * (x_f + coeff * x_cubed);
  float result = 0.5f * x_f * (1.0f + tanhf(inner));
  return __float2bfloat16(result);
}

// Specializations for half precision
#ifdef __CUDA_ARCH__
template <>
__device__ __forceinline__ half relu(half x) {
  return __hmax(x, __float2half(0.0f));
}

template <>
__device__ __forceinline__ half silu(half x) {
  float x_f = __half2float(x);
  float result = x_f / (1.0f + expf(-x_f));
  return __float2half(result);
}

template <>
__device__ __forceinline__ half gelu(half x) {
  float x_f = __half2float(x);
  const float sqrt_2_over_pi = 0.7978845608f;
  const float coeff = 0.044715f;
  float x_cubed = x_f * x_f * x_f;
  float inner = sqrt_2_over_pi * (x_f + coeff * x_cubed);
  float result = 0.5f * x_f * (1.0f + tanhf(inner));
  return __float2half(result);
}
#endif

template <typename T, typename fragment_t>
__host__ __device__ void warp_activation(ActFunc activation,
                                         const fragment_t& frag,
                                         fragment_t& result) {
  switch (activation) {
    case ActFunc::ReLU:
#pragma unroll
      for (int t = 0; t < result.num_elements; t++) {
        result.x[t] = relu(static_cast<T>(frag.x[t]));
      }
      return;
    case ActFunc::SiLU:
#pragma unroll
      for (int t = 0; t < result.num_elements; t++) {
        result.x[t] = silu(static_cast<T>(frag.x[t]));
      }
      return;
    case ActFunc::GeLU:
#pragma unroll
      for (int t = 0; t < result.num_elements; t++) {
        result.x[t] = gelu(static_cast<T>(frag.x[t]));
      }
      return;
    default:
      // Unsupported activation
#ifdef __CUDA_ARCH__
      // On device, we can't use assert, so we'll set result to zero
      for (int t = 0; t < result.num_elements; t++) {
        result.x[t] = T(0);
      }
#else
      assert(false && "Unsupported activation function");
#endif
      return;
  }
}
