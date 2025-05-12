#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

#include "memory/device_caching_allocator.h"

#define TILE_DIM 32
#define NUM_THREADS 32

__device__ inline float to_float(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

__device__ inline __nv_bfloat16 from_float(float x) {
  return __float2bfloat16(x);
}

// __global__ void fused_moe_ffn_kernel(
//     const __nv_bfloat16* __restrict__ hidden,  // [M, K]
//     const __nv_bfloat16* __restrict__ w1,      // [N, K]
//     const __nv_bfloat16* __restrict__ w2,      // [N, K]
//     const __nv_bfloat16* __restrict__ w3,      // [K, N]
//     __nv_bfloat16* __restrict__ output,        // [M, K]
//     int M, int K, int N) {

//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;

//     if (row >= M || col >= K) return;

//     float acc1 = 0.f, acc2 = 0.f;
//     for (int i = 0; i < N; ++i) {
//         float h = 0.f;
//         for (int k = 0; k < K; ++k) {
//             h += __bfloat162float(hidden[row * K + k]) *
//             __bfloat162float(w1[i * K + k]);
//         }
//         acc1 = h;

//         h = 0.f;
//         for (int k = 0; k < K; ++k) {
//             h += __bfloat162float(hidden[row * K + k]) *
//             __bfloat162float(w2[i * K + k]);
//         }
//         acc2 = h;

//         float silu = acc1 / (1.0f + expf(-acc1));
//         float fused = silu * acc2;

//         float out = 0.0f;
//         for (int k = 0; k < K; ++k) {
//             out += fused * __bfloat162float(w3[k * N + col]);
//         }

//         output[row * K + col] = __float2bfloat16(out);
//     }
// }

__global__ void fused_moe_ffn_kernel(
    const __nv_bfloat16* __restrict__ hidden,  // [M, K]
    const __nv_bfloat16* __restrict__ w1,      // [N, K]
    const __nv_bfloat16* __restrict__ w2,      // [N, K]
    const __nv_bfloat16* __restrict__ w3,      // [K, N]
    __nv_bfloat16* __restrict__ output,        // [M, K]
    int M, int K, int N) {
  extern __shared__ __nv_bfloat16 smem[];

  int local_row = threadIdx.y;
  int local_col = threadIdx.x;
  int row = blockIdx.y * blockDim.y + local_row;
  int col = blockIdx.x * blockDim.x + local_col;

  if (row >= M || col >= K) return;

  // Shared memory slice for tile of hidden vector
  __nv_bfloat16* tile_hidden = smem;  // [TILE_DIM * TILE_DIM]
  int tile_idx = local_row * TILE_DIM + local_col;
  tile_hidden[tile_idx] =
      (row < M && col < K) ? hidden[row * K + col] : __float2bfloat16(0.f);

  __syncthreads();

  float acc1 = 0.f, acc2 = 0.f;
  for (int i = 0; i < N; ++i) {
    float h1 = 0.f, h2 = 0.f;
    for (int k = 0; k < TILE_DIM && k + col < K; ++k) {
      int widx = i * K + (col + k);
      float x = to_float(tile_hidden[local_row * TILE_DIM + k]);
      h1 += x * to_float(w1[widx]);
      h2 += x * to_float(w2[widx]);
    }
    acc1 += h1;
    acc2 += h2;
  }

  acc1 = to_float(from_float(acc1));
  acc2 = to_float(from_float(acc2));

  float silu = acc1 / (1.0f + expf(-acc1));
  float fused = silu * acc2;

  float out = 0.0f;
  for (int k = 0; k < N; ++k) {
    int widx = k * N + col;
    out += fused * to_float(w3[widx]);
  }

  output[row * K + col] = from_float(out);
}

struct torch_deleter {
  void operator()(void* ptr) const {
    if (ptr != nullptr) {
      // auto allocator = c10::cuda::CUDACachingAllocator::get();
      // TORCH_CHECK(allocator != nullptr, "CUDACachingAllocator is not
      // initialized"); allocator->raw_delete(ptr);
      c10::DeviceCachingAllocator::get(device_id)->free(ptr);
    }
  }
  int device_id;
};

// // Batch multiple FFNs in one kernel call
// void launch_batch_fused_moe_ffns(
//     const std::vector<torch::Tensor>& hiddens,
//     const std::vector<torch::Tensor>& w1s,
//     const std::vector<torch::Tensor>& w2s,
//     const std::vector<torch::Tensor>& w3s,
//     const std::vector<torch::Tensor>& outputs,
//     cudaStream_t stream,
//     bool sync_after = true)
// {
//     TORCH_CHECK(hiddens.size() == w1s.size() && w1s.size() == w2s.size() &&
//     w2s.size() == w3s.size() && w3s.size() == outputs.size(), "Mismatched
//     batch sizes");

//     for (size_t i = 0; i < hiddens.size(); ++i) {
//         launch_fused_moe_ffn(hiddens[i], w1s[i], w2s[i], w3s[i], outputs[i],
//         stream);
//     }

//     if (sync_after) {
//         cudaStreamSynchronize(stream);
//     }
// }
