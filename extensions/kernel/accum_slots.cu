// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

#include "kernel/accum_slots.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ float as_float(float v) { return v; }
__device__ __forceinline__ float as_float(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
__device__ __forceinline__ float as_float(__half v) { return __half2float(v); }

// One block per token: rank of every active expert (block-wide exclusive scan
// of the mask row), the token's slot count and overflow flag, and the zero
// fill of the token's slice of the slot buffer.
__global__ void __launch_bounds__(kThreads)
    prepare_kernel(const uint8_t* __restrict__ mask, int64_t mask_ld,
                   int64_t mask_cs, int E, int R, int rows, int H,
                   int* __restrict__ slot_index, float* __restrict__ slots,
                   unsigned long long* __restrict__ overflow_counter) {
  const int t = blockIdx.x;
  __shared__ int warp_total[kWarps];
  __shared__ int carry;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (threadIdx.x == 0) carry = 0;
  __syncthreads();
  for (int base = 0; base < E; base += kThreads) {
    const int e = base + threadIdx.x;
    const int flag =
        (e < E && mask[int64_t(t) * mask_ld + int64_t(e) * mask_cs] != 0) ? 1
                                                                          : 0;
    int v = flag;  // inclusive scan within the warp
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int n = __shfl_up_sync(0xffffffffu, v, o);
      if (lane >= o) v += n;
    }
    if (lane == 31) warp_total[warp] = v;
    __syncthreads();
    int before = 0;
    int total = 0;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      if (w < warp) before += warp_total[w];
      total += warp_total[w];
    }
    if (e < E) {
      const int rank = carry + before + v - flag;
      slot_index[int64_t(t) * E + e] = rank < R ? rank : R;
    }
    __syncthreads();
    if (threadIdx.x == 0) carry += total;
    __syncthreads();
  }
  if (threadIdx.x == 0 && carry > R && overflow_counter != nullptr) {
    atomicAdd(overflow_counter, 1ULL);
  }
  float* slice = slots + int64_t(t) * rows * H;
  const int64_t n = int64_t(rows) * H;
  if ((H & 3) == 0) {
    float4* p = reinterpret_cast<float4*>(slice);
    for (int64_t i = threadIdx.x; i < n / 4; i += kThreads) {
      p[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
  } else {
    for (int64_t i = threadIdx.x; i < n; i += kThreads) slice[i] = 0.f;
  }
}

template <typename Out, typename W>
__global__ void __launch_bounds__(kThreads)
    store_kernel(const Out* __restrict__ out, const int64_t* __restrict__ tok,
                 int H, int E, int expert, const W* __restrict__ rw,
                 int64_t rw_ld, int64_t rw_cs,
                 const int* __restrict__ slot_index, float* __restrict__ slots,
                 int R, int rows) {
  const int h = blockIdx.y * kThreads + threadIdx.x;
  if (h >= H) return;
  const int j = blockIdx.x;
  const int64_t t = tok[j];
  const float w = as_float(rw[t * rw_ld + int64_t(expert) * rw_cs]);
  const int idx = slot_index[t * E + expert];
  // Same value as OutputFunc's float(output) * weight.
  const float v = as_float(out[int64_t(j) * H + h]) * w;
  if (idx < R) {
    slots[(t * rows + idx) * H + h] = v;
  } else if (rows > R) {
    atomicAdd(slots + (t * rows + R) * H + h, v);
  }
}

template <typename Out>
void launch_store(const torch::Tensor& output, const torch::Tensor& tok,
                  int64_t expert, const torch::Tensor& rw,
                  const torch::Tensor& slot_index, torch::Tensor& slots,
                  int64_t R, cudaStream_t stream) {
  const int64_t n = output.size(0);
  const int H = static_cast<int>(output.size(1));
  const int E = static_cast<int>(rw.size(1));
  const int rows = static_cast<int>(slots.size(1));
  const dim3 grid(static_cast<unsigned>(n), (H + kThreads - 1) / kThreads);
  const Out* o = reinterpret_cast<const Out*>(output.data_ptr());
#define ACCUM_STORE_LAUNCH(WT)                                                \
  store_kernel<Out, WT><<<grid, kThreads, 0, stream>>>(                       \
      o, tok.data_ptr<int64_t>(), H, E, static_cast<int>(expert),             \
      reinterpret_cast<const WT*>(rw.data_ptr()), rw.stride(0), rw.stride(1), \
      slot_index.data_ptr<int>(), slots.data_ptr<float>(),                    \
      static_cast<int>(R), rows)
  switch (rw.scalar_type()) {
    case at::kFloat:
      ACCUM_STORE_LAUNCH(float);
      break;
    case at::kBFloat16:
      ACCUM_STORE_LAUNCH(__nv_bfloat16);
      break;
    default:
      ACCUM_STORE_LAUNCH(__half);
      break;
  }
#undef ACCUM_STORE_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void __launch_bounds__(kThreads)
    reduce_kernel(const float* __restrict__ slots, int rows, int R, int H,
                  int dump, float* __restrict__ out, int64_t ldo) {
  const int h = blockIdx.y * kThreads + threadIdx.x;
  if (h >= H) return;
  const int t = blockIdx.x;
  const float* base = slots + int64_t(t) * rows * H + h;
  float acc = dump ? base[int64_t(R) * H] : 0.f;
  for (int r = 0; r < R; ++r) acc += base[int64_t(r) * H];
  out[int64_t(t) * ldo + h] = acc;
}

bool is_fp(const torch::Tensor& t) {
  return t.scalar_type() == at::kFloat || t.scalar_type() == at::kBFloat16 ||
         t.scalar_type() == at::kHalf;
}

}  // namespace

void accum_slots_prepare(const torch::Tensor& router_mask, torch::Tensor& slots,
                         torch::Tensor& slot_index, int64_t slot_count,
                         const torch::Tensor& overflow_counter,
                         cudaStream_t stream) {
  TORCH_CHECK(router_mask.is_cuda() && router_mask.dim() == 2 &&
                  (router_mask.scalar_type() == at::kBool ||
                   router_mask.scalar_type() == at::kByte),
              "accum_slots_prepare: router_mask must be bool/uint8 [T, E] on "
              "CUDA");
  const int64_t T = router_mask.size(0);
  const int64_t E = router_mask.size(1);
  TORCH_CHECK(slots.is_cuda() && slots.scalar_type() == at::kFloat &&
                  slots.dim() == 3 && slots.is_contiguous() &&
                  slots.size(0) == T,
              "accum_slots_prepare: slots must be contiguous float32 [T, rows, "
              "H]");
  TORCH_CHECK(
      slot_index.is_cuda() && slot_index.scalar_type() == at::kInt &&
          slot_index.is_contiguous() && slot_index.numel() == T * E,
      "accum_slots_prepare: slot_index must be contiguous int32 [T, E]");
  TORCH_CHECK(slot_count >= 1 && slot_count <= slots.size(1),
              "accum_slots_prepare: slot_count must be in [1, rows]");
  TORCH_CHECK(!overflow_counter.defined() ||
                  (overflow_counter.is_cuda() &&
                   overflow_counter.scalar_type() == at::kLong &&
                   overflow_counter.numel() == 1),
              "accum_slots_prepare: overflow_counter must be int64 [1]");
  if (T == 0) return;
  c10::cuda::CUDAGuard guard(router_mask.device());
  prepare_kernel<<<static_cast<unsigned>(T), kThreads, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(router_mask.data_ptr()),
      router_mask.stride(0), router_mask.stride(1), static_cast<int>(E),
      static_cast<int>(slot_count), static_cast<int>(slots.size(1)),
      static_cast<int>(slots.size(2)), slot_index.data_ptr<int>(),
      slots.data_ptr<float>(),
      overflow_counter.defined() ? reinterpret_cast<unsigned long long*>(
                                       overflow_counter.data_ptr<int64_t>())
                                 : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void accum_slot_store(const torch::Tensor& output,
                      const torch::Tensor& token_indices, int64_t expert,
                      const torch::Tensor& router_weight,
                      const torch::Tensor& slot_index, torch::Tensor& slots,
                      int64_t slot_count, cudaStream_t stream) {
  TORCH_CHECK(output.is_cuda() && output.dim() == 2 && output.is_contiguous() &&
                  is_fp(output),
              "accum_slot_store: output must be contiguous float32/bf16/fp16 "
              "[n, H] on CUDA");
  TORCH_CHECK(
      token_indices.is_cuda() && token_indices.scalar_type() == at::kLong &&
          token_indices.dim() == 1 && token_indices.size(0) == output.size(0) &&
          token_indices.is_contiguous(),
      "accum_slot_store: token_indices must be int64 [n]");
  TORCH_CHECK(router_weight.is_cuda() && router_weight.dim() == 2 &&
                  is_fp(router_weight) && expert >= 0 &&
                  expert < router_weight.size(1),
              "accum_slot_store: router_weight must be float [T, E]");
  TORCH_CHECK(
      slots.is_cuda() && slots.scalar_type() == at::kFloat &&
          slots.dim() == 3 && slots.is_contiguous() &&
          slots.size(2) == output.size(1) && slot_count >= 1 &&
          slot_count <= slots.size(1),
      "accum_slot_store: slots must be contiguous float32 [T, rows, H]");
  TORCH_CHECK(slot_index.is_cuda() && slot_index.scalar_type() == at::kInt &&
                  slot_index.is_contiguous() &&
                  slot_index.numel() == slots.size(0) * router_weight.size(1),
              "accum_slot_store: slot_index must be contiguous int32 [T, E]");
  if (output.size(0) == 0) return;
  c10::cuda::CUDAGuard guard(output.device());
  switch (output.scalar_type()) {
    case at::kFloat:
      launch_store<float>(output, token_indices, expert, router_weight,
                          slot_index, slots, slot_count, stream);
      break;
    case at::kBFloat16:
      launch_store<__nv_bfloat16>(output, token_indices, expert, router_weight,
                                  slot_index, slots, slot_count, stream);
      break;
    default:
      launch_store<__half>(output, token_indices, expert, router_weight,
                           slot_index, slots, slot_count, stream);
      break;
  }
}

void accum_slots_reduce(const torch::Tensor& slots, int64_t slot_count,
                        bool has_dump_row, torch::Tensor& out,
                        cudaStream_t stream) {
  TORCH_CHECK(slots.is_cuda() && slots.scalar_type() == at::kFloat &&
                  slots.dim() == 3 && slots.is_contiguous(),
              "accum_slots_reduce: slots must be contiguous float32 [T, rows, "
              "H]");
  const int64_t T = slots.size(0);
  const int64_t H = slots.size(2);
  TORCH_CHECK(
      slot_count >= 1 && slot_count <= slots.size(1) &&
          (!has_dump_row || slots.size(1) > slot_count),
      "accum_slots_reduce: slot_count / dump row do not fit the buffer");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kFloat &&
                  out.dim() == 2 && out.size(0) == T && out.size(1) == H &&
                  out.stride(1) == 1,
              "accum_slots_reduce: out must be float32 [T, H] with stride(1) "
              "== 1");
  if (T == 0) return;
  c10::cuda::CUDAGuard guard(slots.device());
  const dim3 grid(static_cast<unsigned>(T),
                  static_cast<unsigned>((H + kThreads - 1) / kThreads));
  reduce_kernel<<<grid, kThreads, 0, stream>>>(
      slots.data_ptr<float>(), static_cast<int>(slots.size(1)),
      static_cast<int>(slot_count), static_cast<int>(H), has_dump_row ? 1 : 0,
      out.data_ptr<float>(), out.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
