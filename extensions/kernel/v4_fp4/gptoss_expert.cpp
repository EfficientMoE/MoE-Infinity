// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

#include "gptoss_expert.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <optional>

#include "mxfp4_gptoss_moe.h"

namespace {

bool is_mxfp4(const torch::Tensor& blocks, const torch::Tensor& scales,
              int64_t rows, int64_t k, const torch::Device& device) {
  return blocks.device() == device && scales.device() == device &&
         blocks.scalar_type() == torch::kUInt8 &&
         scales.scalar_type() == torch::kUInt8 && blocks.is_contiguous() &&
         scales.is_contiguous() && blocks.dim() == 2 && scales.dim() == 2 &&
         blocks.size(0) == rows && blocks.size(1) == k / 2 &&
         scales.size(0) == rows && scales.size(1) == k / 32;
}

bool is_vec(const torch::Tensor& t, int64_t n, const torch::Device& device) {
  return t.device() == device && t.numel() == n;
}

// Biases are BF16 in GPT-OSS checkpoints; anything else is converted.
torch::Tensor as_bf16(const torch::Tensor& t) {
  if (t.scalar_type() == torch::kBFloat16 && t.is_contiguous()) return t;
  return t.reshape({-1}).to(torch::kBFloat16).contiguous();
}

void ensure_bytes(torch::Tensor& t, size_t bytes, const torch::Device& d) {
  if (!t.defined() || static_cast<size_t>(t.numel()) < bytes || t.device() != d)
    t = torch::empty({static_cast<int64_t>(bytes)},
                     torch::TensorOptions().dtype(torch::kUInt8).device(d));
}

Mxfp4Workspace workspace(torch::Tensor& partials, int* counters) {
  return Mxfp4Workspace{partials.data_ptr(),
                        static_cast<size_t>(partials.numel()), counters,
                        kMxfp4StreamKCounters};
}

}  // namespace

bool gptoss_mxfp4_expert_forward(
    const torch::Tensor& x, const torch::Tensor& gu_blocks,
    const torch::Tensor& gu_scales, const torch::Tensor& gu_bias,
    const torch::Tensor& d_blocks, const torch::Tensor& d_scales,
    const torch::Tensor& d_bias, torch::Tensor& act, torch::Tensor& out,
    GptOssExpertScratch& scratch, cudaStream_t stream) {
  if (!x.is_cuda() || x.dim() != 2 || x.scalar_type() != torch::kBFloat16 ||
      !x.is_contiguous())
    return false;
  const auto device = x.device();
  const int64_t M = x.size(0);
  const int64_t H = x.size(1);
  const int64_t I = d_blocks.dim() == 2 ? d_blocks.size(1) * 2 : 0;
  if (M == 0) return true;
  if (I == 0 || H % 32 != 0 || I % 32 != 0 ||
      !is_mxfp4(gu_blocks, gu_scales, 2 * I, H, device) ||
      !is_mxfp4(d_blocks, d_scales, H, I, device) ||
      !is_vec(gu_bias, 2 * I, device) || !is_vec(d_bias, H, device))
    return false;
  for (const auto* t : {&act, &out}) {
    if (t->device() != device || t->scalar_type() != torch::kBFloat16 ||
        !t->is_contiguous() || t->dim() != 2 || t->size(0) != M)
      return false;
  }
  if (act.size(1) != I || out.size(1) != H) return false;

  // The launchers query the current device; executor threads already run on
  // the expert's device, so only switch when needed.
  std::optional<c10::cuda::CUDAGuard> device_guard;
  if (c10::cuda::current_device() != device.index())
    device_guard.emplace(device);
  const auto gu_b = as_bf16(gu_bias);
  const auto d_b = as_bf16(d_bias);

  // Stream-K partials only for the tiled kernels.
  const bool tiled = M > kMxfp4GemvMaxTokens;
  const size_t ws_bytes = tiled ? mxfp4_linear_workspace_bytes() : 0;
  if (tiled) ensure_bytes(scratch.partials_gu, ws_bytes, device);
  if (!scratch.counters.defined() || scratch.counters.device() != device) {
    scratch.counters = torch::zeros(
        {2 * kMxfp4StreamKCounters},
        torch::TensorOptions().dtype(torch::kInt32).device(device));
  }
  int* counters = scratch.counters.data_ptr<int>();
  Mxfp4Workspace ws_gu =
      tiled ? workspace(scratch.partials_gu, counters) : Mxfp4Workspace{};

  cudaError_t err;
  if (M >= kGptOssDenseDownMinTokens) {
    // gate_up (MXFP4, fused SwiGLU) de-quantises Wd on the side; the down
    // projection is then one cuBLAS GEMM with the bias in its epilogue.
    if (!scratch.wd_bf16.defined() || scratch.wd_bf16.device() != device ||
        scratch.wd_bf16.size(0) != H || scratch.wd_bf16.size(1) != I) {
      scratch.wd_bf16 = torch::empty(
          {H, I},
          torch::TensorOptions().dtype(torch::kBFloat16).device(device));
    }
    const Mxfp4Side side{d_blocks.data_ptr(), d_scales.data_ptr(),
                         scratch.wd_bf16.data_ptr(),
                         static_cast<int>(H * I / 32)};
    err = mxfp4_linear_cuda(x.data_ptr(), M, H, gu_blocks.data_ptr(),
                            gu_scales.data_ptr(), gu_b.data_ptr(), 2 * I,
                            act.data_ptr(), true, false, &ws_gu, &side, -1,
                            stream);
    if (err == cudaErrorNotSupported) return false;
    TORCH_CHECK(err == cudaSuccess,
                "GPT-OSS gate_up kernel: ", cudaGetErrorString(err));
    c10::cuda::CUDAStreamGuard stream_guard(
        c10::cuda::getStreamFromExternal(stream, device.index()));
    at::addmm_out(out, d_b, act, scratch.wd_bf16.t());
    return true;
  }

  if (tiled) ensure_bytes(scratch.partials_d, ws_bytes, device);
  Mxfp4Workspace ws_d =
      tiled ? workspace(scratch.partials_d, counters + kMxfp4StreamKCounters)
            : Mxfp4Workspace{};
  err = gptoss_mxfp4_expert_cuda(
      x.data_ptr(), M, H, I, gu_blocks.data_ptr(), gu_scales.data_ptr(),
      gu_b.data_ptr(), d_blocks.data_ptr(), d_scales.data_ptr(), d_b.data_ptr(),
      act.data_ptr(), out.data_ptr(), &ws_gu, &ws_d, stream);
  if (err == cudaErrorNotSupported) return false;
  TORCH_CHECK(err == cudaSuccess,
              "GPT-OSS MXFP4 expert kernels: ", cudaGetErrorString(err));
  return true;
}
