// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// GPT-OSS MXFP4 expert forward on the stored blocks/scales (no per-call
// de-quantisation of gate_up). See mxfp4_gptoss_moe.cu for the kernels.

#pragma once

#include <cuda_runtime.h>
#include <torch/torch.h>

// Scratch owned by one expert executor; used by one stream at a time.
// Allocated lazily on the device of the first call.
struct GptOssExpertScratch {
  torch::Tensor partials_gu;  // uint8, stream-K partials of the gate_up GEMM
  torch::Tensor partials_d;   // uint8, stream-K partials of the down GEMM
  torch::Tensor counters;     // int32 [2 * kMxfp4StreamKCounters], zeroed
  torch::Tensor wd_bf16;      // [H, I] BF16 down weights (dense-down path)
};

// From this many tokens the down projection runs as a BF16 cuBLAS GEMM on
// weights de-quantised by the gate_up kernel (overlapped with its MMAs);
// below it both projections stay MXFP4.
constexpr int64_t kGptOssDenseDownMinTokens = 320;

// out[M, H] = SwiGLU-OAI(x . Wgu^T + b_gu) . Wd^T + b_d with act[M, I] as
// the intermediate. x / act / out: BF16, row-major, on one CUDA device;
// blocks / scales: uint8 MXFP4 (gate_up [2I, H/2] / [2I, H/32], down
// [H, I/2] / [H, I/32]); biases: [2I] / [H]. Runs on `stream`. Returns false
// (and launches nothing) when the inputs are outside what the kernels cover,
// so the caller can fall back to de-quantise + matmul.
bool gptoss_mxfp4_expert_forward(
    const torch::Tensor& x, const torch::Tensor& gu_blocks,
    const torch::Tensor& gu_scales, const torch::Tensor& gu_bias,
    const torch::Tensor& d_blocks, const torch::Tensor& d_scales,
    const torch::Tensor& d_bias, torch::Tensor& act, torch::Tensor& out,
    GptOssExpertScratch& scratch, cudaStream_t stream);
