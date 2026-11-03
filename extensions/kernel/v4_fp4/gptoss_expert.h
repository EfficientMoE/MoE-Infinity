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
  // uint8 arena holding the op's intermediates, in order: BF16 Wd [H, I]
  // (dense-down path), act [M, I] BF16, the gate_up then the down stream-K
  // partials. Its prefix is kept L2-resident while the op runs.
  torch::Tensor arena;
  torch::Tensor counters;  // int32 [2 * kMxfp4StreamKCounters], zeroed
};

// From this many tokens the down projection runs as a BF16 cuBLAS GEMM on
// weights de-quantised by the gate_up kernel (overlapped with its MMAs);
// below it both projections stay MXFP4.
constexpr int64_t kGptOssDenseDownMinTokens = 320;

// From this many tokens the arena gets an L2 persisting window for the op
// (Wd, act and partials stay in L2 between the launches), and Wd / act are
// discarded from L2 after it (one extra launch), so they never reach DRAM.
constexpr int64_t kGptOssL2ResidentMinTokens = 257;

// Bytes of GptOssExpertScratch::arena for M tokens.
size_t gptoss_expert_arena_bytes(int64_t M, int64_t H, int64_t I);

// out[M, H] = SwiGLU-OAI(x . Wgu^T + b_gu) . Wd^T + b_d; the intermediate
// act[M, I] lives in the scratch arena. x / out: BF16, row-major, on one CUDA
// device; blocks / scales: uint8 MXFP4 (gate_up [2I, H/2] / [2I, H/32], down
// [H, I/2] / [H, I/32]); biases: [2I] / [H]. Runs on `stream`. Returns false
// (and launches nothing) when the inputs are outside what the kernels cover,
// so the caller can fall back to de-quantise + matmul.
bool gptoss_mxfp4_expert_forward(
    const torch::Tensor& x, const torch::Tensor& gu_blocks,
    const torch::Tensor& gu_scales, const torch::Tensor& gu_bias,
    const torch::Tensor& d_blocks, const torch::Tensor& d_scales,
    const torch::Tensor& d_bias, torch::Tensor& out,
    GptOssExpertScratch& scratch, cudaStream_t stream);
