// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team
//
// Native FP8 (e4m3, 128x128 block-scaled) SwiGLU expert:
//   output = (silu(x @ gate^T) * (x @ up^T)) @ down^T
// The FP8 weights are read once in their storage format and the block scales
// are applied inside the kernels; no BF16 weight copy is materialised.
//
//   M <= kFp8GemvMaxTokens : weight-only FP8 GEMV (BF16 activations, fp32
//                            accumulation; exact weight de-quantisation).
//   M >  kFp8GemvMaxTokens : CUTLASS sm_120 blockwise-scaled FP8 GEMM; the
//                            activations are quantised per token and per 128
//                            channels to e4m3 with an fp32 scale (W8A8-block).
//
// Tensor conventions (HuggingFace weight shapes, row-major):
//   hidden  [M, H] bf16          output [M, H] bf16
//   gate_w  [I, H] e4m3 / uint8  gate_s [I/128, H/128] fp32
//   up_w    [I, H] e4m3 / uint8  up_s   [I/128, H/128] fp32
//   down_w  [H, I] e4m3 / uint8  down_s [H/128, I/128] fp32

#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

constexpr int64_t kFp8GemvMaxTokens = 8;

// True when this build carries the sm_120a FP8 kernels, the current device is
// sm_120 and both dims are multiples of 128.
bool fp8_moe_native_supported(int64_t hidden, int64_t inter);

void fused_fp8_moe_ffn_into(
    const torch::Tensor& hidden, const torch::Tensor& gate_w,
    const torch::Tensor& gate_s, const torch::Tensor& up_w,
    const torch::Tensor& up_s, const torch::Tensor& down_w,
    const torch::Tensor& down_s, torch::Tensor& output, cudaStream_t stream);
