// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team
//
// Native FP8 (e4m3, 128x128 block-scaled) SwiGLU expert for the RTX PRO 6000
// Blackwell (sm_120, 188 SMs, 128 MiB L2). Hardware-gated: SM120 (CUTLASS
// sm120 collectives) + RTX_PRO_6000 (tuning); other GPUs keep the previous
// per-call dequant-to-BF16 path. See docs/hardware-gated-kernels.md.
//   output = (silu(x @ gate^T) * (x @ up^T)) @ down^T
// The FP8 weights are read once in their storage format and the block scales
// are applied inside the kernels; no BF16 weight copy is materialised.
//
//   M <= kFp8Pro6000GemvMaxTokens : weight-only FP8 GEMV on bf16 mma.sync
//   (exact
//                            weight de-quantisation in registers, BF16
//                            activations, fp32 accumulation).
//   M >  kFp8Pro6000GemvMaxTokens : CUTLASS sm_120 blockwise-scaled FP8 GEMM;
//   the
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

constexpr int64_t kFp8Pro6000GemvMaxTokens = 8;

// True when this build carries the sm_120a FP8 kernels (MOE_ENABLE_SM120_FP8),
// `device` (-1 = current) passes both hardware gates (Gate::kSm120 and
// Gate::kRtxPro6000, extensions/kernel/hw_profile.h) and both dims are
// multiples of 128. Otherwise callers keep the dequant-to-BF16 path.
bool fp8_moe_sm120_pro6000_supported(int64_t hidden, int64_t inter,
                                     int device = -1);

void fused_fp8_moe_ffn_sm120_pro6000_into(
    const torch::Tensor& hidden, const torch::Tensor& gate_w,
    const torch::Tensor& gate_s, const torch::Tensor& up_w,
    const torch::Tensor& up_s, const torch::Tensor& down_w,
    const torch::Tensor& down_s, torch::Tensor& output, cudaStream_t stream);
