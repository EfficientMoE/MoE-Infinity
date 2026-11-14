// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// Grouped block-128 FP8 SwiGLU on the RTX PRO 6000.
// Average rows per expert <= 8: weight-only GEMV (bf16 mma.sync).
// Average rows > 8: the same math with the 16-row weight tile staged in
// smem so each expert's weights are read once. No BF16 weight copy.

#pragma once

#include <torch/extension.h>

// x: bf16 [T, H]
// topk_ids: int32 [T, K]
// topk_w: fp32 [T, K]
// table: int64 [E, 6] = gate, gate_scale, up, up_scale, down, down_scale
// Returns fp32 [T, H], the sum over slots of weight * FFN(x).
torch::Tensor fp8_moe_grouped_forward_sm120_pro6000(
    const torch::Tensor& x, const torch::Tensor& topk_ids,
    const torch::Tensor& topk_w, const torch::Tensor& table, int64_t inter);
