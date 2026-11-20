// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// RTX PRO 6000 ONLY (gate RTX_PRO_6000, see docs/hardware-gated-kernels.md).
// The instructions are generic sm_80+ (mma.sync m16n8k16 BF16), but the
// split-K / row-split warp layouts, unroll depths and launch table were
// measured on the RTX PRO 6000 Blackwell Max-Q (188 SMs, 128 MiB L2). On any
// other device grouped_moe_decode_pro6000() refuses to run and
// ExpertDispatcher keeps per-expert execution (fused_moe_ffn_into).

#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

// Grouped routed-expert FFN for decode-sized batches: every expert of one MoE
// layer in two kernel launches (gate/up + SiLU, then down + routing weight +
// scatter-add), reading each expert's weights in place through a pointer table.
//
// Math and rounding points match fused_moe_ffn_into + ExpertDispatcher's
// accumulate (bf16 gate, bf16 silu(gate)*up, bf16 down, fp32 weighted sum).
//
// Tensor conventions:
//   hidden        [T, H]  bf16, stride(1) == 1, 16-byte aligned rows
//   router_mask   [T, E]  bool / uint8 (any strides)
//   router_weight [T, E]  float32 / bfloat16 / float16 (any strides)
//   expert_table  [n, 4]  int64 on the device: {gate*, up*, down*, expert_id}
//                         gate/up [I, H], down [H, I], contiguous bf16
//   out           [T, H]  float32, accumulated into (+=), stride(1) == 1
//   kcap          upper bound on routed experts per token (top-k); it sizes
//                 the [T * kcap, I] activation workspace.
//   zero_out      zero `out` inside the first launch instead of accumulating.
//   slots, slot_rank, slot_count   (optional; slot mode, undefined = off)
//                 Deterministic accumulation: instead of atomicAdd into `out`,
//                 the down launch stores each expert's weighted value into
//                 slots[t, slot_rank[t, e], :]. slots is float32 [T, rows, H]
//                 (zero-filled by the caller, see accum_slots.h), slot_rank
//                 int32 [T, E] holds the rank of expert e among ALL of token
//                 t's active experts (ascending expert index) clamped to
//                 slot_count, slot_count (<= rows) is the number of usable
//                 ranks. Rank == slot_count adds atomically to the dump row
//                 slots[t, slot_count, :] when rows > slot_count (otherwise to
//                 `out`). The caller reduces the slots in a fixed order.

constexpr int kGroupedMoePro6000MaxTokens = 256;

// True when `device` is an RTX PRO 6000 (moe_infinity::hw::Gate::kRtxPro6000).
bool grouped_moe_decode_pro6000_device_ok(int device);

// True when the shapes are supported by the grouped kernels (the caller falls
// back to per-expert execution otherwise).
bool grouped_moe_decode_pro6000_supported(int64_t tokens, int64_t hidden,
                                          int64_t inter);

void grouped_moe_decode_pro6000(
    const torch::Tensor& hidden, const torch::Tensor& router_mask,
    const torch::Tensor& router_weight, const torch::Tensor& expert_table,
    int64_t inter, int64_t kcap, torch::Tensor& out, bool zero_out,
    cudaStream_t stream, const torch::Tensor& slots = torch::Tensor(),
    const torch::Tensor& slot_rank = torch::Tensor(), int64_t slot_count = 0);
