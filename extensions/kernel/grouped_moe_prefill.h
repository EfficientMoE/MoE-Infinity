// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

// Grouped routed-expert FFN for prefill: every expert of one MoE layer in
// three launches (routing → sorted-token layout, grouped gate/up dual GEMM with
// act(g)·u epilogue, grouped down GEMM with routing weight + scatter-add
// epilogue). Each expert's weights are read in place through a pointer table.
//
// Same interface as grouped_moe_decode (#306):
//   hidden        [T, H]  bf16, stride(1) == 1, 16-byte aligned rows
//   router_mask   [T, E]  bool / uint8 (any strides)
//   router_weight [T, E]  float32 / bfloat16 / float16 (any strides)
//   expert_table  [n, 4]  int64 on the device: {gate*, up*, down*, expert_id}
//                         gate/up [I, H], down [H, I], contiguous bf16
//   out           [T, H]  float32, accumulated into (+=), stride(1) == 1
//   kcap          upper bound on routed experts per token (top-k); it sizes
//                 the [T * kcap, I] activation workspace.
//   zero_out      zero `out` inside the first launch instead of accumulating.
//
// RTX PRO 6000 only (gate RTX_PRO_6000, docs/hardware-gated-kernels.md): the
// grouped kernels, their tile tables, the persistent scheduler (grids sized to
// 188 SMs) and the L2 pass / combine thresholds (128 MiB L2) were tuned on that
// card. grouped_moe_prefill_pro6000_enabled(device) is evaluated once per
// device; elsewhere the same entry points run the experts one at a time
// through fused_moe_ffn_into (the previous per-expert execution).
//
// Rounding matches fused_moe_ffn_into + ExpertDispatcher's accumulate: the
// gate/up accumulators stay fp32 into act(g)·u, the activation and the down
// output are rounded to bf16, and the weighted sum is fp32.

bool grouped_moe_prefill_pro6000_enabled(int device);

// The fallback of grouped_moe_prefill on other GPUs (exported for tests): one
// fused_moe_ffn_into per expert and a weighted fp32 index_add; syncs the host.
void grouped_moe_prefill_per_expert(const torch::Tensor& hidden,
                                    const torch::Tensor& router_mask,
                                    const torch::Tensor& router_weight,
                                    const torch::Tensor& expert_table,
                                    int64_t inter, torch::Tensor& out,
                                    bool zero_out, cudaStream_t stream);

bool grouped_moe_prefill_supported(int64_t tokens, int64_t hidden,
                                   int64_t inter, int64_t experts);

void grouped_moe_prefill(const torch::Tensor& hidden,
                         const torch::Tensor& router_mask,
                         const torch::Tensor& router_weight,
                         const torch::Tensor& expert_table, int64_t inter,
                         int64_t kcap, torch::Tensor& out, bool zero_out,
                         cudaStream_t stream);

// As above with an explicit tile config per GEMM and combine mode (0 = fp32
// scatter-add in the down epilogue, 1 = bf16 rows + per-token combine; -1 =
// automatic for all three). For tests and tuning; explicit values need the
// RTX PRO 6000 path.
void grouped_moe_prefill_cfg(const torch::Tensor& hidden,
                             const torch::Tensor& router_mask,
                             const torch::Tensor& router_weight,
                             const torch::Tensor& expert_table, int64_t inter,
                             int64_t kcap, torch::Tensor& out, bool zero_out,
                             int dual_cfg, int down_cfg, int combine_mode,
                             cudaStream_t stream);
