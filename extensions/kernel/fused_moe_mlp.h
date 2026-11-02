#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

#include <utility>

// Fused MoE MLP forward pass using CUTLASS.
//
// Computes: output = (silu(input @ gate_proj^T) * (input @ up_proj^T)) @
// down_proj^T
//
// Two launches: a dual GEMM computes both projections from one read of the
// input and writes only silu(gate) * up to fused_buf; a second GEMM applies
// down_proj. Tile shape and split-K are chosen per (M, N, K).
//
// Tensor conventions (HuggingFace weight shapes):
//   hidden    [M, H]   row-major   (M = batch, H = hidden_size)
//   gate_proj [I, H]   row-major   (I = intermediate_size)
//   up_proj   [I, H]   row-major
//   down_proj [H, I]   row-major
//   gate_buf  [M, I]   row-major   (unused; kept for API compatibility)
//   fused_buf [M, I]   row-major   (intermediate: silu(gate)*up result)
//   output    [M, H]   row-major
void fused_moe_ffn_into(torch::Tensor& hidden,     // [M, H]
                        torch::Tensor& gate_proj,  // [I, H]
                        torch::Tensor& up_proj,    // [I, H]
                        torch::Tensor& down_proj,  // [H, I]
                        torch::Tensor& gate_buf,   // [M, I]
                        torch::Tensor& fused_buf,  // [M, I]
                        torch::Tensor& output,     // [M, H]
                        cudaStream_t stream);

// Same as fused_moe_ffn_into with an explicit tile config / split-K factor per
// GEMM (cfg < 0 selects automatically). For tuning and tests.
void fused_moe_ffn_tuned(torch::Tensor& hidden, torch::Tensor& gate_proj,
                         torch::Tensor& up_proj, torch::Tensor& down_proj,
                         torch::Tensor& fused_buf, torch::Tensor& output,
                         cudaStream_t stream, int dual_cfg, int dual_splits,
                         int down_cfg, int down_splits);

// Number of (gate/up, down) tile configs accepted by fused_moe_ffn_tuned.
std::pair<int, int> fused_moe_ffn_num_configs();
