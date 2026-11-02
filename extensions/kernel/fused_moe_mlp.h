#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

#include "kernel/moe_activation.h"

// Fused MoE MLP forward pass using CUTLASS.
//
// Computes: output = act(input @ gate_proj^T, input @ up_proj^T) @ down_proj^T
// with act = silu(g) * u unless an MoEActivation is given
// (kernel/moe_activation.h).
//
// The intermediate gate_out and act(gate, up) fusion are computed with custom
// CUTLASS epilogues to eliminate 2 kernel launches and avoid writing
// gate_act_out to global memory.
//
// Tensor conventions (HuggingFace weight shapes):
//   hidden    [M, H]   row-major   (M = batch, H = hidden_size)
//   gate_proj [I, H]   row-major   (I = intermediate_size)
//   up_proj   [I, H]   row-major
//   down_proj [H, I]   row-major
//   gate_buf  [M, I]   row-major   (intermediate: gate projection result)
//   fused_buf [M, I]   row-major   (intermediate: act(gate, up) result)
//   output    [M, H]   row-major
void fused_moe_ffn_into(torch::Tensor& hidden,     // [M, H]
                        torch::Tensor& gate_proj,  // [I, H]
                        torch::Tensor& up_proj,    // [I, H]
                        torch::Tensor& down_proj,  // [H, I]
                        torch::Tensor& gate_buf,   // [M, I]
                        torch::Tensor& fused_buf,  // [M, I]
                        torch::Tensor& output,     // [M, H]
                        cudaStream_t stream);

void fused_moe_ffn_into(torch::Tensor& hidden,     // [M, H]
                        torch::Tensor& gate_proj,  // [I, H]
                        torch::Tensor& up_proj,    // [I, H]
                        torch::Tensor& down_proj,  // [H, I]
                        torch::Tensor& gate_buf,   // [M, I]
                        torch::Tensor& fused_buf,  // [M, I]
                        torch::Tensor& output,     // [M, H]
                        const MoEActivation& activation, cudaStream_t stream);

// Throws (TORCH_CHECK) if the descriptor is invalid for its kind.
void check_moe_activation(const MoEActivation& activation);
