#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

// Fused MoE MLP forward pass using CUTLASS.
//
// Computes: output = (silu(input @ gate_proj^T) * (input @ up_proj^T)) @
// down_proj^T
//
// The intermediate gate_out and silu*up fusion are computed with custom CUTLASS
// epilogues to eliminate 2 kernel launches and avoid writing gate_act_out to
// global memory.
//
// Tensor conventions (HuggingFace weight shapes):
//   hidden    [M, H]   row-major   (M = batch, H = hidden_size)
//   gate_proj [I, H]   row-major   (I = intermediate_size)
//   up_proj   [I, H]   row-major
//   down_proj [H, I]   row-major
//   gate_buf  [M, I]   row-major   (intermediate: gate projection result)
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
