#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

#include <array>
#include <utility>
#include <vector>

#include "kernel/moe_activation.h"

// Fused MoE MLP forward pass using CUTLASS.
//
// Computes: output = act(input @ gate_proj^T, input @ up_proj^T) @ down_proj^T
// with act = silu(g) * u unless an MoEActivation is given
// (kernel/moe_activation.h).
//
// RTX PRO 6000, SiLU only: two launches. A dual GEMM computes both projections
// from one read of the input and writes only silu(gate) * up to fused_buf; a
// second GEMM applies down_proj. Tile shape and split-K are chosen per
// (M, N, K) from a table tuned on that card. Clamped SiLU and SwiGLU-OAI, and
// every other GPU, use the previous 3-GEMM path (fused_moe_ffn_into_generic),
// which fuses act(gate, up) into the up-projection epilogue and writes
// gate_buf.
//
// Tensor conventions (HuggingFace weight shapes):
//   hidden    [M, H]   row-major   (M = batch, H = hidden_size)
//   gate_proj [I, H]   row-major   (I = intermediate_size)
//   up_proj   [I, H]   row-major
//   down_proj [H, I]   row-major
//   gate_buf  [M, I]   row-major   (written only by the generic path)
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

// The RTX PRO 6000 path (fused_moe_ffn_pro6000) runs only where
// fused_moe_ffn_pro6000_enabled(device) holds (gate RTX_PRO_6000, evaluated
// once per device) and the activation is SiLU. Every other case runs the
// previous 3-GEMM implementation.
bool fused_moe_ffn_pro6000_enabled(int device);

// Previous implementation: 3 CUTLASS GEMMs with fixed 64x64 tiles, writes
// gate_buf. Generic; the fallback of fused_moe_ffn_into. The overload with
// MoEActivation selects the up-projection epilogue (SiLU, clamped SiLU, or
// SwiGLU-OAI).
void fused_moe_ffn_into_generic(torch::Tensor& hidden, torch::Tensor& gate_proj,
                                torch::Tensor& up_proj,
                                torch::Tensor& down_proj,
                                torch::Tensor& gate_buf,
                                torch::Tensor& fused_buf, torch::Tensor& output,
                                cudaStream_t stream);

void fused_moe_ffn_into_generic(torch::Tensor& hidden, torch::Tensor& gate_proj,
                                torch::Tensor& up_proj,
                                torch::Tensor& down_proj,
                                torch::Tensor& gate_buf,
                                torch::Tensor& fused_buf, torch::Tensor& output,
                                const MoEActivation& activation,
                                cudaStream_t stream);

// RTX PRO 6000 path of fused_moe_ffn_into with an explicit tile config /
// split-K factor per GEMM (cfg < 0 selects automatically). For tuning and
// tests. SiLU only.
void fused_moe_ffn_pro6000(torch::Tensor& hidden, torch::Tensor& gate_proj,
                           torch::Tensor& up_proj, torch::Tensor& down_proj,
                           torch::Tensor& fused_buf, torch::Tensor& output,
                           cudaStream_t stream, int dual_cfg, int dual_splits,
                           int down_cfg, int down_splits);

// Number of (gate/up, down) tile configs accepted by fused_moe_ffn_pro6000.
std::pair<int, int> fused_moe_ffn_num_configs();

// Threadblock (M, N, K) of every (gate/up, down) config, in config order.
std::pair<std::vector<std::array<int, 3>>, std::vector<std::array<int, 3>>>
fused_moe_ffn_tile_shapes();
