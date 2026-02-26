// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// Legacy launch_fused_moe_ffn wrapper.  Delegates to the CUTLASS-based
// fused_moe_ffn_into() which eliminates 2 kernel launches by fusing
// silu(gate) * up into the epilogue of the up-projection GEMM.

#include "model/fused_mlp.h"
#include "kernel/fused_moe_mlp.h"

torch::Tensor launch_fused_moe_ffn(torch::Tensor hidden,  // [M, K]
                                   torch::Tensor w1,      // [N, K] gate proj
                                   torch::Tensor w2,      // [N, K] up   proj
                                   torch::Tensor w3,      // [K, N] down proj
                                   cudaStream_t stream) {
  TORCH_CHECK(hidden.scalar_type() == at::kBFloat16,
              "launch_fused_moe_ffn: BF16 only");

  const int M = static_cast<int>(hidden.size(0));
  const int K = static_cast<int>(hidden.size(1));  // hidden dim
  const int N = static_cast<int>(w1.size(0));      // intermediate dim

  auto opts = hidden.options();
  auto gate_buf = torch::empty({M, N}, opts);
  auto fused_buf = torch::empty({M, N}, opts);
  auto output = torch::empty({M, K}, opts);

  fused_moe_ffn_into(hidden, w1, w2, w3, gate_buf, fused_buf, output, stream);
  return output;
}
