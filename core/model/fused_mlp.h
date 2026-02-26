#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

// Legacy entry point kept for external callers.  The implementation in
// fused_mlp.cu now delegates to fused_moe_ffn_into() (CUTLASS 3-GEMM path).
torch::Tensor launch_fused_moe_ffn(torch::Tensor hidden,  // [M, K]
                                   torch::Tensor w1,      // [N, K] gate proj
                                   torch::Tensor w2,      // [N, K] up   proj
                                   torch::Tensor w3,      // [K, N] down proj
                                   cudaStream_t stream);
