// expert_gemm.cu - Per-Expert CUDA Kernel
// Target: Ampere (Sm80) with optional Hopper (Sm90) support
// Expert MLP: output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
// Uses cuBLAS for GEMM operations

#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

// ============================================================================
// Configuration
// ============================================================================

using ElementAccumulator = float;

// ============================================================================
// Helper: SiLU Activation
// ============================================================================

__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

__global__ void silu_multiply_kernel(
    const __nv_bfloat16* __restrict__ gate_output,  // [num_tokens, inter_dim]
    const __nv_bfloat16* __restrict__ up_output,    // [num_tokens, inter_dim]
    __nv_bfloat16* __restrict__ output,             // [num_tokens, inter_dim]
    int num_tokens, int inter_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_tokens * inter_dim) {
    float gate = __bfloat162float(gate_output[idx]);
    float up = __bfloat162float(up_output[idx]);
    float result = silu(gate) * up;
    output[idx] = __float2bfloat16(result);
  }
}

// ============================================================================
// Main Function: Expert MLP Forward
// ============================================================================

// Forward: output = W2 @ (SiLU(W1 @ x) * (W3 @ x))
//
// Input shapes:
//   input: [num_tokens, D]
//   w1:    [inter_dim, D] - gate_proj (will apply SiLU)
//   w3:    [inter_dim, D] - up_proj
//   w2:    [D, inter_dim] - down_proj
//
// Output shapes:
//   output: [num_tokens, D]

at::Tensor expert_fused_mlp(torch::Tensor input,  // [num_tokens, D]
                            torch::Tensor w1,     // [inter_dim, D] - gate_proj
                            torch::Tensor w3,     // [inter_dim, D] - up_proj
                            torch::Tensor w2      // [D, inter_dim] - down_proj
) {
  TORCH_CHECK(input.is_cuda(), "Input must be CUDA");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "Expected BF16 input");
  TORCH_CHECK(w1.scalar_type() == at::kBFloat16, "Expected BF16 w1");
  TORCH_CHECK(w3.scalar_type() == at::kBFloat16, "Expected BF16 w3");
  TORCH_CHECK(w2.scalar_type() == at::kBFloat16, "Expected BF16 w2");

  const int num_tokens = input.size(0);
  const int D = input.size(1);
  const int inter_dim = w1.size(0);

  // Allocate intermediate buffers
  auto gate_output = torch::empty({num_tokens, inter_dim},
                                  input.options());  // gate_proj result
  auto up_output =
      torch::empty({num_tokens, inter_dim}, input.options());  // up_proj result
  auto gate_up_fused = torch::empty({num_tokens, inter_dim},
                                    input.options());  // SiLU(gate) * up
  auto output = torch::empty({num_tokens, D}, input.options());

  // =========================================================================
  // GEMM 1: gate_proj (input @ w1.T) -> gate_output
  // Using cuBLAS (simpler integration, similar performance)
  // =========================================================================

  cublasHandle_t handle;
  cublasCreate(&handle);

  float alpha = 1.0f, beta = 0.0f;

  // gate_output = input @ w1.T
  // input: [num_tokens, D], w1: [inter_dim, D] -> gate_output: [num_tokens,
  // inter_dim] CUBLAS_OP_T means we use w1.T
  cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, inter_dim, num_tokens, D,
               &alpha, w1.data_ptr<at::BFloat16>(), CUDA_R_16BF, D,
               input.data_ptr<at::BFloat16>(), CUDA_R_16BF, D, &beta,
               gate_output.data_ptr<at::BFloat16>(), CUDA_R_16BF, inter_dim,
               CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  // =========================================================================
  // GEMM 2: up_proj (input @ w3.T) -> up_output
  // =========================================================================

  cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, inter_dim, num_tokens, D,
               &alpha, w3.data_ptr<at::BFloat16>(), CUDA_R_16BF, D,
               input.data_ptr<at::BFloat16>(), CUDA_R_16BF, D, &beta,
               up_output.data_ptr<at::BFloat16>(), CUDA_R_16BF, inter_dim,
               CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  cublasDestroy(handle);

  // =========================================================================
  // Epilogue: SiLU(gate_output) * up_output -> gate_up_fused
  // =========================================================================

  const int threads = 256;
  const int blocks = (num_tokens * inter_dim + threads - 1) / threads;
  silu_multiply_kernel<<<blocks, threads>>>(
      reinterpret_cast<__nv_bfloat16*>(gate_output.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(up_output.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(gate_up_fused.data_ptr<at::BFloat16>()),
      num_tokens, inter_dim);

  // =========================================================================
  // GEMM 3: down_proj (gate_up_fused @ w2.T) -> output
  // =========================================================================

  cublasCreate(&handle);

  // output = gate_up_fused @ w2.T
  // gate_up_fused: [num_tokens, inter_dim], w2: [D, inter_dim] -> output:
  // [num_tokens, D]
  cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, D, num_tokens, inter_dim,
               &alpha, w2.data_ptr<at::BFloat16>(), CUDA_R_16BF, inter_dim,
               gate_up_fused.data_ptr<at::BFloat16>(), CUDA_R_16BF, inter_dim,
               &beta, output.data_ptr<at::BFloat16>(), CUDA_R_16BF, D,
               CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  cublasDestroy(handle);
  cudaDeviceSynchronize();

  return output;
}

// =========================================================================
// Batch version: Process multiple experts efficiently
// =========================================================================

at::Tensor expert_fused_mlp_batched(
    torch::Tensor input,      // [num_tokens, D]
    torch::Tensor w1_list,    // [num_experts, inter_dim, D]
    torch::Tensor w3_list,    // [num_experts, inter_dim, D]
    torch::Tensor w2_list,    // [num_experts, D, inter_dim]
    torch::Tensor expert_ids  // [num_tokens] - which expert each token uses
) {
  TORCH_CHECK(input.is_cuda(), "Input must be CUDA");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "Expected BF16");

  const int num_tokens = input.size(0);
  const int D = input.size(1);
  const int inter_dim = w1_list.size(1);
  const int num_experts = w1_list.size(0);

  auto output = torch::empty({num_tokens, D}, input.options());

  // Process each expert
  for (int expert_id = 0; expert_id < num_experts; expert_id++) {
    // Find tokens that use this expert
    auto mask = (expert_ids == expert_id);
    auto token_indices = mask.nonzero().squeeze();

    if (token_indices.numel() == 0) continue;

    // Extract input for this expert
    auto expert_input = input.index_select(0, token_indices);

    // Get weights
    auto w1 = w1_list[expert_id];
    auto w3 = w3_list[expert_id];
    auto w2 = w2_list[expert_id];

    // Compute expert output
    auto expert_output = expert_fused_mlp(expert_input, w1, w3, w2);

    // Store result
    output.index_copy_(0, token_indices, expert_output);
  }

  return output;
}

// PyTorch Binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("expert_fused_mlp", &expert_fused_mlp,
        "Fused Expert MLP (gate+up fused, then down) - cuBLAS version");
  m.def("expert_fused_mlp_batched", &expert_fused_mlp_batched,
        "Fused Expert MLP for batched experts");
}
