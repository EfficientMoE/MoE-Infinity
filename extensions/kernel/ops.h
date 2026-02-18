#pragma once

#include <torch/all.h>

// Activation and gating kernel functions
void silu_and_mul(torch::Tensor& out,     // [..., d]
                  torch::Tensor& input);  // [..., 2 * d]

void mul_and_silu(torch::Tensor& out,     // [..., d]
                  torch::Tensor& input);  // [..., 2 * d]

void gelu_and_mul(torch::Tensor& out,     // [..., d]
                  torch::Tensor& input);  // [..., 2 * d]

void gelu_tanh_and_mul(torch::Tensor& out,     // [..., d]
                       torch::Tensor& input);  // [..., 2 * d]

void fatrelu_and_mul(torch::Tensor& out,    // [..., d],
                     torch::Tensor& input,  // [..., 2 * d]
                     double threshold);

// Element-wise activation kernel functions
void gelu_new(torch::Tensor& out,     // [..., d]
              torch::Tensor& input);  // [..., d]

void gelu_fast(torch::Tensor& out,     // [..., d]
               torch::Tensor& input);  // [..., d]

void gelu_quick(torch::Tensor& out,     // [..., d]
                torch::Tensor& input);  // [..., d]

// TopK softmax kernel functions
void topk_softmax(torch::Tensor& topk_weights,          // [num_tokens, topk]
                  torch::Tensor& topk_indices,          // [num_tokens, topk]
                  torch::Tensor& token_expert_indices,  // [num_tokens, topk]
                  torch::Tensor& gating_output);  // [num_tokens, num_experts]
