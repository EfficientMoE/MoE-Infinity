#include "moe.h"

void InitMoELayer(int num_experts, int topk, int max_tokens, int64_t hidden_dim,
                  int64_t intermediate_dim) {
  std::call_once(moe_layer_init_flag, [&]() {
    moe_layer_ptr = std::make_unique<MoELayer>(num_experts, topk, max_tokens,
                                               hidden_dim, intermediate_dim);
  });
}

std::tuple<torch::Tensor, torch::Tensor> TopKSoftmax(
    torch::Tensor& gating_outputs) {
  if (!moe_layer_ptr) {
    throw std::runtime_error(
        "MoELayer is not initialized. Call InitMoELayer first.");
  }
  return moe_layer_ptr->TopKSoftmax(gating_outputs);
}
