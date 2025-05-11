// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include <torch/torch.h>
#include <ATen/cuda/CUDAGraph.h>
#include "model/model_topology.h"

#ifndef EXPERT_TYPE
  #define EXPERT_TYPE 0
#endif

#define SWITCH_TRANSFORMERS_DENSE_ACT_DENSE 0
#define SWITCH_TRANSFORMERS_DENSE_GATED_ACT_DENSE 1
#define NLLB_MOE_DENSE_ACT_DENSE 2
#define FSGPT_MOE_DENSE_ACT_DENSE 3
#define MIXTRAL_MOE_DENSE_ACT_DENSE 4
#define DEEPSEEK_MOE_DENSE_ACT_DENSE 5

// forward declarations
torch::Tensor launch_fused_moe_ffn(torch::Tensor hidden,  // [M, K]
                                   torch::Tensor w1,      // [N, K]
                                   torch::Tensor w2,      // [N, K]
                                   torch::Tensor w3,      // [K, N]
                                   cudaStream_t stream);  // CUDA stream

struct ModuleUtils {
  virtual void SetTensorsFromBlob(void* ptr,
                                  const std::vector<std::uint32_t>& tensor_ids,
                                  const torch::Device& device) = 0;
  virtual void SetModuleFromBlob(torch::jit::script::Module* ptr) = 0;
};

#define DECLARE_MODULE(name, ...)                                         \
  struct name : public torch::nn::Module, public ModuleUtils {            \
    name(int dtype);                                                      \
    torch::Tensor forward(torch::Tensor hidden_states,                    \
                          cudaStream_t stream = nullptr);                 \
    torch::Tensor __VA_ARGS__;                                            \
    void SetTensorsFromBlob(void* ptr,                                    \
                            const std::vector<std::uint32_t>& tensor_ids, \
                            const torch::Device& device) override;        \
    void SetModuleFromBlob(torch::jit::script::Module* ptr) override;     \
  };

DECLARE_MODULE(SwitchTransformersDenseActDense, wi, wo)
DECLARE_MODULE(SwitchTransformersDenseGatedActDense, wi_0, wi_1, wo)
DECLARE_MODULE(NllbMoeDenseActDense, fc1, fc2, fc1_bias, fc2_bias)
DECLARE_MODULE(FSGPTMoEDenseActDense, fc1, fc2, fc1_bias, fc2_bias)
DECLARE_MODULE(MixtralMoEDenseActDense, w1, w2, w3)
DECLARE_MODULE(DeepSeekMoEDenseActDense, gate_proj, up_proj, down_proj)

struct MoEMLP : public torch::nn::Module {
  explicit MoEMLP(int dtype, int expert_type);
  torch::Tensor forward(torch::Tensor hidden_states, cudaStream_t stream);

  void SetTensorsFromIds(const std::vector<std::uint32_t>& tensor_ids);

 private:
  void ForwardHelper();

 private:
  std::vector<torch::Tensor> buffer_;
  std::vector<torch::Tensor> param_;
  // torch::Tensor input_;
  // torch::Tensor output_;
  // torch::Tensor gate_proj_;
  // torch::Tensor up_proj_;
  // torch::Tensor down_proj_;

  // torch::Tensor fc1_bias_;
  // torch::Tensor fc2_bias_;
  // torch::Tensor fc3_bias_;

  at::cuda::CUDAGraph graph_;
  int warmup_count_ = 5;
  bool graph_mode_ = false;
  // bool data_initialized_ = false;
  bool param_init_ = false;
  bool param_set_ = false;

  int dtype_;
  int expert_type_;
};

// struct SwitchTransformersDenseActDense : public torch::nn::Module,
//                                          public ModuleUtils {
//   SwitchTransformersDenseActDense(int dtype);
//   torch::Tensor forward(torch::Tensor hidden_states);
//   torch::Tensor wi, wo;

//   void SetTensorsFromBlob(void* ptr,
//                           const std::vector<std::uint32_t>& tensor_ids,
//                           const torch::Device& device) override;
//   void SetModuleFromBlob(torch::jit::script::Module* ptr) override;
// };

// struct SwitchTransformersDenseGatedActDense : public torch::nn::Module,
//                                               public ModuleUtils {
//   SwitchTransformersDenseGatedActDense(int dtype);
//   torch::Tensor forward(torch::Tensor hidden_states);
//   torch::Tensor wi_0, wi_1, wo;

//   void SetTensorsFromBlob(void* ptr,
//                           const std::vector<std::uint32_t>& tensor_ids,
//                           const torch::Device& device) override;
//   void SetModuleFromBlob(torch::jit::script::Module* ptr) override;
// };

// struct NllbMoeDenseActDense : public torch::nn::Module, public ModuleUtils {
//   NllbMoeDenseActDense(int dtype);
//   torch::Tensor forward(torch::Tensor hidden_states);
//   torch::Tensor fc1, fc2;
//   torch::Tensor fc1_bias, fc2_bias;

//   void SetTensorsFromBlob(void* ptr,
//                           const std::vector<std::uint32_t>& tensor_ids,
//                           const torch::Device& device) override;
//   void SetModuleFromBlob(torch::jit::script::Module* ptr) override;
// };

// struct FSGPTMoEDenseActDense : public torch::nn::Module, public ModuleUtils {
//   FSGPTMoEDenseActDense(int dtype);
//   torch::Tensor forward(torch::Tensor hidden_states);
//   torch::Tensor fc1, fc2;
//   torch::Tensor fc1_bias, fc2_bias;

//   void SetTensorsFromBlob(void* ptr,
//                           const std::vector<std::uint32_t>& tensor_ids,
//                           const torch::Device& device) override;
//   void SetModuleFromBlob(torch::jit::script::Module* ptr) override;
// };

// struct MixtralMoEDenseActDense : public torch::nn::Module, public ModuleUtils
// {
//   MixtralMoEDenseActDense(int dtype);
//   torch::Tensor forward(torch::Tensor hidden_states);
//   torch::Tensor w1, w2, w3;

//   void SetTensorsFromBlob(void* ptr,
//                           const std::vector<std::uint32_t>& tensor_ids,
//                           const torch::Device& device) override;
//   void SetModuleFromBlob(torch::jit::script::Module* ptr) override;
// };

// struct DeepSeekMoEDenseActDense : public torch::nn::Module, public
// ModuleUtils {
//   DeepSeekMoEDenseActDense(int dtype);
//   torch::Tensor forward(torch::Tensor hidden_states, cudaStream_t stream);
//   torch::Tensor gate_proj, up_proj, down_proj;

//   void SetTensorsFromBlob(void* ptr,
//                           const std::vector<std::uint32_t>& tensor_ids,
//                           const torch::Device& device) override;
//   void SetModuleFromBlob(torch::jit::script::Module* ptr) override;
// };

struct ExpertNode {
  NodePtr node;
  torch::nn::Module* module;
  void SetTensorsFromBlob(const torch::Device& device);
  int layer_idx;
  int expert_idx;
  int expert_type;
  torch::jit::script::Module* jit_module;
};

typedef std::shared_ptr<ExpertNode> ExpertNodePtr;
