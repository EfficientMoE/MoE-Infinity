// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <cstdint>
#include <vector>
#include <c10/cuda/CUDAStream.h>

#include "utils/cuda_utils.h"
#include "common/pytorch.h"
#include "kernel/ops.h"
#include "utils/logger.h"
#include "base/noncopyable.h"

#define BUFFER_PTR(buf_type, ptr_type) \
  (buffer_[static_cast<int>(BufferType::buf_type)])

#define TENSOR_INS(buf_type) tensors_[static_cast<int>(BufferType::buf_type)]

#define CUDA_ALLOCATE_BUFFER(type, size, dtype)                               \
  CUDA_CHECK(cudaMalloc(                                                      \
      reinterpret_cast<void**>(&buffer_[static_cast<int>(BufferType::type)]), \
      size * sizeof(dtype)));  // always allocate max 4 bytes per element, can
                               // use less

// The abstraction of MoE (Mixture of Experts) layer with fixed buffers.
class MoELayer : public base::noncopyable {
 public:
  enum class BufferType {

    // MoE buffers
    HiddenStates = 0,  // Buffer for hidden states
    // GatingWeights,       // Buffer for gate weights
    FinalHiddenStates,   // Buffer for final hidden states
    GatingOutput,        // Buffer for gating output
    TopKWeights,         // Buffer for top-k weights
    TopKIndices,         // Buffer for top-k indices
    TokenExpertIndices,  // Buffer for token expert indices

    // expert buffers
    ExpertInput,           // Buffer for input to experts
    ExpertUpProjOutput,    // Buffer for up projection output
    ExpertGateProjInput,   // Buffer for gate projection input
    ExpertDownProjOutput,  // Buffer for down projection output
    ExpertActMulOutput,    // Buffer for gated activation output

    // backward capability
    ExpertRouterMask,    // Buffer for router mask
    ExpertRouterWeight,  // Buffer for router weights

    NumBuffers  // Total number of buffer types
  };

  explicit MoELayer(int num_experts, int topk, int max_tokens,
                    int64_t hidden_dim, int64_t intermediate_dim,
                    bool use_bf16 = false, bool norm_topk_prob = false)
      : num_experts_(num_experts),
        topk_(topk),
        max_tokens_(max_tokens),
        hidden_dim_(hidden_dim),
        intermediate_dim_(intermediate_dim),
        buffer_(static_cast<int>(BufferType::NumBuffers)) {
    CUDA_ALLOCATE_BUFFER(HiddenStates, max_tokens * hidden_dim, float);
    // CUDA_ALLOCATE_BUFFER(GatingWeights, num_experts * hidden_dim);
    CUDA_ALLOCATE_BUFFER(FinalHiddenStates, max_tokens * hidden_dim, float);
    CUDA_ALLOCATE_BUFFER(GatingOutput, max_tokens * num_experts, float);
    CUDA_ALLOCATE_BUFFER(TopKWeights, max_tokens * topk, float);
    CUDA_ALLOCATE_BUFFER(TopKIndices, max_tokens * topk, int64_t);
    CUDA_ALLOCATE_BUFFER(TokenExpertIndices, max_tokens * topk, int32_t);
    CUDA_ALLOCATE_BUFFER(ExpertInput, max_tokens * hidden_dim, float);
    CUDA_ALLOCATE_BUFFER(ExpertUpProjOutput, max_tokens * intermediate_dim,
                         float);
    CUDA_ALLOCATE_BUFFER(ExpertGateProjInput, max_tokens * intermediate_dim,
                         float);
    CUDA_ALLOCATE_BUFFER(ExpertDownProjOutput, max_tokens * hidden_dim, float);
    CUDA_ALLOCATE_BUFFER(ExpertActMulOutput, max_tokens * hidden_dim, float);

    CUDA_ALLOCATE_BUFFER(ExpertRouterMask, max_tokens * num_experts, uint32_t);
    CUDA_ALLOCATE_BUFFER(ExpertRouterWeight, max_tokens * num_experts, float);

    device_id_ = GetDevice();
    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);

    DLOG_INFO(
        "MoELayer initialized with num_experts:", num_experts_, " topk:", topk,
        " max_tokens:", max_tokens, " hidden_dim:", hidden_dim,
        " intermediate_dim:", intermediate_dim, " on device:", device_id_);

    scalar_types_ = {
        torch::kFloat32,  // HiddenStates
        // torch::kFloat32,  // GatingWeights
        torch::kFloat32,  // FinalHiddenStates
        torch::kFloat32,  // GatingOutput
        torch::kFloat32,  // TopKWeights
        torch::kUInt32,   // TopKIndices
        torch::kInt32,    // TokenExpertIndices
        torch::kFloat32,  // ExpertInput
        torch::kFloat32,  // ExpertUpProjOutput
        torch::kFloat32,  // ExpertGateProjInput
        torch::kFloat32,  // ExpertDownProjOutput
        torch::kFloat32,  // ExpertActMulOutput
        torch::kBool,     // ExpertRouterMask
        torch::kFloat32   // ExpertRouterWeight
    };

    // Create tensors for each buffer for easy access
    for (int i = 0; i < static_cast<int>(BufferType::NumBuffers); ++i) {
      tensors_.emplace_back(
          torch::zeros({1}, torch::TensorOptions()
                                .dtype(scalar_types_[i])
                                .device(CUDA_DEVICE(device_id_))));
    }
  }

  torch::Tensor& _tensor(BufferType type) {
    return tensors_[static_cast<int>(type)];
  }

  void* _buffer(BufferType type) { return buffer_[static_cast<int>(type)]; }

  std::tuple<torch::Tensor, torch::Tensor> TopKSoftmax(
      torch::Tensor& gating_outputs) {
    // Perform the gating operation on stream_
    c10::cuda::CUDAStream torch_stream =
        c10::cuda::getStreamFromExternal(stream_, device_id_);
    c10::cuda::setCurrentCUDAStream(torch_stream);

    int64_t num_tokens = gating_outputs.size(0);
    assert(num_tokens <= max_tokens_);

    auto logits = gating_outputs.to(torch::kFloat32);
    // if (gating_outputs.dtype() != torch::kFloat32) {
    //   auto logits = torch::from_blob(
    //       BUFFER_PTR(GatingOutput, void), {num_tokens, num_experts_},
    //       DoNothingDeleter<void>{},
    //       torch::TensorOptions()
    //           .dtype(torch::kFloat32)
    //           .device(CUDA_DEVICE(device_id_)));
    //   logits = gating_outputs.to(torch::kFloat32);
    // }

    TENSOR_INS(TopKWeights)
        .set_data(torch::from_blob(BUFFER_PTR(TopKWeights, void),
                                   {num_tokens, topk_},
                                   DoNothingDeleter<void>{},
                                   torch::TensorOptions()
                                       .dtype(torch::kFloat32)
                                       .device(CUDA_DEVICE(device_id_))));

    TENSOR_INS(TopKIndices)
        .set_data(torch::from_blob(BUFFER_PTR(TopKIndices, void),
                                   {num_tokens, topk_},
                                   DoNothingDeleter<void>{},
                                   torch::TensorOptions()
                                       .dtype(torch::kInt64)
                                       .device(CUDA_DEVICE(device_id_))));
    TENSOR_INS(TokenExpertIndices)
        .set_data(torch::from_blob(
            BUFFER_PTR(TokenExpertIndices, void), {num_tokens, topk_},
            DoNothingDeleter<void>{},
            torch::TensorOptions()
                .dtype(torch::kInt32)
                .device(CUDA_DEVICE(device_id_))));  // Use Int32 for indices

    // Perform top-k softmax to get top-k gating weights and indices
    topk_softmax(TENSOR_INS(TopKWeights), TENSOR_INS(TopKIndices),
                 TENSOR_INS(TokenExpertIndices),
                 logits);  // [max_tokens, topk]
    // std::cout << "TopKIndices started on device: "
    //           << TENSOR_INS(TopKIndices) << std::endl;

    TENSOR_INS(TopKWeights) =
        TENSOR_INS(TopKWeights) /
        TENSOR_INS(TopKWeights).sum(1, true);  // Normalize top-k weights

    auto routing_weights = TENSOR_INS(TopKWeights).to(torch::kBFloat16);

    TENSOR_INS(ExpertRouterMask)
        .set_data(torch::from_blob(BUFFER_PTR(ExpertRouterMask, void),
                                   {num_tokens, num_experts_},
                                   DoNothingDeleter<void>{},
                                   torch::TensorOptions()
                                       .dtype(torch::kBool)
                                       .device(CUDA_DEVICE(device_id_))));
    // std::cout << "TokenExpertIndices started on device: "
    //           << TENSOR_INS(TokenExpertIndices) << std::endl;

    cudaMemsetAsync(BUFFER_PTR(ExpertRouterMask, void), 0,
                    num_tokens * num_experts_ * sizeof(uint32_t),
                    stream_);  // Initialize router mask
    cudaMemsetAsync(BUFFER_PTR(ExpertRouterWeight, void), 0,
                    num_tokens * num_experts_ * sizeof(float),
                    stream_);  // Initialize router weights

    TENSOR_INS(ExpertRouterMask)
        .scatter_(1, TENSOR_INS(TopKIndices),
                  true);  // Set router mask based on top-k indices

    TENSOR_INS(ExpertRouterWeight)
        .set_data(torch::from_blob(BUFFER_PTR(ExpertRouterWeight, void),
                                   {num_tokens, num_experts_},
                                   DoNothingDeleter<void>{},
                                   torch::TensorOptions()
                                       .dtype(torch::kBFloat16)
                                       .device(CUDA_DEVICE(device_id_))));

    cudaStreamSynchronize(stream_);  // Ensure all operations are complete

    TENSOR_INS(ExpertRouterWeight)
        .scatter_add_(1, TENSOR_INS(TopKIndices),
                      routing_weights);  // Set routing weights mask
    // std::cout << "TopKSoftmax completed on device: " <<
    // TENSOR_INS(ExpertRouterMask)
    //           << std::endl;
    return std::make_tuple(TENSOR_INS(ExpertRouterMask),
                           TENSOR_INS(ExpertRouterWeight));
  }

  // void ForwardGating() {
  //   // Forward pass for gating mechanism
  //   // This function will use the buffers to compute gating weights and
  //   outputs

  //   // create temperal wrappers as tensor
  //   auto hidden_states =
  //       torch::from_blob(BUFFER_PTR(HiddenStates, void),
  //                        {max_tokens_, hidden_dim_},
  //                        DoNothingDeleter<void>{}, torch::TensorOptions()
  //                            .dtype(torch::dtype<param_t>())
  //                            .device(CUDA_DEVICE(device_id_)));

  //   auto gating_weights =
  //       torch::from_blob(BUFFER_PTR(GatingWeights, void),
  //                        {num_experts_, hidden_dim_},
  //                        DoNothingDeleter<void>{}, torch::TensorOptions()
  //                            .dtype(torch::dtype<param_t>())
  //                            .device(CUDA_DEVICE(device_id_)));

  //   auto gating_output =
  //       torch::from_blob(BUFFER_PTR(GatingOutput, void),
  //                        {max_tokens_, num_experts_},
  //                        DoNothingDeleter<void>{}, torch::TensorOptions()
  //                            .dtype(torch::dtype<param_t>())
  //                            .device(CUDA_DEVICE(device_id_)));

  //   // Perform the gating operation on stream_
  //   c10::cuda::CUDAStream torch_stream =
  //       c10::cuda::getStreamFromExternal(stream_, device_id_);
  //   c10::cuda::setCurrentCUDAStream(torch_stream);
  //   torch::matmul_out(gating_output, hidden_states,
  //                     gating_weights.t());  // [max_tokens, num_experts]

  //   auto topk_weights =
  //       torch::from_blob(BUFFER_PTR(TopKWeights, void), {max_tokens_, topk_},
  //                        DoNothingDeleter<void>{},
  //                        torch::TensorOptions()
  //                            .dtype(torch::kFloat32)
  //                            .device(CUDA_DEVICE(device_id_)));

  //   auto topk_indices =
  //       torch::from_blob(BUFFER_PTR(TopKIndices, void), {max_tokens_, topk_},
  //                        DoNothingDeleter<void>{},
  //                        torch::TensorOptions()
  //                            .dtype(torch::kUInt32)
  //                            .device(CUDA_DEVICE(device_id_)));

  //   auto token_expert_indices =
  //       torch::from_blob(BUFFER_PTR(TokenExpertIndices, void),
  //                        {max_tokens_, topk_}, DoNothingDeleter<void>{},
  //                        torch::TensorOptions()
  //                            .dtype(torch::kUInt32)
  //                            .device(CUDA_DEVICE(device_id_)));

  //   // Perform top-k softmax to get top-k gating weights and indices
  //   topk_softmax(topk_weights, topk_indices, token_expert_indices,
  //                gating_output);  // [max_tokens, topk]

  //   auto router_mask =
  //       torch::from_blob(BUFFER_PTR(ExpertRouterMask, void),
  //                        {max_tokens_, num_experts_},
  //                        DoNothingDeleter<void>{}, torch::TensorOptions()
  //                            .dtype(torch::kBool)
  //                            .device(CUDA_DEVICE(device_id_)));

  //   router_mask.scatter_(1, token_expert_indices,
  //                        true);  // Set router mask based on top-k indices

  //   auto routing_weights_mask =
  //       torch::from_blob(BUFFER_PTR(ExpertRouterWeight, void),
  //                        {max_tokens_, num_experts_},
  //                        DoNothingDeleter<void>{}, torch::TensorOptions()
  //                            .dtype(torch::dtype<param_t>())
  //                            .device(CUDA_DEVICE(device_id_)));

  //   routing_weights_mask.scatter_add_(
  //       1, token_expert_indices,
  //       topk_weights);  // Set routing weights mask
  // }

  ~MoELayer() {
    // Clean up allocated buffers
    for (auto* buffer : buffer_) {
      if (buffer) {
        CUDA_CHECK(cudaFree(buffer));
      }
    }
    if (stream_) {
      CUDA_CHECK(cudaStreamDestroy(stream_));
    }
  }

 private:
  std::vector<void*> buffer_;           // Vector of buffers
  std::vector<torch::Tensor> tensors_;  // Vector of tensors for easy access
  std::vector<torch::ScalarType> scalar_types_;  // Vector of scalar types
  int64_t num_experts_ = 0;  // Number of experts in the MoE layer
  int64_t topk_ = 0;         // Number of top-k experts to select
  int64_t max_tokens_ = 0;   // Maximum number of tokens processed in a batch
  int64_t hidden_dim_ = 0;   // Dimension of hidden states
  int64_t intermediate_dim_ = 0;  // Dimension of intermediate states
  cudaStream_t stream_ = 0;       // CUDA stream for asynchronous operations
  int device_id_ = 0;             // Device ID for the MoE layer
};

static std::unique_ptr<MoELayer> moe_layer_ptr = nullptr;
static std::once_flag moe_layer_init_flag;
void InitMoELayer(int num_experts, int topk, int max_tokens, int64_t hidden_dim,
                  int64_t intermediate_dim);
std::tuple<torch::Tensor, torch::Tensor> TopKSoftmax(
    torch::Tensor& gating_outputs);
