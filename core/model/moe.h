// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <cstdint>
#include <vector>

#include "utils/cuda_utils.h"
#include "common/pytorch.h"
#include "kernel/ops.h"

#define BUFFER_PTR(buf_type, ptr_type) \
  (buffer_[static_cast<int>(BufferType::buf_type)])

#define CUDA_ALLOCATE_BUFFER(type, size)                                      \
  CUDA_CHECK(cudaMalloc(                                                      \
      reinterpret_cast<void**>(&buffer_[static_cast<int>(BufferType::type)]), \
      size * sizeof(param_t)));

// The abstraction of MoE (Mixture of Experts) layer with fixed buffers.
template <typename param_t>
class MoELayer {
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
                    int64_t hidden_dim, int64_t intermediate_dim)
      : num_experts_(num_experts),
        topk_(topk),
        max_tokens_(max_tokens),
        hidden_dim_(hidden_dim),
        intermediate_dim_(intermediate_dim),
        buffer_(static_cast<int>(BufferType::NumBuffers)) {
    CUDA_ALLOCATE_BUFFER(HiddenStates, max_tokens * hidden_dim);
    // CUDA_ALLOCATE_BUFFER(GatingWeights, num_experts * hidden_dim);
    CUDA_ALLOCATE_BUFFER(FinalHiddenStates, max_tokens * hidden_dim);
    CUDA_ALLOCATE_BUFFER(GatingOutput, max_tokens * num_experts);
    CUDA_ALLOCATE_BUFFER(TopKWeights, max_tokens * topk);
    CUDA_ALLOCATE_BUFFER(TopKIndices, max_tokens * topk);
    CUDA_ALLOCATE_BUFFER(TokenExpertIndices, max_tokens * topk);
    CUDA_ALLOCATE_BUFFER(ExpertInput, max_tokens * hidden_dim);
    CUDA_ALLOCATE_BUFFER(ExpertUpProjOutput, max_tokens * intermediate_dim);
    CUDA_ALLOCATE_BUFFER(ExpertGateProjInput, max_tokens * intermediate_dim);
    CUDA_ALLOCATE_BUFFER(ExpertDownProjOutput, max_tokens * hidden_dim);
    CUDA_ALLOCATE_BUFFER(ExpertActMulOutput, max_tokens * hidden_dim);

    CUDA_ALLOCATE_BUFFER(ExpertRouterMask, max_tokens * num_experts);
    CUDA_ALLOCATE_BUFFER(ExpertRouterWeight, max_tokens * num_experts);

    device_id_ = c10::cuda::current_device();
    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
  }

  void ForwardGating() {
    // Forward pass for gating mechanism
    // This function will use the buffers to compute gating weights and outputs

    // create temperal wrappers as tensor
    auto hidden_states =
        torch::from_blob(BUFFER_PTR(HiddenStates, void),
                         {max_tokens_, hidden_dim_}, DoNothingDeleter<void>{},
                         torch::TensorOptions()
                             .dtype(torch::dtype<param_t>())
                             .device(CUDA_DEVICE(device_id_)));

    auto gating_weights =
        torch::from_blob(BUFFER_PTR(GatingWeights, void),
                         {num_experts_, hidden_dim_}, DoNothingDeleter<void>{},
                         torch::TensorOptions()
                             .dtype(torch::dtype<param_t>())
                             .device(CUDA_DEVICE(device_id_)));

    auto gating_output =
        torch::from_blob(BUFFER_PTR(GatingOutput, void),
                         {max_tokens_, num_experts_}, DoNothingDeleter<void>{},
                         torch::TensorOptions()
                             .dtype(torch::dtype<param_t>())
                             .device(CUDA_DEVICE(device_id_)));

    // Perform the gating operation on stream_
    c10::cuda::CUDAStream torch_stream =
        c10::cuda::getStreamFromExternal(stream_, device_id_);
    c10::cuda::setCurrentCUDAStream(torch_stream);
    torch::matmul_out(gating_output, hidden_states,
                      gating_weights.t());  // [max_tokens, num_experts]

    auto topk_weights =
        torch::from_blob(BUFFER_PTR(TopKWeights, void), {max_tokens_, topk_},
                         DoNothingDeleter<void>{},
                         torch::TensorOptions()
                             .dtype(torch::kFloat32)
                             .device(CUDA_DEVICE(device_id_)));

    auto topk_indices =
        torch::from_blob(BUFFER_PTR(TopKIndices, void), {max_tokens_, topk_},
                         DoNothingDeleter<void>{},
                         torch::TensorOptions()
                             .dtype(torch::kUInt32)
                             .device(CUDA_DEVICE(device_id_)));

    auto token_expert_indices =
        torch::from_blob(BUFFER_PTR(TokenExpertIndices, void),
                         {max_tokens_, topk_}, DoNothingDeleter<void>{},
                         torch::TensorOptions()
                             .dtype(torch::kUInt32)
                             .device(CUDA_DEVICE(device_id_)));

    // Perform top-k softmax to get top-k gating weights and indices
    topk_softmax(topk_weights, topk_indices, token_expert_indices,
                 gating_output);  // [max_tokens, topk]

    auto router_mask =
        torch::from_blob(BUFFER_PTR(ExpertRouterMask, void),
                         {max_tokens_, num_experts_}, DoNothingDeleter<void>{},
                         torch::TensorOptions()
                             .dtype(torch::kBool)
                             .device(CUDA_DEVICE(device_id_)));

    router_mask.scatter_(1, token_expert_indices,
                         true);  // Set router mask based on top-k indices

    auto routing_weights_mask =
        torch::from_blob(BUFFER_PTR(ExpertRouterWeight, void),
                         {max_tokens_, num_experts_}, DoNothingDeleter<void>{},
                         torch::TensorOptions()
                             .dtype(torch::dtype<param_t>())
                             .device(CUDA_DEVICE(device_id_)));

    routing_weights_mask.scatter_add_(
        1, token_expert_indices,
        topk_weights);  // Set routing weights mask
  }

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
  std::vector<param_t*> buffer_;  // Vector of buffers
  int num_experts_ = 0;           // Number of experts in the MoE layer
  int topk_ = 0;                  // Number of top-k experts to select
  int max_tokens_ = 0;      // Maximum number of tokens processed in a batch
  int64_t hidden_dim_ = 0;  // Dimension of hidden states
  int64_t intermediate_dim_ = 0;  // Dimension of intermediate states
  cudaStream_t stream_ = 0;       // CUDA stream for asynchronous operations
  int device_id_ = 0;             // Device ID for the MoE layer
};
