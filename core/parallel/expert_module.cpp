// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "expert_module.h"
// #include "memory/caching_allocator.h"
#include "utils/cuda_utils.h"
#include "utils/logger.h"
#include "kernel/fused_moe_mlp.h"

static const int64_t kMaxTokens = 256;

void ExpertNode::SetTensorsFromBlob(const torch::Device& device) {
  auto expert_type = static_cast<ExpertType>(this->expert_type);
  switch (expert_type) {
    case ExpertType::NllbMoeDenseActDense:
      reinterpret_cast<NllbMoeDenseActDense*>(module)->SetTensorsFromBlob(
          node->device_memory_ptr, node->tensor_ids, device);
      break;
    case ExpertType::FSGPTMoeDenseActDense:
      reinterpret_cast<FSGPTMoEDenseActDense*>(module)->SetTensorsFromBlob(
          node->device_memory_ptr, node->tensor_ids, device);
      break;
    case ExpertType::MixtralMoeDenseActDense:
      reinterpret_cast<MixtralMoEDenseActDense*>(module)->SetTensorsFromBlob(
          node->device_memory_ptr, node->tensor_ids, device);
      break;
    case ExpertType::DeepSeekMoeDenseActDense:
      reinterpret_cast<DeepSeekMoEDenseActDense*>(module)->SetTensorsFromBlob(
          node->device_memory_ptr, node->tensor_ids, device);
      break;
    default:
      assert(false);
  }
}

MoEMLP::MoEMLP(int dtype, int expert_type) {
  auto tensor_dtype = dtype_to_torch(dtype);
  auto options = torch::TensorOptions().dtype(tensor_dtype).device(torch::kCPU);

  expert_type_ = expert_type;
  dtype_ = dtype;

  for (int i = 0; i < 8; i++) {
    buffer_.push_back(torch::zeros({1}, options));
  }
  for (int i = 0; i < 4; i++) {
    param_.push_back(torch::zeros({1}, options));
  }
}

void MoEMLP::SetTensorsFromIds(const std::vector<std::uint32_t>& tensor_ids) {
  std::vector<std::tuple<void*, int64_t>> tensor_ptrs;
  std::vector<std::vector<int64_t>> tensor_shapes;
  std::vector<std::vector<int64_t>> data_shapes;
  int device = at::cuda::current_device();
  auto options = torch::TensorOptions()
                     .dtype(dtype_to_torch(dtype_))
                     .device(CUDA_DEVICE(device));
  for (auto& id : tensor_ids) {
    auto tensor = kTensorIndex->find(id)->second.tensor;
    auto tensor_shape = tensor.sizes().vec();
    auto tensor_ptr = tensor.data_ptr();
    auto tensor_size = torch_shape_size(tensor_shape, dtype_);
    tensor_ptrs.push_back(std::make_tuple(tensor_ptr, tensor_size));
    tensor_shapes.push_back(tensor_shape);
  }
  if (!param_init_) {
    // auto allocator = CudaDeviceCachingAllocator::instance(device);
    auto allocator = c10::DeviceCachingAllocator::get(device);
    for (size_t i = 0; i < tensor_ptrs.size(); i++) {
      auto [ptr, tensor_size] = tensor_ptrs[i];
      auto tensor_shape = tensor_shapes[i];
      void* param_ptr = allocator->allocate(tensor_size);
      param_[i].set_data(torch::from_blob(param_ptr, tensor_shape,
                                          DoNothingDeleter<void>{}, options));
      DLOG_DEBUG("MoEMLP::SetTensorsFromBlob: tensor_ids", tensor_ids[i],
                 "tensor_shape", tensor_shape, "tensor_size", tensor_size,
                 "param_", param_[i].sizes().vec(), "device",
                 param_[i].device().str());
    }

    // MLP tensor shape is transposed
    int64_t hdim = tensor_shapes[0][1];
    int64_t idim = tensor_shapes[0][0];
    data_shapes.push_back({kMaxTokens, hdim});
    data_shapes.push_back({kMaxTokens, hdim});

    for (size_t i = 0; i < tensor_shapes.size(); i++) {
      data_shapes.push_back({kMaxTokens, idim});
    }

    // auto allocator = CudaDeviceCachingAllocator::instance(device);
    // auto allocator = c10::DeviceCachingAllocator::get(device);
    // auto data_size = torch_shape_size({1024, hdim}, dtype_);
    for (size_t i = 0; i < data_shapes.size(); i++) {
      auto data_shape = data_shapes[i];
      auto data_size = torch_shape_size(data_shape, dtype_);
      void* buffer_ptr = allocator->allocate(data_size);
      buffer_[i].set_data(torch::from_blob(buffer_ptr, data_shape,
                                           DoNothingDeleter<void>{}, options));
      DLOG_TRACE("MoEMLP::SetTensorsFromBlob: buffer_ tensor", i, "data_shape",
                 data_shape, "data_size", data_size, "device",
                 buffer_[i].device().str());
    }
    param_init_ = true;
  }

  assert(param_init_ == true);
  assert(param_set_ == false);

  cudaStream_t current_stream;
  cudaStreamCreate(&current_stream);
  for (size_t i = 0; i < tensor_ptrs.size(); i++) {
    auto [ptr, tensor_size] = tensor_ptrs[i];
    CUDA_CHECK(cudaMemcpyAsync(param_[i].data_ptr(), ptr, tensor_size,
                               cudaMemcpyDeviceToDevice, current_stream));
  }
  cudaStreamSynchronize(current_stream);
  cudaStreamDestroy(current_stream);
  param_set_ = true;
  // DLOG_FATAL(
  //     "MoEMLP::SetTensorsFromBlob: tensor_ids.size() should be 2,3,4, but got
  //     {}", tensor_ids.size());
}

torch::Tensor MoEMLP::forward(torch::Tensor hidden_states,
                              cudaStream_t stream) {
  DLOG_FATAL_IF(param_set_ == false, "param_set_ should be true");
  DLOG_FATAL_IF(param_init_ == false, "param_init_ should be true");

  auto& input_ = buffer_[0];
  auto& output_ = buffer_[1];

  int64_t batch_size = hidden_states.size(0);
  int64_t hdim = hidden_states.size(1);

  DLOG_FATAL_IF(batch_size > kMaxTokens || batch_size <= 0,
                "batch_size should be (0,", kMaxTokens, "] , but got",
                batch_size);

  // Use async copy with the provided execution stream
  cudaMemcpyAsync(input_.data_ptr(), hidden_states.data_ptr(),
                  hidden_states.numel() * hidden_states.element_size(),
                  cudaMemcpyDeviceToDevice, stream);

  // Dynamically reshape buffers to match actual batch_size, avoiding
  // computation on padded rows. The underlying memory is unchanged.
  for (auto& buffer : buffer_) {
    auto shape_vec = buffer.sizes().vec();
    if (shape_vec.size() != 2) continue;
    int64_t row = buffer.size(0);
    int64_t col = buffer.size(1);
    auto dtype = buffer.dtype();

    if (row == kMaxTokens) {
      buffer.set_data(torch::from_blob(
          buffer.data_ptr(), {batch_size, col}, DoNothingDeleter<void>{},
          torch::TensorOptions().dtype(dtype).device(
              CUDA_DEVICE(at::cuda::current_device()))));
    }
  }
  cudaStreamSynchronize(stream);

  ForwardHelper(stream);
  param_set_ = false;
  cudaStreamSynchronize(stream);

  auto output = output_.clone();

  // Restore buffers to kMaxTokens shape for next invocation
  for (auto& buffer : buffer_) {
    auto shape_vec = buffer.sizes().vec();
    if (shape_vec.size() != 2) continue;
    int64_t row = buffer.size(0);
    int64_t col = buffer.size(1);
    auto dtype = buffer.dtype();

    if (row == batch_size && batch_size != kMaxTokens) {
      buffer.set_data(torch::from_blob(
          buffer.data_ptr(), {kMaxTokens, col}, DoNothingDeleter<void>{},
          torch::TensorOptions().dtype(dtype).device(
              CUDA_DEVICE(at::cuda::current_device()))));
    }
  }

  return output;
}

void MoEMLP::ForwardHelper(cudaStream_t stream) {
  torch::NoGradGuard no_grad;
  auto& input = buffer_[0];
  auto& output = buffer_[1];

  if (expert_type_ == NLLB_MOE_DENSE_ACT_DENSE) {
    auto& fc1 = param_[0];
    auto& fc2 = param_[1];
    output.copy_(
        torch::matmul(torch::relu(torch::matmul(input, fc1.transpose(0, 1))),
                      fc2.transpose(0, 1)));
    return;
  }

  if (expert_type_ == DEEPSEEK_MOE_DENSE_ACT_DENSE ||
      expert_type_ == MIXTRAL_MOE_DENSE_ACT_DENSE) {
    auto& gate_proj = param_[0];
    auto& up_proj =
        (expert_type_ == DEEPSEEK_MOE_DENSE_ACT_DENSE) ? param_[1] : param_[2];
    auto& down_proj =
        (expert_type_ == DEEPSEEK_MOE_DENSE_ACT_DENSE) ? param_[2] : param_[1];

    auto& gate_out = buffer_[2];
    auto& fused_out = buffer_[3];

    fused_moe_ffn_into(input, gate_proj, up_proj, down_proj, gate_out,
                       fused_out, output, stream);
    return;
  }
  DLOG_FATAL("MoEMLP::forward: expert_type not supported", expert_type_);
}
