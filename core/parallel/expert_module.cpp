// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "expert_module.h"
#include "aio/archer_tensor_handle.h"
#include "memory/caching_allocator.h"
#include "utils/logger.h"

static const int64_t kMaxTokens = 128;

SwitchTransformersDenseActDense::SwitchTransformersDenseActDense(int dtype) {
  // auto tensor_dtype = dtype_to_torch(dtype);
  auto options = torch::TensorOptions().device(torch::kCPU);
  wi = register_parameter("wi", torch::zeros({1}, options));
  wo = register_parameter("wo", torch::zeros({1}, options));
}

void SwitchTransformersDenseActDense::SetTensorsFromBlob(
    void* ptr, const std::vector<std::uint32_t>& tensor_ids,
    const torch::Device& device) {
  wi = kTensorIndex->find(tensor_ids[0])->second.tensor;
  wo = kTensorIndex->find(tensor_ids[1])->second.tensor;
}

void SwitchTransformersDenseActDense::SetModuleFromBlob(
    torch::jit::script::Module* ptr) {
  for (auto it = ptr->parameters().begin(); it != ptr->parameters().end();
       ++it) {
    auto tensor = *it;
    if ((*it).name() == "wi") {
      (*it).set_data(wi);
    } else if ((*it).name() == "wo") {
      (*it).set_data(wo);
    }
  }
}

torch::Tensor SwitchTransformersDenseActDense::forward(
    torch::Tensor hidden_states, cudaStream_t stream) {
  // DLOG_TRACE("SwitchTransformersDenseActDense wi {} wo {}, hidden_states {}",
  //                  torch_dtype_to_int(wi.dtype()),
  //                     torch_dtype_to_int(wo.dtype()),
  //                     torch_dtype_to_int(hidden_states.dtype()));
  // DLOG_TRACE("SwitchTransformersDenseActDense wi {} wo {}, hidden_states {}",
  return torch::matmul(
      torch::relu(torch::matmul(hidden_states,
                                wi.transpose(0, 1).to(hidden_states.dtype()))),
      wo.transpose(0, 1).to(hidden_states.dtype()));
}

SwitchTransformersDenseGatedActDense::SwitchTransformersDenseGatedActDense(
    int dtype) {
  auto tensor_dtype = dtype_to_torch(dtype);
  auto options = torch::TensorOptions().dtype(tensor_dtype).device(torch::kCPU);
  wi_0 = register_parameter("wi_0", torch::zeros({1}, options));
  wi_1 = register_parameter("wi_1", torch::zeros({1}, options));
  wo = register_parameter("wo", torch::zeros({1}));
}

void SwitchTransformersDenseGatedActDense::SetTensorsFromBlob(
    void* ptr, const std::vector<std::uint32_t>& tensor_ids,
    const torch::Device& device) {
  wi_0 = kTensorIndex->find(tensor_ids[0])->second.tensor;
  wi_1 = kTensorIndex->find(tensor_ids[1])->second.tensor;
  wo = kTensorIndex->find(tensor_ids[2])->second.tensor;
}

void SwitchTransformersDenseGatedActDense::SetModuleFromBlob(
    torch::jit::script::Module* ptr) {
  for (auto it = ptr->parameters().begin(); it != ptr->parameters().end();
       ++it) {
    auto tensor = *it;
    if ((*it).name() == "wi_0") {
      (*it).set_data(wi_0);
    } else if ((*it).name() == "wi_1") {
      (*it).set_data(wi_1);
    } else if ((*it).name() == "wo") {
      (*it).set_data(wo);
    }
  }
}

torch::Tensor SwitchTransformersDenseGatedActDense::forward(
    torch::Tensor hidden_states, cudaStream_t stream) {
  auto gate = torch::gelu(torch::matmul(hidden_states, wi_0.transpose(0, 1)));
  auto linear = torch::matmul(hidden_states, wi_1.transpose(0, 1));
  return torch::matmul(torch::mul(gate, linear), wo.transpose(0, 1));
}

NllbMoeDenseActDense::NllbMoeDenseActDense(int dtype) {
  auto tensor_dtype = dtype_to_torch(dtype);
  auto options = torch::TensorOptions().dtype(tensor_dtype).device(torch::kCPU);
  fc1 = register_parameter("fc1", torch::zeros({1}, options));
  fc2 = register_parameter("fc2", torch::zeros({1}, options));
  fc1_bias = register_parameter("fc1_bias", torch::zeros({1}, options));
  fc2_bias = register_parameter("fc2_bias", torch::zeros({1}, options));
}

void NllbMoeDenseActDense::SetTensorsFromBlob(
    void* ptr, const std::vector<std::uint32_t>& tensor_ids,
    const torch::Device& device) {
  fc1 = kTensorIndex->find(tensor_ids[0])->second.tensor;
  fc1_bias = kTensorIndex->find(tensor_ids[1])->second.tensor;
  fc2 = kTensorIndex->find(tensor_ids[2])->second.tensor;
  fc2_bias = kTensorIndex->find(tensor_ids[3])->second.tensor;
}

void NllbMoeDenseActDense::SetModuleFromBlob(torch::jit::script::Module* ptr) {
  for (auto it = ptr->parameters().begin(); it != ptr->parameters().end();
       ++it) {
    auto tensor = *it;
    if ((*it).name() == "fc1") {
      (*it).set_data(fc1);
    } else if ((*it).name() == "fc1_bias") {
      (*it).set_data(fc1_bias);
    } else if ((*it).name() == "fc2") {
      (*it).set_data(fc2);
    } else if ((*it).name() == "fc2_bias") {
      (*it).set_data(fc2_bias);
    }
  }
}

torch::Tensor NllbMoeDenseActDense::forward(torch::Tensor hidden_states,
                                            cudaStream_t stream) {
  // DLOG_TRACE("NllbMoeDenseActDense fc1 {} fc1_bias {} fc2 {} fc2_bias {}
  // hidden_states
  // {}",
  //                  fc1.device().str(),
  //                  fc1_bias.device().str(),
  //                  fc2.device().str(),
  //                  fc2_bias.device().str(),
  //                  hidden_states.device().str());
  return torch::matmul(
             torch::relu(torch::matmul(hidden_states, fc1.transpose(0, 1)) +
                         fc1_bias),
             fc2.transpose(0, 1)) +
         fc2_bias;
}

FSGPTMoEDenseActDense::FSGPTMoEDenseActDense(int dtype) {
  auto tensor_dtype = dtype_to_torch(dtype);
  auto options = torch::TensorOptions().dtype(tensor_dtype).device(torch::kCPU);
  fc1 = register_parameter("fc1", torch::zeros({1}, options));
  fc2 = register_parameter("fc2", torch::zeros({1}, options));
  fc1_bias = register_parameter("fc1_bias", torch::zeros({1}, options));
  fc2_bias = register_parameter("fc2_bias", torch::zeros({1}, options));
}

void FSGPTMoEDenseActDense::SetTensorsFromBlob(
    void* ptr, const std::vector<std::uint32_t>& tensor_ids,
    const torch::Device& device) {
  fc1 = kTensorIndex->find(tensor_ids[0])->second.tensor;
  fc1_bias = kTensorIndex->find(tensor_ids[1])->second.tensor;
  fc2 = kTensorIndex->find(tensor_ids[2])->second.tensor;
  fc2_bias = kTensorIndex->find(tensor_ids[3])->second.tensor;
}

void FSGPTMoEDenseActDense::SetModuleFromBlob(torch::jit::script::Module* ptr) {
  for (auto it = ptr->parameters().begin(); it != ptr->parameters().end();
       ++it) {
    auto tensor = *it;
    if ((*it).name() == "fc1") {
      (*it).set_data(fc1);
    } else if ((*it).name() == "fc1_bias") {
      (*it).set_data(fc1_bias);
    } else if ((*it).name() == "fc2") {
      (*it).set_data(fc2);
    } else if ((*it).name() == "fc2_bias") {
      (*it).set_data(fc2_bias);
    }
  }
}

torch::Tensor FSGPTMoEDenseActDense::forward(torch::Tensor hidden_states,
                                             cudaStream_t stream) {
  // DLOG_TRACE("FSGPTMoEDenseActDense fc1 {} fc1_bias {} fc2 {} fc2_bias {}
  // hidden_states
  // {}",
  //                  fc1.device().str(),
  //                  fc1_bias.device().str(),
  //                  fc2.device().str(),
  //                  fc2_bias.device().str(),
  //                  hidden_states.device().str());
  if (hidden_states.dtype() != fc1.dtype())
    hidden_states = hidden_states.to(fc1.dtype());
  return torch::matmul(
             torch::relu(torch::matmul(hidden_states, fc1.transpose(0, 1)) +
                         fc1_bias),
             fc2.transpose(0, 1)) +
         fc2_bias;
}

MixtralMoEDenseActDense::MixtralMoEDenseActDense(int dtype) {
  auto tensor_dtype = dtype_to_torch(dtype);
  auto options = torch::TensorOptions().dtype(tensor_dtype).device(torch::kCPU);
  w1 = register_parameter("w1", torch::zeros({1}, options));
  w2 = register_parameter("w2", torch::zeros({1}, options));
  w3 = register_parameter("w3", torch::zeros({1}, options));
}

void MixtralMoEDenseActDense::SetTensorsFromBlob(
    void* ptr, const std::vector<std::uint32_t>& tensor_ids,
    const torch::Device& device) {
  w1 = kTensorIndex->find(tensor_ids[0])->second.tensor;
  w2 = kTensorIndex->find(tensor_ids[1])->second.tensor;
  w3 = kTensorIndex->find(tensor_ids[2])->second.tensor;
}

void MixtralMoEDenseActDense::SetModuleFromBlob(
    torch::jit::script::Module* ptr) {
  for (auto it = ptr->parameters().begin(); it != ptr->parameters().end();
       ++it) {
    auto tensor = *it;
    if ((*it).name() == "w1") {
      (*it).set_data(w1);
    } else if ((*it).name() == "w2") {
      (*it).set_data(w2);
    } else if ((*it).name() == "w3") {
      (*it).set_data(w3);
    }
  }
}

torch::Tensor MixtralMoEDenseActDense::forward(torch::Tensor hidden_states,
                                               cudaStream_t stream) {
  /*
  current_hidden_states = self.silu(self.w1(hidden_states)) *
  self.w3(hidden_states) current_hidden_states = self.w2(current_hidden_states)
  return current_hidden_states
  */
  // int w1_nan = torch::sum(torch::isnan(w1)).item<int>();
  // int w2_nan = torch::sum(torch::isnan(w2)).item<int>();
  // int w3_nan = torch::sum(torch::isnan(w3)).item<int>();
  // int hidden_states_nan =
  // torch::sum(torch::isnan(hidden_states)).item<int>(); std::cout <<
  // "MixtralMoEDenseActDense w1 " << w1_nan << " w2 " << w2_nan << " w3 " <<
  // w3_nan
  // << " hidden_states " << hidden_states_nan << std::endl;

  // assert(w1_nan == 0);
  // assert(w2_nan == 0);
  // assert(w3_nan == 0);
  // assert(hidden_states_nan == 0);

  // auto gate_proj = torch::silu(torch::matmul(hidden_states, w1.transpose(0,
  // 1))); auto up_proj = torch::matmul(hidden_states, w3.transpose(0, 1)); auto
  // down_proj = torch::matmul(gate_proj * up_proj, w2.transpose(0, 1));

  return torch::matmul(
      torch::silu(torch::matmul(hidden_states, w1.transpose(0, 1))) *
          torch::matmul(hidden_states, w3.transpose(0, 1)),
      w2.transpose(0, 1));
}

DeepSeekMoEDenseActDense::DeepSeekMoEDenseActDense(int dtype) {
  auto tensor_dtype = dtype_to_torch(dtype);
  auto options = torch::TensorOptions().dtype(tensor_dtype).device(torch::kCPU);
  gate_proj = register_parameter("gate_proj", torch::zeros({1}, options));
  up_proj = register_parameter("up_proj", torch::zeros({1}, options));
  down_proj = register_parameter("down_proj", torch::zeros({1}, options));
}

void DeepSeekMoEDenseActDense::SetTensorsFromBlob(
    void* ptr, const std::vector<std::uint32_t>& tensor_ids,
    const torch::Device& device) {
  gate_proj = kTensorIndex->find(tensor_ids[0])->second.tensor;
  up_proj = kTensorIndex->find(tensor_ids[1])->second.tensor;
  down_proj = kTensorIndex->find(tensor_ids[2])->second.tensor;
}

void DeepSeekMoEDenseActDense::SetModuleFromBlob(
    torch::jit::script::Module* ptr) {
  for (auto it = ptr->named_parameters().begin();
       it != ptr->named_parameters().end(); ++it) {
    auto tensor = *it;
    // std::cout << (*it).name << std::endl;
    if ((*it).name.find("gate_proj") != std::string::npos) {
      (*it).value.set_data(gate_proj);
    } else if ((*it).name.find("up_proj") != std::string::npos) {
      (*it).value.set_data(up_proj);
    } else if ((*it).name.find("down_proj") != std::string::npos) {
      (*it).value.set_data(down_proj);
    }
  }
}

torch::Tensor DeepSeekMoEDenseActDense::forward(torch::Tensor hidden_states,
                                                cudaStream_t stream) {
  // DLOG_INFO("DeepSeekMoEDenseActDense gate_proj:", gate_proj.sizes().vec(),
  //           "up_proj:", up_proj.sizes().vec(), "down_proj:",
  //           down_proj.sizes().vec(), "hidden_states:",
  //           hidden_states.sizes().vec());
  // assert(torch::all(gate_proj == 0).item<bool>() == false);
  // assert(torch::all(up_proj == 0).item<bool>() == false);
  // assert(torch::all(down_proj == 0).item<bool>() == false);
  // return launch_fused_moe_ffn(hidden_states, gate_proj, up_proj, down_proj,
  // stream);
  return torch::matmul(
      torch::silu(torch::matmul(hidden_states, gate_proj.transpose(0, 1))) *
          torch::matmul(hidden_states, up_proj.transpose(0, 1)),
      down_proj.transpose(0, 1));
}

void ExpertNode::SetTensorsFromBlob(const torch::Device& device) {
  int expert_type = this->expert_type;
  switch (expert_type) {
    case SWITCH_TRANSFORMERS_DENSE_ACT_DENSE:
      reinterpret_cast<SwitchTransformersDenseActDense*>(module)
          ->SetTensorsFromBlob(node->device_memory_ptr, node->tensor_ids,
                               device);
      break;
    case SWITCH_TRANSFORMERS_DENSE_GATED_ACT_DENSE:
      reinterpret_cast<SwitchTransformersDenseGatedActDense*>(module)
          ->SetTensorsFromBlob(node->device_memory_ptr, node->tensor_ids,
                               device);
      break;
    case NLLB_MOE_DENSE_ACT_DENSE:
      reinterpret_cast<NllbMoeDenseActDense*>(module)->SetTensorsFromBlob(
          node->device_memory_ptr, node->tensor_ids, device);
      break;
    case FSGPT_MOE_DENSE_ACT_DENSE:
      reinterpret_cast<FSGPTMoEDenseActDense*>(module)->SetTensorsFromBlob(
          node->device_memory_ptr, node->tensor_ids, device);
      break;
    case MIXTRAL_MOE_DENSE_ACT_DENSE:
      reinterpret_cast<MixtralMoEDenseActDense*>(module)->SetTensorsFromBlob(
          node->device_memory_ptr, node->tensor_ids, device);
      break;
    case DEEPSEEK_MOE_DENSE_ACT_DENSE:
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
  // input_ = register_parameter("input", torch::zeros({1}, options));
  // output_ = register_parameter("output", torch::zeros({1}, options));
  // gate_proj_ = register_parameter("gate_proj", torch::zeros({1}, options));
  // up_proj_ = register_parameter("up_proj", torch::zeros({1}, options));
  // down_proj_ = register_parameter("down_proj", torch::zeros({1}, options));

  // fc1_bias_ = register_parameter("fc1_bias", torch::zeros({1}, options));
  // fc2_bias_ = register_parameter("fc2_bias", torch::zeros({1}, options));
  // fc3_bias_ = register_parameter("fc3_bias", torch::zeros({1}, options));

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

  for (size_t i = 0; i < tensor_ptrs.size(); i++) {
    auto [ptr, tensor_size] = tensor_ptrs[i];
    auto tensor_shape = tensor_shapes[i];
    CUDA_CHECK(cudaMemcpy(param_[i].data_ptr(), ptr, tensor_size,
                          cudaMemcpyDeviceToDevice));
  }
  param_set_ = true;
  // DLOG_FATAL(
  //     "MoEMLP::SetTensorsFromBlob: tensor_ids.size() should be 2,3,4, but got
  //     {}", tensor_ids.size());
}

torch::Tensor MoEMLP::forward(torch::Tensor hidden_states,
                              cudaStream_t stream) {
  int64_t batch_size = hidden_states.size(0);
  int64_t hdim = hidden_states.size(1);

  DLOG_FATAL_IF(batch_size > kMaxTokens || batch_size <= 0,
                "batch_size should be (0,", kMaxTokens, "] , but got",
                batch_size);

  DLOG_FATAL_IF(param_set_ == false, "param_set_ should be true");
  DLOG_FATAL_IF(param_init_ == false, "param_init_ should be true");

  auto& input_ = buffer_[0];
  auto& output_ = buffer_[1];

  // copy hidden_states to input_ using cudaMemcpy
  cudaMemcpy(input_.data_ptr(), hidden_states.data_ptr(),
             hidden_states.numel() * hidden_states.element_size(),
             cudaMemcpyDeviceToDevice);
  // cudaStreamSynchronize(stream);
  // if (warmup_count_ == 0 && graph_mode_) {
  //   graph_.replay();
  // }

  // if (warmup_count_ == 0 && !graph_mode_) {
  //   graph_.capture_begin();
  //   ForwardHelper();
  //   graph_.capture_end();
  //   graph_mode_ = true;
  // }

  // if (warmup_count_ > 0) {
  //   warmup_count_--;
  //   ForwardHelper();
  // }
  ForwardHelper();
  param_set_ = false;
  // slice until batch_size
  cudaStreamSynchronize(stream);
  // auto options = torch::TensorOptions()
  //                    .dtype(dtype_to_torch(dtype_))
  //                    .device(CUDA_DEVICE(at::cuda::current_device()));
  // auto output = torch::empty({batch_size, hdim}, options);
  // output.copy_(output_.index({torch::indexing::Slice(0, batch_size)}));
  // cudaStreamSynchronize(stream);
  // return std::move(output);
  return output_.index({torch::indexing::Slice(0, batch_size)});
}

void MoEMLP::ForwardHelper() {
  torch::NoGradGuard no_grad;
  if (expert_type_ == DEEPSEEK_MOE_DENSE_ACT_DENSE ||
      expert_type_ == MIXTRAL_MOE_DENSE_ACT_DENSE) {
    auto& gate_proj = param_[0];
    auto& up_proj =
        (expert_type_ == DEEPSEEK_MOE_DENSE_ACT_DENSE) ? param_[1] : param_[2];
    auto& down_proj =
        (expert_type_ == DEEPSEEK_MOE_DENSE_ACT_DENSE) ? param_[2] : param_[1];

    auto& input = buffer_[0];
    auto& output = buffer_[1];
    auto& gate_out = buffer_[2];
    auto& gate_act_out = buffer_[3];
    auto& up_out = buffer_[4];

    DLOG_TRACE("MoEMLP::forward: gate_proj", gate_proj.sizes().vec(), "up_proj",
               up_proj.sizes().vec(), "down_proj", down_proj.sizes().vec(),
               "input", input.sizes().vec(), "gate_out", gate_out.sizes().vec(),
               "gate_act_out", gate_act_out.sizes().vec(), "up_out",
               up_out.sizes().vec(), "output", output.sizes().vec());

    // gate step
    torch::matmul_out(gate_out, input, gate_proj.transpose(0, 1));

    // activation step
    torch::silu_out(gate_act_out, gate_out);

    // up step
    torch::matmul_out(up_out, input, up_proj.transpose(0, 1));

    // multiplication step, reuse gate_out
    torch::mul_out(gate_out, gate_act_out, up_out);

    // down step
    torch::matmul_out(output, gate_out, down_proj.transpose(0, 1));
    return;
  }
  DLOG_FATAL("MoEMLP::forward: expert_type not supported", expert_type_);
}
