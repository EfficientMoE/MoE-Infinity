// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "expert_module.h"
// #include "memory/caching_allocator.h"
#include "utils/cuda_utils.h"
#include "utils/logger.h"
#include "kernel/fused_moe_mlp.h"

static const int64_t kMaxTokens = 256;
static const int64_t kFp8BlockSize = 128;

// Block-wise FP8 dequant: expand [ceil(n/128), ceil(k/128)] scales to [n, k]
// and multiply. Must match moe_infinity.utils.fp8.dequant_fp8_blockwise.
torch::Tensor DequantFp8Blockwise(const torch::Tensor& weight,
                                  const torch::Tensor& scale) {
  int64_t n = weight.size(0);
  int64_t k = weight.size(1);
  auto scale_full = scale.to(torch::kFloat32)
                        .repeat_interleave(kFp8BlockSize, 0)
                        .repeat_interleave(kFp8BlockSize, 1)
                        .narrow(0, 0, n)
                        .narrow(1, 0, k);
  return (weight.to(torch::kFloat32) * scale_full).to(torch::kBFloat16);
}

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
  int device = at::cuda::current_device();
  auto options = torch::TensorOptions()
                     .dtype(dtype_to_torch(dtype_))
                     .device(CUDA_DEVICE(device));

  // Safety: exec_state is FETCHING/EXECUTING throughout forward(), so the
  // expert's GPU memory cannot be evicted until OutputFunc resets to IDLE.
  std::vector<std::vector<int64_t>> tensor_shapes;
  for (size_t i = 0; i < tensor_ids.size(); i++) {
    auto& tensor = kTensorIndex->find(tensor_ids[i])->second.tensor;
    tensor_shapes.push_back(tensor.sizes().vec());
    param_[i].set_data(tensor);
  }

  if (!param_init_) {
    auto allocator = c10::DeviceCachingAllocator::get(device);

    int64_t hdim = tensor_shapes[0][1];
    int64_t idim = tensor_shapes[0][0];

    std::vector<std::vector<int64_t>> data_shapes;
    data_shapes.push_back({kMaxTokens, hdim});
    data_shapes.push_back({kMaxTokens, hdim});

    for (size_t i = 0; i < tensor_shapes.size(); i++) {
      data_shapes.push_back({kMaxTokens, idim});
    }

    for (size_t i = 0; i < data_shapes.size(); i++) {
      auto data_shape = data_shapes[i];
      auto data_size = torch_shape_size(data_shape, dtype_);
      void* buffer_ptr = allocator->allocate(data_size);
      buffer_[i].set_data(torch::from_blob(buffer_ptr, data_shape,
                                           DoNothingDeleter<void>{}, options));
      DLOG_TRACE("MoEMLP::SetTensorsFromIds: buffer_ tensor", i, "data_shape",
                 data_shape, "data_size", data_size, "device",
                 buffer_[i].device().str());
    }
    param_init_ = true;
  }

  if (!param_init_) {
    // Allocate computation buffers (input, output, intermediates) once.
    // These are reused across expert invocations.
    auto allocator = c10::DeviceCachingAllocator::get(device);

    // MLP tensor shape: weight is [intermediate, hidden], so
    //   hdim = tensor_shapes[0][1], idim = tensor_shapes[0][0]
    int64_t hdim = tensor_shapes[0][1];
    int64_t idim = tensor_shapes[0][0];

    std::vector<std::vector<int64_t>> data_shapes;
    data_shapes.push_back({kMaxTokens, hdim});  // input buffer
    data_shapes.push_back({kMaxTokens, hdim});  // output buffer

    for (size_t i = 0; i < tensor_shapes.size(); i++) {
      data_shapes.push_back({kMaxTokens, idim});  // intermediate buffers
    }

    for (size_t i = 0; i < data_shapes.size(); i++) {
      auto data_shape = data_shapes[i];
      auto data_size = torch_shape_size(data_shape, dtype_);
      void* buffer_ptr = allocator->allocate(data_size);
      buffer_[i].set_data(torch::from_blob(buffer_ptr, data_shape,
                                           DoNothingDeleter<void>{}, options));
      DLOG_TRACE("MoEMLP::SetTensorsFromIds: buffer_ tensor", i, "data_shape",
                 data_shape, "data_size", data_size, "device",
                 buffer_[i].device().str());
    }
    param_init_ = true;
  }

  assert(param_init_ == true);
  assert(param_set_ == false);
  param_set_ = true;
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

    torch::Tensor gate_w = gate_proj;
    torch::Tensor up_w = up_proj;
    torch::Tensor down_w = down_proj;
    if (gate_proj.scalar_type() == torch::kFloat8_e4m3fn && has_scales_) {
      // GLM Option B: block-quantized fp8 experts + block scales delivered via
      // expert_dispatcher.set_layer_scales() at init; dequant on-GPU here.
      gate_w = DequantFp8Blockwise(gate_proj, scale_gate_);
      up_w = DequantFp8Blockwise(up_proj, scale_up_);
      down_w = DequantFp8Blockwise(down_proj, scale_down_);
    } else if (gate_proj.scalar_type() == torch::kFloat8_e4m3fn &&
               dtype_ != DTYPE_FP8_E4M3FN) {
      // fp8 weights but no scales registered and the dispatcher was not built
      // for flat-fp8 (DeepSeek-V3, dtype_ == DTYPE_FP8_E4M3FN, which keeps the
      // historical pass-through below): misconfigured scale delivery.
      TORCH_CHECK(false,
                  "fp8 block-quantized expert weights require scales; call "
                  "expert_dispatcher.set_layer_scales() at init");
    }

    fused_moe_ffn_into(input, gate_w, up_w, down_w, gate_out, fused_out,
                       output, stream);
    return;
  }
  DLOG_FATAL("MoEMLP::forward: expert_type not supported", expert_type_);
}
