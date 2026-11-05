// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "expert_module.h"
#include "utils/cuda_utils.h"
#include "utils/logger.h"
#include "kernel/fused_moe_mlp.h"
#include "kernel/hw_profile.h"
#include "kernel/v4_fp4/mxfp4_gptoss_moe.h"
extern void fp8_dequant_blockwise_cuda(const void* weight, const void* scale,
                                       void* out, int N, int K,
                                       cudaStream_t stream);

static const int64_t kMaxTokens = 8192;

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
    case ExpertType::GptOssMoeDenseActDense:
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
  int num_params = 8;
  for (int i = 0; i < num_params; i++) {
    param_.push_back(torch::zeros({1}, options));
  }
}

void MoEMLP::SetRepresentation(const ExpertExecutionDescriptor& descriptor) {
  const std::size_t expected_roles =
      descriptor.execution_kind == 1 || descriptor.execution_kind == 2 ? 6 : 3;
  TORCH_CHECK(descriptor.tensor_ids.size() == descriptor.tensor_roles.size(),
              "expert tensor IDs and roles must have identical lengths");
  TORCH_CHECK(descriptor.tensor_ids.size() == expected_roles,
              "expert representation has an invalid role count");
  representation_execution_kind_ = descriptor.execution_kind;
  SetTensorsFromIds(descriptor.tensor_ids);
}

void MoEMLP::PrepareRepresentation(cudaStream_t stream) {
  if (representation_execution_kind_ != 1) return;
  int device = at::cuda::current_device();
  std::array<torch::Tensor, 3> dequantized;
  for (std::size_t projection = 0; projection < 3; ++projection) {
    auto weight = param_[projection * 2];
    auto scale = param_[projection * 2 + 1];
    TORCH_CHECK(weight.scalar_type() == torch::kFloat8_e4m3fn,
                "generic FP8 representation requires float8_e4m3fn weights");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32,
                "generic FP8 representation requires float32 scales");
    auto output =
        torch::empty(weight.sizes(), torch::TensorOptions()
                                         .dtype(torch::kBFloat16)
                                         .device(CUDA_DEVICE(device)));
    auto weight_bytes = weight.view(torch::kUInt8).contiguous();
    auto scale_f32 = scale.contiguous();
    fp8_dequant_blockwise_cuda(weight_bytes.data_ptr(), scale_f32.data_ptr(),
                               output.data_ptr(), weight.size(0),
                               weight.size(1), stream);
    dequantized[projection] = std::move(output);
  }
  for (std::size_t projection = 0; projection < 3; ++projection) {
    param_[projection] = std::move(dequantized[projection]);
  }
}

void MoEMLP::SetTensorsFromIds(const std::vector<std::uint32_t>& tensor_ids) {
  // Safety: exec_state is FETCHING/EXECUTING throughout forward(), so the
  // expert's GPU memory cannot be evicted until OutputFunc resets to IDLE.
  std::vector<torch::Tensor> tensors;
  tensors.reserve(tensor_ids.size());
  for (auto tensor_id : tensor_ids) {
    tensors.push_back(GetTensorStore()->index().find(tensor_id)->second.tensor);
  }
  SetTensors(tensors);
}

void MoEMLP::SetTensors(const std::vector<torch::Tensor>& tensors) {
  if (expert_type_ == GPT_OSS_MOE_DENSE_ACT_DENSE) {
    TORCH_CHECK(tensors.size() == 6,
                "GPT-OSS expert requires blocks/scales/bias for two "
                "projections");
  }
  TORCH_CHECK(!tensors.empty() && tensors.size() <= param_.size(),
              "MoEMLP expects 1..", param_.size(), " parameter tensors, got ",
              tensors.size());
  for (size_t i = 0; i < tensors.size(); i++) {
    param_[i].set_data(tensors[i]);
  }

  if (!param_init_) {
    // MLP weight is [intermediate, hidden]; GPT-OSS gate_up blocks are
    // [2 * intermediate, hidden / 2] packed MXFP4.
    const auto& shape = tensors[0].sizes();
    int64_t hdim;
    int64_t idim;
    if (expert_type_ == GPT_OSS_MOE_DENSE_ACT_DENSE) {
      hdim = shape[1] * 2;
      idim = shape[0] / 2;
    } else {
      hdim = shape[1];
      idim = shape[0];
    }
    // GPT-OSS keeps its intermediate in its own scratch (GptOssExpertScratch).
    const std::size_t num_inter =
        expert_type_ == GPT_OSS_MOE_DENSE_ACT_DENSE ? 0 : tensors.size();
    AllocateStaging(hdim, idim, num_inter);
    param_init_ = true;
  }

  assert(param_set_ == false);
  param_set_ = true;
}

void MoEMLP::AllocateStaging(int64_t hdim, int64_t idim,
                             std::size_t num_inter) {
  // Staging buffers (input, output, intermediates) are allocated once at
  // kMaxTokens rows and reused across expert invocations; forward() processes
  // larger batches in chunks of at most kMaxTokens rows.
  int device = at::cuda::current_device();
  auto options = torch::TensorOptions()
                     .dtype(dtype_to_torch(dtype_))
                     .device(CUDA_DEVICE(device));
  auto allocator = c10::DeviceCachingAllocator::get(device);

  staging_cols_.assign({hdim, hdim});
  staging_cols_.insert(staging_cols_.end(), num_inter, idim);
  TORCH_CHECK(staging_cols_.size() <= buffer_.size(),
              "MoEMLP staging slot count exceeds buffer_ size");

  for (size_t i = 0; i < staging_cols_.size(); i++) {
    std::vector<int64_t> data_shape = {kMaxTokens, staging_cols_[i]};
    auto data_size = torch_shape_size(data_shape, dtype_);
    void* buffer_ptr = allocator->allocate(data_size);
    buffer_[i].set_data(torch::from_blob(buffer_ptr, data_shape,
                                         DoNothingDeleter<void>{}, options));
    DLOG_TRACE("MoEMLP::AllocateStaging: buffer_ tensor", i, "data_shape",
               data_shape, "data_size", data_size, "device",
               buffer_[i].device().str());
  }
}

void MoEMLP::ResizeStaging(int64_t rows) {
  auto device = CUDA_DEVICE(at::cuda::current_device());
  for (size_t i = 0; i < staging_cols_.size(); i++) {
    auto& buffer = buffer_[i];
    if (buffer.size(0) == rows) continue;
    buffer.set_data(torch::from_blob(
        buffer.data_ptr(), {rows, staging_cols_[i]}, DoNothingDeleter<void>{},
        torch::TensorOptions().dtype(buffer.dtype()).device(device)));
  }
}

// Binds the stored MXFP4 tensors (gate_up blocks/scales/bias, down
// blocks/scales/bias). ForwardHelper consumes them directly; the BF16
// de-quantisation only runs on the fallback path.
void MoEMLP::DequantMxfp4Params(cudaStream_t stream, bool allow_pro6000) {
  if (expert_type_ != GPT_OSS_MOE_DENSE_ACT_DENSE) return;
  int device = at::cuda::current_device();
  if (!gpt_oss_gate_set_) {
    gpt_oss_mxfp4_pro6000_ =
        allow_pro6000 &&
        moe_infinity::hw::Supports(moe_infinity::hw::Gate::kRtxPro6000, device);
    gpt_oss_gate_set_ = true;
  }
  gpt_oss_param_.clear();
  for (int i = 0; i < 6; ++i) {
    gpt_oss_param_.push_back(param_[i].to(CUDA_DEVICE(device)));
  }
  DLOG_FATAL_IF(gpt_oss_param_[0].scalar_type() != torch::kUInt8 ||
                    gpt_oss_param_[1].scalar_type() != torch::kUInt8 ||
                    gpt_oss_param_[3].scalar_type() != torch::kUInt8 ||
                    gpt_oss_param_[4].scalar_type() != torch::kUInt8,
                "GPT-OSS MXFP4 blocks/scales must be uint8");
}

static torch::Tensor DequantMxfp4(const torch::Tensor& blocks,
                                  const torch::Tensor& scales,
                                  cudaStream_t stream) {
  auto packed = blocks.contiguous();
  auto sc = scales.contiguous();
  auto output = torch::empty({packed.size(0), packed.size(1) * 2},
                             packed.options().dtype(torch::kBFloat16));
  mxfp4_dequant_cuda(packed.data_ptr(), sc.data_ptr(), output.data_ptr(),
                     packed.size(0), packed.size(1), sc.size(1),
                     packed.size(1) * 2 / sc.size(1), stream);
  return output;
}

torch::Tensor MoEMLP::forward(torch::Tensor hidden_states, cudaStream_t stream,
                              cudaEvent_t kernel_start,
                              cudaEvent_t kernel_stop) {
  // Whatever happens below, the parameters are consumed and the staging
  // buffers return to kMaxTokens rows, so a failed call (TORCH_CHECK or a
  // kernel error) leaves the module usable for the next expert.
  struct Restore {
    MoEMLP* self;
    ~Restore() {
      self->param_set_ = false;
      try {
        self->ResizeStaging(kMaxTokens);
      } catch (const std::exception& e) {
        DLOG_ERROR("MoEMLP::forward: failed to restore staging buffers: ",
                   e.what());
      }
    }
  } restore{this};

  TORCH_CHECK(param_set_, "MoEMLP::forward called before SetTensors");
  TORCH_CHECK(param_init_, "MoEMLP::forward: staging buffers not allocated");

  auto& input_ = buffer_[0];
  auto& output_ = buffer_[1];

  TORCH_CHECK(hidden_states.dim() == 2,
              "hidden_states must be 2-D [tokens, hidden], got ",
              hidden_states.sizes());
  int64_t batch_size = hidden_states.size(0);
  TORCH_CHECK(batch_size > 0, "MoEMLP::forward: batch_size must be > 0");
  TORCH_CHECK(hidden_states.size(1) == input_.size(1),
              "hidden_states hidden dim ", hidden_states.size(1),
              " does not match expert hidden dim ", input_.size(1));
  TORCH_CHECK(hidden_states.scalar_type() == input_.scalar_type(),
              "hidden_states dtype must match expert input dtype");
  hidden_states = hidden_states.contiguous();

  const int64_t in_row_bytes = input_.size(1) * input_.element_size();
  const int64_t out_row_bytes = output_.size(1) * output_.element_size();
  // Experts that receive more than kMaxTokens tokens (long prefills, routing
  // skew) run in chunks of at most kMaxTokens rows through the same buffers.
  const bool chunked = batch_size > kMaxTokens;
  torch::Tensor output;
  if (chunked) {
    output = torch::empty({batch_size, output_.size(1)}, output_.options());
  }

  for (int64_t start = 0; start < batch_size; start += kMaxTokens) {
    const int64_t rows = std::min(kMaxTokens, batch_size - start);
    CUDA_CHECK(
        cudaMemcpyAsync(input_.data_ptr(),
                        static_cast<const char*>(hidden_states.data_ptr()) +
                            start * in_row_bytes,
                        rows * in_row_bytes, cudaMemcpyDeviceToDevice, stream));
    ResizeStaging(rows);
    if (start == 0 && kernel_start != nullptr) {
      cudaEventRecord(kernel_start, stream);
    }
    ForwardHelper(stream);
    if (chunked) {
      CUDA_CHECK(cudaMemcpyAsync(
          static_cast<char*>(output.data_ptr()) + start * out_row_bytes,
          output_.data_ptr(), rows * out_row_bytes, cudaMemcpyDeviceToDevice,
          stream));
    }
  }
  if (kernel_stop != nullptr) {
    cudaEventRecord(kernel_stop, stream);
  }
  if (!chunked) {
    output = output_.clone();
  }
  return output;
}

void MoEMLP::ForwardHelper(cudaStream_t stream) {
  torch::NoGradGuard no_grad;
  auto& input = buffer_[0];
  auto& output = buffer_[1];

  if (expert_type_ == GPT_OSS_MOE_DENSE_ACT_DENSE) {
    const auto& p = gpt_oss_param_;
    // RTX PRO 6000: MXFP4 tensor-core kernels on the stored blocks/scales
    // (bias, interleaved gate/up split and clamped SwiGLU-OAI fused).
    if (gpt_oss_mxfp4_pro6000_ && gptoss_mxfp4_expert_forward_pro6000(
                                      input, p[0], p[1], p[2], p[3], p[4], p[5],
                                      output, gpt_oss_scratch_, stream)) {
      return;
    }
    // Previous path (other devices, or layouts the kernels do not cover):
    // de-quantise to BF16 + matmul.
    auto gate_up =
        torch::matmul(input, DequantMxfp4(p[0], p[1], stream).transpose(0, 1)) +
        p[2];
    auto gate =
        gate_up.index({torch::indexing::Slice(),
                       torch::indexing::Slice(0, torch::indexing::None, 2)});
    auto up =
        gate_up.index({torch::indexing::Slice(),
                       torch::indexing::Slice(1, torch::indexing::None, 2)});
    gate = torch::clamp_max(gate, 7.0);
    up = torch::clamp(up, -7.0, 7.0);
    auto activated = (up + 1.0) * (gate * torch::sigmoid(gate * 1.702));
    output.copy_(
        torch::matmul(activated,
                      DequantMxfp4(p[3], p[4], stream).transpose(0, 1)) +
        p[5]);
    return;
  }

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

    TORCH_CHECK(gate_proj.scalar_type() == torch::kBFloat16 &&
                    up_proj.scalar_type() == torch::kBFloat16 &&
                    down_proj.scalar_type() == torch::kBFloat16,
                "expert projections must be BF16 before execution; got gate=",
                gate_proj.scalar_type(), ", up=", up_proj.scalar_type(),
                ", down=", down_proj.scalar_type(), ", representation=",
                static_cast<int>(representation_execution_kind_));

    auto& gate_out = buffer_[2];
    auto& fused_out = buffer_[3];

    fused_moe_ffn_into(input, gate_proj, up_proj, down_proj, gate_out,
                       fused_out, output, stream);
    return;
  }
  DLOG_FATAL("MoEMLP::forward: expert_type not supported", expert_type_);
}

void MoEMLP::SetFp8Scales(const std::vector<torch::Tensor>& scales) {
  fp8_scales_ = scales;
  has_fp8_scales_ = !scales.empty();
}

void MoEMLP::DequantFp8Params(cudaStream_t stream) {
  if (!has_fp8_scales_) return;
  int device = at::cuda::current_device();
  for (size_t i = 0; i < fp8_scales_.size() && i < param_.size(); ++i) {
    auto& w = param_[i];
    if (!w.defined()) continue;
    auto wdtype = w.scalar_type();
    if (wdtype != torch::kFloat8_e4m3fn && wdtype != torch::kUInt8) continue;
    auto& s = fp8_scales_[i];
    if (!s.defined()) continue;
    // Scales are expected to already be fp32 on this device (cached per-GPU
    // at ExpertDispatcher::SetScales time), making both .to() calls no-ops.
    auto s_gpu = s.to(torch::kFloat32).to(CUDA_DEVICE(device));
    int N = w.size(0);
    int K = w.size(1);
    auto out = torch::empty({N, K}, torch::TensorOptions()
                                        .dtype(torch::kBFloat16)
                                        .device(CUDA_DEVICE(device)));
    auto w_u8 = w.view(torch::kUInt8).contiguous();
    fp8_dequant_blockwise_cuda(w_u8.data_ptr(), s_gpu.data_ptr(),
                               out.data_ptr(), N, K, stream);
    param_[i] = out;
  }
}
