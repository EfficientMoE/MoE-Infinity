// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/stl.h>
#include <map>
#include <string>

void fp4_dequant_to_bf16(const void* packed, const void* scale_e8m0, void* out,
                         int N, int K, cudaStream_t stream);

void fp8_dequant_blockwise_cuda(const void* weight, const void* scale,
                                void* out, int N, int K, cudaStream_t stream);

// packed: [N, K/2] uint8 (view of float4_e2m1fn_x2); scale: [N, K/32] e8m0.
// Returns dequantized BF16 weight [N, K].
torch::Tensor fp4_dequant(torch::Tensor packed, torch::Tensor scale,
                          int64_t K) {
  TORCH_CHECK(packed.is_cuda(), "packed must be CUDA");
  int N = packed.size(0);
  auto out = torch::empty(
      {N, K},
      torch::TensorOptions().dtype(torch::kBFloat16).device(packed.device()));
  auto stream = at::cuda::getCurrentCUDAStream(packed.device().index());
  auto packed_u8 = packed.view(torch::kUInt8).contiguous();
  auto scale_u8 = scale.view(torch::kUInt8).contiguous();
  fp4_dequant_to_bf16(packed_u8.data_ptr(), scale_u8.data_ptr(), out.data_ptr(),
                      N, (int)K, stream);
  return out;
}

// Full routed-expert forward: dequant FP4 w1/w2/w3 -> BF16, then SwiGLU MLP.
//   gate = x @ w1^T ; up = x @ w3^T ; h = silu(clamp(gate)) * clamp(up)
//   out  = h @ w2^T
// x: [M, hidden] bf16; w*: packed FP4 [N, K/2]; s*: e8m0 [N, K/32].
torch::Tensor v4_expert_forward(torch::Tensor x, torch::Tensor w1,
                                torch::Tensor s1, torch::Tensor w2,
                                torch::Tensor s2, torch::Tensor w3,
                                torch::Tensor s3, double swiglu_limit) {
  int64_t hidden = x.size(1);
  int64_t inter = w1.size(0);
  auto dw1 = fp4_dequant(w1, s1, hidden);  // [inter, hidden]
  auto dw3 = fp4_dequant(w3, s3, hidden);  // [inter, hidden]
  auto dw2 = fp4_dequant(w2, s2, inter);   // [hidden, inter]
  auto gate = torch::matmul(x, dw1.transpose(0, 1)).to(torch::kFloat32);
  auto up = torch::matmul(x, dw3.transpose(0, 1)).to(torch::kFloat32);
  if (swiglu_limit > 0) {
    gate = torch::clamp_max(gate, swiglu_limit);
    up = torch::clamp(up, -swiglu_limit, swiglu_limit);
  }
  auto h = (torch::silu(gate) * up).to(x.dtype());
  return torch::matmul(h, dw2.transpose(0, 1));
}

// FP8 E4M3 block-scale dequant for GLM-5.2-FP8 routed experts.
// weight: [N, K] float8_e4m3fn (passed as uint8 view); scale: [ceil(N/128),
// ceil(K/128)] float32. Returns dequantized BF16 weight [N, K].
torch::Tensor fp8_dequant_blockwise(torch::Tensor weight, torch::Tensor scale) {
  TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
  TORCH_CHECK(scale.is_cuda(), "scale must be CUDA");
  TORCH_CHECK(weight.dim() == 2, "weight must be 2D [N, K]");
  TORCH_CHECK(scale.dim() == 2, "scale must be 2D [SN, SK]");

  int N = weight.size(0);
  int K = weight.size(1);

  auto out = torch::empty(
      {N, K},
      torch::TensorOptions().dtype(torch::kBFloat16).device(weight.device()));
  auto stream = at::cuda::getCurrentCUDAStream(weight.device().index());

  // Accept fp8 tensor or uint8 view
  auto w_u8 = weight.view(torch::kUInt8).contiguous();
  auto s_f32 = scale.to(torch::kFloat32).contiguous();

  fp8_dequant_blockwise_cuda(w_u8.data_ptr(), s_f32.data_ptr(), out.data_ptr(),
                             N, K, stream);
  return out;
}

// set_scales: no-op stub for T15 dispatcher integration (deferred).
// Receives a dict of base_key -> scale_tensor for fp8-in-store dequant-on-copy.
// Full integration (storing scales in native expert_dispatcher for H2D copy) is
// deferred.
void set_scales(const std::map<std::string, torch::Tensor>& /*scales*/) {
  // No-op: full dispatcher integration deferred to T15-full.
  // This binding satisfies the Python call site: dispatcher.set_scales(scales).
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp4_dequant", &fp4_dequant, "FP4 E2M1 packed -> BF16 dequant");
  m.def("v4_expert_forward", &v4_expert_forward,
        "V4 FP4 routed-expert SwiGLU forward");
  m.def("fp8_dequant_blockwise", &fp8_dequant_blockwise,
        "FP8 E4M3 block-scale (128x128) weight -> BF16 dequant (GLM-5.2-FP8)");
  m.def("set_scales", &set_scales,
        "Store fp8 block scales for dequant-on-copy (no-op stub; full "
        "integration deferred)");
}
