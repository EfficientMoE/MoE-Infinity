// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void fp4_dequant_to_bf16(const void* packed, const void* scale_e8m0, void* out,
                         int N, int K, cudaStream_t stream);
void mxfp4_dequant_cuda(const void* packed, const void* scales, void* output,
                        int rows, int packed_cols, int scale_cols,
                        int block_size, cudaStream_t stream);

torch::Tensor mxfp4_dequant(torch::Tensor packed, torch::Tensor scales) {
  TORCH_CHECK(packed.is_cuda() && scales.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(packed.scalar_type() == torch::kUInt8, "packed must be uint8");
  TORCH_CHECK(scales.scalar_type() == torch::kUInt8, "scales must be uint8");
  TORCH_CHECK(packed.dim() == 2 && scales.dim() == 2, "2D tensors required");
  int rows = packed.size(0);
  int packed_cols = packed.size(1);
  int scale_cols = scales.size(1);
  int block_size = packed_cols * 2 / scale_cols;
  packed = packed.contiguous();
  scales = scales.contiguous();
  auto output = torch::empty({rows, packed_cols * 2},
                             packed.options().dtype(torch::kBFloat16));
  auto stream = at::cuda::getCurrentCUDAStream(packed.device().index());
  mxfp4_dequant_cuda(packed.data_ptr(), scales.data_ptr(), output.data_ptr(),
                     rows, packed_cols, scale_cols, block_size, stream);
  return output;
}

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp4_dequant", &fp4_dequant, "FP4 E2M1 packed -> BF16 dequant");
  m.def("mxfp4_dequant", &mxfp4_dequant, "MXFP4 uint8 blocks/scales to BF16");
  m.def("v4_expert_forward", &v4_expert_forward,
        "V4 FP4 routed-expert SwiGLU forward");
}
