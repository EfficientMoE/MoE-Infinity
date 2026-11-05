// Thin pybind shim that calls the production kernels compiled into
// moe_infinity/_store*.so, so the audit measures the exact shipped binary
// rather than a rebuilt copy.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

#include "kernel/moe_activation.h"
#include "../../../extensions/kernel/v4_fp4/gptoss_expert.h"
#include "../../../extensions/kernel/v4_fp4/mxfp4_gptoss_moe.h"

void fused_moe_ffn_into(at::Tensor& hidden, at::Tensor& gate_proj,
                        at::Tensor& up_proj, at::Tensor& down_proj,
                        at::Tensor& gate_buf, at::Tensor& fused_buf,
                        at::Tensor& output, cudaStream_t stream);
void fused_moe_ffn_into(at::Tensor& hidden, at::Tensor& gate_proj,
                        at::Tensor& up_proj, at::Tensor& down_proj,
                        at::Tensor& gate_buf, at::Tensor& fused_buf,
                        at::Tensor& output, const MoEActivation& activation,
                        cudaStream_t stream);
void mxfp4_dequant_cuda(const void* packed, const void* scales, void* output,
                        int rows, int packed_cols, int scale_cols,
                        int block_size, cudaStream_t stream);
void fp8_dequant_blockwise_cuda(const void* weight, const void* scale,
                                void* out, int N, int K, cudaStream_t stream);

static void py_fused_moe_ffn_into(at::Tensor hidden, at::Tensor gate_proj,
                                  at::Tensor up_proj, at::Tensor down_proj,
                                  at::Tensor gate_buf, at::Tensor fused_buf,
                                  at::Tensor output) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  fused_moe_ffn_into(hidden, gate_proj, up_proj, down_proj, gate_buf, fused_buf,
                     output, stream);
}

static void py_fused_moe_ffn_act_into(at::Tensor hidden, at::Tensor gate_proj,
                                      at::Tensor up_proj, at::Tensor down_proj,
                                      at::Tensor gate_buf, at::Tensor fused_buf,
                                      at::Tensor output, int kind, float alpha,
                                      float limit) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  MoEActivation act{static_cast<MoEActivationKind>(kind), alpha, limit};
  fused_moe_ffn_into(hidden, gate_proj, up_proj, down_proj, gate_buf, fused_buf,
                     output, act, stream);
}

static void py_mxfp4_dequant(at::Tensor packed, at::Tensor scales,
                             at::Tensor output) {
  TORCH_CHECK(packed.is_contiguous() && scales.is_contiguous() &&
              output.is_contiguous());
  int rows = packed.size(0);
  int packed_cols = packed.size(1);
  int scale_cols = scales.size(1);
  int block_size = packed_cols * 2 / scale_cols;
  mxfp4_dequant_cuda(packed.data_ptr(), scales.data_ptr(), output.data_ptr(),
                     rows, packed_cols, scale_cols, block_size,
                     at::cuda::getCurrentCUDAStream().stream());
}

static void py_fp8_dequant_blockwise(at::Tensor weight_u8, at::Tensor scale,
                                     at::Tensor out) {
  TORCH_CHECK(weight_u8.is_contiguous() && scale.is_contiguous() &&
              out.is_contiguous());
  fp8_dequant_blockwise_cuda(
      weight_u8.data_ptr(), scale.data_ptr(), out.data_ptr(), weight_u8.size(0),
      weight_u8.size(1), at::cuda::getCurrentCUDAStream().stream());
}

// MoEMLP's GPT-OSS path (ForwardHelper) with the scratch held by the caller,
// so repeated calls reuse it like an expert executor does.
static void py_gptoss_mxfp4_expert_pro6000(
    at::Tensor x, at::Tensor gu_blocks, at::Tensor gu_scales,
    at::Tensor gu_bias, at::Tensor d_blocks, at::Tensor d_scales,
    at::Tensor d_bias, at::Tensor out, at::Tensor arena, at::Tensor counters) {
  GptOssExpertScratch scratch{arena, counters};
  TORCH_CHECK(gptoss_mxfp4_expert_forward_pro6000(
      x, gu_blocks, gu_scales, gu_bias, d_blocks, d_scales, d_bias, out,
      scratch, at::cuda::getCurrentCUDAStream().stream()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_moe_ffn_into", &py_fused_moe_ffn_into);
  m.def("fused_moe_ffn_act_into", &py_fused_moe_ffn_act_into);
  m.def("mxfp4_dequant", &py_mxfp4_dequant);
  m.def("fp8_dequant_blockwise", &py_fp8_dequant_blockwise);
  m.def("gptoss_mxfp4_expert_pro6000", &py_gptoss_mxfp4_expert_pro6000);
  m.def("gptoss_expert_arena_bytes", [](int64_t M, int64_t H, int64_t I) {
    return static_cast<int64_t>(gptoss_expert_arena_bytes(M, H, I));
  });
  m.attr("mxfp4_stream_k_counters") = kMxfp4StreamKCounters;
}
