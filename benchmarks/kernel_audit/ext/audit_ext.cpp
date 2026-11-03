// Thin pybind shim that calls the production kernels compiled into
// moe_infinity/_store*.so, so the audit measures the exact shipped binary
// rather than a rebuilt copy.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

void fused_moe_ffn_into(at::Tensor& hidden, at::Tensor& gate_proj,
                        at::Tensor& up_proj, at::Tensor& down_proj,
                        at::Tensor& gate_buf, at::Tensor& fused_buf,
                        at::Tensor& output, cudaStream_t stream);
void mxfp4_dequant_cuda(const void* packed, const void* scales, void* output,
                        int rows, int packed_cols, int scale_cols,
                        int block_size, cudaStream_t stream);
void fp8_dequant_blockwise_cuda(const void* weight, const void* scale,
                                void* out, int N, int K, cudaStream_t stream);
void fused_fp8_moe_ffn_sm120_pro6000_into(
    const at::Tensor& hidden, const at::Tensor& gate_w,
    const at::Tensor& gate_s, const at::Tensor& up_w, const at::Tensor& up_s,
    const at::Tensor& down_w, const at::Tensor& down_s, at::Tensor& output,
    cudaStream_t stream);
bool fp8_moe_sm120_pro6000_supported(int64_t hidden, int64_t inter, int device);

static void py_fused_moe_ffn_into(at::Tensor hidden, at::Tensor gate_proj,
                                  at::Tensor up_proj, at::Tensor down_proj,
                                  at::Tensor gate_buf, at::Tensor fused_buf,
                                  at::Tensor output) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  fused_moe_ffn_into(hidden, gate_proj, up_proj, down_proj, gate_buf, fused_buf,
                     output, stream);
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

static void py_fused_fp8_moe_ffn_sm120_pro6000_into(
    at::Tensor hidden, at::Tensor gate_w, at::Tensor gate_s, at::Tensor up_w,
    at::Tensor up_s, at::Tensor down_w, at::Tensor down_s, at::Tensor output) {
  fused_fp8_moe_ffn_sm120_pro6000_into(
      hidden, gate_w, gate_s, up_w, up_s, down_w, down_s, output,
      at::cuda::getCurrentCUDAStream().stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_moe_ffn_into", &py_fused_moe_ffn_into);
  m.def("mxfp4_dequant", &py_mxfp4_dequant);
  m.def("fp8_dequant_blockwise", &py_fp8_dequant_blockwise);
  m.def("fused_fp8_moe_ffn_sm120_pro6000_into",
        &py_fused_fp8_moe_ffn_sm120_pro6000_into);
  m.def("fp8_moe_sm120_pro6000_supported", &fp8_moe_sm120_pro6000_supported);
}
