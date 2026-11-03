// Python binding for the small-M MXFP4 GEMM (moe_infinity._mxfp4_gemv).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <optional>

bool mxfp4_gemv_small_launch(const void* x, const void* w, const void* s,
                             const void* bias, void* out, int M, int N, int K,
                             int64_t stride_xm, int64_t stride_wn,
                             int64_t stride_sn, int64_t stride_om, int warps,
                             cudaStream_t stream);

// out[M, N] = x[M, K] @ dequant(w, s)^T (+ bias). Returns false without
// launching when the inputs are outside the kernel's contract (M > 8, dtype,
// layout, alignment), so the caller can fall back.
bool mxfp4_gemv_small(const torch::Tensor& x, const torch::Tensor& w,
                      const torch::Tensor& s,
                      const std::optional<torch::Tensor>& bias,
                      torch::Tensor& out, int64_t warps) {
  if (!x.is_cuda() || x.scalar_type() != at::kBFloat16 ||
      w.scalar_type() != at::kByte || s.scalar_type() != at::kByte ||
      out.scalar_type() != at::kBFloat16 || x.dim() != 2 || w.dim() != 2 ||
      s.dim() != 2 || x.stride(1) != 1 || w.stride(1) != 1 ||
      s.stride(1) != 1 || out.stride(1) != 1)
    return false;
  const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
  if (w.size(1) * 2 != K || s.size(0) != N || s.size(1) * 32 != K ||
      out.size(0) != M || out.size(1) != N)
    return false;
  const void* b = nullptr;
  if (bias.has_value()) {
    if (bias->scalar_type() != at::kBFloat16 || !bias->is_contiguous() ||
        bias->numel() != N)
      return false;
    b = bias->data_ptr();
  }
  const at::cuda::CUDAGuard guard(x.device());
  return mxfp4_gemv_small_launch(
      x.data_ptr(), w.data_ptr(), s.data_ptr(), b, out.data_ptr(), (int)M,
      (int)N, (int)K, x.stride(0), w.stride(0), s.stride(0), out.stride(0),
      (int)warps, at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mxfp4_gemv_small", &mxfp4_gemv_small, py::arg("x"), py::arg("w"),
        py::arg("s"), py::arg("bias"), py::arg("out"), py::arg("warps") = 8,
        "Small-M (<= 8) MXFP4 x BF16 GEMM, full-K strip per CTA (sm_120a)");
}
