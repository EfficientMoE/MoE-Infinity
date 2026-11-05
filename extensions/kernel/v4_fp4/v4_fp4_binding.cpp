// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/stl.h>
#include <map>
#include <string>

#include "../hw_profile.h"
#include "gptoss_expert.h"

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

void v4_moe_forward_sm120_pro6000_cuda(
    const void* x, const void* topk_ids, bool ids_int64, const float* topk_w,
    const int64_t* table, int T, int topk, int E, int H, int I,
    float swiglu_limit, int bt, uint8_t* xq, uint8_t* xs, int* sorted,
    void* tiles, int* num_tiles, uint8_t* hq, uint8_t* hs, void* ybuf,
    float* out, void* hbuf, int* counters, int n_counters, void* tile_ext,
    int act_block, cudaStream_t stream);
int v4_moe_choose_bt(long long n, int E);
bool v4_moe_use_decode(int bt, int H, int I);
int v4_moe_decode_counters(int bound, int H);
int v4_moe_tile_bound(long long n, int E, int bt);

namespace {

// The grouped MoE kernels (v4_fp4_moe_sm120_pro6000.cu) use sm_120a
// block-scaled MMA and are sized for the RTX PRO 6000 (99 KiB shared memory
// per block, 188-CTA persistent decode grid, 128 MiB L2 prefetch budget).
// Other devices use Path B, v4_expert_forward (BF16 dequant + cuBLAS).
bool v4_moe_supported(int64_t device) {
  using moe_infinity::hw::Gate;
  using moe_infinity::hw::Supports;
  return Supports(Gate::kSm120, (int)device) &&
         Supports(Gate::kRtxPro6000, (int)device);
}

// Runs the native W4A8 block-scaled MoE. Returns the per-slot BF16 expert
// outputs [T * topk, H] and, if `combine`, the FP32 per-token sum [T, H].
std::pair<torch::Tensor, torch::Tensor> run_v4_moe(
    const torch::Tensor& x, const torch::Tensor& topk_ids,
    const torch::Tensor& topk_w, const torch::Tensor& table, int64_t I,
    double swiglu_limit, bool combine, int64_t act_block) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 &&
                  x.dim() == 2 && x.is_contiguous(),
              "x must be a contiguous CUDA bf16 [T, H] tensor");
  TORCH_CHECK(v4_moe_supported(x.device().index()),
              "v4_moe_*_sm120_pro6000 runs only on the RTX PRO 6000 "
              "(sm_120, 188 SMs, 128 MiB L2, 99 KiB shared memory); use "
              "v4_expert_forward (Path B) on this device");
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.dim() == 2 &&
                  topk_ids.is_contiguous() &&
                  (topk_ids.scalar_type() == torch::kInt64 ||
                   topk_ids.scalar_type() == torch::kInt32),
              "topk_ids must be a contiguous CUDA int32/int64 [T, k] tensor");
  TORCH_CHECK(topk_w.is_cuda() && topk_w.scalar_type() == torch::kFloat32 &&
                  topk_w.sizes() == topk_ids.sizes() && topk_w.is_contiguous(),
              "topk_w must be a contiguous CUDA float32 tensor like topk_ids");
  TORCH_CHECK(table.is_cuda() && table.scalar_type() == torch::kInt64 &&
                  table.dim() == 2 && table.size(1) == 6 &&
                  table.is_contiguous(),
              "table must be a contiguous CUDA int64 [E, 6] pointer table");
  const int64_t T = x.size(0), H = x.size(1), topk = topk_ids.size(1);
  const int64_t E = table.size(0);
  TORCH_CHECK(topk_ids.size(0) == T, "topk_ids rows must match x");
  TORCH_CHECK(H % 256 == 0 && I % 128 == 0 && H <= 8192 && I <= 8192,
              "v4_moe requires hidden % 256 == 0, inter % 128 == 0, both <= "
              "8192");
  TORCH_CHECK(E >= 1 && E <= 1024 && topk <= 16, "E <= 1024, topk <= 16");
  TORCH_CHECK(act_block == 32 || act_block == 128,
              "act_block (activation quantisation group) must be 32 or 128");
  const int64_t n = T * topk;
  int bt = v4_moe_choose_bt(n, (int)E);
  const bool decode = v4_moe_use_decode(bt, (int)H, (int)I);
  // one token: tiles of one routed slot each (decode kernel's batch-1 path)
  if (decode && T == 1 && topk <= 8) bt = 1;
  const int bound = v4_moe_tile_bound(n, (int)E, bt);
  const int64_t n1 = std::max<int64_t>(n, 1);
  const int64_t n_counters = v4_moe_decode_counters(bound, (int)H);
  // One scratch allocation: the host time of every allocation before the
  // first launch shows up as GPU idle time in a decode layer.
  int64_t off = 0;
  auto carve = [&](int64_t bytes) {
    const int64_t o = off;
    off += (bytes + 255) / 256 * 256;
    return o;
  };
  // activation scales: one ue8m0 byte per 32 elements
  const int64_t o_xq = carve(T * H), o_xs = carve(T * (H / 32));
  const int64_t o_sorted = carve(n1 * 4);
  const int64_t o_tiles = carve(((int64_t)bound * 3 + 1) * 4);
  const int64_t o_ctr = carve(decode ? n_counters * 4 : 0);
  const int64_t o_ext = carve(decode ? (int64_t)bound * 96 : 0);
  const int64_t o_hbuf = carve(decode ? n1 * I * 2 : 0);
  const int64_t o_hq = carve(decode ? 0 : n1 * I);
  const int64_t o_hs = carve(decode ? 0 : n1 * (I / 32));
  // ybuf is the result of the single-expert call; otherwise scratch too
  const int64_t o_ybuf = carve(combine ? n * H * 2 : 0);
  auto ws = torch::empty({off}, x.options().dtype(torch::kUInt8));
  uint8_t* base = ws.data_ptr<uint8_t>();
  torch::Tensor ybuf, out;
  if (combine)
    out = torch::empty({T, H}, x.options().dtype(torch::kFloat32));
  else
    ybuf = torch::empty({n, H}, x.options());
  if (T == 0) {
    if (combine) out.zero_();
    return {ybuf, out};
  }
  auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
  int* tiles = reinterpret_cast<int*>(base + o_tiles);
  v4_moe_forward_sm120_pro6000_cuda(
      x.data_ptr(), topk_ids.data_ptr(),
      topk_ids.scalar_type() == torch::kInt64, topk_w.data_ptr<float>(),
      table.data_ptr<int64_t>(), (int)T, (int)topk, (int)E, (int)H, (int)I,
      (float)swiglu_limit, bt, base + o_xq, base + o_xs,
      reinterpret_cast<int*>(base + o_sorted), tiles + 1, tiles, base + o_hq,
      base + o_hs, combine ? (void*)(base + o_ybuf) : ybuf.data_ptr(),
      combine ? out.data_ptr<float>() : nullptr,
      decode ? base + o_hbuf : nullptr, reinterpret_cast<int*>(base + o_ctr),
      (int)n_counters, base + o_ext, (int)act_block, stream);
  return {ybuf, out};
}

}  // namespace

// Grouped routed-expert MoE layer (native W4A8, block-scaled FP4 MMA).
//   x [T, H] bf16; topk_ids [T, k] int32/int64 (ids outside [0, E) skipped);
//   topk_w [T, k] float32 routing weights (applied before w2, as upstream);
//   table [E, 6] int64 device pointers per expert: w1, s1, w3, s3, w2, s2
//   (packed e2m1 [N, K/2] and ue8m0 [N, K/32], checkpoint layout).
//   act_block: elements per FP8 activation scale, the official
//   fp8_block_size (128 for V4-Flash, 32 for V4.1-Flash).
// Returns FP32 [T, H] = sum over the token's experts in ascending id order.
torch::Tensor v4_moe_forward_sm120_pro6000(torch::Tensor x,
                                           torch::Tensor topk_ids,
                                           torch::Tensor topk_w,
                                           torch::Tensor table, int64_t inter,
                                           double swiglu_limit,
                                           int64_t act_block) {
  return run_v4_moe(x, topk_ids, topk_w, table, inter, swiglu_limit, true,
                    act_block)
      .second;
}

// Single-expert forward on the native path (same contract as
// v4_expert_forward, routing weight 1). Returns BF16 [M, H].
torch::Tensor v4_expert_forward_fp4_sm120_pro6000(
    torch::Tensor x, torch::Tensor w1, torch::Tensor s1, torch::Tensor w2,
    torch::Tensor s2, torch::Tensor w3, torch::Tensor s3, double swiglu_limit,
    int64_t act_block) {
  for (auto* t : {&w1, &s1, &w2, &s2, &w3, &s3})
    TORCH_CHECK(t->is_cuda() && t->is_contiguous(),
                "expert tensors must be contiguous CUDA tensors");
  const int64_t M = x.size(0), I = w1.size(0);
  auto cpu =
      torch::empty({1, 6}, torch::dtype(torch::kInt64).pinned_memory(true));
  auto* p = cpu.data_ptr<int64_t>();
  p[0] = (int64_t)w1.data_ptr();
  p[1] = (int64_t)s1.data_ptr();
  p[2] = (int64_t)w3.data_ptr();
  p[3] = (int64_t)s3.data_ptr();
  p[4] = (int64_t)w2.data_ptr();
  p[5] = (int64_t)s2.data_ptr();
  auto table = cpu.to(x.device(), /*non_blocking=*/true);
  auto ids = torch::zeros({M, 1}, x.options().dtype(torch::kInt32));
  auto w = torch::ones({M, 1}, x.options().dtype(torch::kFloat32));
  return run_v4_moe(x.contiguous(), ids, w, table, I, swiglu_limit, false,
                    act_block)
      .first;
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

// GPT-OSS MXFP4 expert on the stored blocks/scales (the offloaded-expert
// path of MoEMLP): returns SwiGLU-OAI(x . Wgu^T + b_gu) . Wd^T + b_d.
// RTX PRO 6000 only (RTX_PRO_6000 gate); raises on other devices.
torch::Tensor gptoss_mxfp4_expert_pro6000(
    torch::Tensor x, torch::Tensor gu_blocks, torch::Tensor gu_scales,
    torch::Tensor gu_bias, torch::Tensor d_blocks, torch::Tensor d_scales,
    torch::Tensor d_bias) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2, "x must be a 2D CUDA tensor");
  auto xb = x.to(torch::kBFloat16).contiguous();
  auto out = torch::empty_like(xb);
  GptOssExpertScratch scratch;
  auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
  TORCH_CHECK(gptoss_mxfp4_expert_forward_pro6000(xb, gu_blocks, gu_scales,
                                                  gu_bias, d_blocks, d_scales,
                                                  d_bias, out, scratch, stream),
              "gptoss_mxfp4_expert_pro6000: needs an RTX PRO 6000 "
              "(MOE_INFINITY_UNVALIDATED_HW=1 overrides) and supported "
              "shapes/dtypes/alignment");
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp4_dequant", &fp4_dequant, "FP4 E2M1 packed -> BF16 dequant");
  m.def("mxfp4_dequant", &mxfp4_dequant, "MXFP4 uint8 blocks/scales to BF16");
  m.def("gptoss_mxfp4_expert_pro6000", &gptoss_mxfp4_expert_pro6000,
        "GPT-OSS MXFP4 expert forward (fused W4A16 GEMMs)");
  m.def("v4_expert_forward", &v4_expert_forward,
        "V4 FP4 routed-expert SwiGLU forward (BF16 dequant reference)");
  m.def("v4_moe_forward_sm120_pro6000", &v4_moe_forward_sm120_pro6000,
        py::arg("x"), py::arg("topk_ids"), py::arg("topk_w"), py::arg("table"),
        py::arg("inter"), py::arg("swiglu_limit"), py::arg("act_block") = 128,
        "V4 grouped routed-expert MoE, native W4A8 block-scaled FP4 MMA "
        "(sm_120a); returns FP32 [T, H]");
  m.def("v4_expert_forward_fp4_sm120_pro6000",
        &v4_expert_forward_fp4_sm120_pro6000, py::arg("x"), py::arg("w1"),
        py::arg("s1"), py::arg("w2"), py::arg("s2"), py::arg("w3"),
        py::arg("s3"), py::arg("swiglu_limit"), py::arg("act_block") = 128,
        "V4 single routed expert on the native W4A8 block-scaled FP4 path");
  m.def("v4_moe_supported", &v4_moe_supported, py::arg("device"),
        "True if the *_sm120_pro6000 grouped MoE kernels may run on this "
        "device (SM120 and RTX_PRO_6000 hardware gates)");
  m.def("fp8_dequant_blockwise", &fp8_dequant_blockwise,
        "FP8 E4M3 block-scale (128x128) weight -> BF16 dequant (GLM-5.2-FP8)");
  m.def("set_scales", &set_scales,
        "Store fp8 block scales for dequant-on-copy (no-op stub; full "
        "integration deferred)");
}
