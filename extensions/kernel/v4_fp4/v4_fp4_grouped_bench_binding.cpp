// Bench-only binding for v4_moe_forward_sm120_pro6000. Same launch contract
// as v4_fp4_binding.cpp::run_v4_moe, without the GPT-OSS objects, so a
// trial can relink this symbol alone.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "../hw_profile.h"

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

bool v4_moe_supported(int64_t device) {
  using moe_infinity::hw::Gate;
  using moe_infinity::hw::Supports;
  return Supports(Gate::kSm120, (int)device) &&
         Supports(Gate::kRtxPro6000, (int)device);
}

torch::Tensor v4_moe_forward_sm120_pro6000(torch::Tensor x,
                                           torch::Tensor topk_ids,
                                           torch::Tensor topk_w,
                                           torch::Tensor table, int64_t I,
                                           double swiglu_limit,
                                           int64_t act_block) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 &&
                  x.dim() == 2 && x.is_contiguous(),
              "x must be contiguous CUDA bf16 [T, H]");
  TORCH_CHECK(v4_moe_supported(x.device().index()),
              "v4 grouped bench binding: RTX PRO 6000 only");
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.dim() == 2 &&
                  topk_ids.is_contiguous() &&
                  (topk_ids.scalar_type() == torch::kInt64 ||
                   topk_ids.scalar_type() == torch::kInt32),
              "topk_ids");
  TORCH_CHECK(topk_w.is_cuda() && topk_w.scalar_type() == torch::kFloat32 &&
                  topk_w.sizes() == topk_ids.sizes() && topk_w.is_contiguous(),
              "topk_w");
  TORCH_CHECK(table.is_cuda() && table.scalar_type() == torch::kInt64 &&
                  table.dim() == 2 && table.size(1) == 6 &&
                  table.is_contiguous(),
              "table");
  const int64_t T = x.size(0), H = x.size(1), topk = topk_ids.size(1);
  const int64_t E = table.size(0);
  TORCH_CHECK(topk_ids.size(0) == T, "topk rows");
  TORCH_CHECK(H % 256 == 0 && I % 128 == 0 && H <= 8192 && I <= 8192, "shape");
  TORCH_CHECK(E >= 1 && E <= 1024 && topk <= 16, "E/topk");
  TORCH_CHECK(act_block == 32 || act_block == 128, "act_block");
  const int64_t n = T * topk;
  int bt = v4_moe_choose_bt(n, (int)E);
  const bool decode = v4_moe_use_decode(bt, (int)H, (int)I);
  if (decode && T == 1 && topk <= 8) bt = 1;
  const int bound = v4_moe_tile_bound(n, (int)E, bt);
  const int64_t n1 = std::max<int64_t>(n, 1);
  const int64_t n_counters = v4_moe_decode_counters(bound, (int)H);
  int64_t off = 0;
  auto carve = [&](int64_t bytes) {
    const int64_t o = off;
    off += (bytes + 255) / 256 * 256;
    return o;
  };
  const int64_t o_xq = carve(T * H), o_xs = carve(T * (H / 32));
  const int64_t o_sorted = carve(n1 * 4);
  const int64_t o_tiles = carve(((int64_t)bound * 3 + 1) * 4);
  const int64_t o_ctr = carve(decode ? n_counters * 4 : 0);
  const int64_t o_ext = carve(decode ? (int64_t)bound * 96 : 0);
  const int64_t o_hbuf = carve(decode ? n1 * I * 2 : 0);
  const int64_t o_hq = carve(decode ? 0 : n1 * I);
  const int64_t o_hs = carve(decode ? 0 : n1 * (I / 32));
  const int64_t o_ybuf = carve(n * H * 2);
  auto ws = torch::empty({off}, x.options().dtype(torch::kUInt8));
  uint8_t* base = ws.data_ptr<uint8_t>();
  auto out = torch::empty({T, H}, x.options().dtype(torch::kFloat32));
  if (T == 0) {
    out.zero_();
    return out;
  }
  auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
  int* tiles = reinterpret_cast<int*>(base + o_tiles);
  v4_moe_forward_sm120_pro6000_cuda(
      x.data_ptr(), topk_ids.data_ptr(),
      topk_ids.scalar_type() == torch::kInt64, topk_w.data_ptr<float>(),
      table.data_ptr<int64_t>(), (int)T, (int)topk, (int)E, (int)H, (int)I,
      (float)swiglu_limit, bt, base + o_xq, base + o_xs,
      reinterpret_cast<int*>(base + o_sorted), tiles + 1, tiles, base + o_hq,
      base + o_hs, (void*)(base + o_ybuf), out.data_ptr<float>(),
      decode ? base + o_hbuf : nullptr, reinterpret_cast<int*>(base + o_ctr),
      (int)n_counters, base + o_ext, (int)act_block, stream);
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("v4_moe_forward_sm120_pro6000", &v4_moe_forward_sm120_pro6000,
        py::arg("x"), py::arg("topk_ids"), py::arg("topk_w"), py::arg("table"),
        py::arg("inter"), py::arg("swiglu_limit"), py::arg("act_block") = 128);
}
