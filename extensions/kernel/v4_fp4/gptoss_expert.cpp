// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

#include "gptoss_expert.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <mutex>
#include <optional>

#include "mxfp4_gptoss_moe.h"

namespace {

bool is_mxfp4(const torch::Tensor& blocks, const torch::Tensor& scales,
              int64_t rows, int64_t k, const torch::Device& device) {
  return blocks.device() == device && scales.device() == device &&
         blocks.scalar_type() == torch::kUInt8 &&
         scales.scalar_type() == torch::kUInt8 && blocks.is_contiguous() &&
         scales.is_contiguous() && blocks.dim() == 2 && scales.dim() == 2 &&
         blocks.size(0) == rows && blocks.size(1) == k / 2 &&
         scales.size(0) == rows && scales.size(1) == k / 32;
}

bool is_vec(const torch::Tensor& t, int64_t n, const torch::Device& device) {
  return t.device() == device && t.numel() == n;
}

// Biases are BF16 in GPT-OSS checkpoints; anything else is converted.
torch::Tensor as_bf16(const torch::Tensor& t) {
  if (t.scalar_type() == torch::kBFloat16 && t.is_contiguous()) return t;
  return t.reshape({-1}).to(torch::kBFloat16).contiguous();
}

void ensure_bytes(torch::Tensor& t, size_t bytes, const torch::Device& d) {
  if (!t.defined() || static_cast<size_t>(t.numel()) < bytes || t.device() != d)
    t = torch::empty({static_cast<int64_t>(bytes)},
                     torch::TensorOptions().dtype(torch::kUInt8).device(d));
}

size_t align256(size_t n) { return (n + 255) / 256 * 256; }

struct ArenaLayout {
  size_t wd, act, partials, total;  // offsets; partials: gate_up then down
  size_t ws;                        // bytes of one partials region
};

ArenaLayout arena_layout(int64_t M, int64_t H, int64_t I) {
  ArenaLayout l{};
  const bool tiled = M > kMxfp4GemvMaxTokens;
  const bool dense = M >= kGptOssDenseDownMinTokens;
  l.ws = tiled ? align256(mxfp4_linear_workspace_bytes()) : 0;
  l.wd = 0;
  l.act = dense ? align256(static_cast<size_t>(H) * I * 2) : 0;
  l.partials = l.act + align256(static_cast<size_t>(M) * I * 2);
  l.total = l.partials + (dense ? 1 : 2) * l.ws;
  return l;
}

// Keeps [p, p + bytes) L2-resident for the kernels this stream launches
// while the object lives (persisting access-policy window), then restores
// the stream's previous window. The device's persisting set-aside is raised
// to its maximum on first use; the window is capped to it.
class L2Window {
 public:
  L2Window(cudaStream_t stream, void* p, size_t bytes) : stream_(stream) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev >= 64) return;
    static std::once_flag once[64];
    static size_t limit[64];
    std::call_once(once[dev], [dev] {
      int max_persist = 0, max_window = 0;
      cudaDeviceGetAttribute(&max_persist, cudaDevAttrMaxPersistingL2CacheSize,
                             dev);
      cudaDeviceGetAttribute(&max_window, cudaDevAttrMaxAccessPolicyWindowSize,
                             dev);
      size_t cur = 0;
      cudaDeviceGetLimit(&cur, cudaLimitPersistingL2CacheSize);
      if (cur < static_cast<size_t>(max_persist) &&
          cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, max_persist) ==
              cudaSuccess)
        cur = max_persist;
      limit[dev] = std::min(cur, static_cast<size_t>(max_window));
      cudaGetLastError();
    });
    const size_t n = std::min(bytes, limit[dev]);
    if (n == 0 ||
        cudaStreamGetAttribute(stream_, cudaStreamAttributeAccessPolicyWindow,
                               &old_) != cudaSuccess) {
      cudaGetLastError();
      return;
    }
    cudaStreamAttrValue v = {};
    v.accessPolicyWindow.base_ptr = p;
    v.accessPolicyWindow.num_bytes = n;
    v.accessPolicyWindow.hitRatio = 1.0f;
    v.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    v.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    active_ =
        cudaStreamSetAttribute(stream_, cudaStreamAttributeAccessPolicyWindow,
                               &v) == cudaSuccess;
    if (!active_) cudaGetLastError();
  }
  ~L2Window() {
    if (active_)
      cudaStreamSetAttribute(stream_, cudaStreamAttributeAccessPolicyWindow,
                             &old_);
  }
  L2Window(const L2Window&) = delete;
  L2Window& operator=(const L2Window&) = delete;

 private:
  cudaStream_t stream_;
  cudaStreamAttrValue old_ = {};
  bool active_ = false;
};

}  // namespace

size_t gptoss_expert_arena_bytes(int64_t M, int64_t H, int64_t I) {
  return arena_layout(M, H, I).total;
}

bool gptoss_mxfp4_expert_forward(
    const torch::Tensor& x, const torch::Tensor& gu_blocks,
    const torch::Tensor& gu_scales, const torch::Tensor& gu_bias,
    const torch::Tensor& d_blocks, const torch::Tensor& d_scales,
    const torch::Tensor& d_bias, torch::Tensor& out,
    GptOssExpertScratch& scratch, cudaStream_t stream) {
  if (!x.is_cuda() || x.dim() != 2 || x.scalar_type() != torch::kBFloat16 ||
      !x.is_contiguous())
    return false;
  const auto device = x.device();
  const int64_t M = x.size(0);
  const int64_t H = x.size(1);
  const int64_t I = d_blocks.dim() == 2 ? d_blocks.size(1) * 2 : 0;
  if (M == 0) return true;
  if (I == 0 || H % 32 != 0 || I % 32 != 0 ||
      !is_mxfp4(gu_blocks, gu_scales, 2 * I, H, device) ||
      !is_mxfp4(d_blocks, d_scales, H, I, device) ||
      !is_vec(gu_bias, 2 * I, device) || !is_vec(d_bias, H, device))
    return false;
  if (out.device() != device || out.scalar_type() != torch::kBFloat16 ||
      !out.is_contiguous() || out.dim() != 2 || out.size(0) != M ||
      out.size(1) != H)
    return false;

  // The launchers query the current device; executor threads already run on
  // the expert's device, so only switch when needed.
  std::optional<c10::cuda::CUDAGuard> device_guard;
  if (c10::cuda::current_device() != device.index())
    device_guard.emplace(device);
  const auto gu_b = as_bf16(gu_bias);
  const auto d_b = as_bf16(d_bias);

  const ArenaLayout lay = arena_layout(M, H, I);
  ensure_bytes(scratch.arena, lay.total, device);
  if (!scratch.counters.defined() || scratch.counters.device() != device) {
    scratch.counters = torch::zeros(
        {2 * kMxfp4StreamKCounters},
        torch::TensorOptions().dtype(torch::kInt32).device(device));
  }
  uint8_t* arena = scratch.arena.data_ptr<uint8_t>();
  void* act = arena + lay.act;
  int* counters = scratch.counters.data_ptr<int>();
  // Stream-K partials only for the tiled kernels.
  const bool tiled = M > kMxfp4GemvMaxTokens;
  auto workspace = [&](int i) {
    return tiled ? Mxfp4Workspace{arena + lay.partials + i * lay.ws, lay.ws,
                                  counters + i * kMxfp4StreamKCounters,
                                  kMxfp4StreamKCounters}
                 : Mxfp4Workspace{};
  };
  const Mxfp4Workspace ws_gu = workspace(0);

  // Large M: Wd / act / partials would otherwise be partly written back to
  // DRAM before they are read. Small M: they stay in L2 anyway.
  const bool resident = M >= kGptOssL2ResidentMinTokens;
  std::optional<L2Window> window;
  if (resident) window.emplace(stream, arena, lay.total);

  cudaError_t err;
  if (M >= kGptOssDenseDownMinTokens) {
    // gate_up (MXFP4, fused SwiGLU) de-quantises Wd on the side; the down
    // projection is then one cuBLAS GEMM with the bias in its epilogue.
    auto wd = torch::from_blob(
        arena + lay.wd, {H, I},
        torch::TensorOptions().dtype(torch::kBFloat16).device(device));
    auto act_t = torch::from_blob(
        act, {M, I},
        torch::TensorOptions().dtype(torch::kBFloat16).device(device));
    const Mxfp4Side side{d_blocks.data_ptr(), d_scales.data_ptr(),
                         wd.data_ptr(), static_cast<int>(H * I / 32)};
    err = mxfp4_linear_cuda(x.data_ptr(), M, H, gu_blocks.data_ptr(),
                            gu_scales.data_ptr(), gu_b.data_ptr(), 2 * I, act,
                            true, false, &ws_gu, &side, -1, stream);
    if (err == cudaErrorNotSupported) return false;
    TORCH_CHECK(err == cudaSuccess,
                "GPT-OSS gate_up kernel: ", cudaGetErrorString(err));
    c10::cuda::CUDAStreamGuard stream_guard(
        c10::cuda::getStreamFromExternal(stream, device.index()));
    at::addmm_out(out, d_b, act_t, wd.t());
  } else {
    const Mxfp4Workspace ws_d = workspace(1);
    err = gptoss_mxfp4_expert_cuda(
        x.data_ptr(), M, H, I, gu_blocks.data_ptr(), gu_scales.data_ptr(),
        gu_b.data_ptr(), d_blocks.data_ptr(), d_scales.data_ptr(),
        d_b.data_ptr(), act, out.data_ptr(), &ws_gu, &ws_d, stream);
    if (err == cudaErrorNotSupported) return false;
    TORCH_CHECK(err == cudaSuccess,
                "GPT-OSS MXFP4 expert kernels: ", cudaGetErrorString(err));
  }
  if (resident) {
    // Wd and act are dead now. The stream-K partials are left in place: the
    // next call rewrites the same lines, and discarding them inside the
    // kernels cost 2-6 % at M = 288-1152.
    err = mxfp4_l2_discard_cuda(arena, lay.partials, stream);
    TORCH_CHECK(err == cudaSuccess,
                "GPT-OSS L2 discard: ", cudaGetErrorString(err));
  }
  return true;
}
