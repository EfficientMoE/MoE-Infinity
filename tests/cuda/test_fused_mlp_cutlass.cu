// Benchmark: BF16 CUTLASS fused MoE MLP vs. PyTorch 5-op reference
//
// Measures wall-clock latency and numerical error for fused_moe_ffn_into()
// across representative MoE problem sizes.

#include <torch/torch.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "kernel/fused_moe_mlp.h"

// ---------------------------------------------------------------------------
// Timer — microseconds, same pattern as test_topk_softmax.cu
// ---------------------------------------------------------------------------
class Timer {
 private:
  std::chrono::high_resolution_clock::time_point start_time;

 public:
  void start() { start_time = std::chrono::high_resolution_clock::now(); }

  double elapsed_us() {
    auto end_time = std::chrono::high_resolution_clock::now();
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                              start_time)
            .count());
  }
};

// ---------------------------------------------------------------------------
// PyTorch 5-op reference (no internal allocations)
//   hidden    [M, H]
//   gate_proj [I, H]   up_proj  [I, H]   down_proj [H, I]
//   gate_out  [M, I]   up_out   [M, I]
//   gate_act  [M, I]   fused_tmp[M, I]   output    [M, H]
// ---------------------------------------------------------------------------
static void torch_fused_mlp(torch::Tensor& hidden, torch::Tensor& gate_proj,
                             torch::Tensor& up_proj, torch::Tensor& down_proj,
                             torch::Tensor& gate_out, torch::Tensor& up_out,
                             torch::Tensor& gate_act, torch::Tensor& fused_tmp,
                             torch::Tensor& output) {
  at::mm_out(gate_out, hidden, gate_proj.t());   // gate_out = hidden @ W_g^T
  at::mm_out(up_out, hidden, up_proj.t());        // up_out   = hidden @ W_u^T
  at::silu_out(gate_act, gate_out);               // gate_act = silu(gate_out)
  at::mul_out(fused_tmp, gate_act, up_out);       // fused    = gate_act * up_out
  at::mm_out(output, fused_tmp, down_proj.t());   // output   = fused @ W_d^T
}

// ---------------------------------------------------------------------------
// Report max/mean absolute error and max relative error
// ---------------------------------------------------------------------------
static void report_error(torch::Tensor& torch_out, torch::Tensor& cutlass_out) {
  auto t = torch_out.flatten().to(torch::kFloat32);
  auto c = cutlass_out.flatten().to(torch::kFloat32);

  auto diff = torch::abs(c - t);
  float max_abs = diff.max().item<float>();
  float mean_abs = diff.mean().item<float>();
  float max_rel = (diff / (torch::abs(t) + 1e-6f)).max().item<float>();
  bool pass = max_abs < 0.05f;

  std::cout << std::fixed << std::setprecision(4)
            << "Numerical check:  max_abs=" << max_abs
            << "  mean_abs=" << mean_abs << "  max_rel=" << max_rel << "  "
            << (pass ? "PASS" : "FAIL") << std::endl;
}

// ---------------------------------------------------------------------------
// Per-configuration benchmark
// ---------------------------------------------------------------------------
static void benchmark(int M, int H, int I, const std::string& name,
                      int warmup = 10, int iters = 50) {
  std::cout << "\n=== Config: " << name << " (M=" << M << " H=" << H
            << " I=" << I << ") ===" << std::endl;

  torch::Device device(torch::kCUDA);
  const at::cuda::OptionalCUDAGuard device_guard(device);

  auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  auto f32  = torch::TensorOptions().dtype(torch::kFloat32).device(device);

  // Scale weights by 1/sqrt(dim) so activations stay O(1) and don't overflow
  // BF16 max (~65504).  Without scaling, gate_out std≈sqrt(H)≈45, fused
  // std≈45², output std≈45²·sqrt(I)≈91K — well past BF16 range.
  auto mk_bf16 = [&](torch::Tensor t) { return t.to(torch::kBFloat16); };
  const float scaleH = 1.0f / std::sqrt(static_cast<float>(H));
  const float scaleI = 1.0f / std::sqrt(static_cast<float>(I));

  // Shared inputs
  torch::Tensor hidden    = mk_bf16(torch::randn({M, H}, f32));
  torch::Tensor gate_proj = mk_bf16(torch::randn({I, H}, f32) * scaleH);
  torch::Tensor up_proj   = mk_bf16(torch::randn({I, H}, f32) * scaleH);
  torch::Tensor down_proj = mk_bf16(torch::randn({H, I}, f32) * scaleI);

  // CUTLASS buffers
  torch::Tensor gate_buf = torch::zeros({M, I}, opts);
  torch::Tensor fused_buf = torch::zeros({M, I}, opts);
  torch::Tensor cutlass_out = torch::zeros({M, H}, opts);

  // Torch reference buffers
  torch::Tensor gate_out = torch::zeros({M, I}, opts);
  torch::Tensor up_out = torch::zeros({M, I}, opts);
  torch::Tensor gate_act = torch::zeros({M, I}, opts);
  torch::Tensor fused_tmp = torch::zeros({M, I}, opts);
  torch::Tensor torch_out = torch::zeros({M, H}, opts);

  // --- Correctness ---
  torch_fused_mlp(hidden, gate_proj, up_proj, down_proj, gate_out, up_out,
                  gate_act, fused_tmp, torch_out);
  fused_moe_ffn_into(hidden, gate_proj, up_proj, down_proj, gate_buf,
                     fused_buf, cutlass_out, nullptr);
  cudaDeviceSynchronize();
  report_error(torch_out, cutlass_out);

  // --- Warmup ---
  for (int i = 0; i < warmup; i++) {
    torch_fused_mlp(hidden, gate_proj, up_proj, down_proj, gate_out, up_out,
                    gate_act, fused_tmp, torch_out);
    fused_moe_ffn_into(hidden, gate_proj, up_proj, down_proj, gate_buf,
                       fused_buf, cutlass_out, nullptr);
  }
  cudaDeviceSynchronize();

  Timer timer;
  std::vector<double> torch_times, cutlass_times;
  torch_times.reserve(iters);
  cutlass_times.reserve(iters);

  // --- Benchmark Torch ---
  for (int i = 0; i < iters; i++) {
    cudaDeviceSynchronize();
    timer.start();
    torch_fused_mlp(hidden, gate_proj, up_proj, down_proj, gate_out, up_out,
                    gate_act, fused_tmp, torch_out);
    cudaDeviceSynchronize();
    torch_times.push_back(timer.elapsed_us());
  }

  // --- Benchmark CUTLASS ---
  for (int i = 0; i < iters; i++) {
    cudaDeviceSynchronize();
    timer.start();
    fused_moe_ffn_into(hidden, gate_proj, up_proj, down_proj, gate_buf,
                       fused_buf, cutlass_out, nullptr);
    cudaDeviceSynchronize();
    cutlass_times.push_back(timer.elapsed_us());
  }

  // --- Stats ---
  auto calc_stats = [](const std::vector<double>& times) {
    double sum = 0, mn = times[0], mx = times[0];
    for (double t : times) {
      sum += t;
      mn = std::min(mn, t);
      mx = std::max(mx, t);
    }
    double mean = sum / static_cast<double>(times.size());
    double var = 0;
    for (double t : times) var += (t - mean) * (t - mean);
    double stddev = std::sqrt(var / static_cast<double>(times.size()));
    return std::make_tuple(mean, mn, mx, stddev);
  };

  auto [t_mean, t_min, t_max, t_std] = calc_stats(torch_times);
  auto [c_mean, c_min, c_max, c_std] = calc_stats(cutlass_times);

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "Torch  : mean=" << t_mean << "µs  std=" << t_std
            << "  min=" << t_min << "  max=" << t_max << std::endl;
  std::cout << "CUTLASS: mean=" << c_mean << "µs  std=" << c_std
            << "  min=" << c_min << "  max=" << c_max << std::endl;
  std::cout << "Speedup: " << std::setprecision(2) << (t_mean / c_mean) << "x"
            << std::endl;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
  std::cout << "BF16 CUTLASS Fused MoE MLP vs. Torch-Native Benchmark"
            << std::endl;
  std::cout << "======================================================="
            << std::endl;

  if (!torch::cuda::is_available()) {
    std::cerr << "CUDA is not available!" << std::endl;
    return -1;
  }

  benchmark(1, 2048, 2048, "tiny");
  benchmark(32, 2048, 2048, "small");
  benchmark(128, 2048, 2048, "maxTok");
  benchmark(128, 4096, 4096, "mixtral");

  std::cout << "\nBenchmark completed." << std::endl;
  return 0;
}
