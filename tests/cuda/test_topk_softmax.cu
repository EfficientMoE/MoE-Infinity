#include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <iomanip>

#include "kernel/ops.h"

// PyTorch native implementation for comparison - same interface as custom
// kernel
void torch_topk_softmax(torch::Tensor& topk_weights,
                        torch::Tensor& topk_indices,
                        torch::Tensor& token_expert_indices,
                        torch::Tensor& gating_output) {
  // Convert bf16 to float32 if needed for computation
  torch::Tensor input_for_compute = gating_output;
  if (gating_output.dtype() == torch::kBFloat16) {
    input_for_compute = gating_output.to(torch::kFloat32);
  }

  // Apply softmax
  torch::Tensor softmax_output = torch::softmax(input_for_compute, -1);

  // Get topk values and indices
  auto topk_result = torch::topk(softmax_output, topk_weights.size(-1), -1);
  torch::Tensor temp_weights = std::get<0>(topk_result);
  torch::Tensor temp_indices = std::get<1>(topk_result);

  // Copy results to pre-allocated tensors
  topk_weights.copy_(temp_weights);
  topk_indices.copy_(temp_indices.to(torch::kUInt32));

  // Create token_expert_indices efficiently using PyTorch operations
  int num_tokens = gating_output.size(0);
  int topk = topk_weights.size(-1);

  // Create base indices for tokens: [0, 1, 2, ..., num_tokens-1]
  torch::Tensor token_ids =
      torch::arange(num_tokens, torch::TensorOptions()
                                    .dtype(torch::kInt32)
                                    .device(gating_output.device()));

  // Create k indices: [0, 1, 2, ..., topk-1]
  torch::Tensor k_ids =
      torch::arange(topk, torch::TensorOptions()
                              .dtype(torch::kInt32)
                              .device(gating_output.device()));

  // Broadcast and compute: k_ids[:, None] * num_tokens + token_ids[None, :]
  // This creates the pattern: j * num_tokens + i for all (i,j) combinations
  torch::Tensor k_offset = k_ids.unsqueeze(1) * num_tokens;  // [topk, 1]
  torch::Tensor token_base = token_ids.unsqueeze(0);         // [1, num_tokens]

  // Final result: [topk, num_tokens] then transpose to [num_tokens, topk]
  token_expert_indices.copy_((k_offset + token_base).transpose(0, 1));
}

class Timer {
 private:
  std::chrono::high_resolution_clock::time_point start_time;

 public:
  void start() { start_time = std::chrono::high_resolution_clock::now(); }

  double elapsed_ms() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);
    return duration.count() / 1000.0;  // Convert to milliseconds
  }
};

void benchmark_kernel_vs_torch(int num_tokens, int num_experts, int topk,
                               int num_iterations = 100) {
  std::cout << "\n=== Benchmark: " << num_tokens << " tokens, " << num_experts
            << " experts, topk=" << topk << " ===" << std::endl;

  // Setup CUDA device
  torch::Device device(torch::kCUDA);
  const at::cuda::OptionalCUDAGuard device_guard(device);

  // Create input tensor
  torch::Tensor gating_output = torch::randn(
      {num_tokens, num_experts},
      torch::TensorOptions().dtype(torch::kFloat32).device(device));

  // Pre-allocate output tensors for custom kernel
  torch::Tensor custom_topk_weights = torch::zeros(
      {num_tokens, topk},
      torch::TensorOptions().dtype(torch::kFloat32).device(device));
  torch::Tensor custom_topk_indices =
      torch::zeros({num_tokens, topk},
                   torch::TensorOptions().dtype(torch::kUInt32).device(device));
  torch::Tensor custom_token_expert_indices =
      torch::zeros({num_tokens, topk},
                   torch::TensorOptions().dtype(torch::kInt32).device(device));

  // Pre-allocate output tensors for PyTorch native (same tensors used across
  // iterations)
  torch::Tensor torch_topk_weights = torch::zeros(
      {num_tokens, topk},
      torch::TensorOptions().dtype(torch::kFloat32).device(device));
  torch::Tensor torch_topk_indices =
      torch::zeros({num_tokens, topk},
                   torch::TensorOptions().dtype(torch::kUInt32).device(device));
  torch::Tensor torch_token_expert_indices =
      torch::zeros({num_tokens, topk},
                   torch::TensorOptions().dtype(torch::kInt32).device(device));

  Timer timer;
  std::vector<double> custom_times, torch_times;

  // Warmup runs
  std::cout << "Warming up..." << std::endl;
  for (int i = 0; i < 10; i++) {
    // Warmup custom kernel
    topk_softmax(custom_topk_weights, custom_topk_indices,
                 custom_token_expert_indices, gating_output);

    // Warmup PyTorch
    torch_topk_softmax(torch_topk_weights, torch_topk_indices,
                       torch_token_expert_indices, gating_output);

    torch::cuda::synchronize();
  }

  std::cout << "Running benchmark..." << std::endl;

  // Benchmark custom kernel
  for (int i = 0; i < num_iterations; i++) {
    timer.start();
    topk_softmax(custom_topk_weights, custom_topk_indices,
                 custom_token_expert_indices, gating_output);
    torch::cuda::synchronize();
    custom_times.push_back(timer.elapsed_ms());
  }

  // Benchmark PyTorch native
  for (int i = 0; i < num_iterations; i++) {
    timer.start();
    torch_topk_softmax(torch_topk_weights, torch_topk_indices,
                       torch_token_expert_indices, gating_output);
    torch::cuda::synchronize();
    torch_times.push_back(timer.elapsed_ms());
  }

  // Calculate statistics
  auto calc_stats = [](const std::vector<double>& times) {
    double sum = 0, min_time = times[0], max_time = times[0];
    for (double t : times) {
      sum += t;
      min_time = std::min(min_time, t);
      max_time = std::max(max_time, t);
    }
    double mean = sum / times.size();

    double variance = 0;
    for (double t : times) {
      variance += (t - mean) * (t - mean);
    }
    double stddev = std::sqrt(variance / times.size());

    return std::make_tuple(mean, min_time, max_time, stddev);
  };

  auto [custom_mean, custom_min, custom_max, custom_std] =
      calc_stats(custom_times);
  auto [torch_mean, torch_min, torch_max, torch_std] = calc_stats(torch_times);

  // Print results
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "\nCustom Kernel Results:" << std::endl;
  std::cout << "  Mean: " << custom_mean << " ms" << std::endl;
  std::cout << "  Min:  " << custom_min << " ms" << std::endl;
  std::cout << "  Max:  " << custom_max << " ms" << std::endl;
  std::cout << "  Std:  " << custom_std << " ms" << std::endl;

  std::cout << "\nPyTorch Native Results:" << std::endl;
  std::cout << "  Mean: " << torch_mean << " ms" << std::endl;
  std::cout << "  Min:  " << torch_min << " ms" << std::endl;
  std::cout << "  Max:  " << torch_max << " ms" << std::endl;
  std::cout << "  Std:  " << torch_std << " ms" << std::endl;

  double speedup = torch_mean / custom_mean;
  std::cout << "\nSpeedup: " << speedup << "x ";
  if (speedup > 1.0) {
    std::cout << "(Custom kernel is faster)" << std::endl;
  } else {
    std::cout << "(PyTorch native is faster)" << std::endl;
  }

  // Verify correctness (optional)
  std::cout << "\nVerifying correctness..." << std::endl;

  // Create temporary tensors for verification
  torch::Tensor verify_weights = torch::zeros_like(custom_topk_weights);
  torch::Tensor verify_indices = torch::zeros_like(custom_topk_indices);
  torch::Tensor verify_token_indices =
      torch::zeros_like(custom_token_expert_indices);

  torch_topk_softmax(verify_weights, verify_indices, verify_token_indices,
                     gating_output);

  // Compare a few values
  bool close = torch::allclose(custom_topk_weights, verify_weights, 1e-3, 1e-3);
  std::cout << "Results match PyTorch: " << (close ? "YES" : "NO") << std::endl;

  if (!close) {
    std::cout << "Max difference: "
              << torch::max(torch::abs(custom_topk_weights - verify_weights))
                     .item<float>()
              << std::endl;
  }
}

int main() {
  std::cout << "MoE Kernel vs PyTorch Native Speed Comparison" << std::endl;
  std::cout << "=============================================" << std::endl;

  if (!torch::cuda::is_available()) {
    std::cerr << "CUDA is not available!" << std::endl;
    return -1;
  }

  // std::cout << "CUDA Device: " << torch::cuda::get_device_name(0) <<
  // std::endl;

  // Test different configurations
  std::vector<std::tuple<int, int, int>> test_configs = {
      {32, 128, 8},    {1024, 128, 8},  // Small: 1K tokens, 8 experts, top-2
      {4096, 128, 8},                   // Medium: 4K tokens, 16 experts, top-2
      {8192, 128, 8},                   // Large: 8K tokens, 32 experts, top-4
      {16384, 128, 8},  // Very large: 16K tokens, 64 experts, top-8
      {32768, 128, 8},  // Extreme: 32K tokens, 128 experts, top-2
  };

  for (auto [num_tokens, num_experts, topk] : test_configs) {
    try {
      benchmark_kernel_vs_torch(num_tokens, num_experts, topk, 50);
    } catch (const std::exception& e) {
      std::cerr << "Error in configuration (" << num_tokens << ", "
                << num_experts << ", " << topk << "): " << e.what()
                << std::endl;
    }
  }

  std::cout << "\nBenchmark completed!" << std::endl;
  return 0;
}
