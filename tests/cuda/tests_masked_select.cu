#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/functional.h>

// Include your kernel headers here
#include "kernel/masked_select.h"

// // For testing purposes, we'll define the function signatures
// extern void fused_extract_expert_tokens_bf16(const __nv_bfloat16*
// hidden_states,
//                                              const bool* router_mask,
//                                              __nv_bfloat16* output,
//                                              int* output_count, int
//                                              num_tokens, int hidden_dim, int
//                                              expert_idx, int num_experts);

// extern void extract_expert_tokens_cutlass_v2(const __nv_bfloat16*
// hidden_states,
//                                              const bool* router_mask,
//                                              __nv_bfloat16* output,
//                                              int* output_count, int
//                                              num_tokens, int hidden_dim, int
//                                              expert_idx, int num_experts);

// extern void extract_expert_tokens_cub(const __nv_bfloat16* hidden_states,
//                                       const bool* router_mask,
//                                       __nv_bfloat16* output, int*
//                                       output_count, int num_tokens, int
//                                       hidden_dim, int expert_idx, int
//                                       num_experts);

// Utility class for timing
class CudaTimer {
  cudaEvent_t start, stop;

 public:
  CudaTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~CudaTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start() { cudaEventRecord(start); }

  float Stop() {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds;
  }
};

// Reference CPU implementation for correctness checking
void extract_expert_tokens_cpu(const std::vector<__nv_bfloat16>& hidden_states,
                               const std::vector<bool>& router_mask,
                               std::vector<__nv_bfloat16>& output,
                               int& output_count, int num_tokens,
                               int hidden_dim, int expert_idx,
                               int num_experts) {
  output_count = 0;

  // Count selected tokens
  for (int i = 0; i < num_tokens; i++) {
    if (router_mask[i * num_experts + expert_idx]) {
      output_count++;
    }
  }

  // Extract tokens
  int out_idx = 0;
  for (int i = 0; i < num_tokens; i++) {
    if (router_mask[i * num_experts + expert_idx]) {
      for (int j = 0; j < hidden_dim; j++) {
        output[out_idx * hidden_dim + j] = hidden_states[i * hidden_dim + j];
      }
      out_idx++;
    }
  }
}

// Function to generate random test data
void generate_test_data(std::vector<__nv_bfloat16>& hidden_states,
                        std::vector<bool>& router_mask, int num_tokens,
                        int hidden_dim, int num_experts,
                        float sparsity = 0.1f) {
  std::mt19937 gen(42);  // Fixed seed for reproducibility and faster generation
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  std::bernoulli_distribution mask_dis(sparsity);

  // Generate hidden states
  hidden_states.resize(num_tokens * hidden_dim);
  for (int i = 0; i < num_tokens * hidden_dim; i++) {
    hidden_states[i] = __float2bfloat16(dis(gen));
  }

  // Generate router mask with specified sparsity
  router_mask.resize(num_tokens * num_experts);
  for (int i = 0; i < num_tokens * num_experts; i++) {
    router_mask[i] = mask_dis(gen);
  }
}

// Function to check if two bf16 arrays are approximately equal
bool check_correctness(const std::vector<__nv_bfloat16>& ref,
                       const thrust::host_vector<__nv_bfloat16>& test,
                       int count, int hidden_dim, float tolerance = 1e-3f) {
  for (int i = 0; i < count * hidden_dim; i++) {
    float ref_val = __bfloat162float(ref[i]);
    float test_val = __bfloat162float(test[i]);

    if (std::abs(ref_val - test_val) > tolerance) {
      std::cout << "Mismatch at index " << i << ": ref=" << ref_val
                << " test=" << test_val << std::endl;
      return false;
    }
  }
  return true;
}

// Test configuration structure
struct TestConfig {
  int num_tokens;
  int hidden_dim;
  int num_experts;
  int expert_idx;
  float sparsity;
  const char* name;
};

// Main test function
void run_test(const TestConfig& config) {
  std::cout << "\n=== Testing configuration: " << config.name
            << " ===" << std::endl;
  std::cout << "Tokens: " << config.num_tokens
            << ", Hidden dim: " << config.hidden_dim
            << ", Experts: " << config.num_experts
            << ", Sparsity: " << config.sparsity << std::endl;

  // Generate test data
  std::vector<__nv_bfloat16> h_hidden_states;
  std::vector<bool> h_router_mask;
  generate_test_data(h_hidden_states, h_router_mask, config.num_tokens,
                     config.hidden_dim, config.num_experts, config.sparsity);

  // Compute reference result on using torch tensor
  std::vector<__nv_bfloat16> cpu_output(config.num_tokens * config.hidden_dim);
  int cpu_count = 0;
  extract_expert_tokens_cpu(h_hidden_states, h_router_mask, cpu_output,
                            cpu_count, config.num_tokens, config.hidden_dim,
                            config.expert_idx, config.num_experts);

  std::cout << "Selected tokens: " << cpu_count << " ("
            << (100.0f * cpu_count / config.num_tokens) << "%)" << std::endl;

  // Copy data to GPU
  thrust::device_vector<__nv_bfloat16> d_hidden_states = h_hidden_states;
  thrust::device_vector<bool> d_router_mask = h_router_mask;

  // Prepare output buffers
  thrust::device_vector<__nv_bfloat16> d_output_fused(config.num_tokens *
                                                      config.hidden_dim);
  thrust::device_vector<__nv_bfloat16> d_output_cutlass(config.num_tokens *
                                                        config.hidden_dim);
  thrust::device_vector<__nv_bfloat16> d_output_cub(config.num_tokens *
                                                    config.hidden_dim);

  thrust::device_vector<int> d_count_fused(1);
  thrust::device_vector<int> d_count_cutlass(1);
  thrust::device_vector<int> d_count_cub(1);

  CudaTimer timer;
  const int warmup_iters = 10;
  const int bench_iters = 100;

  // Test 1: Fused kernel
  std::cout << "\n1. Testing fused kernel..." << std::endl;

  // Warmup
  for (int i = 0; i < warmup_iters; i++) {
    thrust::fill(d_count_fused.begin(), d_count_fused.end(), 0);

    // Configure kernel launch
    const int threads = 256;
    const int warps_per_block = threads / 32;
    const int blocks =
        std::min(65535, (config.num_tokens + threads - 1) / threads);
    const int smem_size = sizeof(int) * (warps_per_block + 1);

    fused_extract_expert_tokens_bf16_simple<<<blocks, threads, smem_size>>>(
        thrust::raw_pointer_cast(d_hidden_states.data()),
        thrust::raw_pointer_cast(d_router_mask.data()),
        thrust::raw_pointer_cast(d_output_fused.data()),
        thrust::raw_pointer_cast(d_count_fused.data()), config.num_tokens,
        config.hidden_dim, config.expert_idx, config.num_experts);
  }
  cudaDeviceSynchronize();
  std::cout << "Warmup complete." << std::endl;

  // Benchmark
  timer.Start();
  for (int i = 0; i < bench_iters; i++) {
    thrust::fill(d_count_fused.begin(), d_count_fused.end(), 0);

    const int threads = 256;
    const int warps_per_block = threads / 32;
    const int blocks =
        std::min(65535, (config.num_tokens + threads - 1) / threads);
    const int smem_size = sizeof(int) * (warps_per_block + 1);

    fused_extract_expert_tokens_bf16_simple<<<blocks, threads, smem_size>>>(
        thrust::raw_pointer_cast(d_hidden_states.data()),
        thrust::raw_pointer_cast(d_router_mask.data()),
        thrust::raw_pointer_cast(d_output_fused.data()),
        thrust::raw_pointer_cast(d_count_fused.data()), config.num_tokens,
        config.hidden_dim, config.expert_idx, config.num_experts);
  }
  float fused_time = timer.Stop() / bench_iters;

  // Check correctness
  int fused_count = d_count_fused[0];
  thrust::host_vector<__nv_bfloat16> h_output_fused = d_output_fused;

  bool fused_correct = (fused_count == cpu_count) &&
                       check_correctness(cpu_output, h_output_fused, cpu_count,
                                         config.hidden_dim);

  std::cout << "  Time: " << std::fixed << std::setprecision(3) << fused_time
            << " ms"
            << "  Correctness: " << (fused_correct ? "PASSED" : "FAILED")
            << std::endl;

  // Test 1: pytorch call
  std::cout << "\n2. Testing PyTorch call..." << std::endl;
  torch::Tensor hidden_states_tensor =
      torch::from_blob(h_hidden_states.data(),
                       {config.num_tokens, config.hidden_dim}, torch::kBFloat16)
          .clone()
          .cuda();
  torch::Tensor router_mask_tensor =
      torch::from_blob(h_router_mask.data(),
                       {config.num_tokens, config.num_experts}, torch::kBool)
          .clone()
          .cuda();
  torch::Tensor output_tensor =
      torch::empty({cpu_count, config.hidden_dim}, torch::kBFloat16).cuda();
  // Warmup
  for (int i = 0; i < warmup_iters; i++) {
    auto token_mask =
        router_mask_tensor.index({torch::indexing::Slice(), config.expert_idx});
    output_tensor.index_put_(
        {torch::indexing::Slice(0, cpu_count), torch::indexing::Slice()},
        hidden_states_tensor.index({token_mask}));
  }

  // Benchmark
  timer.Start();
  for (int i = 0; i < bench_iters; i++) {
    auto token_mask =
        router_mask_tensor.index({torch::indexing::Slice(), config.expert_idx});
    output_tensor.index_put_(
        {torch::indexing::Slice(0, cpu_count), torch::indexing::Slice()},
        hidden_states_tensor.index({token_mask}));
  }
  float pytorch_time = timer.Stop() / bench_iters;
  std::cout << "  Time: " << std::fixed << std::setprecision(3) << pytorch_time
            << " ms" << std::endl;
  // Check correctness
  thrust::host_vector<__nv_bfloat16> h_output_pytorch =
      output_tensor.cpu().bfloat16();
  bool pytorch_correct =
      (cpu_count == h_output_pytorch.size() / config.hidden_dim) &&
      check_correctness(cpu_output, h_output_pytorch, cpu_count,
                        config.hidden_dim);
  std::cout << "  Correctness: " << (pytorch_correct ? "PASSED" : "FAILED")
            << std::endl;

  // Performance summary
  std::cout << "\nPerformance Summary:" << std::endl;
  std::cout << "  Fused kernel:   " << fused_time << " ms (1.00x)" << std::endl;
  std::cout << "  PyTorch call:  " << pytorch_time << " ms (" << std::fixed
            << std::setprecision(2) << fused_time / pytorch_time << "x)"
            << std::endl;

  // Calculate bandwidth utilization
  size_t bytes_read =
      config.num_tokens * config.hidden_dim * sizeof(__nv_bfloat16) +
      config.num_tokens * config.num_experts * sizeof(bool);
  size_t bytes_written = cpu_count * config.hidden_dim * sizeof(__nv_bfloat16);
  float bandwidth_gb =
      (bytes_read + bytes_written) / (1024.0f * 1024.0f * 1024.0f);

  std::cout << "\nBandwidth utilization:" << std::endl;
  std::cout << "  Fused:   " << std::fixed << std::setprecision(1)
            << bandwidth_gb / (fused_time / 1000.0f) << " GB/s" << std::endl;
  std::cout << "  PyTorch: " << std::fixed << std::setprecision(1)
            << bandwidth_gb / (pytorch_time / 1000.0f) << " GB/s" << std::endl;
}

int main(int argc, char** argv) {
  // Check CUDA device
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  std::cout << "Running on: " << prop.name << std::endl;
  std::cout << "Compute capability: " << prop.major << "." << prop.minor
            << std::endl;
  std::cout << "Memory bandwidth: "
            << prop.memoryBusWidth * prop.memoryClockRate * 2 / 8e6 << " GB/s"
            << std::endl;

  // Define test configurations
  std::vector<TestConfig> configs = {
      // Small tests
      {1024, 768, 8, 0, 0.1f, "Small (BERT-like)"},
      {1024, 768, 8, 0, 0.01f, "Small (sparse)"},
      {1024, 768, 8, 0, 0.5f, "Small (dense)"},

      // Medium tests
      {4096, 1024, 16, 0, 0.1f, "Medium"},
      {4096, 4096, 8, 0, 0.1f, "Medium (wide)"},

      // Large tests
      {16384, 1024, 32, 0, 0.1f, "Large"},
      {16384, 4096, 16, 0, 0.05f, "Large (LLM-like)"},

      // Stress tests
      {65536, 1024, 64, 0, 0.02f, "Stress test"},
      {32768, 8192, 8, 0, 0.125f, "Stress test (very wide)"},
  };

  // Run all tests
  for (const auto& config : configs) {
    try {
      run_test(config);
    } catch (const std::exception& e) {
      std::cerr << "Test failed with exception: " << e.what() << std::endl;
    }
  }

  std::cout << "\n=== All tests completed ===" << std::endl;

  return 0;
}
