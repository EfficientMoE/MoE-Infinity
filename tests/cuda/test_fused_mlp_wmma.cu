#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cmath>
#include <chrono>

using namespace nvcuda;

// Device SiLU activation for bfloat16
__device__ __forceinline__ __nv_bfloat16 silu(__nv_bfloat16 x) {
  float x_f = __bfloat162float(x);
  float result = x_f / (1.0f + expf(-x_f));
  return __float2bfloat16(result);
}

// WMMA activation function for float accumulator, convert to bfloat16 precision
template <typename fragment_t>
__device__ __forceinline__ void warp_silu_activation_bf16(
    const fragment_t& frag, fragment_t& result) {
#pragma unroll
  for (int t = 0; t < result.num_elements; t++) {
    // Convert to bfloat16 precision, apply SiLU, then back to float for
    // accumulator
    __nv_bfloat16 bf16_val = __float2bfloat16(frag.x[t]);
    __nv_bfloat16 silu_result = silu(bf16_val);
    result.x[t] = __bfloat162float(silu_result);
  }
}

// Pipelined WMMA kernel for bfloat16 precision
__global__ void wmma_silu_add_kernel_bf16_pipelined(
    const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C, int M,
    int N, int K, int ldx, int lda, int ldb, int ldc) {
  const int WMMA_M = 16;
  const int WMMA_N = 16;
  const int WMMA_K = 16;

  // Calculate warp indices within the block
  int warp_id = threadIdx.x / 32;
  int warps_per_block_x = blockDim.x / 32;
  int warps_per_block_y = blockDim.y;

  int warp_row = warp_id % warps_per_block_x;
  int warp_col = warp_id / warps_per_block_x;

  // Global warp coordinates
  int warpM = blockIdx.x * warps_per_block_x + warp_row;
  int warpN = blockIdx.y * warps_per_block_y + warp_col;

  if (warpM >= (M + WMMA_M - 1) / WMMA_M ||
      warpN >= (N + WMMA_N - 1) / WMMA_N) {
    return;
  }

  // Double-buffered fragments for pipelining
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16,
                 wmma::row_major>
      frag_x[2];
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16,
                 wmma::col_major>
      frag_a[2];
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16,
                 wmma::col_major>
      frag_b[2];

  // Accumulator fragments (float precision for computation)
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_xa;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_xb;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_silu_xa;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_result;

  // Shared memory for manual store using float (then convert during global
  // store)
  extern __shared__ float smem_float[];
  float* warp_smem = smem_float + warp_id * WMMA_M * WMMA_N;

  // Initialize accumulators
  wmma::fill_fragment(frag_xa, 0.0f);
  wmma::fill_fragment(frag_xb, 0.0f);

  int row = warpM * WMMA_M;
  int col = warpN * WMMA_N;

  // Pipeline prologue - load first iteration
  int buffer_idx = 0;
  if (row < M && col < N && K > 0) {
    // Load first tiles
    wmma::load_matrix_sync(frag_x[buffer_idx], X + row * ldx, ldx);
    wmma::load_matrix_sync(frag_a[buffer_idx], A + col, lda);
    wmma::load_matrix_sync(frag_b[buffer_idx], B + col, ldb);
  }

  // Pipelined main loop
  for (int i = 0; i < K; i += WMMA_K) {
    int next_buffer_idx = 1 - buffer_idx;

    // Prefetch next iteration (if not last)
    if (i + WMMA_K < K && row < M && col < N) {
      wmma::load_matrix_sync(frag_x[next_buffer_idx],
                             X + row * ldx + (i + WMMA_K), ldx);
      wmma::load_matrix_sync(frag_a[next_buffer_idx],
                             A + (i + WMMA_K) * lda + col, lda);
      wmma::load_matrix_sync(frag_b[next_buffer_idx],
                             B + (i + WMMA_K) * ldb + col, ldb);
    }

    // Compute with current buffers
    if (row < M && col < N && i < K) {
      wmma::mma_sync(frag_xa, frag_x[buffer_idx], frag_a[buffer_idx], frag_xa);
      wmma::mma_sync(frag_xb, frag_x[buffer_idx], frag_b[buffer_idx], frag_xb);

      // Convert accumulator results back to bfloat16 precision after each MM
      // operation
#pragma unroll
      for (int t = 0; t < frag_xa.num_elements; t++) {
        frag_xa.x[t] = __bfloat162float(__float2bfloat16(frag_xa.x[t]));
        frag_xb.x[t] = __bfloat162float(__float2bfloat16(frag_xb.x[t]));
      }
    }

    // Swap buffers
    buffer_idx = next_buffer_idx;
  }

  // Apply SiLU activation to X*A result (with bfloat16 precision behavior)
  warp_silu_activation_bf16(frag_xa, frag_silu_xa);

  // Add silu(X*A) + X*B (convert to bfloat16 precision during addition)
#pragma unroll
  for (int t = 0; t < frag_result.num_elements; t++) {
    // Convert both operands to bfloat16, perform addition, convert back to
    // float
    __nv_bfloat16 silu_bf16 = __float2bfloat16(frag_silu_xa.x[t]);
    __nv_bfloat16 xb_bf16 = __float2bfloat16(frag_xb.x[t]);
    __nv_bfloat16 result_bf16 = __hadd(silu_bf16, xb_bf16);
    frag_result.x[t] = __bfloat162float(result_bf16);
  }

  // Use WMMA store to shared memory with float, then manually copy to global
  // with conversion
  if (row < M && col < N) {
    // Store float result to shared memory using WMMA (this works!)
    wmma::store_matrix_sync(warp_smem, frag_result, WMMA_N,
                            wmma::mem_row_major);

    // Synchronize warp
    __syncwarp();

    // Now cooperatively copy from shared memory to global memory with bfloat16
    // conversion
    int lane_id = threadIdx.x % 32;

    // Each thread handles 8 elements (256 total / 32 threads = 8)
    for (int i = 0; i < 8; i++) {
      int elem_idx = lane_id * 8 + i;
      if (elem_idx < WMMA_M * WMMA_N) {
        int local_row = elem_idx / WMMA_N;
        int local_col = elem_idx % WMMA_N;
        int global_row = row + local_row;
        int global_col = col + local_col;

        if (global_row < M && global_col < N) {
          // Convert float from shared memory to bfloat16 for global memory
          float val = warp_smem[local_row * WMMA_N + local_col];
          C[global_row * ldc + global_col] = __float2bfloat16(val);
        }
      }
    }
  }
}

// Host function for PyTorch interface (bfloat16 only)
void wmma_silu_add_cuda(torch::Tensor X, torch::Tensor A, torch::Tensor B,
                        torch::Tensor C) {
  // Check inputs
  TORCH_CHECK(X.device().is_cuda(), "X must be a CUDA tensor");
  TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
  TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
  TORCH_CHECK(C.device().is_cuda(), "C must be a CUDA tensor");
  TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
  TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
  TORCH_CHECK(C.is_contiguous(), "C must be contiguous");
  TORCH_CHECK(X.dtype() == torch::kBFloat16, "X must be bfloat16");
  TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be bfloat16");
  TORCH_CHECK(B.dtype() == torch::kBFloat16, "B must be bfloat16");
  TORCH_CHECK(C.dtype() == torch::kBFloat16, "C must be bfloat16");

  auto M = X.size(0);
  auto K = X.size(1);
  auto N = A.size(1);

  TORCH_CHECK(A.size(0) == K, "A.size(0) must equal X.size(1)");
  TORCH_CHECK(B.size(0) == K, "B.size(0) must equal X.size(1)");
  TORCH_CHECK(B.size(1) == N, "B.size(1) must equal A.size(1)");
  TORCH_CHECK(C.size(0) == M, "C.size(0) must equal X.size(0)");
  TORCH_CHECK(C.size(1) == N, "C.size(1) must equal A.size(1)");

  // Calculate grid and block dimensions with multiple warps per block
  const int WMMA_M = 16;
  const int WMMA_N = 16;

  // Use 4x2 warps per block (8 warps total, 256 threads)
  const int WARPS_PER_BLOCK_X = 4;
  const int WARPS_PER_BLOCK_Y = 2;
  const int THREADS_PER_BLOCK = WARPS_PER_BLOCK_X * WARPS_PER_BLOCK_Y * 32;

  // Calculate shared memory requirements
  // Each warp needs one 16x16 float tile for intermediate storage
  const int FRAGMENT_SIZE_BYTES =
      16 * 16 * sizeof(float);  // 1024 bytes per tile (float, not bfloat16)
  const int WARPS_PER_BLOCK = WARPS_PER_BLOCK_X * WARPS_PER_BLOCK_Y;
  const int SHARED_MEM_SIZE =
      WARPS_PER_BLOCK * FRAGMENT_SIZE_BYTES;  // 8 warps × 1024 = 8192 bytes

  dim3 blockDim(THREADS_PER_BLOCK, 1);

  // Calculate grid dimensions based on warps per block
  int grid_x =
      ((M + WMMA_M - 1) / WMMA_M + WARPS_PER_BLOCK_X - 1) / WARPS_PER_BLOCK_X;
  int grid_y =
      ((N + WMMA_N - 1) / WMMA_N + WARPS_PER_BLOCK_Y - 1) / WARPS_PER_BLOCK_Y;

  dim3 gridDim(grid_x, grid_y);

  // Get CUDA stream
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  // Launch pipelined bfloat16 kernel with specified shared memory
  wmma_silu_add_kernel_bf16_pipelined<<<gridDim, blockDim, SHARED_MEM_SIZE,
                                        stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(B.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(C.data_ptr<at::BFloat16>()), M, N, K, K,
      N, N, N);

  cudaStreamSynchronize(stream);
}

// Test function to compare CUDA native result vs PyTorch native result
torch::Tensor test_wmma_vs_torch_native(torch::Tensor X, torch::Tensor A,
                                        torch::Tensor B) {
  // Validate inputs
  TORCH_CHECK(X.device().is_cuda(), "X must be a CUDA tensor");
  TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
  TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
  TORCH_CHECK(X.dtype() == torch::kBFloat16, "X must be bfloat16");
  TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be bfloat16");
  TORCH_CHECK(B.dtype() == torch::kBFloat16, "B must be bfloat16");

  auto M = X.size(0);
  auto K = X.size(1);
  auto N = A.size(1);

  // Create output tensors
  auto C_wmma =
      torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(X.device()));
  auto C_torch =
      torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(X.device()));

  // Test 1: WMMA CUDA kernel result
  auto start_wmma = std::chrono::high_resolution_clock::now();
  wmma_silu_add_cuda(X, A, B, C_wmma);
  cudaDeviceSynchronize();
  auto end_wmma = std::chrono::high_resolution_clock::now();
  auto duration_wmma = std::chrono::duration_cast<std::chrono::microseconds>(
      end_wmma - start_wmma);

  // Test 2: PyTorch native implementation
  auto start_torch = std::chrono::high_resolution_clock::now();
  auto XA = torch::mm(X, A);
  auto XB = torch::mm(X, B);
  auto silu_XA = torch::sigmoid(XA) * XA;  // SiLU activation
  C_torch = silu_XA + XB;
  cudaDeviceSynchronize();
  auto end_torch = std::chrono::high_resolution_clock::now();
  auto duration_torch = std::chrono::duration_cast<std::chrono::microseconds>(
      end_torch - start_torch);

  // Calculate differences
  auto diff = torch::abs(C_wmma - C_torch);
  auto max_diff = torch::max(diff);
  auto mean_diff = torch::mean(diff);
  auto relative_diff = diff / (torch::abs(C_torch) + 1e-8f);
  auto max_relative_diff = torch::max(relative_diff);
  auto mean_relative_diff = torch::mean(relative_diff);

  // Convert to CPU for printing
  auto max_diff_cpu = max_diff.cpu().item<float>();
  auto mean_diff_cpu = mean_diff.cpu().item<float>();
  auto max_rel_diff_cpu = max_relative_diff.cpu().item<float>();
  auto mean_rel_diff_cpu = mean_relative_diff.cpu().item<float>();

  // Print test results
  printf("\n=== WMMA vs PyTorch Native Comparison ===\n");
  printf("Matrix dimensions: M=%ld, K=%ld, N=%ld\n", M, K, N);
  printf("WMMA kernel time:  %ld μs\n", duration_wmma.count());
  printf("PyTorch time:      %ld μs\n", duration_torch.count());
  printf("Speedup:           %.2fx\n",
         (float)duration_torch.count() / duration_wmma.count());
  printf("\nAccuracy metrics:\n");
  printf("Max absolute diff:     %.8f\n", max_diff_cpu);
  printf("Mean absolute diff:    %.8f\n", mean_diff_cpu);
  printf("Max relative diff:     %.8f\n", max_rel_diff_cpu);
  printf("Mean relative diff:    %.8f\n", mean_rel_diff_cpu);

  // Determine test status
  const float abs_tolerance = 1e-3f;  // Relaxed for bfloat16
  const float rel_tolerance = 1e-2f;  // 1% relative tolerance

  bool test_passed =
      (max_diff_cpu < abs_tolerance) && (max_rel_diff_cpu < rel_tolerance);

  printf("\nTest result: %s\n", test_passed ? "PASSED" : "FAILED");
  if (!test_passed) {
    printf("Tolerance: abs < %.6f, rel < %.6f\n", abs_tolerance, rel_tolerance);
  }
  printf("==========================================\n\n");

  // Return results as a tensor for further analysis
  auto results =
      torch::tensor({max_diff_cpu, mean_diff_cpu, max_rel_diff_cpu,
                     mean_rel_diff_cpu, (float)duration_wmma.count(),
                     (float)duration_torch.count(), test_passed ? 1.0f : 0.0f},
                    torch::dtype(torch::kFloat32).device(torch::kCPU));

  return results;
}

// Comprehensive benchmark function
torch::Tensor benchmark_wmma_multiple_sizes() {
  printf("\n=== Comprehensive WMMA Benchmark ===\n");

  // Test different matrix sizes
  std::vector<std::tuple<int, int, int>> test_sizes = {
      {512, 512, 512},     // Small
      {1024, 1024, 1024},  // Medium
      {2048, 2048, 2048},  // Large
      {4096, 4096, 4096},  // Very large
      {1024, 512, 2048},   // Rectangular 1
      {2048, 1024, 512},   // Rectangular 2
      {8192, 8192, 8192},  // Huge (if memory allows)
  };

  std::vector<float> all_results;

  for (auto& [M, K, N] : test_sizes) {
    printf("\nTesting size M=%d, K=%d, N=%d\n", M, K, N);

    try {
      // Create test tensors
      auto X = torch::randn(
          {M, K}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
      auto A = torch::randn(
          {K, N}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
      auto B = torch::randn(
          {K, N}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));

      // Run test
      auto results = test_wmma_vs_torch_native(X, A, B);
      auto results_vec = results.accessor<float, 1>();

      // Store results
      for (int i = 0; i < results.size(0); i++) {
        all_results.push_back(results_vec[i]);
      }

      // Add matrix dimensions
      all_results.push_back((float)M);
      all_results.push_back((float)K);
      all_results.push_back((float)N);

    } catch (const std::exception& e) {
      printf("Skipped size %dx%dx%d due to: %s\n", M, K, N, e.what());
      // Add zeros for failed test
      for (int i = 0; i < 10; i++) {
        all_results.push_back(0.0f);
      }
    }
  }

  printf("=====================================\n");

  // Return all results as tensor
  return torch::tensor(all_results,
                       torch::dtype(torch::kFloat32).device(torch::kCPU));
}

int main() {
  // Initialize CUDA
  //   at::cuda::init();

  // Run comprehensive benchmark
  auto results = benchmark_wmma_multiple_sizes();

  // // Print final results
  // printf("\n=== Final Benchmark Results ===\n");
  // printf("Results tensor shape: %s\n", results.sizes().vec());

  return 0;
}
