#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <iostream>
#include <chrono>
#include <cmath>

#define CHECK_CUDA(call)                                                    \
  do {                                                                      \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                << cudaGetErrorString(err) << std::endl;                    \
      exit(EXIT_FAILURE);                                                   \
    }                                                                       \
  } while (0)

#define CHECK_CUBLAS(call)                                           \
  do {                                                               \
    cublasStatus_t status = call;                                    \
    if (status != CUBLAS_STATUS_SUCCESS) {                           \
      std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ \
                << std::endl;                                        \
      exit(EXIT_FAILURE);                                            \
    }                                                                \
  } while (0)

// SiLU activation kernel: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
__global__ void silu_kernel(__nv_bfloat16* data, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float val = __bfloat162float(data[idx]);
    // SiLU: x * sigmoid(x)
    float sigmoid_val = 1.0f / (1.0f + expf(-val));
    float silu_val = val * sigmoid_val;
    data[idx] = __float2bfloat16(silu_val);
  }
}

// Element-wise addition kernel
__global__ void add_kernel(__nv_bfloat16* a, __nv_bfloat16* b,
                           __nv_bfloat16* result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float val_a = __bfloat162float(a[idx]);
    float val_b = __bfloat162float(b[idx]);
    result[idx] = __float2bfloat16(val_a + val_b);
  }
}

// Kernel to pollute L2 cache
__global__ void pollute_cache(__nv_bfloat16* trash, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    __nv_bfloat16 val = trash[idx];
    // Do some computation to force cache pollution
    for (int i = 0; i < 10; i++) {
      val = __hmul(val, __float2bfloat16(1.001f));
      val = __hadd(val, __float2bfloat16(0.001f));
    }
    trash[idx] = val;
  }
}

class Timer {
  std::chrono::high_resolution_clock::time_point start;

 public:
  Timer() : start(std::chrono::high_resolution_clock::now()) {}

  double elapsed() {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
  }

  void reset() { start = std::chrono::high_resolution_clock::now(); }
};

// Test case: (A * B) + silu(A * C) without L2 cache optimization
void shared_input_no_optimization(cublasHandle_t handle, int M, int K, int N) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // Allocate device memory
  __nv_bfloat16 *d_A, *d_B, *d_C, *d_AB, *d_AC, *d_result, *d_trash;
  size_t size_A = M * K * sizeof(__nv_bfloat16);       // M x K
  size_t size_B = K * N * sizeof(__nv_bfloat16);       // K x N
  size_t size_C = K * N * sizeof(__nv_bfloat16);       // K x N
  size_t size_result = M * N * sizeof(__nv_bfloat16);  // M x N
  size_t trash_size = 32 * 1024 * 1024;  // 32MB to pollute L2 cache

  CHECK_CUDA(cudaMalloc(&d_A, size_A));
  CHECK_CUDA(cudaMalloc(&d_B, size_B));
  CHECK_CUDA(cudaMalloc(&d_C, size_C));
  CHECK_CUDA(cudaMalloc(&d_AB, size_result));
  CHECK_CUDA(cudaMalloc(&d_AC, size_result));
  CHECK_CUDA(cudaMalloc(&d_result, size_result));
  CHECK_CUDA(cudaMalloc(&d_trash, trash_size));

  // Initialize with random data
  __nv_bfloat16* h_A = new __nv_bfloat16[M * K];
  __nv_bfloat16* h_B = new __nv_bfloat16[K * N];
  __nv_bfloat16* h_C = new __nv_bfloat16[K * N];

  for (int i = 0; i < M * K; i++) {
    h_A[i] = __float2bfloat16(static_cast<float>(rand()) / RAND_MAX);
  }
  for (int i = 0; i < K * N; i++) {
    h_B[i] = __float2bfloat16(static_cast<float>(rand()) / RAND_MAX);
    h_C[i] = __float2bfloat16(static_cast<float>(rand()) / RAND_MAX);
  }

  CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));

  // Warmup
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            d_B, CUDA_R_16BF, N, d_A, CUDA_R_16BF, K, &beta,
                            d_AB, CUDA_R_16BF, N, CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  CHECK_CUDA(cudaDeviceSynchronize());

  // Test: Without cache optimization
  // A is read twice but likely evicted between operations
  Timer timer;

  // First GEMM: AB = A * B
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            d_B, CUDA_R_16BF, N, d_A, CUDA_R_16BF, K, &beta,
                            d_AB, CUDA_R_16BF, N, CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  CHECK_CUDA(cudaDeviceSynchronize());

  // Pollute L2 cache to simulate realistic scenario
  int blockSize = 256;
  int numBlocks =
      (trash_size / sizeof(__nv_bfloat16) + blockSize - 1) / blockSize;
  pollute_cache<<<numBlocks, blockSize>>>(d_trash,
                                          trash_size / sizeof(__nv_bfloat16));
  CHECK_CUDA(cudaDeviceSynchronize());

  // Second GEMM: AC = A * C (A needs to be re-read from memory)
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            d_C, CUDA_R_16BF, N, d_A, CUDA_R_16BF, K, &beta,
                            d_AC, CUDA_R_16BF, N, CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  CHECK_CUDA(cudaDeviceSynchronize());

  // Apply SiLU activation: silu(AC)
  int total_elements = M * N;
  int silu_blocks = (total_elements + blockSize - 1) / blockSize;
  silu_kernel<<<silu_blocks, blockSize>>>(d_AC, total_elements);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Add: result = AB + silu(AC)
  add_kernel<<<silu_blocks, blockSize>>>(d_AB, d_AC, d_result, total_elements);
  CHECK_CUDA(cudaDeviceSynchronize());

  double time = timer.elapsed();
  std::cout << "No Cache Optimization (A polluted): " << time << " ms"
            << std::endl;

  // Cleanup
  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));
  CHECK_CUDA(cudaFree(d_AB));
  CHECK_CUDA(cudaFree(d_AC));
  CHECK_CUDA(cudaFree(d_result));
  CHECK_CUDA(cudaFree(d_trash));
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
}

// Test case: (A * B) + silu(A * C) with back-to-back execution
void shared_input_sequential(cublasHandle_t handle, int M, int K, int N) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // Allocate device memory
  __nv_bfloat16 *d_A, *d_B, *d_C, *d_AB, *d_AC, *d_result;
  size_t size_A = M * K * sizeof(__nv_bfloat16);
  size_t size_B = K * N * sizeof(__nv_bfloat16);
  size_t size_C = K * N * sizeof(__nv_bfloat16);
  size_t size_result = M * N * sizeof(__nv_bfloat16);

  CHECK_CUDA(cudaMalloc(&d_A, size_A));
  CHECK_CUDA(cudaMalloc(&d_B, size_B));
  CHECK_CUDA(cudaMalloc(&d_C, size_C));
  CHECK_CUDA(cudaMalloc(&d_AB, size_result));
  CHECK_CUDA(cudaMalloc(&d_AC, size_result));
  CHECK_CUDA(cudaMalloc(&d_result, size_result));

  // Initialize with random data
  __nv_bfloat16* h_A = new __nv_bfloat16[M * K];
  __nv_bfloat16* h_B = new __nv_bfloat16[K * N];
  __nv_bfloat16* h_C = new __nv_bfloat16[K * N];

  for (int i = 0; i < M * K; i++) {
    h_A[i] = __float2bfloat16(static_cast<float>(rand()) / RAND_MAX);
  }
  for (int i = 0; i < K * N; i++) {
    h_B[i] = __float2bfloat16(static_cast<float>(rand()) / RAND_MAX);
    h_C[i] = __float2bfloat16(static_cast<float>(rand()) / RAND_MAX);
  }

  CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));

  // Warmup
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            d_B, CUDA_R_16BF, N, d_A, CUDA_R_16BF, K, &beta,
                            d_AB, CUDA_R_16BF, N, CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  CHECK_CUDA(cudaDeviceSynchronize());

  // Test: Sequential execution (A stays in L2 naturally)
  Timer timer;

  // First GEMM: AB = A * B
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            d_B, CUDA_R_16BF, N, d_A, CUDA_R_16BF, K, &beta,
                            d_AB, CUDA_R_16BF, N, CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // Second GEMM immediately: AC = A * C (A likely still in L2)
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            d_C, CUDA_R_16BF, N, d_A, CUDA_R_16BF, K, &beta,
                            d_AC, CUDA_R_16BF, N, CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  CHECK_CUDA(cudaDeviceSynchronize());

  // Apply SiLU activation: silu(AC)
  int total_elements = M * N;
  int blockSize = 256;
  int silu_blocks = (total_elements + blockSize - 1) / blockSize;
  silu_kernel<<<silu_blocks, blockSize>>>(d_AC, total_elements);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Add: result = AB + silu(AC)
  add_kernel<<<silu_blocks, blockSize>>>(d_AB, d_AC, d_result, total_elements);
  CHECK_CUDA(cudaDeviceSynchronize());

  double time = timer.elapsed();
  std::cout << "Sequential (A naturally cached): " << time << " ms"
            << std::endl;

  // Cleanup
  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));
  CHECK_CUDA(cudaFree(d_AB));
  CHECK_CUDA(cudaFree(d_AC));
  CHECK_CUDA(cudaFree(d_result));
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
}

// Test case: (A * B) + silu(A * C) with L2 persistence hint for A
void shared_input_with_persistence(cublasHandle_t handle, int M, int K, int N) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // Allocate device memory
  __nv_bfloat16 *d_A, *d_B, *d_C, *d_AB, *d_AC, *d_result;
  size_t size_A = M * K * sizeof(__nv_bfloat16);
  size_t size_B = K * N * sizeof(__nv_bfloat16);
  size_t size_C = K * N * sizeof(__nv_bfloat16);
  size_t size_result = M * N * sizeof(__nv_bfloat16);

  CHECK_CUDA(cudaMalloc(&d_A, size_A));
  CHECK_CUDA(cudaMalloc(&d_B, size_B));
  CHECK_CUDA(cudaMalloc(&d_C, size_C));
  CHECK_CUDA(cudaMalloc(&d_AB, size_result));
  CHECK_CUDA(cudaMalloc(&d_AC, size_result));
  CHECK_CUDA(cudaMalloc(&d_result, size_result));

  // Initialize with random data
  __nv_bfloat16* h_A = new __nv_bfloat16[M * K];
  __nv_bfloat16* h_B = new __nv_bfloat16[K * N];
  __nv_bfloat16* h_C = new __nv_bfloat16[K * N];

  for (int i = 0; i < M * K; i++) {
    h_A[i] = __float2bfloat16(static_cast<float>(rand()) / RAND_MAX);
  }
  for (int i = 0; i < K * N; i++) {
    h_B[i] = __float2bfloat16(static_cast<float>(rand()) / RAND_MAX);
    h_C[i] = __float2bfloat16(static_cast<float>(rand()) / RAND_MAX);
  }

  CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));

  // Set L2 cache persistence for shared input A (Ampere and newer)
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

  cudaStream_t stream = nullptr;
  if (prop.major >= 8) {  // Ampere or newer
    cudaStreamAttrValue stream_attribute;
    stream_attribute.accessPolicyWindow.base_ptr = d_A;
    stream_attribute.accessPolicyWindow.num_bytes = size_A;
    stream_attribute.accessPolicyWindow.hitRatio = 1.0f;
    stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(cudaStreamSetAttribute(
        stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute));

    CHECK_CUBLAS(cublasSetStream(handle, stream));
  }

  // Warmup
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            d_B, CUDA_R_16BF, N, d_A, CUDA_R_16BF, K, &beta,
                            d_AB, CUDA_R_16BF, N, CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  CHECK_CUDA(cudaDeviceSynchronize());

  // Test: With L2 persistence hint for A
  Timer timer;

  // First GEMM: AB = A * B
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            d_B, CUDA_R_16BF, N, d_A, CUDA_R_16BF, K, &beta,
                            d_AB, CUDA_R_16BF, N, CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // Second GEMM: AC = A * C (A persisted in L2)
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            d_C, CUDA_R_16BF, N, d_A, CUDA_R_16BF, K, &beta,
                            d_AC, CUDA_R_16BF, N, CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  CHECK_CUDA(cudaDeviceSynchronize());

  // Apply SiLU activation: silu(AC)
  int total_elements = M * N;
  int blockSize = 256;
  int silu_blocks = (total_elements + blockSize - 1) / blockSize;
  silu_kernel<<<silu_blocks, blockSize>>>(d_AC, total_elements);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Add: result = AB + silu(AC)
  add_kernel<<<silu_blocks, blockSize>>>(d_AB, d_AC, d_result, total_elements);
  CHECK_CUDA(cudaDeviceSynchronize());

  double time = timer.elapsed();
  std::cout << "L2 Persistence for A: " << time << " ms";
  if (prop.major >= 8) {
    std::cout << " (enabled)";
  } else {
    std::cout << " (not supported on " << prop.name << ")";
  }
  std::cout << std::endl;

  // Reset stream
  if (stream) {
    CHECK_CUBLAS(cublasSetStream(handle, nullptr));
    CHECK_CUDA(cudaStreamDestroy(stream));
  }

  // Cleanup
  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));
  CHECK_CUDA(cudaFree(d_AB));
  CHECK_CUDA(cudaFree(d_AC));
  CHECK_CUDA(cudaFree(d_result));
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
}

int main() {
  // Initialize cuBLAS
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  // Get device properties
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

  std::cout << "========================================" << std::endl;
  std::cout << "Shared Input L2 Cache Persistence Test" << std::endl;
  std::cout << "Pattern: (A * B) + silu(A * C)" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Device: " << prop.name << std::endl;
  std::cout << "L2 Cache Size: " << prop.l2CacheSize / 1024 << " KB"
            << std::endl;
  std::cout << "========================================" << std::endl;

  // Test configurations: (M, K, N)
  struct TestConfig {
    int M, K, N;
    const char* description;
  };

  TestConfig configs[] = {{32, 1024, 768, "Small: 32 x 1K x 768"},
                          {128, 2048, 768, "Medium: 128 x 2K x 768"},
                          {32, 4096, 4096, "Large: 32 x 4K x 4K"},
                          {1024, 4096, 4096, "Wide A: 1K x 4K, B/C: 4K x 4K"},
                          {4096, 1024, 4096, "Tall A: 4K x 1K, B/C: 1K x 4K"}};

  for (const auto& config : configs) {
    std::cout << "\n" << config.description << std::endl;
    std::cout << "Matrix A: " << config.M << " x " << config.K << " ("
              << (config.M * config.K * sizeof(__nv_bfloat16)) / (1024 * 1024)
              << " MB)" << std::endl;
    std::cout << "Matrix B/C: " << config.K << " x " << config.N << " ("
              << (config.K * config.N * sizeof(__nv_bfloat16)) / (1024 * 1024)
              << " MB)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    shared_input_sequential(handle, config.M, config.K, config.N);
    shared_input_no_optimization(handle, config.M, config.K, config.N);
    shared_input_with_persistence(handle, config.M, config.K, config.N);

    std::cout << std::endl;
  }

  // Cleanup
  CHECK_CUBLAS(cublasDestroy(handle));

  return 0;
}
