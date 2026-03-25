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

void sequential_gemm_l2_optimized(cublasHandle_t handle, int N) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // Allocate device memory
  __nv_bfloat16 *d_A, *d_B, *d_C, *d_temp, *d_result;
  size_t bytes = N * N * sizeof(__nv_bfloat16);

  CHECK_CUDA(cudaMalloc(&d_A, bytes));
  CHECK_CUDA(cudaMalloc(&d_B, bytes));
  CHECK_CUDA(cudaMalloc(&d_C, bytes));
  CHECK_CUDA(cudaMalloc(&d_temp, bytes));  // Intermediate result
  CHECK_CUDA(cudaMalloc(&d_result, bytes));

  // Initialize with random data
  __nv_bfloat16* h_data = new __nv_bfloat16[N * N];
  for (int i = 0; i < N * N; i++) {
    h_data[i] = __float2bfloat16(static_cast<float>(rand()) / RAND_MAX);
  }

  CHECK_CUDA(cudaMemcpy(d_A, h_data, bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, h_data, bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_C, h_data, bytes, cudaMemcpyHostToDevice));

  // Warmup
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                            d_A, CUDA_R_16BF, N, d_B, CUDA_R_16BF, N, &beta,
                            d_temp, CUDA_R_16BF, N, CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  CHECK_CUDA(cudaDeviceSynchronize());

  // Test: Sequential GEMM with L2 cache optimization
  // result = (A * B) * C, keeping intermediate in L2
  Timer timer;

  // First GEMM: temp = A * B
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                            d_A, CUDA_R_16BF, N, d_B, CUDA_R_16BF, N, &beta,
                            d_temp, CUDA_R_16BF, N, CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // Second GEMM immediately: result = temp * C
  // temp should still be hot in L2 cache
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                            d_temp, CUDA_R_16BF, N, d_C, CUDA_R_16BF, N, &beta,
                            d_result, CUDA_R_16BF, N, CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  CHECK_CUDA(cudaDeviceSynchronize());
  double time_l2 = timer.elapsed();

  std::cout << "L2 Cache Optimized: " << time_l2 << " ms" << std::endl;

  // Cleanup
  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));
  CHECK_CUDA(cudaFree(d_temp));
  CHECK_CUDA(cudaFree(d_result));
  delete[] h_data;
}

void sequential_gemm_default(cublasHandle_t handle, int N) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // Allocate device memory
  __nv_bfloat16 *d_A, *d_B, *d_C, *d_temp, *d_result, *d_trash;
  size_t bytes = N * N * sizeof(__nv_bfloat16);
  size_t trash_size = 32 * 1024 * 1024;  // 32MB to pollute L2 cache

  CHECK_CUDA(cudaMalloc(&d_A, bytes));
  CHECK_CUDA(cudaMalloc(&d_B, bytes));
  CHECK_CUDA(cudaMalloc(&d_C, bytes));
  CHECK_CUDA(cudaMalloc(&d_temp, bytes));
  CHECK_CUDA(cudaMalloc(&d_result, bytes));
  CHECK_CUDA(cudaMalloc(&d_trash, trash_size));

  // Initialize
  __nv_bfloat16* h_data = new __nv_bfloat16[N * N];
  for (int i = 0; i < N * N; i++) {
    h_data[i] = __float2bfloat16(static_cast<float>(rand()) / RAND_MAX);
  }

  CHECK_CUDA(cudaMemcpy(d_A, h_data, bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, h_data, bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_C, h_data, bytes, cudaMemcpyHostToDevice));

  // Warmup
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                            d_A, CUDA_R_16BF, N, d_B, CUDA_R_16BF, N, &beta,
                            d_temp, CUDA_R_16BF, N, CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  CHECK_CUDA(cudaDeviceSynchronize());

  // Test: Sequential GEMM with cache pollution (system default behavior)
  Timer timer;

  // First GEMM: temp = A * B
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                            d_A, CUDA_R_16BF, N, d_B, CUDA_R_16BF, N, &beta,
                            d_temp, CUDA_R_16BF, N, CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  CHECK_CUDA(cudaDeviceSynchronize());

  // Pollute L2 cache to simulate system default behavior
  int blockSize = 256;
  int numBlocks =
      (trash_size / sizeof(__nv_bfloat16) + blockSize - 1) / blockSize;
  pollute_cache<<<numBlocks, blockSize>>>(d_trash,
                                          trash_size / sizeof(__nv_bfloat16));
  CHECK_CUDA(cudaDeviceSynchronize());

  // Second GEMM: result = temp * C
  // temp is likely evicted from L2 cache now
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                            d_temp, CUDA_R_16BF, N, d_C, CUDA_R_16BF, N, &beta,
                            d_result, CUDA_R_16BF, N, CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  CHECK_CUDA(cudaDeviceSynchronize());
  double time_default = timer.elapsed();

  std::cout << "System Default (cache polluted): " << time_default << " ms"
            << std::endl;

  // Cleanup
  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));
  CHECK_CUDA(cudaFree(d_temp));
  CHECK_CUDA(cudaFree(d_result));
  CHECK_CUDA(cudaFree(d_trash));
  delete[] h_data;
}

void sequential_gemm_with_persistence(cublasHandle_t handle, int N) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // Allocate device memory
  __nv_bfloat16 *d_A, *d_B, *d_C, *d_temp, *d_result;
  size_t bytes = N * N * sizeof(__nv_bfloat16);

  CHECK_CUDA(cudaMalloc(&d_A, bytes));
  CHECK_CUDA(cudaMalloc(&d_B, bytes));
  CHECK_CUDA(cudaMalloc(&d_C, bytes));
  CHECK_CUDA(cudaMalloc(&d_temp, bytes));
  CHECK_CUDA(cudaMalloc(&d_result, bytes));

  // Initialize
  __nv_bfloat16* h_data = new __nv_bfloat16[N * N];
  for (int i = 0; i < N * N; i++) {
    h_data[i] = __float2bfloat16(static_cast<float>(rand()) / RAND_MAX);
  }

  CHECK_CUDA(cudaMemcpy(d_A, h_data, bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, h_data, bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_C, h_data, bytes, cudaMemcpyHostToDevice));

  // Set L2 cache persistence for intermediate result (Ampere and newer)
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

  if (prop.major >= 8) {  // Ampere or newer
    cudaStreamAttrValue stream_attribute;
    stream_attribute.accessPolicyWindow.base_ptr = d_temp;
    stream_attribute.accessPolicyWindow.num_bytes = bytes;
    stream_attribute.accessPolicyWindow.hitRatio = 1.0f;
    stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(cudaStreamSetAttribute(
        stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute));

    CHECK_CUBLAS(cublasSetStream(handle, stream));
  }

  // Warmup
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                            d_A, CUDA_R_16BF, N, d_B, CUDA_R_16BF, N, &beta,
                            d_temp, CUDA_R_16BF, N, CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  CHECK_CUDA(cudaDeviceSynchronize());

  // Test: Sequential GEMM with L2 persistence hint
  Timer timer;

  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                            d_A, CUDA_R_16BF, N, d_B, CUDA_R_16BF, N, &beta,
                            d_temp, CUDA_R_16BF, N, CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                            d_temp, CUDA_R_16BF, N, d_C, CUDA_R_16BF, N, &beta,
                            d_result, CUDA_R_16BF, N, CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  CHECK_CUDA(cudaDeviceSynchronize());
  double time_persist = timer.elapsed();

  std::cout << "L2 Persistence Hint: " << time_persist << " ms";
  if (prop.major >= 8) {
    std::cout << " (enabled)";
  } else {
    std::cout << " (not supported on " << prop.name << ")";
  }
  std::cout << std::endl;

  // Cleanup
  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));
  CHECK_CUDA(cudaFree(d_temp));
  CHECK_CUDA(cudaFree(d_result));
  delete[] h_data;
}

int main() {
  // Initialize cuBLAS
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  // Get device properties
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

  std::cout << "========================================" << std::endl;
  std::cout << "Sequential GEMM L2 Cache Test" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Device: " << prop.name << std::endl;
  std::cout << "L2 Cache Size: " << prop.l2CacheSize / 1024 << " KB"
            << std::endl;
  std::cout << "========================================" << std::endl;

  // Test with different matrix sizes
  int sizes[] = {1024, 2048, 4096};

  for (int N : sizes) {
    std::cout << "\nMatrix Size: " << N << "x" << N << std::endl;
    std::cout << "Memory per matrix: "
              << (N * N * sizeof(__nv_bfloat16)) / (1024 * 1024) << " MB"
              << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    sequential_gemm_l2_optimized(handle, N);
    sequential_gemm_default(handle, N);
    sequential_gemm_with_persistence(handle, N);

    std::cout << std::endl;
  }

  // Cleanup
  CHECK_CUBLAS(cublasDestroy(handle));

  return 0;
}
