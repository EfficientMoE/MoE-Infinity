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

// SiLU activation: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

// BF16 helper functions
__device__ __forceinline__ float bf16_to_fp32(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 fp32_to_bf16(float x) {
  return __float2bfloat16(x);
}

// Unfused baseline: separate kernels with BF16
__global__ void silu_kernel_bf16(const __nv_bfloat16* __restrict__ input,
                                 __nv_bfloat16* __restrict__ output, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float val = bf16_to_fp32(input[idx]);
    output[idx] = fp32_to_bf16(silu(val));
  }
}

__global__ void elementwise_mul_kernel_bf16(const __nv_bfloat16* __restrict__ a,
                                            const __nv_bfloat16* __restrict__ b,
                                            __nv_bfloat16* __restrict__ output,
                                            int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float a_val = bf16_to_fp32(a[idx]);
    float b_val = bf16_to_fp32(b[idx]);
    output[idx] = fp32_to_bf16(a_val * b_val);
  }
}

// Fused kernel: gate * silu(up) in one pass with BF16
__global__ void fused_gated_silu_kernel_bf16(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up, __nv_bfloat16* __restrict__ output,
    int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float gate_val = bf16_to_fp32(gate[idx]);
    float up_val = bf16_to_fp32(up[idx]);
    output[idx] = fp32_to_bf16(gate_val * silu(up_val));
  }
}

// Vectorized fused kernel using __nv_bfloat162
__global__ void fused_gated_silu_vectorized_kernel_bf16(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up, __nv_bfloat16* __restrict__ output,
    int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  if (idx + 1 < N) {
    __nv_bfloat162 gate_val =
        reinterpret_cast<const __nv_bfloat162*>(gate)[idx / 2];
    __nv_bfloat162 up_val =
        reinterpret_cast<const __nv_bfloat162*>(up)[idx / 2];

    float2 gate_fp32 = __bfloat1622float2(gate_val);
    float2 up_fp32 = __bfloat1622float2(up_val);

    float2 result;
    result.x = gate_fp32.x * silu(up_fp32.x);
    result.y = gate_fp32.y * silu(up_fp32.y);

    reinterpret_cast<__nv_bfloat162*>(output)[idx / 2] =
        __floats2bfloat162_rn(result.x, result.y);
  }

  // Handle remainder
  if (idx == (N / 2) * 2 && N % 2 == 1 && threadIdx.x == 0 &&
      blockIdx.x == gridDim.x - 1) {
    float gate_val = bf16_to_fp32(gate[idx]);
    float up_val = bf16_to_fp32(up[idx]);
    output[idx] = fp32_to_bf16(gate_val * silu(up_val));
  }
}

// Vectorized kernel using 8 elements (4x __nv_bfloat162)
__global__ void fused_gated_silu_vectorized8_kernel_bf16(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up, __nv_bfloat16* __restrict__ output,
    int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (idx + 7 < N) {
// Load 8 bf16 values (4x bf162)
#pragma unroll
    for (int i = 0; i < 4; i++) {
      __nv_bfloat162 gate_val =
          reinterpret_cast<const __nv_bfloat162*>(gate)[(idx / 2) + i];
      __nv_bfloat162 up_val =
          reinterpret_cast<const __nv_bfloat162*>(up)[(idx / 2) + i];

      float2 gate_fp32 = __bfloat1622float2(gate_val);
      float2 up_fp32 = __bfloat1622float2(up_val);

      float2 result;
      result.x = gate_fp32.x * silu(up_fp32.x);
      result.y = gate_fp32.y * silu(up_fp32.y);

      reinterpret_cast<__nv_bfloat162*>(output)[(idx / 2) + i] =
          __floats2bfloat162_rn(result.x, result.y);
    }
  }

  // Handle remainder
  int base_idx = (N / 8) * 8;
  if (threadIdx.x == 0 && blockIdx.x == gridDim.x - 1) {
    for (int i = base_idx; i < N; i++) {
      float gate_val = bf16_to_fp32(gate[i]);
      float up_val = bf16_to_fp32(up[i]);
      output[i] = fp32_to_bf16(gate_val * silu(up_val));
    }
  }
}

// Full fused kernel with BF16
__global__ void fused_gemm_gated_silu_kernel_bf16(
    const __nv_bfloat16* __restrict__ AB_result,
    const __nv_bfloat16* __restrict__ AC_result,
    __nv_bfloat16* __restrict__ output, int M, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    int idx = row * N + col;
    float gate_val = bf16_to_fp32(AB_result[idx]);
    float up_val = bf16_to_fp32(AC_result[idx]);
    output[idx] = fp32_to_bf16(gate_val * silu(up_val));
  }
}

class Timer {
  cudaEvent_t start, stop;

 public:
  Timer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
  }

  float elapsed() {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
  }

  ~Timer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
};

// Optimized fully fused kernel with shared memory tiling and register
// accumulation Computes: output = (A * B) * silu(A * C) in one kernel
template <int TILE_M, int TILE_N, int TILE_K>
__global__ void fully_fused_tiled_gemm_gated_silu_kernel(
    const __nv_bfloat16* __restrict__ A,  // [M, K]
    const __nv_bfloat16* __restrict__ B,  // [K, N]
    const __nv_bfloat16* __restrict__ C,  // [K, N]
    __nv_bfloat16* __restrict__ output,   // [M, N]
    int M, int K, int N) {
  // Shared memory for tiles
  __shared__ float tile_A[TILE_M][TILE_K];
  __shared__ float tile_B[TILE_K][TILE_N];
  __shared__ float tile_C[TILE_K][TILE_N];

  // Thread coordinates
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Output coordinates
  int row = by * TILE_M + ty;
  int col = bx * TILE_N + tx;

  // Register accumulators for AB and AC
  float acc_AB = 0.0f;
  float acc_AC = 0.0f;

  // Loop over K dimension in tiles
  int num_tiles = (K + TILE_K - 1) / TILE_K;

  for (int tile = 0; tile < num_tiles; tile++) {
    int k_start = tile * TILE_K;

    // Cooperatively load tile_A into shared memory
    // Each thread loads multiple elements if needed
    for (int i = ty; i < TILE_M; i += blockDim.y) {
      for (int j = tx; j < TILE_K; j += blockDim.x) {
        int global_row = by * TILE_M + i;
        int global_col = k_start + j;
        if (global_row < M && global_col < K) {
          tile_A[i][j] = bf16_to_fp32(A[global_row * K + global_col]);
        } else {
          tile_A[i][j] = 0.0f;
        }
      }
    }

    // Cooperatively load tile_B and tile_C into shared memory
    for (int i = ty; i < TILE_K; i += blockDim.y) {
      for (int j = tx; j < TILE_N; j += blockDim.x) {
        int global_row = k_start + i;
        int global_col = bx * TILE_N + j;
        if (global_row < K && global_col < N) {
          tile_B[i][j] = bf16_to_fp32(B[global_row * N + global_col]);
          tile_C[i][j] = bf16_to_fp32(C[global_row * N + global_col]);
        } else {
          tile_B[i][j] = 0.0f;
          tile_C[i][j] = 0.0f;
        }
      }
    }

    __syncthreads();

    // Compute partial dot products for this tile
    // Each thread computes one element of output
    if (ty < TILE_M && tx < TILE_N) {
#pragma unroll
      for (int k = 0; k < TILE_K; k++) {
        acc_AB += tile_A[ty][k] * tile_B[k][tx];
        acc_AC += tile_A[ty][k] * tile_C[k][tx];
      }
    }

    __syncthreads();
  }

  // Apply gated SiLU and write output
  // output = AB * silu(AC)
  if (row < M && col < N) {
    float result = acc_AB * silu(acc_AC);
    output[row * N + col] = fp32_to_bf16(result);
  }
}

// Wrapper for fully fused tiled kernel
void test_fully_fused_tiled(cublasHandle_t handle, int M, int K, int N,
                            __nv_bfloat16* d_A, __nv_bfloat16* d_B,
                            __nv_bfloat16* d_C, __nv_bfloat16* d_output) {
  Timer timer;

  // Tile sizes
  const int TILE_M = 32;
  const int TILE_N = 32;
  const int TILE_K = 32;

  // Launch configuration
  dim3 blockSize(TILE_N, TILE_M);
  dim3 gridSize((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

  fully_fused_tiled_gemm_gated_silu_kernel<TILE_M, TILE_N, TILE_K>
      <<<gridSize, blockSize>>>(d_A, d_B, d_C, d_output, M, K, N);

  CHECK_CUDA(cudaDeviceSynchronize());
  float time = timer.elapsed();

  std::cout << "Fully fused tiled (shared mem): " << time << " ms" << std::endl;
}

// Advanced: Double buffering version with larger tiles
template <int TILE_M, int TILE_N, int TILE_K>
__global__ void fully_fused_double_buffer_kernel(
    const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
    const __nv_bfloat16* __restrict__ C, __nv_bfloat16* __restrict__ output,
    int M, int K, int N) {
  // Double buffered shared memory
  __shared__ float tile_A[2][TILE_M][TILE_K];
  __shared__ float tile_B[2][TILE_K][TILE_N];
  __shared__ float tile_C[2][TILE_K][TILE_N];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int row = by * TILE_M + ty;
  int col = bx * TILE_N + tx;

  float acc_AB = 0.0f;
  float acc_AC = 0.0f;

  int num_tiles = (K + TILE_K - 1) / TILE_K;
  int write_idx = 0;
  int read_idx = 1;

  // Prefetch first tile
  int k_start = 0;
  for (int i = ty; i < TILE_M; i += blockDim.y) {
    for (int j = tx; j < TILE_K; j += blockDim.x) {
      int global_row = by * TILE_M + i;
      int global_col = k_start + j;
      if (global_row < M && global_col < K) {
        tile_A[write_idx][i][j] = bf16_to_fp32(A[global_row * K + global_col]);
      } else {
        tile_A[write_idx][i][j] = 0.0f;
      }
    }
  }

  for (int i = ty; i < TILE_K; i += blockDim.y) {
    for (int j = tx; j < TILE_N; j += blockDim.x) {
      int global_row = k_start + i;
      int global_col = bx * TILE_N + j;
      if (global_row < K && global_col < N) {
        tile_B[write_idx][i][j] = bf16_to_fp32(B[global_row * N + global_col]);
        tile_C[write_idx][i][j] = bf16_to_fp32(C[global_row * N + global_col]);
      } else {
        tile_B[write_idx][i][j] = 0.0f;
        tile_C[write_idx][i][j] = 0.0f;
      }
    }
  }
  __syncthreads();

  // Main loop with double buffering
  for (int tile = 0; tile < num_tiles; tile++) {
    // Swap buffers
    read_idx = write_idx;
    write_idx = 1 - write_idx;

    // Prefetch next tile while computing current
    if (tile + 1 < num_tiles) {
      k_start = (tile + 1) * TILE_K;

      for (int i = ty; i < TILE_M; i += blockDim.y) {
        for (int j = tx; j < TILE_K; j += blockDim.x) {
          int global_row = by * TILE_M + i;
          int global_col = k_start + j;
          if (global_row < M && global_col < K) {
            tile_A[write_idx][i][j] =
                bf16_to_fp32(A[global_row * K + global_col]);
          } else {
            tile_A[write_idx][i][j] = 0.0f;
          }
        }
      }

      for (int i = ty; i < TILE_K; i += blockDim.y) {
        for (int j = tx; j < TILE_N; j += blockDim.x) {
          int global_row = k_start + i;
          int global_col = bx * TILE_N + j;
          if (global_row < K && global_col < N) {
            tile_B[write_idx][i][j] =
                bf16_to_fp32(B[global_row * N + global_col]);
            tile_C[write_idx][i][j] =
                bf16_to_fp32(C[global_row * N + global_col]);
          } else {
            tile_B[write_idx][i][j] = 0.0f;
            tile_C[write_idx][i][j] = 0.0f;
          }
        }
      }
    }

    // Compute using current tile (read_idx)
    if (ty < TILE_M && tx < TILE_N) {
#pragma unroll
      for (int k = 0; k < TILE_K; k++) {
        acc_AB += tile_A[read_idx][ty][k] * tile_B[read_idx][k][tx];
        acc_AC += tile_A[read_idx][ty][k] * tile_C[read_idx][k][tx];
      }
    }

    __syncthreads();
  }

  // Write final result
  if (row < M && col < N) {
    float result = acc_AB * silu(acc_AC);
    output[row * N + col] = fp32_to_bf16(result);
  }
}

void test_fully_fused_double_buffer(cublasHandle_t handle, int M, int K, int N,
                                    __nv_bfloat16* d_A, __nv_bfloat16* d_B,
                                    __nv_bfloat16* d_C,
                                    __nv_bfloat16* d_output) {
  Timer timer;

  const int TILE_M = 32;
  const int TILE_N = 32;
  const int TILE_K = 32;

  dim3 blockSize(TILE_N, TILE_M);
  dim3 gridSize((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

  fully_fused_double_buffer_kernel<TILE_M, TILE_N, TILE_K>
      <<<gridSize, blockSize>>>(d_A, d_B, d_C, d_output, M, K, N);

  CHECK_CUDA(cudaDeviceSynchronize());
  float time = timer.elapsed();

  std::cout << "Fully fused double-buffered:     " << time << " ms"
            << std::endl;
}

void test_unfused(cublasHandle_t handle, int M, int K, int N,
                  __nv_bfloat16* d_A, __nv_bfloat16* d_B, __nv_bfloat16* d_C,
                  __nv_bfloat16* d_AB, __nv_bfloat16* d_AC,
                  __nv_bfloat16* d_silu_AC, __nv_bfloat16* d_output) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  Timer timer;

  // Step 1: AB = A * B (gate projection)
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            d_B, CUDA_R_16BF, N, d_A, CUDA_R_16BF, K, &beta,
                            d_AB, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // Step 2: AC = A * C (up projection)
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            d_C, CUDA_R_16BF, N, d_A, CUDA_R_16BF, K, &beta,
                            d_AC, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // Step 3: silu(AC)
  int total_elements = M * N;
  int blockSize = 256;
  int numBlocks = (total_elements + blockSize - 1) / blockSize;
  silu_kernel_bf16<<<numBlocks, blockSize>>>(d_AC, d_silu_AC, total_elements);

  // Step 4: output = AB * silu(AC)
  elementwise_mul_kernel_bf16<<<numBlocks, blockSize>>>(
      d_AB, d_silu_AC, d_output, total_elements);

  CHECK_CUDA(cudaDeviceSynchronize());
  float time = timer.elapsed();

  std::cout << "Unfused (separate kernels):     " << time << " ms" << std::endl;
}

void test_fused_activation(cublasHandle_t handle, int M, int K, int N,
                           __nv_bfloat16* d_A, __nv_bfloat16* d_B,
                           __nv_bfloat16* d_C, __nv_bfloat16* d_AB,
                           __nv_bfloat16* d_AC, __nv_bfloat16* d_output) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  Timer timer;

  // Step 1: AB = A * B (gate projection)
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            d_B, CUDA_R_16BF, N, d_A, CUDA_R_16BF, K, &beta,
                            d_AB, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // Step 2: AC = A * C (up projection)
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            d_C, CUDA_R_16BF, N, d_A, CUDA_R_16BF, K, &beta,
                            d_AC, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // Step 3: Fused gate * silu(up)
  int total_elements = M * N;
  int blockSize = 256;
  int numBlocks = (total_elements + blockSize - 1) / blockSize;
  fused_gated_silu_kernel_bf16<<<numBlocks, blockSize>>>(d_AB, d_AC, d_output,
                                                         total_elements);

  CHECK_CUDA(cudaDeviceSynchronize());
  float time = timer.elapsed();

  std::cout << "Fused activation only:           " << time << " ms"
            << std::endl;
}

void test_fused_vectorized(cublasHandle_t handle, int M, int K, int N,
                           __nv_bfloat16* d_A, __nv_bfloat16* d_B,
                           __nv_bfloat16* d_C, __nv_bfloat16* d_AB,
                           __nv_bfloat16* d_AC, __nv_bfloat16* d_output) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  Timer timer;

  // Step 1: AB = A * B (gate projection)
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            d_B, CUDA_R_16BF, N, d_A, CUDA_R_16BF, K, &beta,
                            d_AB, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // Step 2: AC = A * C (up projection)
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            d_C, CUDA_R_16BF, N, d_A, CUDA_R_16BF, K, &beta,
                            d_AC, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // Step 3: Fused vectorized gate * silu(up)
  int total_elements = M * N;
  int blockSize = 256;
  int numBlocks = ((total_elements / 2) + blockSize - 1) / blockSize;
  fused_gated_silu_vectorized_kernel_bf16<<<numBlocks, blockSize>>>(
      d_AB, d_AC, d_output, total_elements);

  CHECK_CUDA(cudaDeviceSynchronize());
  float time = timer.elapsed();

  std::cout << "Fused vectorized (bf162):        " << time << " ms"
            << std::endl;
}

void test_fused_vectorized8(cublasHandle_t handle, int M, int K, int N,
                            __nv_bfloat16* d_A, __nv_bfloat16* d_B,
                            __nv_bfloat16* d_C, __nv_bfloat16* d_AB,
                            __nv_bfloat16* d_AC, __nv_bfloat16* d_output) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  Timer timer;

  // Step 1: AB = A * B (gate projection)
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            d_B, CUDA_R_16BF, N, d_A, CUDA_R_16BF, K, &beta,
                            d_AB, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // Step 2: AC = A * C (up projection)
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            d_C, CUDA_R_16BF, N, d_A, CUDA_R_16BF, K, &beta,
                            d_AC, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // Step 3: Fused vectorized8 gate * silu(up)
  int total_elements = M * N;
  int blockSize = 256;
  int numBlocks = ((total_elements / 8) + blockSize - 1) / blockSize;
  fused_gated_silu_vectorized8_kernel_bf16<<<numBlocks, blockSize>>>(
      d_AB, d_AC, d_output, total_elements);

  CHECK_CUDA(cudaDeviceSynchronize());
  float time = timer.elapsed();

  std::cout << "Fused vectorized8 (4x bf162):    " << time << " ms"
            << std::endl;
}

void test_fully_fused(cublasHandle_t handle, int M, int K, int N,
                      __nv_bfloat16* d_A, __nv_bfloat16* d_B,
                      __nv_bfloat16* d_C, __nv_bfloat16* d_AB,
                      __nv_bfloat16* d_AC, __nv_bfloat16* d_output) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  Timer timer;

  // Step 1: AB = A * B (gate projection)
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            d_B, CUDA_R_16BF, N, d_A, CUDA_R_16BF, K, &beta,
                            d_AB, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // Step 2: AC = A * C (up projection)
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            d_C, CUDA_R_16BF, N, d_A, CUDA_R_16BF, K, &beta,
                            d_AC, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // Step 3: Fully fused 2D kernel
  dim3 blockSize(16, 16);
  dim3 numBlocks((N + blockSize.x - 1) / blockSize.x,
                 (M + blockSize.y - 1) / blockSize.y);
  fused_gemm_gated_silu_kernel_bf16<<<numBlocks, blockSize>>>(d_AB, d_AC,
                                                              d_output, M, N);

  CHECK_CUDA(cudaDeviceSynchronize());
  float time = timer.elapsed();

  std::cout << "Fully fused 2D kernel:           " << time << " ms"
            << std::endl;
}

void verify_results(__nv_bfloat16* d_result1, __nv_bfloat16* d_result2,
                    int size) {
  float* h_result1 = new float[size];
  float* h_result2 = new float[size];
  __nv_bfloat16* h_bf16_1 = new __nv_bfloat16[size];
  __nv_bfloat16* h_bf16_2 = new __nv_bfloat16[size];

  CHECK_CUDA(cudaMemcpy(h_bf16_1, d_result1, size * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_bf16_2, d_result2, size * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));

  for (int i = 0; i < size; i++) {
    h_result1[i] = __bfloat162float(h_bf16_1[i]);
    h_result2[i] = __bfloat162float(h_bf16_2[i]);
  }

  float max_abs_diff = 0.0f;
  float max_rel_diff = 0.0f;
  float avg_abs_diff = 0.0f;
  float avg_rel_diff = 0.0f;
  int mismatch_count = 0;

  for (int i = 0; i < size; i++) {
    float abs_diff = fabs(h_result1[i] - h_result2[i]);
    float rel_diff = 0.0f;

    if (fabs(h_result1[i]) > 1e-6f) {
      rel_diff = abs_diff / fabs(h_result1[i]);
    }

    max_abs_diff = fmax(max_abs_diff, abs_diff);
    max_rel_diff = fmax(max_rel_diff, rel_diff);
    avg_abs_diff += abs_diff;
    avg_rel_diff += rel_diff;

    // Count mismatches (considering BF16 precision ~0.01 or 1%)
    if (rel_diff > 0.02f && abs_diff > 0.01f) {
      mismatch_count++;
    }
  }

  avg_abs_diff /= size;
  avg_rel_diff /= size;

  std::cout << "  Max abs: " << max_abs_diff << ", Avg abs: " << avg_abs_diff
            << ", Max rel: " << (max_rel_diff * 100.0f) << "%"
            << ", Avg rel: " << (avg_rel_diff * 100.0f) << "%";

  if (mismatch_count > 0) {
    std::cout << " ⚠ Mismatches: " << mismatch_count << "/" << size;
  } else {
    std::cout << " ✓";
  }
  std::cout << std::endl;

  delete[] h_result1;
  delete[] h_result2;
  delete[] h_bf16_1;
  delete[] h_bf16_2;
}

// Optimized: Launch GEMMs back-to-back then fuse activation (minimal gap)
void test_fused_minimal_gap(cublasHandle_t handle, int M, int K, int N,
                            __nv_bfloat16* d_A, __nv_bfloat16* d_B,
                            __nv_bfloat16* d_C, __nv_bfloat16* d_AB,
                            __nv_bfloat16* d_AC, __nv_bfloat16* d_output) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  cudaStream_t stream1, stream2;
  CHECK_CUDA(cudaStreamCreate(&stream1));
  CHECK_CUDA(cudaStreamCreate(&stream2));

  Timer timer;

  // Launch both GEMMs concurrently in different streams
  CHECK_CUBLAS(cublasSetStream(handle, stream1));
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            d_B, CUDA_R_16BF, N, d_A, CUDA_R_16BF, K, &beta,
                            d_AB, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  CHECK_CUBLAS(cublasSetStream(handle, stream2));
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            d_C, CUDA_R_16BF, N, d_A, CUDA_R_16BF, K, &beta,
                            d_AC, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // Wait for both GEMMs
  CHECK_CUDA(cudaStreamSynchronize(stream1));
  CHECK_CUDA(cudaStreamSynchronize(stream2));

  // Fuse activation immediately
  int total_elements = M * N;
  int blockSize = 256;
  int numBlocks = ((total_elements / 8) + blockSize - 1) / blockSize;
  fused_gated_silu_vectorized8_kernel_bf16<<<numBlocks, blockSize>>>(
      d_AB, d_AC, d_output, total_elements);

  CHECK_CUDA(cudaDeviceSynchronize());
  float time = timer.elapsed();

  std::cout << "Fused + concurrent GEMMs:        " << time << " ms"
            << std::endl;

  CHECK_CUDA(cudaStreamDestroy(stream1));
  CHECK_CUDA(cudaStreamDestroy(stream2));
  CHECK_CUBLAS(cublasSetStream(handle, nullptr));
}

int main() {
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

  std::cout << "========================================" << std::endl;
  std::cout << "Fused Gated SiLU Projection (BF16)" << std::endl;
  std::cout << "Operation: AB * silu(AC)" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Device: " << prop.name << std::endl;
  std::cout << "Compute Capability: " << prop.major << "." << prop.minor
            << std::endl;
  std::cout << "========================================" << std::endl;

  // Test configurations (typical transformer sizes)
  struct Config {
    int M, K, N;
    const char* name;
  };

  Config configs[] = {
      {1, 4096, 1536, "LLaMA-7B FFN (batch=1024)"},
      {2048, 4096, 11008, "LLaMA-7B FFN (batch=2048)"},
      {512, 5120, 13824, "LLaMA-13B FFN (batch=512)"},
      {4096, 4096, 11008, "LLaMA-7B FFN (batch=4096)"},
  };

  for (const auto& config : configs) {
    int M = config.M;  // Batch size * sequence length
    int K = config.K;  // Hidden dimension
    int N = config.N;  // Intermediate dimension

    std::cout << "\n" << config.name << std::endl;
    std::cout << "Matrix sizes: A[" << M << "x" << K << "], " << "B[" << K
              << "x" << N << "], " << "C[" << K << "x" << N << "]" << std::endl;
    std::cout << "Output size: [" << M << "x" << N << "]" << std::endl;
    std::cout << "Memory per matrix (BF16): A="
              << (M * K * 2) / (1024.0 * 1024.0) << " MB, "
              << "B/C=" << (K * N * 2) / (1024.0 * 1024.0) << " MB, "
              << "Output=" << (M * N * 2) / (1024.0 * 1024.0) << " MB"
              << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B, *d_C;
    __nv_bfloat16 *d_AB, *d_AC, *d_silu_AC;
    __nv_bfloat16 *d_output1, *d_output2, *d_output3, *d_output4, *d_output5,
        *d_output6, *d_output7;

    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_C, K * N * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_AB, M * N * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_AC, M * N * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_silu_AC, M * N * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_output1, M * N * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_output2, M * N * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_output3, M * N * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_output4, M * N * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_output5, M * N * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_output6, M * N * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_output7, M * N * sizeof(__nv_bfloat16)));

    // Initialize with random data
    float* h_data = new float[std::max({M * K, K * N, M * N})];
    __nv_bfloat16* h_bf16_data =
        new __nv_bfloat16[std::max({M * K, K * N, M * N})];

    for (int i = 0; i < M * K; i++) {
      h_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
      h_bf16_data[i] = __float2bfloat16(h_data[i]);
    }
    CHECK_CUDA(cudaMemcpy(d_A, h_bf16_data, M * K * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));

    for (int i = 0; i < K * N; i++) {
      h_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
      h_bf16_data[i] = __float2bfloat16(h_data[i]);
    }
    CHECK_CUDA(cudaMemcpy(d_B, h_bf16_data, K * N * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));

    for (int i = 0; i < K * N; i++) {
      h_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
      h_bf16_data[i] = __float2bfloat16(h_data[i]);
    }
    CHECK_CUDA(cudaMemcpy(d_C, h_bf16_data, K * N * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));

    delete[] h_data;
    delete[] h_bf16_data;

    // Run tests
    test_unfused(handle, M, K, N, d_A, d_B, d_C, d_AB, d_AC, d_silu_AC,
                 d_output1);
    test_fused_activation(handle, M, K, N, d_A, d_B, d_C, d_AB, d_AC,
                          d_output2);
    test_fused_vectorized(handle, M, K, N, d_A, d_B, d_C, d_AB, d_AC,
                          d_output3);
    test_fused_vectorized8(handle, M, K, N, d_A, d_B, d_C, d_AB, d_AC,
                           d_output4);
    test_fully_fused(handle, M, K, N, d_A, d_B, d_C, d_AB, d_AC, d_output5);
    test_fully_fused_tiled(handle, M, K, N, d_A, d_B, d_C, d_output6);
    test_fused_minimal_gap(handle, M, K, N, d_A, d_B, d_C, d_AB, d_AC,
                           d_output7);

    std::cout << "\nVerification (vs unfused baseline):" << std::endl;
    std::cout << "Fused activation:         ";
    verify_results(d_output1, d_output2, M * N);
    std::cout << "Fused vectorized (bf162): ";
    verify_results(d_output1, d_output3, M * N);
    std::cout << "Fused vectorized8:        ";
    verify_results(d_output1, d_output4, M * N);
    std::cout << "Fully fused 2D:           ";
    verify_results(d_output1, d_output5, M * N);
    std::cout << "Fully fused tiled:        ";
    verify_results(d_output1, d_output6, M * N);
    std::cout << "Fully fused double-buf:   ";
    verify_results(d_output1, d_output7, M * N);

    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_AB));
    CHECK_CUDA(cudaFree(d_AC));
    CHECK_CUDA(cudaFree(d_silu_AC));
    CHECK_CUDA(cudaFree(d_output1));
    CHECK_CUDA(cudaFree(d_output2));
    CHECK_CUDA(cudaFree(d_output3));
    CHECK_CUDA(cudaFree(d_output4));
    CHECK_CUDA(cudaFree(d_output5));
    CHECK_CUDA(cudaFree(d_output6));
    CHECK_CUDA(cudaFree(d_output7));
  }

  std::cout << "\n========================================" << std::endl;
  std::cout << "Performance Summary" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "BF16 Benefits:" << std::endl;
  std::cout << "  - 2x less memory bandwidth vs FP32" << std::endl;
  std::cout << "  - 2x higher Tensor Core throughput" << std::endl;
  std::cout << "  - Better cache utilization" << std::endl;
  std::cout << "\nFusion Benefits:" << std::endl;
  std::cout << "  - Eliminates intermediate memory writes" << std::endl;
  std::cout << "  - Reduces kernel launch overhead" << std::endl;
  std::cout << "  - Better L2 cache utilization" << std::endl;
  std::cout << "  - Expected speedup: 1.3-1.8x over unfused" << std::endl;
  std::cout << "========================================" << std::endl;

  CHECK_CUBLAS(cublasDestroy(handle));

  return 0;
}
