// fused_silu_gemm_bf16.cu - Fused Tiled GEMM + SiLU with Shared Memory (CUDA
// Kernel + PyTorch Binding for BF16)

#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <ATen/cuda/CUDAContext.h>

#define TILE_DIM 32
#define NUM_THREADS 32

__device__ inline float silu(float x) { return x / (1.0f + __expf(-x)); }

inline void throwOnCudaError(cudaError_t error, const char* file, int line,
                             const char* function, const char* call) {
  if (error != cudaSuccess) {
    std::stringstream ss;
    ss << "CUDA error " << error << " at " << file << ":" << line
       << " in function " << function << ": " << cudaGetErrorString(error)
       << "\nCall: " << call;
    throw std::runtime_error(ss.str());
  }
};

inline void throwOnCublasError(cublasStatus_t error, const char* file, int line,
                               const char* function, const char* call) {
  if (error != CUBLAS_STATUS_SUCCESS) {
    std::stringstream ss;
    const char* errorString;
    switch (error) {
      case CUBLAS_STATUS_SUCCESS:
        errorString = "CUBLAS_STATUS_SUCCESS";
        break;
      case CUBLAS_STATUS_NOT_INITIALIZED:
        errorString = "CUBLAS_STATUS_NOT_INITIALIZED";
        break;
      case CUBLAS_STATUS_ALLOC_FAILED:
        errorString = "CUBLAS_STATUS_ALLOC_FAILED";
        break;
      case CUBLAS_STATUS_INVALID_VALUE:
        errorString = "CUBLAS_STATUS_INVALID_VALUE";
        break;
      case CUBLAS_STATUS_ARCH_MISMATCH:
        errorString = "CUBLAS_STATUS_ARCH_MISMATCH";
        break;
      case CUBLAS_STATUS_MAPPING_ERROR:
        errorString = "CUBLAS_STATUS_MAPPING_ERROR";
        break;
      case CUBLAS_STATUS_EXECUTION_FAILED:
        errorString = "CUBLAS_STATUS_EXECUTION_FAILED";
        break;
      case CUBLAS_STATUS_INTERNAL_ERROR:
        errorString = "CUBLAS_STATUS_INTERNAL_ERROR";
        break;
      case CUBLAS_STATUS_NOT_SUPPORTED:
        errorString = "CUBLAS_STATUS_NOT_SUPPORTED";
        break;
      case CUBLAS_STATUS_LICENSE_ERROR:
        errorString = "CUBLAS_STATUS_LICENSE_ERROR";
        break;
      default:
        errorString = "Unknown CUBLAS error";
    }
    ss << "CUBLAS error " << error << " (" << errorString << ") at " << file
       << ":" << line << " in function " << function << "\nCall: " << call;
    throw std::runtime_error(ss.str());
  }
}

#define CUDA_CHECK(call) \
  throwOnCudaError(call, __FILE__, __LINE__, __FUNCTION__, #call)

#define CUBLAS_CHECK(call) \
  throwOnCublasError(call, __FILE__, __LINE__, __FUNCTION__, #call)

torch::Tensor linear_cublaslt_bf16(torch::Tensor input, torch::Tensor weight) {
  TORCH_CHECK(input.is_cuda(), "Input must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA");
  // TORCH_CHECK(bias.is_cuda(), "Bias must be CUDA");

  auto m = input.size(0);
  auto k = input.size(1);
  auto n = weight.size(0);  // weight: [out_features, in_features]

  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "Expected BF16 input");
  TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "Expected BF16 weight");
  // TORCH_CHECK(bias.scalar_type() == at::kFloat, "Bias should be float32");

  auto output = torch::empty({m, n}, input.options().dtype(at::kBFloat16));

  float alpha = 1.0f, beta = 0.0f;

  cudaSetDevice(0);

  cublasLtHandle_t handle;
  cublasLtCreate(&handle);

  cublasLtMatmulDesc_t op_desc;
  cublasLtMatrixLayout_t a_desc, b_desc, c_desc;

  cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_T;

  cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa,
                                 sizeof(transa));
  cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb,
                                 sizeof(transb));

  // // Epilogue with bias
  // cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
  // &bias.data_ptr<float>(), sizeof(void*)); cublasLtEpilogue_t epilogue =
  // CUBLASLT_EPILOGUE_BIAS; cublasLtMatmulDescSetAttribute(op_desc,
  // CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));

  // Layouts using BF16
  cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16BF, m, k, k);
  cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16BF, k, n, n);
  cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16BF, m, n, n);

  CUBLAS_CHECK(cublasLtMatmul(
      handle, op_desc, &alpha, input.data_ptr<at::BFloat16>(), a_desc,
      weight.data_ptr<at::BFloat16>(), b_desc, &beta,
      output.data_ptr<at::BFloat16>(), c_desc, output.data_ptr<at::BFloat16>(),
      c_desc, nullptr, nullptr, 0, 0));

  cublasLtMatmulDescDestroy(op_desc);
  cublasLtMatrixLayoutDestroy(a_desc);
  cublasLtMatrixLayoutDestroy(b_desc);
  cublasLtMatrixLayoutDestroy(c_desc);
  cublasLtDestroy(handle);

  cudaDeviceSynchronize();

  return output;
}

__global__ void fused_silu_gemm_kernel_naive(
    const __nv_bfloat16* __restrict__ hidden,     // (B, D)
    const __nv_bfloat16* __restrict__ gate_proj,  // (G, D)
    __nv_bfloat16* __restrict__ output,           // (B, G)
    int B, int D, int G) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= B || col >= G) return;

  float acc = 0.0f;
  for (int k = 0; k < D; ++k) {
    float h = __bfloat162float(hidden[row * D + k]);     // hidden (B, D)
    float g = __bfloat162float(gate_proj[col * D + k]);  // gate_proj (H, D)
    acc += h * g;
  }

  // acc = silu(acc);
  output[row * G + col] = __float2bfloat16(acc);
  // output[row * G + col] += __float2bfloat16(1.0f);
}

__global__ void fused_silu_gemm_kernel(
    const __nv_bfloat16* __restrict__ hidden,     // (B, D)
    const __nv_bfloat16* __restrict__ gate_proj,  // (G, D)
    __nv_bfloat16* __restrict__ output,           // (B, G)
    int B, int D, int G) {
  int row = blockIdx.y * TILE_DIM + threadIdx.y;
  int col = blockIdx.x * TILE_DIM + threadIdx.x;

  if (row >= B || col >= G) return;

  float acc = 0.0f;

  __shared__ __nv_bfloat16 tileA[TILE_DIM][TILE_DIM];
  __shared__ __nv_bfloat16 tileB[TILE_DIM][TILE_DIM];

  for (int tile_k = 0; tile_k < D; tile_k += TILE_DIM) {
    int tiled_row = row;
    int tiled_col = col;

    if (tiled_row < B && (tile_k + threadIdx.x) < D)
      tileA[threadIdx.y][threadIdx.x] =
          hidden[tiled_row * D + tile_k + threadIdx.x];
    else
      tileA[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);

    if (tiled_col < G && (tile_k + threadIdx.y) < D)
      tileB[threadIdx.y][threadIdx.x] =
          gate_proj[tiled_col * D + tile_k + threadIdx.y];
    else
      tileB[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);

    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE_DIM; ++k) {
      float h = __bfloat162float(tileA[threadIdx.y][k]);
      float g = __bfloat162float(tileB[k][threadIdx.x]);
      acc += h * g;
    }

    __syncthreads();
  }

  // acc = silu(acc);
  output[row * G + col] = __float2bfloat16(acc);
}

// PyTorch wrapper
at::Tensor fused_silu_gemm(torch::Tensor hidden, torch::Tensor gate_proj) {
  TORCH_CHECK(hidden.dim() == 2 && gate_proj.dim() == 2, "Expected 2D tensors");
  TORCH_CHECK(hidden.size(1) == gate_proj.size(1),
              "Inner dimensions must match");

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);  // 0 = current device

  int maxThreadsPerBlock = prop.maxThreadsPerBlock;
  printf("Max threads per block: %d\n", maxThreadsPerBlock);

  int B = hidden.size(0);
  int D = hidden.size(1);
  int G = gate_proj.size(0);

  auto options = hidden.options().dtype(torch::kBFloat16);
  auto output = torch::empty({B, G}, options).contiguous();
  // auto output = torch::zeros({B, G}, options);

  dim3 threads(TILE_DIM, TILE_DIM);
  dim3 blocks((G + TILE_DIM - 1) / TILE_DIM, (B + TILE_DIM - 1) / TILE_DIM);

  // cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // int shm_size = D * sizeof(__nv_bfloat16) * TILE_DIM * TILE_DIM;
  fused_silu_gemm_kernel_naive<<<blocks, threads, 0>>>(
      reinterpret_cast<__nv_bfloat16*>(hidden.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(gate_proj.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()), B, D,
      G);
  {
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
      std::cerr << "Kernel launch failed: " << cudaGetErrorString(launch_err)
                << std::endl;
      return output;
    }
  }

  // cudaStreamSynchronize(stream);
  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
  }

  return output;
}

// at::Tensor fused_silu_gemm_cublas(torch::Tensor hidden, torch::Tensor
// gate_proj) {
//     TORCH_CHECK(hidden.dim() == 2 && gate_proj.dim() == 2, "Expected 2D
//     tensors"); TORCH_CHECK(hidden.size(1) == gate_proj.size(1), "Inner
//     dimensions must match");

//     int B = hidden.size(0);
//     int D = hidden.size(1);
//     int G = gate_proj.size(0);

//     auto output = torch::empty({B, G}, hidden.options());

//     const __nv_bfloat16* Amat = reinterpret_cast<const
//     __nv_bfloat16*>(hidden.data_ptr<at::BFloat16>()); const __nv_bfloat16*
//     Bmat = reinterpret_cast<const
//     __nv_bfloat16*>(gate_proj.data_ptr<at::BFloat16>());
//     __nv_bfloat16* C =
//     reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());

//     // Create cuBLAS handle and stream
//     cudaSetDevice(0);
//     cublasHandle_t handle;
//     cublasCreate(&handle);
//     cudaStream_t stream;
//     cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

//     // int device;
//     // cudaGetDevice(&device);
//     // cudaDeviceProp props;
//     // cudaGetDeviceProperties(&props, device);
//     // TORCH_CHECK(props.major >= 8, "BF16 is only supported on Ampere or
//     newer GPUs");

//     // TORCH_CHECK(hidden.stride(1) == 1, "hidden must be contiguous in last
//     dim");
//     // TORCH_CHECK(gate_proj.stride(1) == 1, "gate_proj must be contiguous in
//     last dim");
//     // TORCH_CHECK(output.stride(1) == 1, "output must be contiguous in last
//     dim");

//     float alpha = 1.0f;
//     float beta = 0.0f;

//     // Change GEMM to compute C directly in row-major layout:
//     // C = A @ B^T, row-major → flip to C^T = B @ A^T, then transpose the
//     result
//     // Instead, use CUBLAS_OP_T, CUBLAS_OP_N and swap A, B

//     cublasSetStream(handle, stream);
//     cudaStreamSynchronize(stream);

//     auto start = std::chrono::high_resolution_clock::now();
//     cublasStatus_t status = cublasGemmEx(
//         handle,
//         CUBLAS_OP_T, CUBLAS_OP_N,
//         B, G, D, // m, n, k (C = B x G)
//         &alpha,
//         Amat, CUDA_R_16BF, D,  // A.T → (D x B), lda = D
//         Bmat, CUDA_R_16BF, D, // B → (D x G), ldb = D
//         &beta,
//         C, CUDA_R_16BF, G, // C → (B x G), ldc = G
//         CUDA_R_32F,
//         CUBLAS_GEMM_DEFAULT
//     );

//     TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublasGemmEx failed");

//     // // Launch SiLU kernel on result
//     // dim3 threads(16, 16);
//     // dim3 blocks((G + 15) / 16, (B + 15) / 16);
//     // apply_silu_bf16<<<blocks, threads, 0, stream>>>(C, B, G);
//     cudaStreamSynchronize(stream);
//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> elapsed = end - start;
//     std::cout << "Elapsed time: " << elapsed.count() << " s\n";
//     // # close cublas handle
//     cublasDestroy(handle);
//     //  close stream
//     cudaStreamDestroy(stream);
//     return output;
// }

at::Tensor fused_silu_gemm_cublas(torch::Tensor hidden,
                                  torch::Tensor gate_proj) {
  TORCH_CHECK(hidden.dim() == 2 && gate_proj.dim() == 2, "Expected 2D tensors");
  TORCH_CHECK(hidden.size(1) == gate_proj.size(1),
              "Inner dimensions must match");

  int B = hidden.size(0);
  int D = hidden.size(1);
  int G = gate_proj.size(0);

  auto output = torch::empty({B, G}, hidden.options());

  const __nv_bfloat16* A =
      reinterpret_cast<const __nv_bfloat16*>(hidden.data_ptr<at::BFloat16>());
  const __nv_bfloat16* Bmat = reinterpret_cast<const __nv_bfloat16*>(
      gate_proj.data_ptr<at::BFloat16>());
  __nv_bfloat16* C =
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());

  // Create cuBLAS handle and stream
  cublasHandle_t handle;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cublasCreate(&handle);
  cublasSetStream(handle, stream);

  float alpha = 1.0f;
  float beta = 0.0f;

  // Change GEMM to compute C directly in row-major layout:
  // C = A @ B^T, row-major → flip to C^T = B @ A^T, then transpose the result
  // Instead, use CUBLAS_OP_T, CUBLAS_OP_N and swap A, B
  auto start = std::chrono::high_resolution_clock::now();
  cublasStatus_t status =
      cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, G, B, D,  // m, n, k
                   &alpha, Bmat, CUDA_R_16BF, G,               // lda = G
                   A, CUDA_R_16BF, D,                          // ldb = D
                   &beta, C, CUDA_R_16BF, G,                   // ldc = G
                   CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublasGemmEx failed");

  // dim3 threads(16, 16);
  // dim3 blocks((G + 15) / 16, (B + 15) / 16);
  // apply_silu_bf16<<<blocks, threads, 0, stream>>>(C, B, G);

  cudaStreamSynchronize(stream);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time: " << elapsed.count() << " s\n";
  cublasDestroy(handle);
  cudaStreamDestroy(stream);

  return output;
}

at::Tensor expert_fused_mlp(torch::Tensor input, torch::Tensor w1,
                            torch::Tensor w3, torch::Tensor w2);
at::Tensor expert_fused_mlp_batched(torch::Tensor input, torch::Tensor w1_list,
                                    torch::Tensor w3_list,
                                    torch::Tensor w2_list,
                                    torch::Tensor expert_ids);

// ============================================================================
// Expert GEMM implementations from expert_gemm.cu
// ============================================================================

__device__ __forceinline__ float expert_silu(float x) {
  return x / (1.0f + expf(-x));
}

__global__ void expert_silu_multiply_kernel(
    const __nv_bfloat16* __restrict__ gate_output,
    const __nv_bfloat16* __restrict__ up_output,
    __nv_bfloat16* __restrict__ output, int num_tokens, int inter_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_tokens * inter_dim) {
    float gate = __bfloat162float(gate_output[idx]);
    float up = __bfloat162float(up_output[idx]);
    float result = expert_silu(gate) * up;
    output[idx] = __float2bfloat16(result);
  }
}

at::Tensor expert_fused_mlp(torch::Tensor input, torch::Tensor w1,
                            torch::Tensor w3, torch::Tensor w2) {
  TORCH_CHECK(input.is_cuda(), "Input must be CUDA");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "Expected BF16 input");
  TORCH_CHECK(w1.scalar_type() == at::kBFloat16, "Expected BF16 w1");
  TORCH_CHECK(w3.scalar_type() == at::kBFloat16, "Expected BF16 w3");
  TORCH_CHECK(w2.scalar_type() == at::kBFloat16, "Expected BF16 w2");

  const int num_tokens = input.size(0);
  const int D = input.size(1);
  const int inter_dim = w1.size(0);

  auto gate_output = torch::empty({num_tokens, inter_dim}, input.options());
  auto up_output = torch::empty({num_tokens, inter_dim}, input.options());
  auto gate_up_fused = torch::empty({num_tokens, inter_dim}, input.options());
  auto output = torch::empty({num_tokens, D}, input.options());

  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0f, beta = 0.0f;

  // gate_output = input @ w1.T
  cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, inter_dim, num_tokens, D,
               &alpha, w1.data_ptr<at::BFloat16>(), CUDA_R_16BF, D,
               input.data_ptr<at::BFloat16>(), CUDA_R_16BF, D, &beta,
               gate_output.data_ptr<at::BFloat16>(), CUDA_R_16BF, inter_dim,
               CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  // up_output = input @ w3.T
  cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, inter_dim, num_tokens, D,
               &alpha, w3.data_ptr<at::BFloat16>(), CUDA_R_16BF, D,
               input.data_ptr<at::BFloat16>(), CUDA_R_16BF, D, &beta,
               up_output.data_ptr<at::BFloat16>(), CUDA_R_16BF, inter_dim,
               CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  cublasDestroy(handle);

  // SiLU(gate_output) * up_output
  const int threads = 256;
  const int blocks = (num_tokens * inter_dim + threads - 1) / threads;
  expert_silu_multiply_kernel<<<blocks, threads>>>(
      reinterpret_cast<__nv_bfloat16*>(gate_output.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(up_output.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(gate_up_fused.data_ptr<at::BFloat16>()),
      num_tokens, inter_dim);

  // output = gate_up_fused @ w2.T
  cublasCreate(&handle);
  cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, D, num_tokens, inter_dim,
               &alpha, w2.data_ptr<at::BFloat16>(), CUDA_R_16BF, inter_dim,
               gate_up_fused.data_ptr<at::BFloat16>(), CUDA_R_16BF, inter_dim,
               &beta, output.data_ptr<at::BFloat16>(), CUDA_R_16BF, D,
               CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cublasDestroy(handle);
  cudaDeviceSynchronize();

  return output;
}

at::Tensor expert_fused_mlp_batched(torch::Tensor input, torch::Tensor w1_list,
                                    torch::Tensor w3_list,
                                    torch::Tensor w2_list,
                                    torch::Tensor expert_ids) {
  TORCH_CHECK(input.is_cuda(), "Input must be CUDA");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "Expected BF16");

  const int num_tokens = input.size(0);
  const int D = input.size(1);
  const int inter_dim = w1_list.size(1);
  const int num_experts = w1_list.size(0);

  auto output = torch::empty({num_tokens, D}, input.options());

  for (int expert_id = 0; expert_id < num_experts; expert_id++) {
    auto mask = (expert_ids == expert_id);
    auto token_indices = mask.nonzero().squeeze();
    if (token_indices.numel() == 0) continue;

    auto expert_input = input.index_select(0, token_indices);
    auto w1 = w1_list[expert_id];
    auto w3 = w3_list[expert_id];
    auto w2 = w2_list[expert_id];
    auto expert_output = expert_fused_mlp(expert_input, w1, w3, w2);
    output.index_copy_(0, token_indices, expert_output);
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_cublaslt_bf16", &linear_cublaslt_bf16,
        "Linear layer with cuBLAS LT (BF16)");
  m.def("fused_silu_gemm", &fused_silu_gemm, "Fused GEMM + SiLU (CUDA, BF16)");
  m.def("fused_silu_gemm_cublas", &fused_silu_gemm_cublas,
        "Fused GEMM + SiLU (CUDA, BF16, CUBLAS)");
  m.def("expert_fused_mlp", &expert_fused_mlp,
        "Fused Expert MLP (gate+up fused, then down) - cuBLAS version");
  m.def("expert_fused_mlp_batched", &expert_fused_mlp_batched,
        "Fused Expert MLP for batched experts");
}
