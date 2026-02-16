// Fully fused CUTLASS GEMM kernel where the same A matrix is used with two
// different B matrices, with explicit tiling, iterators, and MMA, optimized for
// tensor cores, and merging C1 and C2 with element-wise multiplication.

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/warp/mma_tensor_op.h>
#include <cutlass/gemm/warp/mma_tensor_op_tile_iterator.h>
#include <cutlass/gemm/threadblock/default_mma_core.h>
#include <cutlass/gemm/threadblock/mma_base.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/matrix_shape.h>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/mma.h>
#include <cutlass/arch/arch.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

int B = 128, K = 2048, Ng = 768, Nu = 768, No = 2048;

// Type definitions
using ElementInput = cutlass::bfloat16_t;
using ElementOutput = cutlass::bfloat16_t;
using ElementInputccumulator = float;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
using MMAOp = cutlass::arch::OpClassTensorOp;
using Arch = cutlass::arch::Sm80;

using WarpMatrixShape = cutlass::MatrixShape<WarpShape::kM, WarpShape::kN>;

using Operator =
    cutlass::arch::Mma<cutlass::gemm::GemmShape<16, 8, 16>, 32, ElementInput,
                       LayoutA, ElementInput, LayoutB, ElementInputccumulator,
                       LayoutC, cutlass::arch::OpMultiplyAdd>;

using WrapMmaPolicy =
    cutlass::gemm::warp::MmaTensorOpPolicy<Operator, WarpMatrixShape>;

using WarpMma = cutlass::gemm::warp::MmaTensorOp<
    WarpShape, ElementInput, LayoutA, ElementInput, LayoutB,
    ElementInputccumulator, LayoutC, WrapMmaPolicy, 1, true>;

__global__ void fused_gemm_kernel(ElementInput const* A, ElementInput const* B1,
                                  ElementInput const* B2, ElementOutput* C,
                                  int M, int N, int K) {
  extern __shared__ char shared_mem[];
  ElementInput* smem_A = reinterpret_cast<ElementInput*>(shared_mem);
  ElementInput* smem_B1 = reinterpret_cast<ElementInput*>(
      shared_mem +
      ThreadblockShape::kM * ThreadblockShape::kK * sizeof(ElementInput));
  ElementInput* smem_B2 = smem_B1 + ThreadblockShape::kK * ThreadblockShape::kN;

  WarpMma mma;
  typename WarpMma::FragmentC accum1;
  typename WarpMma::FragmentC accum2;
  accum1.clear();
  accum2.clear();

  using IteratorA = typename WarpMma::IteratorA;
  using IteratorB = typename WarpMma::IteratorB;

  for (int tile_k = 0;
       tile_k < (K + ThreadblockShape::kK - 1) / ThreadblockShape::kK;
       ++tile_k) {
    int lane = threadIdx.x;
    int num_threads = blockDim.x;

    for (int i = lane; i < ThreadblockShape::kM * ThreadblockShape::kK;
         i += num_threads) {
      int row = i / ThreadblockShape::kK;
      int col = i % ThreadblockShape::kK;
      int global_row = blockIdx.y * ThreadblockShape::kM + row;
      int global_col = tile_k * ThreadblockShape::kK + col;
      smem_A[i] = (global_row < M && global_col < K)
                      ? A[global_row * K + global_col]
                      : ElementInput(0);
    }

    for (int i = lane; i < ThreadblockShape::kK * ThreadblockShape::kN;
         i += num_threads) {
      int row = i / ThreadblockShape::kN;
      int col = i % ThreadblockShape::kN;
      int global_row = tile_k * ThreadblockShape::kK + row;
      int global_col = blockIdx.x * ThreadblockShape::kN + col;
      smem_B1[i] = (global_row < K && global_col < N)
                       ? B1[global_row * N + global_col]
                       : ElementInput(0);
      smem_B2[i] = (global_row < K && global_col < N)
                       ? B2[global_row * N + global_col]
                       : ElementInput(0);
    }

    __syncthreads();

    typename WarpMma::FragmentA frag_A;
    typename WarpMma::FragmentB frag_B1;
    typename WarpMma::FragmentB frag_B2;

#pragma unroll
    for (int i = 0; i < frag_A.size(); ++i) frag_A[i] = smem_A[i];
#pragma unroll
    for (int i = 0; i < frag_B1.size(); ++i) frag_B1[i] = smem_B1[i];
#pragma unroll
    for (int i = 0; i < frag_B2.size(); ++i) frag_B2[i] = smem_B2[i];

    // IteratorA iter_A({smem_A, ThreadblockShape::kK}, lane);
    // IteratorB iter_B1({smem_B1, ThreadblockShape::kN}, lane);
    // IteratorB iter_B2({smem_B2, ThreadblockShape::kN}, lane);

    // iter_A.load(frag_A);
    // iter_B1.load(frag_B1);
    // iter_B2.load(frag_B2);

    mma(accum1, frag_A, frag_B1, accum1);
    mma(accum2, frag_A, frag_B2, accum2);
    __syncthreads();
  }

  int c_row = blockIdx.y * ThreadblockShape::kM + threadIdx.y * WarpShape::kM;
  int c_col = blockIdx.x * ThreadblockShape::kN + threadIdx.z * WarpShape::kN;
  for (int i = 0; i < WarpShape::kM; ++i) {
    for (int j = 0; j < WarpShape::kN; ++j) {
      int global_row = c_row + i;
      int global_col = c_col + j;
      if (global_row < M && global_col < N) {
        size_t idx = i * WarpShape::kN + j;
        float silu = accum1[idx] / (1.0f + expf(-accum1[idx]));
        C[global_row * N + global_col] = ElementOutput(silu * accum2[idx]);
      }
    }
  }
}

template <typename Element, typename Layout>
void fill_host_tensor(cutlass::HostTensor<Element, Layout>& tensor,
                      Element value, float scale = 1.0f, float offset = 0.0f) {
  cutlass::reference::host::TensorFillRandomUniform(tensor.host_view(), value,
                                                    scale, offset);
}

int main() {
  cutlass::HostTensor<ElementInput, LayoutA> X({B, K});
  cutlass::HostTensor<ElementInput, LayoutB> Wg({Ng, K});
  cutlass::HostTensor<ElementInput, LayoutB> Wu({Nu, K});
  // cutlass::HostTensor<ElementInput, LayoutB> Wd({No, Nu});
  cutlass::HostTensor<ElementOutput, LayoutC> C({B, No});

  fill_host_tensor(X, ElementInput(1.0f), 1.0f, 0.0f);
  fill_host_tensor(Wg, ElementInput(1.0f), 1.0f, 0.0f);
  fill_host_tensor(Wu, ElementInput(1.0f), 1.0f, 0.0f);
  // fill_host_tensor(Wd, ElementInput(1.0f), 1.0f, 0.0f);

  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  dim3 grid((Ng + ThreadblockShape::kN - 1) / ThreadblockShape::kN,
            (B + ThreadblockShape::kM - 1) / ThreadblockShape::kM);
  dim3 block(32, 4, 1);  // 128 threads total
  size_t shared_mem_size =
      sizeof(ElementInput) * ThreadblockShape::kM * ThreadblockShape::kK +
      sizeof(ElementInput) * ThreadblockShape::kK * ThreadblockShape::kN * 2;
  fused_gemm_kernel<<<grid, block, shared_mem_size, stream>>>(
      X.device_data(), Wg.device_data(), Wu.device_data(), C.device_data(), B,
      No, K);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    return -1;
  }

  cudaStreamSynchronize(stream);
  C.sync_host();

  // Check results
  std::cout << "First 10 elements of output C:\n" << std::endl;
  for (int i = 0; i < 10; ++i) {
    std::cout << C.host_data()[i] << " ";
  }
  std::cout << std::endl;
}
