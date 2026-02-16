#include <cutlass/cutlass.h>
#include <cutlass/gemm/threadblock/default_mma_core.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/arch/mma.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/reference/host/gemm.h>
#include <cuda_runtime.h>
#include <iostream>

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

// Tile configurations
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

// MmaCore definition
using MmaCore = cutlass::gemm::threadblock::DefaultMmaCore<
    ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA, ElementB,
    LayoutB, ElementAccumulator, LayoutC, cutlass::arch::OpClassTensorOp, 1,
    cutlass::arch::OpMultiplyAdd, false, cutlass::arch::CacheOperation::Global,
    cutlass::arch::CacheOperation::Global>;

using Mma = typename MmaCore::Mma;
using FragmentC = typename MmaCore::FragmentC;
using SharedStorage = typename Mma::SharedStorage;

// CUDA Kernel
__global__ void mma_kernel(cutlass::TensorRef<ElementA, LayoutA> A,
                           cutlass::TensorRef<ElementB, LayoutB> B,
                           cutlass::TensorRef<ElementC, LayoutC> C, int M,
                           int N, int K) {
  extern __shared__ __align__(16) char shared_storage[];
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_storage);

  Mma mma(smem, threadIdx.x, threadIdx.y, threadIdx.z);

  // Threadblock tile offsets
  int block_row = blockIdx.y * ThreadblockShape::kM;
  int block_col = blockIdx.x * ThreadblockShape::kN;

  FragmentC accum;
  cutlass::arch::mma::clear(accum);  // Zero accumulator

  for (int k_tile = 0; k_tile < K; k_tile += ThreadblockShape::kK) {
    typename Mma::FragmentA fragA;
    typename Mma::FragmentB fragB;

    // Load A tile from global memory
    mma.load_a(fragA, A.data() + block_row * K + k_tile, A.stride(0), k_tile);
    // Load B tile from global memory
    mma.load_b(fragB, B.data() + k_tile * N + block_col, B.stride(0), k_tile);

    // Fused multiply-add
    mma(accum, fragA, fragB, accum);
  }

  // Write result C to global memory
  mma.store_c(C.data() + block_row * N + block_col, C.stride(0), accum);
}

// Host code
int main() {
  constexpr int M = 256, N = 256, K = 256;

  cutlass::HostTensor<ElementA, LayoutA> A({M, K});
  cutlass::HostTensor<ElementB, LayoutB> B({K, N});
  cutlass::HostTensor<ElementC, LayoutC> C({M, N});
  cutlass::HostTensor<ElementC, LayoutC> C_ref({M, N});

  cutlass::reference::host::TensorFill(A.host_view(), ElementA(1.0f));
  cutlass::reference::host::TensorFill(B.host_view(), ElementB(1.0f));

  A.sync_device();
  B.sync_device();

  cutlass::TensorRef<ElementA, LayoutA> A_ref(A.device_data(), A.layout());
  cutlass::TensorRef<ElementB, LayoutB> B_ref(B.device_data(), B.layout());
  cutlass::TensorRef<ElementC, LayoutC> C_ref_device(C.device_data(),
                                                     C.layout());

  // Kernel launch
  dim3 grid((N + ThreadblockShape::kN - 1) / ThreadblockShape::kN,
            (M + ThreadblockShape::kM - 1) / ThreadblockShape::kM);
  dim3 block(32, 4, 1);  // Typical for warp-level threading
  size_t smem_size = sizeof(SharedStorage);

  std::cout << "Launching kernel with grid = (" << grid.x << ", " << grid.y
            << "), block = (" << block.x << ", " << block.y << ")\n";
  mma_kernel<<<grid, block, smem_size>>>(A_ref, B_ref, C_ref_device, M, N, K);
  cudaDeviceSynchronize();

  // Copy result back
  C.sync_host();

  // Reference computation
  cutlass::reference::host::Gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC,
                                 LayoutC, ElementAccumulator>(
      {M, N, K}, ElementAccumulator(1.0f), A.host_ref(), B.host_ref(),
      ElementAccumulator(0.0f), C_ref.host_ref());

  // Validate result
  bool passed = true;
  for (int i = 0; i < M * N; ++i) {
    float diff =
        std::abs(float(C.host_data()[i]) - float(C_ref.host_data()[i]));
    if (diff > 1e-2) {
      std::cout << "Mismatch at " << i << ": " << float(C.host_data()[i])
                << " vs " << float(C_ref.host_data()[i]) << "\n";
      passed = false;
      break;
    }
  }

  if (passed) {
    std::cout << "GEMM passed!\n";
  } else {
    std::cout << "GEMM failed!\n";
  }

  return 0;
}
