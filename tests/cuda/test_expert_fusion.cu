#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_silu.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/numeric_types.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <torch/torch.h>
#include <torch/extension.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cassert>

int B = 128, K = 2048, Ng = 768, Nu = 768, No = 2048;

// Type definitions
using ElementInput = cutlass::bfloat16_t;
using ElementOutput = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

// Epilogues
using EpilogueSiLU = cutlass::epilogue::thread::LinearCombinationSilu<
    ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator, ElementCompute,
    cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;

using EpilogueLinear = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator, ElementCompute,
    cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;

// GEMM Definitions
using Gemm1 = cutlass::gemm::device::Gemm<
    ElementInput, LayoutA, ElementInput, LayoutB, ElementOutput, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    ThreadblockShape, WarpShape, InstructionShape, EpilogueSiLU>;

using Gemm2 = cutlass::gemm::device::Gemm<
    ElementInput, LayoutA, ElementInput, LayoutB, ElementOutput, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    ThreadblockShape, WarpShape, InstructionShape>;

using Gemm3 = cutlass::gemm::device::Gemm<
    ElementOutput, LayoutC, ElementInput, LayoutB, ElementOutput, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    ThreadblockShape, WarpShape, InstructionShape>;

// CUDA kernel for elementwise multiplication
__global__ void ElementwiseMultiply(ElementOutput const* G,
                                    ElementOutput const* U,
                                    ElementOutput* Fused, int total_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_elements) {
    float g = static_cast<float>(G[idx]);
    float u = static_cast<float>(U[idx]);
    Fused[idx] = static_cast<ElementOutput>(g * u);
  }
}

/**
 * Panic wrapper for unwinding CUTLASS errors
 */
#define CUTLASS_CHECK(status)                                             \
  {                                                                       \
    cutlass::Status error = status;                                       \
    if (error != cutlass::Status::kSuccess) {                             \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) \
                << " at: " << __LINE__ << std::endl;                      \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  }

void cutlass_mlp(cutlass::HostTensor<ElementInput, LayoutA>& X,
                 cutlass::HostTensor<ElementInput, LayoutB>& Wg,
                 cutlass::HostTensor<ElementInput, LayoutB>& Wu,
                 cutlass::HostTensor<ElementInput, LayoutB>& Wd,
                 cutlass::HostTensor<ElementOutput, LayoutC>& G,
                 cutlass::HostTensor<ElementOutput, LayoutC>& U,
                 cutlass::HostTensor<ElementOutput, LayoutC>& F,
                 cutlass::HostTensor<ElementOutput, LayoutC>& O) {
  // create async stream
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  // auto alpha = ElementAccumulator(1.0f);
  // int split_k_slices = 1;

  auto start = std::chrono::high_resolution_clock::now();
  Gemm1 gemm1;
  Gemm1::Arguments args1({B, Ng, K}, X.device_ref(), Wg.device_ref(),
                         G.device_ref(), G.device_ref());
  CUTLASS_CHECK(gemm1(args1, stream = stream));

  Gemm2 gemm2;
  Gemm2::Arguments args2({B, Nu, K}, X.device_ref(), Wu.device_ref(),
                         U.device_ref(), U.device_ref());
  CUTLASS_CHECK(gemm2(args2, stream = stream));

  //   G.sync_device();
  //   U.sync_device();

  int total_elements = B * Nu;
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;
  ElementwiseMultiply<<<blocks, threads, 0, stream>>>(
      G.device_data(), U.device_data(), F.device_data(), total_elements);

  // cudaError_t err = cudaGetLastError();
  // if (err != cudaSuccess) {
  //   std::cerr << "CUDA error before GEMM: " << cudaGetErrorString(err)
  //             << std::endl;
  // }

  // cudaStreamSynchronize(stream);

  // F.sync_host();
  // G.sync_host();
  // U.sync_host();

  // std::cout << "F output (first 10 elements): ";
  // for (int i = 0; i < 10; ++i) {
  //   std::cout << float(F.host_data()[i]) << " ";
  // }
  // std::cout << std::endl;

  // std::cout << "G output (first 10 elements): ";
  // for (int i = 0; i < 10; ++i) {
  //   std::cout << float(G.host_data()[i]) << " ";
  // }
  // std::cout << std::endl;

  // std::cout << "U output (first 10 elements): ";
  // for (int i = 0; i < 10; ++i) {
  //   std::cout << float(U.host_data()[i]) << " ";
  // }
  // std::cout << std::endl;

  Gemm3 gemm3;
  Gemm3::Arguments args3({B, No, Nu}, F.device_ref(), Wd.device_ref(),
                         O.device_ref(), O.device_ref());

  CUTLASS_CHECK(gemm3(args3, stream = stream));

  cudaStreamSynchronize(stream);
  std::cout << "cudaStreamSynchronize completed." << std::endl;

  // std::cout << "O sync_host completed." << std::endl;

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Gemm1 and Gemm2 execution time: " << elapsed.count()
            << " seconds" << std::endl;

  // Print outputs
  O.sync_host();
  std::cout << "O output (first 10 elements): ";
  for (int i = 0; i < 10; ++i) std::cout << float(O.host_data()[i]) << " ";
  std::cout << std::endl;

  cudaStreamDestroy(stream);
}

void torch_mlp(torch::Tensor& torch_X, torch::Tensor& torch_Wg,
               torch::Tensor& torch_Wu, torch::Tensor& torch_Wd,
               torch::Tensor& torch_G, torch::Tensor& torch_U,
               torch::Tensor& torch_F, torch::Tensor& torch_O) {
  // create async stream
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  // stream guard
  c10::cuda::CUDAStream torch_stream =
      c10::cuda::getStreamFromExternal(stream, 0);
  auto start = std::chrono::high_resolution_clock::now();
  {
    c10::cuda::CUDAStreamGuard guard(torch_stream);
    // gate step
    torch::matmul_out(torch_G, torch_X, torch_Wg.transpose(0, 1));

    // // activation step
    // torch::silu_out(torch_F, torch_G);

    // // up step
    // torch::matmul_out(torch_U, torch_X, torch_Wu.transpose(0, 1));

    // // multiplication step, reuse gate_out
    // torch::mul_out(torch_G, torch_F, torch_U);

    // // down step
    // torch::matmul_out(torch_O, torch_G, torch_Wd.transpose(0, 1));
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Torch MLP execution time: " << elapsed.count() << " seconds"
            << std::endl;

  // Print outputs
  auto flatten_O = torch_O.view({-1});
  std::cout << "Torch O output (first 10 elements): ";
  for (int i = 0; i < 10; ++i) {
    std::cout << flatten_O[i].item<float>() << " ";
  }
  std::cout << std::endl;

  cudaStreamDestroy(stream);
}

int main() {
  //   int B = 128, K = 256, Ng = 512, Nu = 512, No = 256;

  cutlass::HostTensor<ElementInput, LayoutA> X({B, K});
  cutlass::HostTensor<ElementInput, LayoutB> Wg({Ng, K});
  cutlass::HostTensor<ElementInput, LayoutB> Wu({Nu, K});
  cutlass::HostTensor<ElementInput, LayoutB> Wd({No, Nu});
  cutlass::HostTensor<ElementOutput, LayoutC> G({B, Ng});
  cutlass::HostTensor<ElementOutput, LayoutC> U({B, Nu});
  cutlass::HostTensor<ElementOutput, LayoutC> F({B, Nu});
  cutlass::HostTensor<ElementOutput, LayoutC> O({B, No});

  cutlass::reference::host::TensorFillRandomUniform(X.host_view(), 1, 1.0f,
                                                    -0.5f);
  cutlass::reference::host::TensorFillRandomUniform(Wg.host_view(), 1, 1.0f,
                                                    -0.5f);
  cutlass::reference::host::TensorFillRandomUniform(Wu.host_view(), 1, 1.0f,
                                                    -0.5f);
  cutlass::reference::host::TensorFillRandomUniform(Wd.host_view(), 1, 1.0f,
                                                    -0.5f);

  torch::Tensor torch_X = torch::empty({B, K}, torch::kBFloat16).cuda();
  torch::Tensor torch_Wg = torch::empty({Ng, K}, torch::kBFloat16).cuda();
  torch::Tensor torch_Wu = torch::empty({Nu, K}, torch::kBFloat16).cuda();
  torch::Tensor torch_Wd = torch::empty({No, Nu}, torch::kBFloat16).cuda();
  torch::Tensor torch_G = torch::empty({B, Ng}, torch::kBFloat16).cuda();
  torch::Tensor torch_U = torch::empty({B, Nu}, torch::kBFloat16).cuda();
  torch::Tensor torch_F = torch::empty({B, Nu}, torch::kBFloat16).cuda();
  torch::Tensor torch_O = torch::empty({B, No}, torch::kBFloat16).cuda();

  X.sync_device();
  Wg.sync_device();
  Wu.sync_device();
  Wd.sync_device();
  F.sync_device();
  G.sync_device();
  U.sync_device();
  O.sync_device();

  // copy data to torch tensors
  cudaMemcpy(torch_X.data_ptr(), X.device_data(),
             X.size() * sizeof(cutlass::bfloat16_t), cudaMemcpyDeviceToDevice);
  cudaMemcpy(torch_Wg.data_ptr(), Wg.device_data(),
             Wg.size() * sizeof(cutlass::bfloat16_t), cudaMemcpyDeviceToDevice);
  cudaMemcpy(torch_Wu.data_ptr(), Wu.device_data(),
             Wu.size() * sizeof(cutlass::bfloat16_t), cudaMemcpyDeviceToDevice);
  cudaMemcpy(torch_Wd.data_ptr(), Wd.device_data(),
             Wd.size() * sizeof(cutlass::bfloat16_t), cudaMemcpyDeviceToDevice);

  // print all tensors
  std::cout << "Input X (first 10 elements): ";
  for (int i = 0; i < 10; ++i) std::cout << float(X.host_data()[i]) << " ";
  std::cout << std::endl;

  std::cout << "Weight Wg (first 10 elements): ";
  for (int i = 0; i < 10; ++i) std::cout << float(Wg.host_data()[i]) << " ";
  std::cout << std::endl;

  std::cout << "Weight Wu (first 10 elements): ";
  for (int i = 0; i < 10; ++i) std::cout << float(Wu.host_data()[i]) << " ";
  std::cout << std::endl;

  std::cout << "Weight Wd (first 10 elements): ";
  for (int i = 0; i < 10; ++i) std::cout << float(Wd.host_data()[i]) << " ";
  std::cout << std::endl;

  // warm up torch
  auto tmp = torch::matmul(torch_X, torch_Wg.transpose(0, 1));

  torch_mlp(torch_X, torch_Wg, torch_Wu, torch_Wd, torch_G, torch_U, torch_F,
            torch_O);

  cutlass_mlp(X, Wg, Wu, Wd, G, U, F, O);

  //   Gemm1 gemm1;
  //   Gemm1::Arguments args1({B, Ng, K}, {X.device_data(), K},
  //                          {Wg.device_data(), Ng}, {G.device_data(), Ng},
  //                          {G.device_data(), Ng});
  //   gemm1(args1);

  //   Gemm2 gemm2;
  //   Gemm2::Arguments args2({B, Nu, K}, {X.device_data(), K},
  //                          {Wu.device_data(), Nu}, {U.device_data(), Nu},
  //                          {U.device_data(), Nu});
  //   gemm2(args2);

  //   G.sync_device();
  //   U.sync_device();

  //   std::cout << "G output (first 10 elements): ";
  //   for (int i = 0; i < 10; ++i) std::cout << float(G.host_data()[i]) << " ";
  //   std::cout << std::endl;

  //   std::cout << "U output (first 10 elements): ";
  //   for (int i = 0; i < 10; ++i) std::cout << float(U.host_data()[i]) << " ";
  //   std::cout << std::endl;

  //   int total_elements = B * Nu;
  //   int threads = 256;
  //   int blocks = (total_elements + threads - 1) / threads;
  //   ElementwiseMultiply<<<blocks, threads>>>(G.device_data(),
  //   U.device_data(),
  //                                            F.device_data(),
  //                                            total_elements);

  //   F.sync_device();
  //   std::cout << "Fused G * U output (first 10 elements): ";
  //   for (int i = 0; i < 10; ++i) std::cout << float(F.host_data()[i]) << " ";
  //   std::cout << std::endl;

  //   Gemm3 gemm3;
  //   Gemm3::Arguments args3({B, No, Nu}, {F.device_data(), Nu},
  //                          {Wd.device_data(), No}, {O.device_data(), No},
  //                          {O.device_data(), No});
  //   gemm3(args3);

  //   O.sync_host();
  //   std::cout << "Fused MoEMLP output (first 10 elements): ";
  //   for (int i = 0; i < 10; ++i) std::cout << float(O.host_data()[i]) << " ";
  //   std::cout << std::endl;
  //   return 0;
}
