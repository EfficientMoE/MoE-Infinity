#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination_silu.h>
#include <cutlass/layout/row_major.h>
#include <cutlass/layout/column_major.h>

// Data type
using ElementInput = cutlass::bfloat16_t;
using ElementOutput = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = cutlass::bfloat16_t;

// Tile sizes
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

// Layouts
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationSiLU<
    ElementOutput,  // Element type for output
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // Elements per
                                                       // vectorized access
    ElementAccumulator,  // Accumulator (from GEMM)
    ElementCompute       // Compute type (for scale)
    >;

// Define the GEMM with SiLU fused in epilogue
using FusedGemmSiLU = cutlass::gemm::device::Gemm<
    ElementInput, LayoutA, ElementInput, LayoutB, ElementOutput, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOutputOp  // Fused epilogue with SiLU
    >;

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_silu.h>
#include <cutlass/layout/row_major.h>
#include <cutlass/layout/column_major.h>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/arch.h>
#include <cuda_runtime.h>
#include <iostream>

// Define data types
using ElementInput = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute = float;

// Layouts
using LayoutA = cutlass::layout::RowMajor;     // X
using LayoutB = cutlass::layout::ColumnMajor;  // Weights
using LayoutC = cutlass::layout::RowMajor;     // Output

// Tile sizes (adjust for your GPU architecture)
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

// Epilogue for GEMM3 (down projection)
using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator, ElementCompute>;

// GEMM3 definition (fused output * Wd^T)
using Gemm3 = cutlass::gemm::device::Gemm<
    ElementOutput, LayoutA, ElementInput, LayoutB, ElementOutput, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp>;

// Fully fused kernel for GEMM1+SiLU+GEMM2+Mul
__global__ void FusedMoEMLPKernel(
    ElementInput const* X,   // [B, InputSize]
    ElementInput const* Wg,  // [HiddenSize, InputSize]
    ElementInput const* Wu,  // [UpSize, InputSize]
    ElementInput const* Wd,  // [OutSize, UpSize], optional
    ElementOutput* Output,   // [B, OutSize] if Wd != nullptr else [B, UpSize]
    int B, int InputSize, int HiddenSize, int UpSize, int OutSize,
    bool has_Wd) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;  // Batch
  if (row >= B) return;

  // Pointers
  const ElementInput* X_row = X + row * InputSize;
  const ElementInput* Wg_col = Wg;
  const ElementInput* Wu_col = Wu;

  // Accumulators for GEMM1 and GEMM2
  ElementAccumulator acc_g[HiddenSize] = {0};
  ElementAccumulator acc_u[UpSize] = {0};

  // Compute GEMM1 and GEMM2 in registers
  for (int k = 0; k < InputSize; ++k) {
    ElementInput x = X_row[k];
    for (int n = 0; n < HiddenSize; ++n)
      acc_g[n] += static_cast<ElementAccumulator>(x) *
                  static_cast<ElementAccumulator>(Wg_col[n * InputSize + k]);
    for (int n = 0; n < UpSize; ++n)
      acc_u[n] += static_cast<ElementAccumulator>(x) *
                  static_cast<ElementAccumulator>(Wu_col[n * InputSize + k]);
  }

  // Apply SiLU to GEMM1 result
  for (int n = 0; n < HiddenSize; ++n) {
    float x = static_cast<float>(acc_g[n]);
    acc_g[n] = x * (1.0f / (1.0f + expf(-x)));  // SiLU
  }

  // Fused output = SiLU(GEMM1) * GEMM2
  ElementAccumulator fused[UpSize];
  for (int n = 0; n < UpSize; ++n) {
    fused[n] = acc_u[n];
  }

  for (int n = 0; n < min(HiddenSize, UpSize); ++n) {
    fused[n] *= acc_g[n];  // Elementwise multiply
  }

  for (int n = 0; n < OutSize; ++n) {
    ElementAccumulator acc_out = 0;
    for (int k = 0; k < UpSize; ++k)
      acc_out += fused[k] * static_cast<ElementAccumulator>(Wd[n * UpSize + k]);
    Output[row * OutSize + n] = static_cast<ElementOutput>(acc_out);
  }
}
