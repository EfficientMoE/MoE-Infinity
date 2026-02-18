#pragma once

#include <cmath>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_silu.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

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

using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationSilu<
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

// Custom SiLU-Mul epilogue: D[i] = silu(source[i]) * accum[i]
//
// Used as the epilogue for the up-projection GEMM in fused MoE MLP:
//   accum  = input @ up_proj^T     (result of this GEMM)
//   source = gate_out              (result of prior gate GEMM, passed as C)
//   D      = silu(gate_out) * up_accum   -> written to fused_buf
template <typename ElementOutput_, int Count,
          typename ElementAccumulator_ = float,
          typename ElementCompute_ = float>
struct SiLUAndMulEpilogue {
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  static int const kCount = Count;

  using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
  using FragmentCompute = cutlass::Array<ElementCompute, kCount>;

  struct Params {
    ElementCompute alpha;
    ElementCompute beta;
    CUTLASS_HOST_DEVICE Params()
        : alpha(ElementCompute(1)), beta(ElementCompute(1)) {}
    CUTLASS_HOST_DEVICE Params(ElementCompute a, ElementCompute b)
        : alpha(a), beta(b) {}
  };

 private:
  Params params_;

 public:
  CUTLASS_HOST_DEVICE SiLUAndMulEpilogue(Params const& params)
      : params_(params) {}

  // Source (C matrix = gate_out) is always needed
  CUTLASS_HOST_DEVICE bool is_source_needed() const { return true; }

  // No-op for split-K (not used)
  CUTLASS_HOST_DEVICE void set_k_partition(int, int) {}

  // D[i] = silu(source[i]) * accum[i]
  // source = gate_out, accum = up-projection result
  CUTLASS_HOST_DEVICE FragmentOutput operator()(
      FragmentAccumulator const& accum, FragmentOutput const& source) const {
    cutlass::NumericConverter<ElementCompute, ElementAccumulator> to_compute_a;
    cutlass::NumericConverter<ElementCompute, ElementOutput> to_compute_s;
    cutlass::NumericConverter<ElementOutput, ElementCompute> to_output;

    FragmentOutput result;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      ElementCompute x = to_compute_s(source[i]);
      ElementCompute u = to_compute_a(accum[i]);
      // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
      ElementCompute silu_x =
          x * (ElementCompute(1) /
               (ElementCompute(1) + expf(-static_cast<float>(x))));
      result[i] = to_output(silu_x * u);
    }

    return result;
  }

  // Fallback when source is not needed (required by interface, not called here)
  CUTLASS_HOST_DEVICE FragmentOutput
  operator()(FragmentAccumulator const& accum) const {
    cutlass::NumericConverter<ElementCompute, ElementAccumulator> to_compute_a;
    cutlass::NumericConverter<ElementOutput, ElementCompute> to_output;

    FragmentOutput result;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      result[i] = to_output(to_compute_a(accum[i]));
    }

    return result;
  }
};
