// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// Fused MoE MLP with CUTLASS: replaces 5 separate PyTorch ops with 3 CUTLASS
// GEMMs, fusing silu(gate) * up into the epilogue of the up-projection GEMM.

#include "kernel/epilogue_utils.h"
#include "kernel/fused_moe_mlp.h"

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>

// Small-M tile sizes tuned for kMaxTokens = 128.
// Threadblock covers 64 rows of M, so 128-token batches use 2 threadblocks.
// K-tile=32 is efficient for narrow hidden dims (H≤2048).
using MoEThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
using MoEWarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
using MoEInstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

// Large-K tile for H≥3072 (e.g., Mixtral 4096-wide experts).
// K-tile=64 halves k-iterations (128→64 for K=4096), same 4 warps per TB.
// sm_86 budget: (64×64 + 64×64) BF16 × 3 stages = 48 KB/TB → 2 TBs/SM.
using LargeKThreadblockShape = cutlass::gemm::GemmShape<64, 64, 64>;
using LargeKWarpShape = cutlass::gemm::GemmShape<32, 32, 64>;
// InstructionShape unchanged: <16, 8, 16> is valid for K-tile multiples of 16.

// Standard linear-combination epilogue (D = alpha*accum + beta*C).
// Used for gate GEMM (beta=0) and down GEMM (beta=0).
using StdEpilogue = cutlass::epilogue::thread::LinearCombination<
    ElementInput,                                      // D element type
    128 / cutlass::sizeof_bits<ElementInput>::value,  // elements per vector
    ElementAccumulator,                                // accumulator type
    float                                              // compute type
    >;

// SiLU-Mul epilogue: D[i] = silu(C[i]) * accum[i]
// C = gate_out (passed as the "source" matrix), accum = up-projection result.
using SiLUMulEpilogue =
    SiLUAndMulEpilogue<ElementInput,
                       128 / cutlass::sizeof_bits<ElementInput>::value,
                       ElementAccumulator, float>;

// ---------------------------------------------------------------------------
// Small-K path (K-tile=32): best for H≤2048
// NOTE: CUTLASS 2.x DefaultGemm only specialises on Sm80 for BF16 tensor ops;
// Sm86 has no matching partial specialisation in this version. The binary is
// still compiled with -gencode arch=compute_86,code=sm_86 so PTX is optimal.
// ---------------------------------------------------------------------------

// GEMM0: gate_buf = input @ gate_proj^T
//   A: [M, H] RowMajor   B: gate_proj [I,H]→ColMajor [H,I]   D: [M,I]
using GemmGate = cutlass::gemm::device::Gemm<
    ElementInput, LayoutA,   // A
    ElementInput, LayoutB,   // B (ColMajor: gate_proj [I,H] stored as [H,I])
    ElementInput, LayoutC,   // C/D
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    MoEThreadblockShape, MoEWarpShape, MoEInstructionShape, StdEpilogue>;

// GEMM1: fused_buf = silu(gate_buf) * (input @ up_proj^T)
//   A: [M, H] RowMajor   B: up_proj [I,H]→ColMajor [H,I]
//   C: gate_buf [M,I]  (SiLU source in epilogue)    D: fused_buf [M,I]
using GemmUpFused = cutlass::gemm::device::Gemm<
    ElementInput, LayoutA,   // A
    ElementInput, LayoutB,   // B
    ElementInput, LayoutC,   // C (gate_out, read by SiLU epilogue)
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    MoEThreadblockShape, MoEWarpShape, MoEInstructionShape,
    SiLUMulEpilogue  // D = silu(C) * accum
    >;

// GEMM2: output = fused_buf @ down_proj^T
//   A: [M, I] RowMajor   B: down_proj [H,I]→ColMajor [I,H]   D: [M,H]
using GemmDown = cutlass::gemm::device::Gemm<
    ElementInput, LayoutA,   // A
    ElementInput, LayoutB,   // B (ColMajor: down_proj [H,I] stored as [I,H])
    ElementInput, LayoutC,   // C/D
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    MoEThreadblockShape, MoEWarpShape, MoEInstructionShape, StdEpilogue>;

// ---------------------------------------------------------------------------
// Large-K path (K-tile=64): best for H≥3072 (e.g. Mixtral 4096-wide experts)
// ---------------------------------------------------------------------------

using GemmGateLK = cutlass::gemm::device::Gemm<
    ElementInput, LayoutA,
    ElementInput, LayoutB,
    ElementInput, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    LargeKThreadblockShape, LargeKWarpShape, MoEInstructionShape,
    StdEpilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    /*Stages=*/3>;

using GemmUpFusedLK = cutlass::gemm::device::Gemm<
    ElementInput, LayoutA,
    ElementInput, LayoutB,
    ElementInput, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    LargeKThreadblockShape, LargeKWarpShape, MoEInstructionShape,
    SiLUMulEpilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    /*Stages=*/3>;

using GemmDownLK = cutlass::gemm::device::Gemm<
    ElementInput, LayoutA,
    ElementInput, LayoutB,
    ElementInput, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    LargeKThreadblockShape, LargeKWarpShape, MoEInstructionShape,
    StdEpilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    /*Stages=*/3>;

void fused_moe_ffn_into(torch::Tensor& hidden,     // [M, H]
                        torch::Tensor& gate_proj,  // [I, H]
                        torch::Tensor& up_proj,    // [I, H]
                        torch::Tensor& down_proj,  // [H, I]
                        torch::Tensor& gate_buf,   // [M, I]
                        torch::Tensor& fused_buf,  // [M, I]
                        torch::Tensor& output,     // [M, H]
                        cudaStream_t stream) {
  TORCH_CHECK(hidden.scalar_type() == at::kBFloat16,
              "fused_moe_ffn_into: BF16 only");

  const int M = static_cast<int>(hidden.size(0));
  const int H = static_cast<int>(hidden.size(1));
  const int I = static_cast<int>(gate_proj.size(0));
  const int H_out = static_cast<int>(down_proj.size(0));

  TORCH_CHECK(H == H_out, "fused_moe_ffn_into: hidden dim mismatch");
  TORCH_CHECK(gate_proj.size(1) == H && up_proj.size(1) == H,
              "fused_moe_ffn_into: gate/up proj K-dim mismatch");
  TORCH_CHECK(down_proj.size(1) == I,
              "fused_moe_ffn_into: down proj intermediate dim mismatch");

  using Elem = ElementInput;

  auto* input_ptr = reinterpret_cast<Elem*>(hidden.data_ptr());
  auto* gate_ptr = reinterpret_cast<Elem*>(gate_proj.data_ptr());
  auto* up_ptr = reinterpret_cast<Elem*>(up_proj.data_ptr());
  auto* down_ptr = reinterpret_cast<Elem*>(down_proj.data_ptr());
  auto* gate_buf_ptr = reinterpret_cast<Elem*>(gate_buf.data_ptr());
  auto* fused_ptr = reinterpret_cast<Elem*>(fused_buf.data_ptr());
  auto* out_ptr = reinterpret_cast<Elem*>(output.data_ptr());

  // Dispatch: use large-K tile when I >= 3072.
  // K-tile=64 halves k-loop iterations for wide dims (K=4096: 128→64 iters).
  const bool use_large_k = (I >= 3072);

  if (use_large_k) {
    // ------------------------------------------------------------------
    // GEMM0 (large-K): gate_buf = input @ gate_proj^T
    // ------------------------------------------------------------------
    {
      GemmGateLK gemm;
      GemmGateLK::Arguments args{
          {M, I, H},          // problem size [M, N, K]
          {input_ptr, H},     // ref_A: [M, H] RowMajor
          {gate_ptr, H},      // ref_B: [H, I] ColMajor, ldb = K = H
          {gate_buf_ptr, I},  // ref_C: not read (beta=0)
          {gate_buf_ptr, I},  // ref_D: gate_buf output
          {1.0f, 0.0f}        // {alpha, beta}
      };
      cutlass::Status status = gemm(args, nullptr, stream);
      TORCH_CHECK(status == cutlass::Status::kSuccess,
                  "fused_moe_ffn_into GEMM0-LK (gate) failed: ",
                  cutlassGetStatusString(status));
    }

    // ------------------------------------------------------------------
    // GEMM1 (large-K): fused_buf = silu(gate_buf) * (input @ up_proj^T)
    // ------------------------------------------------------------------
    {
      GemmUpFusedLK gemm;
      GemmUpFusedLK::Arguments args{
          {M, I, H},          // problem size
          {input_ptr, H},     // ref_A
          {up_ptr, H},        // ref_B: [H, I] ColMajor, ldb = K = H
          {gate_buf_ptr, I},  // ref_C: gate_buf (SiLU source)
          {fused_ptr, I},     // ref_D: fused_buf output
          {1.0f, 1.0f}        // {alpha, beta} — beta=1 causes C to be loaded
      };
      cutlass::Status status = gemm(args, nullptr, stream);
      TORCH_CHECK(status == cutlass::Status::kSuccess,
                  "fused_moe_ffn_into GEMM1-LK (up+silu-mul) failed: ",
                  cutlassGetStatusString(status));
    }

    // ------------------------------------------------------------------
    // GEMM2 (large-K): output = fused_buf @ down_proj^T
    // ------------------------------------------------------------------
    {
      GemmDownLK gemm;
      GemmDownLK::Arguments args{
          {M, H, I},       // problem size [M, N, K]
          {fused_ptr, I},  // ref_A: [M, I] RowMajor
          {down_ptr, I},   // ref_B: [I, H] ColMajor, ldb = K = I
          {out_ptr, H},    // ref_C: not read (beta=0)
          {out_ptr, H},    // ref_D: output
          {1.0f, 0.0f}     // {alpha, beta}
      };
      cutlass::Status status = gemm(args, nullptr, stream);
      TORCH_CHECK(status == cutlass::Status::kSuccess,
                  "fused_moe_ffn_into GEMM2-LK (down) failed: ",
                  cutlassGetStatusString(status));
    }
  } else {
    // ------------------------------------------------------------------
    // GEMM0: gate_buf = input @ gate_proj^T
    //
    // CUTLASS GEMM: D [M, N] = A [M, K] × B [K, N]
    //   M=batch, N=I (intermediate), K=H (hidden)
    //   A: [M, H] RowMajor,        lda = H
    //   B: [H, I] ColMajor         ldb = H   (gate_proj [I,H] RowMajor ≡ [H,I] ColMajor)
    //   D: [M, I] RowMajor,        ldd = I
    //   beta=0 → C not read
    // ------------------------------------------------------------------
    {
      GemmGate gemm;
      GemmGate::Arguments args{
          {M, I, H},          // problem size [M, N, K]
          {input_ptr, H},     // ref_A: [M, H] RowMajor
          {gate_ptr, H},      // ref_B: [H, I] ColMajor, ldb = K = H
          {gate_buf_ptr, I},  // ref_C: not read (beta=0)
          {gate_buf_ptr, I},  // ref_D: gate_buf output
          {1.0f, 0.0f}        // {alpha, beta}
      };
      cutlass::Status status = gemm(args, nullptr, stream);
      TORCH_CHECK(status == cutlass::Status::kSuccess,
                  "fused_moe_ffn_into GEMM0 (gate) failed: ",
                  cutlassGetStatusString(status));
    }

    // ------------------------------------------------------------------
    // GEMM1: fused_buf = silu(gate_buf) * (input @ up_proj^T)
    //
    //   Same A/B shapes as GEMM0 but uses up_proj
    //   C = gate_buf [M, I]: read as the "source" in SiLUMulEpilogue
    //   D = fused_buf [M, I]: D[i] = silu(C[i]) * accum[i]
    //   beta=1 → C is loaded by the epilogue
    // ------------------------------------------------------------------
    {
      GemmUpFused gemm;
      GemmUpFused::Arguments args{
          {M, I, H},          // problem size
          {input_ptr, H},     // ref_A
          {up_ptr, H},        // ref_B: [H, I] ColMajor, ldb = K = H
          {gate_buf_ptr, I},  // ref_C: gate_buf (SiLU source)
          {fused_ptr, I},     // ref_D: fused_buf output
          {1.0f, 1.0f}        // {alpha, beta} — beta=1 causes C to be loaded
      };
      cutlass::Status status = gemm(args, nullptr, stream);
      TORCH_CHECK(status == cutlass::Status::kSuccess,
                  "fused_moe_ffn_into GEMM1 (up+silu-mul) failed: ",
                  cutlassGetStatusString(status));
    }

    // ------------------------------------------------------------------
    // GEMM2: output = fused_buf @ down_proj^T
    //
    //   M=batch, N=H (hidden), K=I (intermediate)
    //   A: [M, I] RowMajor,        lda = I
    //   B: [I, H] ColMajor,        ldb = I   (down_proj [H,I] RowMajor ≡ [I,H] ColMajor)
    //   D: [M, H] RowMajor,        ldd = H
    //   beta=0 → C not read
    // ------------------------------------------------------------------
    {
      GemmDown gemm;
      GemmDown::Arguments args{
          {M, H, I},       // problem size [M, N, K]
          {fused_ptr, I},  // ref_A: [M, I] RowMajor
          {down_ptr, I},   // ref_B: [I, H] ColMajor, ldb = K = I
          {out_ptr, H},    // ref_C: not read (beta=0)
          {out_ptr, H},    // ref_D: output
          {1.0f, 0.0f}     // {alpha, beta}
      };
      cutlass::Status status = gemm(args, nullptr, stream);
      TORCH_CHECK(status == cutlass::Status::kSuccess,
                  "fused_moe_ffn_into GEMM2 (down) failed: ",
                  cutlassGetStatusString(status));
    }
  }
}
