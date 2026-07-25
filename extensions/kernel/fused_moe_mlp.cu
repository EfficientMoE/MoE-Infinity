// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// Fused MoE MLP with CUTLASS: replaces 5 separate PyTorch ops with 3 CUTLASS
// GEMMs, fusing silu(gate) * up into the epilogue of the up-projection GEMM.

#include "kernel/epilogue_utils.h"
#include "kernel/fused_moe_mlp.h"

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cstdlib>
#include <string>

// The CUTLASS GEMMs below are specialised for cutlass::arch::Sm80 tensor ops.
// On Blackwell (sm_120) those kernels compile and report kSuccess but produce
// numerically wrong results, so route through a cuBLAS/libtorch path there.
// MOE_INFINITY_FORCE_CUBLAS_MOE=1 forces the fallback on any architecture.
static bool moe_use_cublas_fallback(const torch::Tensor& input) {
  const char* force = std::getenv("MOE_INFINITY_FORCE_CUBLAS_MOE");
  if (force != nullptr && std::string(force) != "0") {
    return true;
  }
  const auto* prop = at::cuda::getDeviceProperties(input.get_device());
  return prop->major >= 12;
}

static void fused_moe_ffn_into_torch_fallback(
    torch::Tensor& hidden, torch::Tensor& gate_proj, torch::Tensor& up_proj,
    torch::Tensor& down_proj, torch::Tensor& gate_buf,
    torch::Tensor& fused_buf, torch::Tensor& output, cudaStream_t stream) {
  c10::cuda::CUDAGuard device_guard(hidden.device());
  auto torch_stream =
      c10::cuda::getStreamFromExternal(stream, hidden.get_device());
  c10::cuda::CUDAStreamGuard stream_guard(torch_stream);
  at::NoGradGuard no_grad;

  gate_buf.copy_(at::mm(hidden, gate_proj.transpose(0, 1)));
  auto up = at::mm(hidden, up_proj.transpose(0, 1));
  fused_buf.copy_(at::silu(gate_buf) * up);
  output.copy_(at::mm(fused_buf, down_proj.transpose(0, 1)));
}

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
    ElementInput,                                     // D element type
    128 / cutlass::sizeof_bits<ElementInput>::value,  // elements per vector
    ElementAccumulator,                               // accumulator type
    float                                             // compute type
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
    ElementInput, LayoutA,  // A
    ElementInput, LayoutB,  // B (ColMajor: gate_proj [I,H] stored as [H,I])
    ElementInput, LayoutC,  // C/D
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    MoEThreadblockShape, MoEWarpShape, MoEInstructionShape, StdEpilogue>;

// GEMM1: fused_buf = silu(gate_buf) * (input @ up_proj^T)
//   A: [M, H] RowMajor   B: up_proj [I,H]→ColMajor [H,I]
//   C: gate_buf [M,I]  (SiLU source in epilogue)    D: fused_buf [M,I]
using GemmUpFused = cutlass::gemm::device::Gemm<
    ElementInput, LayoutA,  // A
    ElementInput, LayoutB,  // B
    ElementInput, LayoutC,  // C (gate_out, read by SiLU epilogue)
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    MoEThreadblockShape, MoEWarpShape, MoEInstructionShape,
    SiLUMulEpilogue  // D = silu(C) * accum
    >;

// GEMM2: output = fused_buf @ down_proj^T
//   A: [M, I] RowMajor   B: down_proj [H,I]→ColMajor [I,H]   D: [M,H]
using GemmDown = cutlass::gemm::device::Gemm<
    ElementInput, LayoutA,  // A
    ElementInput, LayoutB,  // B (ColMajor: down_proj [H,I] stored as [I,H])
    ElementInput, LayoutC,  // C/D
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    MoEThreadblockShape, MoEWarpShape, MoEInstructionShape, StdEpilogue>;

// ---------------------------------------------------------------------------
// Large-K path (K-tile=64): best for H≥3072 (e.g. Mixtral 4096-wide experts)
// ---------------------------------------------------------------------------

using GemmGateLK = cutlass::gemm::device::Gemm<
    ElementInput, LayoutA, ElementInput, LayoutB, ElementInput, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    LargeKThreadblockShape, LargeKWarpShape, MoEInstructionShape, StdEpilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    /*Stages=*/3>;

using GemmUpFusedLK = cutlass::gemm::device::Gemm<
    ElementInput, LayoutA, ElementInput, LayoutB, ElementInput, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    LargeKThreadblockShape, LargeKWarpShape, MoEInstructionShape,
    SiLUMulEpilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    /*Stages=*/3>;

using GemmDownLK = cutlass::gemm::device::Gemm<
    ElementInput, LayoutA, ElementInput, LayoutB, ElementInput, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    LargeKThreadblockShape, LargeKWarpShape, MoEInstructionShape, StdEpilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    /*Stages=*/3>;

// ---------------------------------------------------------------------------
// Throughput tiles for large M: the 128x128 threadblock quadruples the work
// per CTA vs the 64x64 latency tile, keeping the tensor cores fed once M is
// large enough to cover many CTAs. 4 warps/CTA (64x64 each).
//   default-K smem: 2*(128*32)*2B*3 stages = 48 KB
//   large-K  smem: 2*(128*64)*2B*3 stages = 96 KB (CUTLASS opts into dynamic
//                  smem; fits sm_80/sm_90, tight on sm_86)
// ---------------------------------------------------------------------------
using TPThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using TPWarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using TPLargeKThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
using TPLargeKWarpShape = cutlass::gemm::GemmShape<64, 64, 64>;

using GemmGateTP = cutlass::gemm::device::Gemm<
    ElementInput, LayoutA, ElementInput, LayoutB, ElementInput, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    TPThreadblockShape, TPWarpShape, MoEInstructionShape, StdEpilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, /*Stages=*/3>;

using GemmUpFusedTP = cutlass::gemm::device::Gemm<
    ElementInput, LayoutA, ElementInput, LayoutB, ElementInput, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    TPThreadblockShape, TPWarpShape, MoEInstructionShape, SiLUMulEpilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, /*Stages=*/3>;

using GemmDownTP = cutlass::gemm::device::Gemm<
    ElementInput, LayoutA, ElementInput, LayoutB, ElementInput, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    TPThreadblockShape, TPWarpShape, MoEInstructionShape, StdEpilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, /*Stages=*/3>;

using GemmGateTPLK = cutlass::gemm::device::Gemm<
    ElementInput, LayoutA, ElementInput, LayoutB, ElementInput, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    TPLargeKThreadblockShape, TPLargeKWarpShape, MoEInstructionShape,
    StdEpilogue, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    /*Stages=*/3>;

using GemmUpFusedTPLK = cutlass::gemm::device::Gemm<
    ElementInput, LayoutA, ElementInput, LayoutB, ElementInput, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    TPLargeKThreadblockShape, TPLargeKWarpShape, MoEInstructionShape,
    SiLUMulEpilogue, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    /*Stages=*/3>;

using GemmDownTPLK = cutlass::gemm::device::Gemm<
    ElementInput, LayoutA, ElementInput, LayoutB, ElementInput, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    TPLargeKThreadblockShape, TPLargeKWarpShape, MoEInstructionShape,
    StdEpilogue, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    /*Stages=*/3>;

// ---------------------------------------------------------------------------
// Size-based dispatch: choose latency (small M) vs throughput (large M) tiles,
// composed with the existing large-K (I>=3072) axis.
// ---------------------------------------------------------------------------

struct FfnLaunchSpec {
  bool torch_fallback;
  bool m_throughput;
  bool k_large;
};

// Token count above which the throughput path is selected. Scales with SM
// count (~128 tokens per band, 1-4 bands) and bumps one band on Hopper+.
// Override with MOE_MLP_M_THRESHOLD.
static int ffn_m_threshold(const torch::Tensor& input) {
  if (const char* env = std::getenv("MOE_MLP_M_THRESHOLD")) {
    const int v = std::atoi(env);
    if (v > 0) return v;
  }
  const auto* prop = at::cuda::getDeviceProperties(input.get_device());
  int band = (prop->multiProcessorCount + 39) / 40;  // ceil(SM_count / 40)
  band = std::max(1, std::min(band, 4));
  if (prop->major >= 9 && band < 4) band += 1;
  return 128 * band;
}

// Override the auto M-based choice with MOE_MLP_DISPATCH=latency|throughput.
static FfnLaunchSpec ffn_select(int M, int I, const torch::Tensor& input,
                                FfnDispatchPolicy policy) {
  FfnLaunchSpec spec;
  spec.torch_fallback = moe_use_cublas_fallback(input);
  spec.k_large = (I >= 3072);

  if (const char* env = std::getenv("MOE_MLP_DISPATCH")) {
    const std::string v(env);
    if (v == "latency") {
      policy = FfnDispatchPolicy::kForceLatency;
    } else if (v == "throughput") {
      policy = FfnDispatchPolicy::kForceThroughput;
    }
  }
  switch (policy) {
    case FfnDispatchPolicy::kForceLatency:
      spec.m_throughput = false;
      break;
    case FfnDispatchPolicy::kForceThroughput:
      spec.m_throughput = true;
      break;
    default:
      spec.m_throughput = (M > ffn_m_threshold(input));
      break;
  }
  return spec;
}

template <typename GemmGateT, typename GemmUpT, typename GemmDownT>
static void launch_ffn_gemms(ElementInput* input_ptr, ElementInput* gate_ptr,
                             ElementInput* up_ptr, ElementInput* down_ptr,
                             ElementInput* gate_buf_ptr, ElementInput* fused_ptr,
                             ElementInput* out_ptr, int M, int H, int I,
                             cudaStream_t stream) {
  {
    GemmGateT gemm;
    typename GemmGateT::Arguments args{{M, I, H},
                                       {input_ptr, H},
                                       {gate_ptr, H},
                                       {gate_buf_ptr, I},
                                       {gate_buf_ptr, I},
                                       {1.0f, 0.0f}};
    cutlass::Status status = gemm(args, nullptr, stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "fused_moe_ffn_into GEMM0 (gate) failed: ",
                cutlassGetStatusString(status));
  }
  {
    GemmUpT gemm;
    typename GemmUpT::Arguments args{{M, I, H},
                                     {input_ptr, H},
                                     {up_ptr, H},
                                     {gate_buf_ptr, I},
                                     {fused_ptr, I},
                                     {1.0f, 1.0f}};
    cutlass::Status status = gemm(args, nullptr, stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "fused_moe_ffn_into GEMM1 (up+silu-mul) failed: ",
                cutlassGetStatusString(status));
  }
  {
    GemmDownT gemm;
    typename GemmDownT::Arguments args{{M, H, I},
                                       {fused_ptr, I},
                                       {down_ptr, I},
                                       {out_ptr, H},
                                       {out_ptr, H},
                                       {1.0f, 0.0f}};
    cutlass::Status status = gemm(args, nullptr, stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "fused_moe_ffn_into GEMM2 (down) failed: ",
                cutlassGetStatusString(status));
  }
}

void fused_moe_ffn_into(torch::Tensor& hidden,     // [M, H]
                        torch::Tensor& gate_proj,  // [I, H]
                        torch::Tensor& up_proj,    // [I, H]
                        torch::Tensor& down_proj,  // [H, I]
                        torch::Tensor& gate_buf,   // [M, I]
                        torch::Tensor& fused_buf,  // [M, I]
                        torch::Tensor& output,     // [M, H]
                        cudaStream_t stream, FfnDispatchPolicy policy) {
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

  const FfnLaunchSpec spec = ffn_select(M, I, hidden, policy);

  if (spec.torch_fallback) {
    fused_moe_ffn_into_torch_fallback(hidden, gate_proj, up_proj, down_proj,
                                      gate_buf, fused_buf, output, stream);
    return;
  }

  using Elem = ElementInput;
  auto* input_ptr = reinterpret_cast<Elem*>(hidden.data_ptr());
  auto* gate_ptr = reinterpret_cast<Elem*>(gate_proj.data_ptr());
  auto* up_ptr = reinterpret_cast<Elem*>(up_proj.data_ptr());
  auto* down_ptr = reinterpret_cast<Elem*>(down_proj.data_ptr());
  auto* gate_buf_ptr = reinterpret_cast<Elem*>(gate_buf.data_ptr());
  auto* fused_ptr = reinterpret_cast<Elem*>(fused_buf.data_ptr());
  auto* out_ptr = reinterpret_cast<Elem*>(output.data_ptr());

  // M-axis dispatch: large M uses the 128x128 throughput tiles; small M keeps
  // the 64x64 latency tiles (bit-identical to the original single-path code).
  // Each M-variant composes with the large-K (I>=3072) axis.
  if (spec.m_throughput) {
    if (spec.k_large) {
      launch_ffn_gemms<GemmGateTPLK, GemmUpFusedTPLK, GemmDownTPLK>(
          input_ptr, gate_ptr, up_ptr, down_ptr, gate_buf_ptr, fused_ptr,
          out_ptr, M, H, I, stream);
    } else {
      launch_ffn_gemms<GemmGateTP, GemmUpFusedTP, GemmDownTP>(
          input_ptr, gate_ptr, up_ptr, down_ptr, gate_buf_ptr, fused_ptr,
          out_ptr, M, H, I, stream);
    }
  } else {
    if (spec.k_large) {
      launch_ffn_gemms<GemmGateLK, GemmUpFusedLK, GemmDownLK>(
          input_ptr, gate_ptr, up_ptr, down_ptr, gate_buf_ptr, fused_ptr,
          out_ptr, M, H, I, stream);
    } else {
      launch_ffn_gemms<GemmGate, GemmUpFused, GemmDown>(
          input_ptr, gate_ptr, up_ptr, down_ptr, gate_buf_ptr, fused_ptr,
          out_ptr, M, H, I, stream);
    }
  }
}
