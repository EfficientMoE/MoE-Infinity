// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// W4A16 MXFP4 GEMM / GPT-OSS expert forward (mxfp4_gptoss_moe.cu).

#pragma once

#include <cuda_runtime.h>

#include <cstddef>

// Stream-K scratch of one GEMM: FP32 partial-tile slots and an int counter
// array that is zero-initialised once by the owner (every launch leaves it
// zeroed). A short `bytes` only shrinks the persistent grid.
struct Mxfp4Workspace {
  void* partials;
  size_t bytes;
  int* counters;
  int n_counters;
};

// Side job for the tiled kernel: de-quantise n_blocks MX blocks (blocks:
// n_blocks x 16 B codes, scales: n_blocks E8M0 bytes) of another MXFP4 tensor
// into `out` (n_blocks x 32 BF16) while the GEMM runs.
struct Mxfp4Side {
  const void* blocks;
  const void* scales;
  void* out;
  int n_blocks;
};

constexpr int kMxfp4StreamKCounters = 1024;

// Up to this many tokens the (workspace-free) small-M kernels run.
constexpr int kMxfp4GemvMaxTokens = 32;

// Bytes of Mxfp4Workspace::partials that let every tile config run a full
// persistent grid on the current device.
size_t mxfp4_linear_workspace_bytes();

// y[M, N] = x[M, K] . W^T + bias, or with swiglu y[M, N/2] =
// SwiGLU-OAI(gate = rows 2j, up = rows 2j+1) (alpha 1.702, limit 7). x, y BF16
// row-major; W as MXFP4 blocks [N, K/2] uint8 + E8M0 scales [N, K/32]; bias
// BF16 [N]. pdl_wait: launched as a programmatic dependent of the previous
// kernel in the stream (x is that kernel's output). work / side may be
// nullptr. config < 0 picks a kernel from M. Returns cudaErrorNotSupported
// for shapes / alignments the kernels do not cover (K % 32, N % 16, 16-B
// aligned pointers).
cudaError_t mxfp4_linear_cuda(const void* x, int M, int K, const void* wq,
                              const void* ws, const void* bias, int N, void* y,
                              bool swiglu, bool pdl_wait,
                              const Mxfp4Workspace* work, const Mxfp4Side* side,
                              int config, cudaStream_t stream);

// GPT-OSS expert with both projections in MXFP4: act = SwiGLU-OAI(x . Wgu^T +
// b_gu); out = act . Wd^T + b_d. x / out [M, H], act [M, I] BF16. gate_up
// blocks [2I, H/2] / scales [2I, H/32]; down blocks [H, I/2] / scales
// [H, I/32]. ws_gu and ws_d must not alias (the down GEMM may start while
// gate_up drains).
cudaError_t gptoss_mxfp4_expert_cuda(const void* x, int M, int H, int I,
                                     const void* gu_blocks,
                                     const void* gu_scales, const void* gu_bias,
                                     const void* d_blocks, const void* d_scales,
                                     const void* d_bias, void* act, void* out,
                                     const Mxfp4Workspace* ws_gu,
                                     const Mxfp4Workspace* ws_d,
                                     cudaStream_t stream);

void mxfp4_dequant_cuda(const void* packed, const void* scales, void* output,
                        int rows, int packed_cols, int scale_cols,
                        int block_size, cudaStream_t stream);
