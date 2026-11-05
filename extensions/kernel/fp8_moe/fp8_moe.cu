// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team
//
// Native FP8 (e4m3, 128x128 block-scaled) SwiGLU expert, see fp8_moe.h.
//
// Hardware specificity (docs/hardware-gated-kernels.md): tuned on the RTX PRO
// 6000 Blackwell Max-Q (sm_120, 188 SMs, 128 MiB L2, 99 KiB smem/block).
//   - CUTLASS sm120 blockwise FP8 GEMM (Sm120*Gemm): SM120 gate, built only
//     with MOE_ENABLE_SM120_FP8 (sm_120a).
//   - tensor-core decode GEMV (fp8_gemv_pro6000_kernel), the tile / stream-K /
//     serialisation choices (blockwise_gemm_sm120_pro6000,
//     pro6000_serialize_min_tokens): RTX_PRO_6000 gate.
// fp8_moe_sm120_pro6000_supported() checks both; on every other device the
// caller (MoEMLP) keeps the previous FP8 path: per-call dequant to BF16 +
// BF16 CUTLASS GEMM.

#include "fp8_moe.h"
#include "kernel/hw_profile.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <cstdlib>
#include <mutex>

#if defined(MOE_ENABLE_SM120_FP8)
  // griddepcontrol (programmatic dependent launch) in the CUTLASS sm_120
  // kernels; a no-op unless the launch carries the PDL attribute.
  #ifndef CUTLASS_ENABLE_GDC_FOR_SM100
    #define CUTLASS_ENABLE_GDC_FOR_SM100 1
  #endif
  #include "cute/tensor.hpp"
  #include "cutlass/cutlass.h"
  #include "cutlass/detail/blockwise_scale_layout.hpp"
  #include "cutlass/epilogue/collective/collective_builder.hpp"
  #include "cutlass/gemm/collective/collective_builder.hpp"
  #include "cutlass/gemm/device/gemm_universal_adapter.h"
  #include "cutlass/gemm/dispatch_policy.hpp"
  #include "cutlass/gemm/kernel/gemm_universal.hpp"
#endif

namespace fp8_moe_detail {

constexpr int kBlock = 128;    // scale block (both dims)
constexpr int kE4M3Max = 448;  // largest finite e4m3fn value

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Programmatic dependent launch: kernels of one expert are chained so the
// next kernel's CTAs launch (and set up) while the previous one drains.
// pdl_wait() must precede the first global access to the predecessor's data.
__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdl_launch_dependents() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// Drops a dead 128-byte L2 line without writing it back to DRAM.
__device__ __forceinline__ void discard_l2_line(const void* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("discard.global.L2 [%0], 128;" ::"l"(p) : "memory");
#endif
}

template <class Kernel, class... Args>
void launch_kernel(Kernel kernel, dim3 grid, dim3 block, bool pdl,
                   cudaStream_t stream, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, args...));
}

__device__ __forceinline__ uint4 ld_stream_u4(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

// 8 packed bf16 -> 8 floats.
__device__ __forceinline__ void bf16x8_to_float(const uint4& v, float* f) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[2 * i] = __uint_as_float(w[i] << 16);
    f[2 * i + 1] = __uint_as_float(w[i] & 0xFFFF0000u);
  }
}

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + __expf(-x));
}

// ---------------------------------------------------------------------------
// Tensor-core weight-only FP8 GEMV (decode, M <= 8 tokens)
//
// Same math as fp8_gemv_kernel (exact e4m3 -> bf16 weights, BF16 activations,
// fp32 accumulation, block scale per 128-K step) on mma.m16n8k16 bf16: the
// weight tile is the A operand (16 output rows), the tokens are the 8 columns
// of B (zero-padded past M), so the FMA work leaves the CUDA cores and the
// kernel stays memory-bound up to 8 tokens. K is consumed in 64-wide chunks
// that each lane loads as 16 contiguous bytes per row; inside a chunk the K
// order is permuted identically for A and B (a dot product does not care),
// so no shuffles are needed to build fragments. Each CTA owns 16 rows and
// splits K over kTcWarps warps, reduced through shared memory.
// ---------------------------------------------------------------------------

constexpr int kTcWarps = 8;

__device__ __forceinline__ uint32_t e4m3x2_to_bf16x2(uint32_t v16) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(
      static_cast<__nv_fp8x2_storage_t>(v16 & 0xFFFFu), __NV_E4M3);
  const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
  const __nv_bfloat162 b = __floats2bfloat162_rn(f.x, f.y);
  return *reinterpret_cast<const uint32_t*>(&b);
}

__device__ __forceinline__ void mma_bf16_16816(float* d, const uint32_t* a,
                                               uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int NMAT, int U, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
    fp8_gemv_pro6000_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx,
                            int M, const uint8_t* __restrict__ w0,
                            const float* __restrict__ s0,
                            const uint8_t* __restrict__ w1,
                            const float* __restrict__ s1,
                            __nv_bfloat16* __restrict__ out, int64_t ldo, int N,
                            int K) {
  __shared__ float red[WARPS][NMAT][32][4];
  pdl_wait();
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int g = lane >> 2;  // fragment row / token
  const int c = lane & 3;   // 16-byte slot inside a 64-wide K chunk
  const int row0 = blockIdx.x * 16;
  const int KB = K / kBlock;
  const int chunks = K / 64;

  const uint8_t* wp[NMAT];
  const float* sp[NMAT];
  wp[0] = w0;
  sp[0] = s0;
  if constexpr (NMAT > 1) {
    wp[1] = w1;
    sp[1] = s1;
  }
  const int64_t r_lo = min(row0 + g, N - 1);
  const int64_t r_hi = min(row0 + g + 8, N - 1);
  const int sblk = (row0 / kBlock) * KB;
  const bool tok = g < M;

  float acc[NMAT][4];
#pragma unroll
  for (int a = 0; a < NMAT; ++a)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[a][i] = 0.f;

  for (int base = warp; base < chunks; base += WARPS * U) {
    uint4 wlo[U][NMAT], whi[U][NMAT], xv[U][2];
    float sc[U][NMAT];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int ch = base + u * WARPS;
      const bool ok = ch < chunks;
      const int k = ch * 64 + c * 16;
#pragma unroll
      for (int a = 0; a < NMAT; ++a) {
        wlo[u][a] =
            ok ? ld_stream_u4(wp[a] + r_lo * K + k) : make_uint4(0, 0, 0, 0);
        whi[u][a] =
            ok ? ld_stream_u4(wp[a] + r_hi * K + k) : make_uint4(0, 0, 0, 0);
        sc[u][a] = ok ? __ldg(sp[a] + sblk + ch / 2) : 0.f;
      }
      const uint4* xp = reinterpret_cast<const uint4*>(x + g * ldx + k);
      xv[u][0] = ok && tok ? __ldg(xp) : make_uint4(0, 0, 0, 0);
      xv[u][1] = ok && tok ? __ldg(xp + 1) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const uint32_t xb[8] = {xv[u][0].x, xv[u][0].y, xv[u][0].z, xv[u][0].w,
                              xv[u][1].x, xv[u][1].y, xv[u][1].z, xv[u][1].w};
#pragma unroll
      for (int a = 0; a < NMAT; ++a) {
        const uint32_t lo[4] = {wlo[u][a].x, wlo[u][a].y, wlo[u][a].z,
                                wlo[u][a].w};
        const uint32_t hi[4] = {whi[u][a].x, whi[u][a].y, whi[u][a].z,
                                whi[u][a].w};
        float t[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          // bytes 4j..4j+3 of this lane's 16: (p, p+1) -> k {2c, 2c+1},
          // (p+2, p+3) -> k {2c+8, 2c+9} of MMA j.
          const uint32_t afr[4] = {
              e4m3x2_to_bf16x2(lo[j]), e4m3x2_to_bf16x2(hi[j]),
              e4m3x2_to_bf16x2(lo[j] >> 16), e4m3x2_to_bf16x2(hi[j] >> 16)};
          mma_bf16_16816(t, afr, xb[2 * j], xb[2 * j + 1]);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[a][i] = fmaf(t[i], sc[u][a], acc[a][i]);
      }
    }
  }

#pragma unroll
  for (int a = 0; a < NMAT; ++a)
#pragma unroll
    for (int i = 0; i < 4; ++i) red[warp][a][lane][i] = acc[a][i];
  __syncthreads();
  pdl_launch_dependents();
  if (warp != 0) return;
#pragma unroll
  for (int a = 0; a < NMAT; ++a)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < WARPS; ++w) v += red[w][a][lane][i];
      acc[a][i] = v;
    }
    // C fragment: (row g, tokens 2c, 2c+1), (row g+8, tokens 2c, 2c+1).
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int row = row0 + g + (i >= 2 ? 8 : 0);
    const int t = 2 * c + (i & 1);
    if (t < M && row < N) {
      float o = acc[0][i];
      if constexpr (NMAT == 2) o = silu(o) * acc[1][i];
      out[t * ldo + row] = __float2bfloat16(o);
    }
  }
}

template <int NMAT>
void launch_gemv_pro6000(int M, const __nv_bfloat16* x, int64_t ldx,
                         const uint8_t* w0, const float* s0, const uint8_t* w1,
                         const float* s1, __nv_bfloat16* out, int64_t ldo,
                         int N, int K, cudaStream_t stream) {
  // 4 chunks in flight per warp, 8 warps per 16 rows: best of a sweep over
  // (U, warps) in {1,2,4} x {4,8} on GLM-5.3 / GLM-5.3-Flash shapes.
  launch_kernel(fp8_gemv_pro6000_kernel<NMAT, 4, kTcWarps>, dim3((N + 15) / 16),
                dim3(kTcWarps * 32), NMAT == 1, stream, x, ldx, M, w0, s0, w1,
                s1, out, ldo, N, K);
}

// ---------------------------------------------------------------------------
// Activation quantisation: a half-warp per (row, 128-channel block), 8
// elements (16 B) per lane. q = e4m3(v * 448 / amax), sf[kb * M + m] =
// amax / 448 (M-major, the CUTLASS SFA layout for 1x128 granularity).
//   SILU_MUL = false: v = x[m, k]
//   SILU_MUL = true : v = silu(gu[m, gate_off + k]) * gu[m, up_off + k]
// Blocks past the quantisation grid transpose the K-major weight scales into
// the MN-major SFB buffers the GEMMs read.
// ---------------------------------------------------------------------------

struct ScaleCopy {
  const float* src;  // [NB, KB] row-major (K-major)
  float* dst;        // [KB, NB] (MN-major)
  int NB;
  int KB;
};

// Dead workspace whose dirty L2 lines are dropped instead of written back.
struct DeadRegion {
  const uint8_t* ptr = nullptr;  // 128-byte aligned
  int64_t bytes = 0;
};

constexpr int kQuantThreads = 256;
constexpr int kQuantGroupsPerBlock = kQuantThreads / 16;

template <bool SILU_MUL>
__global__ void __launch_bounds__(kQuantThreads)
    fp8_quant_act_kernel(const __nv_bfloat16* __restrict__ in, int64_t ldi,
                         int gate_off, int up_off, uint8_t* __restrict__ q,
                         int64_t ldq, float* __restrict__ sf, int M, int K,
                         int quant_blocks, ScaleCopy c0, ScaleCopy c1,
                         ScaleCopy c2, DeadRegion dead) {
  pdl_wait();
  pdl_launch_dependents();
  if ((int)blockIdx.x >= quant_blocks) {
    const int tid = (blockIdx.x - quant_blocks) * blockDim.x + threadIdx.x;
    const int stride = (gridDim.x - quant_blocks) * blockDim.x;
    const ScaleCopy cs[3] = {c0, c1, c2};
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      const int total = cs[c].NB * cs[c].KB;
      for (int i = tid; i < total; i += stride) {
        const int nb = i / cs[c].KB;
        const int kb = i % cs[c].KB;
        cs[c].dst[kb * cs[c].NB + nb] = cs[c].src[i];
      }
    }
    return;
  }

  const int KB = K / kBlock;
  const int group = blockIdx.x * kQuantGroupsPerBlock + (threadIdx.x >> 4);
  const bool active = group < M * KB;
  const int sub = threadIdx.x & 15;
  const int m = active ? group / KB : 0;
  const int kb = active ? group % KB : 0;
  const int k = kb * kBlock + sub * 8;

  float v[8];
  if (active) {
    if constexpr (SILU_MUL) {
      float g[8], u[8];
      bf16x8_to_float(
          *reinterpret_cast<const uint4*>(in + m * ldi + gate_off + k), g);
      bf16x8_to_float(
          *reinterpret_cast<const uint4*>(in + m * ldi + up_off + k), u);
#pragma unroll
      for (int i = 0; i < 8; ++i) v[i] = silu(g[i]) * u[i];
    } else {
      bf16x8_to_float(*reinterpret_cast<const uint4*>(in + m * ldi + k), v);
    }
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = 0.f;
  }

  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(v[i]));
#pragma unroll
  for (int o = 8; o > 0; o >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));

  if constexpr (SILU_MUL) {
    // The [M, 2I] gate/up intermediate is dead once read (the shuffle above
    // orders it after every lane's load), and so is the FP8 copy of x the
    // gate/up GEMM consumed: drop their dirty lines rather than writing them
    // back to DRAM.
    if (active && sub % 8 == 0) {
      discard_l2_line(in + m * ldi + gate_off + k);
      discard_l2_line(in + m * ldi + up_off + k);
    }
    const int64_t lines = dead.bytes / 128;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < lines;
         i += int64_t(gridDim.x) * blockDim.x) {
      discard_l2_line(dead.ptr + i * 128);
    }
  }
  if (!active) return;

  const float inv = amax > 0.f ? kE4M3Max / amax : 1.f;
  uint32_t packed[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(
        make_float2(v[4 * h] * inv, v[4 * h + 1] * inv), __NV_SATFINITE,
        __NV_E4M3);
    const __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(
        make_float2(v[4 * h + 2] * inv, v[4 * h + 3] * inv), __NV_SATFINITE,
        __NV_E4M3);
    packed[h] = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
  }
  *reinterpret_cast<uint2*>(q + m * ldq + k) = make_uint2(packed[0], packed[1]);
  if (sub == 0) sf[kb * M + m] = amax > 0.f ? amax / kE4M3Max : 1.f;
}

template <bool SILU_MUL>
void launch_quant(const __nv_bfloat16* in, int64_t ldi, int gate_off,
                  int up_off, uint8_t* q, int64_t ldq, float* sf, int M, int K,
                  ScaleCopy c0, ScaleCopy c1, ScaleCopy c2, DeadRegion dead,
                  cudaStream_t stream) {
  const int groups = M * (K / kBlock);
  const int quant_blocks =
      (groups + kQuantGroupsPerBlock - 1) / kQuantGroupsPerBlock;
  const int copy_elems = c0.NB * c0.KB + c1.NB * c1.KB + c2.NB * c2.KB;
  const int copy_blocks =
      copy_elems > 0 ? std::min(16, (copy_elems + 255) / 256) : 0;
  // The input quantisation opens the expert (no PDL edge); the silu*up
  // quantisation overlaps the gate/up GEMM tail.
  launch_kernel(fp8_quant_act_kernel<SILU_MUL>,
                dim3(quant_blocks + copy_blocks), dim3(kQuantThreads), SILU_MUL,
                stream, in, ldi, gate_off, up_off, q, ldq, sf, M, K,
                quant_blocks, c0, c1, c2, dead);
}

// ---------------------------------------------------------------------------
// CUTLASS sm_120 blockwise-scaled FP8 GEMM: D[m, n] (bf16) =
//   sum_kb SFA[kb, m] * SFB[kb, n/128] * sum_{k in kb} A[m, k] * B[n, k]
// A row-major e4m3 (1x128 scales), B = weight [N, K] row-major e4m3
// (128x128 scales), batched over L with arbitrary batch strides.
// ---------------------------------------------------------------------------

#if defined(MOE_ENABLE_SM120_FP8)
using namespace cute;

using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementD = cutlass::bfloat16_t;
using ElementAcc = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;
constexpr int kAlignAB = 16;
constexpr int kAlignD = 8;

using ScaleConfig =
    cutlass::detail::Sm120BlockwiseScaleConfig<1, kBlock, kBlock>;
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

template <class TileShape, class Schedule, class Scheduler = void>
struct BlockwiseGemm {
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, TileShape,
      Shape<_1, _1, _1>, cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementAcc, void, LayoutD, kAlignD, ElementD, LayoutD,
      kAlignD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, ElementA,
      cute::tuple<LayoutA, LayoutSFA>, kAlignAB, ElementB,
      cute::tuple<LayoutB, LayoutSFB>, kAlignAB, ElementAcc, TileShape,
      Shape<_1, _1, _1>,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename Epilogue::SharedStorage))>,
      Schedule>::CollectiveOp;
  using Kernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop,
                                           Epilogue, Scheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

using Sm120CoopGemm =
    BlockwiseGemm<Shape<_128, _128, _128>,
                  cutlass::gemm::KernelScheduleSm120Blockwise>::Gemm;
using Sm120StreamKGemm =
    BlockwiseGemm<Shape<_128, _128, _128>,
                  cutlass::gemm::KernelScheduleSm120Blockwise,
                  cutlass::gemm::StreamKScheduler>::Gemm;
using Sm120PingGemm = BlockwiseGemm<
    Shape<_64, _128, _128>,
    cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120>::Gemm;

struct GemmProblem {
  int M, N, K, L;
  const uint8_t* A;
  int64_t lda, batch_a;
  const uint8_t* B;
  int64_t ldb, batch_b;
  const float* SFA;  // [KB, M] per batch
  int32_t batch_sfa;
  const float* SFB;  // [KB, N/128] per batch
  int32_t batch_sfb;
  __nv_bfloat16* D;
  int64_t ldd, batch_d;
};

template <class Gemm>
void run_gemm(const GemmProblem& p, cudaStream_t stream) {
  using GK = typename Gemm::GemmKernel;
  using StrideA = typename GK::StrideA;
  using StrideB = typename GK::StrideB;
  using StrideC = typename GK::StrideC;
  using StrideD = typename GK::StrideD;

  StrideA sa = cute::make_stride(p.lda, cute::Int<1>{}, p.batch_a);
  StrideB sb = cute::make_stride(p.ldb, cute::Int<1>{}, p.batch_b);
  StrideD sd = cute::make_stride(p.ldd, cute::Int<1>{}, p.batch_d);

  auto shape = cute::make_shape(p.M, p.N, p.K, p.L);
  auto lsfa0 = ScaleConfig::tile_atom_to_shape_SFA(shape);
  auto lsfb0 = ScaleConfig::tile_atom_to_shape_SFB(shape);
  LayoutSFA lsfa = cute::make_layout(
      lsfa0.shape(), cute::make_stride(get<0>(lsfa0.stride()),
                                       get<1>(lsfa0.stride()), p.batch_sfa));
  LayoutSFB lsfb = cute::make_layout(
      lsfb0.shape(), cute::make_stride(get<0>(lsfb0.stride()),
                                       get<1>(lsfb0.stride()), p.batch_sfb));

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {p.M, p.N, p.K, p.L},
      {reinterpret_cast<const ElementA*>(p.A), sa,
       reinterpret_cast<const ElementB*>(p.B), sb, p.SFA, lsfa, p.SFB, lsfb},
      {{}, nullptr, StrideC{}, reinterpret_cast<ElementD*>(p.D), sd}};
  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.beta = 0.0f;

  using Kernel = typename Gemm::GemmKernel;
  TORCH_CHECK(Kernel::can_implement(args),
              "fp8 blockwise gemm: unsupported problem M=", p.M, " N=", p.N,
              " K=", p.K, " L=", p.L);
  // GemmUniversalAdapter::initialize() would set the smem attribute on every
  // call; on the decode/short-prefill path that host cost is comparable to
  // the kernel, so set it once per device and launch from Params directly.
  static std::once_flag smem_once[64];
  std::call_once(smem_once[at::cuda::current_device()], [] {
    C10_CUDA_CHECK(
        cudaFuncSetAttribute(cutlass::device_kernel<Kernel>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             Kernel::SharedStorageSize));
  });
  const size_t ws_bytes = Kernel::get_workspace_size(args);
  at::Tensor ws;
  if (ws_bytes > 0) {
    ws = at::empty({static_cast<int64_t>(ws_bytes)},
                   at::TensorOptions().dtype(at::kByte).device(at::kCUDA));
    TORCH_CHECK(Kernel::initialize_workspace(args, ws.data_ptr(), stream) ==
                    cutlass::Status::kSuccess,
                "fp8 blockwise gemm: workspace init failed");
  }
  auto params =
      Kernel::to_underlying_arguments(args, ws_bytes ? ws.data_ptr() : nullptr);
  TORCH_CHECK(Gemm::run(params, stream, nullptr,
                        /*launch_with_pdl=*/true) == cutlass::Status::kSuccess,
              "fp8 blockwise gemm: launch failed");
}

// Tile configs picked from a sweep of the GLM-5.x expert shapes on RTX PRO
// 6000 (188 SMs). `exclusive` = the call is chained to run alone on the
// device (see DeviceChain) or no other stream has an FP8 expert in flight;
// otherwise several exec streams run experts concurrently, where persistent
// stream-K kernels interfere (measured
// on the 4-stream expert_layer harness: 128x128 tiles beat both stream-K and
// 64x128 ping-pong above 256 tokens, ping-pong wins below).
void blockwise_gemm_sm120_pro6000(const GemmProblem& p, bool exclusive,
                                  cudaStream_t stream) {
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t tiles =
      int64_t((p.M + 127) / 128) * (p.N / 128) * std::max(p.L, 1);
  if (tiles >= 6 * int64_t(sms)) {
    run_gemm<Sm120CoopGemm>(p, stream);
  } else if (exclusive && p.K >= 4864) {
    // Long-K gate/up with only a few waves of tiles: stream-K splits K.
    run_gemm<Sm120StreamKGemm>(p, stream);
  } else if (p.M <= (exclusive ? 2048 : 256)) {
    run_gemm<Sm120PingGemm>(p, stream);
  } else {
    run_gemm<Sm120CoopGemm>(p, stream);
  }
}

// Large-M experts already fill every SM; when several exec streams run them
// concurrently their working sets (weights + activations) thrash the L2.
// Chain those calls through one per-device event so they run back to back
// on the device (no host blocking). Skipped while a stream is capturing.
int64_t pro6000_serialize_min_tokens() {
  static const int64_t v = [] {
    const char* env = std::getenv("MOE_FP8_PRO6000_SERIALIZE_MIN_TOKENS");
    return env ? std::atoll(env) : int64_t(3072);
  }();
  return v;
}

class DeviceChain {
 public:
  bool Enter(cudaStream_t stream, int device) {
    cudaStreamCaptureStatus status;
    if (cudaStreamIsCapturing(stream, &status) != cudaSuccess ||
        status != cudaStreamCaptureStatusNone) {
      return false;
    }
    mu_.lock();
    if (event_ == nullptr) {
      c10::cuda::CUDAGuard guard(device);
      C10_CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
    } else if (recorded_) {
      C10_CUDA_CHECK(cudaStreamWaitEvent(stream, event_, 0));
    }
    return true;
  }
  void Exit(cudaStream_t stream) noexcept {
    recorded_ = cudaEventRecord(event_, stream) == cudaSuccess;
    mu_.unlock();
  }

 private:
  std::mutex mu_;
  cudaEvent_t event_ = nullptr;
  bool recorded_ = false;
};

DeviceChain& device_chain(int device) {
  static DeviceChain chains[64];
  return chains[device];
}

struct ChainScope {
  DeviceChain* chain = nullptr;
  cudaStream_t stream = nullptr;
  ~ChainScope() {
    if (chain != nullptr) chain->Exit(stream);
  }
};
// Tracks the GEMM-path expert calls in flight per device. Persistent
// stream-K kernels are the fastest choice when an expert owns the GPU, but
// interfere when several exec streams run experts at once; a call checks
// whether another stream still has one in flight (non-blocking queries).
class InflightRegistry {
 public:
  bool OtherStreamBusy(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& slot : slots_) {
      if (slot.stream == nullptr || slot.stream == stream) continue;
      const cudaError_t q = cudaEventQuery(slot.event);
      if (q == cudaErrorNotReady) return true;
      if (q != cudaSuccess) (void)cudaGetLastError();
    }
    return false;
  }
  void Record(cudaStream_t stream, int device) {
    std::lock_guard<std::mutex> lock(mu_);
    Slot& slot = slots_[next_++ % kSlots];
    if (slot.event == nullptr) {
      c10::cuda::CUDAGuard guard(device);
      C10_CUDA_CHECK(
          cudaEventCreateWithFlags(&slot.event, cudaEventDisableTiming));
    }
    C10_CUDA_CHECK(cudaEventRecord(slot.event, stream));
    slot.stream = stream;
  }

 private:
  static constexpr int kSlots = 16;
  struct Slot {
    cudaEvent_t event = nullptr;
    cudaStream_t stream = nullptr;
  };
  std::mutex mu_;
  Slot slots_[kSlots];
  unsigned next_ = 0;
};

InflightRegistry& inflight(int device) {
  static InflightRegistry registries[64];
  return registries[device];
}

bool is_capturing(cudaStream_t stream) {
  cudaStreamCaptureStatus status;
  return cudaStreamIsCapturing(stream, &status) != cudaSuccess ||
         status != cudaStreamCaptureStatusNone;
}
#endif  // MOE_ENABLE_SM120_FP8

const uint8_t* weight_ptr(const torch::Tensor& w) {
  return reinterpret_cast<const uint8_t*>(w.data_ptr());
}

}  // namespace fp8_moe_detail

using namespace fp8_moe_detail;

bool fp8_moe_sm120_pro6000_supported(int64_t hidden, int64_t inter,
                                     int device) {
#if defined(MOE_ENABLE_SM120_FP8)
  if (hidden <= 0 || inter <= 0 || hidden % kBlock || inter % kBlock)
    return false;
  using moe_infinity::hw::Gate;
  using moe_infinity::hw::Supports;
  return Supports(Gate::kSm120, device) && Supports(Gate::kRtxPro6000, device);
#else
  (void)hidden;
  (void)inter;
  (void)device;
  return false;
#endif
}

void fused_fp8_moe_ffn_sm120_pro6000_into(
    const torch::Tensor& hidden, const torch::Tensor& gate_w,
    const torch::Tensor& gate_s, const torch::Tensor& up_w,
    const torch::Tensor& up_s, const torch::Tensor& down_w,
    const torch::Tensor& down_s, torch::Tensor& output, cudaStream_t stream) {
  const int64_t M = hidden.size(0);
  const int64_t H = hidden.size(1);
  const int64_t I = gate_w.size(0);
  TORCH_CHECK(fp8_moe_sm120_pro6000_supported(H, I, hidden.get_device()),
              "fp8 expert: the sm120/RTX PRO 6000 kernels do not run on this "
              "device or build; use the dequant-to-BF16 path");
  TORCH_CHECK(hidden.scalar_type() == at::kBFloat16 &&
                  output.scalar_type() == at::kBFloat16,
              "fp8 expert: activations must be bf16");
  TORCH_CHECK(hidden.stride(1) == 1 && output.stride(1) == 1 &&
                  hidden.stride(0) == H && output.stride(0) == H,
              "fp8 expert: hidden/output must be contiguous [M, H]");
  TORCH_CHECK(gate_w.size(1) == H && up_w.size(0) == I && up_w.size(1) == H &&
                  down_w.size(0) == H && down_w.size(1) == I,
              "fp8 expert: weight shapes do not match [I,H],[I,H],[H,I]");
  TORCH_CHECK(gate_w.element_size() == 1 && up_w.element_size() == 1 &&
                  down_w.element_size() == 1 && gate_w.is_contiguous() &&
                  up_w.is_contiguous() && down_w.is_contiguous(),
              "fp8 expert: weights must be contiguous 1-byte e4m3");
  TORCH_CHECK(
      gate_s.scalar_type() == at::kFloat && up_s.scalar_type() == at::kFloat &&
          down_s.scalar_type() == at::kFloat && gate_s.is_contiguous() &&
          up_s.is_contiguous() && down_s.is_contiguous(),
      "fp8 expert: scales must be contiguous fp32");
  TORCH_CHECK(gate_s.numel() == (I / kBlock) * (H / kBlock) &&
                  up_s.numel() == gate_s.numel() &&
                  down_s.numel() == gate_s.numel(),
              "fp8 expert: scale shapes must be [N/128, K/128]");
  TORCH_CHECK(H % kBlock == 0 && I % kBlock == 0,
              "fp8 expert: hidden and intermediate sizes must be multiples "
              "of 128");
  if (M == 0) return;

  const auto opts = hidden.options().dtype(at::kByte);
  auto* x = reinterpret_cast<const __nv_bfloat16*>(hidden.data_ptr());
  auto* out = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());

  if (M <= kFp8Pro6000GemvMaxTokens) {
    at::Tensor inter = at::empty({M, I}, hidden.options());
    auto* ip = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());
    launch_gemv_pro6000<2>(M, x, H, weight_ptr(gate_w),
                           gate_s.data_ptr<float>(), weight_ptr(up_w),
                           up_s.data_ptr<float>(), ip, I, I, H, stream);
    launch_gemv_pro6000<1>(M, ip, I, weight_ptr(down_w),
                           down_s.data_ptr<float>(), nullptr, nullptr, out, H,
                           H, I, stream);
    return;
  }

#if defined(MOE_ENABLE_SM120_FP8)
  const int64_t HB = H / kBlock;
  const int64_t IB = I / kBlock;
  // Workspace (one allocation, 256-byte aligned regions).
  auto align = [](int64_t b) { return (b + 255) / 256 * 256; };
  const int64_t sz_gu = align(M * 2 * I * 2);  // bf16 [M, 2I]
  const int64_t sz_xq = align(M * H);          // e4m3 [M, H]
  const int64_t sz_aq = align(M * I);          // e4m3 [M, I]
  const int64_t sz_sfx = align(HB * M * 4);    // fp32 [H/128, M]
  const int64_t sz_sfa = align(IB * M * 4);    // fp32 [I/128, M]
  const int64_t sz_sfb = align(3 * IB * HB * 4);
  at::Tensor ws =
      at::empty({sz_gu + sz_xq + sz_aq + sz_sfx + sz_sfa + sz_sfb}, opts);
  uint8_t* base = reinterpret_cast<uint8_t*>(ws.data_ptr());
  auto* gu = reinterpret_cast<__nv_bfloat16*>(base);
  uint8_t* xq = base + sz_gu;
  uint8_t* aq = xq + sz_xq;
  float* sfx = reinterpret_cast<float*>(aq + sz_aq);
  float* sfa =
      reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(sfx) + sz_sfx);
  float* sfb =
      reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(sfa) + sz_sfa);
  float* sfb_gate = sfb;
  float* sfb_up = sfb + IB * HB;
  float* sfb_down = sfb + 2 * IB * HB;

  // gate and up run as one L = 2 batched GEMM: the batch stride of B is the
  // distance between the two weight tensors, so batch 0 must be the lower
  // address (TMA strides are unsigned). The silu*up kernel reads the halves
  // of [M, 2I] in the matching order.
  const uint8_t* gp = weight_ptr(gate_w);
  const uint8_t* upp = weight_ptr(up_w);
  const bool gate_first = gp < upp;
  const uint8_t* b0 = gate_first ? gp : upp;
  const int64_t b_stride = gate_first ? upp - gp : gp - upp;
  const bool batched = b_stride % 16 == 0 &&
                       reinterpret_cast<uintptr_t>(b0) % 16 == 0 &&
                       b_stride < (int64_t(1) << 40);

  const int device = hidden.get_device();
  ChainScope chain_scope;
  if (M >= pro6000_serialize_min_tokens() &&
      device_chain(device).Enter(stream, device)) {
    chain_scope.chain = &device_chain(device);
    chain_scope.stream = stream;
  }
  const bool capturing = is_capturing(stream);
  const bool exclusive =
      chain_scope.chain != nullptr ||
      (!capturing && !inflight(device).OtherStreamBusy(stream));

  launch_quant<false>(x, H, 0, 0, xq, H, sfx, M, H,
                      {gate_s.data_ptr<float>(), gate_first ? sfb_gate : sfb_up,
                       (int)IB, (int)HB},
                      {up_s.data_ptr<float>(), gate_first ? sfb_up : sfb_gate,
                       (int)IB, (int)HB},
                      {down_s.data_ptr<float>(), sfb_down, (int)HB, (int)IB},
                      DeadRegion{}, stream);

  // Column halves of gu: batch 0 -> [0, I), batch 1 -> [I, 2I).
  if (batched) {
    blockwise_gemm_sm120_pro6000(
        {(int)M, (int)I, (int)H, 2, xq, H, 0, b0, H, b_stride, sfx, 0, sfb,
         (int32_t)(IB * HB), gu, 2 * I, I},
        exclusive, stream);
  } else {
    blockwise_gemm_sm120_pro6000(
        {(int)M, (int)I, (int)H, 1, xq, H, 0, gate_first ? gp : upp, H, 0, sfx,
         0, sfb, 0, gu, 2 * I, 0},
        exclusive, stream);
    blockwise_gemm_sm120_pro6000(
        {(int)M, (int)I, (int)H, 1, xq, H, 0, gate_first ? upp : gp, H, 0, sfx,
         0, sfb + IB * HB, 0, gu + I, 2 * I, 0},
        exclusive, stream);
  }

  const int gate_off = gate_first ? 0 : (int)I;
  const int up_off = gate_first ? (int)I : 0;
  launch_quant<true>(gu, 2 * I, gate_off, up_off, aq, I, sfa, M, I,
                     {nullptr, nullptr, 0, 0}, {nullptr, nullptr, 0, 0},
                     {nullptr, nullptr, 0, 0}, DeadRegion{xq, sz_xq}, stream);

  blockwise_gemm_sm120_pro6000(
      {(int)M, (int)H, (int)I, 1, aq, I, 0, weight_ptr(down_w), I, 0, sfa, 0,
       sfb_down, 0, out, H, 0},
      exclusive, stream);

  if (!capturing) inflight(device).Record(stream, device);

  const c10::cuda::CUDAStream cur = c10::cuda::getCurrentCUDAStream();
  if (cur.stream() != stream) {
    ws.record_stream(
        c10::cuda::getStreamFromExternal(stream, hidden.get_device()));
  }
#else
  TORCH_CHECK(false,
              "fp8 expert: this build has no sm_120a FP8 GEMM (rebuild with "
              "MOE_ENABLE_SM120=1)");
#endif
}
