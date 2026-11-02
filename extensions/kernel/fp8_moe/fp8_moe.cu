// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team
//
// Native FP8 (e4m3, 128x128 block-scaled) SwiGLU expert, see fp8_moe.h.

#include "fp8_moe.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <cstdlib>
#include <mutex>

#if defined(MOE_ENABLE_SM120_FP8)
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

__device__ __forceinline__ uint4 ld_stream_u4(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

// 4 packed e4m3 bytes (byte 0 = lowest element) -> 4 floats, exact.
__device__ __forceinline__ void e4m3x4_to_float(uint32_t v, float* f) {
  __half2_raw lo = __nv_cvt_fp8x2_to_halfraw2(
      static_cast<__nv_fp8x2_storage_t>(v & 0xFFFFu), __NV_E4M3);
  __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2(
      static_cast<__nv_fp8x2_storage_t>(v >> 16), __NV_E4M3);
  float2 a = __half22float2(*reinterpret_cast<__half2*>(&lo));
  float2 b = __half22float2(*reinterpret_cast<__half2*>(&hi));
  f[0] = a.x;
  f[1] = a.y;
  f[2] = b.x;
  f[3] = b.y;
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
// Weight-only FP8 GEMV (decode, M <= kFp8GemvMaxTokens)
//
// out[m, n] = act(sum_k x[m, k] * W0[n, k] * S0[n/128, k/128], same for W1)
// One warp owns R output rows of each of the NMAT weight matrices and streams
// them along K with 16-byte loads (lane l covers k = l*16 .. l*16+15 of every
// 512-wide step); U steps are loaded before any math so each lane keeps
// U*NMAT*R*16 bytes in flight. NMAT == 2 fuses silu(W0 x) * (W1 x).
// ---------------------------------------------------------------------------

constexpr int kGemvWarps = 4;

template <int M, int NMAT, int R, int U>
__global__ void __launch_bounds__(kGemvWarps * 32)
    fp8_gemv_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx,
                    const uint8_t* __restrict__ w0,
                    const float* __restrict__ s0,
                    const uint8_t* __restrict__ w1,
                    const float* __restrict__ s1,
                    __nv_bfloat16* __restrict__ out, int64_t ldo, int N,
                    int K) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row0 = (blockIdx.x * kGemvWarps + warp) * R;
  if (row0 >= N) return;
  const int KB = K / kBlock;

  const uint8_t* wp[NMAT];
  const float* sp[NMAT];
  wp[0] = w0;
  sp[0] = s0;
  if constexpr (NMAT > 1) {
    wp[1] = w1;
    sp[1] = s1;
  }

  float acc[NMAT][R][M];
#pragma unroll
  for (int a = 0; a < NMAT; ++a)
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int m = 0; m < M; ++m) acc[a][r][m] = 0.f;

  for (int kbase = 0; kbase < K; kbase += 512 * U) {
    uint4 wv[U][NMAT][R];
    float sc[U][NMAT][R];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int k = kbase + u * 512 + lane * 16;
      const bool ok = k < K;
#pragma unroll
      for (int a = 0; a < NMAT; ++a)
#pragma unroll
        for (int r = 0; r < R; ++r) {
          const int n = min(row0 + r, N - 1);
          wv[u][a][r] = ok ? ld_stream_u4(wp[a] + (int64_t)n * K + k)
                           : make_uint4(0, 0, 0, 0);
          sc[u][a][r] =
              ok ? __ldg(sp[a] + (n / kBlock) * KB + k / kBlock) : 0.f;
        }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int k = kbase + u * 512 + lane * 16;
      if (k >= K) continue;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float xf[M][8];
#pragma unroll
        for (int m = 0; m < M; ++m) {
          const uint4 xv =
              __ldg(reinterpret_cast<const uint4*>(x + m * ldx + k + h * 8));
          bf16x8_to_float(xv, xf[m]);
        }
#pragma unroll
        for (int a = 0; a < NMAT; ++a)
#pragma unroll
          for (int r = 0; r < R; ++r) {
            float wf[8];
            const uint4& v = wv[u][a][r];
            e4m3x4_to_float(h == 0 ? v.x : v.z, wf);
            e4m3x4_to_float(h == 0 ? v.y : v.w, wf + 4);
#pragma unroll
            for (int m = 0; m < M; ++m) {
              float p = 0.f;
#pragma unroll
              for (int j = 0; j < 8; ++j) p = fmaf(wf[j], xf[m][j], p);
              acc[a][r][m] = fmaf(p, sc[u][a][r], acc[a][r][m]);
            }
          }
      }
    }
  }

#pragma unroll
  for (int a = 0; a < NMAT; ++a)
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int m = 0; m < M; ++m) {
        float v = acc[a][r][m];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
          v += __shfl_xor_sync(0xffffffffu, v, o);
        acc[a][r][m] = v;
      }

      // Spread the R*M stores over lanes.
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) {
      if (lane == (r * M + m) % 32 && row0 + r < N) {
        float o = acc[0][r][m];
        if constexpr (NMAT == 2) o = silu(o) * acc[1][r][m];
        out[m * ldo + row0 + r] = __float2bfloat16(o);
      }
    }
}

template <int M, int NMAT, int R, int U>
void launch_gemv(const __nv_bfloat16* x, int64_t ldx, const uint8_t* w0,
                 const float* s0, const uint8_t* w1, const float* s1,
                 __nv_bfloat16* out, int64_t ldo, int N, int K,
                 cudaStream_t stream) {
  const int rows_per_cta = kGemvWarps * R;
  const int grid = (N + rows_per_cta - 1) / rows_per_cta;
  fp8_gemv_kernel<M, NMAT, R, U><<<grid, kGemvWarps * 32, 0, stream>>>(
      x, ldx, w0, s0, w1, s1, out, ldo, N, K);
}

// (rows per warp R, unrolled 512-wide K steps U) per token count, from a
// sweep of 8-32 GLM-5.x experts on 4 streams (RTX PRO 6000): the gate/up
// kernel wants fewer registers as M grows, the down GEMV (K = I is short)
// wants more rows per warp.
template <int NMAT>
void dispatch_gemv(int M, const __nv_bfloat16* x, int64_t ldx,
                   const uint8_t* w0, const float* s0, const uint8_t* w1,
                   const float* s1, __nv_bfloat16* out, int64_t ldo, int N,
                   int K, cudaStream_t stream) {
#define FP8_GEMV_CASE(MM, R2, U2, R1, U1)                                   \
  case MM:                                                                  \
    if constexpr (NMAT == 2) {                                              \
      launch_gemv<MM, NMAT, R2, U2>(x, ldx, w0, s0, w1, s1, out, ldo, N, K, \
                                    stream);                                \
    } else {                                                                \
      launch_gemv<MM, NMAT, R1, U1>(x, ldx, w0, s0, w1, s1, out, ldo, N, K, \
                                    stream);                                \
    }                                                                       \
    break;
  switch (M) {
    FP8_GEMV_CASE(1, 2, 4, 4, 1)
    FP8_GEMV_CASE(2, 2, 4, 4, 1)
    FP8_GEMV_CASE(3, 2, 2, 4, 1)
    FP8_GEMV_CASE(4, 2, 2, 4, 1)
    FP8_GEMV_CASE(5, 1, 1, 4, 1)
    FP8_GEMV_CASE(6, 1, 1, 4, 1)
    FP8_GEMV_CASE(7, 1, 1, 4, 1)
    FP8_GEMV_CASE(8, 1, 1, 4, 1)
    default:
      TORCH_CHECK(false, "fp8 gemv: unsupported token count ", M);
  }
#undef FP8_GEMV_CASE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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

constexpr int kQuantThreads = 256;
constexpr int kQuantGroupsPerBlock = kQuantThreads / 16;

template <bool SILU_MUL>
__global__ void __launch_bounds__(kQuantThreads)
    fp8_quant_act_kernel(const __nv_bfloat16* __restrict__ in, int64_t ldi,
                         int gate_off, int up_off, uint8_t* __restrict__ q,
                         int64_t ldq, float* __restrict__ sf, int M, int K,
                         int quant_blocks, ScaleCopy c0, ScaleCopy c1,
                         ScaleCopy c2) {
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
                  ScaleCopy c0, ScaleCopy c1, ScaleCopy c2,
                  cudaStream_t stream) {
  const int groups = M * (K / kBlock);
  const int quant_blocks =
      (groups + kQuantGroupsPerBlock - 1) / kQuantGroupsPerBlock;
  const int copy_elems = c0.NB * c0.KB + c1.NB * c1.KB + c2.NB * c2.KB;
  const int copy_blocks =
      copy_elems > 0 ? std::min(16, (copy_elems + 255) / 256) : 0;
  fp8_quant_act_kernel<SILU_MUL>
      <<<quant_blocks + copy_blocks, kQuantThreads, 0, stream>>>(
          in, ldi, gate_off, up_off, q, ldq, sf, M, K, quant_blocks, c0, c1,
          c2);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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

using CoopGemm =
    BlockwiseGemm<Shape<_128, _128, _128>,
                  cutlass::gemm::KernelScheduleSm120Blockwise>::Gemm;
using StreamKGemm = BlockwiseGemm<Shape<_128, _128, _128>,
                                  cutlass::gemm::KernelScheduleSm120Blockwise,
                                  cutlass::gemm::StreamKScheduler>::Gemm;
using PingGemm = BlockwiseGemm<
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

  Gemm gemm;
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess,
              "fp8 blockwise gemm: unsupported problem M=", p.M, " N=", p.N,
              " K=", p.K, " L=", p.L);
  const size_t ws_bytes = Gemm::get_workspace_size(args);
  at::Tensor ws;
  if (ws_bytes > 0) {
    ws = at::empty({static_cast<int64_t>(ws_bytes)},
                   at::TensorOptions().dtype(at::kByte).device(at::kCUDA));
  }
  TORCH_CHECK(gemm.initialize(args, ws_bytes ? ws.data_ptr() : nullptr,
                              stream) == cutlass::Status::kSuccess,
              "fp8 blockwise gemm: initialize failed");
  TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess,
              "fp8 blockwise gemm: launch failed");
}

// Tile configs picked from a sweep of the GLM-5.x expert shapes on RTX PRO
// 6000 (188 SMs). `exclusive` = the call is chained to run alone on the
// device (see DeviceChain); otherwise several exec streams usually run
// experts concurrently, where persistent stream-K kernels interfere (measured
// on the 4-stream expert_layer harness: 128x128 tiles beat both stream-K and
// 64x128 ping-pong above 256 tokens, ping-pong wins below).
void blockwise_gemm(const GemmProblem& p, bool exclusive, cudaStream_t stream) {
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t tiles =
      int64_t((p.M + 127) / 128) * (p.N / 128) * std::max(p.L, 1);
  if (tiles >= 6 * int64_t(sms)) {
    run_gemm<CoopGemm>(p, stream);
  } else if (exclusive && p.K >= 4864) {
    // Long-K gate/up with only a few waves of tiles: stream-K splits K.
    run_gemm<StreamKGemm>(p, stream);
  } else if (p.M <= 256) {
    run_gemm<PingGemm>(p, stream);
  } else {
    run_gemm<CoopGemm>(p, stream);
  }
}

// Large-M experts already fill every SM; when several exec streams run them
// concurrently their working sets (weights + activations) thrash the L2.
// Chain those calls through one per-device event so they run back to back
// on the device (no host blocking). Skipped while a stream is capturing.
int64_t serialize_min_tokens() {
  static const int64_t v = [] {
    const char* env = std::getenv("MOE_FP8_SERIALIZE_MIN_TOKENS");
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
#endif  // MOE_ENABLE_SM120_FP8

const uint8_t* weight_ptr(const torch::Tensor& w) {
  return reinterpret_cast<const uint8_t*>(w.data_ptr());
}

}  // namespace fp8_moe_detail

using namespace fp8_moe_detail;

bool fp8_moe_native_supported(int64_t hidden, int64_t inter) {
#if defined(MOE_ENABLE_SM120_FP8)
  if (hidden <= 0 || inter <= 0 || hidden % kBlock || inter % kBlock)
    return false;
  const auto* prop = at::cuda::getCurrentDeviceProperties();
  return prop->major == 12 && prop->minor == 0;
#else
  (void)hidden;
  (void)inter;
  return false;
#endif
}

void fused_fp8_moe_ffn_into(
    const torch::Tensor& hidden, const torch::Tensor& gate_w,
    const torch::Tensor& gate_s, const torch::Tensor& up_w,
    const torch::Tensor& up_s, const torch::Tensor& down_w,
    const torch::Tensor& down_s, torch::Tensor& output, cudaStream_t stream) {
  const int64_t M = hidden.size(0);
  const int64_t H = hidden.size(1);
  const int64_t I = gate_w.size(0);
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

  if (M <= kFp8GemvMaxTokens) {
    at::Tensor inter = at::empty({M, I}, hidden.options());
    auto* ip = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());
    dispatch_gemv<2>(M, x, H, weight_ptr(gate_w), gate_s.data_ptr<float>(),
                     weight_ptr(up_w), up_s.data_ptr<float>(), ip, I, I, H,
                     stream);
    dispatch_gemv<1>(M, ip, I, weight_ptr(down_w), down_s.data_ptr<float>(),
                     nullptr, nullptr, out, H, H, I, stream);
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
  if (M >= serialize_min_tokens() &&
      device_chain(device).Enter(stream, device)) {
    chain_scope.chain = &device_chain(device);
    chain_scope.stream = stream;
  }
  const bool exclusive = chain_scope.chain != nullptr;

  launch_quant<false>(x, H, 0, 0, xq, H, sfx, M, H,
                      {gate_s.data_ptr<float>(), gate_first ? sfb_gate : sfb_up,
                       (int)IB, (int)HB},
                      {up_s.data_ptr<float>(), gate_first ? sfb_up : sfb_gate,
                       (int)IB, (int)HB},
                      {down_s.data_ptr<float>(), sfb_down, (int)HB, (int)IB},
                      stream);

  // Column halves of gu: batch 0 -> [0, I), batch 1 -> [I, 2I).
  if (batched) {
    blockwise_gemm({(int)M, (int)I, (int)H, 2, xq, H, 0, b0, H, b_stride, sfx,
                    0, sfb, (int32_t)(IB * HB), gu, 2 * I, I},
                   exclusive, stream);
  } else {
    blockwise_gemm({(int)M, (int)I, (int)H, 1, xq, H, 0, gate_first ? gp : upp,
                    H, 0, sfx, 0, sfb, 0, gu, 2 * I, 0},
                   exclusive, stream);
    blockwise_gemm({(int)M, (int)I, (int)H, 1, xq, H, 0, gate_first ? upp : gp,
                    H, 0, sfx, 0, sfb + IB * HB, 0, gu + I, 2 * I, 0},
                   exclusive, stream);
  }

  const int gate_off = gate_first ? 0 : (int)I;
  const int up_off = gate_first ? (int)I : 0;
  launch_quant<true>(gu, 2 * I, gate_off, up_off, aq, I, sfa, M, I,
                     {nullptr, nullptr, 0, 0}, {nullptr, nullptr, 0, 0},
                     {nullptr, nullptr, 0, 0}, stream);

  blockwise_gemm({(int)M, (int)H, (int)I, 1, aq, I, 0, weight_ptr(down_w), I, 0,
                  sfa, 0, sfb_down, 0, out, H, 0},
                 exclusive, stream);

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
