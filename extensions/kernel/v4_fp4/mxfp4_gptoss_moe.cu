// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// W4A16 MXFP4 GEMM for GPT-OSS experts (BF16 activations, FP32 accumulate).
//
// RTX PRO 6000 only: the instructions are generic sm_80 / sm_90+ ISA, but the
// tile configs (MXFP4_PRO6000_TILE_CONFIGS, pick_pro6000_config), the 99 KiB
// smem stage budgets and the persistent 188-SM grids were tuned on the RTX
// PRO 6000 Blackwell. The callers gate on RTX_PRO_6000; other devices run the
// previous de-quantise + matmul path.
//
// Weights stay in their storage format: blocks [N, K/2] uint8 (two E2M1 codes
// per byte, low nibble first) and E8M0 scales [N, K/32]. Every byte is read
// once per GEMM and de-quantised in registers straight into mma.sync A
// fragments (swap-AB: weights are the MMA M dimension, tokens the N
// dimension), so small token batches waste no MMA rows.
//
// K permutation: a lane owns 8 consecutive K of one row (one 32-bit word of
// codes). The word is expanded into four BF16x2 pairs (k, k+4) and the
// activation side is permuted identically with byte_perm, so the dot product
// is unchanged.
//
// gate_up epilogue: rows 2j / 2j+1 (interleaved gate / up) are mapped to MMA
// rows g / g+8, so the clamped SwiGLU-OAI (alpha 1.702, limit 7) and both
// biases are applied thread-locally on the FP32 accumulators.

#include "mxfp4_gptoss_moe.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdint>

namespace gptoss_mxfp4 {
namespace {

constexpr uint32_t kBf16x2Pow126 = 0x7E807E80u;
constexpr uint32_t kBf16x2NegZero = 0x80008000u;

__device__ __forceinline__ uint32_t bf16x2_fma(uint32_t a, uint32_t b,
                                               uint32_t c) {
  uint32_t d;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

// E8M0 byte -> BF16x2 2^(e-127); clamped like the legacy de-quantiser.
__device__ __forceinline__ uint32_t e8m0_bf16x2(uint32_t e) {
  e = min(max(e, 1u), 254u);
  return (e << 7) | (e << 23);
}

// Codes I and I+4 of the word -> BF16x2 (low = code I, high = code I+4),
// multiplied by the block scale. The 3 magnitude bits are dropped into the
// BF16 exponent/mantissa boundary (a subnormal for E2M1 exponent 0), and the
// exact multiply by 2^126 restores the E2M1 value.
template <int I>
__device__ __forceinline__ uint32_t e2m1_pair(uint32_t w, uint32_t scale2) {
  uint32_t a;
  if constexpr (I <= 1) {
    a = w << (6 - 4 * I);
  } else {
    a = w >> (4 * I - 6);
  }
  uint32_t b = w << (12 - 4 * I);
  uint32_t raw = (a & 0x01C001C0u) | (b & 0x80008000u);
  return bf16x2_fma(bf16x2_fma(raw, kBf16x2Pow126, kBf16x2NegZero), scale2,
                    kBf16x2NegZero);
}

// A fragments (rows g and g+8) of the two k16 MMA steps covered by one word.
__device__ __forceinline__ void make_a(uint32_t wg, uint32_t wg8, uint32_t sg,
                                       uint32_t sg8, uint32_t (&a)[2][4]) {
  a[0][0] = e2m1_pair<0>(wg, sg);
  a[0][1] = e2m1_pair<0>(wg8, sg8);
  a[0][2] = e2m1_pair<1>(wg, sg);
  a[0][3] = e2m1_pair<1>(wg8, sg8);
  a[1][0] = e2m1_pair<2>(wg, sg);
  a[1][1] = e2m1_pair<2>(wg8, sg8);
  a[1][2] = e2m1_pair<3>(wg, sg);
  a[1][3] = e2m1_pair<3>(wg8, sg8);
}

// Matching B fragments from 8 consecutive BF16 activations x0..x7.
__device__ __forceinline__ void make_b(const uint4& x, uint32_t (&b)[2][2]) {
  b[0][0] = __byte_perm(x.x, x.z, 0x5410);  // (x0, x4)
  b[0][1] = __byte_perm(x.x, x.z, 0x7632);  // (x1, x5)
  b[1][0] = __byte_perm(x.y, x.w, 0x5410);  // (x2, x6)
  b[1][1] = __byte_perm(x.y, x.w, 0x7632);  // (x3, x7)
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4],
                                         const uint32_t (&b)[2]) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint4 ldg_stream(const void* p) {
  uint4 v;
  // Not volatile: weights are immutable, so the compiler may batch loads.
  asm("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "l"(p));
  return v;
}

// Activations may be the previous kernel's output (PDL): a volatile, coherent
// load cannot be hoisted above griddepcontrol.wait, unlike ld.global.nc.
__device__ __forceinline__ uint4 ldg_act(const void* p) {
  uint4 v;
  asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

// Release/acquire ordering for the stream-K hand-off (cheaper than the
// sequentially consistent fence behind __threadfence()).
__device__ __forceinline__ void fence_acq_rel_gpu() {
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
}

__device__ __forceinline__ void pdl_launch_dependents() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void prefetch_l2_bulk(const void* p, uint32_t n) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(n)
               : "memory");
#endif
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem,
                                           bool valid) {
  uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  int n = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s),
               "l"(gmem), "r"(n));
}

__device__ __forceinline__ void cp_async4(void* smem, const void* gmem,
                                          bool valid) {
  uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  int n = valid ? 4 : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(s),
               "l"(gmem), "r"(n));
}

__device__ __forceinline__ void cp_async8(void* smem, const void* gmem,
                                          bool valid) {
  uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  int n = valid ? 8 : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(s),
               "l"(gmem), "r"(n));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Drops 128-B L2 lines without writing them back (their data is dead).
__device__ __forceinline__ void discard_l2(const void* p, size_t bytes, int tid,
                                           int nthreads) {
  const char* c = static_cast<const char*>(p);
  for (size_t i = static_cast<size_t>(tid) * 128; i < bytes;
       i += static_cast<size_t>(nthreads) * 128)
    asm volatile("discard.global.L2 [%0], 128;" ::"l"(c + i) : "memory");
}

__device__ __forceinline__ float swiglu_oai(float gate, float up) {
  // Match the BF16 module: gate/up are rounded to BF16 after the bias add.
  gate = __bfloat162float(__float2bfloat16(gate));
  up = __bfloat162float(__float2bfloat16(up));
  gate = fminf(gate, 7.0f);
  up = fminf(fmaxf(up, -7.0f), 7.0f);
  float glu = gate / (1.0f + __expf(-1.702f * gate));
  return (up + 1.0f) * glu;
}

// --------------------------------------------------------------------------
// Small-M kernel (M <= 8 * NT): one CTA per 16 * MT weight rows, WK warps
// split K in 128-wide granules (lane t owns MX block 4G + t). Each warp
// issues the loads of U granules before computing, operands come straight
// from global memory (activation fragments are reused across the MT row
// tiles), and partial sums are reduced across warps in shared memory.
// --------------------------------------------------------------------------
template <int WK, int U, int NT, int MT, bool SWIGLU, bool PDL_WAIT>
__global__ void __launch_bounds__(WK * 32)
    mxfp4_gemv_pro6000_kernel(const __nv_bfloat16* __restrict__ x, int M, int K,
                              const uint8_t* __restrict__ wq,
                              const uint8_t* __restrict__ ws,
                              const __nv_bfloat16* __restrict__ bias, int N,
                              __nv_bfloat16* __restrict__ y,
                              int prefetch_aligned) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int m0 = blockIdx.x * 16 * MT;
  const int KB = K >> 5;
  const int KP = K >> 1;
  const int nG = (KB + 3) >> 2;
  // blockIdx.y selects a slice of 8 * NT tokens (the other slices re-read
  // the weights from L2).
  {
    const int t_base = blockIdx.y * 8 * NT;
    x += static_cast<size_t>(t_base) * K;
    y += static_cast<size_t>(t_base) * (SWIGLU ? N >> 1 : N);
    M -= t_base;
  }

  // Weight rows of MMA rows g / g+8 in each 16-row tile.
  int rg[MT], rg8[MT];
  bool ok_g[MT], ok_g8[MT];
#pragma unroll
  for (int mi = 0; mi < MT; ++mi) {
    const int base = m0 + 16 * mi;
    int a = SWIGLU ? base + 2 * g : base + g;
    int b = SWIGLU ? a + 1 : a + 8;
    ok_g[mi] = a < N;
    ok_g8[mi] = b < N;
    rg[mi] = ok_g[mi] ? a : 0;
    rg8[mi] = ok_g8[mi] ? b : 0;
  }

  pdl_launch_dependents();
  if constexpr (PDL_WAIT) {
    // Weights do not depend on the previous kernel: start streaming them
    // into L2 while it drains, then wait for its output (our activations).
    if (prefetch_aligned && blockIdx.y == 0 && threadIdx.x < 16 * MT) {
      const int r = m0 + threadIdx.x;
      if (r < N) prefetch_l2_bulk(wq + static_cast<size_t>(r) * KP, KP);
    }
  }

  float acc[MT][NT][4];
#pragma unroll
  for (int mi = 0; mi < MT; ++mi)
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[mi][i][j] = 0.0f;

  const uint4 zero4 = make_uint4(0, 0, 0, 0);
  bool waited = !PDL_WAIT;
  for (int G0 = warp; G0 < nG; G0 += WK * U) {
    uint4 wa[U][MT], wb[U][MT];
    uint32_t sa[U][MT], sb[U][MT];
    int blk[U];
    bool oku[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int b = 4 * (G0 + u * WK) + t;
      oku[u] = b < KB;
      blk[u] = oku[u] ? b : 0;
#pragma unroll
      for (int mi = 0; mi < MT; ++mi) {
        const bool va = oku[u] && ok_g[mi];
        const bool vb = oku[u] && ok_g8[mi];
        wa[u][mi] =
            va ? ldg_stream(wq + static_cast<size_t>(rg[mi]) * KP + 16 * b)
               : zero4;
        wb[u][mi] =
            vb ? ldg_stream(wq + static_cast<size_t>(rg8[mi]) * KP + 16 * b)
               : zero4;
        sa[u][mi] =
            va ? e8m0_bf16x2(__ldg(ws + static_cast<size_t>(rg[mi]) * KB + b))
               : 0u;
        sb[u][mi] =
            vb ? e8m0_bf16x2(__ldg(ws + static_cast<size_t>(rg8[mi]) * KB + b))
               : 0u;
      }
    }
    if (!waited) {
      pdl_wait();
      waited = true;
    }
    uint4 xv[U][NT][4];
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int tok = nt * 8 + g;
        const bool okx = oku[u] && tok < M;
        const uint4* xp = reinterpret_cast<const uint4*>(
            x + static_cast<size_t>(okx ? tok : 0) * K + 32 * blk[u]);
#pragma unroll
        for (int q = 0; q < 4; ++q)
          xv[u][nt][q] = okx ? ldg_act(xp + q) : zero4;
      }
#pragma unroll
    for (int u = 0; u < U; ++u) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        uint32_t bb[NT][2][2];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) make_b(xv[u][nt][q], bb[nt]);
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
          const uint32_t wav[4] = {wa[u][mi].x, wa[u][mi].y, wa[u][mi].z,
                                   wa[u][mi].w};
          const uint32_t wbv[4] = {wb[u][mi].x, wb[u][mi].y, wb[u][mi].z,
                                   wb[u][mi].w};
          uint32_t a[2][4];
          make_a(wav[q], wbv[q], sa[u][mi], sb[u][mi], a);
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            mma_bf16(acc[mi][nt], a[0], bb[nt][0]);
            mma_bf16(acc[mi][nt], a[1], bb[nt][1]);
          }
        }
      }
    }
  }
  if (!waited) pdl_wait();

  if constexpr (WK > 1) {
    __shared__ float red[WK - 1][MT * NT * 4][32];
    if (warp > 0) {
#pragma unroll
      for (int mi = 0; mi < MT; ++mi)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
          for (int j = 0; j < 4; ++j)
            red[warp - 1][(mi * NT + nt) * 4 + j][lane] = acc[mi][nt][j];
    }
    __syncthreads();
    if (warp > 0) return;
#pragma unroll
    for (int w = 0; w < WK - 1; ++w)
#pragma unroll
      for (int mi = 0; mi < MT; ++mi)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
          for (int j = 0; j < 4; ++j)
            acc[mi][nt][j] += red[w][(mi * NT + nt) * 4 + j][lane];
  }

#pragma unroll
  for (int mi = 0; mi < MT; ++mi) {
    const float bg = ok_g[mi] ? __bfloat162float(bias[rg[mi]]) : 0.0f;
    const float bg8 = ok_g8[mi] ? __bfloat162float(bias[rg8[mi]]) : 0.0f;
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tok = nt * 8 + 2 * t + h;
        if (tok >= M) continue;
        if constexpr (SWIGLU) {
          if (ok_g[mi])
            y[static_cast<size_t>(tok) * (N >> 1) + (rg[mi] >> 1)] =
                __float2bfloat16(
                    swiglu_oai(acc[mi][nt][h] + bg, acc[mi][nt][2 + h] + bg8));
        } else {
          if (ok_g[mi])
            y[static_cast<size_t>(tok) * N + rg[mi]] =
                __float2bfloat16(acc[mi][nt][h] + bg);
          if (ok_g8[mi])
            y[static_cast<size_t>(tok) * N + rg8[mi]] =
                __float2bfloat16(acc[mi][nt][2 + h] + bg8);
        }
      }
    }
  }
}

// --------------------------------------------------------------------------
// Tiled kernel: CTA tile 128 weight rows x BT tokens (128 threads), K stepped
// one MX block (32) per iteration. Per iteration each thread de-quantises
// the 32 codes of one weight row into a BF16 smem tile (one iteration ahead
// of its use), and the warps run a plain ldmatrix + mma.sync BF16 mainloop
// on that tile and the cp.async-staged activations, so the de-quant work is
// shared by all warps and overlaps the tensor pipe.
//
// Work is scheduled stream-K style over a persistent grid: tiles
// [0, n_dp) are data-parallel (whole tiles), the K iterations of the
// remaining tiles are split evenly across the grid. A CTA that finishes a
// partial tile writes its FP32 partial to its own workspace slot and bumps
// the tile's counter; the CTA completing the tile sums the other slots and
// runs the epilogue (no CTA ever waits on another). Counters are reset by
// the finishing CTA, so the counter array only has to be zeroed once.
// --------------------------------------------------------------------------
// Side job: per K iteration a CTA bulk-copies one chunk of 16 MX blocks
// (256 B codes + 16 B scales) into a small ring and de-quantises the chunk
// copied two iterations earlier, so its DRAM latency never stalls the
// cp.async pipeline of the GEMM.
constexpr int kSideLanes = 16;
constexpr int kSideRing = 4;
constexpr int kSideLag = 3;  // iterations between bulk copy and use (< ring)
constexpr int kSideChunkBytes = kSideLanes * 17;  // 272: codes, then scales

template <int BN, int BT, int WARPS_N, int WARPS_T, int STAGES>
struct TileCfg {
  static constexpr int kBN = BN;
  static constexpr int kBT = BT;
  static constexpr int kWarpsN = WARPS_N;
  static constexpr int kThreads = 32 * WARPS_N * WARPS_T;
  static constexpr int kSplit = kThreads / BN;  // threads per weight row
  static constexpr int kStages = STAGES;        // raw (cp.async) stages
  static constexpr int kWN = BN / WARPS_N;
  static constexpr int kMT = kWN / 16;
  static constexpr int kWT = BT / WARPS_T;
  static constexpr int kNT = kWT / 8;
  static constexpr int kRegs = kMT * kNT * 4;
  static constexpr int kXStage = BT * 64;     // BT rows x 32 BF16
  static constexpr int kWRaw = BN * 16;       // BN rows x 16 B of codes
  static constexpr int kSRaw = kThreads * 4;  // 4-B word holding the scale
  static constexpr int kRawStage = kXStage + kWRaw + kSRaw;
  static constexpr int kWbf = BN * 64;  // BN rows x 32 BF16
  static constexpr int kWbfOffset = kStages * kRawStage;

  static_assert(kSplit * BN == kThreads && (kSplit == 1 || kSplit == 2),
                "one or two threads per weight row");
  static_assert(kMT * 16 == kWN && kNT * 8 == kWT && kNT % 2 == 0,
                "bad warp tile");

  static size_t smem_bytes(bool swiglu) {
    size_t pipe = static_cast<size_t>(kWbfOffset) + 2 * kWbf;
    size_t out = static_cast<size_t>(BT) * ((swiglu ? BN / 2 : BN) + 8) * 2;
    return std::max(pipe, out);
  }
  // FP32 partial slots: two per CTA (first and last segment).
  __host__ __device__ static size_t slot_floats() {
    return static_cast<size_t>(kRegs) * kThreads;
  }
};

struct StreamK {
  int tiles_n;         // tiles along N (tile id = tn + tiles_n * tt)
  int n_tiles;         // all tiles
  int n_dp;            // tiles [0, n_dp) are processed whole
  long long sk_iters;  // (n_tiles - n_dp) * KB
  float* partials;     // [2][grid][slot_floats] (first / last segment)
  int* counters;       // [n_tiles - n_dp], zero on entry and on exit
  // Optional side job: de-quantise `side_blocks` MX blocks (16 B codes +
  // 1 B scale -> 32 BF16) of another MXFP4 tensor while the GEMM runs, so
  // the DRAM traffic hides under a compute-bound mainloop.
  const uint8_t* side_q;
  const uint8_t* side_s;
  __nv_bfloat16* side_out;
  int side_blocks;  // multiple of 16
  int side_per;     // blocks per CTA, multiple of 16
};

__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(count)
               : "memory");
#endif
}

__device__ __forceinline__ void side_fetch(uint8_t* dst, const uint8_t* codes,
                                           const uint8_t* scales,
                                           uint64_t* bar) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b),
      "r"(kSideChunkBytes)
      : "memory");
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];" ::"r"(d),
      "l"(codes), "r"(kSideLanes * 16), "r"(b)
      : "memory");
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];" ::"r"(d + kSideLanes * 16),
      "l"(scales), "r"(kSideLanes), "r"(b)
      : "memory");
#endif
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  uint32_t done = 0;
  while (!done) {
    asm volatile(
        "{\n .reg .pred p;\n"
        " mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        " selp.u32 %0, 1, 0, p;\n}"
        : "=r"(done)
        : "r"(a), "r"(parity));
  }
#endif
}

// Weight row (inside the 128-row tile) de-quantised into smem row r. For
// SWIGLU the interleaved gate (2j) / up (2j+1) rows of each 16-row group are
// placed at MMA rows j / j+8, so both halves of a pair land in one thread's
// accumulators.
template <bool SWIGLU>
__device__ __forceinline__ int weight_row_of(int r) {
  if constexpr (SWIGLU) {
    return (r & ~15) | (((r & 7) << 1) | ((r >> 3) & 1));
  } else {
    return r;
  }
}

// 16-B chunk c (0..3) of a 64-B smem row, XOR-swizzled so ldmatrix and the
// row-per-thread stores are bank-conflict free.
__device__ __forceinline__ int swz(int row, int c) {
  return row * 64 + ((c ^ ((row >> 1) & 3)) << 4);
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* p) {
  uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(s));
}

// 8 E2M1 codes (k .. k+7) -> 4 BF16x2 words in natural K order, scaled.
__device__ __forceinline__ uint4 e2m1x8_bf16(uint32_t w, uint32_t s2) {
  const uint32_t p0 = e2m1_pair<0>(w, s2);  // (k0, k4)
  const uint32_t p1 = e2m1_pair<1>(w, s2);  // (k1, k5)
  const uint32_t p2 = e2m1_pair<2>(w, s2);  // (k2, k6)
  const uint32_t p3 = e2m1_pair<3>(w, s2);  // (k3, k7)
  uint4 o;
  o.x = __byte_perm(p0, p1, 0x5410);
  o.y = __byte_perm(p2, p3, 0x5410);
  o.z = __byte_perm(p0, p1, 0x7632);
  o.w = __byte_perm(p2, p3, 0x7632);
  return o;
}

__device__ __forceinline__ void mma_bf16_2(float (&c)[4],
                                           const uint32_t (&a)[4], uint32_t b0,
                                           uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ long long sk_begin(long long c, long long T,
                                              long long G) {
  return c * T / G;
}

// CTA whose stream-K range contains iteration `it`.
__device__ __forceinline__ int sk_owner(long long it, long long T,
                                        long long G) {
  return static_cast<int>(((it + 1) * G - 1) / T);
}

// One CTA's view of a 128/256-row x BT-token tile: the cp.async pipeline,
// per-row de-quantisation into the BF16 smem tile, the ldmatrix + mma.sync
// mainloop and the bias (+ SwiGLU) epilogue, used by the stream-K GEMM.
// `Hooks` lets the caller run work at fixed points
// of the pipeline (PDL wait before the first activation load, the side job
// once per K step).
template <class Cfg, bool SWIGLU>
struct TileEngine {
  static constexpr int BN = Cfg::kBN;
  static constexpr int BT = Cfg::kBT;
  static constexpr int kThreads = Cfg::kThreads;
  static constexpr int S = Cfg::kStages;
  static constexpr int MT = Cfg::kMT;
  static constexpr int NT = Cfg::kNT;
  static constexpr int kSplit = Cfg::kSplit;
  static constexpr int kWords = 4 / kSplit;

  uint8_t* smem;
  int tid, lane, wn, wt, g, t;

  __device__ __forceinline__ explicit TileEngine(uint8_t* smem_) : smem(smem_) {
    tid = threadIdx.x;
    lane = tid & 31;
    const int warp = tid >> 5;
    wn = warp % Cfg::kWarpsN;
    wt = warp / Cfg::kWarpsN;
    g = lane >> 2;
    t = lane & 3;
  }

  // Accumulate K iterations [k_lo, k_hi) of tile (n0, t0) of
  // y = x[M, K] . W[N, K]^T into acc.
  template <class Hooks>
  __device__ __forceinline__ void mainloop(const __nv_bfloat16* __restrict__ x,
                                           int M, int K,
                                           const uint8_t* __restrict__ wq,
                                           const uint8_t* __restrict__ ws,
                                           int N, int n0, int t0, int k_lo,
                                           int k_hi, float (&acc)[MT][NT][4],
                                           Hooks& hooks) {
    const int KB = K >> 5;
    const int KP = K >> 1;
    // This thread stages and de-quantises (part of) weight row `wrow` into
    // smem row `srow`: words [wq0, wq0 + 4 / kSplit) of its 16-B MX block.
    const int srow = tid / kSplit;
    const int wq0 = (tid % kSplit) * kWords;
    const int wrow = n0 + weight_row_of<SWIGLU>(srow);
    const bool wvalid = wrow < N;
    const uint8_t* wsrc = wq + static_cast<size_t>(wvalid ? wrow : 0) * KP;
    // The scale byte travels with the codes: cp.async of the 4-B aligned
    // word holding it (scales are 16-B aligned with N * K/32 % 4 == 0,
    // checked on the host, so the word never leaves the tensor).
    const uintptr_t sbase = reinterpret_cast<uintptr_t>(
        ws + static_cast<size_t>(wvalid ? wrow : 0) * KB);
    auto raw = [&](int stage) { return smem + stage * Cfg::kRawStage; };
    auto load_w = [&](int stage, int kb) {
      uint8_t* dst = raw(stage) + Cfg::kXStage + tid * 4 * kWords;
      if constexpr (kSplit == 1) {
        cp_async16(dst, wsrc + 16 * kb, wvalid);
      } else {
        cp_async8(dst, wsrc + 16 * kb + 4 * wq0, wvalid);
      }
      cp_async4(raw(stage) + Cfg::kXStage + Cfg::kWRaw + tid * 4,
                reinterpret_cast<const void*>((sbase + kb) & ~uintptr_t{3}),
                wvalid);
    };
    auto load_x = [&](int stage, int kb) {
#pragma unroll
      for (int i = tid; i < BT * 4; i += kThreads) {
        const int r = i >> 2;
        const int c = i & 3;
        const int tok = t0 + r;
        const bool v = tok < M;
        cp_async16(raw(stage) + swz(r, c),
                   x + static_cast<size_t>(v ? tok : 0) * K + 32 * kb + 8 * c,
                   v);
      }
    };
    auto convert = [&](int stage, int buf, int kb) {
      const uint8_t* src = raw(stage) + Cfg::kXStage + tid * 4 * kWords;
      const uint32_t sc =
          raw(stage)[Cfg::kXStage + Cfg::kWRaw + tid * 4 + ((sbase + kb) & 3)];
      uint32_t q[kWords];
      if constexpr (kSplit == 1) {
        const uint4 v = *reinterpret_cast<const uint4*>(src);
        q[0] = v.x;
        q[1] = v.y;
        q[2] = v.z;
        q[3] = v.w;
      } else {
        const uint2 v = *reinterpret_cast<const uint2*>(src);
        q[0] = v.x;
        q[1] = v.y;
      }
      const uint32_t s2 = e8m0_bf16x2(sc);
      uint8_t* dst = smem + Cfg::kWbfOffset + buf * Cfg::kWbf;
#pragma unroll
      for (int w = 0; w < kWords; ++w)
        *reinterpret_cast<uint4*>(dst + swz(srow, wq0 + w)) =
            e2m1x8_bf16(q[w], s2);
    };

#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int k = 0; k < 4; ++k) acc[i][j][k] = 0.0f;

    __syncthreads();  // previous segment / epilogue is done with smem
    // Weights and scales do not depend on the previous kernel.
#pragma unroll
    for (int s = 0; s < S - 1; ++s)
      if (k_lo + s < k_hi) load_w(s, k_lo + s);
    hooks.before_activations();
#pragma unroll
    for (int s = 0; s < S - 1; ++s) {
      if (k_lo + s < k_hi) load_x(s, k_lo + s);
      cp_async_commit();
    }
    // De-quantise the first iteration's weights (own row only).
    cp_async_wait<S - 2>();
    convert(0, 0, k_lo);

    for (int kb = k_lo; kb < k_hi; ++kb) {
      const int i = kb - k_lo;
      __syncthreads();
      hooks.per_iteration();
      const int nxt = kb + S - 1;
      if (nxt < k_hi) {
        const int ns = (i + S - 1) % S;
        load_w(ns, nxt);
        load_x(ns, nxt);
      }
      cp_async_commit();

      const uint8_t* xs_s = raw(i % S);
      const uint8_t* wb_s = smem + Cfg::kWbfOffset + (i & 1) * Cfg::kWbf;
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        uint32_t a[MT][4];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          const int r = wn * Cfg::kWN + mt * 16 + (lane & 15);
          ldmatrix_x4(a[mt], wb_s + swz(r, 2 * kk + (lane >> 4)));
        }
#pragma unroll
        for (int np = 0; np < NT / 2; ++np) {
          const int r =
              wt * Cfg::kWT + np * 16 + (lane & 7) + ((lane >> 4) << 3);
          uint32_t b[4];
          ldmatrix_x4(b, xs_s + swz(r, 2 * kk + ((lane >> 3) & 1)));
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            mma_bf16_2(acc[mt][2 * np], a[mt], b[0], b[1]);
            mma_bf16_2(acc[mt][2 * np + 1], a[mt], b[2], b[3]);
          }
        }
      }
      // With this iteration's MMAs issued, wait for our own weight codes of
      // kb + 1 and de-quantise them for the next iteration.
      cp_async_wait<S - 2>();
      if (kb + 1 < k_hi) convert((i + 1) % S, (i + 1) & 1, kb + 1);
    }
    cp_async_wait<0>();
    __syncthreads();
  }

  // Bias (+ SwiGLU) on FP32, staged through smem for 16-B stores.
  __device__ __forceinline__ void epilogue(
      const __nv_bfloat16* __restrict__ bias, int N,
      __nv_bfloat16* __restrict__ y, int M, int n0, int t0,
      const float (&acc)[MT][NT][4]) {
    constexpr int kOutCols = SWIGLU ? BN / 2 : BN;
    constexpr int kOutStride = kOutCols + 8;
    __nv_bfloat16* out_s = reinterpret_cast<__nv_bfloat16*>(smem);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      const int rl = wn * Cfg::kWN + mt * 16;
      int ra, rb;  // weight rows of MMA rows g / g+8
      if constexpr (SWIGLU) {
        ra = n0 + rl + 2 * g;
        rb = ra + 1;
      } else {
        ra = n0 + rl + g;
        rb = ra + 8;
      }
      const float ba = ra < N ? __bfloat162float(bias[ra]) : 0.0f;
      const float bb = rb < N ? __bfloat162float(bias[rb]) : 0.0f;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int tl = wt * Cfg::kWT + nt * 8 + 2 * t + h;
          if constexpr (SWIGLU) {
            out_s[tl * kOutStride + rl / 2 + g] = __float2bfloat16(
                swiglu_oai(acc[mt][nt][h] + ba, acc[mt][nt][2 + h] + bb));
          } else {
            out_s[tl * kOutStride + rl + g] =
                __float2bfloat16(acc[mt][nt][h] + ba);
            out_s[tl * kOutStride + rl + g + 8] =
                __float2bfloat16(acc[mt][nt][2 + h] + bb);
          }
        }
      }
    }
    __syncthreads();
    const int n_out = SWIGLU ? N >> 1 : N;
    const int c0 = SWIGLU ? n0 >> 1 : n0;
    constexpr int kChunks = kOutCols / 8;
    for (int i = tid; i < BT * kChunks; i += kThreads) {
      const int r = i / kChunks;
      const int c = i - r * kChunks;
      const int tok = t0 + r;
      const int col = c0 + c * 8;
      if (tok < M && col < n_out) {
        *reinterpret_cast<uint4*>(y + static_cast<size_t>(tok) * n_out + col) =
            *reinterpret_cast<const uint4*>(out_s + r * kOutStride + c * 8);
      }
    }
  }
};

template <int BN_, int BT, int WARPS_N, int WARPS_T, int STAGES, bool SWIGLU,
          bool PDL_WAIT>
__global__ void __launch_bounds__(32 * WARPS_N * WARPS_T)
    mxfp4_tile_pro6000_kernel(const __nv_bfloat16* __restrict__ x, int M, int K,
                              const uint8_t* __restrict__ wq,
                              const uint8_t* __restrict__ ws,
                              const __nv_bfloat16* __restrict__ bias, int N,
                              __nv_bfloat16* __restrict__ y, StreamK sk) {
  using Cfg = TileCfg<BN_, BT, WARPS_N, WARPS_T, STAGES>;
  using Engine = TileEngine<Cfg, SWIGLU>;
  constexpr int BN = Cfg::kBN;
  constexpr int kThreads = Cfg::kThreads;
  constexpr int MT = Cfg::kMT;
  constexpr int NT = Cfg::kNT;
  extern __shared__ uint4 smem_u4[];
  uint8_t* smem = reinterpret_cast<uint8_t*>(smem_u4);
  __shared__ int s_last;
  Engine eng(smem);
  const int tid = threadIdx.x;
  const int KB = K >> 5;

  // Side job (see kSideLanes): this CTA owns blocks [side_lo, side_hi).
  __shared__ alignas(16) uint8_t side_buf[kSideRing][kSideChunkBytes + 16];
  __shared__ alignas(8) uint64_t side_bar[kSideRing];
  const long long side_lo = static_cast<long long>(blockIdx.x) * sk.side_per;
  const int side_chunks =
      sk.side_blocks > 0
          ? static_cast<int>(
                (std::min<long long>(sk.side_blocks, side_lo + sk.side_per) -
                 std::min<long long>(sk.side_blocks, side_lo)) /
                kSideLanes)
          : 0;
  auto side_store = [&](long long b, const uint4& q, uint32_t sc) {
    const uint32_t s2 = e8m0_bf16x2(sc);
    uint4* dst = reinterpret_cast<uint4*>(sk.side_out + 32 * b);
    __stcg(dst + 0, e2m1x8_bf16(q.x, s2));
    __stcg(dst + 1, e2m1x8_bf16(q.y, s2));
    __stcg(dst + 2, e2m1x8_bf16(q.z, s2));
    __stcg(dst + 3, e2m1x8_bf16(q.w, s2));
  };
  // Every thread converts 256 / kThreads code bytes of a chunk, so the work
  // is divergence-free and the 1 KB of BF16 output is written coalesced.
  auto side_consume = [&](int j) {
    if (j < 0 || j >= side_chunks) return;
    mbar_wait(&side_bar[j % kSideRing], (j / kSideRing) & 1);
    // Volatile reads: they must stay after the (clobber-free) barrier wait.
    const volatile uint8_t* buf = side_buf[j % kSideRing];
    uint32_t* out = reinterpret_cast<uint32_t*>(
        sk.side_out + 32 * (side_lo + static_cast<long long>(j) * kSideLanes));
#pragma unroll
    for (int k = tid; k < kSideLanes * 16; k += kThreads) {
      const uint32_t byte = buf[k];
      const uint32_t w = (byte & 0xFu) | ((byte >> 4) << 16);
      __stcg(out + k,
             e2m1_pair<0>(w, e8m0_bf16x2(buf[kSideLanes * 16 + (k >> 4)])));
    }
  };
  struct Hooks {
    bool waited;
    int iter;  // K iterations run so far (uniform across the CTA)
    int tid;
    int chunks;
    long long lo;
    const StreamK* sk;
    uint8_t (*buf)[kSideChunkBytes + 16];
    uint64_t* bar;
    decltype(side_consume)* consume;
    __device__ __forceinline__ void before_activations() {
      if (!waited) {
        pdl_wait();
        waited = true;
      }
    }
    __device__ __forceinline__ void per_iteration() {
      if (tid == 0 && iter < chunks) {
        const long long b0 = lo + static_cast<long long>(iter) * kSideLanes;
        side_fetch(buf[iter % kSideRing], sk->side_q + 16 * b0, sk->side_s + b0,
                   &bar[iter % kSideRing]);
      }
      // Overlaps the MMAs that follow (independent instructions).
      (*consume)(iter - kSideLag);
      ++iter;
    }
  };
  Hooks hooks{!PDL_WAIT, 0,        tid,      side_chunks,  side_lo,
              &sk,       side_buf, side_bar, &side_consume};
  if (side_chunks > 0 && tid == 0) {
    for (int r = 0; r < kSideRing; ++r) mbar_init(&side_bar[r], 1);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
  }

  pdl_launch_dependents();

  float acc[MT][NT][4];
  auto tile_origin = [&](int tile, int& n0, int& t0) {
    n0 = (tile % sk.tiles_n) * BN;
    t0 = (tile / sk.tiles_n) * BT;
  };

  for (int tile = blockIdx.x; tile < sk.n_dp; tile += gridDim.x) {
    int n0, t0;
    tile_origin(tile, n0, t0);
    eng.mainloop(x, M, K, wq, ws, N, n0, t0, 0, KB, acc, hooks);
    eng.epilogue(bias, N, y, M, n0, t0, acc);
  }

  if (sk.sk_iters > 0) {
    const long long T = sk.sk_iters;
    const long long G = gridDim.x;
    const long long b = sk_begin(blockIdx.x, T, G);
    const long long e = sk_begin(blockIdx.x + 1, T, G);
    const size_t slot = Cfg::slot_floats();
    for (long long it = b; it < e;) {
      const int ts = static_cast<int>(it / KB);
      const int k_lo = static_cast<int>(it - static_cast<long long>(ts) * KB);
      const int k_hi =
          static_cast<int>(std::min<long long>(KB, k_lo + (e - it)));
      int n0, t0;
      tile_origin(sk.n_dp + ts, n0, t0);
      eng.mainloop(x, M, K, wq, ws, N, n0, t0, k_lo, k_hi, acc, hooks);
      if (k_lo == 0 && k_hi == KB) {
        eng.epilogue(bias, N, y, M, n0, t0, acc);
      } else {
        const int which = (it == b) ? 0 : 1;
        float* mine =
            sk.partials + (which * static_cast<size_t>(G) + blockIdx.x) * slot;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int k = 0; k < 4; ++k)
              __stcg(mine + ((mt * NT + nt) * 4 + k) * kThreads + tid,
                     acc[mt][nt][k]);
        fence_acq_rel_gpu();
        __syncthreads();
        if (tid == 0) {
          const int add = k_hi - k_lo;
          const int prev = atomicAdd(sk.counters + ts, add);
          s_last = (prev + add == KB);
          if (s_last) sk.counters[ts] = 0;
        }
        __syncthreads();
        if (s_last) {
          fence_acq_rel_gpu();
          const long long tb = static_cast<long long>(ts) * KB;
          const int c_first = sk_owner(tb, T, G);
          const int c_last = sk_owner(tb + KB - 1, T, G);
          for (int c = c_first; c <= c_last; ++c) {
            if (c == static_cast<int>(blockIdx.x)) continue;
            const int w = sk_begin(c, T, G) >= tb ? 0 : 1;
            const float* other =
                sk.partials + (w * static_cast<size_t>(G) + c) * slot;
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
              for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int k = 0; k < 4; ++k)
                  acc[mt][nt][k] +=
                      __ldcg(other + ((mt * NT + nt) * 4 + k) * kThreads + tid);
          }
          eng.epilogue(bias, N, y, M, n0, t0, acc);
        }
      }
      it += k_hi - k_lo;
    }
  }
  // Side-job chunks still in flight, then the ones this CTA had no
  // iteration for.
  if (side_chunks > 0) {
    for (int j = hooks.iter - kSideLag; j < hooks.iter; ++j) side_consume(j);
    for (long long k = static_cast<long long>(hooks.iter) * kSideLanes + tid;
         k < static_cast<long long>(side_chunks) * kSideLanes; k += kThreads) {
      const long long b = side_lo + k;
      side_store(b, __ldg(reinterpret_cast<const uint4*>(sk.side_q) + b),
                 __ldg(sk.side_s + b));
    }
  }
  if (!hooks.waited) pdl_wait();
}

__global__ void l2_discard_kernel(void* p, size_t bytes) {
  discard_l2(p, bytes, blockIdx.x * blockDim.x + threadIdx.x,
             gridDim.x * blockDim.x);
}

// --------------------------------------------------------------------------
// Host side
// --------------------------------------------------------------------------
struct Device {
  int sms = 0;
  int major = 0;
  size_t smem_max = 0;
};

Device query_device(int dev) {
  Device d;
  cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, dev);
  cudaDeviceGetAttribute(&d.major, cudaDevAttrComputeCapabilityMajor, dev);
  int smem_max = 0;
  cudaDeviceGetAttribute(&smem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                         dev);
  d.smem_max = static_cast<size_t>(smem_max);
  return d;
}

int current_device() {
  int dev = 0;
  cudaGetDevice(&dev);
  return dev;
}

const Device& device_info(int dev) {
  static thread_local int cached_dev = -1;
  static thread_local Device cached;
  if (dev != cached_dev) {
    cached = query_device(dev);
    cached_dev = dev;
  }
  return cached;
}

template <typename Kernel, typename... Args>
cudaError_t launch(Kernel kernel, dim3 grid, dim3 block, size_t smem,
                   cudaStream_t stream, bool pdl, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  return cudaLaunchKernelEx(&cfg, kernel, args...);
}

template <int WK, int U, int NT, int MT, bool SWIGLU, bool PDL>
cudaError_t run_gemv(const __nv_bfloat16* x, int M, int K, const uint8_t* wq,
                     const uint8_t* ws, const __nv_bfloat16* bias, int N,
                     __nv_bfloat16* y, cudaStream_t stream) {
  const int aligned =
      (reinterpret_cast<uintptr_t>(wq) % 16 == 0) && ((K / 2) % 16 == 0);
  dim3 grid((N + 16 * MT - 1) / (16 * MT), (M + 8 * NT - 1) / (8 * NT));
  return launch(mxfp4_gemv_pro6000_kernel<WK, U, NT, MT, SWIGLU, PDL>, grid,
                dim3(WK * 32), 0, stream, PDL, x, M, K, wq, ws, bias, N, y,
                aligned);
}

// Per-(instantiation, device) launch facts: smem attribute raised, CTAs/SM.
struct KernelInfo {
  std::atomic<int> blocks_per_sm[64];
  std::atomic<int> smem_set[64];
};

template <int BN_, int BT, int WARPS_N, int WARPS_T, int STAGES, bool SWIGLU,
          bool PDL>
struct TileLauncher {
  using Cfg = TileCfg<BN_, BT, WARPS_N, WARPS_T, STAGES>;
  static constexpr auto kernel =
      mxfp4_tile_pro6000_kernel<BN_, BT, WARPS_N, WARPS_T, STAGES, SWIGLU, PDL>;

  // Resident CTAs per SM (raises the dynamic smem limit on first use).
  static cudaError_t occupancy(int dev, int* occ) {
    static KernelInfo info{};
    const size_t smem = Cfg::smem_bytes(SWIGLU);
    if (smem > device_info(dev).smem_max) return cudaErrorInvalidConfiguration;
    if (dev < 64 && info.smem_set[dev].load() >= static_cast<int>(smem)) {
      *occ = info.blocks_per_sm[dev].load();
      return cudaSuccess;
    }
    cudaError_t e = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem));
    if (e != cudaSuccess) return e;
    e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(occ, kernel,
                                                      Cfg::kThreads, smem);
    if (e != cudaSuccess) return e;
    *occ = std::max(*occ, 1);
    if (dev < 64) {
      info.blocks_per_sm[dev].store(*occ);
      info.smem_set[dev].store(static_cast<int>(smem));
    }
    return cudaSuccess;
  }

  static size_t workspace_bytes(int dev) {
    int occ = 0;
    if (occupancy(dev, &occ) != cudaSuccess) return 0;
    return static_cast<size_t>(device_info(dev).sms) * occ * 2 *
           Cfg::slot_floats() * sizeof(float);
  }

  static cudaError_t run(const __nv_bfloat16* x, int M, int K,
                         const uint8_t* wq, const uint8_t* ws,
                         const __nv_bfloat16* bias, int N, __nv_bfloat16* y,
                         const Mxfp4Workspace* work, const Mxfp4Side* side,
                         cudaStream_t stream) {
    constexpr int BN = Cfg::kBN;
    const int dev = current_device();
    const Device& d = device_info(dev);
    const int KB = K / 32;
    int occ = 0;
    cudaError_t e = occupancy(dev, &occ);
    if (e != cudaSuccess) return e;

    StreamK sk = {};
    sk.tiles_n = (N + BN - 1) / BN;
    const int tiles_t = (M + BT - 1) / BT;
    sk.n_tiles = sk.tiles_n * tiles_t;
    const int slots = d.sms * occ;
    const size_t slot_bytes = 2 * Cfg::slot_floats() * sizeof(float);
    // Persistent grid; a short workspace may cap it below full occupancy.
    int grid = slots;
    const bool have_ws = work != nullptr && work->partials != nullptr &&
                         work->counters != nullptr;
    if (have_ws)
      grid = static_cast<int>(std::min<size_t>(grid, work->bytes / slot_bytes));
    // Stream-K only pays off for a few waves; with many, the tail wave is
    // cheap and the FP32 partials would only add L2 traffic.
    const bool can_sk = have_ws && grid >= slots / 2 &&
                        sk.n_tiles % slots != 0 && sk.n_tiles < 16 * slots;
    if (!can_sk) {
      sk.n_dp = sk.n_tiles;
      grid = std::min(slots * 8, sk.n_tiles);
    } else {
      // Whole waves data-parallel, the remainder stream-K.
      sk.n_dp = (sk.n_tiles / grid) * grid;
      sk.sk_iters = static_cast<long long>(sk.n_tiles - sk.n_dp) * KB;
      if (sk.n_dp == 0) {
        // Keep >= 8 K iterations per CTA so pipeline fill stays amortised.
        grid = static_cast<int>(
            std::max<long long>(1, std::min<long long>(grid, sk.sk_iters / 8)));
      }
      if (sk.n_tiles - sk.n_dp > work->n_counters) {
        sk.n_dp = sk.n_tiles;
        sk.sk_iters = 0;
        grid = std::min(slots * 8, sk.n_tiles);
      }
      sk.partials = static_cast<float*>(work->partials);
      sk.counters = work->counters;
    }
    if (side != nullptr && side->n_blocks > 0) {
      sk.side_q = static_cast<const uint8_t*>(side->blocks);
      sk.side_s = static_cast<const uint8_t*>(side->scales);
      sk.side_out = static_cast<__nv_bfloat16*>(side->out);
      sk.side_blocks = side->n_blocks;
      const int per = (side->n_blocks + grid - 1) / grid;
      sk.side_per = (per + kSideLanes - 1) / kSideLanes * kSideLanes;
    }
    return launch(kernel, dim3(grid), dim3(Cfg::kThreads),
                  Cfg::smem_bytes(SWIGLU), stream, PDL, x, M, K, wq, ws, bias,
                  N, y, sk);
  }
};

// id, weight rows, tokens, warps along rows, warps along tokens, stages
#define MXFP4_PRO6000_TILE_CONFIGS(X) \
  X(4, 128, 32, 4, 1, 4)              \
  X(5, 128, 64, 2, 2, 4)              \
  X(6, 128, 64, 2, 2, 6)              \
  X(7, 256, 128, 4, 2, 4)

template <bool SWIGLU, bool PDL>
cudaError_t dispatch(const __nv_bfloat16* x, int M, int K, const uint8_t* wq,
                     const uint8_t* ws, const __nv_bfloat16* bias, int N,
                     __nv_bfloat16* y, const Mxfp4Workspace* work,
                     const Mxfp4Side* side, cudaStream_t stream, int cfg) {
  switch (cfg) {
    case 0:
      return run_gemv<8, 3, 1, 1, SWIGLU, PDL>(x, M, K, wq, ws, bias, N, y,
                                               stream);
    case 1:
      return run_gemv<8, 3, 2, 1, SWIGLU, PDL>(x, M, K, wq, ws, bias, N, y,
                                               stream);
    case 2:
      return run_gemv<8, 2, 4, 1, SWIGLU, PDL>(x, M, K, wq, ws, bias, N, y,
                                               stream);
    case 3:
      return run_gemv<8, 1, 4, 1, SWIGLU, PDL>(x, M, K, wq, ws, bias, N, y,
                                               stream);
#define MXFP4_CASE(ID, BN, BT, WN, WT, ST)                     \
  case ID:                                                     \
    return TileLauncher<BN, BT, WN, WT, ST, SWIGLU, PDL>::run( \
        x, M, K, wq, ws, bias, N, y, work, side, stream);
      MXFP4_PRO6000_TILE_CONFIGS(MXFP4_CASE)
#undef MXFP4_CASE
    default:
      return cudaErrorInvalidValue;
  }
}

size_t workspace_bytes_max(int dev) {
  size_t m = 0;
#define MXFP4_WS(ID, BN, BT, WN, WT, ST)                                       \
  m = std::max(                                                                \
      m, TileLauncher<BN, BT, WN, WT, ST, true, false>::workspace_bytes(dev)); \
  m = std::max(                                                                \
      m, TileLauncher<BN, BT, WN, WT, ST, false, true>::workspace_bytes(dev));
  MXFP4_PRO6000_TILE_CONFIGS(MXFP4_WS)
#undef MXFP4_WS
  return m;
}

// Measured on RTX PRO 6000 Max-Q (sm_120) at H = I = 2880.
int pick_pro6000_config(int M, bool swiglu) {
  static_assert(kMxfp4GemvMaxTokens == 32, "keep in sync with configs 0-3");
  // Up to 32 tokens: the NT = 1 kernel over ceil(M / 8) token slices
  // (blockIdx.y) beats the wider NT = 2 / 4 variants on this GPU.
  if (M <= kMxfp4GemvMaxTokens) return 0;
  if (M <= 128) return swiglu ? 6 : 4;
  if (M <= 256) return 5;
  return 7;
}

}  // namespace
}  // namespace gptoss_mxfp4

size_t mxfp4_linear_workspace_bytes() {
  return gptoss_mxfp4::workspace_bytes_max(gptoss_mxfp4::current_device());
}

cudaError_t mxfp4_linear_pro6000_cuda(const void* x, int M, int K,
                                      const void* wq, const void* ws,
                                      const void* bias, int N, void* y,
                                      bool swiglu, bool pdl_wait,
                                      const Mxfp4Workspace* work,
                                      const Mxfp4Side* side, int config,
                                      cudaStream_t stream) {
  using namespace gptoss_mxfp4;
  if (M <= 0) return cudaSuccess;
  auto misaligned = [](const void* p) {
    return reinterpret_cast<uintptr_t>(p) % 16 != 0;
  };
  if (K % 32 != 0 || N % 16 != 0 || misaligned(x) || misaligned(wq) ||
      misaligned(ws) || misaligned(y) ||
      (static_cast<long long>(N) * (K / 32)) % 4 != 0)
    return cudaErrorNotSupported;
  if (side != nullptr && side->n_blocks > 0 &&
      (misaligned(side->blocks) || misaligned(side->scales) ||
       misaligned(side->out)))
    return cudaErrorNotSupported;
  const int major = device_info(current_device()).major;
  const bool pdl = pdl_wait && major >= 9;
  if (config < 0) config = pick_pro6000_config(M, swiglu);
  const bool gemv = config < 4;
  if (side != nullptr && side->n_blocks > 0 &&
      (gemv || major < 9 || side->n_blocks % kSideLanes != 0)) {
    // No in-kernel side job (small-M kernels, pre-sm_90 bulk copies, ragged
    // block count): de-quantise up front.
    mxfp4_dequant_cuda(side->blocks, side->scales, side->out, side->n_blocks,
                       16, 1, 32, stream);
    side = nullptr;
  }
  auto xb = static_cast<const __nv_bfloat16*>(x);
  auto wqb = static_cast<const uint8_t*>(wq);
  auto wsb = static_cast<const uint8_t*>(ws);
  auto bb = static_cast<const __nv_bfloat16*>(bias);
  auto yb = static_cast<__nv_bfloat16*>(y);
  if (swiglu) {
    return pdl ? dispatch<true, true>(xb, M, K, wqb, wsb, bb, N, yb, work, side,
                                      stream, config)
               : dispatch<true, false>(xb, M, K, wqb, wsb, bb, N, yb, work,
                                       side, stream, config);
  }
  return pdl ? dispatch<false, true>(xb, M, K, wqb, wsb, bb, N, yb, work, side,
                                     stream, config)
             : dispatch<false, false>(xb, M, K, wqb, wsb, bb, N, yb, work, side,
                                      stream, config);
}

cudaError_t mxfp4_l2_discard_cuda(void* p, size_t bytes, cudaStream_t stream) {
  using namespace gptoss_mxfp4;
  bytes &= ~size_t{127};
  if (bytes == 0) return cudaSuccess;
  if (reinterpret_cast<uintptr_t>(p) % 128 != 0) return cudaErrorInvalidValue;
  const int blocks = static_cast<int>(std::min<size_t>(
      device_info(current_device()).sms * 2, (bytes / 128 + 255) / 256));
  l2_discard_kernel<<<blocks, 256, 0, stream>>>(p, bytes);
  return cudaGetLastError();
}

cudaError_t gptoss_mxfp4_expert_pro6000_cuda(
    const void* x, int M, int H, int I, const void* gu_blocks,
    const void* gu_scales, const void* gu_bias, const void* d_blocks,
    const void* d_scales, const void* d_bias, void* act, void* out,
    const Mxfp4Workspace* ws_gu, const Mxfp4Workspace* ws_d,
    cudaStream_t stream) {
  cudaError_t e =
      mxfp4_linear_pro6000_cuda(x, M, H, gu_blocks, gu_scales, gu_bias, 2 * I,
                                act, true, false, ws_gu, nullptr, -1, stream);
  if (e != cudaSuccess) return e;
  return mxfp4_linear_pro6000_cuda(act, M, I, d_blocks, d_scales, d_bias, H,
                                   out, false, true, ws_d, nullptr, -1, stream);
}
