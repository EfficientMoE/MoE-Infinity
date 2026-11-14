// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// Grouped block-128 FP8 SwiGLU for the RTX PRO 6000.
//
// Size split, same cut as the per-expert kernel (M <= 8 vs M > 8):
//   average routed rows per expert <= 8: one weight-only GEMV (bf16
//   mma.sync, exact fp32 block scale). This is the b1 / b64 path. Two
//   launches cover every expert (gate+up, then down).
//   average rows > 8: the same math, but each CTA stages its 16 weight
//   rows in shared memory and reuses them across token chunks, so the
//   expert weights move once. A third pass is the up projection; gate
//   is written to a temporary and consumed by the SiLU epilogue.
// Both paths write per-slot BF16 down outputs and sum them in FP32 in
// slot order, multiplied by the routing weight. No BF16 copy of W.

#include "fp8_moe_grouped_sm120_pro6000.h"

#include "kernel/hw_profile.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <vector>

void fp8_grouped_large_sm120(const __nv_bfloat16* x, int T, int H, int I,
                             int topk, int E, const int* offsets,
                             const int* slot, const float* topk_w,
                             const int64_t* table, float* out,
                             cudaStream_t stream);

namespace {

constexpr int kBlock = 128;
constexpr int kWarps = 8;
constexpr int kMaxSlots = 8;
constexpr int kSmemCap = 101376;

__device__ __forceinline__ uint4 ld_u4(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__device__ __forceinline__ uint32_t e4m3x2_to_bf16x2(uint32_t v16) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(
      static_cast<__nv_fp8x2_storage_t>(v16 & 0xFFFFu), __NV_E4M3);
  const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
  const __nv_bfloat162 b = __floats2bfloat162_rn(f.x, f.y);
  return *reinterpret_cast<const uint32_t*>(&b);
}

__device__ __forceinline__ void mma_bf16(float* d, const uint32_t* a,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float silu(float x) {
  return x / (1.f + __expf(-x));
}

struct Expert6 {
  const uint8_t* gate;
  const float* gate_s;
  const uint8_t* up;
  const float* up_s;
  const uint8_t* down;
  const float* down_s;
};
static_assert(sizeof(Expert6) == 48, "table row is int64[6]");

__global__ void count_ids_kernel(const int* __restrict__ ids, int nslots,
                                 int E, int* __restrict__ counts) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nslots;
       i += blockDim.x * gridDim.x) {
    const int e = ids[i];
    if (static_cast<unsigned>(e) < static_cast<unsigned>(E))
      atomicAdd(counts + e, 1);
  }
}

// E <= 1024. Exclusive prefix of counts, plus the total at offsets[E].
__global__ void exclusive_scan_kernel(const int* __restrict__ counts,
                                      int* __restrict__ offsets, int n) {
  __shared__ int buf[2][1024];
  const int tid = threadIdx.x;
  const int orig = tid < n ? counts[tid] : 0;
  buf[0][tid] = orig;
  __syncthreads();
  int src = 0;
  for (int offset = 1; offset < 1024; offset <<= 1) {
    int v = buf[src][tid];
    if (tid >= offset) v += buf[src][tid - offset];
    buf[src ^ 1][tid] = v;
    src ^= 1;
    __syncthreads();
  }
  if (tid < n) offsets[tid] = buf[src][tid] - orig;
  if (tid == 0) offsets[n] = n > 0 ? buf[src][n - 1] : 0;
}

__global__ void fill_slots_kernel(const int* __restrict__ ids, int nslots,
                                  int E, int* __restrict__ cursor,
                                  int* __restrict__ slot) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nslots;
       i += blockDim.x * gridDim.x) {
    const int e = ids[i];
    if (static_cast<unsigned>(e) < static_cast<unsigned>(E)) {
      const int p = atomicAdd(cursor + e, 1);
      slot[p] = i;
    }
  }
}

// One 64-wide K chunk: 4x m16n8k16, scale applied to the partial.
__device__ __forceinline__ void mma_chunk(float* partial, uint4 wlo, uint4 whi,
                                          uint4 xv0, uint4 xv1) {
  const uint32_t lo[4] = {wlo.x, wlo.y, wlo.z, wlo.w};
  const uint32_t hi[4] = {whi.x, whi.y, whi.z, whi.w};
  const uint32_t xb[8] = {xv0.x, xv0.y, xv0.z, xv0.w, xv1.x, xv1.y, xv1.z, xv1.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) partial[i] = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t afr[4] = {
        e4m3x2_to_bf16x2(lo[j]), e4m3x2_to_bf16x2(hi[j]),
        e4m3x2_to_bf16x2(lo[j] >> 16), e4m3x2_to_bf16x2(hi[j] >> 16)};
    mma_bf16(partial, afr, xb[2 * j], xb[2 * j + 1]);
  }
}

// NMAT=2 writes silu(gate)*up into act[slot, N]. NMAT=1 writes the down
// projection into ybuf[slot, N] (routing weight applied in the combine).
template <int NMAT>
__global__ void __launch_bounds__(kWarps* 32)
    fp8_grouped_gemv_kernel(const __nv_bfloat16* __restrict__ x, int ldx,
                            int Kdiv, const Expert6* __restrict__ experts,
                            const int* __restrict__ offsets,
                            const int* __restrict__ slot,
                            __nv_bfloat16* __restrict__ act, int act_ld,
                            __nv_bfloat16* __restrict__ ybuf, int N, int K) {
  const int e = blockIdx.y;
  const int begin = offsets[e];
  const int nslots = offsets[e + 1] - begin;
  if (nslots <= 0) return;
  const Expert6 expert = experts[e];
  const uint8_t* w0 = NMAT == 2 ? expert.gate : expert.down;
  const float* s0 = NMAT == 2 ? expert.gate_s : expert.down_s;
  const uint8_t* w1 = NMAT == 2 ? expert.up : nullptr;
  const float* s1 = NMAT == 2 ? expert.up_s : nullptr;

  __shared__ float red[kWarps][NMAT][32][4];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int g = lane >> 2;
  const int c = lane & 3;
  const int row0 = blockIdx.x * 16;
  if (row0 >= N) return;
  const int KB = K / kBlock;
  const int chunks = K / 64;
  const int sblk = (row0 / kBlock) * KB;
  const int64_t r_lo = min(row0 + g, N - 1);
  const int64_t r_hi = min(row0 + g + 8, N - 1);

  for (int m0 = 0; m0 < nslots; m0 += kMaxSlots) {
    const int M = min(kMaxSlots, nslots - m0);
    const bool live = g < M;
    const int id = live ? slot[begin + m0 + g] : 0;
    const int xrow = NMAT == 2 ? id / Kdiv : id;

    float acc[NMAT][4];
#pragma unroll
    for (int a = 0; a < NMAT; ++a)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[a][i] = 0.f;

    for (int base = warp; base < chunks; base += kWarps * 4) {
      uint4 wlo[4][NMAT], whi[4][NMAT], xv0[4], xv1[4];
      float sc[4][NMAT];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int ch = base + u * kWarps;
        const bool ok = ch < chunks;
        const int k = ch * 64 + c * 16;
#pragma unroll
        for (int a = 0; a < NMAT; ++a) {
          const uint8_t* wp = a == 0 ? w0 : w1;
          const float* sp = a == 0 ? s0 : s1;
          wlo[u][a] = ok ? ld_u4(wp + r_lo * K + k) : make_uint4(0, 0, 0, 0);
          whi[u][a] = ok ? ld_u4(wp + r_hi * K + k) : make_uint4(0, 0, 0, 0);
          sc[u][a] = ok ? __ldg(sp + sblk + ch / 2) : 0.f;
        }
        const __nv_bfloat16* xp = x + static_cast<int64_t>(xrow) * ldx + k;
        xv0[u] = ok && live ? __ldg(reinterpret_cast<const uint4*>(xp))
                            : make_uint4(0, 0, 0, 0);
        xv1[u] = ok && live ? __ldg(reinterpret_cast<const uint4*>(xp) + 1)
                            : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) {
#pragma unroll
        for (int a = 0; a < NMAT; ++a) {
          float partial[4];
          mma_chunk(partial, wlo[u][a], whi[u][a], xv0[u], xv1[u]);
#pragma unroll
          for (int i = 0; i < 4; ++i)
            acc[a][i] = fmaf(partial[i], sc[u][a], acc[a][i]);
        }
      }
    }

#pragma unroll
    for (int a = 0; a < NMAT; ++a)
#pragma unroll
      for (int i = 0; i < 4; ++i) red[warp][a][lane][i] = acc[a][i];
    __syncthreads();
    if (warp == 0) {
      float sum[NMAT][4];
#pragma unroll
      for (int a = 0; a < NMAT; ++a)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float v = 0.f;
#pragma unroll
          for (int w = 0; w < kWarps; ++w) v += red[w][a][lane][i];
          sum[a][i] = v;
        }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int row = row0 + g + (i >= 2 ? 8 : 0);
        const int t = 2 * c + (i & 1);
        if (t < M && row < N) {
          const int sid = slot[begin + m0 + t];
          if constexpr (NMAT == 2) {
            const float o = silu(sum[0][i]) * sum[1][i];
            act[static_cast<int64_t>(sid) * act_ld + row] = __float2bfloat16(o);
          } else {
            ybuf[static_cast<int64_t>(sid) * act_ld + row] =
                __float2bfloat16(sum[0][i]);
          }
        }
      }
    }
    __syncthreads();
  }
}

// MODE 0: gate -> tmp. MODE 1: up, silu(tmp)*up -> act. MODE 2: down -> ybuf.
// Weight tile (16 rows x K) lives in dynamic smem for the whole token loop.
template <int MODE>
__global__ void __launch_bounds__(kWarps* 32)
    fp8_grouped_cached_kernel(const __nv_bfloat16* __restrict__ x, int ldx,
                              int Kdiv, const Expert6* __restrict__ experts,
                              const int* __restrict__ offsets,
                              const int* __restrict__ slot,
                              __nv_bfloat16* __restrict__ tmp,
                              __nv_bfloat16* __restrict__ act, int act_ld,
                              __nv_bfloat16* __restrict__ ybuf, int N, int K) {
  const int e = blockIdx.y;
  const int begin = offsets[e];
  const int nslots = offsets[e + 1] - begin;
  const int row0 = blockIdx.x * 16;
  if (nslots <= 0 || row0 >= N) return;

  const Expert6 expert = experts[e];
  const uint8_t* w;
  const float* scs;
  if constexpr (MODE == 0) {
    w = expert.gate;
    scs = expert.gate_s;
  } else if constexpr (MODE == 1) {
    w = expert.up;
    scs = expert.up_s;
  } else {
    w = expert.down;
    scs = expert.down_s;
  }

  extern __shared__ __align__(16) uint8_t sW[];
  const int tid = threadIdx.x;
  const int vecs = (16 * K) >> 4;
  uint4* dst = reinterpret_cast<uint4*>(sW);
  for (int i = tid; i < vecs; i += blockDim.x) {
    const int byte = i << 4;
    const int r = byte / K;
    const int c = byte - r * K;
    const int gr = row0 + r;
    dst[i] = gr < N ? ld_u4(w + static_cast<int64_t>(gr) * K + c)
                    : make_uint4(0, 0, 0, 0);
  }
  __syncthreads();

  __shared__ float red[kWarps][32][4];
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int g = lane >> 2;
  const int c = lane & 3;
  const int KB = K / kBlock;
  const int chunks = K / 64;
  const int sblk = (row0 / kBlock) * KB;
  const int r_lo = g;
  const int r_hi = g + 8;

  for (int m0 = 0; m0 < nslots; m0 += kMaxSlots) {
    const int M = min(kMaxSlots, nslots - m0);
    const bool live = g < M;
    const int id = live ? slot[begin + m0 + g] : 0;
    const int xrow = MODE == 2 ? id : id / Kdiv;

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int base = warp; base < chunks; base += kWarps * 4) {
      uint4 wlo[4], whi[4], xv0[4], xv1[4];
      float sc[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int ch = base + u * kWarps;
        const bool ok = ch < chunks;
        const int k = ch * 64 + c * 16;
        wlo[u] = ok ? *reinterpret_cast<const uint4*>(sW + r_lo * K + k)
                    : make_uint4(0, 0, 0, 0);
        whi[u] = ok ? *reinterpret_cast<const uint4*>(sW + r_hi * K + k)
                    : make_uint4(0, 0, 0, 0);
        sc[u] = ok ? __ldg(scs + sblk + ch / 2) : 0.f;
        const __nv_bfloat16* xp = x + static_cast<int64_t>(xrow) * ldx + k;
        xv0[u] = ok && live ? __ldg(reinterpret_cast<const uint4*>(xp))
                            : make_uint4(0, 0, 0, 0);
        xv1[u] = ok && live ? __ldg(reinterpret_cast<const uint4*>(xp) + 1)
                            : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        float partial[4];
        mma_chunk(partial, wlo[u], whi[u], xv0[u], xv1[u]);
#pragma unroll
        for (int i = 0; i < 4; ++i)
          acc[i] = fmaf(partial[i], sc[u], acc[i]);
      }
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) red[warp][lane][i] = acc[i];
    __syncthreads();
    if (warp == 0) {
      float sum[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) v += red[w][lane][i];
        sum[i] = v;
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int row = row0 + g + (i >= 2 ? 8 : 0);
        const int t = 2 * c + (i & 1);
        if (t < M && row < N) {
          const int sid = slot[begin + m0 + t];
          const int64_t oi = static_cast<int64_t>(sid) * act_ld + row;
          if constexpr (MODE == 0) {
            tmp[oi] = __float2bfloat16(sum[i]);
          } else if constexpr (MODE == 1) {
            const float gate = __bfloat162float(tmp[oi]);
            act[oi] = __float2bfloat16(silu(gate) * sum[i]);
          } else {
            ybuf[oi] = __float2bfloat16(sum[i]);
          }
        }
      }
    }
    __syncthreads();
  }
}

__global__ void combine_slots_kernel(const __nv_bfloat16* __restrict__ ybuf,
                                     const float* __restrict__ topk_w, int K,
                                     int H, float* __restrict__ out) {
  const int t = blockIdx.x;
  const int col = blockIdx.y * blockDim.x + threadIdx.x;
  if (col >= H) return;
  float s = 0.f;
  for (int k = 0; k < K; ++k) {
    const int64_t slot = static_cast<int64_t>(t) * K + k;
    s += __bfloat162float(ybuf[slot * H + col]) * topk_w[slot];
  }
  out[static_cast<int64_t>(t) * H + col] = s;
}

// Static reduction smem sits beside the dynamic weight tile, so the opt-in
// cap cannot be requested in full. 64 KiB covers GLM-5.3-Flash (16*4096).
constexpr int kCachedSmem = 64 * 1024;

template <int MODE>
void launch_cached_mode(dim3 grid, int smem, cudaStream_t stream,
                        const __nv_bfloat16* x, int ldx, int Kdiv,
                        const Expert6* experts, const int* offsets,
                        const int* slot, __nv_bfloat16* tmp,
                        __nv_bfloat16* act, int act_ld, __nv_bfloat16* ybuf,
                        int N, int K) {
  auto* k = fp8_grouped_cached_kernel<MODE>;
  static bool once = false;
  if (!once) {
    TORCH_CHECK(cudaFuncSetAttribute(
                    k, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    kCachedSmem) == cudaSuccess,
                "fp8 grouped: shared-memory opt-in failed");
    once = true;
  }
  TORCH_CHECK(smem <= kCachedSmem, "fp8 grouped: weight tile exceeds smem");
  k<<<grid, kWarps * 32, smem, stream>>>(x, ldx, Kdiv, experts, offsets, slot,
                                         tmp, act, act_ld, ybuf, N, K);
}

void launch_cached(int mode, dim3 grid, int smem, cudaStream_t stream,
                   const __nv_bfloat16* x, int ldx, int Kdiv,
                   const Expert6* experts, const int* offsets, const int* slot,
                   __nv_bfloat16* tmp, __nv_bfloat16* act, int act_ld,
                   __nv_bfloat16* ybuf, int N, int K) {
  if (mode == 0)
    launch_cached_mode<0>(grid, smem, stream, x, ldx, Kdiv, experts, offsets,
                          slot, tmp, act, act_ld, ybuf, N, K);
  else if (mode == 1)
    launch_cached_mode<1>(grid, smem, stream, x, ldx, Kdiv, experts, offsets,
                          slot, tmp, act, act_ld, ybuf, N, K);
  else
    launch_cached_mode<2>(grid, smem, stream, x, ldx, Kdiv, experts, offsets,
                          slot, tmp, act, act_ld, ybuf, N, K);
}

}  // namespace

torch::Tensor fp8_moe_grouped_forward_sm120_pro6000(
    const torch::Tensor& x, const torch::Tensor& topk_ids,
    const torch::Tensor& topk_w, const torch::Tensor& table, int64_t inter) {
  using moe_infinity::hw::Gate;
  using moe_infinity::hw::Supports;
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 &&
                  x.is_contiguous(),
              "fp8 grouped: x must be contiguous CUDA bf16 [T, H]");
  TORCH_CHECK(Supports(Gate::kSm120, x.get_device()) &&
                  Supports(Gate::kRtxPro6000, x.get_device()),
              "fp8 grouped: RTX PRO 6000 only");
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.scalar_type() == at::kInt &&
                  topk_ids.dim() == 2 && topk_ids.is_contiguous(),
              "fp8 grouped: topk_ids must be contiguous int32 [T, K]");
  TORCH_CHECK(topk_w.is_cuda() && topk_w.scalar_type() == at::kFloat &&
                  topk_w.sizes() == topk_ids.sizes() && topk_w.is_contiguous(),
              "fp8 grouped: topk_w must be contiguous fp32 [T, K]");
  TORCH_CHECK(table.is_cuda() && table.scalar_type() == at::kLong &&
                  table.dim() == 2 && table.size(1) == 6 &&
                  table.is_contiguous(),
              "fp8 grouped: table must be int64 [E, 6]");
  const int T = static_cast<int>(x.size(0));
  const int H = static_cast<int>(x.size(1));
  const int K = static_cast<int>(topk_ids.size(1));
  const int E = static_cast<int>(table.size(0));
  const int I = static_cast<int>(inter);
  TORCH_CHECK(topk_ids.size(0) == T, "fp8 grouped: topk rows must match T");
  TORCH_CHECK(H % kBlock == 0 && I % kBlock == 0 && H <= 8192 && I <= 8192,
              "fp8 grouped: H and I must be multiples of 128 and <= 8192");
  TORCH_CHECK(E >= 1 && E <= 1024 && K >= 1 && K <= 16, "fp8 grouped: E/K");
  auto out = torch::empty({T, H}, x.options().dtype(at::kFloat));
  if (T == 0) {
    out.zero_();
    return out;
  }

  const int slots = T * K;
  auto counts = torch::zeros({E}, topk_ids.options());
  auto offsets = torch::empty({E + 1}, topk_ids.options());
  auto slot = torch::empty({slots}, topk_ids.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const int blocks = std::max(1, std::min(2048, (slots + 255) / 256));
  count_ids_kernel<<<blocks, 256, 0, stream>>>(topk_ids.data_ptr<int>(), slots,
                                               E, counts.data_ptr<int>());
  exclusive_scan_kernel<<<1, 1024, 0, stream>>>(counts.data_ptr<int>(),
                                                offsets.data_ptr<int>(), E);
  auto cursor = offsets.narrow(/*dim=*/0, /*start=*/0, E).clone();
  fill_slots_kernel<<<blocks, 256, 0, stream>>>(
      topk_ids.data_ptr<int>(), slots, E, cursor.data_ptr<int>(),
      slot.data_ptr<int>());

  // Small rows/expert: the grouped BF16 GEMV, which is at the DRAM roof at b64.
  // Large rows/expert: one CUTLASS sm_120 blockwise FP8 grouped GEMM per
  // matmul (gate and up together, then down).
  if (slots > 8 * E) {
    fp8_grouped_large_sm120(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), T, H, I, K, E,
        offsets.data_ptr<int>(), slot.data_ptr<int>(), topk_w.data_ptr<float>(),
        table.data_ptr<int64_t>(), out.data_ptr<float>(), stream);
    return out;
  }

  auto act = torch::empty({slots, I}, x.options());
  auto ybuf = torch::zeros({slots, H}, x.options());

  const auto* experts =
      reinterpret_cast<const Expert6*>(table.data_ptr<int64_t>());
  // 64 KiB dynamic tile: 16 weight rows of K, GLM H=4096 and I=2048.
  const bool cache = slots > 8 * E && 16 * H <= 65536 && 16 * I <= 65536;
  if (!cache) {
    fp8_grouped_gemv_kernel<2>
        <<<dim3((I + 15) / 16, E), kWarps * 32, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), H, K, experts,
            offsets.data_ptr<int>(), slot.data_ptr<int>(),
            reinterpret_cast<__nv_bfloat16*>(act.data_ptr()), I,
            reinterpret_cast<__nv_bfloat16*>(ybuf.data_ptr()), I, H);
    fp8_grouped_gemv_kernel<1>
        <<<dim3((H + 15) / 16, E), kWarps * 32, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(act.data_ptr()), I, K,
            experts, offsets.data_ptr<int>(), slot.data_ptr<int>(), nullptr, H,
            reinterpret_cast<__nv_bfloat16*>(ybuf.data_ptr()), H, I);
  } else {
    auto tmp = torch::empty({slots, I}, x.options());
    auto* xptr = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    auto* actp = reinterpret_cast<__nv_bfloat16*>(act.data_ptr());
    auto* tmpp = reinterpret_cast<__nv_bfloat16*>(tmp.data_ptr());
    auto* yp = reinterpret_cast<__nv_bfloat16*>(ybuf.data_ptr());
    launch_cached(0, dim3((I + 15) / 16, E), 16 * H, stream, xptr, H, K,
                  experts, offsets.data_ptr<int>(), slot.data_ptr<int>(), tmpp,
                  actp, I, yp, I, H);
    launch_cached(1, dim3((I + 15) / 16, E), 16 * H, stream, xptr, H, K,
                  experts, offsets.data_ptr<int>(), slot.data_ptr<int>(), tmpp,
                  actp, I, yp, I, H);
    launch_cached(2, dim3((H + 15) / 16, E), 16 * I, stream, actp, I, K,
                  experts, offsets.data_ptr<int>(), slot.data_ptr<int>(), tmpp,
                  actp, H, yp, H, I);
  }
  combine_slots_kernel<<<dim3(T, (H + 127) / 128), 128, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(ybuf.data_ptr()),
      topk_w.data_ptr<float>(), K, H, out.data_ptr<float>());
  C10_CUDA_CHECK(cudaGetLastError());
  return out;
}
