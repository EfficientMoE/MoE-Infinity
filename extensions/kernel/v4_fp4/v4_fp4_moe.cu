// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team
//
// Native block-scaled FP4 routed-expert MoE for DeepSeek-V4 / V4.1 on sm_120a.
//
// Numerics follow the official DeepSeek-V4 inference (W4A8):
//   x  -> act_quant (FP8 e4m3, one ue8m0 power-of-two scale per 128 along K)
//   gate/up = fp4_gemm(x, w1/w3)  (FP32 accumulate, rounded to BF16)
//   h  = bf16( silu(clamp(gate)) * clamp(up) * routing_weight )
//   h  -> act_quant -> fp4_gemm(h, w2) -> BF16, summed in FP32 per token in
//   ascending expert order.
// Weights stay in their checkpoint layout: [N, K/2] packed e2m1 (low nibble =
// even k) with [N, K/32] ue8m0 scales. They are consumed by
//   mma.sync.m16n8k32.kind::mxf8f6f4.block_scale.scale_vec::1X (e2m1 x e4m3)
// with the weight rows as the 16-row operand and tokens as the 8-column
// operand; both per-32 weight scales and per-128 activation scales are fed to
// the MMA, so FP4 weights are read exactly once and never materialised.
//
// K permutation: ldmatrix gives lane (g, t) bytes 4t..4t+3 of a 16-byte row
// chunk, i.e. packed FP4 elements 8t..8t+7 of a 32-element block. These are
// used as MMA k-slots {4t..4t+3} and {16+4t..16+4t+3}. Activations are stored
// with the matching permutation inside each 32-element block (stored byte
// 4t+i = element 8t+i, stored byte 16+4t+i = element 8t+4+i), so a plain
// ldmatrix of the activation tile yields the matching B fragment.
//
// Per layer, no host synchronisation:
//   prep     : routing histogram/scan/scatter + tile list (block 0),
//              FP8 quantisation of x (other blocks)
// then, for <= 8 routed rows per expert on average (decode), one persistent
// kernel (moe_decode_kernel, programmatic dependent launch) that streams
// gate|up and down weights of all active experts; otherwise (prefill):
//   gemm<1>  : gate|up for 64/128 features x BT routed rows per CTA, fused
//              clamp/SiLU*up/routing weight and FP8 re-quantisation of h
//   gemm<2>  : down projection for 128/256 outputs x BT rows -> BF16
//   combine  : per-token FP32 sum over its top-k slots

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cooperative_groups.h>
#include <cstdint>

namespace cg = cooperative_groups;

namespace v4moe {

constexpr int kBK = 128;  // K per pipeline stage (one activation scale)

struct Tile {
  int expert;
  int row_start;  // offset into the expert-sorted slot list
  int nrows;
};

// Self-contained tile record for the decode kernel (BT = 8): one L2 round
// trip gives a work unit everything it needs.
struct TileExt {
  int64_t ptr[6];  // w1, s1, w3, s3, w2, s2 of the tile's expert
  int slot[8];     // token slots (t * topk + k); padded rows repeat slot 0
  int row_start, nrows, pad[2];
};
static_assert(sizeof(TileExt) == 96, "TileExt layout");

// ---------------------------------------------------------------------------
// small device helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src,
                                           int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst),
               "l"(src), "r"(src_bytes));
}

// Every thread of the CTA prefetches a share of [src, src + bytes) into L2.
__device__ __forceinline__ void prefetch_l2_range(const uint8_t* src,
                                                  long long bytes, int tid,
                                                  int nthreads) {
  for (long long o = (long long)tid * 128; o < bytes; o += nthreads * 128ll)
    asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(src + o));
}

__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dst),
               "l"(src));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0,
                                        uint32_t& r1, uint32_t& r2,
                                        uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

// 8 packed e2m1 codes (elements 8t..8t+7) -> two 4x8-bit mxf8f6f4 containers
// (code in bits [5:2]): lo = elements 0..3, hi = elements 4..7.
__device__ __forceinline__ void expand_fp4(uint32_t w, uint32_t& lo,
                                           uint32_t& hi) {
  uint32_t even = (w << 2) & 0x3C3C3C3Cu;  // elements 0,2,4,6
  uint32_t odd = (w >> 2) & 0x3C3C3C3Cu;   // elements 1,3,5,7
  lo = __byte_perm(even, odd, 0x5140);
  hi = __byte_perm(even, odd, 0x7362);
}

template <int BID>
__device__ __forceinline__ void mma_fp4_fp8(float* d, const uint32_t* a,
                                            uint32_t b0, uint32_t b1,
                                            uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32."
      "row.col.f32.e2m1.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
      "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa),
        "h"((uint16_t)BID), "h"((uint16_t)0), "r"(sfb), "h"((uint16_t)0),
        "h"((uint16_t)0));
}

// Official act_quant scale: 2^ceil(log2(max(amax, 1e-4) / 448)) computed on
// the IEEE bits of the FP32 product. Returns the unbiased exponent.
__device__ __forceinline__ int ue8m0_exponent(float amax) {
  amax = fmaxf(amax, 1e-4f);
  uint32_t bits = __float_as_uint(amax * (1.0f / 448.0f));
  int e = (int)((bits >> 23) & 0xFF) - 127;
  return e + ((bits & 0x7FFFFF) != 0 ? 1 : 0);
}

__device__ __forceinline__ float pow2(int e) {
  return __uint_as_float((uint32_t)(e + 127) << 23);
}

__device__ __forceinline__ uint32_t pack_fp8x4(float a, float b, float c,
                                               float d) {
  uint32_t lo =
      __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
  uint32_t hi =
      __nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3);
  return lo | (hi << 16);
}

__device__ __forceinline__ float bf16_round(float v) {
  return __bfloat162float(__float2bfloat16(v));
}

// Position of logical element e (0..31) of a 32-block in the permuted layout.
__device__ __forceinline__ int perm32(int e) {
  int t = e >> 3, i = e & 7;
  return i < 4 ? 4 * t + i : 16 + 4 * t + (i - 4);
}

// One 128-element activation group is quantised by 16 consecutive lanes, 8
// BF16 values each (elements 8t'..8t'+7 of a 32-block, t' = lane & 3).
// Writes the permuted FP8 bytes into the 32-byte block at `dst` and returns
// the unbiased ue8m0 exponent (identical in all 16 lanes).
__device__ __forceinline__ int quant8_group16(uint4 raw, uint8_t* dst, int t) {
  const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&raw);
  float f[8];
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    f[i] = __bfloat162float(v[i]);
    amax = fmaxf(amax, fabsf(f[i]));
  }
#pragma unroll
  for (int o = 1; o < 16; o <<= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  const int e = ue8m0_exponent(amax);
  const float inv = pow2(-e);
  *reinterpret_cast<uint32_t*>(dst + 4 * t) =
      pack_fp8x4(f[0] * inv, f[1] * inv, f[2] * inv, f[3] * inv);
  *reinterpret_cast<uint32_t*>(dst + 16 + 4 * t) =
      pack_fp8x4(f[4] * inv, f[5] * inv, f[6] * inv, f[7] * inv);
  return e;
}

// ---------------------------------------------------------------------------
// prep: routing (block 0) + x FP8 quantisation (blocks 1..)
// ---------------------------------------------------------------------------

constexpr int kPrepThreads = 1024;
constexpr int kMaxExperts = 1024;

template <typename IdT>
__global__ void __launch_bounds__(kPrepThreads)
    prep_kernel(const __nv_bfloat16* __restrict__ x,
                const IdT* __restrict__ topk_ids, int T, int H, int topk, int E,
                int BT, int* __restrict__ sorted, Tile* __restrict__ tiles,
                int* __restrict__ num_tiles, uint8_t* __restrict__ xq,
                uint8_t* __restrict__ xs, int* __restrict__ zero_buf,
                int zero_n, const int64_t* __restrict__ table,
                TileExt* __restrict__ tile_ext) {
  const int tid = threadIdx.x;
  // Let a programmatic-dependent successor launch now; it waits for this
  // grid's completion (griddepcontrol.wait) before reading our outputs.
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (blockIdx.x == 0) {
    for (int i = tid; i < zero_n; i += kPrepThreads) zero_buf[i] = 0;
    __shared__ int cnt[kMaxExperts];
    __shared__ int cur[kMaxExperts];
    __shared__ int tile_off[kMaxExperts];
    __shared__ int warp_sum[2][32];
    for (int e = tid; e < E; e += kPrepThreads) cnt[e] = 0;
    __syncthreads();
    const int n = T * topk;
    for (int i = tid; i < n; i += kPrepThreads) {
      long long e = (long long)topk_ids[i];
      if (e >= 0 && e < E) atomicAdd(&cnt[e], 1);
    }
    __syncthreads();
    // exclusive scans of counts and tile counts (E <= 1024: one per thread)
    int c = tid < E ? cnt[tid] : 0;
    int tc = (c + BT - 1) / BT;
    int lane = tid & 31, wid = tid >> 5;
    int sc = c, st = tc;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      int y0 = __shfl_up_sync(0xffffffffu, sc, o);
      int y1 = __shfl_up_sync(0xffffffffu, st, o);
      if (lane >= o) {
        sc += y0;
        st += y1;
      }
    }
    if (lane == 31) {
      warp_sum[0][wid] = sc;
      warp_sum[1][wid] = st;
    }
    __syncthreads();
    if (wid == 0) {
      int a = warp_sum[0][lane], b = warp_sum[1][lane];
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        int y0 = __shfl_up_sync(0xffffffffu, a, o);
        int y1 = __shfl_up_sync(0xffffffffu, b, o);
        if (lane >= o) {
          a += y0;
          b += y1;
        }
      }
      warp_sum[0][lane] = a;
      warp_sum[1][lane] = b;
    }
    __syncthreads();
    int base_c = (wid > 0 ? warp_sum[0][wid - 1] : 0) + sc - c;
    int base_t = (wid > 0 ? warp_sum[1][wid - 1] : 0) + st - tc;
    if (tid < E) {
      cur[tid] = base_c;
      tile_off[tid] = base_t;
    }
    if (tid == kPrepThreads - 1) *num_tiles = warp_sum[1][31];
    __syncthreads();
    if (tid < E) {
      for (int j = 0; j < tc; ++j) {
        Tile t;
        t.expert = tid;
        t.row_start = base_c + j * BT;
        t.nrows = min(BT, c - j * BT);
        tiles[base_t + j] = t;
      }
    }
    for (int i = tid; i < n; i += kPrepThreads) {
      long long e = (long long)topk_ids[i];
      if (e >= 0 && e < E) sorted[atomicAdd(&cur[e], 1)] = i;
    }
    if (tile_ext != nullptr) {
      __syncthreads();
      const int nt = warp_sum[1][31];
      for (int i = tid; i < nt; i += kPrepThreads) {
        const Tile t = tiles[i];
        TileExt x;
#pragma unroll
        for (int j = 0; j < 6; ++j) x.ptr[j] = table[(int64_t)t.expert * 6 + j];
#pragma unroll
        for (int j = 0; j < 8; ++j)
          x.slot[j] = sorted[t.row_start + (j < t.nrows ? j : 0)];
        x.row_start = t.row_start;
        x.nrows = t.nrows;
        x.pad[0] = x.pad[1] = 0;
        tile_ext[i] = x;
      }
    }
    return;
  }
  // FP8 quantisation of x: 16 threads per 128-element group, 8 elements each.
  const long long groups = (long long)T * (H / kBK);
  long long g = (long long)(blockIdx.x - 1) * (kPrepThreads / 16) + tid / 16;
  if (g >= groups) return;
  const int sub = tid & 15;
  const long long row = g / (H / kBK);
  const int col = (int)(g % (H / kBK)) * kBK + sub * 8;
  const uint4 raw =
      *reinterpret_cast<const uint4*>(x + row * (long long)H + col);
  const int e =
      quant8_group16(raw, xq + row * (long long)H + (col & ~31), sub & 3);
  if (sub == 0) xs[row * (H / kBK) + col / kBK] = (uint8_t)(e + 127);
}

// ---------------------------------------------------------------------------
// block-scaled grouped GEMM
// ---------------------------------------------------------------------------
//
// PHASE 1: A rows 0..127 = w1[f0 + r], rows 128..255 = w3[f0 + r]; B = x rows
//          of the tile's tokens; epilogue -> hq/hs (FP8 h, permuted layout).
// PHASE 2: A rows = w2[n0 + r]; B = hq rows; epilogue -> ybuf (BF16).
// Warp layout WM x WN, each warp MT m16 tiles x NT n8 tiles. WM*MT*16 = 256.

template <int WM, int WN, int MT, int NT, int STAGES>
struct Cfg {
  static constexpr int BT = WN * NT * 8;
  static constexpr int ROWS = WM * MT * 16;  // weight rows per CTA
  static constexpr int NTHR = WM * WN * 32;
  static constexpr int kA = ROWS * (kBK / 2);  // FP4 bytes per stage
  static constexpr int kB = BT * kBK;          // FP8 bytes per stage
  static constexpr int kSA = ROWS * 4;         // 4 ue8m0 per row per stage
  static constexpr int kStage = kA + kB + kSA;
  static constexpr int kSmem = kStage * STAGES;
  static constexpr int kMaxKC = 64;  // K <= 8192
  static constexpr int kSmemTotal = kSmem + (BT <= 64 ? BT * kMaxKC : 0);
  // Phase 1 covers ROWS/2 features; a 128-feature activation-scale block
  // spans CLUSTER CTAs that exchange their per-token amax through DSMEM.
  static constexpr int CLUSTER = 128 / (ROWS / 2);
  static_assert(ROWS == NTHR, "one A-scale row per thread");
  static_assert(CLUSTER == 1 || CLUSTER == 2, "64 or 128 features per CTA");
};

struct GemmArgs {
  const int64_t* table;  // [E, 6]: w1, s1, w3, s3, w2, s2 device pointers
  const int* sorted;
  const Tile* tiles;
  const int* num_tiles;
  const float* topk_w;  // [T*topk]
  int topk;
  int K;              // reduction dim (H for phase 1, I for phase 2)
  int N;              // output features (I for phase 1, H for phase 2)
  const uint8_t* bq;  // activations FP8 (x for phase 1, h for phase 2)
  const uint8_t* bs;  // activation ue8m0 scales [rows, K/128]
  uint8_t* hq;        // phase 1 out
  uint8_t* hs;
  __nv_bfloat16* ybuf;  // phase 2 out [T*topk, H]
  float swiglu_limit;
};

template <int PHASE, int WM, int WN, int MT, int NT, int STAGES>
__global__ void __launch_bounds__(WM* WN * 32)
    moe_gemm_kernel(const GemmArgs args) {
  using C = Cfg<WM, WN, MT, NT, STAGES>;
  constexpr int BT = C::BT;
  constexpr int ROWS = C::ROWS;
  constexpr int kThreads = C::NTHR;
  extern __shared__ __align__(128) uint8_t smem[];

  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int wm = warp % WM, wn = warp / WM;
  const int g = lane >> 2, t4 = lane & 3;
  const int K = args.K;
  const int KC = K / kBK;
  const int ntiles = *args.num_tiles;

  for (int tile_id = blockIdx.y; tile_id < ntiles; tile_id += gridDim.y) {
    const Tile tile = args.tiles[tile_id];
    const int64_t* ent = args.table + (int64_t)tile.expert * 6;
    const int n0 = blockIdx.x * (PHASE == 1 ? ROWS / 2 : ROWS);
    const uint8_t *wa, *wb, *sa_g, *sb_g;
    if (PHASE == 1) {
      wa = reinterpret_cast<const uint8_t*>(ent[0]);
      sa_g = reinterpret_cast<const uint8_t*>(ent[1]);
      wb = reinterpret_cast<const uint8_t*>(ent[2]);
      sb_g = reinterpret_cast<const uint8_t*>(ent[3]);
    } else {
      wa = wb = reinterpret_cast<const uint8_t*>(ent[4]);
      sa_g = sb_g = reinterpret_cast<const uint8_t*>(ent[5]);
    }
    const int kpack = K / 2, kscale = K / 32;

    // B row source for this thread's copy slots; out-of-range rows zero-fill.
    auto b_row = [&](int r) -> long long {
      int rr = r < tile.nrows ? r : 0;
      int p = tile.row_start + rr;
      return PHASE == 1 ? (long long)(args.sorted[p] / args.topk) : p;
    };

    constexpr int kBChunks = (BT * 8 + kThreads - 1) / kThreads;
    long long b_src[kBChunks];
#pragma unroll
    for (int i = 0; i < kBChunks; ++i) {
      int c = tid + i * kThreads;
      b_src[i] = b_row(c >> 3) * K + (c & 7) * 16;
    }

    auto load_stage = [&](int kc, int buf) {
      uint8_t* sA = smem + buf * C::kStage;
      uint8_t* sB = sA + C::kA;
      uint8_t* sS = sB + C::kB;
      // A: 256 rows x 64 B
#pragma unroll
      for (int i = 0; i < C::kA / 16 / kThreads; ++i) {
        int c = tid + i * kThreads;
        int r = c >> 2, cc = c & 3;
        const uint8_t* src;
        if (PHASE == 1)
          src = (r < ROWS / 2 ? wa + (long long)(n0 + r) * kpack
                              : wb + (long long)(n0 + r - ROWS / 2) * kpack);
        else
          src = wa + (long long)(n0 + r) * kpack;
        src += kc * (kBK / 2) + cc * 16;
        cp_async16(smem_u32(sA + r * 64 + ((cc ^ ((r >> 1) & 3)) << 4)), src,
                   16);
      }
      // A scales: 256 rows x 4 B
      {
        int r = tid;
        const uint8_t* src;
        if (PHASE == 1)
          src = (r < ROWS / 2 ? sa_g + (long long)(n0 + r) * kscale
                              : sb_g + (long long)(n0 + r - ROWS / 2) * kscale);
        else
          src = sa_g + (long long)(n0 + r) * kscale;
        cp_async4(smem_u32(sS + r * 4), src + kc * 4);
      }
      // B: BT rows x 128 B
#pragma unroll
      for (int i = 0; i < kBChunks; ++i) {
        int c = tid + i * kThreads;
        if (c < BT * 8) {
          int r = c >> 3, cc = c & 7;
          cp_async16(smem_u32(sB + r * 128 + ((cc ^ (r & 7)) << 4)),
                     args.bq + b_src[i] + kc * kBK, r < tile.nrows ? 16 : 0);
        }
      }
    };

    // activation scale row offsets for this lane's columns (token g of n8)
    long long sb_off[NT];
#pragma unroll
    for (int ni = 0; ni < NT; ++ni)
      sb_off[ni] = b_row(wn * NT * 8 + ni * 8 + g) * KC;

    float acc[MT][NT][4];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi)
#pragma unroll
      for (int ni = 0; ni < NT; ++ni)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[mi][ni][q] = 0.f;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
      if (s < KC) load_stage(s, s);
      cp_async_commit();
    }
    // Activation scales: small token tiles keep all K/128 scales of their
    // rows in shared memory (a per-stage global load would put one L2
    // round trip on the critical path of every short stage); wide tiles
    // prefetch them one stage ahead in registers.
    constexpr bool kSbSmem = BT <= 64;
    uint8_t* sSB = smem + C::kSmem;
    uint32_t sb_cur[NT];
    if constexpr (kSbSmem) {
      for (int i = tid; i < BT * KC; i += kThreads) {
        int r = i / KC, c = i - r * KC;
        sSB[i] = args.bs[b_row(r) * KC + c];
      }
    } else {
#pragma unroll
      for (int ni = 0; ni < NT; ++ni) sb_cur[ni] = args.bs[sb_off[ni]];
    }

    // warp's A rows (within the 256-row CTA tile) for m16 tile mi
    auto a_row0 = [&](int mi) -> int {
      if (PHASE == 1) {
        constexpr int H2 = MT / 2;
        return (mi < H2 ? 0 : ROWS / 2) + wm * H2 * 16 + (mi % H2) * 16;
      }
      return wm * MT * 16 + mi * 16;
    };

    for (int kc = 0; kc < KC; ++kc) {
      cp_async_wait<STAGES - 2>();
      __syncthreads();
      {
        int nk = kc + STAGES - 1;
        if (nk < KC) load_stage(nk, nk % STAGES);
        cp_async_commit();
      }
      uint32_t sb_next[NT];
      if constexpr (kSbSmem) {
#pragma unroll
        for (int ni = 0; ni < NT; ++ni)
          sb_cur[ni] = sSB[(wn * NT * 8 + ni * 8 + g) * KC + kc];
      } else if (kc + 1 < KC) {
#pragma unroll
        for (int ni = 0; ni < NT; ++ni)
          sb_next[ni] = args.bs[sb_off[ni] + kc + 1];
      }
      const uint8_t* sA = smem + (kc % STAGES) * C::kStage;
      const uint8_t* sB = sA + C::kA;
      const uint8_t* sS = sB + C::kB;
      uint32_t sfa[MT];
#pragma unroll
      for (int mi = 0; mi < MT; ++mi)
        sfa[mi] = *reinterpret_cast<const uint32_t*>(
            sS + (a_row0(mi) + g + 8 * (t4 & 1)) * 4);
#pragma unroll
      for (int jp = 0; jp < 2; ++jp) {  // pairs of k32 blocks
        uint32_t af[MT][2][4];
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
          int q = lane >> 3;
          int r = a_row0(mi) + (lane & 7) + 8 * (q & 1);
          int ch = 2 * jp + (q >> 1);
          uint32_t w0, w1, w2, w3;
          ldsm_x4(smem_u32(sA + r * 64 + ((ch ^ ((r >> 1) & 3)) << 4)), w0, w1,
                  w2, w3);
          expand_fp4(w0, af[mi][0][0], af[mi][0][2]);
          expand_fp4(w1, af[mi][0][1], af[mi][0][3]);
          expand_fp4(w2, af[mi][1][0], af[mi][1][2]);
          expand_fp4(w3, af[mi][1][1], af[mi][1][3]);
        }
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
          int r = wn * NT * 8 + ni * 8 + (lane & 7);
          int ch = 4 * jp + (lane >> 3);
          uint32_t b0, b1, b2, b3;
          ldsm_x4(smem_u32(sB + r * 128 + ((ch ^ (r & 7)) << 4)), b0, b1, b2,
                  b3);
#pragma unroll
          for (int mi = 0; mi < MT; ++mi) {
            if (jp == 0) {
              mma_fp4_fp8<0>(acc[mi][ni], af[mi][0], b0, b1, sfa[mi],
                             sb_cur[ni]);
              mma_fp4_fp8<1>(acc[mi][ni], af[mi][1], b2, b3, sfa[mi],
                             sb_cur[ni]);
            } else {
              mma_fp4_fp8<2>(acc[mi][ni], af[mi][0], b0, b1, sfa[mi],
                             sb_cur[ni]);
              mma_fp4_fp8<3>(acc[mi][ni], af[mi][1], b2, b3, sfa[mi],
                             sb_cur[ni]);
            }
          }
        }
      }
      if (!kSbSmem && kc + 1 < KC) {
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) sb_cur[ni] = sb_next[ni];
      }
    }
    cp_async_wait<0>();
    __syncthreads();

    // ------------------------------------------------------------------
    // epilogue
    if constexpr (PHASE == 1) {
      constexpr int H2 = MT / 2;
      constexpr int FW = ROWS / 2;                  // features per CTA
      float* red = reinterpret_cast<float*>(smem);  // [WM][BT]
      float* red2 = red + WM * BT;                  // [BT] CTA column max
      uint8_t* outq = reinterpret_cast<uint8_t*>(red2 + BT);  // [BT][FW]
      float cmax[NT][2];
#pragma unroll
      for (int ni = 0; ni < NT; ++ni) {
        cmax[ni][0] = cmax[ni][1] = 0.f;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          int col = wn * NT * 8 + ni * 8 + 2 * t4 + c;
          float w = 0.f;
          if (col < tile.nrows)
            w = args.topk_w[args.sorted[tile.row_start + col]];
#pragma unroll
          for (int mi = 0; mi < H2; ++mi) {
#pragma unroll
            for (int hr = 0; hr < 2; ++hr) {
              float gt = bf16_round(acc[mi][ni][2 * hr + c]);
              float up = bf16_round(acc[mi + H2][ni][2 * hr + c]);
              if (args.swiglu_limit > 0.f) {
                up = fminf(fmaxf(up, -args.swiglu_limit), args.swiglu_limit);
                gt = fminf(gt, args.swiglu_limit);
              }
              float h = gt / (1.0f + __expf(-gt)) * up;
              h = bf16_round(h * w);
              acc[mi][ni][2 * hr + c] = h;
              cmax[ni][c] = fmaxf(cmax[ni][c], fabsf(h));
            }
          }
        }
      }
#pragma unroll
      for (int ni = 0; ni < NT; ++ni)
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          float m = cmax[ni][c];
          m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 4));
          m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 8));
          m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 16));
          if (g == 0) red[wm * BT + wn * NT * 8 + ni * 8 + 2 * t4 + c] = m;
        }
      __syncthreads();
      for (int col = tid; col < BT; col += kThreads) {
        float m = 0.f;
#pragma unroll
        for (int w = 0; w < WM; ++w) m = fmaxf(m, red[w * BT + col]);
        red2[col] = m;
      }
      bool write_scale = true;
      if constexpr (C::CLUSTER == 2) {
        cg::cluster_group cl = cg::this_cluster();
        cl.sync();
        const float* peer = cl.map_shared_rank(red2, cl.block_rank() ^ 1);
        for (int col = tid; col < BT; col += kThreads)
          red[col] = fmaxf(red2[col], peer[col]);
        write_scale = cl.block_rank() == 0;
        __syncthreads();
      } else {
        __syncthreads();
        for (int col = tid; col < BT; col += kThreads) red[col] = red2[col];
        __syncthreads();
      }
      const int fblk = n0 / kBK;  // which 128-block of I
#pragma unroll
      for (int ni = 0; ni < NT; ++ni)
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          int col = wn * NT * 8 + ni * 8 + 2 * t4 + c;
          int e = ue8m0_exponent(red[col]);
          float inv = pow2(-e);
#pragma unroll
          for (int mi = 0; mi < H2; ++mi)
#pragma unroll
            for (int hr = 0; hr < 2; ++hr) {
              int f = wm * H2 * 16 + mi * 16 + g + 8 * hr;  // 0..FW-1
              float q = acc[mi][ni][2 * hr + c] * inv;
              __nv_fp8_storage_t b =
                  __nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
              outq[col * FW + (f & ~31) + perm32(f & 31)] = b;
            }
          if (write_scale && wm == 0 && g == 0 && col < tile.nrows)
            args.hs[(long long)(tile.row_start + col) * (args.N / kBK) + fblk] =
                (uint8_t)(e + 127);
        }
      __syncthreads();
      for (int c = tid; c < BT * (FW / 16); c += kThreads) {
        int col = c / (FW / 16), cc = c % (FW / 16);
        if (col < tile.nrows)
          *reinterpret_cast<uint4*>(args.hq +
                                    (long long)(tile.row_start + col) * args.N +
                                    n0 + cc * 16) =
              *reinterpret_cast<const uint4*>(outq + col * FW + cc * 16);
      }
      if constexpr (C::CLUSTER == 2) {
        cg::this_cluster().sync();  // peer finished reading red2
      } else {
        __syncthreads();
      }
    } else {
      __nv_bfloat16* outb = reinterpret_cast<__nv_bfloat16*>(smem);
      constexpr int LD = ROWS + 8;  // [BT][LD] bf16
#pragma unroll
      for (int mi = 0; mi < MT; ++mi)
#pragma unroll
        for (int ni = 0; ni < NT; ++ni)
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            int row = a_row0(mi) + g + 8 * (q >> 1);
            int col = wn * NT * 8 + ni * 8 + 2 * t4 + (q & 1);
            outb[col * LD + row] = __float2bfloat16(acc[mi][ni][q]);
          }
      __syncthreads();
      for (int c = tid; c < BT * (ROWS / 8); c += kThreads) {
        int col = c / (ROWS / 8), cc = c % (ROWS / 8);
        if (col < tile.nrows) {
          int slot = args.sorted[tile.row_start + col];
          *reinterpret_cast<uint4*>(args.ybuf + (long long)slot * args.N + n0 +
                                    cc * 8) =
              *reinterpret_cast<const uint4*>(outb + col * LD + cc * 8);
        }
      }
      __syncthreads();
    }
  }
}

// ---------------------------------------------------------------------------
// combine: out[t] = sum over slots (ascending expert id) of ybuf[slot]
// ---------------------------------------------------------------------------

// Sum of a token's slot outputs (8 columns) in ascending expert-id order,
// starting from 0 like the reference `y = zeros; y[idx] += expert(...)`.
template <bool kCoherent, typename IdT>
__device__ __forceinline__ void combine8(const __nv_bfloat16* ybuf,
                                         const IdT* topk_ids, int t, int topk,
                                         int E, int H, int col, float* out) {
  int order[16];
  long long key[16];
  int n = 0;
  for (int k = 0; k < topk && k < 16; ++k) {
    long long e = (long long)topk_ids[(long long)t * topk + k];
    if (e < 0 || e >= E) continue;
    int j = n++;
    while (j > 0 && key[j - 1] > e) {
      key[j] = key[j - 1];
      order[j] = order[j - 1];
      --j;
    }
    key[j] = e;
    order[j] = k;
  }
  float s[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) s[i] = 0.f;
  for (int j = 0; j < n; ++j) {
    const uint4* src = reinterpret_cast<const uint4*>(
        ybuf + ((long long)t * topk + order[j]) * H + col);
    uint4 raw = kCoherent ? __ldcg(src) : *src;
    const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
    for (int i = 0; i < 8; ++i) s[i] += __bfloat162float(v[i]);
  }
  float4* o = reinterpret_cast<float4*>(out + (long long)t * H + col);
  o[0] = make_float4(s[0], s[1], s[2], s[3]);
  o[1] = make_float4(s[4], s[5], s[6], s[7]);
}

template <typename IdT>
__global__ void combine_kernel(const __nv_bfloat16* __restrict__ ybuf,
                               const IdT* __restrict__ topk_ids, int topk,
                               int E, int H, float* __restrict__ out) {
  const int t = blockIdx.x;
  const int col = (blockIdx.y * blockDim.x + threadIdx.x) * 8;
  if (col >= H) return;
  combine8<false>(ybuf, topk_ids, t, topk, E, H, col, out);
}

// ---------------------------------------------------------------------------
// decode: one persistent kernel for small token tiles (<= 8 rows per expert)
// ---------------------------------------------------------------------------
//
// Work units are pulled from an atomic queue: first every gate|up unit
// (32 features = 64 weight rows of one tile), then every down unit (64 rows
// of w2). A down unit waits until all gate|up units of its tile have
// published h (BF16 in global memory) and quantises those h rows to FP8 in
// shared memory itself. The last down unit of each 64-column block sums the
// token slots (combine). Units are taken in queue order and gate|up units
// never wait, so a waiting unit only depends on units that are running.
//
// The cp.async ring runs across unit boundaries: the producer side walks the
// stream of (unit, stage) pairs and dequeues the next unit as soon as the
// current one has no stage left to issue, so weight loads of the next unit
// are in flight during the current unit's tail, epilogue and dependency wait.
// Weight stages are 256 K-elements wide: every weight row contributes a full
// 128-byte line per stage, which keeps the DRAM access pattern streaming.
// Warps 0-1 take the first 128 K of each stage, warps 2-3 the second; the two
// partial sums are added once per unit.

constexpr int kDecThreads = 256;  // 8 warps, each 2 m16 tiles
constexpr int kDecRows = 64;      // weight rows per work unit
constexpr int kDecRowWarps = kDecRows / 32;
constexpr int kDecKSplit = kDecThreads / 32 / kDecRowWarps;  // K shares
static_assert(kDecKSplit >= 1 && kDecKSplit <= 4, "k64 quarters per stage");
constexpr int kDecK = 256;
constexpr int kDecStages = 5;
constexpr int kDecMaxI = 2560;
constexpr int kDecStageA = kDecRows * kDecK / 2;    // 8 KB
constexpr int kDecStageSA = kDecRows * kDecK / 32;  // 512 B
constexpr int kDecStageB = 8 * kDecK;               // 2 KB
constexpr int kDecStage = kDecStageA + kDecStageSA + kDecStageB;
constexpr int kDecBres = 8 * (kDecMaxI + 16);  // row stride I + 16 bytes
constexpr int kDecRed = (kDecKSplit - 1) * kDecRowWarps * 32 * 8 * 4;
constexpr int kDecOut = 8 * (kDecRows + 8) * 2;
// in-flight work units: the producer runs up to kDecStages - 1 stages (so up
// to that many units of >= 1 stage) ahead of the consumer
constexpr int kDecRing = 16;
static_assert(kDecRing > kDecStages, "unit ring must cover the lookahead");
constexpr long long kDecL2Budget = 64ll << 20;  // w2 L2 prefetch limit
constexpr int kDecSmem = kDecStage * kDecStages + kDecBres +
                         kDecRing * (8 * 64 + (int)sizeof(TileExt)) + kDecRed +
                         kDecOut + 128;
static_assert(kDecSmem <= 101376, "decode kernel shared memory budget");

struct DecodeArgs {
  const int64_t* table;
  const TileExt* tiles;
  const int* num_tiles;
  const float* topk_w;
  const void* topk_ids;
  int topk, E, T, H, I;
  const uint8_t* xq;
  const uint8_t* xs;
  __nv_bfloat16* hbuf;  // [n_slots_sorted, I]
  __nv_bfloat16* ybuf;  // [T * topk, H]
  float* out;           // [T, H]
  int* queue;           // [1]
  int* tile_done;       // [max tiles]
  int* rb_done;         // [H / 64]
  float swiglu_limit;
};

// Decoded work unit.
struct DecUnit {
  bool valid, up;
  int tl, blk, n0, KS;
};

// Queue order: gate|up units of tiles 0..L-1, then gate|up of tile L+i
// followed by the down units of tile i, ..., then the last L tiles' down
// units. Every down unit comes after all gate|up units of its tile (the
// minimal-index unfinished unit can always run, so waiting cannot deadlock),
// and with a lag of L tiles (about one wave of CTAs) a tile's h is normally
// ready when its down units are dequeued: down units overlap with the
// gate|up stream instead of forming a tail.
__device__ __forceinline__ DecUnit decode_unit(const DecodeArgs& a, int u,
                                               int nt, int U1, int U2, int L) {
  DecUnit d;
  d.valid = u < nt * (U1 + U2);
  if (u < L * U1) {
    d.up = true;
    d.tl = u / U1;
    d.blk = u % U1;
  } else {
    const int v = u - L * U1;
    const int pair = v / (U1 + U2);
    if (pair < nt - L) {
      const int r = v - pair * (U1 + U2);
      d.up = r < U1;
      d.tl = d.up ? L + pair : pair;
      d.blk = d.up ? r : r - U1;
    } else {
      const int w = v - (nt - L) * (U1 + U2);
      d.up = false;
      d.tl = nt - L + w / U2;
      d.blk = w % U2;
    }
  }
  d.n0 = d.up ? d.blk * (kDecRows / 2) : d.blk * kDecRows;
  d.KS = (d.up ? a.H : a.I) / kDecK;
  return d;
}

template <typename IdT>
__global__ void __launch_bounds__(kDecThreads, 1)
    moe_decode_kernel(const DecodeArgs a) {
  extern __shared__ __align__(128) uint8_t smem[];
  // Bres is only touched by the consumer between units: single buffer.
  uint8_t* sBres = smem + kDecStage * kDecStages;  // [8][I + 16] FP8
  uint8_t* sSB0 = sBres + kDecBres;                // ring x [8][64] ue8m0
  TileExt* sTE = reinterpret_cast<TileExt*>(sSB0 + kDecRing * 8 * 64);
  float* sRed = reinterpret_cast<float*>(sTE + kDecRing);
  __nv_bfloat16* sOut = reinterpret_cast<__nv_bfloat16*>(
      reinterpret_cast<uint8_t*>(sRed) + kDecRed);
  int* sMisc = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sOut) +
                                      kDecOut);  // [0..15] tickets, [16] flag

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  // warp -> (row group wr, K share kq): K share kq takes the k64 quarters
  // [kq * kQ, (kq + 1) * kQ) of every 256-wide stage
  const int wr = warp % kDecRowWarps, kq = warp / kDecRowWarps;
  constexpr int kQ = 4 / kDecKSplit;
  const int ldb = a.I + 16;
  const int KCx = a.H / kBK;

  asm volatile("griddepcontrol.wait;" ::: "memory");
  const int ntiles = *a.num_tiles;
  const int U1 = a.I / (kDecRows / 2), U2 = a.H / kDecRows;
  // lag (in tiles) between a tile's gate|up units and its down units
  const int lag = min(ntiles, (int)((gridDim.x + U1 - 1) / U1));
  // When all down-projection weights of the layer fit comfortably in L2
  // (few active experts, e.g. batch 1), gate|up units also pull a 1/U1 slice
  // of their tile's w2 into L2, so the last down units (which must wait for
  // h) read from L2 instead of extending the layer after the gate|up stream.
  // With more experts the down units overlap the stream anyway and the
  // prefetch only competes with it (measured: b1 -3 %, b4 +3 % time).
  const long long w2_bytes = (long long)a.H * a.I / 2;
  const bool pf_w2 = (long long)ntiles * w2_bytes <= kDecL2Budget;

  // ---- producer state
  int p_count = 0;  // units dequeued so far
  DecUnit pu;
  int pks = 0;
  long long xoff = 0;
  const uint8_t *w_lo = nullptr, *w_hi = nullptr, *s_lo = nullptr,
                *s_hi = nullptr;
  int pnrows = 0;
  // activation scales of the newest gate|up unit, stored to its smem slot
  // one stage later (their load latency hides behind that stage)
  uint32_t pf_sb[4];
  int pf_slot = -1;
  // The first unit of CTA i is unit i (so the first wave is exactly the head
  // of the queue); later units come from the atomic queue. Thread 0 keeps one
  // ticket in flight so the atomic's round trip is never on the critical
  // path.
  int ticket = blockIdx.x;

  auto flush_sb = [&]() {
    if (pf_slot < 0) return;
    pf_slot = tid < 128 ? pf_slot : -1;
    if (pf_slot < 0) return;
    uint8_t* d = sSB0 + pf_slot * 8 * 64;
    const int r = tid >> 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int c = (tid & 15) + 16 * j;
      if (c < KCx) d[r * 64 + c] = (uint8_t)pf_sb[j];
    }
    pf_slot = -1;
  };
  auto next_unit = [&]() {
    flush_sb();
    const int slot = p_count & (kDecRing - 1);
    if (tid == 0) {
      sMisc[slot] = ticket;
      ticket = gridDim.x + atomicAdd(a.queue, 1);
    }
    __syncthreads();
    pu = decode_unit(a, sMisc[slot], ntiles, U1, U2, lag);
    ++p_count;
    pks = 0;
    if (!pu.valid) return;
    const TileExt* tex = a.tiles + pu.tl;
    if (tid < (int)sizeof(TileExt) / 4)
      reinterpret_cast<int*>(sTE + slot)[tid] =
          reinterpret_cast<const int*>(tex)[tid];
    const int j0 = pu.up ? 0 : 4;
    w_lo = reinterpret_cast<const uint8_t*>(tex->ptr[j0]);
    s_lo = reinterpret_cast<const uint8_t*>(tex->ptr[j0 + 1]);
    w_hi = pu.up ? reinterpret_cast<const uint8_t*>(tex->ptr[2]) : w_lo;
    s_hi = pu.up ? reinterpret_cast<const uint8_t*>(tex->ptr[3]) : s_lo;
    pnrows = tex->nrows;
    if (pu.up && pf_w2) {
      const long long b0 = w2_bytes * pu.blk / U1 & ~127ll;
      const long long b1 = w2_bytes * (pu.blk + 1) / U1 & ~127ll;
      prefetch_l2_range(reinterpret_cast<const uint8_t*>(tex->ptr[4]) + b0,
                        b1 - b0, tid, kDecThreads);
      const long long c0 = w2_bytes / 16 * pu.blk / U1 & ~127ll;
      const long long c1 = w2_bytes / 16 * (pu.blk + 1) / U1 & ~127ll;
      prefetch_l2_range(reinterpret_cast<const uint8_t*>(tex->ptr[5]) + c0,
                        c1 - c0, tid, kDecThreads);
    }
    if (pu.up && tid < 128) {
      const long long tok = tex->slot[tid >> 4] / a.topk;
      xoff = tok * a.H + (tid & 15) * 16;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        int c = (tid & 15) + 16 * j;
        pf_sb[j] = c < KCx ? a.xs[tok * KCx + c] : 0;
      }
      pf_slot = slot;
    }
  };
  auto issue = [&](int buf) {  // one stage of pu, then advance
    flush_sb();
    if (pu.valid) {
      const int K = pu.up ? a.H : a.I;
      const int kpack = K / 2, kscale = K / 32;
      uint8_t* sA = smem + buf * kDecStage;
      uint8_t* sS = sA + kDecStageA;
      uint8_t* sB = sS + kDecStageSA;
#pragma unroll
      for (int i = 0; i < kDecStageA / 16 / kDecThreads; ++i) {
        int c = tid + i * kDecThreads;
        int r = c >> 3, cc = c & 7;
        bool hi = pu.up && r >= kDecRows / 2;
        int wrow = pu.up ? (r & (kDecRows / 2 - 1)) : r;
        const uint8_t* src = (hi ? w_hi : w_lo) +
                             (pu.n0 + wrow) * (long long)kpack +
                             pks * (kDecK / 2) + cc * 16;
        cp_async16(smem_u32(sA + r * 128 + ((cc ^ (r & 7)) << 4)), src, 16);
      }
      if (tid < kDecRows) {
        int r = tid;
        bool hi = pu.up && r >= kDecRows / 2;
        int wrow = pu.up ? (r & (kDecRows / 2 - 1)) : r;
        const uint8_t* src =
            (hi ? s_hi : s_lo) + (pu.n0 + wrow) * (long long)kscale + pks * 8;
        cp_async4(smem_u32(sS + r * 8), src);
        cp_async4(smem_u32(sS + r * 8 + 4), src + 4);
      }
      if (pu.up && tid < 128) {
        int r = tid >> 4, cc = tid & 15;
        cp_async16(smem_u32(sB + r * 256 + ((cc ^ (r & 7)) << 4)),
                   a.xq + xoff + pks * kDecK, r < pnrows ? 16 : 0);
      }
      ++pks;
    }
    cp_async_commit();
  };

  next_unit();
#pragma unroll 1
  for (int s = 0; s < kDecStages - 1; ++s) {
    if (pu.valid && pks == pu.KS) next_unit();
    issue(s);
  }

  // Combine of a 64-column block, run by the unit that completed it last.
  // The completion counter's result is checked one unit later (or at exit)
  // so its round trip does not stall the CTA.
  int last_ticket = -1, last_blk = 0;  // thread 0 only
  auto maybe_combine = [&]() {
    if (tid == 0) sMisc[kDecRing] = last_ticket == ntiles - 1 ? last_blk : -1;
    __syncthreads();
    const int cblk = sMisc[kDecRing];
    __syncthreads();
    if (cblk < 0 || a.out == nullptr) return;
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
    const IdT* ids = reinterpret_cast<const IdT*>(a.topk_ids);
    for (int i = tid; i < a.T * (kDecRows / 8); i += kDecThreads) {
      int t = i / (kDecRows / 8), c8 = i % (kDecRows / 8);
      combine8<true>(a.ybuf, ids, t, a.topk, a.E, a.H, cblk * kDecRows + c8 * 8,
                     a.out);
    }
  };

  // ---- consumer
  int c_count = 0;  // units consumed
  int stage = 0;    // global stage counter (ring slot = stage % kDecStages)
  for (;;) {
    const int cslot = c_count & (kDecRing - 1);
    const DecUnit cu = decode_unit(a, sMisc[cslot], ntiles, U1, U2, lag);
    maybe_combine();
    if (tid == 0) last_ticket = -1;
    if (!cu.valid) break;
    const TileExt& tex = sTE[cslot];
    const int nrows = tex.nrows, row_start = tex.row_start;
    uint8_t* sSB = sSB0 + cslot * 8 * 64;
    if (!cu.up) {
      // wait for every gate|up unit of this tile, then quantise its h rows
      if (tid == 0) {
        const int* flag = a.tile_done + cu.tl;
        int v;
        for (;;) {
          asm volatile("ld.acquire.gpu.global.s32 %0, [%1];"
                       : "=r"(v)
                       : "l"(flag)
                       : "memory");
          if (v >= U1) break;
          __nanosleep(32);
        }
      }
      __syncthreads();
      const int nkb = a.I / kBK;
      const int groups = nrows * nkb;  // padded rows are never stored
      for (int g0 = tid / 16; g0 < groups; g0 += 4 * (kDecThreads / 16)) {
        uint4 raw[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          int gi = g0 + j * (kDecThreads / 16);
          raw[j] = make_uint4(0, 0, 0, 0);
          if (gi < groups) {
            int r = gi / nkb, kb = gi - r * nkb;
            raw[j] = __ldcg(reinterpret_cast<const uint4*>(
                a.hbuf + (long long)(row_start + r) * a.I + kb * kBK +
                (tid & 15) * 8));
          }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          int gi = g0 + j * (kDecThreads / 16);
          if (gi < groups) {  // uniform across the 16 lanes of a group
            int r = gi / nkb, kb = gi - r * nkb;
            int sub = tid & 15;
            int e = quant8_group16(
                raw[j], sBres + r * ldb + kb * kBK + (sub >> 2) * 32, sub & 3);
            if (sub == 0) sSB[r * 64 + kb] = (uint8_t)(e + 127);
          }
        }
      }
    }

    float acc[2][4];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[mi][q] = 0.f;
    // warp's 2 m16 tiles: gate|up -> rows 16wr (gate), R/2+16wr (up)
    const int ar0 = cu.up ? 16 * wr : 32 * wr;
    const int ar1 = cu.up ? kDecRows / 2 + 16 * wr : 32 * wr + 16;

#pragma unroll 1
    for (int ks = 0; ks < cu.KS; ++ks, ++stage) {
      cp_async_wait<kDecStages - 2>();
      __syncthreads();
      if (pu.valid && pks == pu.KS) next_unit();
      issue((stage + kDecStages - 1) % kDecStages);
      const uint8_t* sA = smem + (stage % kDecStages) * kDecStage;
      const uint8_t* sS = sA + kDecStageA;
      const uint8_t* sB = sS + kDecStageSA;
#pragma unroll
      for (int qi = 0; qi < kQ; ++qi) {
        const int Q = kq * kQ + qi;  // k64 quarter of the stage
        const int kh = Q >> 1, jp = Q & 1;
        const int kc = ks * 2 + kh;
        const uint32_t sfb = sSB[g * 64 + kc];
        uint32_t sfa[2];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
          sfa[mi] = *reinterpret_cast<const uint32_t*>(
              sS + ((mi ? ar1 : ar0) + g + 8 * (t4 & 1)) * 8 + 4 * kh);
        uint32_t af[2][2][4];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
          int q = lane >> 3;
          int r = (mi ? ar1 : ar0) + (lane & 7) + 8 * (q & 1);
          int ch = 4 * kh + 2 * jp + (q >> 1);
          uint32_t w0, w1, w2, w3;
          ldsm_x4(smem_u32(sA + r * 128 + ((ch ^ (r & 7)) << 4)), w0, w1, w2,
                  w3);
          expand_fp4(w0, af[mi][0][0], af[mi][0][2]);
          expand_fp4(w1, af[mi][0][1], af[mi][0][3]);
          expand_fp4(w2, af[mi][1][0], af[mi][1][2]);
          expand_fp4(w3, af[mi][1][1], af[mi][1][3]);
        }
        uint32_t b0, b1, b2, b3;
        {
          int r = lane & 7;
          if (cu.up) {
            int ch = 8 * kh + 4 * jp + (lane >> 3);
            ldsm_x4(smem_u32(sB + r * 256 + ((ch ^ (r & 7)) << 4)), b0, b1, b2,
                    b3);
          } else {
            int ch = (kc * kBK + 64 * jp) / 16 + (lane >> 3);
            ldsm_x4(smem_u32(sBres + r * ldb + ch * 16), b0, b1, b2, b3);
          }
        }
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
          if (jp == 0) {
            mma_fp4_fp8<0>(acc[mi], af[mi][0], b0, b1, sfa[mi], sfb);
            mma_fp4_fp8<1>(acc[mi], af[mi][1], b2, b3, sfa[mi], sfb);
          } else {
            mma_fp4_fp8<2>(acc[mi], af[mi][0], b0, b1, sfa[mi], sfb);
            mma_fp4_fp8<3>(acc[mi], af[mi][1], b2, b3, sfa[mi], sfb);
          }
        }
      }
    }

    // add the other K shares into share 0 (fixed order: deterministic)
    if (kq > 0)
#pragma unroll
      for (int i = 0; i < 8; ++i)
        sRed[(((kq - 1) * kDecRowWarps + wr) * 32 + lane) * 8 + i] =
            acc[i >> 2][i & 3];
    __syncthreads();
    if (kq == 0)
#pragma unroll
      for (int j = 0; j < kDecKSplit - 1; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i)
          acc[i >> 2][i & 3] +=
              sRed[((j * kDecRowWarps + wr) * 32 + lane) * 8 + i];

    if (cu.up) {
      // h = bf16(silu(clamp(gate)) * clamp(up) * w) -> hbuf (BF16)
      if (kq == 0) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          int col = 2 * t4 + c;
          if (col >= nrows) continue;
          float w = a.topk_w[tex.slot[col]];
#pragma unroll
          for (int hr = 0; hr < 2; ++hr) {
            float gt = bf16_round(acc[0][2 * hr + c]);
            float up = bf16_round(acc[1][2 * hr + c]);
            if (a.swiglu_limit > 0.f) {
              up = fminf(fmaxf(up, -a.swiglu_limit), a.swiglu_limit);
              gt = fminf(gt, a.swiglu_limit);
            }
            float hv = gt / (1.0f + __expf(-gt)) * up;
            int f = cu.n0 + 16 * wr + g + 8 * hr;
            a.hbuf[(long long)(row_start + col) * a.I + f] =
                __float2bfloat16(hv * w);
          }
        }
      }
      __syncthreads();
      if (tid == 0)
        asm volatile(
            "red.release.gpu.global.add.s32 [%0], 1;" ::"l"(a.tile_done + cu.tl)
            : "memory");
    } else {
      constexpr int LD = kDecRows + 8;
      if (kq == 0)
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            int row = (mi ? ar1 : ar0) + g + 8 * (q >> 1);
            int col = 2 * t4 + (q & 1);
            sOut[col * LD + row] = __float2bfloat16(acc[mi][q]);
          }
      __syncthreads();
      if (tid < 8 * (kDecRows / 8)) {
        int col = tid / (kDecRows / 8), cc = tid % (kDecRows / 8);
        if (col < nrows) {
          int slot = tex.slot[col];
          *reinterpret_cast<uint4*>(a.ybuf + (long long)slot * a.H + cu.n0 +
                                    cc * 8) =
              *reinterpret_cast<const uint4*>(sOut + col * LD + cc * 8);
        }
      }
      __syncthreads();
      if (tid == 0) {
        asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;"
                     : "=r"(last_ticket)
                     : "l"(a.rb_done + cu.blk)
                     : "memory");
        last_blk = cu.blk;
      }
    }
    ++c_count;
  }
  cp_async_wait<0>();
}

// ---------------------------------------------------------------------------
// host launcher
// ---------------------------------------------------------------------------

template <int PHASE, int WM, int WN, int MT, int NT, int STAGES>
void launch_gemm(const GemmArgs& a, int grid_x, int grid_y, cudaStream_t s) {
  using C = Cfg<WM, WN, MT, NT, STAGES>;
  auto k = moe_gemm_kernel<PHASE, WM, WN, MT, NT, STAGES>;
  static bool init = false;  // per instantiation
  if (!init) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         C::kSmemTotal);
    if (PHASE == 1 && C::CLUSTER > 1)
      cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed,
                           1);
    init = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid_x, grid_y);
  cfg.blockDim = dim3(C::NTHR);
  cfg.dynamicSmemBytes = C::kSmemTotal;
  cfg.stream = s;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = PHASE == 1 ? C::CLUSTER : 1;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, k, a);
}

template <int WM, int WN, int MT, int NT, int STAGES>
void run_cfg(GemmArgs a1, GemmArgs a2, int T, int topk, int E, int H, int I,
             cudaStream_t s) {
  using C = Cfg<WM, WN, MT, NT, STAGES>;
  long long n = (long long)T * topk;
  long long bound = (n + C::BT - 1) / C::BT + (n < E ? n : E);
  int gy = (int)(bound < 65535 ? bound : 65535);
  launch_gemm<1, WM, WN, MT, NT, STAGES>(a1, I / (C::ROWS / 2), gy, s);
  launch_gemm<2, WM, WN, MT, NT, STAGES>(a2, H / C::ROWS, gy, s);
}

template <typename IdT>
void launch_decode(const DecodeArgs& a, cudaStream_t s) {
  auto k = moe_decode_kernel<IdT>;
  static int sms = 0;
  if (sms == 0) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kDecSmem);
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(sms);
  cfg.blockDim = dim3(kDecThreads);
  cfg.dynamicSmemBytes = kDecSmem;
  cfg.stream = s;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, k, a);
}

int choose_bt(long long n, int E) {
  long long active = n < E ? n : E;
  long long avg = (n + active - 1) / (active > 0 ? active : 1);
  if (avg <= 8) return 8;
  if (avg <= 16) return 16;
  if (avg <= 32) return 32;
  if (avg <= 64) return 64;
  return 128;
}

}  // namespace v4moe

// Workspace pointers are carved by the caller (binding) from one allocation.
void v4_moe_forward_cuda(const void* x, const void* topk_ids, bool ids_int64,
                         const float* topk_w, const int64_t* table, int T,
                         int topk, int E, int H, int I, float swiglu_limit,
                         int bt, uint8_t* xq, uint8_t* xs, int* sorted,
                         void* tiles, int* num_tiles, uint8_t* hq, uint8_t* hs,
                         void* ybuf, float* out, void* hbuf, int* counters,
                         int n_counters, void* tile_ext, cudaStream_t stream) {
  using namespace v4moe;
  const bool decode = hbuf != nullptr;
  const long long groups = (long long)T * (H / kBK);
  const long long qblocks =
      (groups + kPrepThreads / 16 - 1) / (kPrepThreads / 16);
  auto* xb = reinterpret_cast<const __nv_bfloat16*>(x);
  if (ids_int64)
    prep_kernel<int64_t><<<1 + qblocks, kPrepThreads, 0, stream>>>(
        xb, reinterpret_cast<const int64_t*>(topk_ids), T, H, topk, E, bt,
        sorted, reinterpret_cast<Tile*>(tiles), num_tiles, xq, xs, counters,
        decode ? n_counters : 0, table,
        decode ? reinterpret_cast<TileExt*>(tile_ext) : nullptr);
  else
    prep_kernel<int32_t><<<1 + qblocks, kPrepThreads, 0, stream>>>(
        xb, reinterpret_cast<const int32_t*>(topk_ids), T, H, topk, E, bt,
        sorted, reinterpret_cast<Tile*>(tiles), num_tiles, xq, xs, counters,
        decode ? n_counters : 0, table,
        decode ? reinterpret_cast<TileExt*>(tile_ext) : nullptr);

  if (decode) {
    DecodeArgs d{};
    d.table = table;
    d.tiles = reinterpret_cast<const TileExt*>(tile_ext);
    d.num_tiles = num_tiles;
    d.topk_w = topk_w;
    d.topk_ids = topk_ids;
    d.topk = topk;
    d.E = E;
    d.T = T;
    d.H = H;
    d.I = I;
    d.xq = xq;
    d.xs = xs;
    d.hbuf = reinterpret_cast<__nv_bfloat16*>(hbuf);
    d.ybuf = reinterpret_cast<__nv_bfloat16*>(ybuf);
    d.out = out;
    d.queue = counters;
    d.tile_done = counters + 1;
    d.rb_done = counters + 1 + (n_counters - 1 - H / kDecRows);
    d.swiglu_limit = swiglu_limit;
    if (ids_int64)
      launch_decode<int64_t>(d, stream);
    else
      launch_decode<int32_t>(d, stream);
    return;
  }

  GemmArgs a1{};
  a1.table = table;
  a1.sorted = sorted;
  a1.tiles = reinterpret_cast<const Tile*>(tiles);
  a1.num_tiles = num_tiles;
  a1.topk_w = topk_w;
  a1.topk = topk;
  a1.K = H;
  a1.N = I;
  a1.bq = xq;
  a1.bs = xs;
  a1.hq = hq;
  a1.hs = hs;
  a1.swiglu_limit = swiglu_limit;
  GemmArgs a2 = a1;
  a2.K = I;
  a2.N = H;
  a2.bq = hq;
  a2.bs = hs;
  a2.ybuf = reinterpret_cast<__nv_bfloat16*>(ybuf);

  switch (bt) {
    case 8:
      run_cfg<4, 1, 2, 1, 5>(a1, a2, T, topk, E, H, I, stream);
      break;
    case 16:
      run_cfg<8, 1, 2, 2, 5>(a1, a2, T, topk, E, H, I, stream);
      break;
    case 32:
      run_cfg<8, 1, 2, 4, 4>(a1, a2, T, topk, E, H, I, stream);
      break;
    case 64:
      run_cfg<4, 2, 4, 4, 3>(a1, a2, T, topk, E, H, I, stream);
      break;
    default:
      run_cfg<4, 2, 4, 8, 3>(a1, a2, T, topk, E, H, I, stream);
      break;
  }

  if (out == nullptr) return;  // single-slot callers read ybuf directly
  const int cthreads = 128;
  dim3 cgrid(T, (H / 8 + cthreads - 1) / cthreads);
  auto* yb = reinterpret_cast<const __nv_bfloat16*>(ybuf);
  if (ids_int64)
    combine_kernel<int64_t><<<cgrid, cthreads, 0, stream>>>(
        yb, reinterpret_cast<const int64_t*>(topk_ids), topk, E, H, out);
  else
    combine_kernel<int32_t><<<cgrid, cthreads, 0, stream>>>(
        yb, reinterpret_cast<const int32_t*>(topk_ids), topk, E, H, out);
}

int v4_moe_choose_bt(long long n, int E) { return v4moe::choose_bt(n, E); }
int v4_moe_decode_counters(int bound, int H) {
  return 1 + bound + H / v4moe::kDecRows;
}
bool v4_moe_use_decode(int bt, int H, int I) {
  return bt == 8 && H % v4moe::kDecK == 0 && I % v4moe::kDecK == 0 &&
         I <= v4moe::kDecMaxI && H <= 8192;
}
int v4_moe_tile_bound(long long n, int E, int bt) {
  return (int)((n + bt - 1) / bt + (n < E ? n : E));
}
