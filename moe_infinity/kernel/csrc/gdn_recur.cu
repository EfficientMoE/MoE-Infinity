// Chunked delta-rule recurrence with fused output, sm_80+ mma.sync (#304).
//
// Hardware: RTX_PRO_6000. Tuned on the RTX PRO 6000 Blackwell (188 SMs,
// 99 KiB shared memory per block): 92.5 KiB of shared tiles per CTA and one
// wave of 128 CTAs. Launched only via moe_infinity/kernel/gdn_recur_cuda.py,
// which checks hw_profile.supports(RTX_PRO_6000); other GPUs run fla.
//
// One CTA = one (sequence, head, 32-column value slice); 4 warps, each warp
// owns 8 value columns of the state h [K=128, V] and runs its recurrence in
// registers (fp32 accumulators). Per 64-token chunk (operands precomputed by
// the prep kernel, block-local buffers):
//   x = w @ h;  v = u - x;  o = qg @ h + S @ v;  h = d * h + kd^T @ v
// with d = exp2(g_last) (GDN scalar gate; S must be tril-masked).
// h and v are the B operands (n = 8 value columns): their fp32 accumulators
// are packed to bf16 and transposed in registers with movmatrix; w, qg, S
// (row-major) and kd^T (ldmatrix.trans) are A operands from shared memory,
// prefetched with cp.async (w / qg double-buffered, kd / S / u single).
// recur_ws_pro6000_kernel (default) moves the cp.async issue to 4 producer
// warps and hands tiles over with named barriers.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace gdn {

constexpr int BT = 64;
constexpr int KD = 128;
constexpr int NW = 4;          // warps per CTA
constexpr int BV = 8 * NW;     // value columns per CTA
constexpr int TILE = BT * KD;  // elements of a [64, 128] tile

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem,
                                           bool valid) {
  int n = valid ? 16 : 0;
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(smem)),
      "l"(gmem), "r"(n));
}
__device__ __forceinline__ void cp_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1,
                                        uint32_t& r2, uint32_t& r3,
                                        uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t& r0, uint32_t& r1,
                                          uint32_t& r2, uint32_t& r3,
                                          uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(addr));
}
__device__ __forceinline__ uint32_t movt(uint32_t x) {
  uint32_t y;
  asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(y) : "r"(x));
  return y;
}
__device__ __forceinline__ uint32_t pack(float a, float b) {
  __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ void mma(float* d, uint32_t a0, uint32_t a1,
                                    uint32_t a2, uint32_t a3, uint32_t b0,
                                    uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// XOR-swizzled offset (elements) of the 16-byte chunk `c` of row `r` in a
// tile with `cpr` chunks per row.
__device__ __forceinline__ int swz(int r, int c, int cpr) {
  return r * cpr * 8 + ((c ^ (r & 7)) & (cpr - 1)) * 8;
}

// Async copy of a [64, cols] bf16 tile (row stride `ld` elements) into a
// swizzled shared tile; rows >= nt are zero-filled.
template <int COLS, int NTHR = NW * 32>
__device__ __forceinline__ void load_tile(__nv_bfloat16* s,
                                          const __nv_bfloat16* g, long ld,
                                          int nt) {
  constexpr int CPR = COLS / 8;
  for (int i = threadIdx.x; i < BT * CPR; i += NTHR) {
    int r = i / CPR, c = i % CPR;
    bool ok = r < nt;
    cp_async16(s + swz(r, c, CPR), g + (ok ? r : 0) * ld + c * 8, ok);
  }
}

// B fragments (k16 x n8) of a [16*KT, 8] fp32 accumulator stack.
template <int KT>
__device__ __forceinline__ void to_b(const float (*acc)[4], uint32_t (*b)[2]) {
#pragma unroll
  for (int i = 0; i < KT; ++i) {
    b[i][0] = movt(pack(acc[i][0], acc[i][1]));
    b[i][1] = movt(pack(acc[i][2], acc[i][3]));
  }
}

__global__ void __launch_bounds__(NW * 32) recur_pro6000_kernel(
    const __nv_bfloat16* __restrict__ qg, const __nv_bfloat16* __restrict__ kd,
    const __nv_bfloat16* __restrict__ w, const __nv_bfloat16* __restrict__ u,
    const __nv_bfloat16* __restrict__ S, const float* __restrict__ g,
    __nv_bfloat16* __restrict__ o, float* __restrict__ h, int T, int t_base,
    int TB, int nb, int H) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  __nv_bfloat16* sW = reinterpret_cast<__nv_bfloat16*>(smem_raw);  // 2 tiles
  __nv_bfloat16* sQ = sW + 2 * TILE;                               // 2 tiles
  __nv_bfloat16* sK = sQ + 2 * TILE;                               // 1 tile
  __nv_bfloat16* sS = sK + TILE;                                   // 64x64
  __nv_bfloat16* sU = sS + BT * BT;                                // 64xBV

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int gq = lane >> 2, qq = lane & 3;
  const int nv = KD / BV;
  const int i_v = blockIdx.x % nv;
  const int i_nh = blockIdx.x / nv;
  const int i_n = i_nh / H, i_h = i_nh % H;
  const int vcol = i_v * BV + warp * 8 + 2 * qq;  // this thread's 2 columns

  const long ldk = (long)H * KD;  // row stride of [*, H, 128] tensors
  const long ldS = (long)H * BT;
  const __nv_bfloat16* pw = w + ((long)i_n * TB * H + i_h) * KD;
  const __nv_bfloat16* pq = qg + ((long)i_n * TB * H + i_h) * KD;
  const __nv_bfloat16* pk = kd + ((long)i_n * TB * H + i_h) * KD;
  const __nv_bfloat16* pu = u + ((long)i_n * TB * H + i_h) * KD + i_v * BV;
  const __nv_bfloat16* pS = S + ((long)i_n * TB * H + i_h) * BT;
  __nv_bfloat16* po = o + ((long)i_n * T * H + i_h) * KD;
  const float* pg = g + (long)i_n * T * H + i_h;
  float* ph = h + (long)i_nh * KD * KD;

  // state: 8 row tiles (16 state rows each) x [c0..c3]
  float hs[8][4];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float2 a = *reinterpret_cast<const float2*>(ph + (16 * i + gq) * KD + vcol);
    float2 b =
        *reinterpret_cast<const float2*>(ph + (16 * i + gq + 8) * KD + vcol);
    hs[i][0] = a.x;
    hs[i][1] = a.y;
    hs[i][2] = b.x;
    hs[i][3] = b.y;
  }

  const int NC = (nb + BT - 1) / BT;
  auto issue_wq = [&](int c, int buf) {
    int nt = min(BT, nb - c * BT);
    load_tile<KD>(sW + buf * TILE, pw + (long)c * BT * ldk, ldk, nt);
    load_tile<KD>(sQ + buf * TILE, pq + (long)c * BT * ldk, ldk, nt);
  };
  auto issue_ksu = [&](int c) {
    int nt = min(BT, nb - c * BT);
    load_tile<KD>(sK, pk + (long)c * BT * ldk, ldk, nt);
    load_tile<BT>(sS, pS + (long)c * BT * ldS, ldS, nt);
    for (int i = threadIdx.x; i < BT * (BV / 8); i += NW * 32) {
      int r = i / (BV / 8), cc = i % (BV / 8);
      bool ok = r < nt;
      cp_async16(sU + r * BV + cc * 8,
                 pu + ((long)c * BT + (ok ? r : 0)) * ldk + cc * 8, ok);
    }
  };

  issue_wq(0, 0);
  cp_commit();
  issue_ksu(0);
  cp_commit();

  for (int c = 0; c < NC; ++c) {
    const int buf = c & 1;
    const int t0 = t_base + c * BT;
    const int nt = min(BT, nb - c * BT);
    cp_wait<1>();  // w / qg of chunk c
    __syncthreads();
    if (c + 1 < NC) issue_wq(c + 1, buf ^ 1);
    cp_commit();

    // B fragments of h (bf16, transposed in registers)
    uint32_t hb[8][2];
    to_b<8>(hs, hb);

    // x = w @ h  and  o = qg @ h   (m: 4 tiles of 16 tokens, k: 8 x 16)
    float x[4][4] = {}, oo[4][4] = {};
    const __nv_bfloat16* tW = sW + buf * TILE;
    const __nv_bfloat16* tQ = sQ + buf * TILE;
    // k-step outer, m-tile inner: 8 independent accumulator chains per
    // k-step; the A fragments of a k-step are loaded together (ldmatrix is
    // ordered, mma is not volatile, so the next loads overlap the MMAs).
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      uint32_t aw[4][4], aq[4][4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        int r = 16 * j + (lane & 15);
        int cch = 2 * i + (lane >> 4);
        ldsm_x4(aw[j][0], aw[j][1], aw[j][2], aw[j][3],
                smem_u32(tW + swz(r, cch, 16)));
        ldsm_x4(aq[j][0], aq[j][1], aq[j][2], aq[j][3],
                smem_u32(tQ + swz(r, cch, 16)));
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        mma(x[j], aw[j][0], aw[j][1], aw[j][2], aw[j][3], hb[i][0], hb[i][1]);
        mma(oo[j], aq[j][0], aq[j][1], aq[j][2], aq[j][3], hb[i][0], hb[i][1]);
      }
    }

    cp_wait<1>();  // kd / S / u of chunk c (issued one group earlier)
    __syncthreads();

    // v = u - x  (fp32), then B fragments of v
    float vv[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int r0 = 16 * j + gq, r1 = r0 + 8;
      __nv_bfloat162 u0 = *reinterpret_cast<const __nv_bfloat162*>(
          sU + r0 * BV + warp * 8 + 2 * qq);
      __nv_bfloat162 u1 = *reinterpret_cast<const __nv_bfloat162*>(
          sU + r1 * BV + warp * 8 + 2 * qq);
      vv[j][0] = __bfloat162float(u0.x) - x[j][0];
      vv[j][1] = __bfloat162float(u0.y) - x[j][1];
      vv[j][2] = __bfloat162float(u1.x) - x[j][2];
      vv[j][3] = __bfloat162float(u1.y) - x[j][3];
    }
    uint32_t vb[4][2];
    to_b<4>(vv, vb);

    // o += tril(S) @ v ; store o
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (i > j) continue;
        int r = 16 * j + (lane & 15);
        int cch = 2 * i + (lane >> 4);
        uint32_t a0, a1, a2, a3;
        ldsm_x4(a0, a1, a2, a3, smem_u32(sS + swz(r, cch, 8)));
        mma(oo[j], a0, a1, a2, a3, vb[i][0], vb[i][1]);
      }
      int r0 = 16 * j + gq, r1 = r0 + 8;
      if (r0 < nt)
        *reinterpret_cast<uint32_t*>(po + (long)(t0 + r0) * ldk + vcol) =
            pack(oo[j][0], oo[j][1]);
      if (r1 < nt)
        *reinterpret_cast<uint32_t*>(po + (long)(t0 + r1) * ldk + vcol) =
            pack(oo[j][2], oo[j][3]);
    }

    // h = d * h + kd^T @ v   (m: 8 state-row tiles, k: 4 x 16 tokens)
    const float dl = exp2f(pg[(long)(min(t0 + BT, T) - 1) * H]);
#pragma unroll
    for (int m = 0; m < 8; ++m) {
      hs[m][0] *= dl;
      hs[m][1] *= dl;
      hs[m][2] *= dl;
      hs[m][3] *= dl;
    }
    // k-step (16 tokens) outer, state-row tile inner: 8 independent chains
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      uint32_t ak[8][4];
      const int mi = lane >> 3;  // which 8x8 matrix this lane addresses
      const int t = 16 * i + (mi >> 1) * 8 + (lane & 7);
#pragma unroll
      for (int m = 0; m < 8; ++m) {
        // A[m=d][k=t] = kd[t][d]: 8x8 blocks (t, d) loaded transposed
        int dch = 2 * m + (mi & 1);
        ldsm_x4_t(ak[m][0], ak[m][1], ak[m][2], ak[m][3],
                  smem_u32(sK + swz(t, dch, 16)));
      }
#pragma unroll
      for (int m = 0; m < 8; ++m)
        mma(hs[m], ak[m][0], ak[m][1], ak[m][2], ak[m][3], vb[i][0], vb[i][1]);
    }

    __syncthreads();  // everyone is done with kd / S / u of chunk c
    if (c + 1 < NC) issue_ksu(c + 1);
    cp_commit();
  }
  cp_wait<0>();

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    *reinterpret_cast<float2*>(ph + (16 * i + gq) * KD + vcol) =
        make_float2(hs[i][0], hs[i][1]);
    *reinterpret_cast<float2*>(ph + (16 * i + gq + 8) * KD + vcol) =
        make_float2(hs[i][2], hs[i][3]);
  }
}

// Warp-specialised variant: one extra producer warp issues every cp.async
// (the consumer warps' instruction streams carry only the MMA chain; a probe
// with the loads removed ran 1.71 vs 2.35 us per chunk step). Hand-offs use
// named barriers over all 160 threads: FULL_* (producer arrives, consumers
// sync) and EMPTY_* (consumers arrive, producer syncs). w / qg are double-
// buffered, kd / S / u single-buffered, as in recur_pro6000_kernel.
constexpr int WS_NP = 4;  // producer warps
constexpr int WS_NTHR = (NW + WS_NP) * 32;
enum : int {
  kFullWQ0 = 1,
  kFullWQ1,
  kEmptyWQ0,
  kEmptyWQ1,
  kFullKSU,
  kEmptyKSU
};

__device__ __forceinline__ void nb_sync(int id) {
  asm volatile("bar.sync %0, %1;\n" ::"r"(id), "n"(WS_NTHR) : "memory");
}
__device__ __forceinline__ void nb_arrive(int id) {
  asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "n"(WS_NTHR) : "memory");
}

template <int COLS>
__device__ __forceinline__ void load_tile_w(__nv_bfloat16* s,
                                            const __nv_bfloat16* g, long ld,
                                            int nt, int lane) {
  constexpr int CPR = COLS / 8;
#pragma unroll 4
  for (int i = lane; i < BT * CPR; i += WS_NP * 32) {
    int r = i / CPR, c = i % CPR;
    bool ok = r < nt;
    cp_async16(s + swz(r, c, CPR), g + (ok ? r : 0) * ld + c * 8, ok);
  }
}

__global__ void __launch_bounds__(WS_NTHR) recur_ws_pro6000_kernel(
    const __nv_bfloat16* __restrict__ qg, const __nv_bfloat16* __restrict__ kd,
    const __nv_bfloat16* __restrict__ w, const __nv_bfloat16* __restrict__ u,
    const __nv_bfloat16* __restrict__ S, const float* __restrict__ g,
    __nv_bfloat16* __restrict__ o, float* __restrict__ h, int T, int t_base,
    int TB, int nb, int H) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  __nv_bfloat16* sW = reinterpret_cast<__nv_bfloat16*>(smem_raw);  // 2 tiles
  __nv_bfloat16* sQ = sW + 2 * TILE;                               // 2 tiles
  __nv_bfloat16* sK = sQ + 2 * TILE;                               // 1 tile
  __nv_bfloat16* sS = sK + TILE;                                   // 64x64
  __nv_bfloat16* sU = sS + BT * BT;                                // 64xBV

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int gq = lane >> 2, qq = lane & 3;
  const int nv = KD / BV;
  const int i_v = blockIdx.x % nv;
  const int i_nh = blockIdx.x / nv;
  const int i_n = i_nh / H, i_h = i_nh % H;
  const long ldk = (long)H * KD;
  const long ldS = (long)H * BT;
  const __nv_bfloat16* pw = w + ((long)i_n * TB * H + i_h) * KD;
  const __nv_bfloat16* pq = qg + ((long)i_n * TB * H + i_h) * KD;
  const __nv_bfloat16* pk = kd + ((long)i_n * TB * H + i_h) * KD;
  const __nv_bfloat16* pu = u + ((long)i_n * TB * H + i_h) * KD + i_v * BV;
  const __nv_bfloat16* pS = S + ((long)i_n * TB * H + i_h) * BT;
  const int NC = (nb + BT - 1) / BT;

  if (warp >= NW) {  // ---------------- producers
    const int ptid = threadIdx.x - NW * 32;
    auto load_wq = [&](int c, int buf) {
      int nt = min(BT, nb - c * BT);
      load_tile_w<KD>(sW + buf * TILE, pw + (long)c * BT * ldk, ldk, nt, ptid);
      load_tile_w<KD>(sQ + buf * TILE, pq + (long)c * BT * ldk, ldk, nt, ptid);
      cp_commit();
      cp_wait<0>();
      nb_arrive(buf ? kFullWQ1 : kFullWQ0);
    };
    auto load_ksu = [&](int c) {
      int nt = min(BT, nb - c * BT);
      load_tile_w<KD>(sK, pk + (long)c * BT * ldk, ldk, nt, ptid);
      load_tile_w<BT>(sS, pS + (long)c * BT * ldS, ldS, nt, ptid);
      for (int i = ptid; i < BT * (BV / 8); i += WS_NP * 32) {
        int r = i / (BV / 8), cc = i % (BV / 8);
        bool ok = r < nt;
        cp_async16(sU + r * BV + cc * 8,
                   pu + ((long)c * BT + (ok ? r : 0)) * ldk + cc * 8, ok);
      }
      cp_commit();
      cp_wait<0>();
      nb_arrive(kFullKSU);
    };
    load_wq(0, 0);
    load_ksu(0);
    if (NC > 1) load_wq(1, 1);
    for (int c = 0; c < NC; ++c) {
      if (c + 2 < NC) {
        nb_sync((c & 1) ? kEmptyWQ1 : kEmptyWQ0);
        load_wq(c + 2, c & 1);
      }
      if (c + 1 < NC) {
        nb_sync(kEmptyKSU);
        load_ksu(c + 1);
      }
    }
    return;
  }

  // ---------------- consumers (4 warps x 8 value columns)
  const int vcol = i_v * BV + warp * 8 + 2 * qq;
  __nv_bfloat16* po = o + ((long)i_n * T * H + i_h) * KD;
  const float* pg = g + (long)i_n * T * H + i_h;
  float* ph = h + (long)i_nh * KD * KD;
  float hs[8][4];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float2 a = *reinterpret_cast<const float2*>(ph + (16 * i + gq) * KD + vcol);
    float2 b =
        *reinterpret_cast<const float2*>(ph + (16 * i + gq + 8) * KD + vcol);
    hs[i][0] = a.x;
    hs[i][1] = a.y;
    hs[i][2] = b.x;
    hs[i][3] = b.y;
  }

  for (int c = 0; c < NC; ++c) {
    const int buf = c & 1;
    const int t0 = t_base + c * BT;
    const int nt = min(BT, nb - c * BT);
    nb_sync(buf ? kFullWQ1 : kFullWQ0);

    uint32_t hb[8][2];
    to_b<8>(hs, hb);
    float x[4][4] = {}, oo[4][4] = {};
    const __nv_bfloat16* tW = sW + buf * TILE;
    const __nv_bfloat16* tQ = sQ + buf * TILE;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      uint32_t aw[4][4], aq[4][4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        int r = 16 * j + (lane & 15);
        int cch = 2 * i + (lane >> 4);
        ldsm_x4(aw[j][0], aw[j][1], aw[j][2], aw[j][3],
                smem_u32(tW + swz(r, cch, 16)));
        ldsm_x4(aq[j][0], aq[j][1], aq[j][2], aq[j][3],
                smem_u32(tQ + swz(r, cch, 16)));
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        mma(x[j], aw[j][0], aw[j][1], aw[j][2], aw[j][3], hb[i][0], hb[i][1]);
        mma(oo[j], aq[j][0], aq[j][1], aq[j][2], aq[j][3], hb[i][0], hb[i][1]);
      }
    }
    if (c + 2 < NC) nb_arrive(buf ? kEmptyWQ1 : kEmptyWQ0);

    nb_sync(kFullKSU);
    float vv[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int r0 = 16 * j + gq, r1 = r0 + 8;
      __nv_bfloat162 u0 = *reinterpret_cast<const __nv_bfloat162*>(
          sU + r0 * BV + warp * 8 + 2 * qq);
      __nv_bfloat162 u1 = *reinterpret_cast<const __nv_bfloat162*>(
          sU + r1 * BV + warp * 8 + 2 * qq);
      vv[j][0] = __bfloat162float(u0.x) - x[j][0];
      vv[j][1] = __bfloat162float(u0.y) - x[j][1];
      vv[j][2] = __bfloat162float(u1.x) - x[j][2];
      vv[j][3] = __bfloat162float(u1.y) - x[j][3];
    }
    uint32_t vb[4][2];
    to_b<4>(vv, vb);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (i > j) continue;
        int r = 16 * j + (lane & 15);
        int cch = 2 * i + (lane >> 4);
        uint32_t a0, a1, a2, a3;
        ldsm_x4(a0, a1, a2, a3, smem_u32(sS + swz(r, cch, 8)));
        mma(oo[j], a0, a1, a2, a3, vb[i][0], vb[i][1]);
      }
      int r0 = 16 * j + gq, r1 = r0 + 8;
      if (r0 < nt)
        *reinterpret_cast<uint32_t*>(po + (long)(t0 + r0) * ldk + vcol) =
            pack(oo[j][0], oo[j][1]);
      if (r1 < nt)
        *reinterpret_cast<uint32_t*>(po + (long)(t0 + r1) * ldk + vcol) =
            pack(oo[j][2], oo[j][3]);
    }
    const float dl = exp2f(pg[(long)(min(t0 + BT, T) - 1) * H]);
#pragma unroll
    for (int m = 0; m < 8; ++m) {
      hs[m][0] *= dl;
      hs[m][1] *= dl;
      hs[m][2] *= dl;
      hs[m][3] *= dl;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      uint32_t ak[8][4];
      const int mi = lane >> 3;
      const int t = 16 * i + (mi >> 1) * 8 + (lane & 7);
#pragma unroll
      for (int m = 0; m < 8; ++m) {
        int dch = 2 * m + (mi & 1);
        ldsm_x4_t(ak[m][0], ak[m][1], ak[m][2], ak[m][3],
                  smem_u32(sK + swz(t, dch, 16)));
      }
#pragma unroll
      for (int m = 0; m < 8; ++m)
        mma(hs[m], ak[m][0], ak[m][1], ak[m][2], ak[m][3], vb[i][0], vb[i][1]);
    }
    if (c + 1 < NC) nb_arrive(kEmptyKSU);
  }

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    *reinterpret_cast<float2*>(ph + (16 * i + gq) * KD + vcol) =
        make_float2(hs[i][0], hs[i][1]);
    *reinterpret_cast<float2*>(ph + (16 * i + gq + 8) * KD + vcol) =
        make_float2(hs[i][2], hs[i][3]);
  }
}

constexpr size_t smem_bytes() {
  return (size_t)(5 * TILE + BT * BT + BT * BV) * 2;
}

}  // namespace gdn

// Host entry: shapes as in moe_infinity/kernel/gdn_blocked.py.
void gdn_recur_launch(const void* qg, const void* kd, const void* w,
                      const void* u, const void* S, const float* g, void* o,
                      float* h, int B, int H, int T, int t_base, int TB, int nb,
                      int variant, cudaStream_t stream) {
  dim3 grid(B * H * (gdn::KD / gdn::BV));
  if (variant == 2) {
    size_t sm = gdn::smem_bytes();
    cudaFuncSetAttribute(gdn::recur_ws_pro6000_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    gdn::recur_ws_pro6000_kernel<<<grid, gdn::WS_NTHR, sm, stream>>>(
        (const __nv_bfloat16*)qg, (const __nv_bfloat16*)kd,
        (const __nv_bfloat16*)w, (const __nv_bfloat16*)u,
        (const __nv_bfloat16*)S, g, (__nv_bfloat16*)o, h, T, t_base, TB, nb, H);
    return;
  }
  dim3 block(gdn::NW * 32);
  size_t sm = gdn::smem_bytes();
  cudaFuncSetAttribute(gdn::recur_pro6000_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
  gdn::recur_pro6000_kernel<<<grid, block, sm, stream>>>(
      (const __nv_bfloat16*)qg, (const __nv_bfloat16*)kd,
      (const __nv_bfloat16*)w, (const __nv_bfloat16*)u, (const __nv_bfloat16*)S,
      g, (__nv_bfloat16*)o, h, T, t_base, TB, nb, H);
}
