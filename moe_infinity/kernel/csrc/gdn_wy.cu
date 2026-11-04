// GDN WY build for one 64-token chunk, CUDA mma.sync (#304, round 7 S3).
//
// Hardware: RTX_PRO_6000. Tuned on the RTX PRO 6000 Blackwell (188 SMs,
// 99 KiB shared memory per block): ~41 KiB of shared memory per CTA so two
// CTAs (16 warps) share an SM. Launched only via
// moe_infinity/kernel/gdn_recur_cuda.py, which checks
// hw_profile.supports(RTX_PRO_6000); other GPUs run fla.
//
// Per (chunk, head): k_n = k / |k| (optional), L = strict_tril(beta_i k_n,i .
// k_n,j exp2(g_i - g_j)), A = (I + L)^{-1} (fp32 block forward substitution,
// rounded to bf16 like fla), w = A (beta exp2(g) k_n), u = A (beta v).
// kkt and w / u are mma.sync m16n8k16 (bf16 in, fp32 accumulate).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace gdnwy {

constexpr int BT = 64;
constexpr int KD = 128;
constexpr int NW = 8;
constexpr int NTHR = NW * 32;
constexpr int LP = 65;  // fp32 L / A pitch

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
__device__ __forceinline__ void cp_wait_all() {
  asm volatile("cp.async.wait_group 0;\n" ::);
}
__device__ __forceinline__ void ldsm_x4(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t* r, uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(addr));
}
__device__ __forceinline__ void mma(float* d, const uint32_t* a, uint32_t b0,
                                    uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ int swz(int r, int c, int cpr) {
  return r * cpr * 8 + ((c ^ (r & 7)) & (cpr - 1)) * 8;
}
// element offset of (r, col) in a swizzled tile with cpr 16-byte chunks/row
__device__ __forceinline__ int swe(int r, int col, int cpr) {
  return swz(r, col >> 3, cpr) + (col & 7);
}
__device__ __forceinline__ uint32_t pack(float a, float b) {
  __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ void load_rows(__nv_bfloat16* s,
                                          const __nv_bfloat16* g, long ld,
                                          int nt) {
  for (int i = threadIdx.x; i < BT * 16; i += NTHR) {
    int r = i >> 4, c = i & 15;
    bool ok = r < nt;
    cp_async16(s + swz(r, c, 16), g + (ok ? r : 0) * ld + c * 8, ok);
  }
}

// out[64, 128] = A[64, 64] (bf16, swizzled, lower block-triangular) @ X
// (bf16 [64, 128] swizzled, row = token); warp: row tile j, 64-col half.
__device__ __forceinline__ void a_times(const __nv_bfloat16* sA,
                                        const __nv_bfloat16* sX,
                                        __nv_bfloat16* out, long ldo, int nt,
                                        int warp, int lane) {
  const int j = warp >> 1, half = warp & 1;
  const int gq = lane >> 2, qq = lane & 3;
  float acc[8][4] = {};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    if (i > j) break;
    uint32_t a[4];
    ldsm_x4(a,
            smem_u32(sA + swz(16 * j + (lane & 15), 2 * i + (lane >> 4), 8)));
#pragma unroll
    for (int p = 0; p < 8; p += 2) {
      // B[k = token][n = col]: rows 16i.., cols 64*half + 8p ..; transposed
      int mi = lane >> 3;
      int t = 16 * i + (mi & 1) * 8 + (lane & 7);
      int ch = 8 * half + p + (mi >> 1);
      uint32_t b[4];
      ldsm_x4_t(b, smem_u32(sX + swz(t, ch, 16)));
      mma(acc[p], a, b[0], b[1]);
      mma(acc[p + 1], a, b[2], b[3]);
    }
  }
  const int r0 = 16 * j + gq, r1 = r0 + 8;
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    int col = 64 * half + 8 * p + 2 * qq;
    if (r0 < nt)
      *reinterpret_cast<uint32_t*>(out + (long)r0 * ldo + col) =
          pack(acc[p][0], acc[p][1]);
    if (r1 < nt)
      *reinterpret_cast<uint32_t*>(out + (long)r1 * ldo + col) =
          pack(acc[p][2], acc[p][3]);
  }
}

template <bool NORM>
__global__ void __launch_bounds__(NTHR, 2)
    wy_pro6000_kernel(const __nv_bfloat16* __restrict__ k,
                      const __nv_bfloat16* __restrict__ v,
                      const float* __restrict__ gc,
                      const __nv_bfloat16* __restrict__ beta,
                      __nv_bfloat16* __restrict__ w,
                      __nv_bfloat16* __restrict__ u, int T, int t_base, int TB,
                      int H) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(smem_raw);  // 16 KB
  float* sL = reinterpret_cast<float*>(sK + BT * KD);        // 64 x 65 fp32
  __nv_bfloat16* sV = reinterpret_cast<__nv_bfloat16*>(sL);  // reuses sL
  __nv_bfloat16* sA = reinterpret_cast<__nv_bfloat16*>(sL + BT * LP);  // 8 KB
  float* sT = reinterpret_cast<float*>(sA + BT * BT);  // 16 x 17 scratch
  float* sg = sT + 16 * 17;
  float* sb = sg + BT;
  float* sr = sb + BT;

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int gq = lane >> 2, qq = lane & 3;
  const int i_b = blockIdx.y / H, i_h = blockIdx.y % H;
  const int t0 = t_base + blockIdx.x * BT;  // global token of row 0
  const int nt = min(BT, T - t0);
  const long ldk = (long)H * KD;
  const long row0 = (long)i_b * T + t0;
  const __nv_bfloat16* pk = k + (row0 * H + i_h) * KD;
  const __nv_bfloat16* pv = v + (row0 * H + i_h) * KD;
  const long orow0 = (long)i_b * TB + (t0 - t_base);
  __nv_bfloat16* pw = w + (orow0 * H + i_h) * KD;
  __nv_bfloat16* pu = u + (orow0 * H + i_h) * KD;

  load_rows(sK, pk, ldk, nt);
  cp_commit();
  if (threadIdx.x < BT) {
    int r = threadIdx.x;
    bool ok = r < nt;
    sg[r] = ok ? gc[(row0 + r) * H + i_h] : 0.f;
    sb[r] = ok ? __bfloat162float(beta[(row0 + r) * H + i_h]) : 0.f;
  }
  cp_wait_all();
  __syncthreads();

  // row norms: 4 threads per row, 32 columns each
  {
    int r = threadIdx.x >> 2, part = threadIdx.x & 3;
    float ss = 0.f;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const __nv_bfloat16* p = sK + swz(r, 4 * part + c, 16);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        float x = __bfloat162float(p[e]);
        ss += x * x;
      }
    }
    ss += __shfl_xor_sync(0xffffffff, ss, 1);
    ss += __shfl_xor_sync(0xffffffff, ss, 2);
    if (part == 0) sr[r] = NORM ? rsqrtf(ss + 1e-6f) : 1.f;
  }
  __syncthreads();

  // L = strict_tril(beta_i rk_i rk_j (k k^T)_ij exp2(g_i - g_j)), fp32 in sL
  {
    const int j = warp >> 1, half = warp & 1;  // row tile, 32-col half
    float acc[4][4] = {};
    if (32 * half < 16 * j + 16) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        uint32_t a[4];
        ldsm_x4(a, smem_u32(sK + swz(16 * j + (lane & 15), 2 * i + (lane >> 4),
                                     16)));
#pragma unroll
        for (int p = 0; p < 4; p += 2) {
          // B[k = d][n = token]: token rows, d contiguous (non-transposed)
          int mi = lane >> 3;
          int n = 32 * half + 8 * p + (mi >> 1) * 8 + (lane & 7);
          uint32_t b[4];
          ldsm_x4(b, smem_u32(sK + swz(n, 2 * i + (mi & 1), 16)));
          mma(acc[p], a, b[0], b[1]);
          mma(acc[p + 1], a, b[2], b[3]);
        }
      }
    }
#pragma unroll
    for (int p = 0; p < 4; ++p) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        int r = 16 * j + gq + (e >> 1) * 8;
        int c = 32 * half + 8 * p + 2 * qq + (e & 1);
        float val = 0.f;
        if (c < r)
          val = sb[r] * sr[r] * sr[c] * acc[p][e] * exp2f(sg[r] - sg[c]);
        sL[r * LP + c] = val;
      }
    }
  }
  __syncthreads();

  // A = (I + L)^{-1}, in place on the strict lower part of sL.
  // (1) diagonal 16x16 blocks, all four at once: X = (I + L_bb)^{-1} - I.
  {
    const int bb = threadIdx.x >> 4, c = threadIdx.x & 15;  // 16 blocks*16
    for (int r = 1; r < 16; ++r) {
      float x = 0.f;
      if (bb < 4 && c < r) {
        const int R = 16 * bb + r, C = 16 * bb + c;
        x = -sL[R * LP + C];
        for (int kk = c + 1; kk < r; ++kk)
          x -= sL[R * LP + 16 * bb + kk] * sL[(16 * bb + kk) * LP + C];
      }
      __syncthreads();
      if (bb < 4 && c < r) sL[(16 * bb + r) * LP + 16 * bb + c] = x;
      __syncthreads();
    }
  }
  // (2) off-diagonal blocks, row block b, column block jb ascending:
  //     A_b,jb = -(I + X_bb) * sum_{kb = jb..b-1} L_b,kb A_kb,jb
  //     (A_jb,jb = I + X_jb,jb; L_b,kb for kb > jb is still unmodified)
  for (int b = 1; b < 4; ++b) {
    for (int jb = 0; jb < b; ++jb) {
      const int r = threadIdx.x >> 4, c = threadIdx.x & 15;
      float t = 0.f;
      for (int kb = jb; kb < b; ++kb) {
        for (int kk = 0; kk < 16; ++kk) {
          float l = sL[(16 * b + r) * LP + 16 * kb + kk];
          float a;
          if (kb == jb)
            a = (kk == c ? 1.f : 0.f) +
                (kk > c ? sL[(16 * kb + kk) * LP + 16 * jb + c] : 0.f);
          else
            a = sL[(16 * kb + kk) * LP + 16 * jb + c];
          t += l * a;
        }
      }
      sT[r * 17 + c] = t;
      __syncthreads();
      float y = sT[r * 17 + c];
      for (int kk = 0; kk < r; ++kk)
        y += sL[(16 * b + r) * LP + 16 * b + kk] * sT[kk * 17 + c];
      __syncthreads();
      sL[(16 * b + r) * LP + 16 * jb + c] = -y;
      __syncthreads();
    }
  }

  // A to bf16 (fla rounds A to the input dtype before forming w / u)
  for (int i = threadIdx.x; i < BT * BT; i += NTHR) {
    int r = i >> 6, c = i & 63;
    float a = c < r ? sL[r * LP + c] : (c == r ? 1.f : 0.f);
    sA[swe(r, c, 8)] = __float2bfloat16(a);
  }
  __syncthreads();

  // v into the (now free) L region; meanwhile scale k in place
  load_rows(sV, pv, ldk, nt);
  cp_commit();
  for (int i = threadIdx.x; i < BT * KD / 2; i += NTHR) {
    int r = (2 * i) >> 7, col = (2 * i) & 127;
    float s = sb[r] * exp2f(sg[r]) * sr[r];
    __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(sK + swe(r, col, 16));
    float2 x = __bfloat1622float2(*p);
    *p = __floats2bfloat162_rn(x.x * s, x.y * s);
  }
  __syncthreads();
  a_times(sA, sK, pw, ldk, nt, warp, lane);
  cp_wait_all();
  __syncthreads();
  for (int i = threadIdx.x; i < BT * KD / 2; i += NTHR) {
    int r = (2 * i) >> 7, col = (2 * i) & 127;
    float s = sb[r];
    __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(sV + swe(r, col, 16));
    float2 x = __bfloat1622float2(*p);
    *p = __floats2bfloat162_rn(x.x * s, x.y * s);
  }
  __syncthreads();
  a_times(sA, sV, pu, ldk, nt, warp, lane);
}

constexpr size_t smem_bytes() {
  return (size_t)BT * KD * 2 + (size_t)BT * LP * 4 + (size_t)BT * BT * 2 +
         (16 * 17 + 3 * BT) * 4;
}

}  // namespace gdnwy

void gdn_wy_launch(const void* k, const void* v, const float* gc,
                   const void* beta, void* w, void* u, int B, int H, int T,
                   int t_base, int TB, int nb, bool norm, cudaStream_t stream) {
  dim3 grid((nb + gdnwy::BT - 1) / gdnwy::BT, B * H);
  size_t sm = gdnwy::smem_bytes();
  auto launch = [&](auto kern) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)sm);
    kern<<<grid, gdnwy::NTHR, sm, stream>>>(
        (const __nv_bfloat16*)k, (const __nv_bfloat16*)v, gc,
        (const __nv_bfloat16*)beta, (__nv_bfloat16*)w, (__nv_bfloat16*)u, T,
        t_base, TB, H);
  };
  if (norm)
    launch(gdnwy::wy_pro6000_kernel<true>);
  else
    launch(gdnwy::wy_pro6000_kernel<false>);
}
