// Fused prefill routing for wide routers (E in (128, 256]): one CUTLASS 2.x
// tensor-core mainloop (mma.sync bf16, fp32 accumulate -- exact products)
// whose CTA covers ALL experts of BM = 64 tokens, followed by a scoring +
// top-k epilogue on the CTA's logits tile in shared memory. Logits never
// leave the SM: X and W are read once per tile, only the routing outputs are
// written (#315).
//
// Included by fused_router.cu (needs its EpilogueParams / helpers).

#include "cutlass/gemm/threadblock/default_mma.h"

namespace moe_router {

template <int BN>
struct WideCfg {
  static constexpr int kBM = 64;
  static constexpr int kBK = 32;
  static constexpr int kStages = BN <= 256 ? 4 : (BN <= 384 ? 3 : 2);
  using TB = cutlass::gemm::GemmShape<kBM, BN, kBK>;
  using Warp = cutlass::gemm::GemmShape<32, 64, kBK>;
  using Inst = cutlass::gemm::GemmShape<16, 8, 16>;
  using Mma = typename cutlass::gemm::threadblock::DefaultMma<
      cutlass::bfloat16_t, cutlass::layout::RowMajor, 8, cutlass::bfloat16_t,
      cutlass::layout::ColumnMajor, 8, float, cutlass::layout::RowMajor,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, Warp, Inst,
      kStages, cutlass::arch::OpMultiplyAdd>::ThreadblockMma;
  static constexpr int kWarpsM = kBM / Warp::kM;  // 2
  static constexpr int kWarpsN = BN / Warp::kN;
  static constexpr int kThreads = 32 * kWarpsM * kWarpsN;
  static constexpr int kRowsPerRound = kBM / kWarpsM;  // 32
  static constexpr int kLanesPerRow = 8;               // epilogue
  // The whole [BM, BN] fp32 tile is staged at once so the accumulators die
  // before the epilogue (keeping them live across passes spilled).
  static constexpr int kTileBytes = kBM * BN * 4;
  static constexpr int kSmem = sizeof(typename Mma::SharedStorage) > kTileBytes
                                   ? sizeof(typename Mma::SharedStorage)
                                   : kTileBytes;
};

constexpr int kWideMaxK = 16;  // picks per row: lanes x 2

__device__ __forceinline__ uint64_t wide_key(float sel, int e) {
  return (static_cast<uint64_t>(ord32(sel)) << 32) |
         (0xFFFFFFFFu - static_cast<uint32_t>(e));
}

template <int SC>
__device__ __forceinline__ float wide_score(float v) {
  if (SC == kSoftmax) return v;
  if (SC == kSigmoid) return __frcp_rn(1.0f + expf(-v));
  return sqrtf(v > 20.f ? v : log1pf(expf(v)));
}

template <int BN, int SC, int KMAX>
__global__ void __launch_bounds__(WideCfg<BN>::kThreads, 1)
    wide_route_kernel(const cutlass::bfloat16_t* __restrict__ x, int64_t ldx,
                      const cutlass::bfloat16_t* __restrict__ w, int64_t ldw,
                      int H, EpilogueParams p) {
  using C = WideCfg<BN>;
  using Mma = typename C::Mma;
  extern __shared__ __align__(128) unsigned char smem[];
  __shared__ uint32_t sel_bits[C::kRowsPerRound][BN / 32];
  __shared__ float sbias[kMaxE];
  auto& storage = *reinterpret_cast<typename Mma::SharedStorage*>(smem);
  const int T = p.T, E = p.E, K = p.K;
  const int tid = threadIdx.x, lane = tid & 31;
  const int warp = __shfl_sync(0xffffffffu, tid >> 5, 0);
  const int t_base = blockIdx.x * C::kBM;
  if (p.bias != nullptr)
    for (int e = tid; e < E; e += C::kThreads) sbias[e] = p.bias[e];

  const cutlass::layout::RowMajor la(static_cast<int>(ldx));
  const cutlass::layout::ColumnMajor lb(static_cast<int>(ldw));
  typename Mma::IteratorA::Params pa{la};
  typename Mma::IteratorB::Params pb{lb};
  typename Mma::IteratorA it_a(pa, const_cast<cutlass::bfloat16_t*>(x),
                               cutlass::MatrixCoord(T, H), tid,
                               cutlass::MatrixCoord(t_base, 0));
  typename Mma::IteratorB it_b(pb, const_cast<cutlass::bfloat16_t*>(w),
                               cutlass::MatrixCoord(H, E), tid,
                               cutlass::MatrixCoord(0, 0));
  Mma mma(storage, tid, warp, lane);
  typename Mma::FragmentC acc;
  acc.clear();
  mma((H + C::kBK - 1) / C::kBK, acc, it_a, it_b, acc);
  __syncthreads();  // mainloop smem is reused for the logits tile

  // Same warp -> tile mapping as the CUTLASS multistage mainloop.
  const int warp_mn = warp % (C::kWarpsM * C::kWarpsN);
  const int warp_m = warp_mn % C::kWarpsM, warp_n = warp_mn / C::kWarpsM;
  float* cs = reinterpret_cast<float*>(smem);  // [32][BN]
  const bool want_flags = p.flags_host != nullptr;

  {
    typename Mma::Operator::IteratorC it_c(
        typename Mma::Operator::IteratorC::TensorRef(
            cs, cutlass::layout::RowMajor(BN)),
        lane);
    it_c.add_tile_offset({warp_m, warp_n});
    it_c.store(acc);
  }
  __syncthreads();
  for (int round = 0; round < C::kWarpsM;
       ++round, cs += C::kRowsPerRound * BN) {
    const int row = tid / C::kLanesPerRow, j = tid % C::kLanesPerRow;
    const int t = t_base + round * C::kRowsPerRound + row;
    if (tid < C::kRowsPerRound * C::kLanesPerRow) {
      // K selection rounds over the row's experts (lane j scans j, j + 8,
      // ...; chosen experts are marked in sel_bits). Recomputing the score
      // per round keeps everything in registers -- per-thread top-k arrays
      // ended up in local memory and cost ~1000 cycles per element.
      // Selection values of this lane's experts j, j + 8, ... in registers;
      // K rounds of an 8-lane argmax (lowest index wins ties), chosen
      // experts tracked in a register bitmask.
      constexpr int kPer = BN / C::kLanesPerRow;
      uint32_t selk[kPer];  // ord32(selection value); 0 = not a candidate
#pragma unroll
      for (int i = 0; i < kPer; ++i) {
        const int e = j + C::kLanesPerRow * i;
        float v = e < E ? cs[row * BN + e] : 0.f;
        if (p.round_logits) v = __bfloat162float(__float2bfloat16_rn(v));
        float sel = wide_score<SC>(v);
        if (SC != kSoftmax && p.bias != nullptr && e < E) sel += sbias[e];
        selk[i] = e < E ? ord32(sel) : 0u;
      }
      int mine[2] = {0, 0};  // this lane keeps picks k = j and k = j + 8
      float top_logit = 0.f;
#pragma unroll 1
      for (int k = 0; k < K; ++k) {
        // tree argmax over the lane's slots; equal keys -> lower slot
        uint32_t tk[kPer];
        int ti[kPer];
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
          tk[i] = selk[i];
          ti[i] = i;
        }
#pragma unroll
        for (int w2 = kPer / 2; w2 > 0; w2 >>= 1)
#pragma unroll
          for (int i = 0; i < w2; ++i)
            if (tk[i + w2] > tk[i] ||
                (tk[i + w2] == tk[i] && ti[i + w2] < ti[i])) {
              tk[i] = tk[i + w2];
              ti[i] = ti[i + w2];
            }
        uint64_t best =
            tk[0] == 0u
                ? 0ull
                : (static_cast<uint64_t>(tk[0]) << 32) |
                      (0xFFFFFFFFu -
                       static_cast<uint32_t>(j + C::kLanesPerRow * ti[0]));
#pragma unroll
        for (int o = 1; o < C::kLanesPerRow; o <<= 1) {
          const uint64_t other = __shfl_xor_sync(0xffffffffu, best, o);
          best = other > best ? other : best;
        }
        const int idx =
            static_cast<int>(0xFFFFFFFFu - static_cast<uint32_t>(best));
        if (k == 0) top_logit = unord32(static_cast<uint32_t>(best >> 32));
        if ((idx & (C::kLanesPerRow - 1)) == j) {
          const int slot = idx / C::kLanesPerRow;
#pragma unroll
          for (int i = 0; i < kPer; ++i)
            if (i == slot) selk[i] = 0u;
        }
        if ((k & 7) == j) {
          if (k < 8)
            mine[0] = idx;
          else
            mine[1] = idx;
        }
      }
      // per-row selection bitmask for the dense pass (lane j owns word j)
      if (j < BN / 32) sel_bits[row][j] = 0u;
      __syncwarp();
      // Weights: lane j owns picks j and j + 8; the row sum is an 8-lane
      // xor-tree (fixed order).
      float raw[2] = {0.f, 0.f};
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const int k = j + 8 * q;
        if (k < K) {
          float v = cs[row * BN + mine[q]];
          if (p.round_logits) v = __bfloat162float(__float2bfloat16_rn(v));
          raw[q] = SC == kSoftmax ? expf(v - top_logit) : wide_score<SC>(v);
          atomicOr(&sel_bits[row][mine[q] >> 5], 1u << (mine[q] & 31));
        }
      }
      float inv_all = 1.f;
      if (SC == kSoftmax && !p.renorm) {
        float all = 0.f;
        for (int e = j; e < E; e += C::kLanesPerRow) {
          float v = cs[row * BN + e];
          if (p.round_logits) v = __bfloat162float(__float2bfloat16_rn(v));
          all += expf(v - top_logit);
        }
#pragma unroll
        for (int o = 1; o < C::kLanesPerRow; o <<= 1)
          all += __shfl_xor_sync(0xffffffffu, all, o);
        inv_all = 1.f / all;
      }
      float den = raw[0] + raw[1];
#pragma unroll
      for (int o = 1; o < C::kLanesPerRow; o <<= 1)
        den += __shfl_xor_sync(0xffffffffu, den, o);
      den += p.eps;
      __syncwarp();  // all logits of the row read before weights replace them
      if (t < T) {
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          const int k = j + 8 * q;
          if (k < K) {
            const float r = raw[q] * inv_all;
            const float v = (p.renorm ? r / den : r) * p.scale;
            cs[row * BN + mine[q]] = v;  // the weight replaces its logit
            if (p.ids != nullptr) {
              p.ids[static_cast<int64_t>(t) * K + k] = mine[q];
              p.tw[static_cast<int64_t>(t) * K + k] = v;
            }
            if (want_flags) p.state->flags[mine[q]] = 1;
          }
        }
      }
    }
    __syncthreads();
    // Dense outputs, coalesced over the 32 x E block.
    if (p.mask != nullptr) {
      for (int e = tid; e < E; e += C::kThreads) {
#pragma unroll 4
        for (int r = 0; r < C::kRowsPerRound; ++r) {
          const int tt = t_base + round * C::kRowsPerRound + r;
          if (tt < T) {
            const bool on = (sel_bits[r][e >> 5] >> (e & 31)) & 1u;
            const float v = on ? cs[r * BN + e] : 0.f;
            const int64_t o = static_cast<int64_t>(tt) * E + e;
            p.mask[o] = on;
            store_typed(p.dw, o, v, p.dw_dtype);
          }
        }
      }
    }
    __syncthreads();
  }
  publish_flags(p, gridDim.x);
}

template <int BN, int SC, int KMAX>
void launch_wide(const at::Tensor& x, const at::Tensor& w, int H,
                 const EpilogueParams& p, cudaStream_t stream, int device) {
  using C = WideCfg<BN>;
  auto kernel = wide_route_kernel<BN, SC, KMAX>;
  static bool attr[16] = {false};
  if (!attr[device & 15]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::kSmem));
    attr[device & 15] = true;
  }
  kernel<<<(p.T + C::kBM - 1) / C::kBM, C::kThreads, C::kSmem, stream>>>(
      reinterpret_cast<const cutlass::bfloat16_t*>(x.data_ptr()), x.stride(0),
      reinterpret_cast<const cutlass::bfloat16_t*>(w.data_ptr()), w.stride(0),
      H, p);
}

// Returns false when the wide kernel does not apply.
inline bool try_launch_wide(const at::Tensor& x, const at::Tensor& w, int H,
                            const EpilogueParams& p, int sc,
                            cudaStream_t stream, int device) {
  if (x.scalar_type() != at::kBFloat16 || p.n_group > 1 ||
      p.logits_out != nullptr || p.K > kWideMaxK || p.E <= 128 || p.E > 256)
    return false;
  auto go = [&](auto bn) {
    constexpr int BN = decltype(bn)::value;
    auto by_k = [&](auto km) {
      constexpr int KM = decltype(km)::value;
      if (sc == kSoftmax)
        launch_wide<BN, kSoftmax, KM>(x, w, H, p, stream, device);
      else if (sc == kSigmoid)
        launch_wide<BN, kSigmoid, KM>(x, w, H, p, stream, device);
      else
        launch_wide<BN, kSqrtSoftplus, KM>(x, w, H, p, stream, device);
    };
    if (p.K <= 8)
      by_k(std::integral_constant<int, 8>{});
    else
      by_k(std::integral_constant<int, kWideMaxK>{});
  };
  // 384 / 512-wide tiles need more threads than the 64 x 32 A tile has
  // 16-byte accesses (CUTLASS thread map); those routers keep cuBLASLt.
  go(std::integral_constant<int, 256>{});
  return true;
}

}  // namespace moe_router
