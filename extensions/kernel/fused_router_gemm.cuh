// Fused prefill routing for wide routers (E in (128, 256]): one CUTLASS 2.x
// tensor-core mainloop (mma.sync bf16, fp32 accumulate -- exact products)
// whose CTA covers ALL experts of BM = 64 tokens, followed by a scoring +
// top-k epilogue on the CTA's logits tile in shared memory. Logits never
// leave the SM: X and W are read once per tile, only the routing outputs are
// written (#315).
//
// Hardware: RTX_PRO_6000 gate. The 64 x 256 tile, 4 stages, the 99 KiB
// shared-memory budget and the "tiles fit one wave of 188 SMs" dispatch rule
// were tuned on the RTX PRO 6000 Blackwell; fused_route() checks the gate, so
// other devices never reach this kernel (they run the torch routing).
//
// Included by fused_router.cu (needs its EpilogueParams / helpers).

#include "cutlass/gemm/threadblock/default_mma.h"

namespace moe_router {

// Tile tables (RTX PRO 6000): 64 x 256 x 4 stages while the tiles fit one
// wave; 32 x 256 x 5 stages for 1-4 waves (smaller last-wave loss);
// 128 x 256 x 3 stages (half the gate-weight re-reads per token) for many
// waves.
template <int BM, int BN>
struct WideCfg {
  static constexpr int kBM = BM;
  static constexpr int kBK = 32;
  static constexpr int kStages = BM >= 128 ? 3 : (BM <= 32 ? 5 : 4);
  using TB = cutlass::gemm::GemmShape<kBM, BN, kBK>;
  using Warp = cutlass::gemm::GemmShape<kBM / 2, 64, kBK>;
  using Inst = cutlass::gemm::GemmShape<16, 8, 16>;
  using Mma = typename cutlass::gemm::threadblock::DefaultMma<
      cutlass::bfloat16_t, cutlass::layout::RowMajor, 8, cutlass::bfloat16_t,
      cutlass::layout::ColumnMajor, 8, float, cutlass::layout::RowMajor,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, Warp, Inst,
      kStages, cutlass::arch::OpMultiplyAdd>::ThreadblockMma;
  static constexpr int kWarpsM = kBM / Warp::kM;  // 2
  static constexpr int kWarpsN = BN / Warp::kN;
  static constexpr int kThreads = 32 * kWarpsM * kWarpsN;
  static constexpr int kRowsPerRound = kBM / kWarpsM;  // rows per staging
  // epilogue: 8 lanes per row (16 for 32-row tiles, so all 256 threads work)
  static constexpr int kLanesPerRow = kThreads / (kBM / kWarpsM) > 8 ? 16 : 8;
  static constexpr int kRowsPerPass = kThreads / kLanesPerRow;  // epilogue rows
  // One warp row of the fp32 tile is staged at a time.
  static constexpr int kTileBytes = kRowsPerRound * BN * 4;
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

template <int BM, int BN, int SC, int KMAX>
__global__ void __launch_bounds__(WideCfg<BM, BN>::kThreads, 1)
    wide_route_pro6000_kernel(const cutlass::bfloat16_t* __restrict__ x,
                              int64_t ldx,
                              const cutlass::bfloat16_t* __restrict__ w,
                              int64_t ldw, int H, EpilogueParams p) {
  using C = WideCfg<BM, BN>;
  using Mma = typename C::Mma;
  extern __shared__ __align__(128) unsigned char smem[];
  __shared__ uint32_t sel_bits[C::kRowsPerRound][BN / 32];
  __shared__ float sbias[kMaxE];
  auto& storage = *reinterpret_cast<typename Mma::SharedStorage*>(smem);
  const int T = p.T, E = p.E, K = p.K;
  const int tid = threadIdx.x, lane = tid & 31;
  const int warp = __shfl_sync(0xffffffffu, tid >> 5, 0);
  if (p.bias != nullptr)
    for (int e = tid; e < E; e += C::kThreads) sbias[e] = p.bias[e];
  const int n_tiles = (T + C::kBM - 1) / C::kBM;
  const bool want_flags = p.flags_host != nullptr;
  // Same warp -> tile mapping as the CUTLASS multistage mainloop.
  const int warp_mn = warp % (C::kWarpsM * C::kWarpsN);
  const int warp_m = warp_mn % C::kWarpsM, warp_n = warp_mn / C::kWarpsM;
  for (int tile = blockIdx.x; tile < n_tiles; tile += gridDim.x) {
    const int t_base = tile * C::kBM;

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

    float* cs = reinterpret_cast<float*>(smem);  // [kRowsPerRound][BN]
    for (int round = 0; round < C::kWarpsM; ++round) {
      if (warp_m == round) {
        typename Mma::Operator::IteratorC it_c(
            typename Mma::Operator::IteratorC::TensorRef(
                cs, cutlass::layout::RowMajor(BN)),
            lane);
        it_c.add_tile_offset({0, warp_n});
        it_c.store(acc);
      }
      __syncthreads();
      for (int pass = 0; pass < C::kRowsPerRound / C::kRowsPerPass; ++pass) {
        const int row = pass * C::kRowsPerPass + tid / C::kLanesPerRow;
        const int j = tid % C::kLanesPerRow;
        const int t = t_base + round * C::kRowsPerRound + row;
        {
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
            if ((k & (C::kLanesPerRow - 1)) == j) {
              if (k < C::kLanesPerRow)
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
            const int k = j + C::kLanesPerRow * q;
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
          __syncwarp();  // all logits of the row read before weights replace
                         // them
          if (t < T) {
#pragma unroll
            for (int q = 0; q < 2; ++q) {
              const int k = j + C::kLanesPerRow * q;
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
      }
      __syncthreads();
      // Dense outputs, coalesced over the staged rows x E block.
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
  }
  publish_flags(p, gridDim.x);
}

template <int BM, int BN, int SC, int KMAX>
void launch_wide(const at::Tensor& x, const at::Tensor& w, int H,
                 const EpilogueParams& p, cudaStream_t stream, int device) {
  using C = WideCfg<BM, BN>;
  auto kernel = wide_route_pro6000_kernel<BM, BN, SC, KMAX>;
  static bool attr[16] = {false};
  if (!attr[device & 15]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::kSmem));
    attr[device & 15] = true;
  }
  const int grid = (p.T + C::kBM - 1) / C::kBM;
  kernel<<<grid, C::kThreads, C::kSmem, stream>>>(
      reinterpret_cast<const cutlass::bfloat16_t*>(x.data_ptr()), x.stride(0),
      reinterpret_cast<const cutlass::bfloat16_t*>(w.data_ptr()), w.stride(0),
      H, p);
}

// Returns false when the wide kernel does not apply.
inline bool try_launch_wide(const at::Tensor& x, const at::Tensor& w, int H,
                            const EpilogueParams& p, int sc, bool big_tiles,
                            cudaStream_t stream, int device) {
  if (x.scalar_type() != at::kBFloat16 || p.n_group > 1 ||
      p.logits_out != nullptr || p.K > kWideMaxK || p.E <= 128 || p.E > 256)
    return false;
  // The call site passes a tile height, but this argument is a bool, so every
  // non-zero value used to select the 128-row tile. Pick 64 vs 128 by wave
  // fill on this GPU: 9K is 144/188 with 64-row tiles and 72/188 with 128-row
  // tiles; 50K is 0.85 vs 0.71. Equal fill keeps 128-row (half the gate
  // re-reads). 12K ties; 200K stays 128-row.
  const int sms = at::cuda::getDeviceProperties(device)->multiProcessorCount;
  const int n64 = (p.T + 63) / 64;
  const int n128 = (p.T + 127) / 128;
  const int w64 = sms > 0 ? (n64 + sms - 1) / sms : 1;
  const int w128 = sms > 0 ? (n128 + sms - 1) / sms : 1;
  const bool use128 =
      static_cast<long long>(n128) * w64 >= static_cast<long long>(n64) * w128;
  auto go = [&](auto bm, auto bn) {
    constexpr int BM = decltype(bm)::value;
    constexpr int BN = decltype(bn)::value;
    auto by_k = [&](auto km) {
      constexpr int KM = decltype(km)::value;
      if (sc == kSoftmax)
        launch_wide<BM, BN, kSoftmax, KM>(x, w, H, p, stream, device);
      else if (sc == kSigmoid)
        launch_wide<BM, BN, kSigmoid, KM>(x, w, H, p, stream, device);
      else
        launch_wide<BM, BN, kSqrtSoftplus, KM>(x, w, H, p, stream, device);
    };
    if (p.K <= 8)
      by_k(std::integral_constant<int, 8>{});
    else
      by_k(std::integral_constant<int, kWideMaxK>{});
  };
  // 384 / 512-wide tiles need more threads than the 64 x 32 A tile has
  // 16-byte accesses (CUTLASS thread map); those routers keep cuBLASLt.
  (void)big_tiles;
  if (use128)
    go(std::integral_constant<int, 128>{}, std::integral_constant<int, 256>{});
  else
    go(std::integral_constant<int, 64>{}, std::integral_constant<int, 256>{});
  return true;
}

}  // namespace moe_router
