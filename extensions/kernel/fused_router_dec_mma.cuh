// Decode phase 1 for T >= kBT64MinTokens (b64 / b256): one split-K work item
// (64 tokens x 64 / 128 experts over one H slice) on a CUTLASS 2.x multistage
// mainloop (mma.sync m16n8k16 + ldmatrix, cp.async, 3 stages of BK = 64; fp32
// accumulate -- bf16 x bf16 products are exact, so only the summation order
// differs from an fp32 GEMM). Replaces the WMMA 16x16x16 item, which ran at
// 21 % tensor-pipe / 24 % DRAM utilisation at b256 (#315 round-6 audit).
//
// Hardware: RTX_PRO_6000 gate (tile and stage counts sized for 188 SMs and a
// 99 KiB shared-memory budget); reached only through fused_route(), which
// returns {} elsewhere.
//
// Included by fused_router.cu (needs DecodeArgs / EpilogueParams).

#include "cutlass/gemm/threadblock/default_mma.h"

namespace moe_router {

template <typename Tin>
struct CutlassElem;
template <>
struct CutlassElem<__nv_bfloat16> {
  using T = cutlass::bfloat16_t;
};
template <>
struct CutlassElem<__half> {
  using T = cutlass::half_t;
};

// BN = 128 from b65 (fewer, fatter items); BN = 64 at b64 and below, where
// a 128-wide tile would halve the number of CTAs streaming the weight.
template <typename Tin, int BN>
struct DecMmaCfg {
  using El = typename CutlassElem<Tin>::T;
  // BK = 64 x 3 stages measured 0.3-1.3 us faster than 32 x 4 at b64 / b256.
  static constexpr int kBM = 64, kBN = BN, kBK = 64, kStages = 3;
  using TB = cutlass::gemm::GemmShape<kBM, kBN, kBK>;
  using Warp = cutlass::gemm::GemmShape<32, BN / 4, kBK>;  // 2 x 4 warps
  using Inst = cutlass::gemm::GemmShape<16, 8, 16>;
  using Mma = typename cutlass::gemm::threadblock::DefaultMma<
      El, cutlass::layout::RowMajor, 8, El, cutlass::layout::ColumnMajor, 8,
      float, cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, TB, Warp, Inst, kStages,
      cutlass::arch::OpMultiplyAdd>::ThreadblockMma;
  static constexpr int kWarpsM = kBM / Warp::kM;
  static constexpr int kWarpsN = kBN / Warp::kN;
  static_assert(kWarpsM * kWarpsN * 32 == 256, "decode CTA is 256 threads");
  static constexpr int kSmem =
      sizeof(typename Mma::SharedStorage) > kBM* kBN * 4
          ? static_cast<int>(sizeof(typename Mma::SharedStorage))
          : kBM* kBN * 4;
};

template <typename Tin, int BN>
__device__ void logits_item_mma(const DecodeArgs& a, const EpilogueParams& p,
                                int item, unsigned char* smem) {
  using C = DecMmaCfg<Tin, BN>;
  using Mma = typename C::Mma;
  using El = typename C::El;
  const int tile_t = item % a.n_t;
  const int rest = item / a.n_t;
  const int tile_e = rest % a.n_e, split = rest / a.n_e;
  const int t0 = tile_t * C::kBM, e0 = tile_e * C::kBN;
  const int h_lo = split * a.h_per_split;
  const int h_hi = min(a.H, h_lo + a.h_per_split);
  const int T = p.T, E = p.E;
  const int tid = threadIdx.x, lane = tid & 31;
  const int warp = __shfl_sync(0xffffffffu, tid >> 5, 0);

  auto& storage = *reinterpret_cast<typename Mma::SharedStorage*>(smem);
  const cutlass::layout::RowMajor la(static_cast<int>(a.ldx));
  const cutlass::layout::ColumnMajor lb(static_cast<int>(a.ldw));
  typename Mma::IteratorA::Params pa{la};
  typename Mma::IteratorB::Params pb{lb};
  // Extents end at h_hi so the iterators predicate this split's K range.
  typename Mma::IteratorA it_a(pa, const_cast<El*>(static_cast<const El*>(a.x)),
                               cutlass::MatrixCoord(T, h_hi), tid,
                               cutlass::MatrixCoord(t0, h_lo));
  typename Mma::IteratorB it_b(pb, const_cast<El*>(static_cast<const El*>(a.w)),
                               cutlass::MatrixCoord(h_hi, E), tid,
                               cutlass::MatrixCoord(h_lo, e0));
  Mma mma(storage, tid, warp, lane);
  typename Mma::FragmentC acc;
  acc.clear();
  const int k_iters = h_hi > h_lo ? (h_hi - h_lo + C::kBK - 1) / C::kBK : 0;
  mma(k_iters, acc, it_a, it_b, acc);
  __syncthreads();  // mainloop smem is reused for the accumulator tile

  const int warp_mn = warp % (C::kWarpsM * C::kWarpsN);
  const int warp_m = warp_mn % C::kWarpsM, warp_n = warp_mn / C::kWarpsM;
  float* cs = reinterpret_cast<float*>(smem);  // [kBM][kBN]
  {
    typename Mma::Operator::IteratorC it_c(
        typename Mma::Operator::IteratorC::TensorRef(
            cs, cutlass::layout::RowMajor(C::kBN)),
        lane);
    it_c.add_tile_offset({warp_m, warp_n});
    it_c.store(acc);
  }
  __syncthreads();
  for (int i = tid; i < C::kBM * C::kBN; i += 256) {
    const int r = i / C::kBN, ce = i % C::kBN;
    const int t = t0 + r, e = e0 + ce;
    if (t < T && e < E)
      a.partial[(static_cast<int64_t>(split) * T + t) * E + e] = cs[i];
  }
  __syncthreads();  // smem is reused by the next item
}

}  // namespace moe_router
