// Grouped sm_120 blockwise FP8 GEMM for the large-M half of the GLM layer.
// One CUTLASS ptr-array launch covers every expert's gate and up (2 groups
// per expert, same M/N/K). A second launch is the down projection. The BF16
// GEMV stays on the small-M path.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "cute/tensor.hpp"

using namespace cute;

namespace fp8g {

constexpr int kBlock = 128;
constexpr float kFp8Max = 448.f;

struct Expert6 {
  const uint8_t* gate;
  const float* gate_s;
  const uint8_t* up;
  const float* up_s;
  const uint8_t* down;
  const float* down_s;
};

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
constexpr int AlignmentA = 16;
constexpr int AlignmentB = 16;
constexpr int AlignmentC = 8;
constexpr int AlignmentD = 8;

using MmaTileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;
using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<1, kBlock, kBlock>;
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, MmaTileShape,
        ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, float, ElementC, LayoutC*, AlignmentC, ElementD,
        LayoutD*, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, ElementA,
        cute::tuple<LayoutA*, LayoutSFA*>, AlignmentA, ElementB,
        cute::tuple<LayoutB*, LayoutSFB*>, AlignmentB, ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelScheduleSm120Blockwise>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape,
                                                        CollectiveMainloop,
                                                        CollectiveEpilogue,
                                                        void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;

struct Group {
  int expert;
  int M;
  int row0;
};

__device__ __forceinline__ int find_expert(const int* offsets, int E, int row) {
  int lo = 0, hi = E;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (offsets[mid + 1] <= row) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

__global__ void pack_rows_kernel(const __nv_bfloat16* __restrict__ x, int H,
                                 int topk, const int* __restrict__ offsets,
                                 const int* __restrict__ slot, int E, int rows,
                                 __nv_bfloat16* __restrict__ packed) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int e = find_expert(offsets, E, row);
  const int local = row - offsets[e];
  const int id = slot[offsets[e] + local];
  const int token = id / topk;
  const int col = threadIdx.x;
  for (int c = col; c < H; c += blockDim.x) {
    packed[static_cast<int64_t>(row) * H + c] =
        x[static_cast<int64_t>(token) * H + c];
  }
}

__global__ void quant_grouped_kernel(const __nv_bfloat16* __restrict__ in,
                                     int ld, const int* __restrict__ offsets,
                                     int E, int K, uint8_t* __restrict__ q,
                                     float* __restrict__ sfa,
                                     const int* __restrict__ sfa_off) {
  const int e = blockIdx.y;
  const int M = offsets[e + 1] - offsets[e];
  const int KB = K / kBlock;
  const int group = blockIdx.x * (blockDim.x / 16) + (threadIdx.x >> 4);
  const bool active = M > 0 && group < M * KB;
  const int local = active ? group / KB : 0;
  const int kb = active ? group - local * KB : 0;
  const int sub = threadIdx.x & 15;
  const int k = kb * kBlock + sub * 8;
  const int grow = active ? offsets[e] + local : 0;
  float v[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) v[i] = 0.f;
  if (active) {
    const uint4 raw = *reinterpret_cast<const uint4*>(
        in + static_cast<int64_t>(grow) * ld + k);
    const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = __bfloat162float(b[i]);
  }
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(v[i]));
#pragma unroll
  for (int o = 8; o > 0; o >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  if (!active) return;
  const float inv = amax > 0.f ? kFp8Max / amax : 1.f;
  uint32_t packed[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(
        make_float2(v[4 * h] * inv, v[4 * h + 1] * inv), __NV_SATFINITE, __NV_E4M3);
    const __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(
        make_float2(v[4 * h + 2] * inv, v[4 * h + 3] * inv), __NV_SATFINITE, __NV_E4M3);
    packed[h] = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
  }
  *reinterpret_cast<uint2*>(q + static_cast<int64_t>(grow) * ld + k) =
      make_uint2(packed[0], packed[1]);
  if (sub == 0) {
    sfa[sfa_off[e] + kb * M + local] = amax > 0.f ? amax / kFp8Max : 1.f;
  }
}

__global__ void silu_quant_kernel(const __nv_bfloat16* __restrict__ gate,
                                  const __nv_bfloat16* __restrict__ up, int I,
                                  const int* __restrict__ offsets, int E,
                                  uint8_t* __restrict__ q, float* __restrict__ sfa,
                                  const int* __restrict__ sfa_off) {
  const int e = blockIdx.y;
  const int M = offsets[e + 1] - offsets[e];
  const int KB = I / kBlock;
  const int group = blockIdx.x * (blockDim.x / 16) + (threadIdx.x >> 4);
  const bool active = M > 0 && group < M * KB;
  const int local = active ? group / KB : 0;
  const int kb = active ? group - local * KB : 0;
  const int sub = threadIdx.x & 15;
  const int k = kb * kBlock + sub * 8;
  const int grow = active ? offsets[e] + local : 0;
  float v[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) v[i] = 0.f;
  if (active) {
    float g[8], u[8];
    const uint4 rg = *reinterpret_cast<const uint4*>(
        gate + static_cast<int64_t>(grow) * I + k);
    const uint4 ru = *reinterpret_cast<const uint4*>(
        up + static_cast<int64_t>(grow) * I + k);
    const __nv_bfloat16* bg = reinterpret_cast<const __nv_bfloat16*>(&rg);
    const __nv_bfloat16* bu = reinterpret_cast<const __nv_bfloat16*>(&ru);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      g[i] = __bfloat162float(bg[i]);
      u[i] = __bfloat162float(bu[i]);
      v[i] = g[i] / (1.f + __expf(-g[i])) * u[i];
    }
  }
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(v[i]));
#pragma unroll
  for (int o = 8; o > 0; o >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  if (!active) return;
  const float inv = amax > 0.f ? kFp8Max / amax : 1.f;
  uint32_t packed[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(
        make_float2(v[4 * h] * inv, v[4 * h + 1] * inv), __NV_SATFINITE, __NV_E4M3);
    const __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(
        make_float2(v[4 * h + 2] * inv, v[4 * h + 3] * inv), __NV_SATFINITE, __NV_E4M3);
    packed[h] = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
  }
  *reinterpret_cast<uint2*>(q + static_cast<int64_t>(grow) * I + k) =
      make_uint2(packed[0], packed[1]);
  if (sub == 0)
    sfa[sfa_off[e] + kb * M + local] = amax > 0.f ? amax / kFp8Max : 1.f;
}

__global__ void transpose_sfb_kernel(const float* __restrict__ src,
                                     float* __restrict__ dst, int NB, int KB) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= NB * KB) return;
  const int nb = i / KB;
  const int kb = i - nb * KB;
  dst[kb * NB + nb] = src[i];
}

__global__ void scatter_down_kernel(const __nv_bfloat16* __restrict__ down, int H,
                                    const int* __restrict__ offsets,
                                    const int* __restrict__ slot, int E,
                                    __nv_bfloat16* __restrict__ ybuf) {
  const int row = blockIdx.x;
  const int e = find_expert(offsets, E, row);
  const int local = row - offsets[e];
  const int id = slot[offsets[e] + local];
  for (int c = threadIdx.x; c < H; c += blockDim.x)
    ybuf[static_cast<int64_t>(id) * H + c] =
        down[static_cast<int64_t>(row) * H + c];
}

__global__ void combine_kernel(const __nv_bfloat16* __restrict__ ybuf,
                               const float* __restrict__ w, int K, int H,
                               float* __restrict__ out) {
  const int t = blockIdx.x;
  const int col = blockIdx.y * blockDim.x + threadIdx.x;
  if (col >= H) return;
  float s = 0.f;
  for (int k = 0; k < K; ++k) {
    const int64_t slot = static_cast<int64_t>(t) * K + k;
    s += __bfloat162float(ybuf[slot * H + col]) * w[slot];
  }
  out[static_cast<int64_t>(t) * H + col] = s;
}

void launch_grouped(int groups, const std::vector<cute::Shape<int, int, int>>& problems,
                    const std::vector<const ElementA*>& A,
                    const std::vector<const ElementB*>& B,
                    const std::vector<const float*>& SFA,
                    const std::vector<const float*>& SFB,
                    const std::vector<ElementD*>& D, const std::vector<StrideA>& sA,
                    const std::vector<StrideB>& sB, const std::vector<StrideC>& sC,
                    const std::vector<StrideD>& sD, const std::vector<LayoutSFA>& lA,
                    const std::vector<LayoutSFB>& lB, cudaStream_t stream) {
  auto dev = [&](auto& host) {
    using T = typename std::decay_t<decltype(host)>::value_type;
    auto t = torch::empty({static_cast<int64_t>(host.size() * sizeof(T))},
                          torch::dtype(torch::kUInt8).device(torch::kCUDA));
    cudaMemcpyAsync(t.data_ptr(), host.data(), host.size() * sizeof(T),
                    cudaMemcpyHostToDevice, stream);
    return t;
  };
  // Keep device storage alive until the launch returns.
  auto p_prob = dev(problems);
  auto p_A = dev(A);
  auto p_B = dev(B);
  auto p_SFA = dev(SFA);
  auto p_SFB = dev(SFB);
  auto p_D = dev(D);
  auto p_sA = dev(sA);
  auto p_sB = dev(sB);
  auto p_sC = dev(sC);
  auto p_sD = dev(sD);
  auto p_lA = dev(lA);
  auto p_lB = dev(lB);

  cutlass::KernelHardwareInfo hw;
  hw.device_id = at::cuda::current_device();
  hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw.device_id);

  typename Gemm::Arguments arguments{};
  arguments.mode = cutlass::gemm::GemmUniversalMode::kGrouped;
  arguments.problem_shape.num_groups = groups;
  arguments.problem_shape.problem_shapes =
      reinterpret_cast<cute::Shape<int, int, int>*>(p_prob.data_ptr());
  arguments.problem_shape.host_problem_shapes = problems.data();
  arguments.mainloop.ptr_A = reinterpret_cast<ElementA const**>(p_A.data_ptr());
  arguments.mainloop.dA = reinterpret_cast<decltype(arguments.mainloop.dA)>(p_sA.data_ptr());
  arguments.mainloop.ptr_B = reinterpret_cast<ElementB const**>(p_B.data_ptr());
  arguments.mainloop.dB = reinterpret_cast<decltype(arguments.mainloop.dB)>(p_sB.data_ptr());
  arguments.mainloop.ptr_SFA = reinterpret_cast<float const**>(p_SFA.data_ptr());
  arguments.mainloop.layout_SFA =
      reinterpret_cast<decltype(arguments.mainloop.layout_SFA)>(p_lA.data_ptr());
  arguments.mainloop.ptr_SFB = reinterpret_cast<float const**>(p_SFB.data_ptr());
  arguments.mainloop.layout_SFB =
      reinterpret_cast<decltype(arguments.mainloop.layout_SFB)>(p_lB.data_ptr());
  arguments.epilogue.thread.alpha = 1.f;
  arguments.epilogue.thread.beta = 0.f;
  arguments.epilogue.ptr_C = reinterpret_cast<ElementC const**>(p_D.data_ptr());
  arguments.epilogue.dC = reinterpret_cast<decltype(arguments.epilogue.dC)>(p_sC.data_ptr());
  arguments.epilogue.ptr_D = reinterpret_cast<ElementD**>(p_D.data_ptr());
  arguments.epilogue.dD = reinterpret_cast<decltype(arguments.epilogue.dD)>(p_sD.data_ptr());
  arguments.hw_info = hw;

  Gemm gemm;
  TORCH_CHECK(gemm.can_implement(arguments) == cutlass::Status::kSuccess,
              "fp8 grouped blockwise: can_implement failed");
  const size_t ws_bytes = Gemm::get_workspace_size(arguments);
  auto ws = torch::empty({static_cast<int64_t>(ws_bytes)},
                         torch::dtype(torch::kUInt8).device(torch::kCUDA));
  TORCH_CHECK(gemm.initialize(arguments, ws.data_ptr()) == cutlass::Status::kSuccess,
              "fp8 grouped blockwise: initialize failed");
  TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess,
              "fp8 grouped blockwise: launch failed");
  // The GEMM reads these pointers asynchronously.
  for (auto* t : {&p_prob, &p_A, &p_B, &p_SFA, &p_SFB, &p_D, &p_sA, &p_sB, &p_sC,
                  &p_sD, &p_lA, &p_lB, &ws}) {
    t->record_stream(at::cuda::getCurrentCUDAStream());
  }
}

}  // namespace fp8g

using namespace fp8g;

void fp8_grouped_large_sm120(const __nv_bfloat16* x, int T, int H, int I,
                             int topk, int E, const int* offsets,
                             const int* slot, const float* topk_w,
                             const int64_t* table, float* out,
                             cudaStream_t stream) {
  std::vector<int> hoff(E + 1);
  cudaMemcpyAsync(hoff.data(), offsets, (E + 1) * sizeof(int),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  const int rows = hoff[E];
  std::vector<Group> groups;
  groups.reserve(E);
  for (int e = 0; e < E; ++e) {
    const int M = hoff[e + 1] - hoff[e];
    if (M > 0) groups.push_back({e, M, hoff[e]});
  }
  TORCH_CHECK(!groups.empty(), "fp8 grouped: no routed tokens");
  const int G = static_cast<int>(groups.size());
  // Weight pointers and block scales are fixed for a loaded table. Rebuilding
  // them on every layer is 3*E tiny launches plus a host sync. The bench and
  // a resident expert cache hit this on the second call.
  struct ScaleCache {
    const void* table = nullptr;
    int E = 0;
    int H = 0;
    int I = 0;
    torch::Tensor sfb_g, sfb_u, sfb_d;
    std::vector<Expert6> experts;
  };
  static ScaleCache scale_cache;
  const bool scale_hit = scale_cache.table == table && scale_cache.E == E &&
                         scale_cache.H == H && scale_cache.I == I;

  auto packed = torch::empty({rows, H}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
  pack_rows_kernel<<<rows, 256, 0, stream>>>(
      x, H, topk, offsets, slot, E, rows,
      reinterpret_cast<__nv_bfloat16*>(packed.data_ptr()));

  const int KB_H = H / kBlock;
  const int KB_I = I / kBlock;
  const int NB_I = I / kBlock;
  const int NB_H = H / kBlock;
  std::vector<int> sfa_h_off(E + 1, 0), sfa_i_off(E + 1, 0);
  for (int e = 0; e < E; ++e) {
    const int M = hoff[e + 1] - hoff[e];
    sfa_h_off[e + 1] = sfa_h_off[e] + KB_H * M;
    sfa_i_off[e + 1] = sfa_i_off[e] + KB_I * M;
  }
  auto sfa_h_off_d = torch::tensor(sfa_h_off, torch::dtype(torch::kInt32).device(torch::kCUDA));
  auto sfa_i_off_d = torch::tensor(sfa_i_off, torch::dtype(torch::kInt32).device(torch::kCUDA));
  auto xq = torch::empty({rows, H}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
  auto sfa_h = torch::empty({sfa_h_off[E]}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  const int qblocks = std::max(1, ([&] {
    int m = 0;
    for (const Group& g : groups) m = std::max(m, g.M);
    return (m * KB_H + 15) / 16;
  })());
  quant_grouped_kernel<<<dim3(qblocks, E), 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(packed.data_ptr()), H, offsets, E, H,
      xq.data_ptr<uint8_t>(), sfa_h.data_ptr<float>(), sfa_h_off_d.data_ptr<int>());

  torch::Tensor sfb_g, sfb_u, sfb_d;
  const Expert6* experts = nullptr;
  if (scale_hit) {
    sfb_g = scale_cache.sfb_g;
    sfb_u = scale_cache.sfb_u;
    sfb_d = scale_cache.sfb_d;
    experts = scale_cache.experts.data();
  } else {
    std::vector<Expert6> hexperts(E);
    cudaMemcpyAsync(hexperts.data(), table, E * sizeof(Expert6),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    sfb_g = torch::empty({static_cast<int64_t>(E) * NB_I * KB_H},
                         torch::dtype(torch::kFloat32).device(torch::kCUDA));
    sfb_u = torch::empty_like(sfb_g);
    sfb_d = torch::empty({static_cast<int64_t>(E) * NB_H * KB_I},
                         torch::dtype(torch::kFloat32).device(torch::kCUDA));
    for (int e = 0; e < E; ++e) {
      transpose_sfb_kernel<<<(NB_I * KB_H + 255) / 256, 256, 0, stream>>>(
          hexperts[e].gate_s,
          sfb_g.data_ptr<float>() + static_cast<int64_t>(e) * NB_I * KB_H, NB_I,
          KB_H);
      transpose_sfb_kernel<<<(NB_I * KB_H + 255) / 256, 256, 0, stream>>>(
          hexperts[e].up_s,
          sfb_u.data_ptr<float>() + static_cast<int64_t>(e) * NB_I * KB_H, NB_I,
          KB_H);
      transpose_sfb_kernel<<<(NB_H * KB_I + 255) / 256, 256, 0, stream>>>(
          hexperts[e].down_s,
          sfb_d.data_ptr<float>() + static_cast<int64_t>(e) * NB_H * KB_I, NB_H,
          KB_I);
    }
    scale_cache.table = table;
    scale_cache.E = E;
    scale_cache.H = H;
    scale_cache.I = I;
    scale_cache.sfb_g = sfb_g;
    scale_cache.sfb_u = sfb_u;
    scale_cache.sfb_d = sfb_d;
    scale_cache.experts = std::move(hexperts);
    experts = scale_cache.experts.data();
  }

  auto gate_o = torch::empty({rows, I}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
  auto up_o = torch::empty({rows, I}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));

  const int GG = G * 2;
  std::vector<cute::Shape<int, int, int>> problems(GG);
  std::vector<const ElementA*> A(GG);
  std::vector<const ElementB*> B(GG);
  std::vector<const float*> SFA(GG), SFB(GG);
  std::vector<ElementD*> D(GG);
  std::vector<StrideA> sA(GG);
  std::vector<StrideB> sB(GG);
  std::vector<StrideC> sC(GG);
  std::vector<StrideD> sD(GG);
  std::vector<LayoutSFA> lA(GG);
  std::vector<LayoutSFB> lB(GG);
  auto* xq_p = reinterpret_cast<const ElementA*>(xq.data_ptr());
  auto* go = reinterpret_cast<ElementD*>(gate_o.data_ptr());
  auto* uo = reinterpret_cast<ElementD*>(up_o.data_ptr());
  for (int i = 0; i < G; ++i) {
    const Group g = groups[i];
    for (int half = 0; half < 2; ++half) {
      const int dst = half == 0 ? i : G + i;
      problems[dst] = cute::make_shape(g.M, I, H);
      sA[dst] = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(g.M, H, 1));
      sB[dst] = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(I, H, 1));
      sC[dst] = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(g.M, I, 1));
      sD[dst] = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(g.M, I, 1));
      lA[dst] = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(g.M, I, H, 1));
      lB[dst] = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(g.M, I, H, 1));
      A[dst] = xq_p + static_cast<int64_t>(g.row0) * H;
      B[dst] = reinterpret_cast<const ElementB*>(half == 0 ? experts[g.expert].gate
                                                           : experts[g.expert].up);
      SFA[dst] = sfa_h.data_ptr<float>() + sfa_h_off[g.expert];
      SFB[dst] = (half == 0 ? sfb_g : sfb_u).data_ptr<float>() +
                 static_cast<int64_t>(g.expert) * NB_I * KB_H;
      D[dst] = (half == 0 ? go : uo) + static_cast<int64_t>(g.row0) * I;
    }
  }
  launch_grouped(GG, problems, A, B, SFA, SFB, D, sA, sB, sC, sD, lA, lB, stream);

  auto hq = torch::empty({rows, I}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
  auto sfa_i = torch::empty({sfa_i_off[E]}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  const int qblocks_i = std::max(1, ([&] {
    int m = 0;
    for (const Group& g : groups) m = std::max(m, g.M);
    return (m * KB_I + 15) / 16;
  })());
  silu_quant_kernel<<<dim3(qblocks_i, E), 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(gate_o.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(up_o.data_ptr()), I, offsets, E,
      hq.data_ptr<uint8_t>(), sfa_i.data_ptr<float>(), sfa_i_off_d.data_ptr<int>());

  auto down_o = torch::empty({rows, H}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
  problems.resize(G);
  A.resize(G);
  B.resize(G);
  SFA.resize(G);
  SFB.resize(G);
  D.resize(G);
  sA.resize(G);
  sB.resize(G);
  sC.resize(G);
  sD.resize(G);
  lA.resize(G);
  lB.resize(G);
  auto* hq_p = reinterpret_cast<const ElementA*>(hq.data_ptr());
  auto* dnp = reinterpret_cast<ElementD*>(down_o.data_ptr());
  for (int i = 0; i < G; ++i) {
    const Group g = groups[i];
    problems[i] = cute::make_shape(g.M, H, I);
    sA[i] = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(g.M, I, 1));
    sB[i] = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(H, I, 1));
    sC[i] = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(g.M, H, 1));
    sD[i] = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(g.M, H, 1));
    lA[i] = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(g.M, H, I, 1));
    lB[i] = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(g.M, H, I, 1));
    A[i] = hq_p + static_cast<int64_t>(g.row0) * I;
    B[i] = reinterpret_cast<const ElementB*>(experts[g.expert].down);
    SFA[i] = sfa_i.data_ptr<float>() + sfa_i_off[g.expert];
    SFB[i] = sfb_d.data_ptr<float>() + static_cast<int64_t>(g.expert) * NB_H * KB_I;
    D[i] = dnp + static_cast<int64_t>(g.row0) * H;
  }
  launch_grouped(G, problems, A, B, SFA, SFB, D, sA, sB, sC, sD, lA, lB, stream);

  auto ybuf = torch::empty({static_cast<int64_t>(T) * topk, H},
                           torch::dtype(torch::kBFloat16).device(torch::kCUDA));
  scatter_down_kernel<<<rows, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(down_o.data_ptr()), H, offsets, slot, E,
      reinterpret_cast<__nv_bfloat16*>(ybuf.data_ptr()));
  combine_kernel<<<dim3(T, (H + 127) / 128), 128, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(ybuf.data_ptr()), topk_w, topk, H, out);
  C10_CUDA_CHECK(cudaGetLastError());
}
