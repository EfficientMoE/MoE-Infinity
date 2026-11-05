"""Kernel cases for the sm_120 audit, built from real model shapes.

Each Case builds its tensors lazily and returns a zero-arg launcher that
issues the same kernels MoE-Infinity issues for that op on sm_120
(torch 2.11 / cu130, no flash_attn / flashinfer / fla installed).

Impl names:
  production paths   cutlass_fused, cutlass_fused_act, grouped_prefill, fp8_native, fp8_prod,
                     gptoss_resident, gptoss_offload, triton_mxfp4,
                     v4_fp4_prod, v4_fp4_native, sdpa_shim, hf_eager, paged_fwd,
                     paged_fwd_int8, hf_eager_decode, router_*, router_fused,
                     gdn_torch,
                     kv_write_loop, kv_write_vec, moe_gqa, moe_gqa_swa,
                     moe_gqa_masked, moe_gqa_decode, moe_gqa_masked_decode
                     (HF models without a paged shim, runtime/hf_attention.py)
  previous paths     fp8_prod (per-call FP8 -> BF16 dequant + CUTLASS; still
                     the production path off sm_120 / RTX PRO 6000)
  FP8 reference      fp8_scaled_mm_ref (cuBLASLt torch._scaled_mm, Rule B)
  (paged_v1 / sdpa_fallback rows in results/ are the pre-#298/#299 kernels)
  references         cublas_ref, sdpa_gqa, sdpa_flash_decode, triton_decode,
                     read_ceiling, kv_copy_ref,
                     gptoss_offload_legacy (pre-#311 type-6 path)
"""

from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass, field
from typing import Callable

import torch

from benchmarks.kernel_audit.models import (
    DECODE_BATCHES,
    DECODE_CONTEXTS,
    MODELS,
    PREFILL_TOKENS,
    MoEModel,
    active_experts,
)

BF16 = 2
EXEC_STREAMS = 4  # ArcherConfig.num_threads default: exec workers per GPU


@dataclass
class Case:
    cid: str
    suite: str
    model: str
    phase: str  # prefill | decode
    size: str  # 1K..200K | b1..b256 | b1@200K ...
    impl: str
    flops: float  # useful FLOPs of one invocation
    bytes_min: float  # compulsory DRAM bytes of one invocation (cold cache)
    build: Callable[[], Callable[[], None]]
    per_layer: float = 1.0  # invocations of this op per layer
    layers: int = 1  # layers in the model that run this op
    meta: dict = field(default_factory=dict)
    skip: str | None = None  # reason the case cannot run (e.g. OOM)
    iters: int | None = None  # override timing iterations (slow paths)


def _rand(*shape, dtype=torch.bfloat16, scale=0.02):
    return (torch.randn(*shape, device="cuda") * scale).to(dtype)


def _free_bytes() -> int:
    free, _ = torch.cuda.mem_get_info()
    return free


# --------------------------------------------------------------------------
# Routed-expert FFN
# --------------------------------------------------------------------------


def expert_tokens(m: MoEModel, n: int) -> tuple[int, float]:
    """(tokens per active expert M, active experts A) for n routed tokens."""
    a = active_experts(m, n)
    return max(1, round(n * m.top_k / a)), a


def _cutlass_builder(M: int, H: int, I: int):
    def build():
        from benchmarks.kernel_audit.ext import load

        ext = load()
        x = _rand(M, H, scale=1.0)
        g, u, d = _rand(I, H), _rand(I, H), _rand(H, I)
        gb = torch.empty(M, I, device="cuda", dtype=torch.bfloat16)
        fb = torch.empty_like(gb)
        out = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)
        return lambda: ext.fused_moe_ffn_into(x, g, u, d, gb, fb, out)

    return build


# Routed-expert gated activation of the checkpoint (moe_infinity.runtime.
# expert_activation kinds: 1 = clamped_silu, 2 = swiglu_oai); absent = silu.
_EXPERT_ACT = {
    "glm-5.3-flash": (1, 1.0, 10.0),
    "minimax-m3": (2, 1.702, 7.0),
}


def _cutlass_act_builder(act: tuple[int, float, float]):
    """CUTLASS path with the checkpoint's activation in the epilogue."""

    def builder(M: int, H: int, I: int):
        def build():
            from benchmarks.kernel_audit.ext import load

            ext = load()
            x = _rand(M, H, scale=1.0)
            g, u, d = _rand(I, H), _rand(I, H), _rand(H, I)
            gb = torch.empty(M, I, device="cuda", dtype=torch.bfloat16)
            fb = torch.empty_like(gb)
            out = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)
            return lambda: ext.fused_moe_ffn_act_into(
                x, g, u, d, gb, fb, out, *act
            )

        return build

    return builder


_MOE_MLP_MODULES: dict = {}


def _moemlp_native_builder(M: int, H: int, I: int):
    """Full C++ MoEMLP::forward (staging copies + CUTLASS FFN + output clone),
    chunked in kMaxTokens rows for M > 8192."""

    def build():
        from moe_infinity import _store

        # Staging buffers are sized on first use and never freed, so reuse
        # one module per expert shape.
        mlp = _MOE_MLP_MODULES.get((H, I))
        if mlp is None:
            mlp = _MOE_MLP_MODULES[(H, I)] = _store._moe_mlp_for_test(0, 5)
        x = _rand(M, H, scale=1.0)
        params = [_rand(I, H), _rand(I, H), _rand(H, I)]
        return lambda: mlp.forward(x, params)

    return build


def _cublas_ref_builder(M: int, H: int, I: int):
    """Same math as the CUTLASS path through cuBLAS with gate|up concatenated."""

    def build():
        x = _rand(M, H, scale=1.0)
        gu = _rand(2 * I, H)
        d = _rand(H, I)

        def run():
            h = x @ gu.T
            a = torch.nn.functional.silu(h[:, :I]) * h[:, I:]
            return a @ d.T

        return run

    return build


def _fp8_prod_builder(M: int, H: int, I: int):
    """Native path for FP8 stores: per-execution block dequant + CUTLASS."""

    def build():
        from benchmarks.kernel_audit.ext import load

        ext = load()
        x = _rand(M, H, scale=1.0)
        shapes = ((I, H), (I, H), (H, I))
        w = [
            torch.randint(0, 120, s, device="cuda", dtype=torch.uint8)
            for s in shapes
        ]
        sc = [
            torch.rand(-(-s[0] // 128), -(-s[1] // 128), device="cuda")
            for s in shapes
        ]
        gb = torch.empty(M, I, device="cuda", dtype=torch.bfloat16)
        fb = torch.empty_like(gb)
        out = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)

        def run():
            deq = []
            for wi, si in zip(w, sc):
                o = torch.empty(wi.shape, device="cuda", dtype=torch.bfloat16)
                ext.fp8_dequant_blockwise(wi, si, o)
                deq.append(o)
            ext.fused_moe_ffn_into(x, deq[0], deq[1], deq[2], gb, fb, out)

        return run

    return build


def _fp8_native_builder(M: int, H: int, I: int):
    """Native FP8 experts (#310): block-scaled GEMV (M <= 8) or CUTLASS
    sm_120 blockwise FP8 GEMM, reading the e4m3 weights once. Gated to
    sm_120 + RTX PRO 6000 (other GPUs run fp8_prod in production)."""

    def build():
        from benchmarks.kernel_audit.ext import load

        ext = load()
        if not ext.fp8_moe_sm120_pro6000_supported(H, I, -1):
            raise RuntimeError(
                "fp8_native runs only on sm_120 + RTX PRO 6000 builds/devices"
            )
        x = _rand(M, H, scale=1.0)
        shapes = ((I, H), (I, H), (H, I))
        w = [
            torch.randint(0, 120, s, device="cuda", dtype=torch.uint8)
            for s in shapes
        ]
        sc = [
            torch.rand(-(-s[0] // 128), -(-s[1] // 128), device="cuda")
            for s in shapes
        ]
        out = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)
        return lambda: ext.fused_fp8_moe_ffn_sm120_pro6000_into(
            x, w[0], sc[0], w[1], sc[1], w[2], sc[2], out
        )

    return build


def _scaled_mm_ref_builder(M: int, H: int, I: int):
    """Rule-B reference for fp8_native: the same W8A8 expert through cuBLASLt
    (torch._scaled_mm, per-tensor scales, gate|up concatenated) with eager
    SiLU*up and e4m3 casts in between."""

    def build():
        f8 = torch.float8_e4m3fn
        x = _rand(M, H, scale=1.0)
        gu = torch.randint(0, 120, (2 * I, H), device="cuda", dtype=torch.uint8)
        d = torch.randint(0, 120, (H, I), device="cuda", dtype=torch.uint8)
        gu, d = gu.view(f8), d.view(f8)
        one = torch.ones((), device="cuda")

        def run():
            h = torch._scaled_mm(
                x.to(f8), gu.T, one, one, out_dtype=torch.bfloat16
            )
            a = (torch.nn.functional.silu(h[:, :I]) * h[:, I:]).to(f8)
            return torch._scaled_mm(a, d.T, one, one, out_dtype=torch.bfloat16)

        return run

    return build


def _mxfp4_weights(H: int, I: int):
    gu_packed = torch.randint(
        0, 255, (2 * I, H // 2), device="cuda", dtype=torch.uint8
    )
    gu_scale = torch.randint(
        118, 124, (2 * I, H // 32), device="cuda", dtype=torch.uint8
    )
    d_packed = torch.randint(
        0, 255, (H, I // 2), device="cuda", dtype=torch.uint8
    )
    d_scale = torch.randint(
        118, 124, (H, I // 32), device="cuda", dtype=torch.uint8
    )
    return gu_packed, gu_scale, _rand(2 * I), d_packed, d_scale, _rand(H)


def _gptoss_swiglu(gate, up, limit=7.0, alpha=1.702):
    gate = gate.clamp(max=limit)
    up = up.clamp(-limit, limit)
    return (up + 1) * (gate * torch.sigmoid(gate * alpha))


def _gptoss_legacy_builder(M: int, H: int, I: int, dequant: bool = True):
    """Pre-#311 type-6 path: per-call MXFP4 dequant + cuBLAS + elementwise."""

    def build():
        from benchmarks.kernel_audit.ext import load

        ext = load()
        x = _rand(M, H, scale=1.0)
        gu_packed, gu_scale, gu_bias, d_packed, d_scale, d_bias = (
            _mxfp4_weights(H, I)
        )
        gu_w = torch.empty(2 * I, H, device="cuda", dtype=torch.bfloat16)
        d_w = torch.empty(H, I, device="cuda", dtype=torch.bfloat16)
        ext.mxfp4_dequant(gu_packed, gu_scale, gu_w)
        ext.mxfp4_dequant(d_packed, d_scale, d_w)
        out = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)

        def run():
            if dequant:
                gw = torch.empty(2 * I, H, device="cuda", dtype=torch.bfloat16)
                dw = torch.empty(H, I, device="cuda", dtype=torch.bfloat16)
                ext.mxfp4_dequant(gu_packed, gu_scale, gw)
                ext.mxfp4_dequant(d_packed, d_scale, dw)
            else:
                gw, dw = gu_w, d_w
            gate_up = torch.matmul(x, gw.transpose(0, 1)) + gu_bias
            act = _gptoss_swiglu(gate_up[:, 0::2], gate_up[:, 1::2])
            out.copy_(torch.matmul(act, dw.transpose(0, 1)) + d_bias)

        return run

    return build


def _gptoss_offload_builder(M: int, H: int, I: int):
    """C++ type-6 path (device_memory_ratio < 0.9): MoEMLP::ForwardHelper ->
    gptoss_mxfp4_expert_forward_pro6000 (fused W4A16 MXFP4 GEMMs, scratch reused
    across calls like an expert executor)."""

    def build():
        from benchmarks.kernel_audit.ext import load

        ext = load()
        x = _rand(M, H, scale=1.0)
        gu_packed, gu_scale, gu_bias, d_packed, d_scale, d_bias = (
            _mxfp4_weights(H, I)
        )
        out = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)
        # Per-executor scratch, sized like gptoss_mxfp4_expert_forward_pro6000
        # allocates it (BF16 Wd, act and stream-K partials in one arena).
        arena = torch.empty(
            ext.gptoss_expert_arena_bytes(M, H, I),
            device="cuda",
            dtype=torch.uint8,
        )
        counters = torch.zeros(
            2 * ext.mxfp4_stream_k_counters, device="cuda", dtype=torch.int32
        )

        def run():
            ext.gptoss_mxfp4_expert_pro6000(
                x,
                gu_packed,
                gu_scale,
                gu_bias,
                d_packed,
                d_scale,
                d_bias,
                out,
                arena,
                counters,
            )

        return run

    return build


def _gptoss_resident_builder(M: int, H: int, I: int):
    """Resident default (device_memory_ratio >= 0.9): Triton fused MXFP4 GEMM x2."""

    def build():
        from moe_infinity.kernel.mxfp4_gemm import fused_mxfp4_gemm

        x = _rand(M, H, scale=1.0)
        gu_packed, gu_scale, gu_bias, d_packed, d_scale, d_bias = (
            _mxfp4_weights(H, I)
        )

        def run():
            gu = fused_mxfp4_gemm(x, gu_packed, gu_scale, gu_bias)
            act = _gptoss_swiglu(gu[..., ::2], gu[..., 1::2])
            return fused_mxfp4_gemm(
                act.to(torch.bfloat16), d_packed, d_scale, d_bias
            )

        return run

    return build


def _v4_fp4_builder(M: int, H: int, I: int):
    """DeepSeek-V4 Path B: _v4_fp4.v4_expert_forward (dequant x3 + cuBLAS)."""

    def build():
        import moe_infinity._v4_fp4 as v4

        x = _rand(M, H, scale=1.0)
        w1 = torch.randint(
            0, 255, (I, H // 2), device="cuda", dtype=torch.uint8
        )
        w3 = torch.randint(
            0, 255, (I, H // 2), device="cuda", dtype=torch.uint8
        )
        w2 = torch.randint(
            0, 255, (H, I // 2), device="cuda", dtype=torch.uint8
        )
        s1 = torch.randint(
            118, 124, (I, H // 32), device="cuda", dtype=torch.uint8
        )
        s3 = torch.randint(
            118, 124, (I, H // 32), device="cuda", dtype=torch.uint8
        )
        s2 = torch.randint(
            118, 124, (H, I // 32), device="cuda", dtype=torch.uint8
        )
        return lambda: v4.v4_expert_forward(x, w1, s1, w2, s2, w3, s3, 10.0)

    return build


def _v4_fp4_expert_weights(H: int, I: int) -> dict:
    """Random packed e2m1 weights + ue8m0 scales of one V4 routed expert."""
    w = {}
    for name, (n, k) in (("w1", (I, H)), ("w3", (I, H)), ("w2", (H, I))):
        w[name] = torch.randint(
            0, 255, (n, k // 2), device="cuda", dtype=torch.uint8
        )
        w["s" + name[1]] = torch.randint(
            118, 124, (n, k // 32), device="cuda", dtype=torch.uint8
        )
    return w


def _v4_fp4_native_builder(M: int, H: int, I: int):
    """DeepSeek-V4 native W4A8 block-scaled path, one expert per call."""

    def build():
        import moe_infinity._v4_fp4 as v4

        x = _rand(M, H, scale=1.0)
        w = _v4_fp4_expert_weights(H, I)
        args = [w[k] for k in ("w1", "s1", "w2", "s2", "w3", "s3")]
        return lambda: v4.v4_expert_forward_fp4_sm120_pro6000(x, *args, 10.0)

    return build


def _v4_fp4_native_layer_builder(T: int, n_exp: int, H: int, I: int, k: int):
    """One MoE layer on the native path: a single grouped call for T tokens
    routed top-k over n_exp distinct (cold) experts, routing on device."""

    def build():
        import moe_infinity._v4_fp4 as v4

        experts = [_v4_fp4_expert_weights(H, I) for _ in range(n_exp)]
        table = torch.tensor(
            [
                [e[n].data_ptr() for n in ("w1", "s1", "w3", "s3", "w2", "s2")]
                for e in experts
            ],
            dtype=torch.int64,
            device="cuda",
        )
        # balanced routing: slot j of all tokens goes to expert j % n_exp,
        # so every expert receives ~T*k/n_exp rows (the case's M)
        ids = (torch.arange(T * k, device="cuda") % n_exp).view(T, k)
        ids = ids.to(torch.int32)
        wts = torch.rand(T, k, device="cuda")
        x = _rand(T, H, scale=1.0)

        def run():
            v4.v4_moe_forward_sm120_pro6000(x, ids, wts, table, I, 10.0)

        run.experts = experts  # keep the weights alive
        return run

    return build


# weight bytes per element of one expert in each storage format
_W_BYTES = {
    "cutlass_fused": BF16,
    "cutlass_fused_act": BF16,
    "moemlp_native": BF16,
    "cublas_ref": BF16,
    "gptoss_bf16cached": BF16,
    "fp8_prod": 1 + 4 / (128 * 128),
    "fp8_native": 1 + 4 / (128 * 128),
    "fp8_scaled_mm_ref": 1,
    "gptoss_resident": 0.5 + 1 / 32,
    "gptoss_offload": 0.5 + 1 / 32,
    "gptoss_offload_legacy": 0.5 + 1 / 32,
    "v4_fp4_prod": 0.5 + 1 / 32,
    "v4_fp4_native": 0.5 + 1 / 32,
}


def _expert_impls(m: MoEModel):
    if m.expert_type == 6:
        return [
            ("gptoss_resident", _gptoss_resident_builder),
            ("gptoss_offload", _gptoss_offload_builder),
            ("gptoss_offload_legacy", _gptoss_legacy_builder),
            (
                "gptoss_bf16cached",
                lambda M, H, I: _gptoss_legacy_builder(M, H, I, False),
            ),
            ("cublas_ref", _cublas_ref_builder),
        ]
    impls = []
    if m.expert_dtype.startswith("fp8"):
        impls.append(("fp8_native", _fp8_native_builder))
        impls.append(("fp8_scaled_mm_ref", _scaled_mm_ref_builder))
        impls.append(("fp8_prod", _fp8_prod_builder))
    if m.expert_dtype.startswith("fp4"):
        impls.append(("v4_fp4_prod", _v4_fp4_builder))
        impls.append(("v4_fp4_native", _v4_fp4_native_builder))
    impls.append(("cutlass_fused", _cutlass_builder))
    if m.key in _EXPERT_ACT:
        impls.append(
            ("cutlass_fused_act", _cutlass_act_builder(_EXPERT_ACT[m.key]))
        )
    impls += [
        ("moemlp_native", _moemlp_native_builder),
        ("cublas_ref", _cublas_ref_builder),
    ]
    return impls


def _expert_points(m: MoEModel):
    for lbl, n in PREFILL_TOKENS.items():
        yield "prefill", lbl, n
    for b in DECODE_BATCHES:
        yield "decode", f"b{b}", b


def expert_cases(model_keys: list[str] | None = None) -> list[Case]:
    cases: list[Case] = []
    for key, m in MODELS.items():
        if model_keys and key not in model_keys:
            continue
        H, I = m.hidden, m.inter
        w_elems = 3 * H * I
        for phase, lbl, n in _expert_points(m):
            M, A = expert_tokens(m, n)
            meta = {"M": M, "H": H, "I": I, "active_experts": A, "tokens": n}
            if M > 8192:
                meta["exceeds_kMaxTokens"] = True
            for impl, builder in _expert_impls(m):
                cases.append(
                    Case(
                        f"{key}|{lbl}|{impl}",
                        suite="expert",
                        model=key,
                        phase=phase,
                        size=lbl,
                        impl=impl,
                        flops=2.0 * M * w_elems,
                        bytes_min=w_elems * _W_BYTES[impl] + 2 * M * H * BF16,
                        build=builder(M, H, I),
                        per_layer=A,
                        layers=m.moe_layers,
                        meta=dict(meta),
                    )
                )
    return cases


def _layer_builder(case: Case, n_exec: int, expert_bytes: float, streams: int):
    """One MoE layer: n_exec expert invocations over distinct weights.

    Native-engine impls are spread round-robin over EXEC_STREAMS streams like
    ExpertDispatcher's exec workers; Python-loop impls run on one stream.
    A pool of distinct weight sets (>= 4x L2) keeps every expert's weights
    cold, as in a real forward pass.
    """

    def build():
        l2 = torch.cuda.get_device_properties(0).L2_cache_size
        if case.phase == "decode":
            # Concurrent streams could share a recycled expert through L2, so
            # every decode invocation gets its own weights (activations are tiny).
            pool = n_exec
        else:
            pool = max(1, min(n_exec, math.ceil(4 * l2 / expert_bytes)))
        fns = [case.build() for _ in range(pool)]
        ss = [torch.cuda.Stream() for _ in range(streams)]
        main = torch.cuda.current_stream()

        def run():
            if streams == 1:
                for i in range(n_exec):
                    fns[i % pool]()
                return
            for s in ss:
                s.wait_stream(main)
            for i in range(n_exec):
                with torch.cuda.stream(ss[i % streams]):
                    fns[i % pool]()
            for s in ss:
                main.wait_stream(s)

        return run

    return build


_SEQUENTIAL_IMPLS = {"gptoss_resident", "v4_fp4_prod"}


def _layer_graph_builder(
    case: Case, n_exec: int, expert_bytes: float, streams: int, reps: int = 4
):
    """Device-only layer time: `reps` layers captured in one CUDA graph.

    Each rep uses a fresh slice of a pool of distinct experts sized so the
    weights touched by consecutive reps exceed L2 (no cross-rep L2 reuse).
    Returns a launcher whose single call replays all reps; per_layer = 1/reps.
    """

    def build():
        l2 = torch.cuda.get_device_properties(0).L2_cache_size
        pool = max(n_exec * 2, math.ceil(4 * l2 / expert_bytes))
        fns = [case.build() for _ in range(pool)]
        ss = [torch.cuda.Stream() for _ in range(streams)]
        cap = torch.cuda.Stream()

        def body():
            main = torch.cuda.current_stream()
            idx = 0
            for _ in range(reps):
                for s in ss:
                    s.wait_stream(main)
                for i in range(n_exec):
                    with torch.cuda.stream(ss[i % streams]):
                        fns[idx % pool]()
                    idx += 1
                for s in ss:
                    main.wait_stream(s)

        cap.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(cap):
            body()  # warmup (allocator, cuBLAS handles, CUTLASS attributes)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=cap):
            body()
        torch.cuda.synchronize()
        return g.replay

    return build


def _bmm_layer_builder(n: int, M: int, H: int, I: int):
    """Grouped-GEMM proxy: all n active experts in two cuBLAS batched GEMMs."""

    def build():
        x = _rand(n, M, H, scale=1.0)
        wgu = _rand(n, H, 2 * I)
        wd = _rand(n, I, H)

        def run():
            h = torch.bmm(x, wgu)
            a = torch.nn.functional.silu(h[..., :I]) * h[..., I:]
            return torch.bmm(a, wd)

        return run

    return build


def expert_grouped_cases(model_keys: list[str] | None = None) -> list[Case]:
    out = []
    for c in expert_cases(model_keys):
        if c.impl != "cublas_ref":
            continue
        n = max(1, round(c.per_layer))
        M, H, I = c.meta["M"], c.meta["H"], c.meta["I"]
        mem = n * (3 * H * I + M * (2 * H + 3 * I)) * BF16
        skip = f"needs {mem / 1e9:.0f} GB" if mem > 60e9 else None
        out.append(
            Case(
                cid=c.cid.replace("|cublas_ref", "|bmm_grouped").replace(
                    "|", "|layer|", 1
                ),
                suite="expert_grouped",
                model=c.model,
                phase=c.phase,
                size=c.size,
                impl="bmm_grouped",
                flops=c.flops * n,
                bytes_min=c.bytes_min * n,
                build=_bmm_layer_builder(n, M, H, I),
                layers=c.layers,
                meta={**c.meta, "n_exec": n},
                skip=skip,
                iters=5,
            )
        )
    return out


def expert_layer_graph_cases(model_keys: list[str] | None = None) -> list[Case]:
    out = []
    reps = 4
    for c in expert_cases(model_keys):
        if c.phase != "decode" or c.impl not in (
            "cutlass_fused",
            "cublas_ref",
            "gptoss_offload",
        ):
            continue
        n_exec = max(1, round(c.per_layer))
        H, I = c.meta["H"], c.meta["I"]
        eb = 3 * H * I * _W_BYTES[c.impl]
        out.append(
            Case(
                cid=c.cid.replace("|", "|graph|", 1),
                suite="expert_layer_graph",
                model=c.model,
                phase=c.phase,
                size=c.size,
                impl=c.impl,
                flops=c.flops * n_exec * reps,
                bytes_min=c.bytes_min * n_exec * reps,
                build=_layer_graph_builder(c, n_exec, eb, EXEC_STREAMS, reps),
                per_layer=1.0 / reps,
                layers=c.layers,
                meta={
                    **c.meta,
                    "n_exec": n_exec,
                    "streams": EXEC_STREAMS,
                    "reps": reps,
                },
                iters=5,
            )
        )
    return out


def _grouped_prefill_builder(
    n: int, M: int, T: int, E: int, H: int, I: int, ext=None
):
    """Grouped prefill path (#307): every expert of the layer in 3 launches.

    n experts with their own weight tensors (as in the expert cache), each
    routed exactly M of the T tokens, through moe_infinity._store
    .grouped_moe_prefill (same descriptor interface as grouped_moe_decode,
    #306) with the zero-init of the fp32 output folded in.
    """

    def build():
        import moe_infinity._store as store

        experts = [(_rand(I, H), _rand(I, H), _rand(H, I)) for _ in range(n)]
        mask = torch.zeros(T, E, dtype=torch.bool, device="cuda")
        for j in range(n):
            toks = (torch.arange(M, device="cuda") + j * M) % T
            mask[toks, j] = True
        weights = torch.rand(T, E, device="cuda") * mask
        kcap = int(mask.sum(1).max())
        table = torch.tensor(
            [[t.data_ptr() for t in w] + [j] for j, w in enumerate(experts)],
            dtype=torch.int64,
            device="cuda",
        )
        x = _rand(T, H, scale=1.0)
        out = torch.empty(T, H, device="cuda")

        def run():
            store.grouped_moe_prefill(
                x, mask, weights, table, I, kcap, out, zero_out=True
            )

        run.keep = experts  # weights must outlive the pointer table
        return run

    return build


def _grouped_prefill_cases(c: Case, n_exec: int) -> list[Case]:
    if c.phase != "prefill" or c.impl != "cutlass_fused":
        return []
    m = MODELS[c.model]
    H, I, M, T = c.meta["H"], c.meta["I"], c.meta["M"], c.meta["tokens"]
    mem = n_exec * 3 * H * I * BF16 + T * max(m.experts, n_exec)
    return [
        Case(
            cid=c.cid.replace("|", "|layer|", 1).replace(
                "|cutlass_fused", "|grouped_prefill"
            ),
            suite="expert_layer",
            model=c.model,
            phase=c.phase,
            size=c.size,
            impl="grouped_prefill",
            flops=c.flops * n_exec,
            bytes_min=c.bytes_min * n_exec,
            build=_grouped_prefill_builder(
                n_exec, M, T, max(m.experts, n_exec), H, I
            ),
            per_layer=1.0,
            layers=c.layers,
            meta={**c.meta, "n_exec": n_exec, "streams": 1},
            skip=f"needs {mem / 1e9:.0f} GB" if mem > 60e9 else None,
            iters=5,
        )
    ]


def expert_layer_cases(model_keys: list[str] | None = None) -> list[Case]:
    out = []
    for c in expert_cases(model_keys):
        if c.impl == "gptoss_bf16cached":
            continue
        out += _grouped_prefill_cases(c, max(1, round(c.per_layer)))
        n_exec = max(1, round(c.per_layer))
        H, I = c.meta["H"], c.meta["I"]
        eb = 3 * H * I * _W_BYTES[c.impl]
        streams = 1 if c.impl in _SEQUENTIAL_IMPLS else EXEC_STREAMS
        if c.impl == "v4_fp4_native":  # one grouped launch per layer
            m = MODELS[c.model]
            build = _v4_fp4_native_layer_builder(
                c.meta["tokens"], n_exec, H, I, m.top_k
            )
            streams = 1
        else:
            build = _layer_builder(c, n_exec, eb, streams)
        out.append(
            Case(
                cid=c.cid.replace("|", "|layer|", 1),
                suite="expert_layer",
                model=c.model,
                phase=c.phase,
                size=c.size,
                impl=c.impl,
                flops=c.flops * n_exec,
                bytes_min=c.bytes_min * n_exec,
                build=build,
                per_layer=1.0,
                layers=c.layers,
                meta={**c.meta, "n_exec": n_exec, "streams": streams},
                iters=5,
            )
        )
    return out


# --------------------------------------------------------------------------
# Attention prefill
# --------------------------------------------------------------------------

_SHIM_MODELS = {
    "qwen3-30b-a3b",
    "qwen3-235b-a22b",
    "qwen3.5-35b-a3b",
    "qwen3.5-122b-a10b",
    "qwen3.5-397b-a17b",
}
_EAGER_MODELS = {"gpt-oss-120b", "minimax-m3", "mixtral-8x7b", "glm-5.3"}


def _sdpa_shim_builder(T, H, KVH, d):
    """PagedAttentionBackend._prefill_forward without FlashInfer."""

    def build():
        from moe_infinity.runtime.attention_backend import causal_prefill_sdpa

        query = torch.randn(T, H, d, device="cuda", dtype=torch.bfloat16)
        key = torch.randn(T, KVH, d, device="cuda", dtype=torch.bfloat16)
        value = torch.randn(T, KVH, d, device="cuda", dtype=torch.bfloat16)

        def run():
            q = query.transpose(0, 1).unsqueeze(0)
            k = key.transpose(0, 1).unsqueeze(0)
            v = value.transpose(0, 1).unsqueeze(0)
            out = causal_prefill_sdpa(q, k, v, d**-0.5)
            return out.squeeze(0).transpose(0, 1)

        return run

    return build


def _sdpa_gqa_builder(T, H, KVH, d):
    """Reference: SDPA FA2 with native GQA (no K/V expansion)."""

    def build():
        q = torch.randn(1, H, T, d, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, KVH, T, d, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, KVH, T, d, device="cuda", dtype=torch.bfloat16)
        F = torch.nn.functional
        return lambda: F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=True
        )

    return build


def _repeat_kv(x, n):
    if n == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n, s, d).reshape(b, h * n, s, d)


def _eager_attn(q, k, v, mask, scale, sinks=None):
    """HF eager_attention_forward (Llama-style; GPT-OSS variant with sinks)."""
    n = q.shape[1] // k.shape[1]
    ks, vs = _repeat_kv(k, n), _repeat_kv(v, n)
    w = torch.matmul(q, ks.transpose(2, 3)) * scale
    if mask is not None:
        w = w + mask
    if sinks is not None:
        s = sinks.reshape(1, -1, 1, 1).expand(q.shape[0], -1, q.shape[-2], -1)
        c = torch.cat([w, s], dim=-1)
        c = c - c.max(dim=-1, keepdim=True).values
        p = torch.softmax(c, dim=-1, dtype=c.dtype)[..., :-1].to(vs.dtype)
    else:
        p = torch.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(p, vs).transpose(1, 2).contiguous()


def _hf_eager_prefill_builder(T, H, KVH, dqk, dv, sinks):
    def build():
        q = torch.randn(1, H, T, dqk, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, KVH, T, dqk, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, KVH, T, dv, device="cuda", dtype=torch.bfloat16)
        mask = torch.full(
            (T, T), float("-inf"), device="cuda", dtype=torch.bfloat16
        ).triu_(1)[None, None]
        snk = (
            torch.randn(H, device="cuda", dtype=torch.bfloat16)
            if sinks
            else None
        )
        return lambda: _eager_attn(q, k, v, mask, dqk**-0.5, snk)

    return build


def _moe_gqa_call(q, k, v, mask, scale, sinks, window=None):
    """Production non-paged HF attention (runtime/hf_attention.py)."""
    from moe_infinity.runtime.hf_attention import moe_gqa_attention_forward

    mod = torch.nn.Module()
    return lambda: moe_gqa_attention_forward(
        mod, q, k, v, mask, scaling=scale, sliding_window=window, s_aux=sinks
    )


def _moe_gqa_prefill_builder(T, H, KVH, dqk, dv, sinks, window, masked):
    def build():
        q = torch.randn(1, H, T, dqk, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, KVH, T, dqk, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, KVH, T, dv, device="cuda", dtype=torch.bfloat16)
        snk = (
            torch.randn(H, device="cuda", dtype=torch.bfloat16)
            if sinks
            else None
        )
        mask = (
            torch.ones(T, T, device="cuda", dtype=torch.bool).tril_()[
                None, None
            ]
            if masked
            else None
        )
        return _moe_gqa_call(q, k, v, mask, dqk**-0.5, snk, window)

    return build


def _attn_dims(m: MoEModel):
    a = m.attn
    if a.kind == "mla_dsa":  # HF eager MLA: per-head K/V after kv_b_proj
        return (
            a.heads,
            a.heads,
            (a.head_dim if a.rope_dim is None else a.head_dim + a.rope_dim),
            a.v_head_dim,
        )
    return a.heads, a.kv_heads, a.head_dim, a.head_dim


def attn_prefill_cases(model_keys: list[str] | None = None) -> list[Case]:
    out = []
    for key, m in MODELS.items():
        if model_keys and key not in model_keys:
            continue
        if key not in _SHIM_MODELS | _EAGER_MODELS:
            continue
        H, KVH, dqk, dv = _attn_dims(m)
        for lbl, T in PREFILL_TOKENS.items():
            flops = (
                1.0 * T * T * H * (dqk + dv)
            )  # causal: half of 2*T^2*H*(dqk+dv)
            bmin = T * (H * dqk + KVH * (dqk + dv) + H * dv) * BF16
            meta = {"T": T, "H": H, "KVH": KVH, "dqk": dqk, "dv": dv}
            common = dict(
                suite="attn_prefill",
                model=key,
                phase="prefill",
                size=lbl,
                flops=flops,
                bytes_min=bmin,
                layers=m.attn.full_layers + m.attn.swa_layers,
                meta=meta,
                iters=3 if T >= 51200 else 8,
            )
            if key in _SHIM_MODELS:
                out.append(
                    Case(
                        f"{key}|{lbl}|sdpa_shim",
                        impl="sdpa_shim",
                        build=_sdpa_shim_builder(T, H, KVH, dqk),
                        **common,
                    )
                )
            else:
                need = (
                    10.0 * H * T * T + 3 * T * T
                )  # live bf16/fp32 score tensors + mask
                c = Case(
                    f"{key}|{lbl}|hf_eager",
                    impl="hf_eager",
                    build=_hf_eager_prefill_builder(
                        T, H, KVH, dqk, dv, m.attn.sinks
                    ),
                    **common,
                )
                c.meta["est_peak_gb"] = need / 1e9
                if need > 0.85 * 96e9:
                    c.skip = f"OOM: eager scores need ~{need / 1e9:.0f} GB"
                out.append(c)
                if dqk == dv and dqk <= 256:
                    out.append(
                        Case(
                            f"{key}|{lbl}|moe_gqa",
                            impl="moe_gqa",
                            build=_moe_gqa_prefill_builder(
                                T, H, KVH, dqk, dv, m.attn.sinks, None, False
                            ),
                            **common,
                        )
                    )
                if m.attn.window and dqk == dv:
                    W = m.attn.window
                    swa = dict(common)
                    swa["flops"] = 2.0 * H * (dqk + dv) * (T * W - W * W / 2)
                    swa["layers"] = m.attn.swa_layers
                    out.append(
                        Case(
                            f"{key}|{lbl}|moe_gqa_swa",
                            impl="moe_gqa_swa",
                            build=_moe_gqa_prefill_builder(
                                T, H, KVH, dqk, dv, m.attn.sinks, W, False
                            ),
                            **swa,
                        )
                    )
                if m.attn.kind in ("mla_dsa", "minimax_sparse") and dqk == dv:
                    # Sparse layers: dense SDPA under the [1, 1, T, T] index
                    # mask HF builds (sparse kernels: #269 / out of scope).
                    c = Case(
                        f"{key}|{lbl}|moe_gqa_masked",
                        impl="moe_gqa_masked",
                        build=_moe_gqa_prefill_builder(
                            T, H, KVH, dqk, dv, False, None, True
                        ),
                        **common,
                    )
                    if T > 51200:
                        c.skip = "dense [T, T] index mask (> 40 GB)"
                    out.append(c)
            if dqk == dv and dqk <= 256:
                out.append(
                    Case(
                        f"{key}|{lbl}|sdpa_gqa",
                        impl="sdpa_gqa",
                        build=_sdpa_gqa_builder(T, H, KVH, dqk),
                        **common,
                    )
                )
    return out


# --------------------------------------------------------------------------
# Attention decode
# --------------------------------------------------------------------------

KV_BLOCK = 16
MAX_KV_TOKENS = 1 << 22  # cap B*L per decode case (KV memory)


def _paged_kv(B: int, L: int, KVH: int, d: int, layout: str):
    nblk_seq = -(-L // KV_BLOCK)
    nblk = B * nblk_seq
    perm = torch.randperm(nblk, device="cuda").to(torch.int32)
    bt = perm.view(B, nblk_seq).contiguous()
    lens = torch.full((B,), L, device="cuda", dtype=torch.int32)
    if layout == "vllm":
        x = 8
        kc = torch.randn(
            nblk, KVH, d // x, KV_BLOCK, x, device="cuda", dtype=torch.bfloat16
        )
        vc = torch.randn(
            nblk, KVH, d, KV_BLOCK, device="cuda", dtype=torch.bfloat16
        )
    else:
        kc = torch.randn(
            nblk, KV_BLOCK, KVH, d, device="cuda", dtype=torch.bfloat16
        )
        vc = torch.randn(
            nblk, KV_BLOCK, KVH, d, device="cuda", dtype=torch.bfloat16
        )
    return kc, vc, bt, lens


def _paged_fwd_builder(B, L, H, KVH, d, int8=False):
    """Production decode: paged_attention_fwd (split-KV Triton kernel) on the
    vLLM-layout cache of PagedAttentionBackend; int8=True uses int8 payloads
    with per-token fp16 scales (FormatLayeredKVStore int8_sym)."""

    def build():
        from moe_infinity.kernel.paged_attention_ops import paged_attention_fwd

        kc, vc, bt, lens = _paged_kv(B, L, KVH, d, "vllm")
        ks = vs = None
        if int8:
            kc = kc.mul_(40).to(torch.int8)
            vc = vc.mul_(40).to(torch.int8)
            nblk = kc.shape[0]
            ks = torch.rand(nblk, KVH, KV_BLOCK, device="cuda").half()
            vs = torch.rand(nblk, KVH, KV_BLOCK, device="cuda").half()
        q = torch.randn(B, H, d, device="cuda", dtype=torch.bfloat16)
        fn = lambda: paged_attention_fwd(  # noqa: E731
            q, kc, vc, bt, lens, d**-0.5, KVH, KV_BLOCK, L, ks, vs
        )
        fn()  # compile outside timing
        return fn

    return build


def _triton_decode_builder(B, L, H, KVH, d):
    def build():
        from moe_infinity.kernel.fused_decode_attn import fused_decode_attention

        kc, vc, bt, lens = _paged_kv(B, L, KVH, d, "nhd")
        q = torch.randn(B, 1, H, d, device="cuda", dtype=torch.bfloat16)
        fused_decode_attention(
            q, kc, vc, bt, lens, d**-0.5
        )  # autotune outside timing
        return lambda: fused_decode_attention(q, kc, vc, bt, lens, d**-0.5)

    return build


def _sdpa_flash_decode_builder(B, L, H, KVH, d):
    """Reference: contiguous-KV SDPA (flash split-KV kernel) with native GQA."""

    def build():
        q = torch.randn(B, H, 1, d, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, KVH, L, d, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, KVH, L, d, device="cuda", dtype=torch.bfloat16)
        F = torch.nn.functional
        return lambda: F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

    return build


def _hf_eager_decode_builder(B, L, H, KVH, dqk, dv, sinks):
    """HF eager decode over a DynamicCache (contiguous [B, KVH, L, D])."""

    def build():
        q = torch.randn(B, H, 1, dqk, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, KVH, L, dqk, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, KVH, L, dv, device="cuda", dtype=torch.bfloat16)
        snk = (
            torch.randn(H, device="cuda", dtype=torch.bfloat16)
            if sinks
            else None
        )
        return lambda: _eager_attn(q, k, v, None, dqk**-0.5, snk)

    return build


def _moe_gqa_decode_builder(B, L, H, KVH, dqk, dv, sinks, masked):
    """runtime/hf_attention.py over a DynamicCache (contiguous [B, KVH, L, D])."""

    def build():
        q = torch.randn(B, H, 1, dqk, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, KVH, L, dqk, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, KVH, L, dv, device="cuda", dtype=torch.bfloat16)
        snk = (
            torch.randn(H, device="cuda", dtype=torch.bfloat16)
            if sinks
            else None
        )
        # All-visible mask: prices the mask path itself (no skipped keys).
        mask = (
            torch.ones(B, 1, 1, L, device="cuda", dtype=torch.bool)
            if masked
            else None
        )
        return _moe_gqa_call(q, k, v, mask, dqk**-0.5, snk)

    return build


def attn_decode_cases(model_keys: list[str] | None = None) -> list[Case]:
    out = []
    for key, m in MODELS.items():
        if model_keys and key not in model_keys:
            continue
        if key not in _SHIM_MODELS | _EAGER_MODELS:
            continue
        H, KVH, dqk, dv = _attn_dims(m)
        for lbl, L in DECODE_CONTEXTS.items():
            for B in DECODE_BATCHES:
                if B * L > MAX_KV_TOKENS:
                    continue
                kv_bytes = B * L * KVH * (dqk + dv) * BF16
                if key == "glm-5.3" and kv_bytes > 24e9:
                    continue
                meta = {
                    "B": B,
                    "L": L,
                    "H": H,
                    "KVH": KVH,
                    "dqk": dqk,
                    "dv": dv,
                }
                common = dict(
                    suite="attn_decode",
                    model=key,
                    phase="decode",
                    size=f"b{B}@{lbl}",
                    flops=2.0 * B * L * H * (dqk + dv),
                    bytes_min=kv_bytes + B * H * (dqk + dv) * BF16,
                    layers=m.attn.full_layers + m.attn.swa_layers,
                    meta=meta,
                )
                if key in _SHIM_MODELS:
                    out.append(
                        Case(
                            f"{key}|b{B}@{lbl}|paged_fwd",
                            impl="paged_fwd",
                            build=_paged_fwd_builder(B, L, H, KVH, dqk),
                            **common,
                        )
                    )
                    int8_common = dict(common)
                    int8_common["bytes_min"] = (
                        B * L * KVH * (dqk + dv + 4) + B * H * (dqk + dv) * BF16
                    )
                    out.append(
                        Case(
                            f"{key}|b{B}@{lbl}|paged_fwd_int8",
                            impl="paged_fwd_int8",
                            build=_paged_fwd_builder(
                                B, L, H, KVH, dqk, int8=True
                            ),
                            **int8_common,
                        )
                    )
                else:
                    out.append(
                        Case(
                            f"{key}|b{B}@{lbl}|hf_eager_decode",
                            impl="hf_eager_decode",
                            build=_hf_eager_decode_builder(
                                B, L, H, KVH, dqk, dv, m.attn.sinks
                            ),
                            **common,
                        )
                    )
                    out.append(
                        Case(
                            f"{key}|b{B}@{lbl}|moe_gqa_decode",
                            impl="moe_gqa_decode",
                            build=_moe_gqa_decode_builder(
                                B, L, H, KVH, dqk, dv, m.attn.sinks, False
                            ),
                            **common,
                        )
                    )
                    if m.attn.kind in ("mla_dsa", "minimax_sparse"):
                        # Sparse layers pass an explicit [B, 1, 1, L] mask.
                        out.append(
                            Case(
                                f"{key}|b{B}@{lbl}|moe_gqa_masked_decode",
                                impl="moe_gqa_masked_decode",
                                build=_moe_gqa_decode_builder(
                                    B, L, H, KVH, dqk, dv, False, True
                                ),
                                **common,
                            )
                        )
                if dqk == dv and dqk in (64, 80, 96, 112, 128, 256):
                    out.append(
                        Case(
                            f"{key}|b{B}@{lbl}|triton_decode",
                            impl="triton_decode",
                            build=_triton_decode_builder(B, L, H, KVH, dqk),
                            **common,
                        )
                    )
                if dqk == dv:
                    out.append(
                        Case(
                            f"{key}|b{B}@{lbl}|sdpa_flash_decode",
                            impl="sdpa_flash_decode",
                            build=_sdpa_flash_decode_builder(B, L, H, KVH, dqk),
                            **common,
                        )
                    )
    return out


# --------------------------------------------------------------------------
# Router, GDN linear attention, KV write
# --------------------------------------------------------------------------

_FP32_ROUTER = {
    "glm-5.3",
    "glm-5.3-flash",
    "deepseek-v4-flash",
    "deepseek-v4.1-flash",
}


def _router_builder(T, H, E, k, fp32):
    def build():
        x = _rand(T, H, scale=1.0)
        w = _rand(E, H)
        wf = w.float()
        bias = torch.zeros(E, device="cuda")

        def run():
            if fp32:  # GLM / DeepSeek: fp32 logits, sigmoid + bias, top-k
                logits = torch.nn.functional.linear(x.float(), wf)
                s = logits.sigmoid()
                _, idx = torch.topk(s + bias, k, dim=-1)
                wts = s.gather(1, idx)
                wts = wts / wts.sum(-1, keepdim=True)
            else:  # Qwen / GPT-OSS / MiniMax / Mixtral: bf16 logits, softmax top-k
                logits = torch.nn.functional.linear(x, w)
                p = torch.softmax(logits, dim=-1, dtype=torch.float32)
                wts, idx = torch.topk(p, k, dim=-1)
                wts = wts / wts.sum(-1, keepdim=True)
            mask = torch.zeros(T, E, dtype=torch.bool, device="cuda")
            mask.scatter_(1, idx, True)
            torch.sum(mask).cpu()  # expert_executor host sync per MoE layer
            return wts

        return run

    return build


def _router_scoring(key: str) -> str:
    if key.startswith("deepseek-v4"):
        return "sqrtsoftplus"
    return "sigmoid" if key in _FP32_ROUTER else "softmax"


def _router_fused_builder(T, H, E, k, scoring):
    """Fused router (#315), the whole per-layer routing step.

    `moe_infinity.kernel.router.fused_route` writes compact ids/weights, the
    dense mask/weights and the active-expert flags (straight into pinned host
    memory) in one launch (two at T > 1024: cuBLAS logits + routing). The
    dispatcher consumes those flags (`dispatch_experts(layer, flags)`), so it
    only enqueues a host callback: no mask reduction, no D2H copy, no host
    sync.
    """

    def build():
        from moe_infinity.kernel.router import make_router

        x = _rand(T, H, scale=1.0)
        w = _rand(E, H)
        bias = torch.zeros(E, device="cuda") if scoring != "softmax" else None
        # Built once per layer in the wrappers, like here.
        route = make_router(
            w, k, scoring=scoring, bias=bias,
            eps=0.0 if scoring == "softmax" else 1e-20,
            active_flags=True, compact=False,
        )  # fmt: skip

        def run():
            return route(x)[3]

        return run

    return build


def router_cases(model_keys: list[str] | None = None) -> list[Case]:
    out = []
    for key, m in MODELS.items():
        if model_keys and key not in model_keys:
            continue
        fp32 = key in _FP32_ROUTER
        pts = [("prefill", lbl, n) for lbl, n in PREFILL_TOKENS.items()]
        pts += [("decode", f"b{b}", b) for b in DECODE_BATCHES]
        for phase, lbl, T in pts:
            common = dict(
                suite="router",
                model=key,
                phase=phase,
                size=lbl,
                flops=2.0 * T * m.hidden * m.experts,
                layers=m.moe_layers,
                meta={"T": T, "H": m.hidden, "E": m.experts},
            )
            out.append(
                Case(
                    f"{key}|{lbl}|router_{'fp32' if fp32 else 'bf16'}",
                    impl=f"router_{'fp32' if fp32 else 'bf16'}",
                    bytes_min=T * m.hidden * BF16
                    + m.experts * m.hidden * (4 if fp32 else 2),
                    build=_router_builder(
                        T, m.hidden, m.experts, m.top_k, fp32
                    ),
                    **common,
                )
            )
            # bf16 gate weight read once (fp32 logits come from bf16 MMA)
            out.append(
                Case(
                    f"{key}|{lbl}|router_fused",
                    impl="router_fused",
                    bytes_min=T * m.hidden * BF16 + m.experts * m.hidden * BF16,
                    build=_router_fused_builder(
                        T, m.hidden, m.experts, m.top_k, _router_scoring(key)
                    ),
                    **common,
                )
            )
    return out


def _gdn_builder(T, Hv, dk, dv):
    def build():
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            torch_chunk_gated_delta_rule,
        )

        q = torch.randn(1, T, Hv, dk, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, T, Hv, dk, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, T, Hv, dv, device="cuda", dtype=torch.bfloat16)
        g = -torch.rand(1, T, Hv, device="cuda", dtype=torch.float32)
        beta = torch.rand(1, T, Hv, device="cuda", dtype=torch.bfloat16)
        return lambda: torch_chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

    return build


def _fla_missing() -> str | None:
    try:
        import fla.ops  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"flash-linear-attention unavailable: {exc}"
    return None


def _gdn_fla_builder(T, Hv, dk, dv, fused=False):
    def build():
        if fused:
            from moe_infinity.kernel.linear_attention_fused import (
                chunk_gated_delta_rule_fused as chunk_gated_delta_rule,
            )
        else:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        q = torch.randn(1, T, Hv, dk, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, T, Hv, dk, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, T, Hv, dv, device="cuda", dtype=torch.bfloat16)
        g = -torch.rand(1, T, Hv, device="cuda", dtype=torch.float32)
        beta = torch.rand(1, T, Hv, device="cuda", dtype=torch.bfloat16)

        def run():
            # opt-in CUDA recurrence path is chosen per call
            os.environ["MOE_LINEAR_ATTN_CUDA_PRO6000"] = (
                "1" if fused == "cuda" else "0"
            )
            return chunk_gated_delta_rule(
                q,
                k,
                v,
                g=g,
                beta=beta,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )

        return run

    return build


def _kda_inputs(T, H, d):
    q = torch.randn(1, T, H, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, T, H, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, T, H, d, device="cuda", dtype=torch.bfloat16)
    # GLM-5.3-Flash safe gate: -5 * sigmoid(.) per channel, fp32
    g = -5.0 * torch.sigmoid(torch.randn(1, T, H, d, device="cuda") - 4.0)
    beta = torch.rand(1, T, H, device="cuda", dtype=torch.bfloat16)
    return q, k, v, g, beta


def _kda_builder(T, H, d, impl):
    def build():
        if impl == "kda_fla":
            from fla.ops.kda import chunk_kda as fn
        elif impl == "kda_fused":
            from moe_infinity.kernel.linear_attention_fused import (
                chunk_kda_fused as fn,
            )
        else:
            import inspect

            from transformers.models.glm5_next import modeling_glm5_next as mg

            # undecorated torch reference (the module global may be fla)
            fn = inspect.unwrap(mg.chunk_kimi_delta_attention)
        q, k, v, g, beta = _kda_inputs(T, H, d)
        return lambda: fn(
            q,
            k,
            v,
            g=g,
            beta=beta,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

    return build


def _gdn_decode_builder(B, Hv, dk, dv, impl):
    def build():
        if impl == "gdn_fla_decode":
            from fla.ops.gated_delta_rule import (
                fused_recurrent_gated_delta_rule as fn,
            )
        else:
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
                torch_recurrent_gated_delta_rule as fn,
            )
        q = torch.randn(B, 1, Hv, dk, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, 1, Hv, dk, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, 1, Hv, dv, device="cuda", dtype=torch.bfloat16)
        g = -torch.rand(B, 1, Hv, device="cuda", dtype=torch.float32)
        beta = torch.rand(B, 1, Hv, device="cuda", dtype=torch.bfloat16)
        h0 = torch.zeros(B, Hv, dk, dv, device="cuda", dtype=torch.float32)
        return lambda: fn(
            q,
            k,
            v,
            g=g,
            beta=beta,
            initial_state=h0,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

    return build


def gdn_cases(model_keys: list[str] | None = None) -> list[Case]:
    """Linear attention: Qwen3.5 GDN (prefill + decode), GLM-5.3-Flash KDA.

    ``*_torch`` is the transformers fallback, ``*_fla`` the chunked / fused
    Triton kernels of flash-linear-attention, ``*_fused`` MoE-Infinity's
    fused prefill (moe_infinity.kernel.linear_attention_fused, #304).
    """
    out = []
    fla_skip = _fla_missing()
    m = MODELS["qwen3.5-35b-a3b"]
    if not model_keys or m.key in model_keys:
        Hv, dk, dv = 32, 128, 128
        for lbl in ("1K", "2K", "9K", "12K", "50K"):
            T = PREFILL_TOKENS[lbl]
            # chunked delta rule: ~ (4*C*dk + 2*dk*dv*... ) per token; use 6*T*Hv*dk*dv
            for impl, builder in (
                ("gdn_torch", _gdn_builder),
                ("gdn_fla", _gdn_fla_builder),
                ("gdn_fused", functools.partial(_gdn_fla_builder, fused=True)),
                (
                    "gdn_fused_cuda",
                    functools.partial(_gdn_fla_builder, fused="cuda"),
                ),
            ):
                if impl == "gdn_torch" and T > 9216:
                    continue
                out.append(
                    Case(
                        f"{m.key}|{lbl}|{impl}",
                        suite="gdn",
                        model=m.key,
                        phase="prefill",
                        size=lbl,
                        impl=impl,
                        flops=6.0 * T * Hv * dk * dv,
                        bytes_min=T * Hv * (2 * dk + 2 * dv) * BF16,
                        build=builder(T, Hv, dk, dv),
                        layers=m.attn.linear_layers,
                        meta={"T": T},
                        skip=fla_skip if impl != "gdn_torch" else None,
                        iters=3 if impl == "gdn_torch" else None,
                    )
                )
        for B in (1, 16):
            for impl in ("gdn_torch_decode", "gdn_fla_decode"):
                out.append(
                    Case(
                        f"{m.key}|b{B}|{impl}",
                        suite="gdn",
                        model=m.key,
                        phase="decode",
                        size=f"b{B}",
                        impl=impl,
                        flops=6.0 * B * Hv * dk * dv,
                        # fp32 state read + write dominates
                        bytes_min=2 * B * Hv * dk * dv * 4
                        + B * Hv * (2 * dk + 2 * dv) * BF16,
                        build=_gdn_decode_builder(B, Hv, dk, dv, impl),
                        layers=m.attn.linear_layers,
                        meta={"T": B},
                        skip=fla_skip if impl == "gdn_fla_decode" else None,
                    )
                )
    m = MODELS["glm-5.3-flash"]
    if not model_keys or m.key in model_keys:
        H, d = 64, 128  # linear_attn_config: num_heads=64, head_dim=128
        for lbl in ("1K", "2K", "9K", "12K"):
            T = PREFILL_TOKENS[lbl]
            for impl in ("kda_torch", "kda_fla", "kda_fused"):
                skip = fla_skip if impl != "kda_torch" else None
                if impl == "kda_torch":
                    try:
                        from transformers.models.glm5_next import (  # noqa: F401
                            modeling_glm5_next,
                        )
                    except Exception:  # noqa: BLE001
                        skip = "transformers < 5.16 (no glm5_next)"
                    if T > 2048:
                        # [H, T/64, 64, 64, d] fp32 decay tensors: ~19 GB each
                        skip = "torch KDA reference: O(T*64*d) fp32 temporaries"
                out.append(
                    Case(
                        f"{m.key}|{lbl}|{impl}",
                        suite="gdn",
                        model=m.key,
                        phase="prefill",
                        size=lbl,
                        impl=impl,
                        flops=6.0 * T * H * d * d,
                        bytes_min=T * H * (4 * d * BF16 + d * 4),
                        build=_kda_builder(T, H, d, impl),
                        layers=m.attn.linear_layers,
                        meta={"T": T},
                        skip=skip,
                        iters=3 if impl == "kda_torch" else None,
                    )
                )
    return out


def _kv_write_builder(T, KVH, d):
    """PagedAttentionBackend.write_kv: per-token Python loop with .item()."""

    def build():
        x = 8
        nblk = -(-T // KV_BLOCK) + 1
        k_store = torch.zeros(
            nblk, KVH, d // x, KV_BLOCK, x, device="cuda", dtype=torch.bfloat16
        )
        v_store = torch.zeros(
            nblk, KVH, d, KV_BLOCK, device="cuda", dtype=torch.bfloat16
        )
        k_src = torch.randn(T, KVH, d, device="cuda", dtype=torch.bfloat16)
        v_src = torch.randn(T, KVH, d, device="cuda", dtype=torch.bfloat16)
        slots = torch.arange(T, device="cuda", dtype=torch.long)

        def run():
            for i in range(T):
                slot = int(slots[i].item())
                b, o = slot // KV_BLOCK, slot % KV_BLOCK
                k_store[b, :, :, o, :] = k_src[i].reshape(KVH, d // x, x)
                v_store[b, :, :, o] = v_src[i]

        return run

    return build


def _kv_write_vec_builder(T, KVH, d, decode):
    """PagedAttentionBackend.write_kv (single Triton scatter launch).

    Decode: one token per sequence, each in its own randomly placed block.
    Prefill: one sequence over a shuffled block table.
    """

    def build():
        from moe_infinity.runtime.attention_backend import PagedAttentionBackend
        from moe_infinity.runtime.attention_types import KVCacheSpec

        nblk = (T if decode else -(-T // KV_BLOCK)) + 1
        spec = KVCacheSpec(
            num_kv_heads=KVH,
            head_dim=d,
            dtype=torch.bfloat16,
            block_size=KV_BLOCK,
        )
        backend = PagedAttentionBackend(
            spec, nblk, torch.device("cuda"), num_layers=1
        )
        k_src = torch.randn(T, KVH, d, device="cuda", dtype=torch.bfloat16)
        v_src = torch.randn(T, KVH, d, device="cuda", dtype=torch.bfloat16)
        table = torch.randperm(nblk, device="cuda")
        if decode:
            slots = table[:T] * KV_BLOCK + torch.randint(
                0, KV_BLOCK, (T,), device="cuda"
            )
        else:
            pos = torch.arange(T, device="cuda")
            slots = table[pos // KV_BLOCK] * KV_BLOCK + pos % KV_BLOCK

        def run():
            backend.write_kv(k_src, v_src, slots, 0)

        return run

    return build


def _kv_copy_ref_builder(T, KVH, d):
    """Roof reference: one contiguous copy of the same K+V bytes."""

    def build():
        src = torch.randn(2 * T * KVH * d, device="cuda", dtype=torch.bfloat16)
        dst = torch.empty_like(src)
        return lambda: dst.copy_(src)

    return build


def kv_write_cases(model_keys: list[str] | None = None) -> list[Case]:
    out = []
    for key in ("qwen3-235b-a22b", "qwen3.5-35b-a3b"):
        if model_keys and key not in model_keys:
            continue
        m = MODELS[key]
        for phase, lbl, T in (
            ("decode", "b1", 1),
            ("decode", "b16", 16),
            ("decode", "b64", 64),
            ("decode", "b256", 256),
            ("prefill", "1K", 1024),
            ("prefill", "2K", 2048),
            ("prefill", "4K", 4096),
            ("prefill", "9K", 9216),
            ("prefill", "16K", 16384),
            ("prefill", "50K", 51200),
        ):
            h, d = m.attn.kv_heads, m.attn.head_dim
            builds = {}
            if lbl in ("b1", "b16", "1K"):
                builds["kv_write_loop"] = _kv_write_builder(T, h, d)
            builds["kv_write_vec"] = _kv_write_vec_builder(
                T, h, d, phase == "decode"
            )
            builds["kv_copy_ref"] = _kv_copy_ref_builder(T, h, d)
            for impl, build in builds.items():
                out.append(
                    Case(
                        f"{key}|{lbl}|{impl}",
                        suite="kv_write",
                        model=key,
                        phase=phase,
                        size=lbl,
                        impl=impl,
                        flops=0.0,
                        bytes_min=2
                        * T
                        * m.attn.kv_heads
                        * m.attn.head_dim
                        * BF16
                        * 2,
                        build=build,
                        layers=m.attn.full_layers,
                        meta={"T": T},
                        iters=5 if impl == "kv_write_loop" else None,
                    )
                )
    return out


# --------------------------------------------------------------------------
# Expert layer with real routing (captured Qwen3-30B-A3B trace)
# --------------------------------------------------------------------------

# Captured Qwen3-30B-A3B routing trace: the npz written by
# `benchmarks/expert_drop/trace_parser.py` from a `trace_capture.py` run
# (keys `layer`, `mask[records, experts]`). The suite is skipped when unset.
REAL_TRACE = os.environ.get("KAUDIT_REAL_TRACE", "")


def real_routing_counts(tokens: int, layer: int, seed: int = 0):
    """Per-expert token counts for `tokens` real tokens of one layer."""
    import numpy as np

    z = np.load(REAL_TRACE)
    sel = z["layer"] == layer
    mask = z["mask"][sel]
    if tokens <= mask.shape[0] and tokens >= 64:
        start = (seed * 997) % max(1, mask.shape[0] - tokens + 1)
        rows = mask[start : start + tokens]
    else:  # decode batch: random distinct tokens
        rng = np.random.default_rng(seed)
        rows = mask[rng.choice(mask.shape[0], size=tokens, replace=False)]
    return [int(c) for c in rows.sum(0) if c > 0]


def _real_layer_builder(
    counts: list[int], H: int, I: int, streams: int, balanced: bool
):
    def build():
        n = len(counts)
        Ms = [max(1, round(sum(counts) / n))] * n if balanced else counts
        eb = 3 * H * I * BF16
        l2 = torch.cuda.get_device_properties(0).L2_cache_size
        pool = max(1, min(n, math.ceil(4 * l2 / eb)))
        from benchmarks.kernel_audit.ext import load

        ext = load()
        W = [(_rand(I, H), _rand(I, H), _rand(H, I)) for _ in range(pool)]
        bufs = []
        for i, M in enumerate(Ms):
            x = _rand(M, H, scale=1.0)
            gb = torch.empty(M, I, device="cuda", dtype=torch.bfloat16)
            bufs.append(
                (
                    x,
                    gb,
                    torch.empty_like(gb),
                    torch.empty(M, H, device="cuda", dtype=torch.bfloat16),
                )
            )
        ss = [torch.cuda.Stream() for _ in range(streams)]
        main = torch.cuda.current_stream()
        # Largest experts first, as a greedy schedule would.
        order = sorted(range(n), key=lambda i: -Ms[i])

        def run():
            for s in ss:
                s.wait_stream(main)
            for j, i in enumerate(order):
                g, u, d = W[i % pool]
                x, gb, fb, out = bufs[i]
                with torch.cuda.stream(ss[j % streams]):
                    ext.fused_moe_ffn_into(x, g, u, d, gb, fb, out)
            for s in ss:
                main.wait_stream(s)

        return run

    return build


def expert_real_cases(model_keys: list[str] | None = None) -> list[Case]:
    m = MODELS["qwen3-30b-a3b"]
    if (model_keys and m.key not in model_keys) or not os.path.exists(
        REAL_TRACE
    ):
        return []
    out = []
    H, I = m.hidden, m.inter
    pts = [("prefill", "1K", 1024), ("prefill", "2K", 2048)]
    pts += [("decode", f"b{b}", b) for b in DECODE_BATCHES]
    for phase, lbl, n in pts:
        for layer in (0, 24, 47):
            counts = real_routing_counts(n, layer)
            flops = 2.0 * sum(counts) * 3 * H * I
            bmin = len(counts) * 3 * H * I * BF16 + 2 * sum(counts) * H * BF16
            meta = {
                "layer": layer,
                "active": len(counts),
                "M_max": max(counts),
                "M_mean": sum(counts) / len(counts),
                "tokens": n,
            }
            for bal in (False, True):
                impl = "cutlass_balanced" if bal else "cutlass_real"
                out.append(
                    Case(
                        f"{m.key}|real|{lbl}|L{layer}|{impl}",
                        suite="expert_real",
                        model=m.key,
                        phase=phase,
                        size=lbl,
                        impl=impl,
                        flops=flops,
                        bytes_min=bmin,
                        build=_real_layer_builder(
                            counts, H, I, EXEC_STREAMS, bal
                        ),
                        layers=m.moe_layers,
                        meta=dict(meta),
                        iters=10,
                    )
                )
    return out


def _gptoss_moe_loop_builder(T: int, H: int, I: int, E: int, k: int):
    """SyncGptOssMLP resident forward: router + Python loop over ALL experts
    with a `token_mask.any()` host sync each, Triton MXFP4 GEMMs for hits."""

    def build():
        from moe_infinity.kernel.mxfp4_gemm import fused_mxfp4_gemm

        x = _rand(T, H, scale=1.0)
        wr, br = _rand(E, H), _rand(E)
        experts = [_mxfp4_weights(H, I) for _ in range(min(E, 16))]
        F = torch.nn.functional

        def run():
            logits = F.linear(x, wr, br)
            p = F.softmax(logits, dim=-1, dtype=torch.float32)
            w, sel = torch.topk(p, k, dim=-1)
            w = (w / w.sum(-1, keepdim=True)).to(x.dtype)
            mask = torch.zeros(T, E, dtype=torch.bool, device="cuda").scatter_(
                1, sel, True
            )
            wmask = torch.zeros(T, E, dtype=x.dtype, device="cuda").scatter_(
                1, sel, w
            )
            out = torch.zeros_like(x)
            for e in range(E):
                tm = mask[:, e]
                if not tm.any():
                    continue
                gp, gs, gb, dp, ds, db = experts[e % len(experts)]
                xe = x[tm]
                gu = fused_mxfp4_gemm(xe, gp, gs, gb)
                act = _gptoss_swiglu(gu[..., ::2], gu[..., 1::2])
                ye = fused_mxfp4_gemm(act.to(torch.bfloat16), dp, ds, db)
                out[tm] += wmask[tm, e].unsqueeze(-1) * ye
            return out

        return run

    return build


def _gptoss_moe_grouped_builder(
    T: int,
    H: int,
    I: int,
    E: int,
    k: int,
    graph: bool = False,
    router: bool = True,
):
    """Same router, then device-side dispatch (#313): moe_align + grouped
    MXFP4 GEMM1 (+SwiGLU) + GEMM2 (weighted scatter-add), no host sync.
    `graph=True` replays the layer as one CUDA graph; `router=False` times
    the expert block alone (top-k ids/weights computed once outside)."""

    def build():
        from moe_infinity.kernel.mxfp4_moe import fused_moe_mxfp4

        # Fixed routing: the active-expert count (and so the bytes actually
        # read) would otherwise change from run to run.
        torch.manual_seed(0)
        x = _rand(T, H, scale=1.0)
        wr, br = _rand(E, H), _rand(E)
        gu_p = torch.randint(
            0, 255, (E, 2 * I, H // 2), device="cuda", dtype=torch.uint8
        )
        gu_s = torch.randint(
            118, 124, (E, 2 * I, H // 32), device="cuda", dtype=torch.uint8
        )
        d_p = torch.randint(
            0, 255, (E, H, I // 2), device="cuda", dtype=torch.uint8
        )
        d_s = torch.randint(
            118, 124, (E, H, I // 32), device="cuda", dtype=torch.uint8
        )
        gu_b, d_b = _rand(E, 2 * I), _rand(E, H)
        F = torch.nn.functional

        def route():
            logits = F.linear(x, wr, br)
            p = F.softmax(logits, dim=-1, dtype=torch.float32)
            w, sel = torch.topk(p, k, dim=-1)
            return sel, (w / w.sum(-1, keepdim=True)).to(x.dtype)

        fixed = None if router else route()

        def run():
            sel, w = fixed or route()
            return fused_moe_mxfp4(x, sel, w, gu_p, gu_s, gu_b, d_p, d_s, d_b)

        if not graph:
            return run
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                run()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run()
        return g.replay

    return build


def gptoss_loop_cases(model_keys: list[str] | None = None) -> list[Case]:
    out = []
    impls = (
        ("gptoss_moe_loop", _gptoss_moe_loop_builder, {}, 5),
        ("gptoss_moe_grouped", _gptoss_moe_grouped_builder, {}, None),
        (
            "gptoss_moe_grouped_graph",
            _gptoss_moe_grouped_builder,
            {"graph": True},
            None,
        ),
        (
            "gptoss_moe_experts_graph",
            _gptoss_moe_grouped_builder,
            {"graph": True, "router": False},
            None,
        ),
    )
    for key in ("gpt-oss-20b", "gpt-oss-120b"):
        if model_keys and key not in model_keys:
            continue
        m = MODELS[key]
        for phase, lbl, T in (
            ("decode", "b1", 1),
            ("decode", "b16", 16),
            ("decode", "b64", 64),
            ("prefill", "1K", 1024),
        ):
            M, A = expert_tokens(m, T)
            for impl, builder, kw, iters in impls:
                if impl == "gptoss_moe_loop" and lbl == "b64":
                    continue
                out.append(
                    Case(
                        f"{key}|{lbl}|{impl}",
                        suite="gptoss_loop",
                        model=key,
                        phase=phase,
                        size=lbl,
                        impl=impl,
                        flops=2.0 * T * m.top_k * 3 * m.hidden * m.inter,
                        bytes_min=round(A)
                        * 3
                        * m.hidden
                        * m.inter
                        * _W_BYTES["gptoss_resident"],
                        build=builder(
                            T, m.hidden, m.inter, m.experts, m.top_k, **kw
                        ),
                        layers=m.moe_layers,
                        meta={"T": T, "active": A},
                        iters=iters,
                    )
                )
    return out


def _mxfp4_gemm_builder(M: int, N: int, K: int):
    """One `fused_mxfp4_gemm` launch (GPT-OSS resident path, #312)."""

    def build():
        from moe_infinity.kernel.mxfp4_gemm import fused_mxfp4_gemm

        x = _rand(M, K, scale=1.0)
        w = torch.randint(0, 255, (N, K // 2), device="cuda", dtype=torch.uint8)
        s = torch.randint(
            118, 124, (N, K // 32), device="cuda", dtype=torch.uint8
        )
        b = _rand(N)
        return lambda: fused_mxfp4_gemm(x, w, s, b)

    return build


_read_kernel = None


def _read_ceiling_builder(M: int, N: int, K: int):
    """Streaming read of the same MXFP4 weight bytes (no math): the DRAM
    bandwidth one cold-L2 launch of this size can reach on this GPU."""

    def build():
        global _read_kernel
        import triton
        import triton.language as tl

        if _read_kernel is None:

            @triton.jit
            def _read(p, o, n, BLOCK: tl.constexpr):
                offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
                v = tl.load(p + offs, mask=offs < n, other=0)
                tl.store(o + tl.program_id(0), tl.xor_sum(v, axis=0))

            _read_kernel = _read

        n = (N * (K // 2) + N * (K // 32)) // 4
        buf = torch.randint(0, 255, (n,), device="cuda", dtype=torch.int32)
        grid = (triton.cdiv(n, 512),)
        out = torch.empty(grid[0], device="cuda", dtype=torch.int32)
        return lambda: _read_kernel[grid](buf, out, n, BLOCK=512, num_warps=8)

    return build


def mxfp4_gemm_cases(model_keys: list[str] | None = None) -> list[Case]:
    """Single GPT-OSS MXFP4 GEMMs (gate_up: N=2I, K=H; down: N=H, K=I) at the
    per-expert token counts of every expert-suite point."""
    cases: list[Case] = []
    for key in ("gpt-oss-20b", "gpt-oss-120b"):
        if model_keys and key not in model_keys:
            continue
        m = MODELS[key]
        H, I = m.hidden, m.inter
        for phase, lbl, n in _expert_points(m):
            M, A = expert_tokens(m, n)
            for proj, N, K in (("gate_up", 2 * I, H), ("down", H, I)):
                w_bytes = N * K * _W_BYTES["gptoss_resident"]
                for impl, builder, flops, act in (
                    (
                        "triton_mxfp4",
                        _mxfp4_gemm_builder,
                        2.0 * M * N * K,
                        M * (K + N) * BF16,
                    ),
                    ("read_ceiling", _read_ceiling_builder, 0.0, 0),
                ):
                    if impl == "read_ceiling" and phase != "decode":
                        continue
                    cases.append(
                        Case(
                            f"{key}|{lbl}|{proj}|{impl}",
                            suite="mxfp4_gemm",
                            model=key,
                            phase=phase,
                            size=f"{lbl}/{proj}",
                            impl=impl,
                            flops=flops,
                            bytes_min=w_bytes + act,
                            build=builder(M, N, K),
                            per_layer=A,
                            layers=m.moe_layers,
                            meta={"M": M, "N": N, "K": K, "tokens": n},
                        )
                    )
    return cases


SUITES: dict[str, Callable[..., list[Case]]] = {
    "expert": expert_cases,
    "mxfp4_gemm": mxfp4_gemm_cases,
    "gptoss_loop": gptoss_loop_cases,
    "expert_layer": expert_layer_cases,
    "expert_layer_graph": expert_layer_graph_cases,
    "expert_grouped": expert_grouped_cases,
    "expert_real": expert_real_cases,
    "attn_prefill": attn_prefill_cases,
    "attn_decode": attn_decode_cases,
    "router": router_cases,
    "gdn": gdn_cases,
    "kv_write": kv_write_cases,
}


def all_cases(suite: str, model_keys: list[str] | None = None) -> list[Case]:
    return SUITES[suite](model_keys)
