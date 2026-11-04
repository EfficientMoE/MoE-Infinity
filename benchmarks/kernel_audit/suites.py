"""Kernel cases for the sm_120 audit, built from real model shapes.

Each Case builds its tensors lazily and returns a zero-arg launcher that
issues the same kernels MoE-Infinity issues for that op on sm_120
(torch 2.11 / cu130, no flash_attn / flashinfer / fla installed).

Impl names:
  production paths   cutlass_fused, fp8_prod, gptoss_resident, gptoss_offload,
                     v4_fp4_prod, sdpa_shim, hf_eager, paged_v1,
                     sdpa_fallback, hf_eager_decode, router_*, gdn_torch,
                     kv_write_loop, moe_gqa, moe_gqa_swa, moe_gqa_masked,
                     moe_gqa_decode, moe_gqa_masked_decode (HF models
                     without a paged shim, runtime/hf_attention.py)
  references         cublas_ref, sdpa_gqa, sdpa_flash_decode, triton_decode
"""

from __future__ import annotations

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


def _gptoss_offload_builder(M: int, H: int, I: int, dequant: bool = True):
    """C++ type-6 path (device_memory_ratio < 0.9): MXFP4 dequant + cuBLAS."""

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


# weight bytes per element of one expert in each storage format
_W_BYTES = {
    "cutlass_fused": BF16,
    "cublas_ref": BF16,
    "gptoss_bf16cached": BF16,
    "fp8_prod": 1 + 4 / (128 * 128),
    "gptoss_resident": 0.5 + 1 / 32,
    "gptoss_offload": 0.5 + 1 / 32,
    "v4_fp4_prod": 0.5 + 1 / 32,
}


def _expert_impls(m: MoEModel):
    if m.expert_type == 6:
        return [
            ("gptoss_resident", _gptoss_resident_builder),
            ("gptoss_offload", _gptoss_offload_builder),
            (
                "gptoss_bf16cached",
                lambda M, H, I: _gptoss_offload_builder(M, H, I, False),
            ),
            ("cublas_ref", _cublas_ref_builder),
        ]
    impls = []
    if m.expert_dtype.startswith("fp8"):
        impls.append(("fp8_prod", _fp8_prod_builder))
    if m.expert_dtype.startswith("fp4"):
        impls.append(("v4_fp4_prod", _v4_fp4_builder))
    impls += [
        ("cutlass_fused", _cutlass_builder),
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
        if c.phase != "decode" or c.impl not in ("cutlass_fused", "cublas_ref"):
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


def expert_layer_cases(model_keys: list[str] | None = None) -> list[Case]:
    out = []
    for c in expert_cases(model_keys):
        if c.impl == "gptoss_bf16cached":
            continue
        n_exec = max(1, round(c.per_layer))
        H, I = c.meta["H"], c.meta["I"]
        eb = 3 * H * I * _W_BYTES[c.impl]
        streams = 1 if c.impl in _SEQUENTIAL_IMPLS else EXEC_STREAMS
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
                build=_layer_builder(c, n_exec, eb, streams),
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
        query = torch.randn(T, H, d, device="cuda", dtype=torch.bfloat16)
        key = torch.randn(T, KVH, d, device="cuda", dtype=torch.bfloat16)
        value = torch.randn(T, KVH, d, device="cuda", dtype=torch.bfloat16)
        F = torch.nn.functional
        ratio = H // KVH

        def run():
            q = query.transpose(0, 1).unsqueeze(0)
            k = key.transpose(0, 1).unsqueeze(0)
            v = value.transpose(0, 1).unsqueeze(0)
            if ratio > 1:
                k = k.repeat_interleave(ratio, dim=1)
                v = v.repeat_interleave(ratio, dim=1)
            out = F.scaled_dot_product_attention(
                q, k, v, scale=d**-0.5, is_causal=True
            )
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


def _paged_fwd_builder(B, L, H, KVH, d):
    """Production decode: paged_attention_fwd (v1 kernel for D<=128, SDPA
    Python fallback for D=256)."""

    def build():
        from moe_infinity.kernel.paged_attention_ops import paged_attention_fwd

        kc, vc, bt, lens = _paged_kv(B, L, KVH, d, "vllm")
        q = torch.randn(B, H, d, device="cuda", dtype=torch.bfloat16)
        return lambda: paged_attention_fwd(
            q, kc, vc, bt, lens, d**-0.5, KVH, KV_BLOCK, L
        )

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


# Production paged decode is extremely slow; sample a few points only.
_PAGED_POINTS = {(1, "1K"), (1, "2K"), (16, "1K"), (1, "9K")}
_FALLBACK_POINTS = {
    (1, "1K"),
    (1, "9K"),
    (16, "1K"),
    (16, "9K"),
    (1, "50K"),
    (1, "200K"),
}


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
                    pts = _PAGED_POINTS if dqk <= 128 else _FALLBACK_POINTS
                    if (B, lbl) in pts:
                        impl = "paged_v1" if dqk <= 128 else "sdpa_fallback"
                        out.append(
                            Case(
                                f"{key}|b{B}@{lbl}|{impl}",
                                impl=impl,
                                build=_paged_fwd_builder(B, L, H, KVH, dqk),
                                iters=2,
                                **common,
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
                if dqk == dv and dqk in (64, 128):
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


def router_cases(model_keys: list[str] | None = None) -> list[Case]:
    out = []
    for key, m in MODELS.items():
        if model_keys and key not in model_keys:
            continue
        fp32 = key in _FP32_ROUTER
        pts = [("prefill", lbl, n) for lbl, n in PREFILL_TOKENS.items()]
        pts += [("decode", f"b{b}", b) for b in DECODE_BATCHES]
        for phase, lbl, T in pts:
            out.append(
                Case(
                    f"{key}|{lbl}|router_{'fp32' if fp32 else 'bf16'}",
                    suite="router",
                    model=key,
                    phase=phase,
                    size=lbl,
                    impl=f"router_{'fp32' if fp32 else 'bf16'}",
                    flops=2.0 * T * m.hidden * m.experts,
                    bytes_min=T * m.hidden * BF16
                    + m.experts * m.hidden * (4 if fp32 else 2),
                    build=_router_builder(
                        T, m.hidden, m.experts, m.top_k, fp32
                    ),
                    layers=m.moe_layers,
                    meta={"T": T, "H": m.hidden, "E": m.experts},
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


def gdn_cases(model_keys: list[str] | None = None) -> list[Case]:
    out = []
    m = MODELS["qwen3.5-35b-a3b"]
    if model_keys and m.key not in model_keys:
        return out
    Hv, dk, dv = 32, 128, 128
    for lbl in ("1K", "2K", "9K"):
        T = PREFILL_TOKENS[lbl]
        # chunked delta rule: ~ (4*C*dk + 2*dk*dv*... ) per token; use 6*T*Hv*dk*dv
        out.append(
            Case(
                f"{m.key}|{lbl}|gdn_torch",
                suite="gdn",
                model=m.key,
                phase="prefill",
                size=lbl,
                impl="gdn_torch",
                flops=6.0 * T * Hv * dk * dv,
                bytes_min=T * Hv * (2 * dk + 2 * dv) * BF16,
                build=_gdn_builder(T, Hv, dk, dv),
                layers=m.attn.linear_layers,
                meta={"T": T},
                iters=3,
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


def kv_write_cases(model_keys: list[str] | None = None) -> list[Case]:
    out = []
    for key in ("qwen3-235b-a22b", "qwen3.5-35b-a3b"):
        if model_keys and key not in model_keys:
            continue
        m = MODELS[key]
        for phase, lbl, T in (
            ("decode", "b1", 1),
            ("decode", "b16", 16),
            ("prefill", "1K", 1024),
        ):
            out.append(
                Case(
                    f"{key}|{lbl}|kv_write_loop",
                    suite="kv_write",
                    model=key,
                    phase=phase,
                    size=lbl,
                    impl="kv_write_loop",
                    flops=0.0,
                    bytes_min=2
                    * T
                    * m.attn.kv_heads
                    * m.attn.head_dim
                    * BF16
                    * 2,
                    build=_kv_write_builder(
                        T, m.attn.kv_heads, m.attn.head_dim
                    ),
                    layers=m.attn.full_layers,
                    meta={"T": T},
                    iters=5,
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


def gptoss_loop_cases(model_keys: list[str] | None = None) -> list[Case]:
    out = []
    for key in ("gpt-oss-20b", "gpt-oss-120b"):
        if model_keys and key not in model_keys:
            continue
        m = MODELS[key]
        for phase, lbl, T in (
            ("decode", "b1", 1),
            ("decode", "b16", 16),
            ("prefill", "1K", 1024),
        ):
            M, A = expert_tokens(m, T)
            out.append(
                Case(
                    f"{key}|{lbl}|gptoss_moe_loop",
                    suite="gptoss_loop",
                    model=key,
                    phase=phase,
                    size=lbl,
                    impl="gptoss_moe_loop",
                    flops=2.0 * T * m.top_k * 3 * m.hidden * m.inter,
                    bytes_min=round(A)
                    * 3
                    * m.hidden
                    * m.inter
                    * _W_BYTES["gptoss_resident"],
                    build=_gptoss_moe_loop_builder(
                        T, m.hidden, m.inter, m.experts, m.top_k
                    ),
                    layers=m.moe_layers,
                    meta={"T": T, "active": A},
                    iters=5,
                )
            )
    return out


SUITES: dict[str, Callable[..., list[Case]]] = {
    "expert": expert_cases,
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
