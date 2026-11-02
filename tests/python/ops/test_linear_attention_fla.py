"""flash-linear-attention kernels for Qwen3.5 GDN and GLM-5.3-Flash KDA (#304).

Checks that transformers binds the fla chunked / fused-recurrent kernels for
the linear-attention layers, and that they match the transformers torch
references on random inputs. GLM-5.3-Flash needs transformers >= 5.16.
"""

from __future__ import annotations

import contextlib
import importlib.util
import inspect
import os
import sys

import pytest
import torch

from moe_infinity.kernel.linear_attention import (
    enable_linear_attention_kernels,
    linear_attention_kernels,
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        importlib.util.find_spec("fla") is None,
        reason="flash-linear-attention not installed",
    ),
]

FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "..", "fixtures", "glm_5_3_flash"
)


def _has_glm5_next() -> bool:
    try:
        from transformers.models.glm5_next import (  # noqa: F401
            modeling_glm5_next,
        )
    except Exception:  # noqa: BLE001
        return False
    return True


requires_glm5_next = pytest.mark.skipif(
    not _has_glm5_next(), reason="glm5_next needs transformers >= 5.16"
)


def _assert_close(out, ref, atol, rtol=0.0, frac=1.0):
    out, ref = out.float(), ref.float()
    tol = atol + rtol * ref.abs()
    ok = ((out - ref).abs() <= tol).float().mean().item()
    err = (out - ref).abs().max().item()
    assert ok >= frac, f"{(1 - ok) * 100:.3f}% outside tol, max err {err:.3e}"


def _gdn_inputs(B, T, H, dk, dv, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def r(*s):
        return torch.randn(*s, device="cuda", generator=g)

    q = r(B, T, H, dk).bfloat16()
    k = r(B, T, H, dk).bfloat16()
    v = r(B, T, H, dv).bfloat16()
    gate = -torch.rand(B, T, H, device="cuda", generator=g) * 0.5
    beta = torch.rand(B, T, H, device="cuda", generator=g).bfloat16()
    h0 = r(B, H, dk, dv) * 0.1
    return q, k, v, gate, beta, h0


@contextlib.contextmanager
def _force_torch(layer):
    """Bind the transformers torch references on ``layer`` (both styles)."""
    patched = []
    names = (
        "chunk_gated_delta_rule",
        "recurrent_gated_delta_rule",
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
        "chunk_kimi_delta_attention",
        "recurrent_kimi_delta_attention",
    )
    mod = sys.modules[type(layer).__module__]
    for name in names:
        if name in vars(layer):
            ref = getattr(mod, "torch_" + name, None)
            if ref is not None:
                patched.append((layer.__dict__, name, vars(layer)[name]))
                layer.__dict__[name] = inspect.unwrap(ref)
        elif hasattr(mod, name):
            fn = getattr(mod, name)
            patched.append((mod.__dict__, name, fn))
            setattr(mod, name, inspect.unwrap(fn))
    try:
        yield
    finally:
        for ns, name, fn in patched:
            ns[name] = fn


# --------------------------------------------------------------------------
# Gated DeltaNet ops (Qwen3.5-35B-A3B shapes: 32 v-heads, dk = dv = 128)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("T", [64, 1000, 4096])
def test_gdn_chunk_matches_torch_reference(T):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        torch_chunk_gated_delta_rule,
    )

    q, k, v, g, beta, h0 = _gdn_inputs(1, T, 32, 128, 128)
    kw = dict(
        initial_state=h0, output_final_state=True, use_qk_l2norm_in_kernel=True
    )
    o, ht = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, **kw)
    ref = inspect.unwrap(torch_chunk_gated_delta_rule)
    o_ref, ht_ref = ref(q, k, v, g=g, beta=beta, **kw)
    assert o.dtype == torch.bfloat16 and o.shape == o_ref.shape
    _assert_close(o, o_ref, atol=2e-2, rtol=2e-2, frac=0.999)
    _assert_close(ht, ht_ref, atol=2e-2, rtol=2e-2, frac=0.999)


def test_gdn_decode_matches_torch_reference_and_is_o1_launches():
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
    from torch.profiler import ProfilerActivity, profile
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        torch_recurrent_gated_delta_rule,
    )

    q, k, v, g, beta, h0 = _gdn_inputs(4, 1, 32, 128, 128)
    kw = dict(output_final_state=True, use_qk_l2norm_in_kernel=True)
    ref = inspect.unwrap(torch_recurrent_gated_delta_rule)
    o_ref, ht_ref = ref(q, k, v, g=g, beta=beta, initial_state=h0.clone(), **kw)
    o, ht = fused_recurrent_gated_delta_rule(
        q, k, v, g=g, beta=beta, initial_state=h0.clone(), **kw
    )
    _assert_close(o, o_ref, atol=2e-2, rtol=2e-2)
    _assert_close(ht, ht_ref, atol=1e-2, rtol=1e-2)

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fused_recurrent_gated_delta_rule(
            q, k, v, g=g, beta=beta, initial_state=h0, **kw
        )
        torch.cuda.synchronize()
    launches = [e for e in prof.events() if e.device_type.name == "CUDA"]
    # one fused kernel (+ at most a couple of tiny dtype / alloc helpers),
    # independent of how many tokens produced the recurrent state
    assert 1 <= len(launches) <= 3, [e.name for e in launches]


@pytest.mark.parametrize("T", [64, 1000, 4096])
@pytest.mark.parametrize("norm", [True, False])
def test_gdn_fused_matches_torch_reference(T, norm):
    """Kernel-level parity (the call-site threshold MIN_FUSED_TOKENS does
    not apply to the direct entry point)."""
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        torch_chunk_gated_delta_rule,
    )

    from moe_infinity.kernel.linear_attention_fused import (
        chunk_gated_delta_rule_fused,
    )

    q, k, v, g, beta, h0 = _gdn_inputs(1, T, 32, 128, 128)
    if not norm:  # the kernels assume unit-norm q / k
        q = torch.nn.functional.normalize(q.float(), dim=-1).bfloat16()
        k = torch.nn.functional.normalize(k.float(), dim=-1).bfloat16()
    kw = dict(
        initial_state=h0, output_final_state=True, use_qk_l2norm_in_kernel=norm
    )
    o, ht = chunk_gated_delta_rule_fused(q, k, v, g, beta, **kw)
    ref = inspect.unwrap(torch_chunk_gated_delta_rule)
    o_ref, ht_ref = ref(q, k, v, g=g, beta=beta, **kw)
    assert o.dtype == torch.bfloat16 and o.shape == o_ref.shape
    _assert_close(o, o_ref, atol=2e-2, rtol=2e-2, frac=0.999)
    _assert_close(ht, ht_ref, atol=2e-2, rtol=2e-2, frac=0.999)


def test_gdn_fused_is_deterministic():
    """Bitwise-identical outputs over repeated calls at a 9K, 128-CTA launch
    (the shape class of the sm_100/103 tl.dot recurrence race)."""
    from moe_infinity.kernel.linear_attention_fused import (
        chunk_gated_delta_rule_fused,
    )

    q, k, v, g, beta, h0 = _gdn_inputs(1, 9216, 32, 128, 128)
    kw = dict(
        initial_state=h0, output_final_state=True, use_qk_l2norm_in_kernel=True
    )
    o0, h_0 = chunk_gated_delta_rule_fused(q, k, v, g, beta, **kw)
    for _ in range(10):
        o, ht = chunk_gated_delta_rule_fused(q, k, v, g, beta, **kw)
        assert torch.equal(o, o0) and torch.equal(ht, h_0)


def test_fused_wrapper_falls_back_to_fla_for_grad_and_varlen():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    from moe_infinity.kernel.linear_attention_fused import make_fused

    calls = []

    def fallback(*a, **kw):
        calls.append(kw.get("cu_seqlens"))
        return chunk_gated_delta_rule(*a, **kw)

    fn = make_fused("gdn", fallback)
    q, k, v, g, beta, _ = _gdn_inputs(1, 4096, 4, 128, 128)
    cu = torch.tensor([0, 1000, 4096], device="cuda")
    fn(q, k, v, g=g, beta=beta, cu_seqlens=cu, use_qk_l2norm_in_kernel=True)
    fn(q.requires_grad_(), k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    assert len(calls) == 2
    with torch.no_grad():  # inference, >= MIN_FUSED_TOKENS: fused
        fn(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
        assert len(calls) == 2
        fn(
            q[:, :1000],
            k[:, :1000],
            v[:, :1000],
            g=g[:, :1000],
            beta=beta[:, :1000],
            use_qk_l2norm_in_kernel=True,
        )
    assert len(calls) == 3


# --------------------------------------------------------------------------
# KDA ops (GLM-5.3-Flash shapes: 64 heads, head_dim 128, safe gate >= -5)
# --------------------------------------------------------------------------


def _kda_inputs(T, H=64, d=128, seed=0):
    q, k, v, _, beta, h0 = _gdn_inputs(1, T, H, d, d, seed)
    gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    g = -5.0 * torch.sigmoid(
        torch.randn(1, T, H, d, device="cuda", generator=gen) - 4.0
    )
    return q, k, v, g, beta, h0


@requires_glm5_next
@pytest.mark.parametrize("T", [64, 1000])
def test_kda_chunk_matches_torch_reference(T):
    from fla.ops.kda import chunk_kda
    from transformers.models.glm5_next import modeling_glm5_next as mg

    q, k, v, g, beta, h0 = _kda_inputs(T)
    kw = dict(
        initial_state=h0, output_final_state=True, use_qk_l2norm_in_kernel=True
    )
    o, ht = chunk_kda(q, k, v, g=g, beta=beta, **kw)
    ref = inspect.unwrap(mg.chunk_kimi_delta_attention)
    o_ref, ht_ref = ref(q, k, v, g=g, beta=beta, **kw)
    _assert_close(o, o_ref, atol=2e-2, rtol=2e-2, frac=0.999)
    _assert_close(ht, ht_ref, atol=2e-2, rtol=2e-2, frac=0.999)


@requires_glm5_next
@pytest.mark.parametrize("T", [64, 1000])
def test_kda_fused_matches_torch_reference(T):
    from transformers.models.glm5_next import modeling_glm5_next as mg

    from moe_infinity.kernel.linear_attention_fused import chunk_kda_fused

    q, k, v, g, beta, h0 = _kda_inputs(T)
    kw = dict(
        initial_state=h0, output_final_state=True, use_qk_l2norm_in_kernel=True
    )
    o, ht = chunk_kda_fused(q, k, v, g, beta, **kw)
    ref = inspect.unwrap(mg.chunk_kimi_delta_attention)
    o_ref, ht_ref = ref(q, k, v, g=g, beta=beta, **kw)
    _assert_close(o, o_ref, atol=2e-2, rtol=2e-2, frac=0.999)
    _assert_close(ht, ht_ref, atol=2e-2, rtol=2e-2, frac=0.999)
    o2, _ = chunk_kda_fused(q, k, v, g, beta, **kw)
    assert torch.equal(o, o2)


@requires_glm5_next
def test_kda_decode_matches_torch_reference():
    from fla.ops.kda import fused_recurrent_kda
    from transformers.models.glm5_next import modeling_glm5_next as mg

    q, k, v, g, beta, h0 = _kda_inputs(1)
    kw = dict(output_final_state=True, use_qk_l2norm_in_kernel=True)
    ref = inspect.unwrap(mg.recurrent_kimi_delta_attention)
    o_ref, ht_ref = ref(q, k, v, g=g, beta=beta, initial_state=h0.clone(), **kw)
    o, ht = fused_recurrent_kda(
        q, k, v, g=g, beta=beta, initial_state=h0.clone(), **kw
    )
    _assert_close(o, o_ref, atol=2e-2, rtol=2e-2)
    _assert_close(ht, ht_ref, atol=1e-2, rtol=1e-2)


# --------------------------------------------------------------------------
# Layers: transformers binds fla, and the layer output matches torch
# --------------------------------------------------------------------------


def _qwen3_5_layer():
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeGatedDeltaNet,
    )

    cfg = Qwen3_5MoeTextConfig(
        hidden_size=512,
        num_hidden_layers=4,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        vocab_size=256,
    )
    cfg.dtype = torch.bfloat16
    torch.manual_seed(0)
    return Qwen3_5MoeGatedDeltaNet(cfg, layer_idx=0).cuda().bfloat16()


def _glm5_next_layer():
    from transformers import AutoConfig
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextTextLinearAttention,
    )

    payload = AutoConfig.from_pretrained(FIXTURE).to_dict()
    text = payload["text_config"]
    text.update(
        hidden_size=512,
        num_hidden_layers=4,
        first_k_dense_replace=3,
        num_nextn_predict_layers=1,
        layer_types=["linear_attention"] * 3 + ["deepseek_sparse_attention"],
        mlp_layer_types=["dense"] * 3 + ["sparse"],
        indexer_types=["full"] * 4,
    )
    text["linear_attn_config"] = dict(
        text["linear_attn_config"],
        num_heads=4,
        head_dim=128,
        kda_layers=[0, 1, 2],
        full_attn_layers=[3],
    )
    text.pop("quantization_config", None)
    payload.pop("quantization_config", None)
    cfg = type(AutoConfig.from_pretrained(FIXTURE)).from_dict(payload)
    torch.manual_seed(0)
    layer = Glm5NextTextLinearAttention(cfg.text_config, layer_idx=0)
    for name, p in layer.named_parameters():  # weights are torch.empty
        if name.endswith("A_log"):
            torch.nn.init.uniform_(p, -1.0, 1.0)
        elif name.endswith("dt_bias"):
            torch.nn.init.zeros_(p)
        elif p.dim() == 1:
            torch.nn.init.ones_(p)
        else:
            torch.nn.init.normal_(p, std=0.02)
    return layer.cuda().bfloat16()


@pytest.mark.parametrize(
    "build, cls",
    [
        (_qwen3_5_layer, "Qwen3_5MoeGatedDeltaNet"),
        pytest.param(
            _glm5_next_layer,
            "Glm5NextTextLinearAttention",
            marks=requires_glm5_next,
        ),
    ],
)
def test_layer_binds_fused_kernels_and_matches_torch(build, cls):
    layer = build()
    report = enable_linear_attention_kernels(torch.nn.Sequential(layer))
    assert report[cls]["chunk"] == "fused", report
    assert report[cls]["recurrent"] == "fla", report

    # > MIN_FUSED_TOKENS so the GDN layer takes the fused path
    x = torch.randn(1, 4200, layer.hidden_size, device="cuda").bfloat16()
    with torch.no_grad():
        out = layer(x)
        with _force_torch(layer):
            assert (
                linear_attention_kernels(torch.nn.Sequential(layer))[cls][
                    "chunk"
                ]
                == "torch"
            )
            ref = layer(x)
    out = out[0] if isinstance(out, tuple) else out
    ref = ref[0] if isinstance(ref, tuple) else ref
    scale = ref.float().abs().max().item()
    _assert_close(out, ref, atol=2e-2 * scale, rtol=2e-2, frac=0.999)
