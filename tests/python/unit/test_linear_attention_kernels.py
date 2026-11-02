"""Linear-attention kernel report and fallback warning (#304), CPU only."""

from __future__ import annotations

import functools
import logging
import sys
import types

import pytest
import torch

from moe_infinity.kernel import linear_attention as la


def _fla_fn(name):
    def fn(*args, **kwargs):
        return None

    fn.__module__ = "fla.ops.gated_delta_rule"
    fn.__name__ = name
    return fn


def _hub_with_fallback(implementation):
    """Shape of transformers' ``use_kernel_func_from_hub_with_fallback``."""

    def decorator(torch_function):
        impl = implementation or torch_function

        @functools.wraps(torch_function)
        def wrapped(*args, **kwargs):
            return impl(*args, **kwargs)

        return wrapped

    return decorator


def _make_module_style(use_fla: bool) -> type:
    """transformers >= 5.16 style: kernels are module-level globals."""
    mod = types.ModuleType(f"_fake_glm5_next_{int(use_fla)}")

    def chunk_kimi_delta_attention(*a, **k):
        return None

    def recurrent_kimi_delta_attention(*a, **k):
        return None

    def causal_conv1d_fn(*a, **k):
        return None

    mod.chunk_kimi_delta_attention = _hub_with_fallback(
        _fla_fn("chunk_kda") if use_fla else None
    )(chunk_kimi_delta_attention)
    mod.recurrent_kimi_delta_attention = _hub_with_fallback(
        _fla_fn("fused_recurrent_kda") if use_fla else None
    )(recurrent_kimi_delta_attention)
    mod.causal_conv1d_fn = _hub_with_fallback(None)(causal_conv1d_fn)

    class FakeTextLinearAttention(torch.nn.Module):
        pass

    FakeTextLinearAttention.__module__ = mod.__name__
    mod.FakeTextLinearAttention = FakeTextLinearAttention
    sys.modules[mod.__name__] = mod
    return FakeTextLinearAttention


class FakeGatedDeltaNet(torch.nn.Module):
    """transformers <= 5.12 style: kernels bound as instance attributes."""

    def __init__(self, use_fla: bool):
        super().__init__()
        self.chunk_gated_delta_rule = (
            _fla_fn("chunk_gated_delta_rule") if use_fla else lambda: None
        )
        self.recurrent_gated_delta_rule = (
            _fla_fn("fused_recurrent_gated_delta_rule")
            if use_fla
            else lambda: None
        )
        self.causal_conv1d_fn = None


@pytest.fixture(autouse=True)
def _reset_warned():
    la._warned.clear()
    yield
    la._warned.clear()


def test_instance_attr_style_report():
    model = torch.nn.Sequential(
        FakeGatedDeltaNet(True), torch.nn.Linear(2, 2), FakeGatedDeltaNet(True)
    )
    report = la.linear_attention_kernels(model)
    assert report == {
        "FakeGatedDeltaNet": {
            "layers": 2,
            "chunk": "fla",
            "recurrent": "fla",
            "conv": "torch",
        }
    }
    report = la.linear_attention_kernels(
        torch.nn.Sequential(FakeGatedDeltaNet(False))
    )
    assert report["FakeGatedDeltaNet"]["chunk"] == "torch"


@pytest.mark.parametrize("use_fla", [True, False])
def test_module_global_style_report(use_fla):
    cls = _make_module_style(use_fla)
    report = la.linear_attention_kernels(torch.nn.Sequential(cls(), cls()))
    want = "fla" if use_fla else "torch"
    assert report["FakeTextLinearAttention"] == {
        "layers": 2,
        "chunk": want,
        "recurrent": want,
        "conv": "torch",
    }


def test_non_linear_attention_modules_ignored():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.LayerNorm(2))
    assert la.linear_attention_kernels(model) == {}


def test_warns_once_on_torch_fallback(monkeypatch, caplog):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model = torch.nn.Sequential(FakeGatedDeltaNet(False))
    with caplog.at_level(logging.WARNING, logger=la.__name__):
        la.check_linear_attention_kernels(model)
        la.check_linear_attention_kernels(model)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "flash-linear-attention" in warnings[0].getMessage()


def test_no_warning_when_fla_bound(monkeypatch, caplog):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with caplog.at_level(logging.WARNING, logger=la.__name__):
        la.check_linear_attention_kernels(
            torch.nn.Sequential(FakeGatedDeltaNet(True))
        )
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_real_qwen3_5_layer_reports_fallback_without_fla():
    pytest.importorskip("transformers.models.qwen3_5_moe")
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as mq
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig,
    )

    if torch.cuda.is_available():
        pytest.skip("CPU check; GPU binding is covered in tests/python/ops")
    cfg = Qwen3_5MoeTextConfig(
        hidden_size=64,
        num_hidden_layers=4,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        vocab_size=64,
    )
    layer = mq.Qwen3_5MoeGatedDeltaNet(cfg, layer_idx=0)
    report = la.linear_attention_kernels(torch.nn.Sequential(layer))
    assert report["Qwen3_5MoeGatedDeltaNet"]["chunk"] == "torch"
    assert report["Qwen3_5MoeGatedDeltaNet"]["recurrent"] == "torch"


def test_device_safe_dispatch_routes_cpu_to_torch(monkeypatch):
    calls = []
    mod = types.ModuleType("_fake_qwen3_5_modeling")

    def fla_chunk(query, *a, **k):
        calls.append("fla")

    fla_chunk.__module__ = "fla.ops.gated_delta_rule"

    def torch_chunk_gated_delta_rule(query, *a, **k):
        calls.append("torch")

    class FusedRMSNormGated(torch.nn.Module):
        def __init__(self, hidden, eps=1e-6, activation="silu"):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.rand(hidden) + 0.5)
            self.eps, self.activation = eps, activation

        def forward(self, x, g=None):
            raise AssertionError("Triton path must not run on CPU")

    mod.chunk_gated_delta_rule = fla_chunk
    mod.torch_chunk_gated_delta_rule = torch_chunk_gated_delta_rule
    mod.fused_recurrent_gated_delta_rule = None
    mod.FusedRMSNormGated = FusedRMSNormGated
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setattr(la, "_HF_MODULES", (mod.__name__,))

    patched = la.make_linear_attention_device_safe()
    assert patched == [
        f"{mod.__name__}.chunk_gated_delta_rule",
        f"{mod.__name__}.FusedRMSNormGated",
    ]
    assert la.make_linear_attention_device_safe() == []  # idempotent
    assert la._backend_of(mod.chunk_gated_delta_rule) == "fla"

    mod.chunk_gated_delta_rule(torch.zeros(2), beta=None)
    mod.chunk_gated_delta_rule(query=torch.zeros(2))
    assert calls == ["torch", "torch"]

    norm = mod.FusedRMSNormGated(8)
    x, g = torch.randn(3, 8), torch.randn(3, 8)
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    ref = ref * norm.weight * torch.nn.functional.silu(g)
    torch.testing.assert_close(norm(x, g), ref)
    assert type(norm).__name__ == "FusedRMSNormGated"


def test_make_fused_routes_unsupported_calls_to_fallback(monkeypatch):
    from moe_infinity.kernel import linear_attention_fused as fused_mod

    calls = []

    def fallback(q, k, v, g=None, beta=None, **kw):
        calls.append(sorted(kw))
        return "fallback"

    fallback.__module__ = "fla.ops.gated_delta_rule"
    fn = fused_mod.make_fused("gdn", fallback)
    assert la._backend_of(fn) == "fused"
    x = torch.zeros(1, 4, 2, 128)  # CPU: never the fused path
    out = fn(
        x,
        x,
        x,
        g=torch.zeros(1, 4, 2),
        beta=torch.zeros(1, 4, 2),
        cu_seqlens=None,
    )
    assert out == "fallback" and calls == [["cu_seqlens"]]

    monkeypatch.setenv(fused_mod.ENV_DISABLE, "0")
    assert not fused_mod.fused_enabled()


def test_install_fused_wraps_only_fla_globals(monkeypatch):
    mod = types.ModuleType("_fake_qwen3_5_modeling_fused")
    mod.chunk_gated_delta_rule = _fla_fn("chunk_gated_delta_rule")
    mod.chunk_kimi_delta_attention = lambda *a, **k: None  # torch: untouched
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setattr(la, "_HF_MODULES", (mod.__name__,))

    assert la.install_fused_linear_attention() == [
        f"{mod.__name__}.chunk_gated_delta_rule"
    ]
    assert la.install_fused_linear_attention() == []  # idempotent
    assert la._backend_of(mod.chunk_gated_delta_rule) == "fused"
    assert la._backend_of(mod.chunk_kimi_delta_attention) == "torch"

    layer = FakeGatedDeltaNet(True)
    layer.__class__.__module__ = mod.__name__
    try:
        assert la._bind_instances(torch.nn.Sequential(layer)) == 1
    finally:
        layer.__class__.__module__ = __name__
    assert layer.chunk_gated_delta_rule is mod.chunk_gated_delta_rule
