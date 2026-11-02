import pytest
import torch
import torch.nn.functional as F
from transformers import PretrainedConfig

from moe_infinity.runtime.expert_activation import (
    CLAMPED_SILU,
    SILU,
    SWIGLU_OAI,
    ExpertActivation,
    apply_expert_activation,
    reference_gated_act,
    resolve_expert_activation,
)


def _cfg(**kwargs):
    return PretrainedConfig(**kwargs)


def test_plain_silu_families():
    for model_type in ("qwen3_moe", "mixtral", "deepseek_v3", "olmoe"):
        cfg = _cfg(hidden_act="silu")
        cfg.model_type = model_type
        assert resolve_expert_activation(cfg, 5) == ExpertActivation()


def test_glm5_next_nested_text_config_is_clamped():
    cfg = _cfg(text_config={"hidden_act": "silu", "swiglu_limit": 10.0})
    cfg.model_type = "glm5_next"
    act = resolve_expert_activation(cfg, 5)
    assert act == ExpertActivation(CLAMPED_SILU, 1.0, 10.0)
    assert act.as_args() == (CLAMPED_SILU, 1.0, 10.0)


def test_deepseek_v4_flat_config_is_clamped():
    cfg = _cfg(hidden_act="silu", swiglu_limit=10.0)
    cfg.model_type = "deepseek_v4"
    assert resolve_expert_activation(cfg, 5).kind == CLAMPED_SILU


def test_minimax_m3_is_swiglu_oai():
    cfg = _cfg(
        text_config={
            "hidden_act": "silu",
            "swiglu_alpha": 1.702,
            "swiglu_limit": 7.0,
        }
    )
    cfg.model_type = "minimax_m3_vl"
    assert resolve_expert_activation(cfg, 5) == ExpertActivation(
        SWIGLU_OAI, 1.702, 7.0
    )


def test_minimax_m3_hf_config_class():
    mod = pytest.importorskip(
        "transformers.models.minimax_m3_vl.configuration_minimax_m3_vl"
    )
    cfg = mod.MiniMaxM3VLConfig()
    act = resolve_expert_activation(cfg, 5)
    assert act.kind == SWIGLU_OAI
    assert act.alpha == pytest.approx(cfg.text_config.swiglu_alpha)
    assert act.limit == pytest.approx(cfg.text_config.swiglu_limit)


def test_qwen3_omni_thinker_text_config_wins():
    # The audio/vision sub-configs use gelu; the thinker LLM experts use silu.
    cfg = _cfg(
        thinker_config={
            "text_config": {"hidden_act": "silu"},
            "audio_config": {"hidden_act": "gelu"},
        }
    )
    cfg.model_type = "qwen3_omni_moe"
    assert resolve_expert_activation(cfg, 5) == ExpertActivation()


def test_non_gated_expert_types_are_untouched():
    cfg = _cfg(hidden_act="gelu", swiglu_limit=7.0)
    for expert_type in (2, 3, 6):
        assert resolve_expert_activation(cfg, expert_type) == ExpertActivation()


def test_unsupported_hidden_act_warns():
    cfg = _cfg(hidden_act="gelu")
    with pytest.warns(UserWarning, match="SiLU-gated"):
        assert resolve_expert_activation(cfg, 5).kind == SILU


class _Dispatcher:
    def __init__(self):
        self.calls = []

    def set_expert_activation(self, kind, alpha, limit):
        self.calls.append((kind, alpha, limit))


def test_apply_expert_activation_forwards_descriptor():
    d = _Dispatcher()
    apply_expert_activation(d, ExpertActivation(SWIGLU_OAI, 1.702, 7.0))
    assert d.calls == [(SWIGLU_OAI, 1.702, 7.0)]


def test_apply_expert_activation_requires_binding_for_non_silu():
    apply_expert_activation(object(), ExpertActivation())
    with pytest.raises(RuntimeError, match="set_expert_activation"):
        apply_expert_activation(object(), ExpertActivation(CLAMPED_SILU, 1, 10))


def test_reference_matches_hf_formulations():
    torch.manual_seed(0)
    gate = torch.randn(64, 32) * 20
    up = torch.randn(64, 32) * 20

    torch.testing.assert_close(
        reference_gated_act(gate, up, ExpertActivation()), F.silu(gate) * up
    )
    # GLM-5.3-Flash Glm5NextTextMLP.forward
    torch.testing.assert_close(
        reference_gated_act(gate, up, ExpertActivation(CLAMPED_SILU, 1, 10)),
        F.silu(gate.clamp(max=10)) * up.clamp(-10, 10),
    )
    # MiniMax-M3 MiniMaxM3VLExperts._apply_gate
    g = gate.clamp(max=7)
    torch.testing.assert_close(
        reference_gated_act(gate, up, ExpertActivation(SWIGLU_OAI, 1.702, 7)),
        (up.clamp(-7, 7) + 1) * g * torch.sigmoid(1.702 * g),
    )
