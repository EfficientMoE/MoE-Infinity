from types import SimpleNamespace

import pytest

from moe_infinity.utils.hf_config import parse_expert_id, parse_moe_param


def make_glm_config(num_hidden_layers=78, n_routed_experts=256):
    return SimpleNamespace(
        architectures=["GlmMoeDsaForCausalLM"],
        num_hidden_layers=num_hidden_layers,
        n_routed_experts=n_routed_experts,
    )


def test_parse_moe_param_glm_returns_correct_tuple():
    cfg = make_glm_config()
    num_layers, num_experts, num_encoder_layers = parse_moe_param(cfg)
    assert num_layers == 78
    assert num_experts == 256
    assert num_encoder_layers == 0


def test_parse_expert_id_glm_valid_moe_layer():
    cfg = make_glm_config()
    layer_id, expert_id = parse_expert_id(
        "model.layers.10.mlp.experts.3.gate_proj.weight", cfg
    )
    assert layer_id == 10
    assert expert_id == 3


def test_parse_expert_id_glm_mtp_layer_skipped():
    cfg = make_glm_config()
    layer_id, expert_id = parse_expert_id(
        "model.layers.78.mlp.experts.3.gate_proj.weight", cfg
    )
    assert layer_id is None
    assert expert_id is None


def test_parse_expert_id_glm_indexer_layer_skipped():
    cfg = make_glm_config()
    layer_id, expert_id = parse_expert_id(
        "model.layers.5.self_attn.indexer.k_norm.weight", cfg
    )
    assert layer_id is None
    assert expert_id is None


def test_parse_expert_id_glm_shared_expert_skipped():
    cfg = make_glm_config()
    layer_id, expert_id = parse_expert_id(
        "model.layers.5.mlp.shared_expert.gate_proj.weight", cfg
    )
    assert layer_id is None
    assert expert_id is None


def test_parse_moe_param_unknown_arch_raises():
    cfg = SimpleNamespace(architectures=["UnknownModelForCausalLM"])
    with pytest.raises(RuntimeError, match="Unsupported architecture"):
        parse_moe_param(cfg)


def test_parse_expert_id_glm_first_layer():
    cfg = make_glm_config()
    layer_id, expert_id = parse_expert_id(
        "model.layers.0.mlp.experts.0.gate_proj.weight", cfg
    )
    assert layer_id == 0
    assert expert_id == 0


def test_parse_expert_id_glm_last_valid_layer():
    cfg = make_glm_config()
    layer_id, expert_id = parse_expert_id(
        "model.layers.77.mlp.experts.255.down_proj.weight", cfg
    )
    assert layer_id == 77
    assert expert_id == 255
