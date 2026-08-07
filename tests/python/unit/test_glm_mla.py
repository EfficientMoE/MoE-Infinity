from types import SimpleNamespace

import pytest

pytest.importorskip(
    "transformers.models.glm_moe_dsa.modeling_glm_moe_dsa",
    reason="transformers >= 5.12 required",
)

from moe_infinity.utils.hf_config import parse_expert_id

GLM = SimpleNamespace(
    architectures=["GlmMoeDsaForCausalLM"],
    num_hidden_layers=78,
    n_routed_experts=256,
)

_MLA_NAMES = [
    "model.layers.5.self_attn.q_a_proj.weight",
    "model.layers.5.self_attn.q_a_layernorm.weight",
    "model.layers.5.self_attn.q_b_proj.weight",
    "model.layers.5.self_attn.kv_a_proj_with_mqa.weight",
    "model.layers.5.self_attn.kv_a_layernorm.weight",
    "model.layers.5.self_attn.kv_b_proj.weight",
    "model.layers.5.self_attn.o_proj.weight",
    "model.layers.5.self_attn.indexer.wq_b.weight",
    "model.layers.5.self_attn.indexer.wk.weight",
    "model.layers.5.self_attn.indexer.k_norm.weight",
    "model.layers.5.self_attn.indexer.weights_proj.weight",
]


@pytest.mark.parametrize("name", _MLA_NAMES)
def test_mla_and_indexer_not_experts(name):
    assert parse_expert_id(name, GLM) == (
        None,
        None,
    ), f"MLA tensor misrouted as expert: {name}"


def test_shared_expert_not_routed():
    name = "model.layers.5.mlp.shared_experts.gate_proj.weight"
    assert parse_expert_id(name, GLM) == (None, None)


def test_real_expert_still_routed():
    assert parse_expert_id(
        "model.layers.5.mlp.experts.7.gate_proj.weight", GLM
    ) == (5, 7)


def test_real_expert_layer_0():
    assert parse_expert_id(
        "model.layers.0.mlp.experts.0.up_proj.weight", GLM
    ) == (0, 0)


def test_real_expert_last_layer():
    name = f"model.layers.{GLM.num_hidden_layers - 1}.mlp.experts.255.down_proj.weight"
    assert parse_expert_id(name, GLM) == (GLM.num_hidden_layers - 1, 255)
