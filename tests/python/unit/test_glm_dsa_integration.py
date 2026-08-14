import pytest

from moe_infinity.models.glm_dsa import (
    get_indexer_types,
    indexer_owner_map,
    num_owned_indexers,
    owns_indexer,
)
from moe_infinity.utils.hf_config import parse_expert_id


def _real_cfg():
    try:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(
            "zai-org/GLM-5.2-FP8", trust_remote_code=True
        )
    except Exception:
        pytest.skip("GLM config unavailable offline")


def test_indexshare_ownership_consistent_with_config():
    cfg = _real_cfg()
    types = get_indexer_types(cfg)
    owner = indexer_owner_map(cfg)
    for layer, own in owner.items():
        if own is not None:
            assert owns_indexer(cfg, own)
            assert types[own] == "full"
    assert num_owned_indexers(cfg) == sum(1 for t in types if t == "full")


def test_no_attention_or_indexer_tensor_is_expert():
    cfg = _real_cfg()
    n = cfg.num_hidden_layers
    non_expert = [
        f"model.layers.5.self_attn.q_a_proj.weight",
        f"model.layers.5.self_attn.kv_a_layernorm.weight",
        f"model.layers.5.self_attn.indexer.k_norm.weight",
        f"model.layers.{n}.mlp.experts.0.gate_proj.weight",
    ]
    for name in non_expert:
        assert parse_expert_id(name, cfg) == (None, None)


def test_routed_experts_are_identified():
    cfg = _real_cfg()
    layer_id, expert_id = parse_expert_id(
        "model.layers.5.mlp.experts.42.gate_proj.weight", cfg
    )
    assert layer_id == 5 and expert_id == 42
