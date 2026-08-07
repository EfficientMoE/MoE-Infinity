from types import SimpleNamespace

import pytest


def test_glm_registry_to_expert_type():
    from moe_infinity.common.constants import (
        MODEL_MAPPING_NAMES,
        parse_expert_type,
    )

    if "glmmoedsa" not in MODEL_MAPPING_NAMES:
        pytest.skip("GLM not registered (transformers < 5.12)")
    cfg = SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"])
    assert parse_expert_type(cfg) == 5


def test_glm_parse_moe_param_and_expert_id():
    from moe_infinity.utils.hf_config import parse_expert_id, parse_moe_param

    cfg = SimpleNamespace(
        architectures=["GlmMoeDsaForCausalLM"],
        num_hidden_layers=78,
        n_routed_experts=256,
    )
    assert parse_moe_param(cfg) == (78, 256, 0)
    assert parse_expert_id(
        "model.layers.3.mlp.experts.0.gate_proj.weight", cfg
    ) == (3, 0)
    assert parse_expert_id(
        "model.layers.78.mlp.experts.0.gate_proj.weight", cfg
    ) == (None, None)
    assert parse_expert_id(
        "model.layers.3.self_attn.kv_a_layernorm.weight", cfg
    ) == (None, None)


def test_glm_wrapper_and_offload_importable():
    import moe_infinity.runtime.model_offload  # noqa: F401
    from moe_infinity.models import SyncGlmMoeDsaMoEBlock

    assert SyncGlmMoeDsaMoEBlock is not None
