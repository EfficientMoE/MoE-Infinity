# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from types import SimpleNamespace

from moe_infinity.utils.hf_config import parse_expert_id, parse_moe_param


def _cfg(arch, **kw):
    return SimpleNamespace(architectures=[arch], **kw)


def test_v4_native_expert_key_resolves():
    cfg = _cfg(
        "DeepseekV4ForCausalLM", num_hidden_layers=43, n_routed_experts=256
    )
    assert parse_expert_id("layers.5.ffn.experts.42.w1.weight", cfg) == (5, 42)
    assert parse_expert_id("layers.5.ffn.experts.42.w2.scale", cfg) == (5, 42)
    assert parse_expert_id("layers.5.ffn.experts.42.w3.weight", cfg) == (5, 42)


def test_v4_non_expert_keys_return_none():
    cfg = _cfg(
        "DeepseekV4ForCausalLM", num_hidden_layers=43, n_routed_experts=256
    )
    assert parse_expert_id("layers.5.ffn.gate.weight", cfg) == (None, None)
    assert parse_expert_id("layers.5.attn.wq_a.weight", cfg) == (None, None)
    assert parse_expert_id("embed.weight", cfg) == (None, None)


def test_v3_legacy_regex_unchanged():
    cfg = _cfg(
        "DeepseekV3ForCausalLM", num_hidden_layers=61, n_routed_experts=256
    )
    assert parse_expert_id(
        "model.layers.3.mlp.experts.7.gate_proj.weight", cfg
    ) == (3, 7)


def test_v2_legacy_regex_unchanged():
    cfg = _cfg(
        "DeepseekV2ForCausalLM", num_hidden_layers=27, n_routed_experts=64
    )
    assert parse_expert_id(
        "model.layers.10.mlp.experts.3.up_proj.weight", cfg
    ) == (10, 3)


def test_v4_parse_moe_param():
    cfg = _cfg(
        "DeepseekV4ForCausalLM", num_hidden_layers=43, n_routed_experts=256
    )
    num_layers, num_experts, num_encoder = parse_moe_param(cfg)
    assert (num_layers, num_experts, num_encoder) == (43, 256, 0)


def test_v4_does_not_match_v2_mlp_experts_key():
    cfg = _cfg(
        "DeepseekV4ForCausalLM", num_hidden_layers=43, n_routed_experts=256
    )
    assert parse_expert_id(
        "model.layers.5.mlp.experts.42.gate_proj.weight", cfg
    ) == (None, None)
