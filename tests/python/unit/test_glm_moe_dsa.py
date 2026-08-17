# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import warnings
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import torch


def _glm_config(num_hidden_layers=78, n_routed_experts=256):
    return cast(
        Any,
        SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"],
            model_type="glm_moe_dsa",
            num_hidden_layers=num_hidden_layers,
            n_routed_experts=n_routed_experts,
            first_k_dense_replace=3,
            n_shared_experts=1,
            num_experts_per_tok=8,
        ),
    )


def test_glm_registered_when_transformers_supports_it():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    pytest.importorskip("transformers.models.glm_moe_dsa")
    from moe_infinity.common.constants import (
        MODEL_MAPPING_NAMES,
        MODEL_MAPPING_TYPES,
        parse_expert_type,
    )

    assert "glmmoedsa" in MODEL_MAPPING_NAMES
    assert MODEL_MAPPING_TYPES["glmmoedsa"] == 5
    assert parse_expert_type(_glm_config()) == 5


def test_glm_parse_moe_param():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from moe_infinity.utils.hf_config import parse_moe_param

    assert parse_moe_param(_glm_config()) == (78, 256, 0)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("model.layers.5.mlp.experts.42.gate_proj.weight", (5, 42)),
        ("model.layers.77.mlp.experts.255.down_proj.weight", (77, 255)),
        ("model.layers.3.mlp.shared_experts.up_proj.weight", (None, None)),
        ("model.layers.10.mlp.gate.weight", (None, None)),
        ("model.layers.78.mlp.experts.0.gate_proj.weight", (None, None)),
        ("model.layers.5.self_attn.q_a_proj.weight", (None, None)),
    ],
)
def test_glm_parse_expert_id(name, expected):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from moe_infinity.utils.hf_config import parse_expert_id

    assert parse_expert_id(name, _glm_config()) == expected


def test_glm_fp8_dequant_blockwise():
    from moe_infinity.utils.fp8 import dequant_fp8_blockwise

    torch.manual_seed(0)
    weight = torch.randn(256, 384).to(torch.float8_e4m3fn)
    scale = torch.rand(2, 3, dtype=torch.float32) + 0.5
    scale_full = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[
        :256, :384
    ]
    ref = (weight.to(torch.float32) * scale_full).to(torch.bfloat16)
    out = dequant_fp8_blockwise(
        weight, scale, dtype=torch.bfloat16, block_size=128
    )
    assert out.dtype == torch.bfloat16
    assert torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_glm_fp8_dequant_state_dict():
    from moe_infinity.utils.fp8 import dequant_fp8_state_dict

    weight = torch.randn(256, 384).to(torch.float8_e4m3fn)
    scale = torch.rand(2, 3, dtype=torch.float32) + 0.5
    state_dict = {
        "m.gate_proj.weight": weight,
        "m.gate_proj.weight_scale_inv": scale,
        "m.norm.weight": torch.ones(8, dtype=torch.bfloat16),
    }
    dequant_fp8_state_dict(state_dict, dtype=torch.bfloat16, block_size=128)

    assert "m.gate_proj.weight_scale_inv" not in state_dict
    assert state_dict["m.gate_proj.weight"].dtype == torch.bfloat16
    assert state_dict["m.norm.weight"].numel() == 8
    assert not any("weight_scale_inv" in k for k in state_dict)


def test_glm_fp8_selective_dequant_keeps_experts_fp8():
    from moe_infinity.utils.fp8 import (
        EXPERT_SCALE_KEY_RE,
        dequant_fp8_state_dict,
        stack_expert_scales,
    )

    expert_w = torch.randn(256, 384).to(torch.float8_e4m3fn)
    expert_s = torch.rand(2, 3, dtype=torch.float32) + 0.5
    dense_w = torch.randn(256, 384).to(torch.float8_e4m3fn)
    dense_s = torch.rand(2, 3, dtype=torch.float32) + 0.5
    state_dict = {
        "model.layers.3.mlp.experts.7.gate_proj.weight": expert_w,
        "model.layers.3.mlp.experts.7.gate_proj.weight_scale_inv": expert_s,
        "model.layers.0.mlp.gate_proj.weight": dense_w,
        "model.layers.0.mlp.gate_proj.weight_scale_inv": dense_s,
    }
    kept = dequant_fp8_state_dict(
        state_dict,
        dtype=torch.bfloat16,
        block_size=128,
        keep_fp8=EXPERT_SCALE_KEY_RE.match,
    )

    expert_key = "model.layers.3.mlp.experts.7.gate_proj.weight"
    assert state_dict[expert_key].dtype == torch.float8_e4m3fn
    assert expert_key in kept
    assert torch.equal(kept[expert_key], expert_s)
    assert "model.layers.0.mlp.gate_proj.weight_scale_inv" not in state_dict
    assert state_dict["model.layers.0.mlp.gate_proj.weight"].dtype == (
        torch.bfloat16
    )
    assert not any("weight_scale_inv" in k for k in state_dict)

    stacked = stack_expert_scales(kept)
    assert list(stacked.keys()) == [3]
    assert stacked[3]["gate"].shape == (1, 2, 3)
    assert torch.equal(stacked[3]["gate"][0], expert_s)


def test_glm_cpp_dequant_matches_python():
    _store = pytest.importorskip("moe_infinity._store")
    if isinstance(_store, MagicMock):
        pytest.skip(
            "moe_infinity._store is stubbed by conftest; native "
            "extension not built"
        )
    from moe_infinity.utils.fp8 import dequant_fp8_blockwise

    torch.manual_seed(0)
    shapes = [
        ((256, 384), (2, 3)),
        ((2048, 6144), (16, 48)),  # GLM-5.2 gate/up proj
        ((6144, 2048), (48, 16)),  # GLM-5.2 down proj
    ]
    for w_shape, s_shape in shapes:
        weight = torch.randn(*w_shape).to(torch.float8_e4m3fn)
        scale = torch.rand(*s_shape, dtype=torch.float32) + 0.5
        ref = dequant_fp8_blockwise(weight, scale, dtype=torch.bfloat16)
        out = _store.dequant_fp8_blockwise(weight, scale)
        assert out.dtype == torch.bfloat16
        assert torch.equal(out, ref)
        if torch.cuda.is_available():
            out_gpu = _store.dequant_fp8_blockwise(weight.cuda(), scale.cuda())
            assert out_gpu.dtype == torch.bfloat16
            assert torch.equal(out_gpu.cpu(), ref)


def test_glm_routing_parity_vs_hf():
    pytest.importorskip("transformers.models.glm_moe_dsa")
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaMoE,
    )

    try:
        from transformers import GlmMoeDsaConfig
    except ImportError:
        from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
            GlmMoeDsaConfig,
        )

    from moe_infinity.models import SyncGlmMoeDsaMoEBlock

    if not hasattr(SyncGlmMoeDsaMoEBlock, "route_tokens_to_experts"):
        pytest.skip(
            "canonical SyncGlmMoeDsaMoEBlock delegates routing to HF and has no "
            "standalone route_tokens_to_experts; parity covered by "
            "test_glm_routing.py::test_routing_parity"
        )

    torch.manual_seed(0)
    config = GlmMoeDsaConfig(
        hidden_size=128,
        moe_intermediate_size=64,
        intermediate_size=256,
        n_routed_experts=8,
        num_local_experts=8,
        num_experts_per_tok=4,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        hidden_act="silu",
        num_hidden_layers=4,
    )
    hf = GlmMoeDsaMoE(config).eval()
    sync = SyncGlmMoeDsaMoEBlock(config).eval()
    sync.gate.load_state_dict(hf.gate.state_dict())
    with torch.no_grad():
        bias = torch.randn(config.n_routed_experts)
        hf.gate.e_score_correction_bias.copy_(bias)
        sync.gate.e_score_correction_bias.copy_(bias)

    router_logits = torch.randn(32, config.n_routed_experts)
    hf_idx, hf_w = hf.route_tokens_to_experts(router_logits)
    our_idx, our_w = sync.route_tokens_to_experts(router_logits)

    hf_sorted = torch.sort(hf_idx, dim=-1).values
    our_sorted = torch.sort(our_idx, dim=-1).values
    assert torch.equal(hf_sorted, our_sorted)

    hf_ws = torch.gather(hf_w, 1, torch.argsort(hf_idx, dim=-1))
    our_ws = torch.gather(our_w, 1, torch.argsort(our_idx, dim=-1))
    assert torch.allclose(hf_ws, our_ws, rtol=1e-4, atol=1e-5)
