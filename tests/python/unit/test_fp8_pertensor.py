# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import torch

from moe_infinity.runtime.model_offload import (
    _dequant_fp8_pertensor,
    _expand_pertensor_scale_to_blocks,
    _identify_fp8_pertensor_pairs,
    _split_fp8_pertensor_state_dict,
)


def test_identify_pertensor_pairs_matches_weight_scale_suffix():
    keys = [
        "model.layers.2.mlp.experts.0.gate_proj.weight",
        "model.layers.2.mlp.experts.0.gate_proj.weight_scale",
        "model.layers.2.mlp.experts.0.gate_proj.input_scale",
        "model.layers.2.self_attn.q_proj.weight",
        "model.layers.2.self_attn.q_proj.weight_scale",
        # blockwise key must NOT be picked up by the per-tensor identifier
        "model.layers.2.mlp.experts.1.up_proj.weight",
        "model.layers.2.mlp.experts.1.up_proj.weight_scale_inv",
        # orphan scale without a weight
        "model.layers.3.mlp.experts.0.up_proj.weight_scale",
    ]
    pairs = _identify_fp8_pertensor_pairs(keys)
    assert pairs == [
        (
            "model.layers.2.mlp.experts.0.gate_proj.weight",
            "model.layers.2.mlp.experts.0.gate_proj.weight_scale",
        ),
        (
            "model.layers.2.self_attn.q_proj.weight",
            "model.layers.2.self_attn.q_proj.weight_scale",
        ),
    ]


def test_expand_scalar_scale_to_block_grid():
    scale = torch.tensor(0.5)
    grid = _expand_pertensor_scale_to_blocks((1536, 4096), scale)
    assert grid.shape == (12, 32)
    assert grid.dtype == torch.float32
    assert torch.all(grid == 0.5)
    # non-multiple dims round up
    assert _expand_pertensor_scale_to_blocks((130, 100), scale).shape == (2, 1)


def test_dequant_fp8_pertensor_roundtrip():
    w_ref = torch.tensor([[0.5, -1.0], [2.0, 0.25]], dtype=torch.float32)
    scale = torch.tensor(2.0)
    w_fp8 = (w_ref / scale).to(torch.float8_e4m3fn)
    out = _dequant_fp8_pertensor(w_fp8, scale, dtype=torch.bfloat16)
    assert out.dtype == torch.bfloat16
    assert torch.allclose(out.float(), w_ref, atol=1e-2)


def test_split_routes_experts_to_scales_and_dequants_resident():
    expert_w = torch.ones(256, 128).to(torch.float8_e4m3fn)
    attn_w = (torch.ones(128, 128) * 0.5).to(torch.float8_e4m3fn)
    state_dict = {
        "model.layers.2.mlp.experts.0.gate_proj.weight": expert_w,
        "model.layers.2.mlp.experts.0.gate_proj.weight_scale": torch.tensor(
            2.0
        ),
        "model.layers.2.mlp.experts.0.gate_proj.input_scale": torch.tensor(1.0),
        "model.layers.2.self_attn.q_proj.weight": attn_w,
        "model.layers.2.self_attn.q_proj.weight_scale": torch.tensor(4.0),
        "model.layers.2.self_attn.q_proj.input_scale": torch.tensor(1.0),
        "model.norm.weight": torch.ones(128, dtype=torch.bfloat16),
    }
    scales = {}
    _split_fp8_pertensor_state_dict(state_dict, scales)

    # routed expert: weight stays FP8, scale expanded to the block grid
    k = "model.layers.2.mlp.experts.0.gate_proj.weight"
    assert state_dict[k].dtype == torch.float8_e4m3fn
    assert scales[k].shape == (2, 1)
    assert torch.all(scales[k] == 2.0)

    # resident weight: dequantized to BF16 in place
    q = state_dict["model.layers.2.self_attn.q_proj.weight"]
    assert q.dtype == torch.bfloat16
    assert torch.allclose(q.float(), torch.full((128, 128), 2.0), atol=1e-2)

    # every *_scale / input_scale key consumed; untouched keys survive
    assert not [key for key in state_dict if key.endswith("_scale")]
    assert not [key for key in state_dict if key.endswith(".input_scale")]
    assert "model.norm.weight" in state_dict
