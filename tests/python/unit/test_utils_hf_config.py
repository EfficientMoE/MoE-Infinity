from types import SimpleNamespace
from typing import cast

import pytest
import torch
from transformers import PretrainedConfig

from moe_infinity.utils.hf_config import (
    parse_expert_dtype,
    parse_expert_id,
    parse_moe_param,
    resolve_config_dtype,
)


def _cfg(**kwargs) -> PretrainedConfig:
    return cast(PretrainedConfig, cast(object, SimpleNamespace(**kwargs)))


def test_parse_expert_dtype_supported():
    cfg = _cfg(torch_dtype=torch.bfloat16)
    assert parse_expert_dtype(cfg) == 0

    cfg = _cfg(torch_dtype=torch.float32)
    assert parse_expert_dtype(cfg) == 1

    cfg = _cfg(torch_dtype=torch.float16)
    assert parse_expert_dtype(cfg) == 2


def test_parse_expert_dtype_unsupported():
    cfg = _cfg(torch_dtype=torch.int32)
    with pytest.raises(AssertionError):
        parse_expert_dtype(cfg)


def test_resolve_config_dtype_v5_shape():
    cfg = _cfg(dtype=torch.bfloat16)
    assert resolve_config_dtype(cfg) == torch.bfloat16


def test_resolve_config_dtype_v4_shape():
    cfg = _cfg(torch_dtype=torch.float16)
    assert resolve_config_dtype(cfg) == torch.float16


def test_resolve_config_dtype_prefers_new_name():
    cfg = _cfg(dtype=torch.float32, torch_dtype=torch.float16)
    assert resolve_config_dtype(cfg) == torch.float32


def test_resolve_config_dtype_missing_returns_none():
    cfg = _cfg(hidden_size=128)
    assert resolve_config_dtype(cfg) is None


def test_parse_expert_dtype_v5_shape():
    cfg = _cfg(dtype=torch.bfloat16)
    assert parse_expert_dtype(cfg) == 0

    cfg = _cfg(dtype=torch.float16)
    assert parse_expert_dtype(cfg) == 2


def test_parse_moe_param_nllb():
    nllb = _cfg(
        architectures=["NllbMoe"],
        encoder_layers=12,
        decoder_layers=6,
        encoder_sparse_step=2,
        decoder_sparse_step=3,
        num_experts=4,
    )
    assert parse_moe_param(nllb) == (8, 4, 6)


def test_parse_moe_param_mixtral_and_deepseek():
    mixtral = _cfg(
        architectures=["Mixtral"],
        num_hidden_layers=7,
        num_local_experts=6,
    )
    assert parse_moe_param(mixtral) == (7, 6, 0)

    deepseek = _cfg(
        architectures=["DeepSeek"],
        num_hidden_layers=9,
        n_routed_experts=12,
    )
    assert parse_moe_param(deepseek) == (9, 12, 0)


def test_parse_expert_id_mixtral():
    mixtral = _cfg(
        architectures=["Mixtral"],
        num_hidden_layers=4,
        num_local_experts=8,
    )
    layer_id, expert_id = parse_expert_id(
        "model.layers.2.block_sparse_moe.experts.5.w1.weight",
        mixtral,
    )
    assert (layer_id, expert_id) == (2, 5)


def test_parse_expert_id_deepseek():
    deepseek = _cfg(
        architectures=["Deepseek"],
        num_hidden_layers=3,
        n_routed_experts=4,
    )
    layer_id, expert_id = parse_expert_id(
        "model.layers.2.mlp.experts.3.gate_proj.weight",
        deepseek,
    )
    assert (layer_id, expert_id) == (2, 3)
