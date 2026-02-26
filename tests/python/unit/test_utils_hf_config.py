from types import SimpleNamespace
from typing import cast

import pytest
import torch
from transformers import PretrainedConfig

from moe_infinity.utils.hf_config import (
    parse_expert_dtype,
    parse_expert_id,
    parse_moe_param,
)


def _cfg(**kwargs) -> PretrainedConfig:
    return cast(PretrainedConfig, SimpleNamespace(**kwargs))


def test_parse_expert_dtype_supported():
    cfg = _cfg(torch_dtype=torch.bfloat16)
    assert parse_expert_dtype(cfg) == 0

    cfg = _cfg(torch_dtype=torch.float32)
    assert parse_expert_dtype(cfg) == 1

    cfg = _cfg(torch_dtype=torch.float16)
    assert parse_expert_dtype(cfg) == 2

    if hasattr(torch, "float8_e4m3fn"):
        cfg = _cfg(torch_dtype=torch.float8_e4m3fn)
        assert parse_expert_dtype(cfg) == 3


def test_parse_expert_dtype_unsupported():
    cfg = _cfg(torch_dtype=torch.int32)
    with pytest.raises(AssertionError):
        parse_expert_dtype(cfg)


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


def test_parse_moe_param_minimax_m2():
    minimax = _cfg(
        architectures=["MiniMaxM2ForCausalLM"],
        model_type="minimax_m2",
        num_hidden_layers=62,
        num_local_experts=256,
    )
    assert parse_moe_param(minimax) == (62, 256, 0)


def test_parse_moe_param_kimi_vl():
    text_cfg = _cfg(num_hidden_layers=27, n_routed_experts=64)
    kimi = _cfg(
        architectures=["KimiVLForConditionalGeneration"],
        model_type="kimi_vl",
        text_config=text_cfg,
    )
    assert parse_moe_param(kimi) == (27, 64, 0)


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


def test_parse_expert_id_minimax_m2():
    minimax = _cfg(
        architectures=["MiniMaxM2ForCausalLM"],
        model_type="minimax_m2",
        num_hidden_layers=62,
        num_local_experts=256,
    )
    layer_id, expert_id = parse_expert_id(
        "model.layers.5.feed_forward.experts.7.w1.weight",
        minimax,
    )
    assert (layer_id, expert_id) == (5, 7)


def test_parse_expert_id_kimi_vl():
    text_cfg = _cfg(num_hidden_layers=27, n_routed_experts=64)
    kimi = _cfg(
        architectures=["KimiVLForConditionalGeneration"],
        model_type="kimi_vl",
        text_config=text_cfg,
    )
    layer_id, expert_id = parse_expert_id(
        "language_model.model.layers.3.mlp.experts.10.gate_proj.weight",
        kimi,
    )
    assert (layer_id, expert_id) == (3, 10)


def test_parse_expert_id_grok_and_deepseek():
    grok = _cfg(
        architectures=["Grok"],
        num_hidden_layers=3,
        num_experts=2,
    )
    layer_id, expert_id = parse_expert_id(
        "model.layers.1.moe_block.experts.0.linear_1.weight",
        grok,
    )
    assert (layer_id, expert_id) == (1, 0)

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
