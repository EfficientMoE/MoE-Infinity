from __future__ import annotations

import os
import sys
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

sys.path.insert(0, os.path.dirname(__file__))

from fixtures_tiny import (  # noqa: E402
    build_tiny_drafter,
    make_tiny_drafter_config,
    plain_greedy_decode,
    set_determinism,
)

from moe_infinity.spec_decode import (  # noqa: E402
    DFlashSpeculator,
    read_dflash_config,
)
from moe_infinity.entrypoints.big_modeling import MoE  # noqa: E402


def _tiny_qwen35_target(seed: int = 0) -> Qwen3_5MoeForCausalLM:
    set_determinism(seed)
    config = Qwen3_5MoeTextConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
        eos_token_id=None,
        pad_token_id=0,
        bos_token_id=1,
        attn_implementation="eager",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        target = Qwen3_5MoeForCausalLM(config)
    return target.to(torch.float32).eval()


def _qwen35_shell() -> MoE:
    shell = MoE.__new__(MoE)
    shell.model = MagicMock()
    shell.model.config = SimpleNamespace(model_type="qwen3_5_moe")
    shell.model.generate.return_value = torch.tensor([[1, 2, 9]])
    shell.use_native_engine = True
    shell._native_generation_engine = MagicMock()
    shell._native_generation_engine.generate.return_value = SimpleNamespace(
        output_token_ids=[8]
    )
    shell._resolve_spec_strategy = MagicMock()
    shell._configure_hook = MagicMock()
    shell._cached_past_key_values = None
    shell.max_seq_length = 64
    return shell


def test_qwen35_spec_off_generate_stays_on_hf_path() -> None:
    shell = _qwen35_shell()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        actual = shell.generate(
            torch.tensor([[1, 2]]), do_sample=False, max_new_tokens=1
        )

    assert actual.tolist() == [[1, 2, 9]]
    shell.model.generate.assert_called_once()
    shell._native_generation_engine.generate.assert_not_called()


def test_qwen35_greedy_dflash_uses_native_path() -> None:
    shell = _qwen35_shell()
    draft = object()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        actual = shell.generate(
            torch.tensor([[1, 2]]),
            do_sample=False,
            max_new_tokens=1,
            speculative_draft=draft,
        )

    assert actual.tolist() == [[1, 2, 8]]
    shell._resolve_spec_strategy.assert_called_once_with(draft)
    shell._native_generation_engine.generate.assert_called_once()
    shell.model.generate.assert_not_called()


def test_qwen35_sampled_dflash_is_rejected() -> None:
    shell = _qwen35_shell()

    with warnings.catch_warnings(), pytest.raises(ValueError, match="greedy"):
        warnings.simplefilter("ignore", DeprecationWarning)
        shell.generate(
            torch.tensor([[1, 2]]),
            do_sample=True,
            temperature=0.7,
            speculative_draft=object(),
        )


def test_qwen35_hybrid_dflash_is_token_identical_to_plain_greedy() -> None:
    target = _tiny_qwen35_target(seed=1)
    drafter = build_tiny_drafter(
        target,
        seed=21,
        block_size=4,
        target_layer_ids=(0, 1, 2, 3),
    )
    config = read_dflash_config(
        make_tiny_drafter_config(
            target.config,
            block_size=4,
            target_layer_ids=(0, 1, 2, 3),
        )
    )
    spec = DFlashSpeculator.from_models(
        target, drafter, config=config, device="cpu"
    )
    prompt = torch.tensor([[3, 7, 11, 2, 5]], dtype=torch.long)

    plain = plain_greedy_decode(target, prompt, max_new_tokens=32)
    actual = spec.generate(prompt, max_new_tokens=32, temperature=0.0)

    assert actual.tolist() == plain.tolist()
