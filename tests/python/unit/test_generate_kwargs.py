from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("torch")

import torch

from moe_infinity.engine.types import SamplingParams


class _StubResult:
    output_token_ids = [42]


def _make_moe_with_engine_stub():
    from moe_infinity.entrypoints.big_modeling import MoE

    instance = MoE.__new__(MoE)
    instance.model = MagicMock()
    instance.model.eval = MagicMock()
    instance._configure_hook = MagicMock()
    instance._cached_past_key_values = None
    instance.max_seq_length = 1024
    instance.use_native_engine = True
    instance.tokenizer = MagicMock()
    instance.tokenizer.eos_token_id = 0
    instance.eos_token_id = 0
    instance._native_generation_engine = MagicMock()
    instance._native_generation_engine.generate.return_value = _StubResult()
    return instance


def _captured_sampling_params(moe):
    call = moe._native_generation_engine.generate.call_args
    assert call is not None, "engine.generate was not called"
    return call.kwargs["sampling_params"]


def test_do_sample_false_forces_argmax_temperature():
    moe = _make_moe_with_engine_stub()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    moe.generate(input_ids, max_new_tokens=4, do_sample=False)

    sp = _captured_sampling_params(moe)
    assert isinstance(sp, SamplingParams)
    assert sp.temperature == 0.0


def test_explicit_temperature_overrides_do_sample_default():
    moe = _make_moe_with_engine_stub()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    moe.generate(input_ids, max_new_tokens=4, temperature=0.7)

    sp = _captured_sampling_params(moe)
    assert sp.temperature == 0.7


def test_do_sample_true_uses_caller_temperature():
    moe = _make_moe_with_engine_stub()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    moe.generate(input_ids, max_new_tokens=4, do_sample=True, temperature=0.5)

    sp = _captured_sampling_params(moe)
    assert sp.temperature == 0.5


def test_default_temperature_is_one():
    moe = _make_moe_with_engine_stub()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    moe.generate(input_ids, max_new_tokens=4)

    sp = _captured_sampling_params(moe)
    assert sp.temperature == 1.0
