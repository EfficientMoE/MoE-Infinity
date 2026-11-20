"""MoE.generate on the Hugging Face path moves its inputs to the model's device.

The HF path (use_native_engine=False, or a batch the native engine does not
take) used to forward ``input_ids`` untouched, so CPU ids made transformers'
rotary embedding mix a CPU ``position_ids`` with CUDA weights. The native path
moves the prompt to the input-embedding device and returns the result on the
caller's device; the HF path does the same.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("torch")

import torch


class _RecordingModel:
    """Stands in for the HF model: records the device of every tensor."""

    def __init__(self, output_device="cpu"):
        self.seen: dict[str, torch.device] = {}
        self.output_device = output_device
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    def generate(self, input_ids, **kwargs):
        self.seen["input_ids"] = input_ids.device
        for name, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                self.seen[name] = value.device
        # Data-free output (inputs may be on "meta"): prompt + 2 new tokens.
        return torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1] + 2,
            dtype=input_ids.dtype,
            device=self.output_device,
        )


def _make_hf_moe(model, target_device):
    from moe_infinity.entrypoints.big_modeling import MoE

    instance = MoE.__new__(MoE)
    instance.model = model
    instance._configure_hook = MagicMock()
    instance._cached_past_key_values = None
    instance.max_seq_length = 1024
    instance.use_native_engine = False
    instance._native_generation_engine = None
    instance._resolve_native_input_device = lambda: target_device
    return instance


def test_cpu_inputs_are_moved_to_the_model_device_and_output_returns_to_caller():
    # "meta" needs no GPU: any device other than the caller's shows the move.
    target = torch.device("meta")
    model = _RecordingModel()
    moe = _make_hf_moe(model, target)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    out = moe.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=2,
        pad_token_id=0,
    )

    assert model.eval_called
    assert model.seen["input_ids"] == target
    assert model.seen["attention_mask"] == target
    assert set(model.seen) == {
        "input_ids",
        "attention_mask",
    }  # no other tensors
    assert out.shape == (1, 5)
    # Same as the native path: the result is on the device the ids came from.
    assert out.device == input_ids.device


def test_non_tensor_kwargs_pass_through_unchanged():
    model = _RecordingModel()
    seen_kwargs = {}
    original = model.generate

    def generate(input_ids, **kwargs):
        seen_kwargs.update(kwargs)
        return original(input_ids)

    model.generate = generate
    moe = _make_hf_moe(model, torch.device("cpu"))
    generator = torch.Generator().manual_seed(1)

    moe.generate(
        torch.tensor([[1, 2]], dtype=torch.long),
        max_new_tokens=3,
        do_sample=False,
        generator=generator,
    )

    assert seen_kwargs["max_new_tokens"] == 3
    assert seen_kwargs["do_sample"] is False
    assert seen_kwargs["generator"] is generator


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("ids_device", ["cpu", "cuda:0"])
def test_resolver_target_is_the_embedding_device(ids_device):
    """Uses the real resolver: the target is the input embedding's device."""
    from moe_infinity.entrypoints.big_modeling import MoE

    embed = torch.nn.Embedding(8, 4).to("cuda:0")
    model = _RecordingModel(output_device="cuda:0")
    model.get_input_embeddings = lambda: embed
    moe = _make_hf_moe(model, None)
    del moe._resolve_native_input_device  # back to the class method

    out = moe.generate(
        torch.tensor([[1, 2, 3]], dtype=torch.long, device=ids_device),
        attention_mask=torch.ones(1, 3, dtype=torch.long, device=ids_device),
        max_new_tokens=2,
    )

    assert model.seen["input_ids"] == torch.device("cuda:0")
    assert model.seen["attention_mask"] == torch.device("cuda:0")
    assert out.device == torch.device(ids_device)
