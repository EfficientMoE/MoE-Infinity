"""CPU, deterministic checks for the OLMoE paged-attention shim.

The paged backend is replaced by a causal SDPA over the packed tokens, so
these tests pin down what the shim itself is responsible for: the OLMoE
pre-attention math (whole-projection q/k norms before the head split,
``clip_qkv``), the padded-token packing, the fallback to the stock
attention when no paged context is installed, and the KV-cache spec.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")

from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
from transformers.models.olmoe.modeling_olmoe import (
    OlmoeAttention,
    OlmoeRotaryEmbedding,
)

from moe_infinity.models.olmoe_paged_attention import OlmoePagedAttention


def _config(**overrides) -> OlmoeConfig:
    params = dict(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        rms_norm_eps=1e-5,
        clip_qkv=None,
        attention_bias=False,
    )
    params.update(overrides)
    config = OlmoeConfig(**params)
    config._attn_implementation = "eager"
    return config


class _CausalSdpaBackend:
    """Stand-in for PagedAttentionBackend: causal attention per sequence over
    packed [tokens, heads, head_dim] inputs, GQA expanded to the query heads.
    Records what it was called with so packing can be asserted."""

    def __init__(self, query_lengths: list[int]):
        self.query_lengths = query_lengths
        self.calls: list[dict] = []

    def forward(self, query, key, value, attention_metadata=None, scale=None, layer_idx=0):
        self.calls.append(
            {"q": query.shape, "k": key.shape, "v": value.shape, "layer_idx": layer_idx}
        )
        groups = query.shape[1] // key.shape[1]
        outputs = []
        start = 0
        for length in self.query_lengths:
            q = query[start : start + length].transpose(0, 1)  # [H, T, D]
            k = key[start : start + length].repeat_interleave(groups, dim=1).transpose(0, 1)
            v = value[start : start + length].repeat_interleave(groups, dim=1).transpose(0, 1)
            scores = (q @ k.transpose(-1, -2)) * scale
            mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
            outputs.append((torch.softmax(scores, dim=-1) @ v).transpose(0, 1))
            start += length
        return torch.cat(outputs, dim=0)


def _metadata(query_lengths: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        lengths=SimpleNamespace(query_lengths=torch.tensor(query_lengths, dtype=torch.int32))
    )


def _causal_mask(length: int) -> torch.Tensor:
    mask = torch.zeros(1, 1, length, length)
    mask[..., torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)] = float("-inf")
    return mask


def _rope(config: OlmoeConfig, hidden: torch.Tensor, positions: torch.Tensor):
    return OlmoeRotaryEmbedding(config)(hidden, positions)


@pytest.fixture(autouse=True)
def _clear_context():
    OlmoePagedAttention.clear_paged_context()
    yield
    OlmoePagedAttention.clear_paged_context()


@pytest.mark.parametrize("clip_qkv", [None, 0.5])
def test_paged_forward_matches_stock_attention_single_sequence(clip_qkv) -> None:
    torch.manual_seed(0)
    config = _config(clip_qkv=clip_qkv)
    shim = OlmoePagedAttention(config, layer_idx=0).eval()
    stock = OlmoeAttention(config, layer_idx=0).eval()
    stock.load_state_dict(shim.state_dict())

    length = 5
    hidden = torch.randn(1, length, config.hidden_size)
    positions = torch.arange(length).unsqueeze(0)
    cos, sin = _rope(config, hidden, positions)

    with torch.no_grad():
        expected, _ = stock(
            hidden_states=hidden,
            position_embeddings=(cos, sin),
            attention_mask=_causal_mask(length),
        )
        backend = _CausalSdpaBackend([length])
        OlmoePagedAttention.set_paged_context(backend, _metadata([length]))
        actual, weights = shim(
            hidden_states=hidden,
            position_embeddings=(cos, sin),
            attention_mask=None,
        )

    assert weights is None
    assert len(backend.calls) == 1
    assert backend.calls[0]["layer_idx"] == 0
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_padded_batch_is_packed_and_scattered_back() -> None:
    torch.manual_seed(1)
    config = _config()
    shim = OlmoePagedAttention(config, layer_idx=3).eval()
    lengths = [3, 2]
    q_len = max(lengths)
    hidden = torch.randn(len(lengths), q_len, config.hidden_size)
    positions = torch.arange(q_len).unsqueeze(0).expand(len(lengths), -1)
    cos, sin = _rope(config, hidden, positions)

    backend = _CausalSdpaBackend(lengths)
    OlmoePagedAttention.set_paged_context(backend, _metadata(lengths))
    with torch.no_grad():
        out, _ = shim(hidden_states=hidden, position_embeddings=(cos, sin), attention_mask=None)

    # the backend saw only the 5 valid tokens, not the 6 padded slots
    assert backend.calls[0]["q"][0] == sum(lengths)
    assert backend.calls[0]["layer_idx"] == 3
    assert out.shape == (len(lengths), q_len, config.hidden_size)
    # padded slot of the shorter row: attention output zero -> o_proj(0) = 0 (no bias)
    torch.testing.assert_close(out[1, 2], torch.zeros(config.hidden_size))

    # each valid row equals the same sequence run alone (unpadded)
    for row, length in enumerate(lengths):
        solo_backend = _CausalSdpaBackend([length])
        OlmoePagedAttention.set_paged_context(solo_backend, _metadata([length]))
        with torch.no_grad():
            solo, _ = shim(
                hidden_states=hidden[row : row + 1, :length],
                position_embeddings=(cos[row : row + 1, :length], sin[row : row + 1, :length]),
                attention_mask=None,
            )
        torch.testing.assert_close(out[row, :length], solo[0], rtol=1e-5, atol=1e-5)


def test_without_paged_context_defers_to_stock_forward() -> None:
    torch.manual_seed(2)
    config = _config(clip_qkv=1.0)
    shim = OlmoePagedAttention(config, layer_idx=0).eval()
    stock = OlmoeAttention(config, layer_idx=0).eval()
    stock.load_state_dict(shim.state_dict())
    length = 4
    hidden = torch.randn(1, length, config.hidden_size)
    cos, sin = _rope(config, hidden, torch.arange(length).unsqueeze(0))
    with torch.no_grad():
        a, _ = shim(hidden_states=hidden, position_embeddings=(cos, sin), attention_mask=_causal_mask(length))
        b, _ = stock(hidden_states=hidden, position_embeddings=(cos, sin), attention_mask=_causal_mask(length))
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_kv_cache_spec_follows_olmoe_head_geometry() -> None:
    config = _config()
    spec = OlmoePagedAttention.get_kv_cache_spec_for_config(config)
    assert spec == {"num_kv_heads": 2, "head_dim": 8}
    assert spec["head_dim"] == config.hidden_size // config.num_attention_heads
    assert math.isclose(OlmoePagedAttention(config, layer_idx=0).scaling, 8**-0.5)


def test_runner_match_set_and_registry_know_the_shim() -> None:
    from moe_infinity.serving import model_runner as runner_mod

    source = open(runner_mod.__file__, encoding="utf-8").read()
    assert '"OlmoePagedAttention"' in source, "serving runner must match the OLMoE shim by name"
