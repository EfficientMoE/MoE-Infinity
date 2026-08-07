from __future__ import annotations

import torch

from moe_infinity.serving.kv_cache import PagedKVCache
from moe_infinity.serving.spec_state import SpecDecodeState
from moe_infinity.serving.spec_verify import apply_verify_step
from moe_infinity.spec_decode._dflash_ops import (
    acceptance_length,
    committed_tokens,
)

BLOCK_SIZE = 4


def _make_cache(prompt_len: int) -> PagedKVCache:
    cache = PagedKVCache(
        num_blocks=64,
        block_size=4,
        num_layers=1,
        num_heads=1,
        head_dim=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    cache.allocate_sequence(0, prompt_len)
    return cache


def _step(cache: PagedKVCache, state: SpecDecodeState, block, posterior):
    cache.append_tokens(0, BLOCK_SIZE)
    return apply_verify_step(
        kv_cache=cache,
        seq_id=0,
        state=state,
        block=block,
        posterior=posterior,
        block_size=BLOCK_SIZE,
    )


def test_partial_accept_matches_canonical_ops() -> None:
    block = torch.tensor([[10, 11, 12, 99]])
    posterior = torch.tensor([[11, 12, 50, 7]])
    expected_accept = acceptance_length(block, posterior)
    expected = committed_tokens(block, posterior, expected_accept)

    cache = _make_cache(6)
    state = SpecDecodeState(seq_id=0, prompt_len=6)
    res = _step(cache, state, block, posterior)

    assert res.accept == 2 == expected_accept
    assert res.emitted_tokens == [int(t) for t in expected.emitted[0].tolist()]
    assert res.emitted_tokens == [11, 12, 50]
    assert res.next_anchor == 50
    assert res.cache_committed == 3
    assert cache._sequence_tables[0].num_computed_tokens() == 6 + 3
    assert res.cached_len == 9


def test_accept_zero_emits_only_bonus() -> None:
    block = torch.tensor([[10, 99, 98, 97]])
    posterior = torch.tensor([[50, 1, 2, 3]])
    cache = _make_cache(5)
    state = SpecDecodeState(seq_id=0, prompt_len=5)
    res = _step(cache, state, block, posterior)

    assert res.accept == 0
    assert res.emitted_tokens == [50]
    assert res.next_anchor == 50
    assert res.cache_committed == 1
    assert cache._sequence_tables[0].num_computed_tokens() == 6


def test_full_accept_keeps_whole_block() -> None:
    block = torch.tensor([[10, 11, 12, 13]])
    posterior = torch.tensor([[11, 12, 13, 77]])
    cache = _make_cache(4)
    state = SpecDecodeState(seq_id=0, prompt_len=4)
    res = _step(cache, state, block, posterior)

    assert res.accept == 3
    assert res.emitted_tokens == [11, 12, 13, 77]
    assert res.next_anchor == 77
    assert res.cache_committed == 4
    assert cache._sequence_tables[0].num_computed_tokens() == 8


def test_multi_step_chains_anchor_and_accumulates_cache() -> None:
    cache = _make_cache(3)
    state = SpecDecodeState(seq_id=0, prompt_len=3)

    block1 = torch.tensor([[20, 21, 55, 54]])
    posterior1 = torch.tensor([[21, 30, 31, 32]])
    res1 = _step(cache, state, block1, posterior1)
    assert res1.accept == 1
    assert res1.next_anchor == 30
    assert cache._sequence_tables[0].num_computed_tokens() == 3 + 2

    block2 = torch.tensor([[res1.next_anchor, 41, 42, 43]])
    posterior2 = torch.tensor([[41, 42, 43, 9]])
    res2 = _step(cache, state, block2, posterior2)
    assert res2.accept == 3
    assert cache._sequence_tables[0].num_computed_tokens() == 5 + 4
    assert res2.cached_len == 9
    assert state.emitted_len == res1.cache_committed + res2.cache_committed
