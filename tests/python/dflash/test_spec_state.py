from __future__ import annotations

import pytest

from moe_infinity.serving.spec_state import SpecDecodeState


def test_initial_cached_len_defaults_to_prompt_len() -> None:
    state = SpecDecodeState(seq_id=0, prompt_len=5)
    assert state.cached_len == 5
    assert state.emitted_len == 0
    assert state.invariant_holds()


def test_full_accept_advances_by_block_len() -> None:
    state = SpecDecodeState(seq_id=0, prompt_len=5)
    acc = state.record_verify(block_len=9, committed=9)
    assert acc.committed == 9
    assert acc.truncate_target == 14
    assert state.cached_len == 14
    assert state.emitted_len == 9
    assert state.invariant_holds()


def test_partial_accept_truncate_target_drops_rejected_tail() -> None:
    state = SpecDecodeState(seq_id=0, prompt_len=10)
    acc = state.record_verify(block_len=9, committed=3)
    assert acc.truncate_target == 13
    assert state.cached_len == 13
    assert state.emitted_len == 3
    assert state.invariant_holds()


def test_single_token_commit() -> None:
    state = SpecDecodeState(seq_id=0, prompt_len=4)
    acc = state.record_verify(block_len=9, committed=1)
    assert acc.truncate_target == 5
    assert state.invariant_holds()


def test_multi_step_accumulates() -> None:
    state = SpecDecodeState(seq_id=0, prompt_len=8)
    state.record_verify(block_len=9, committed=4)
    state.record_verify(block_len=9, committed=2)
    acc = state.record_verify(block_len=9, committed=9)
    assert state.emitted_len == 15
    assert state.cached_len == 23
    assert acc.truncate_target == 23
    assert state.invariant_holds()


def test_committed_out_of_range_raises() -> None:
    state = SpecDecodeState(seq_id=0, prompt_len=5)
    with pytest.raises(ValueError, match="committed must be"):
        state.record_verify(block_len=9, committed=0)
    with pytest.raises(ValueError, match="committed must be"):
        state.record_verify(block_len=9, committed=10)


def test_bad_block_len_raises() -> None:
    state = SpecDecodeState(seq_id=0, prompt_len=5)
    with pytest.raises(ValueError, match="block_len must be"):
        state.record_verify(block_len=0, committed=1)


def test_negative_prompt_len_raises() -> None:
    with pytest.raises(ValueError, match="prompt_len must be"):
        SpecDecodeState(seq_id=0, prompt_len=-1)


def test_truncate_target_composes_with_kv_truncate() -> None:
    import torch

    from moe_infinity.serving.kv_cache import PagedKVCache

    cache = PagedKVCache(
        num_blocks=32,
        block_size=4,
        num_layers=1,
        num_heads=1,
        head_dim=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    prompt_len = 6
    cache.allocate_sequence(0, prompt_len)
    state = SpecDecodeState(seq_id=0, prompt_len=prompt_len)

    block_len, committed = 9, 3
    cache.append_tokens(0, block_len)
    acc = state.record_verify(block_len=block_len, committed=committed)
    cache.truncate_tokens(0, acc.truncate_target)

    assert (
        cache._sequence_tables[0].num_computed_tokens()
        == prompt_len + committed
    )
    assert acc.truncate_target == prompt_len + committed
    assert state.invariant_holds()
