from __future__ import annotations

import pytest
import torch

from moe_infinity.serving.kv_cache import PagedKVCache

BLOCK_SIZE = 4
NUM_BLOCKS = 16


def _make_cache() -> PagedKVCache:
    return PagedKVCache(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        num_layers=2,
        num_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def _blocks_for(num_tokens: int) -> int:
    return (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE


def _num_tokens(cache: PagedKVCache, seq_id: int) -> int:
    return cache._sequence_tables[seq_id].num_computed_tokens()


def _num_blocks(cache: PagedKVCache, seq_id: int) -> int:
    return len(cache.get_block_table(seq_id))


def test_partial_block_shrink_frees_no_blocks() -> None:
    cache = _make_cache()
    cache.allocate_sequence(0, 10)
    free_before = cache.block_allocator.num_free_blocks
    assert _num_blocks(cache, 0) == 3

    cache.truncate_tokens(0, 9)

    assert _num_tokens(cache, 0) == 9
    assert _num_blocks(cache, 0) == 3
    assert cache.block_allocator.num_free_blocks == free_before


def test_full_block_shrink_frees_tail_blocks() -> None:
    cache = _make_cache()
    cache.allocate_sequence(0, 10)
    kept_head = cache.get_block_table(0)[:1]
    free_before = cache.block_allocator.num_free_blocks

    cache.truncate_tokens(0, 4)

    assert _num_tokens(cache, 0) == 4
    assert _num_blocks(cache, 0) == 1
    assert cache.get_block_table(0) == kept_head
    assert cache.block_allocator.num_free_blocks == free_before + 2


def test_shrink_to_zero_returns_all_blocks() -> None:
    cache = _make_cache()
    cache.allocate_sequence(0, 8)
    cache.truncate_tokens(0, 0)

    assert _num_tokens(cache, 0) == 0
    assert _num_blocks(cache, 0) == 0
    assert cache.block_allocator.num_free_blocks == NUM_BLOCKS


def test_noop_when_length_unchanged() -> None:
    cache = _make_cache()
    cache.allocate_sequence(0, 8)
    blocks_before = cache.get_block_table(0)
    free_before = cache.block_allocator.num_free_blocks

    cache.truncate_tokens(0, 8)

    assert cache.get_block_table(0) == blocks_before
    assert cache.block_allocator.num_free_blocks == free_before
    assert _num_tokens(cache, 0) == 8


def test_grow_raises() -> None:
    cache = _make_cache()
    cache.allocate_sequence(0, 8)
    with pytest.raises(ValueError, match="cannot grow"):
        cache.truncate_tokens(0, 9)


def test_negative_raises() -> None:
    cache = _make_cache()
    cache.allocate_sequence(0, 8)
    with pytest.raises(ValueError, match=">= 0"):
        cache.truncate_tokens(0, -1)


def test_unknown_sequence_raises() -> None:
    cache = _make_cache()
    with pytest.raises(KeyError):
        cache.truncate_tokens(99, 0)


def test_other_sequences_unaffected() -> None:
    cache = _make_cache()
    cache.allocate_sequence(0, 8)
    cache.allocate_sequence(1, 6)
    seq1_blocks = cache.get_block_table(1)

    cache.truncate_tokens(0, 2)

    assert cache.get_block_table(1) == seq1_blocks
    assert _num_tokens(cache, 1) == 6
    assert _num_tokens(cache, 0) == 2


def test_truncate_while_swapped_out_updates_cpu_state() -> None:
    cache = _make_cache()
    cache.allocate_sequence(0, 8)
    cache.swap_out(0)
    cache.free_gpu_blocks(0)
    free_after_free = cache.block_allocator.num_free_blocks

    cache.truncate_tokens(0, 4)

    assert cache._swapped_num_tokens[0] == 4
    assert int(cache._swapped_cpu_buffers[0].shape[1]) == 1
    assert cache.block_allocator.num_free_blocks == free_after_free

    cache.swap_in(0)
    assert _num_tokens(cache, 0) == 4
    assert _num_blocks(cache, 0) == 1


def test_repeated_truncate_and_free_reconciles_pool() -> None:
    cache = _make_cache()
    cache.allocate_sequence(0, 12)
    cache.truncate_tokens(0, 5)
    assert _num_blocks(cache, 0) == 2
    cache.truncate_tokens(0, 1)
    assert _num_blocks(cache, 0) == 1
    cache.free_sequence(0)
    assert cache.block_allocator.num_free_blocks == NUM_BLOCKS
