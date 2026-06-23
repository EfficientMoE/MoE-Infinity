from typing import cast

import torch

from moe_infinity.serving.kv_cache import PagedKVCache


def _make_kv_cache(num_blocks: int = 6) -> PagedKVCache:
    return PagedKVCache(
        num_blocks=num_blocks,
        block_size=4,
        num_layers=1,
        num_heads=1,
        head_dim=8,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )


def test_free_returns_blocks() -> None:
    kv_cache = _make_kv_cache(num_blocks=4)
    kv_cache.allocate_sequence(seq_id=1, num_tokens=8)

    before_free = kv_cache.block_allocator.num_free_blocks
    kv_cache.free_gpu_blocks(seq_id=1)
    after_free = kv_cache.block_allocator.num_free_blocks

    assert before_free == 2
    assert after_free == 4


def test_sequence_table_preserved_after_free() -> None:
    kv_cache = _make_kv_cache(num_blocks=4)
    kv_cache.allocate_sequence(seq_id=2, num_tokens=8)
    sequence_tables = cast(
        dict[int, object],
        getattr(kv_cache, "_sequence_tables"),
    )

    kv_cache.free_gpu_blocks(seq_id=2)

    assert 2 in sequence_tables
    assert kv_cache.get_block_table(2) == []


def test_free_idempotent() -> None:
    kv_cache = _make_kv_cache(num_blocks=4)
    kv_cache.allocate_sequence(seq_id=3, num_tokens=8)

    kv_cache.free_gpu_blocks(seq_id=3)
    after_first_free = kv_cache.block_allocator.num_free_blocks
    kv_cache.free_gpu_blocks(seq_id=3)
    after_second_free = kv_cache.block_allocator.num_free_blocks

    assert after_first_free == 4
    assert after_second_free == 4


def test_free_nonexistent_seq_is_noop() -> None:
    kv_cache = _make_kv_cache(num_blocks=4)
    kv_cache.allocate_sequence(seq_id=11, num_tokens=4)

    free_before_unknown = kv_cache.block_allocator.num_free_blocks
    kv_cache.free_gpu_blocks(seq_id=999)
    free_after_unknown = kv_cache.block_allocator.num_free_blocks

    assert free_after_unknown == free_before_unknown

    kv_cache.free_gpu_blocks(seq_id=11)
    assert kv_cache.block_allocator.num_free_blocks == 4
    assert kv_cache.get_block_table(11) == []


def test_swap_in_reallocates_after_free() -> None:
    kv_cache = _make_kv_cache(num_blocks=6)
    kv_cache.allocate_sequence(seq_id=55, num_tokens=8)
    initial_block_ids = kv_cache.get_block_table(55)
    assert len(initial_block_ids) == 2

    payload_shape = (
        kv_cache.num_layers,
        len(initial_block_ids),
        2,
        kv_cache.block_size,
        kv_cache.num_heads,
        kv_cache.head_dim,
    )
    payload = (
        torch.arange(
            int(torch.tensor(payload_shape).prod().item()),
            dtype=torch.float32,
        )
        .reshape(payload_shape)
        .to(dtype=kv_cache.dtype)
    )

    kv_tensor = kv_cache.get_kv_cache_tensors()
    kv_tensor[:, initial_block_ids, ...] = payload

    kv_cache.swap_out(seq_id=55)
    kv_tensor[:, initial_block_ids, ...] = torch.zeros_like(payload)

    kv_cache.free_gpu_blocks(seq_id=55)
    assert kv_cache.get_block_table(55) == []
    assert kv_cache.block_allocator.num_free_blocks == kv_cache.num_blocks

    kv_cache.swap_in(seq_id=55)
    restored_block_ids = kv_cache.get_block_table(55)
    assert len(restored_block_ids) == len(initial_block_ids)
    assert kv_cache.block_allocator.num_free_blocks == (
        kv_cache.num_blocks - len(initial_block_ids)
    )

    restored = kv_cache.get_kv_cache_tensors()[:, restored_block_ids, ...]
    assert torch.equal(restored, payload)
