# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false

from moe_infinity.memory.block_pool import BlockPool, hash_block_tokens


def test_allocate_free_cycle() -> None:
    pool = BlockPool(num_blocks=10)
    blocks = [pool.allocate_block() for _ in range(10)]
    assert all(b is not None for b in blocks)
    assert pool.num_free_blocks() == 0

    for block in blocks:
        pool.free_block(block)

    assert pool.num_free_blocks() == 10


def test_prefix_cache_hit() -> None:
    pool = BlockPool(num_blocks=5)
    block = pool.allocate_block()
    assert block is not None

    block_hash = hash_block_tokens(0, (1, 2, 3))
    pool.cache_full_block(block, block_hash)
    pool.free_block(block)

    reused = pool.get_cached_block(block_hash)
    assert reused is block
    assert reused.ref_cnt == 1


def test_over_allocation_returns_none() -> None:
    pool = BlockPool(num_blocks=3)
    b1, b2, b3 = [pool.allocate_block() for _ in range(3)]
    assert b1 is not None and b2 is not None and b3 is not None

    b4 = pool.allocate_block()
    assert b4 is None


def test_ref_counting_prevents_eviction() -> None:
    pool = BlockPool(num_blocks=3)
    block = pool.allocate_block()
    assert block is not None

    block_hash = hash_block_tokens(0, (1, 2))
    pool.cache_full_block(block, block_hash)

    assert pool.get_cached_block(block_hash) is block


def test_hash_block_tokens() -> None:
    h1 = hash_block_tokens(0, (1, 2, 3))
    h2 = hash_block_tokens(0, (1, 2, 3))
    h3 = hash_block_tokens(0, (1, 2, 4))

    assert h1 == h2
    assert h1 != h3
