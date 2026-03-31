from __future__ import annotations

import threading

from moe_infinity.memory.block_pool import BlockPool
from moe_infinity.memory.cpu_block_cache import CPUBlockCache


def test_store_and_load() -> None:
    pool = BlockPool(num_blocks=10)
    cache = CPUBlockCache(cpu_pool=pool, capacity=5)

    cpu_id = cache.store(block_hash=12345, gpu_block_id=0)
    assert cpu_id is not None
    result = cache.load(block_hash=12345)
    assert result == cpu_id


def test_eviction_when_full() -> None:
    pool = BlockPool(num_blocks=5)
    cache = CPUBlockCache(cpu_pool=pool, capacity=3)

    for i in range(3):
        _ = cache.store(block_hash=i, gpu_block_id=i)

    _ = cache.store(block_hash=99, gpu_block_id=0)
    assert len(cache) <= 3
    assert cache.load(block_hash=0) is None
    assert cache.load(block_hash=99) is not None


def test_load_returns_none_for_missing() -> None:
    pool = BlockPool(num_blocks=5)
    cache = CPUBlockCache(cpu_pool=pool)
    assert cache.load(block_hash=999) is None


def test_invalidate() -> None:
    pool = BlockPool(num_blocks=10)
    cache = CPUBlockCache(cpu_pool=pool)

    cpu_id = cache.store(block_hash=42, gpu_block_id=0)
    assert cache.load(block_hash=42) == cpu_id

    cache.invalidate(block_hash=42)
    assert cache.load(block_hash=42) is None


def test_thread_safety() -> None:
    pool = BlockPool(num_blocks=64)
    cache = CPUBlockCache(cpu_pool=pool, capacity=16)
    errors: list[Exception] = []

    def worker(thread_id: int) -> None:
        try:
            for i in range(300):
                block_hash = (thread_id * 1000 + i) % 40
                _ = cache.store(block_hash=block_hash, gpu_block_id=i)
                _ = cache.load(block_hash=block_hash)
                if i % 17 == 0:
                    cache.invalidate(block_hash=(block_hash + 1) % 40)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(idx,)) for idx in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert len(cache) <= 16
