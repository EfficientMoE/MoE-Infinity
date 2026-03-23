import sys
from pathlib import Path

import pytest

ROOT = str(Path(__file__).resolve().parents[3])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
_ = sys.modules.pop("moe_infinity", None)
_ = sys.modules.pop("moe_infinity.memory", None)


def test_import_block_pool():
    from moe_infinity.memory.kv_cache_manager import BlockPool

    assert BlockPool is not None


def test_import_kv_cache_manager():
    from moe_infinity.memory.kv_cache_manager import KVCacheManager

    assert KVCacheManager is not None


def test_import_memory_budget():
    from moe_infinity.memory.kv_cache_manager import MemoryBudget

    assert MemoryBudget is not None


def test_memory_budget_defaults():
    from moe_infinity.memory.kv_cache_manager import MemoryBudget

    mb = MemoryBudget()
    assert mb.expert_cache_ratio == 0.75
    assert mb.kv_cache_ratio == 0.0
    assert mb.total_gpu_memory_bytes > 0


def test_memory_budget_custom():
    from moe_infinity.memory.kv_cache_manager import MemoryBudget

    mb = MemoryBudget(
        expert_cache_ratio=0.3,
        kv_cache_ratio=0.5,
        total_gpu_memory_bytes=24 * 1024**3,
    )
    assert mb.expert_cache_ratio == 0.3
    assert mb.kv_cache_ratio == 0.5


def test_memory_budget_sum_check():
    from moe_infinity.memory.kv_cache_manager import MemoryBudget

    with pytest.raises(ValueError):
        _ = MemoryBudget(expert_cache_ratio=0.7, kv_cache_ratio=0.5)


def test_block_pool_creation():
    from moe_infinity.memory.kv_cache_manager import BlockPool

    pool = BlockPool(num_blocks=8, block_size=1024, device="cpu")
    assert pool.num_blocks == 8
    assert pool.block_size == 1024
    assert pool.device == "cpu"


def test_block_pool_num_free_blocks():
    from moe_infinity.memory.kv_cache_manager import BlockPool

    pool = BlockPool(num_blocks=8, block_size=1024, device="cpu")
    assert pool.num_free_blocks == 8


def test_block_pool_allocate_returns_ids():
    from moe_infinity.memory.kv_cache_manager import BlockPool

    pool = BlockPool(num_blocks=8, block_size=1024, device="cpu")
    block_ids = pool.allocate(3)
    assert isinstance(block_ids, list)
    assert len(block_ids) == 3
    assert pool.num_free_blocks == 5


def test_block_pool_free_returns_blocks():
    from moe_infinity.memory.kv_cache_manager import BlockPool

    pool = BlockPool(num_blocks=8, block_size=1024, device="cpu")
    block_ids = pool.allocate(3)
    pool.free(block_ids)
    assert pool.num_free_blocks == 8


def test_block_pool_exhaustion_raises():
    from moe_infinity.memory.kv_cache_manager import BlockPool

    pool = BlockPool(num_blocks=4, block_size=1024, device="cpu")
    with pytest.raises(RuntimeError):
        _ = pool.allocate(5)


def test_kv_cache_manager_creation():
    from moe_infinity.memory.kv_cache_manager import BlockPool, KVCacheManager

    gpu_pool = BlockPool(num_blocks=4, block_size=1024, device="cpu")
    cpu_pool = BlockPool(num_blocks=8, block_size=1024, device="cpu")
    manager = KVCacheManager(gpu_pool=gpu_pool, cpu_pool=cpu_pool)
    assert manager is not None


def test_kv_cache_manager_allocate_blocks():
    from moe_infinity.memory.kv_cache_manager import BlockPool, KVCacheManager

    gpu_pool = BlockPool(num_blocks=4, block_size=1024, device="cpu")
    cpu_pool = BlockPool(num_blocks=8, block_size=1024, device="cpu")
    manager = KVCacheManager(gpu_pool=gpu_pool, cpu_pool=cpu_pool)
    block_ids = manager.allocate_blocks(seq_id=0, num_blocks=2)
    assert isinstance(block_ids, list)
    assert len(block_ids) == 2


def test_kv_cache_manager_free_blocks():
    from moe_infinity.memory.kv_cache_manager import BlockPool, KVCacheManager

    gpu_pool = BlockPool(num_blocks=4, block_size=1024, device="cpu")
    cpu_pool = BlockPool(num_blocks=8, block_size=1024, device="cpu")
    manager = KVCacheManager(gpu_pool=gpu_pool, cpu_pool=cpu_pool)
    _ = manager.allocate_blocks(seq_id=1, num_blocks=2)
    manager.free_blocks(seq_id=1)
    assert gpu_pool.num_free_blocks == 4


def test_kv_cache_manager_swap_out_not_implemented():
    from moe_infinity.memory.kv_cache_manager import BlockPool, KVCacheManager

    gpu_pool = BlockPool(num_blocks=4, block_size=1024, device="cpu")
    cpu_pool = BlockPool(num_blocks=8, block_size=1024, device="cpu")
    manager = KVCacheManager(gpu_pool=gpu_pool, cpu_pool=cpu_pool)
    with pytest.raises(NotImplementedError):
        _ = manager.swap_out([0, 1])


def test_kv_cache_manager_swap_in_not_implemented():
    from moe_infinity.memory.kv_cache_manager import BlockPool, KVCacheManager

    gpu_pool = BlockPool(num_blocks=4, block_size=1024, device="cpu")
    cpu_pool = BlockPool(num_blocks=8, block_size=1024, device="cpu")
    manager = KVCacheManager(gpu_pool=gpu_pool, cpu_pool=cpu_pool)
    with pytest.raises(NotImplementedError):
        _ = manager.swap_in([0, 1])
