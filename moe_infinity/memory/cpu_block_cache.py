from __future__ import annotations

import threading
from typing import Optional

from moe_infinity.memory.block_pool import BlockPool, KVCacheBlock
from moe_infinity.memory.offloading_policy import CachePolicy, LRUPolicy


class CPUBlockCache:
    def __init__(
        self,
        cpu_pool: BlockPool,
        policy: Optional[CachePolicy[int, int]] = None,
        capacity: Optional[int] = None,
    ):
        self._cpu_pool: BlockPool = cpu_pool
        cap = capacity or cpu_pool.total_blocks()
        self._policy: CachePolicy[int, int] = policy or LRUPolicy(capacity=cap)
        self._lock: threading.Lock = threading.Lock()
        self._hash_to_cpu_block: dict[int, KVCacheBlock] = {}
        self._cpu_id_to_hash: dict[int, int] = {}

    def store(self, block_hash: int, gpu_block_id: int) -> Optional[int]:
        _ = gpu_block_id
        with self._lock:
            cached_block = self._hash_to_cpu_block.get(block_hash)
            if cached_block is not None:
                _ = self._policy.get(block_hash)
                return cached_block.block_id

            self._evict_if_needed_locked()

            cpu_block = self._cpu_pool.allocate_block()
            if cpu_block is None:
                return None

            cpu_block_id = cpu_block.block_id
            self._hash_to_cpu_block[block_hash] = cpu_block
            self._cpu_id_to_hash[cpu_block_id] = block_hash
            self._policy.put(block_hash, cpu_block_id)
            return cpu_block_id

    def load(self, block_hash: int) -> Optional[int]:
        with self._lock:
            cpu_block_id = self._policy.get(block_hash)
            if cpu_block_id is None:
                return None
            block = self._hash_to_cpu_block.get(block_hash)
            if block is None or block.block_id != cpu_block_id:
                return None
            return cpu_block_id

    def evict_if_needed(self) -> None:
        with self._lock:
            self._evict_if_needed_locked()

    def _evict_if_needed_locked(self) -> None:
        while len(self._policy) >= self._policy.capacity:
            evicted = self._policy.evict()
            if evicted is None:
                break

            block_hash, cpu_block_id = evicted
            cpu_block = self._hash_to_cpu_block.pop(block_hash, None)
            _ = self._cpu_id_to_hash.pop(cpu_block_id, None)

            if cpu_block is not None and cpu_block.ref_cnt > 0:
                self._cpu_pool.free_block(cpu_block)

    def invalidate(self, block_hash: int) -> None:
        with self._lock:
            cpu_block = self._hash_to_cpu_block.pop(block_hash, None)
            if cpu_block is None:
                return
            cpu_block_id = cpu_block.block_id

            _ = self._cpu_id_to_hash.pop(cpu_block_id, None)
            self._remove_from_policy_locked(block_hash)

            if cpu_block.ref_cnt > 0:
                self._cpu_pool.free_block(cpu_block)

    def _remove_from_policy_locked(self, target_hash: int) -> None:
        retained: list[tuple[int, int]] = []
        while True:
            item = self._policy.evict()
            if item is None:
                break
            if item[0] != target_hash:
                retained.append(item)

        for key, value in retained:
            self._policy.put(key, value)

    def __len__(self) -> int:
        with self._lock:
            return len(self._hash_to_cpu_block)

    @property
    def num_cached_blocks(self) -> int:
        return len(self)
