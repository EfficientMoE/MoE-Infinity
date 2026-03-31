from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Protocol, cast

from .block_pool import BlockPool, KVCacheBlock


class _CudaDeviceProperties(Protocol):
    total_memory: int


class _CudaInterface(Protocol):
    def is_available(self) -> bool: ...

    def get_device_properties(self, device: int) -> _CudaDeviceProperties: ...


@dataclass
class MemoryBudget:
    device_memory_ratio: float = 0.75
    kv_cache_memory_ratio: float = 0.15
    host_memory_ratio: float = 0.25

    @property
    def total_gpu_memory_bytes(self) -> int:
        try:
            import torch

            get_properties = cast(
                Callable[[int], _CudaDeviceProperties] | None,
                getattr(torch.cuda, "get_device_properties", None),
            )
            cuda: _CudaInterface = torch.cuda
            if cuda.is_available():
                if get_properties is not None:
                    return int(get_properties(0).total_memory)
        except Exception:
            pass
        return 24 * 1024**3

    @property
    def kv_cache_gpu_bytes(self) -> int:
        return int(self.total_gpu_memory_bytes * self.kv_cache_memory_ratio)


class KVCacheManager:
    block_size: int
    _gpu_pool: BlockPool
    _cpu_pool: BlockPool
    _seq_to_gpu_blocks: dict[str, list[int]]
    _swap_map_gpu_to_cpu: dict[int, int]
    _seq_to_swapped_gpu_blocks: dict[str, list[int]]
    _gpu_allocated_blocks: dict[int, KVCacheBlock]
    _cpu_allocated_blocks: dict[int, KVCacheBlock]

    def __init__(
        self,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        block_size: int = 16,
    ):
        self.block_size = block_size
        self._gpu_pool = BlockPool(num_blocks=num_gpu_blocks)
        self._cpu_pool = BlockPool(num_blocks=num_cpu_blocks)
        self._seq_to_gpu_blocks = {}
        self._swap_map_gpu_to_cpu = {}
        self._seq_to_swapped_gpu_blocks = {}
        self._gpu_allocated_blocks = {}
        self._cpu_allocated_blocks = {}

    def allocate_blocks_for_sequence(
        self, seq_id: str, num_tokens: int
    ) -> bool:
        num_blocks = math.ceil(num_tokens / self.block_size)
        blocks: list[KVCacheBlock] = []
        for _ in range(num_blocks):
            block = self._gpu_pool.allocate_block()
            if block is None:
                for allocated in blocks:
                    _ = self._gpu_allocated_blocks.pop(allocated.block_id, None)
                    self._gpu_pool.free_block(allocated)
                return False
            blocks.append(block)
            self._gpu_allocated_blocks[block.block_id] = block

        self._seq_to_gpu_blocks[seq_id] = [b.block_id for b in blocks]
        return True

    def append_token_block(self, seq_id: str) -> bool:
        block = self._gpu_pool.allocate_block()
        if block is None:
            return False

        self._gpu_allocated_blocks[block.block_id] = block
        self._seq_to_gpu_blocks.setdefault(seq_id, []).append(block.block_id)
        return True

    def free_sequence(self, seq_id: str) -> None:
        for block_id in self._seq_to_gpu_blocks.pop(seq_id, []):
            block = self._gpu_allocated_blocks.pop(block_id, None)
            if block is None:
                continue
            self._gpu_pool.free_block(block)

        for swapped_gpu_id in self._seq_to_swapped_gpu_blocks.pop(seq_id, []):
            cpu_id = self._swap_map_gpu_to_cpu.pop(swapped_gpu_id, None)
            if cpu_id is None:
                continue
            cpu_block = self._cpu_allocated_blocks.pop(cpu_id, None)
            if cpu_block is None:
                continue
            self._cpu_pool.free_block(cpu_block)

    def get_block_table(self, seq_id: str) -> list[int]:
        return list(self._seq_to_gpu_blocks.get(seq_id, []))

    def prepare_swap_out(self, seq_id: str) -> list[tuple[int, int]]:
        gpu_block_ids = self._seq_to_gpu_blocks.get(seq_id, [])
        if not gpu_block_ids:
            return []

        pairs: list[tuple[int, int]] = []
        cpu_blocks_allocated: list[KVCacheBlock] = []
        for gpu_id in gpu_block_ids:
            cpu_block = self._cpu_pool.allocate_block()
            if cpu_block is None:
                for allocated in cpu_blocks_allocated:
                    _ = self._cpu_allocated_blocks.pop(allocated.block_id, None)
                    self._cpu_pool.free_block(allocated)
                return []

            pairs.append((gpu_id, cpu_block.block_id))
            cpu_blocks_allocated.append(cpu_block)
            self._cpu_allocated_blocks[cpu_block.block_id] = cpu_block
        return pairs

    def commit_swap_out(
        self, seq_id: str, pairs: list[tuple[int, int]]
    ) -> None:
        swapped_gpu_ids: list[int] = []
        for gpu_id, cpu_id in pairs:
            self._swap_map_gpu_to_cpu[gpu_id] = cpu_id
            swapped_gpu_ids.append(gpu_id)
            block = self._gpu_allocated_blocks.pop(gpu_id, None)
            if block is None:
                continue
            self._gpu_pool.free_block(block)

        if swapped_gpu_ids:
            self._seq_to_swapped_gpu_blocks[seq_id] = swapped_gpu_ids
        _ = self._seq_to_gpu_blocks.pop(seq_id, None)

    def prepare_swap_in(
        self,
        seq_id: str,
        swapped_gpu_block_ids: list[int],
    ) -> list[tuple[int, int]]:
        _ = seq_id
        pairs: list[tuple[int, int]] = []
        new_gpu_blocks: list[KVCacheBlock] = []
        for orig_gpu_id in swapped_gpu_block_ids:
            cpu_id = self._swap_map_gpu_to_cpu.get(orig_gpu_id)
            if cpu_id is None:
                for allocated in new_gpu_blocks:
                    _ = self._gpu_allocated_blocks.pop(allocated.block_id, None)
                    self._gpu_pool.free_block(allocated)
                return []

            new_gpu_block = self._gpu_pool.allocate_block()
            if new_gpu_block is None:
                for allocated in new_gpu_blocks:
                    _ = self._gpu_allocated_blocks.pop(allocated.block_id, None)
                    self._gpu_pool.free_block(allocated)
                return []

            pairs.append((cpu_id, new_gpu_block.block_id))
            new_gpu_blocks.append(new_gpu_block)
            self._gpu_allocated_blocks[new_gpu_block.block_id] = new_gpu_block
        return pairs

    def commit_swap_in(
        self,
        seq_id: str,
        orig_gpu_block_ids: list[int],
        pairs: list[tuple[int, int]],
    ) -> None:
        new_gpu_ids: list[int] = []
        for orig_gpu_id, (cpu_id, new_gpu_id) in zip(orig_gpu_block_ids, pairs):
            cpu_block = self._cpu_allocated_blocks.pop(cpu_id, None)
            if cpu_block is not None:
                self._cpu_pool.free_block(cpu_block)
            if orig_gpu_id in self._swap_map_gpu_to_cpu:
                del self._swap_map_gpu_to_cpu[orig_gpu_id]
            new_gpu_ids.append(new_gpu_id)

        _ = self._seq_to_swapped_gpu_blocks.pop(seq_id, None)
        self._seq_to_gpu_blocks[seq_id] = new_gpu_ids

    @property
    def num_free_gpu_blocks(self) -> int:
        return self._gpu_pool.num_free_blocks()

    @property
    def num_free_cpu_blocks(self) -> int:
        return self._cpu_pool.num_free_blocks()

    @property
    def num_gpu_blocks(self) -> int:
        return self._gpu_pool.total_blocks()

    @property
    def num_cpu_blocks(self) -> int:
        return self._cpu_pool.total_blocks()


__all__ = ["BlockPool", "KVCacheManager", "MemoryBudget"]
