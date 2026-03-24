from __future__ import annotations

from dataclasses import dataclass, field

import torch


def _detect_total_gpu_memory_bytes() -> int:
    return 1024 * 1024 * 1024


@dataclass
class MemoryBudget:
    expert_cache_ratio: float = 0.75
    kv_cache_ratio: float = 0.0
    total_gpu_memory_bytes: int = field(
        default_factory=_detect_total_gpu_memory_bytes
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.expert_cache_ratio <= 1.0:
            raise ValueError(
                f"expert_cache_ratio must be in [0, 1], got {self.expert_cache_ratio}"
            )
        if not 0.0 <= self.kv_cache_ratio <= 1.0:
            raise ValueError(
                f"kv_cache_ratio must be in [0, 1], got {self.kv_cache_ratio}"
            )
        if self.expert_cache_ratio + self.kv_cache_ratio > 1.0:
            raise ValueError(
                f"expert_cache_ratio ({self.expert_cache_ratio}) + kv_cache_ratio ({self.kv_cache_ratio}) > 1.0"
            )


class BlockPool:
    num_blocks: int
    block_size: int
    device: str
    _free_list: list[int]

    def __init__(self, num_blocks: int, block_size: int, device: str):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = device
        self._free_list = list(range(num_blocks))

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_list)

    def allocate(self, num_blocks: int) -> list[int]:
        if num_blocks > self.num_free_blocks:
            raise RuntimeError(
                f"BlockPool exhausted: requested {num_blocks} blocks but only {self.num_free_blocks} available on {self.device}"
            )
        allocated = self._free_list[:num_blocks]
        self._free_list = self._free_list[num_blocks:]
        return allocated

    def free(self, block_ids: list[int]) -> None:
        self._free_list.extend(block_ids)


class KVCacheManager:
    gpu_pool: BlockPool
    cpu_pool: BlockPool
    _seq_blocks: dict[int, list[int]]

    def __init__(self, gpu_pool: BlockPool, cpu_pool: BlockPool):
        self.gpu_pool = gpu_pool
        self.cpu_pool = cpu_pool
        self._seq_blocks = {}

    def allocate_blocks(self, seq_id: int, num_blocks: int) -> list[int]:
        block_ids = self.gpu_pool.allocate(num_blocks)
        self._seq_blocks.setdefault(seq_id, []).extend(block_ids)
        return block_ids

    def free_blocks(self, seq_id: int) -> None:
        block_ids = self._seq_blocks.pop(seq_id, None)
        if block_ids is not None:
            self.gpu_pool.free(block_ids)

    def get_block_table(self, seq_id: int) -> torch.Tensor | None:
        blocks = self._seq_blocks.get(seq_id)
        if blocks is None:
            return None
        return torch.tensor(blocks, dtype=torch.int32)

    def swap_out(self, block_ids: list[int]) -> list[int]:
        _ = block_ids
        raise NotImplementedError(
            "KV cache swap_out requires C++ extension support. Enable by implementing kv_cache_swap_out in moe_infinity._store."
        )

    def swap_in(self, block_ids: list[int]) -> list[int]:
        _ = block_ids
        raise NotImplementedError(
            "KV cache swap_in requires C++ extension support. Enable by implementing kv_cache_swap_in in moe_infinity._store."
        )
