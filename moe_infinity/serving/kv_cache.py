from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import torch

try:
    import flashinfer
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
    )
except ImportError:
    flashinfer = None
    BatchDecodeWithPagedKVCacheWrapper = None
    BatchPrefillWithPagedKVCacheWrapper = None

HAS_FLASHINFER = flashinfer is not None


@dataclass
class BlockAllocator:
    num_blocks: int
    block_size: int
    device: torch.device
    _free_block_heap: list[int] = field(init=False, repr=False)
    _free_block_set: set[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {self.num_blocks}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {self.block_size}")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        self._free_block_heap = list(range(self.num_blocks))
        heapq.heapify(self._free_block_heap)
        self._free_block_set = set(self._free_block_heap)

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_block_heap)

    def allocate(self, num_blocks: int) -> list[int]:
        if num_blocks < 0:
            raise ValueError(f"num_blocks must be >= 0, got {num_blocks}")
        if num_blocks == 0:
            return []
        if num_blocks > self.num_free_blocks:
            raise RuntimeError(
                f"BlockAllocator exhausted: requested {num_blocks} blocks but only {self.num_free_blocks} available"
            )

        allocated: list[int] = []
        for _ in range(num_blocks):
            block_id = heapq.heappop(self._free_block_heap)
            self._free_block_set.remove(block_id)
            allocated.append(block_id)
        return allocated

    def free(self, block_ids: list[int]) -> None:
        for block_id in block_ids:
            if not 0 <= block_id < self.num_blocks:
                raise ValueError(
                    f"invalid block id {block_id}; expected [0, {self.num_blocks})"
                )
            if block_id in self._free_block_set:
                raise ValueError(f"block id {block_id} is already free")

            heapq.heappush(self._free_block_heap, block_id)
            self._free_block_set.add(block_id)


@dataclass
class BlockTable:
    block_allocator: BlockAllocator
    _block_ids: list[int] = field(default_factory=list, init=False, repr=False)
    _num_tokens: int = field(default=0, init=False, repr=False)

    @property
    def block_size(self) -> int:
        return self.block_allocator.block_size

    def append_token(self) -> None:
        if self._num_tokens % self.block_size == 0:
            new_block_ids = self.block_allocator.allocate(1)
            self._block_ids.append(new_block_ids[0])
        self._num_tokens += 1

    def get_block_ids(self) -> list[int]:
        return list(self._block_ids)

    def num_computed_tokens(self) -> int:
        return self._num_tokens

    def has_blocks(self) -> bool:
        return bool(self._block_ids)

    def restore_blocks(self, block_ids: list[int], num_tokens: int) -> None:
        self._block_ids = list(block_ids)
        self._num_tokens = num_tokens

    def release(self) -> None:
        if self._block_ids:
            self.block_allocator.free(self._block_ids)
        self._block_ids = []
        self._num_tokens = 0


@dataclass
class PagedKVCache:
    num_blocks: int
    block_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device | None = None
    block_allocator: BlockAllocator = field(init=False)
    _sequence_tables: dict[int, BlockTable] = field(
        init=False, default_factory=dict
    )
    _swapped_cpu_buffers: dict[int, torch.Tensor] = field(
        init=False, default_factory=dict
    )
    _swapped_out_sequences: set[int] = field(init=False, default_factory=set)
    _kv_cache: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {self.num_layers}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {self.num_heads}")
        if self.head_dim <= 0:
            raise ValueError(f"head_dim must be > 0, got {self.head_dim}")

        self.device = self._resolve_device(self.device)
        self.block_allocator = BlockAllocator(
            num_blocks=self.num_blocks,
            block_size=self.block_size,
            device=self.device,
        )
        self._kv_cache = torch.zeros(
            (
                self.num_layers,
                self.num_blocks,
                2,
                self.block_size,
                self.num_heads,
                self.head_dim,
            ),
            dtype=self.dtype,
            device=self.device,
        )

    def allocate_sequence(self, seq_id: int, num_tokens: int) -> None:
        if seq_id in self._sequence_tables:
            raise ValueError(f"sequence {seq_id} already exists")
        if num_tokens < 0:
            raise ValueError(f"num_tokens must be >= 0, got {num_tokens}")

        block_table = BlockTable(block_allocator=self.block_allocator)
        for _ in range(num_tokens):
            block_table.append_token()
        self._sequence_tables[seq_id] = block_table

    def append_tokens(self, seq_id: int, num_new_tokens: int) -> None:
        if num_new_tokens < 0:
            raise ValueError(
                f"num_new_tokens must be >= 0, got {num_new_tokens}"
            )
        block_table = self._require_sequence(seq_id)
        for _ in range(num_new_tokens):
            block_table.append_token()

    def free_sequence(self, seq_id: int) -> None:
        block_table = self._sequence_tables.pop(seq_id, None)
        if block_table is None:
            return

        block_table.release()
        _ = self._swapped_cpu_buffers.pop(seq_id, None)
        self._swapped_out_sequences.discard(seq_id)

    def free_gpu_blocks(self, seq_id: int) -> None:
        block_table = self._sequence_tables.get(seq_id)
        if block_table is None:
            return

        if block_table._block_ids:
            self.block_allocator.free(block_table._block_ids)
            block_table._block_ids = []

    def get_block_table(self, seq_id: int) -> list[int]:
        block_table = self._require_sequence(seq_id)
        return block_table.get_block_ids()

    def get_kv_cache_tensors(self) -> torch.Tensor:
        return self._kv_cache

    def swap_out(self, seq_id: int) -> None:
        block_table = self._require_sequence(seq_id)
        if seq_id in self._swapped_out_sequences:
            return

        block_ids = block_table.get_block_ids()
        if block_ids:
            self._swapped_cpu_buffers[seq_id] = (
                self._kv_cache[:, block_ids, ...].detach().to("cpu").clone()
            )
        self._swapped_out_sequences.add(seq_id)

    def swap_in(self, seq_id: int) -> None:
        block_table = self._require_sequence(seq_id)
        if seq_id not in self._swapped_out_sequences:
            return

        # Re-allocate blocks if they were freed during swap-out
        if not block_table._block_ids and block_table._num_tokens > 0:
            from math import ceil

            needed = ceil(block_table._num_tokens / block_table.block_size)
            block_table._block_ids = self.block_allocator.allocate(needed)

        cpu_buffer = self._swapped_cpu_buffers.pop(seq_id, None)
        if cpu_buffer is not None:
            block_ids = block_table.get_block_ids()
            if block_ids:
                self._kv_cache[:, block_ids, ...] = cpu_buffer.to(
                    device=self._kv_cache.device,
                    dtype=self._kv_cache.dtype,
                )

        self._swapped_out_sequences.discard(seq_id)

    def _compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        # Use FlashInfer paged attention if available, else fall back to torch SDPA.
        if HAS_FLASHINFER:
            # FlashInfer path (future: use BatchDecodeWithPagedKVCacheWrapper).
            _ = (
                flashinfer,
                BatchPrefillWithPagedKVCacheWrapper,
                BatchDecodeWithPagedKVCacheWrapper,
            )
            pass  # currently falls through to SDPA until integration complete

        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )

    def _require_sequence(self, seq_id: int) -> BlockTable:
        block_table = self._sequence_tables.get(seq_id)
        if block_table is None:
            raise KeyError(f"unknown sequence id: {seq_id}")
        return block_table

    @staticmethod
    def _resolve_device(device: torch.device | None) -> torch.device:
        if device is None:
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")

        if device.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return device


__all__ = ["BlockAllocator", "BlockTable", "PagedKVCache"]
