from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import torch


@dataclass
class AttentionMetadata:
    is_prefill: bool
    block_table: Optional[torch.Tensor]
    slot_mapping: Optional[torch.Tensor]
    seq_lens: Optional[torch.Tensor] = None


@runtime_checkable
class AttentionBackend(Protocol):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        scale: Optional[float] = None,
    ) -> Optional[torch.Tensor]: ...

    def get_kv_cache_shape(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]: ...

    def supports_dtype(self, dtype: torch.dtype) -> bool: ...


class PlaceholderAttentionBackend:
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        scale: Optional[float] = None,
    ) -> Optional[torch.Tensor]:
        _ = (query, key, value, kv_cache, attn_metadata, scale)
        return None

    def get_kv_cache_shape(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        return (2, num_blocks, num_kv_heads, block_size, head_size)

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        _ = dtype
        return True
