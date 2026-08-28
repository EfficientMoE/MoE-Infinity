from __future__ import annotations

from dataclasses import dataclass

import torch

from moe_infinity.runtime.kv_cache_format import KVCacheFormat


@dataclass
class KVCacheSpec:
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    block_size: int
    format_name: str = "native"

    @property
    def page_size_bytes(self) -> int:
        return KVCacheFormat.parse(self.format_name).page_size_bytes(
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            execution_dtype=self.dtype,
        )


@dataclass
class AttentionMetadata:
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    max_seq_len: int
    num_prefill_tokens: int
    num_decode_tokens: int
    slot_mapping: torch.Tensor
    is_prefill: bool
