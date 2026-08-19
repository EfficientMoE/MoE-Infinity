from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class KVCacheSpec:
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    block_size: int

    @property
    def page_size_bytes(self) -> int:
        dtype_size = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}[
            self.dtype
        ]
        return (
            self.block_size * self.num_kv_heads * self.head_dim * dtype_size * 2
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
    seq_id: int | None = None
