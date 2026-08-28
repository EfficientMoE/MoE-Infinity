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


@dataclass(frozen=True)
class PagedBatchLengths:
    query_lengths: torch.Tensor | list[int]
    query_offsets: torch.Tensor | list[int]
    context_lengths: torch.Tensor | list[int]
    kv_seq_lengths: torch.Tensor | list[int]

    def validate(self) -> None:
        query = [int(value) for value in self.query_lengths]
        offsets = [int(value) for value in self.query_offsets]
        context = [int(value) for value in self.context_lengths]
        kv = [int(value) for value in self.kv_seq_lengths]
        if len(context) != len(query) or len(kv) != len(query):
            raise ValueError("paged length vectors must have equal batch size")
        expected_offsets = [0]
        for length in query:
            if length < 0:
                raise ValueError("query lengths must be non-negative")
            expected_offsets.append(expected_offsets[-1] + length)
        if offsets != expected_offsets:
            raise ValueError(
                "query_offsets must be the prefix sum of query_lengths"
            )
        if any(prior < 0 for prior in context):
            raise ValueError("context lengths must be non-negative")
        if any(
            total != prior + current
            for total, prior, current in zip(kv, context, query)
        ):
            raise ValueError(
                "kv_seq_lengths must equal context_lengths + query_lengths"
            )


@dataclass
class AttentionMetadata:
    block_tables: torch.Tensor
    lengths: PagedBatchLengths
    max_seq_len: int
    num_prefill_tokens: int
    num_decode_tokens: int
    slot_mapping: torch.Tensor
    is_prefill: bool


@dataclass(frozen=True)
class FlashInferPlanMetadata:
    lengths: PagedBatchLengths
    kv_indptr: torch.Tensor
    kv_last_page_len: torch.Tensor
