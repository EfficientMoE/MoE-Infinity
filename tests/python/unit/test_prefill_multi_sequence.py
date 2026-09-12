"""Packed multi-sequence prefill through PagedAttentionBackend (SDPA path).

``ModelRunner.prepare_inputs`` pads a batch to its longest query and the
paged shims pack the valid tokens back into one ``[tokens, heads, head_dim]``
tensor. The SDPA prefill branch used to run one causal attention over that
whole packed tensor, so sequence ``k`` also attended to the tokens of the
sequences packed before it (measured: in a 4-request batch the first
sequence matched the reference, the other three did not). The backend must
attend each sequence to itself, using ``lengths.query_offsets``.

CPU, deterministic, no model weights.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from moe_infinity.runtime.attention_backend import PagedAttentionBackend
from moe_infinity.runtime.attention_types import (
    AttentionMetadata,
    KVCacheSpec,
    PagedBatchLengths,
)


def _causal_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    # q/k/v: [tokens, heads, head_dim] of ONE sequence
    groups = q.shape[1] // k.shape[1]
    qh = q.transpose(0, 1)
    kh = k.repeat_interleave(groups, dim=1).transpose(0, 1)
    vh = v.repeat_interleave(groups, dim=1).transpose(0, 1)
    return F.scaled_dot_product_attention(qh, kh, vh, scale=scale, is_causal=True).transpose(0, 1)


def _metadata(lengths: list[int], block_size: int) -> AttentionMetadata:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    slots, tables = [], []
    for seq_idx, length in enumerate(lengths):
        base = seq_idx * 4 * block_size  # 4 blocks per sequence, far apart
        slots.extend(range(base, base + length))
        tables.append([seq_idx * 4 + b for b in range(4)])
    return AttentionMetadata(
        block_tables=torch.tensor(tables, dtype=torch.int32),
        max_seq_len=max(lengths),
        num_prefill_tokens=sum(lengths),
        num_decode_tokens=0,
        slot_mapping=torch.tensor(slots, dtype=torch.int64),
        is_prefill=True,
        lengths=PagedBatchLengths(
            query_lengths=torch.tensor(lengths, dtype=torch.int32),
            query_offsets=torch.tensor(offsets, dtype=torch.int32),
            context_lengths=torch.zeros(len(lengths), dtype=torch.int32),
            kv_seq_lengths=torch.tensor(lengths, dtype=torch.int32),
        ),
    )


def _run(lengths: list[int], kv_heads: int) -> None:
    torch.manual_seed(0)
    heads, head_dim, block_size = 4, 16, 4
    total = sum(lengths)
    backend = PagedAttentionBackend(
        spec=KVCacheSpec(kv_heads, head_dim, torch.float32, block_size),
        num_gpu_blocks=4 * len(lengths),
        num_layers=1,
        device=torch.device("cpu"),
    )
    q = torch.randn(total, heads, head_dim)
    k = torch.randn(total, kv_heads, head_dim)
    v = torch.randn(total, kv_heads, head_dim)
    scale = head_dim**-0.5

    out = backend.forward(q, k, v, attention_metadata=_metadata(lengths, block_size), scale=scale, layer_idx=0)

    start = 0
    for length in lengths:
        expected = _causal_sdpa(q[start : start + length], k[start : start + length], v[start : start + length], scale)
        torch.testing.assert_close(out[start : start + length], expected, rtol=1e-5, atol=1e-5)
        start += length


def test_single_sequence_prefill_unchanged() -> None:
    _run([5], kv_heads=4)


def test_two_packed_sequences_attend_only_to_themselves() -> None:
    _run([3, 2], kv_heads=4)


def test_four_packed_sequences_with_gqa() -> None:
    _run([5, 4, 13, 5], kv_heads=2)


def test_second_sequence_is_not_a_continuation_of_the_first() -> None:
    """The regression itself: with one causal mask over the packed tensor the
    second sequence's first token would attend to every token of the first
    sequence, so its output would differ from its own single-token attention."""
    torch.manual_seed(1)
    heads, head_dim, block_size = 2, 8, 4
    backend = PagedAttentionBackend(
        spec=KVCacheSpec(heads, head_dim, torch.float32, block_size),
        num_gpu_blocks=8,
        num_layers=1,
        device=torch.device("cpu"),
    )
    q = torch.randn(3 + 1, heads, head_dim)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = backend.forward(q, k, v, attention_metadata=_metadata([3, 1], block_size), scale=1.0, layer_idx=0)
    # a single-token sequence attends only to itself: output == its own value
    torch.testing.assert_close(out[3], v[3], rtol=1e-5, atol=1e-5)
