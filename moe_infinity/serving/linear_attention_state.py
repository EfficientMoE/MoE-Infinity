# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Per-sequence recurrent state for the linear-attention layers of a hybrid
model served with paged attention.

The paged KV cache only covers the full-attention layers. The linear-attention
layers (Qwen3.5 Gated DeltaNet) carry a conv state and a recurrent state that
HuggingFace keeps in its hybrid ``DynamicCache``; the serving runner builds a
fresh HF cache for every forward, so this module snapshots each sequence's
state after a forward and restores it into the next one.

Snapshots are keyed by the number of tokens they have consumed. A forward over
``context_len`` cached tokens reads the snapshot at ``context_len`` and writes
one at ``context_len + query_len``; the snapshot it read is kept until the
sequence moves past it, so re-executing the same batch (the engine re-runs a
step's batch, relying on paged KV writes being idempotent) restores the same
input state instead of advancing it twice. A missing snapshot fails loudly:
paths that skip tokens (prefix KV reuse, rolled-back speculative tokens)
cannot be served correctly for a hybrid model.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch
from transformers.cache_utils import Cache, DynamicCache

# Per linear layer: (conv_states, recurrent_states), each with batch dim 1.
_Snapshot = tuple[tuple[torch.Tensor, torch.Tensor], ...]


class LinearAttentionStateCache:
    def __init__(
        self, config: Any, linear_layer_indices: Sequence[int]
    ) -> None:
        if not linear_layer_indices:
            raise ValueError("linear_layer_indices must not be empty")
        self.config = config
        self.linear_layer_indices = tuple(int(i) for i in linear_layer_indices)
        self._states: dict[int, dict[int, _Snapshot]] = {}

    def __contains__(self, seq_id: object) -> bool:
        return seq_id in self._states

    def __len__(self) -> int:
        return len(self._states)

    def has_state(self, seq_id: int, num_tokens: int) -> bool:
        return num_tokens in self._states.get(seq_id, {})

    def build_cache(
        self, seq_ids: Sequence[int], context_lengths: Sequence[int]
    ) -> Cache:
        """Return the HF hybrid cache for one homogeneous row group.

        Every row must either start a sequence (``context_len == 0``) or
        continue one; HF tracks "has previous state" per layer for the whole
        batch, so the two cannot share a forward.
        """
        cache = DynamicCache(config=self.config)
        fresh = [int(context) == 0 for context in context_lengths]
        if all(fresh):
            return cache
        if any(fresh):
            raise ValueError(
                "a linear-attention forward cannot mix new and continuing rows"
            )

        snapshots: list[_Snapshot] = []
        for seq_id, context_len in zip(seq_ids, context_lengths):
            snapshot = self._states.get(seq_id, {}).get(int(context_len))
            if snapshot is None:
                raise RuntimeError(
                    f"no linear-attention state for sequence {seq_id} at "
                    f"token {int(context_len)}; hybrid linear-attention "
                    "models cannot resume from KV that skipped their "
                    "recurrent layers (prefix reuse, chunk history, or "
                    "speculative rollback)"
                )
            snapshots.append(snapshot)

        for slot, layer_idx in enumerate(self.linear_layer_indices):
            conv_states = torch.cat(
                [snapshot[slot][0] for snapshot in snapshots], dim=0
            )
            recurrent_states = torch.cat(
                [snapshot[slot][1] for snapshot in snapshots], dim=0
            )
            # The first update copies into fresh layer storage and marks the
            # layer as having previous state; snapshots are never aliased.
            cache.update_conv_state(conv_states, layer_idx)
            cache.update_recurrent_state(recurrent_states, layer_idx)
        return cache

    def commit(
        self,
        seq_ids: Sequence[int],
        context_lengths: Sequence[int],
        query_lengths: Sequence[int],
        cache: Cache,
    ) -> None:
        """Snapshot each row's state after a successful forward."""
        layers = [cache.layers[idx] for idx in self.linear_layer_indices]
        for layer_idx, layer in zip(self.linear_layer_indices, layers):
            conv_states = getattr(layer, "conv_states", None)
            recurrent_states = getattr(layer, "recurrent_states", None)
            if conv_states is None or recurrent_states is None:
                raise RuntimeError(
                    f"linear-attention layer {layer_idx} produced no state"
                )
            if conv_states.shape[0] != len(seq_ids):
                raise RuntimeError(
                    "linear-attention state batch does not match the rows"
                )
        for row, (seq_id, context_len, query_len) in enumerate(
            zip(seq_ids, context_lengths, query_lengths)
        ):
            snapshot: _Snapshot = tuple(
                (
                    layer.conv_states[row : row + 1].clone(),
                    layer.recurrent_states[row : row + 1].clone(),
                )
                for layer in layers
            )
            base = int(context_len)
            end = base + int(query_len)
            entries = self._states.setdefault(seq_id, {})
            for num_tokens in list(entries):
                if num_tokens != end and (base == 0 or num_tokens != base):
                    del entries[num_tokens]
            entries[end] = snapshot

    def release(self, seq_ids: Iterable[int]) -> None:
        for seq_id in seq_ids:
            self._states.pop(seq_id, None)

    def retain(self, live_seq_ids: Iterable[int]) -> None:
        live = set(live_seq_ids)
        for seq_id in [seq_id for seq_id in self._states if seq_id not in live]:
            del self._states[seq_id]


__all__ = ["LinearAttentionStateCache"]
