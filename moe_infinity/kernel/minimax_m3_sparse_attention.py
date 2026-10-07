# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Reference MiniMax-M3 Lightning Indexer block-sparse attention.

This module provides a pure-torch reference that consumes the Lightning
Indexer's left-packed ``block_indices`` (``[B, S_q, topk]`` with ``-1``
right-padding) directly, instead of the dense additive mask the Hugging Face
eager path builds. It reproduces ``MiniMaxM3VLIndexer.build_block_mask`` and
``eager_attention_forward`` from ``transformers==5.12.0`` exactly, so the
output matches the eager correctness reference. A future FlashInfer or local
CUDA kernel slots in behind the same entry point; the provider is chosen
separately once the decision-checkpoint evidence is recorded on the PR.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _hf_minimax_module():
    from transformers.models.minimax_m3_vl import modeling_minimax_m3_vl

    return modeling_minimax_m3_vl


def minimax_m3_sparse_attention(
    module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    block_indices: torch.Tensor,
    position_ids: torch.Tensor,
    scaling: float,
    attention_mask: torch.Tensor | None = None,
    dropout: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-sparse attention over the indexer-selected key blocks.

    ``block_indices`` is the indexer contract (left-packed valid key-block ids,
    ``-1`` right-padding). It is composed into the additive attention mask by
    the installed ``MiniMaxM3VLIndexer.build_block_mask`` -- the exact function
    the eager path calls -- so the kernel matches eager and stays robust across
    transformers versions that change the indexer selection layout (the
    per-query ``[B, S_q, topk]`` of 5.12 vs the per-head ``[B, H, S_q, topk]``
    of 5.19). The composed mask then feeds the standard scaled-dot-product math.
    """
    hf = _hf_minimax_module()
    dtype = query_states.dtype
    device = query_states.device
    key_length = key_states.shape[2]

    additive_mask = module.indexer.build_block_mask(
        block_indices, attention_mask, key_length, dtype, device, position_ids
    )

    key = hf.repeat_kv(key_states, module.num_key_value_groups)
    value = hf.repeat_kv(value_states, module.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key.transpose(2, 3)) * scaling
    attn_weights = attn_weights + additive_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        dtype
    )
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def indexer_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Drop-in ``MiniMaxM3VLAttention.forward`` routed through the kernel.

    Mirrors the recorded hook: projections, QK-norm, partial RoPE, and indexer
    selection are identical to the eager path; only the dense mask build plus
    attention interface is replaced by :func:`minimax_m3_sparse_attention`.
    """
    hf = _hf_minimax_module()
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(
        self.q_proj(hidden_states).view(hidden_shape)
    ).transpose(1, 2)
    key_states = self.k_norm(
        self.k_proj(hidden_states).view(hidden_shape)
    ).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = hf.apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx
        )

    position_ids = kwargs.get("position_ids")
    if position_ids is None:
        position_ids = torch.arange(
            key_states.shape[2] - query_states.shape[2],
            key_states.shape[2],
            device=query_states.device,
        )
    position_ids = (
        position_ids if position_ids.ndim > 1 else position_ids.unsqueeze(0)
    ).expand(query_states.shape[0], -1)

    block_indices = self.indexer(
        hidden_states, position_embeddings, past_key_values, position_ids
    )

    attn_output, attn_weights = minimax_m3_sparse_attention(
        self,
        query_states,
        key_states,
        value_states,
        block_indices,
        position_ids,
        scaling=self.scaling,
        attention_mask=attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), attn_weights


def make_minimax_m3_indexer_forward(original_forward):
    """Wrap ``MiniMaxM3VLAttention.forward`` behind the opt-in flag.

    The wrapper is a verified no-op unless ``enable_minimax_m3_indexer_kernel``
    is set on the module config and the layer owns an indexer; otherwise it
    delegates to the untouched eager ``original_forward``.
    """

    def forward(self, *args, **kwargs):
        if not getattr(self.config, "enable_minimax_m3_indexer_kernel", False):
            return original_forward(self, *args, **kwargs)
        if getattr(self, "indexer", None) is None:
            return original_forward(self, *args, **kwargs)
        return indexer_attention_forward(self, *args, **kwargs)

    forward._minimax_m3_indexer_wrapped = original_forward
    return forward
