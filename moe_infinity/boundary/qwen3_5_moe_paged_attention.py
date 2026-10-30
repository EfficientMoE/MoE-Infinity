# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Paged-attention shim for Qwen3.5-MoE (``Qwen3_5MoeForConditionalGeneration``
and ``Qwen3_5MoeForCausalLM``).

Follows ``moe_store.wrappers.qwen3_paged_attention.Qwen3PagedAttention``: when
the serving runner installs a paged backend and per-batch metadata on the
class, K/V go through the paged KV cache instead of the HuggingFace cache;
otherwise the stock ``Qwen3_5MoeAttention.forward`` runs unchanged.

Only the gated full-attention layers are replaced. The Gated DeltaNet
linear-attention layers and the vision tower keep their HuggingFace classes;
their recurrent/conv state stays in the HF hybrid cache, which the serving
runner carries per sequence (``moe_infinity.serving.linear_attention_state``).

What differs from Qwen3, taken from
``transformers.models.qwen3_5_moe.modeling_qwen3_5_moe`` (v5.12): ``q_proj``
emits the query and an output gate per head (``head_dim * 2``), the attention
output is multiplied by ``sigmoid(gate)`` before ``o_proj``, and
``apply_rotary_pos_emb`` rotates only the leading ``partial_rotary_factor``
of each head. M-RoPE (text/height/width positions for the VL wrapper) is
folded into ``position_embeddings`` by the text model, so this module only
consumes the resulting ``cos``/``sin``.

The paged backend is addressed by the dense paged layer index (the ordinal of
this layer among the full-attention layers), so the paged KV cache holds one
plane per full-attention layer rather than one per decoder layer.

This lives at the moe-store boundary until moe-store ships the shim.
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional, Protocol, Union, cast

import torch
from moe_store.wrappers.qwen3_paged_attention import Qwen3PagedAttention
from transformers.cache_utils import Cache
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeAttention,
    apply_rotary_pos_emb,
)

from moe_infinity.boundary.hybrid_attention_layout import (
    paged_layer_index_for,
)

AttentionMetadata = Any
RuntimeAttentionMetadata = Any

_Metadata = Union[AttentionMetadata, RuntimeAttentionMetadata]


class _SupportsPagedAttention(Protocol):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
        attn_metadata: Optional[_Metadata] = None,
        scale: Optional[float] = None,
        attention_metadata: Optional[_Metadata] = None,
        layer_idx: int = 0,
    ) -> Optional[torch.Tensor]: ...


class Qwen3_5MoePagedAttention(Qwen3_5MoeAttention):
    _paged_backend: ClassVar[Optional[_SupportsPagedAttention]] = None
    _attention_metadata: ClassVar[Optional[_Metadata]] = None

    def __init__(self, config: Any, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        self.paged_layer_idx = paged_layer_index_for(config, layer_idx)

    @classmethod
    def set_paged_context(
        cls,
        backend: _SupportsPagedAttention,
        metadata: _Metadata,
    ) -> None:
        cls._paged_backend = backend
        cls._attention_metadata = metadata

    @classmethod
    def clear_paged_context(cls) -> None:
        cls._paged_backend = None
        cls._attention_metadata = None

    @staticmethod
    def _valid_token_index(
        metadata: object,
        bsz: int,
        q_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        # The runner right-pads every batch to its longest query, except fresh
        # hybrid prefill groups, which it left-pads so the linear-attention
        # layers see padding before (not after) the real tokens.
        if getattr(metadata, "query_padding", "right") != "left":
            return Qwen3PagedAttention._valid_token_index(
                metadata, bsz, q_len, device
            )
        lengths = getattr(metadata, "lengths", None)
        query_lengths = (
            getattr(lengths, "query_lengths", None)
            if lengths is not None
            else None
        )
        if query_lengths is None:
            return None
        per_seq = [int(v) for v in query_lengths.reshape(-1).tolist()]
        if len(per_seq) != bsz or all(length == q_len for length in per_seq):
            return None
        indices = [
            row * q_len + q_len - length + col
            for row, length in enumerate(per_seq)
            for col in range(length)
        ]
        return torch.tensor(indices, dtype=torch.long, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        paged_backend = self.__class__._paged_backend
        attention_metadata = self.__class__._attention_metadata
        if paged_backend is None or attention_metadata is None:
            return super().forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(
                *input_shape, -1, self.head_dim * 2
            ),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = cast(
            torch.Tensor, self.q_norm(query_states.view(hidden_shape))
        ).transpose(1, 2)
        key_states = cast(
            torch.Tensor,
            self.k_norm(self.k_proj(hidden_states).view(hidden_shape)),
        ).transpose(1, 2)
        value_states = (
            self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        )

        cos, sin = position_embeddings
        query_states, key_states = cast(
            tuple[torch.Tensor, torch.Tensor],
            apply_rotary_pos_emb(query_states, key_states, cos, sin),
        )

        # The paged store is the only KV owner here. ``past_key_values`` still
        # reaches this layer because the runner hands the HF hybrid cache to
        # the linear-attention layers; its full-attention slots stay empty.

        bsz, q_len = int(input_shape[0]), int(input_shape[1])
        num_attention_heads = query_states.shape[1]
        num_key_value_heads = key_states.shape[1]

        query_tokens = (
            query_states.transpose(1, 2)
            .contiguous()
            .view(-1, num_attention_heads, self.head_dim)
        )
        key_tokens = (
            key_states.transpose(1, 2)
            .contiguous()
            .view(-1, num_key_value_heads, self.head_dim)
        )
        value_tokens = (
            value_states.transpose(1, 2)
            .contiguous()
            .view(-1, num_key_value_heads, self.head_dim)
        )

        valid_index = self._valid_token_index(
            attention_metadata, bsz, q_len, query_tokens.device
        )
        packed = valid_index is not None
        if packed:
            query_tokens = query_tokens.index_select(0, valid_index)
            key_tokens = key_tokens.index_select(0, valid_index)
            value_tokens = value_tokens.index_select(0, valid_index)

        attn_output_tokens = paged_backend.forward(
            query_tokens,
            key_tokens,
            value_tokens,
            attention_metadata=attention_metadata,
            scale=cast(float, self.scaling),
            layer_idx=int(self.paged_layer_idx),
        )

        if attn_output_tokens is None or attn_output_tokens.ndim != 3:
            raise ValueError(
                "paged attention backend must return rank-3 tensor"
            )

        if packed:
            scattered = attn_output_tokens.new_zeros(
                bsz * q_len,
                attn_output_tokens.shape[1],
                attn_output_tokens.shape[2],
            )
            scattered.index_copy_(0, valid_index, attn_output_tokens)
            attn_output_tokens = scattered

        if attn_output_tokens.shape != (
            bsz * q_len,
            num_attention_heads,
            self.head_dim,
        ):
            raise ValueError(
                "`attn_output` should be of size "
                f"{(bsz * q_len, num_attention_heads, self.head_dim)}, "
                f"but is {tuple(attn_output_tokens.shape)}"
            )

        attn_output = attn_output_tokens.to(gate.dtype).reshape(
            *input_shape, -1
        )
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = cast(torch.Tensor, self.o_proj(attn_output))
        return attn_output, None

    @classmethod
    def get_kv_cache_spec_for_config(cls, config: Any) -> dict[str, int]:
        text_config = (
            config.get_text_config()
            if callable(getattr(config, "get_text_config", None))
            else config
        )
        head_dim = getattr(text_config, "head_dim", None) or (
            text_config.hidden_size // text_config.num_attention_heads
        )
        return {
            "num_kv_heads": int(text_config.num_key_value_heads),
            "head_dim": int(head_dim),
        }


__all__ = ["Qwen3_5MoePagedAttention"]
