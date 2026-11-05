# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Paged-attention shim for Mixtral (``MixtralForCausalLM``).

Follows ``moe_store.wrappers.qwen3_paged_attention.Qwen3PagedAttention``: when
the serving runner installs a paged backend and per-batch metadata on the
class, K/V go through the paged KV cache instead of the HuggingFace cache;
otherwise the stock ``MixtralAttention.forward`` runs unchanged.

What differs from Qwen3, taken from
``transformers.models.mixtral.modeling_mixtral`` (v5.12): there is no
per-head ``q_norm``/``k_norm``, and the attention interface receives
``config.sliding_window``. The paged backend has no window mask, so a config
whose window is shorter than its context is reported unsupported
(``paged_unsupported_reason``) and the serving entry points refuse it at
startup. The released Mixtral checkpoints set ``sliding_window`` to null.

This lives at the moe-store boundary until moe-store ships the shim.
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional, Protocol, Union, cast

import torch
from moe_store.wrappers.qwen3_paged_attention import Qwen3PagedAttention
from transformers.cache_utils import Cache
from transformers.models.mixtral.modeling_mixtral import (
    MixtralAttention,
    apply_rotary_pos_emb,
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


class MixtralPagedAttention(MixtralAttention):
    _paged_backend: ClassVar[Optional[_SupportsPagedAttention]] = None
    _attention_metadata: ClassVar[Optional[_Metadata]] = None

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

    @classmethod
    def paged_unsupported_reason(cls, config: Any) -> Optional[str]:
        window = getattr(config, "sliding_window", None)
        if window is None:
            return None
        max_positions = getattr(config, "max_position_embeddings", None)
        if isinstance(max_positions, int) and int(window) >= max_positions:
            return None
        return (
            f"sliding_window={window} is shorter than the context "
            f"(max_position_embeddings={max_positions}); the paged attention "
            "backend has no sliding-window mask"
        )

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

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        cos, sin = position_embeddings
        query_states, key_states = cast(
            tuple[torch.Tensor, torch.Tensor],
            apply_rotary_pos_emb(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                cos,
                sin,
            ),
        )

        # The paged store is the only KV owner here; ``past_key_values`` is
        # not updated.

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
        value_tokens = value_states.reshape(
            -1, num_key_value_heads, self.head_dim
        )

        valid_index = Qwen3PagedAttention._valid_token_index(
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
            scale=float(self.scaling),
            layer_idx=int(self.layer_idx),
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

        attn_output = attn_output_tokens.to(hidden_states.dtype).reshape(
            *input_shape, -1
        )
        attn_output = cast(torch.Tensor, self.o_proj(attn_output))
        return attn_output, None

    @classmethod
    def get_kv_cache_spec_for_config(cls, config: Any) -> dict[str, int]:
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        return {
            "num_kv_heads": int(config.num_key_value_heads),
            "head_dim": int(head_dim),
        }


__all__ = ["MixtralPagedAttention"]
