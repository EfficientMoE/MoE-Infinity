# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Paged-attention shim for OLMoE (``OlmoeForCausalLM``).

Mirrors ``qwen3_paged_attention.Qwen3PagedAttention``: when the serving
runner installs a paged backend and per-batch metadata on the class, the
forward routes attention through the paged KV cache instead of the
HuggingFace cache; otherwise it defers to the stock ``OlmoeAttention``.

What differs from Qwen3, taken from ``transformers.models.olmoe.modeling_olmoe``
(v5.17): ``q_norm``/``k_norm`` are RMSNorms over the *whole* projection and
run *before* the reshape into heads (Qwen3 normalises per head, after the
reshape); ``config.clip_qkv`` clamps q, k and v after the norm; ``head_dim``
is ``getattr(config, "head_dim", hidden_size // num_attention_heads)``.

The padded-token packing (``_valid_token_index`` -> ``index_select`` ->
backend -> ``index_copy_``) is required because ``ModelRunner.prepare_inputs``
pads every batch to its longest query.
"""

from __future__ import annotations

from typing import ClassVar, Optional, Protocol, Union, cast

import torch
from transformers.cache_utils import Cache
from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
from transformers.models.olmoe.modeling_olmoe import (
    OlmoeAttention,
    apply_rotary_pos_emb,
)

from moe_infinity.runtime.attention_backend import AttentionMetadata
from moe_infinity.runtime.attention_types import (
    AttentionMetadata as RuntimeAttentionMetadata,
)

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


class OlmoePagedAttention(OlmoeAttention):
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

    @staticmethod
    def _valid_token_index(
        metadata: object,
        bsz: int,
        q_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
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
            row * q_len + col
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

        # OLMoE: RMSNorm over the whole projection, before the split into
        # heads; then the optional clamp.  Same order as OlmoeAttention.
        query_states = cast(torch.Tensor, self.q_norm(self.q_proj(hidden_states)))
        key_states = cast(torch.Tensor, self.k_norm(self.k_proj(hidden_states)))
        value_states = cast(torch.Tensor, self.v_proj(hidden_states))

        clip_qkv = getattr(self.config, "clip_qkv", None)
        if clip_qkv is not None:
            query_states.clamp_(min=-clip_qkv, max=clip_qkv)
            key_states.clamp_(min=-clip_qkv, max=clip_qkv)
            value_states.clamp_(min=-clip_qkv, max=clip_qkv)

        query_states = query_states.view(*hidden_shape).transpose(1, 2)
        key_states = key_states.view(*hidden_shape).transpose(1, 2)
        value_states = value_states.view(*hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = cast(
            tuple[torch.Tensor, torch.Tensor],
            apply_rotary_pos_emb(query_states, key_states, cos, sin),
        )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

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

        layer_idx = int(self.layer_idx or 0)
        try:
            attn_output_tokens = paged_backend.forward(
                query_tokens,
                key_tokens,
                value_tokens,
                attention_metadata=attention_metadata,
                scale=cast(float, self.scaling),
                layer_idx=layer_idx,
            )
        except TypeError:
            attn_output_tokens = paged_backend.forward(
                query_tokens,
                key_tokens,
                value_tokens,
                attention_metadata=attention_metadata,
                scale=cast(float, self.scaling),
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

        attn_output = attn_output_tokens.reshape(
            bsz, q_len, num_attention_heads * self.head_dim
        )
        attn_output = cast(torch.Tensor, self.o_proj(attn_output))
        return attn_output, None

    @classmethod
    def get_kv_cache_spec_for_config(cls, config: OlmoeConfig) -> dict[str, int]:
        head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        return {
            "num_kv_heads": int(config.num_key_value_heads),
            "head_dim": int(head_dim),
        }


__all__ = ["OlmoePagedAttention"]
