from __future__ import annotations

import os

import torch

from .paged_attention_ops import paged_attention_fwd
from .sglang_adapter import sglang_topk_softmax as topk_softmax

# Environment variable gate for fused kernels (set to "1" to disable)
_FUSED_KERNELS_DISABLED = (
    os.environ.get("MOE_DISABLE_FUSED_KERNELS", "0") == "1"
)


def launch_fused_softmax_topk_nobias(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    top_k: int,
    normalize_topk: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    from .router import launch_fused_softmax_topk_nobias as _impl

    return _impl(hidden_states, weight, top_k, normalize_topk)


def fused_qkv_proj(
    hidden_states: torch.Tensor,
    weight_qkv: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused QKV projection. Falls back to 3 matmuls if fused kernels disabled."""
    if _FUSED_KERNELS_DISABLED:
        q_dim = num_q_heads * head_dim
        kv_dim = num_kv_heads * head_dim
        prefix_shape = hidden_states.shape[:-1]
        h = hidden_states.reshape(-1, hidden_states.shape[-1])
        out = h @ weight_qkv
        q = out[:, :q_dim].reshape(*prefix_shape, num_q_heads, head_dim)
        k = out[:, q_dim : q_dim + kv_dim].reshape(
            *prefix_shape, num_kv_heads, head_dim
        )
        v = out[:, q_dim + kv_dim :].reshape(
            *prefix_shape, num_kv_heads, head_dim
        )
        return q, k, v
    from .fused_qkv import fused_qkv_proj as _impl

    return _impl(hidden_states, weight_qkv, num_q_heads, num_kv_heads, head_dim)


def fused_ffn(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """Fused Gate+Up+SiLU+Down FFN. Falls back to eager if disabled."""
    if _FUSED_KERNELS_DISABLED:
        import torch.nn.functional as F

        intermediate = F.silu(x @ gate_weight.T) * (x @ up_weight.T)
        return intermediate @ down_weight.T
    from .fused_ffn import fused_ffn as _impl

    return _impl(x, gate_weight, up_weight, down_weight)


def fused_decode_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Fused decode attention with paged KV. Falls back to FlashInfer/SDPA if disabled."""
    if _FUSED_KERNELS_DISABLED:
        return paged_attention_fwd(
            query, key_cache, value_cache, block_tables, seq_lens, scale
        )
    from .fused_decode_attn import fused_decode_attention as _impl

    return _impl(query, key_cache, value_cache, block_tables, seq_lens, scale)


__all__ = [
    "topk_softmax",
    "launch_fused_softmax_topk_nobias",
    "paged_attention_fwd",
    "fused_qkv_proj",
    "fused_ffn",
    "fused_decode_attention",
]
