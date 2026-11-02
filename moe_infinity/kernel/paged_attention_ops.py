from __future__ import annotations

import importlib
import logging
from typing import Optional, Protocol, cast

import torch

logger = logging.getLogger(__name__)


class _PagedAttnModule(Protocol):
    def paged_attention_v1(
        self,
        out: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        num_kv_heads: int,
        scale: float,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        block_size: int,
        max_seq_len: int,
    ) -> None: ...


def _load_paged_attn_module() -> tuple[_PagedAttnModule | None, bool]:
    try:
        module = importlib.import_module("moe_infinity._paged_attn")
        return cast(_PagedAttnModule, cast(object, module)), True
    except ImportError:
        return None, False


# The legacy CUDA extension is only probed for capability reporting; decode
# attention runs on the split-KV Triton kernel in ``fused_decode_attn``.
_paged_attn_ops, HAS_PAGED_ATTN = _load_paged_attn_module()

_KERNEL_KV_DTYPES = (torch.float16, torch.bfloat16, torch.int8)


def probe_native_int8_binding() -> tuple[bool, str | None]:
    if _paged_attn_ops is None:
        return False, "native_int8_module_unavailable"
    binding = getattr(_paged_attn_ops, "paged_attention_int8_v1", None)
    if not callable(binding):
        return False, "native_int8_binding_missing"
    return True, None


def paged_attention_fwd(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    num_kv_heads: int,
    block_size: int = 16,
    max_seq_len: Optional[int] = None,
    key_scale: torch.Tensor | None = None,
    value_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single-token decode attention over a paged KV cache (vLLM layout).

    Args:
        query: ``[num_seqs, num_heads, head_dim]``.
        key_cache: ``[num_blocks, num_kv_heads, head_dim / x, block_size, x]``.
        value_cache: ``[num_blocks, num_kv_heads, head_dim, block_size]``.
        block_tables: ``[num_seqs, max_blocks_per_seq]``.
        seq_lens: ``[num_seqs]`` context lengths.
        key_scale / value_scale: per-token fp16 scales
            ``[num_blocks, num_kv_heads, block_size]`` for int8 caches.
        max_seq_len: optional host-side upper bound of ``seq_lens`` (sizes
            the KV split count; never read back from the device).

    CUDA inputs run the split-KV Triton kernel without any device -> host
    synchronisation. Other inputs use :func:`paged_attention_reference`.
    Caches may be non-contiguous views; they are read in place.
    """
    if query.ndim != 3:
        raise ValueError(
            "query must have shape [num_seqs, num_heads, head_dim]"
        )
    if block_tables.ndim != 2:
        raise ValueError(
            "block_tables must have shape [num_seqs, max_blocks_per_seq]"
        )
    if seq_lens.ndim != 1:
        raise ValueError("seq_lens must have shape [num_seqs]")
    if key_cache.ndim != 5 or value_cache.ndim != 4:
        raise ValueError(
            "key_cache must be [num_blocks, num_kv_heads, head_dim/x, "
            "block_size, x] and value_cache [num_blocks, num_kv_heads, "
            "head_dim, block_size]"
        )
    if (key_scale is None) != (value_scale is None):
        raise ValueError(
            "key_scale and value_scale must both be provided or both be None"
        )
    if query.shape[1] % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")

    is_int8 = key_scale is not None
    if is_int8 and (
        key_cache.dtype != torch.int8 or value_cache.dtype != torch.int8
    ):
        raise ValueError(
            "int8 paged attention requires int8 key/value payloads"
        )

    if _use_kernel(query, key_cache, value_cache, block_tables, seq_lens):
        if key_scale is not None and not (
            key_scale.is_cuda
            and value_scale is not None
            and value_scale.is_cuda
        ):
            raise ValueError("key_scale/value_scale must be CUDA tensors")
        from .fused_decode_attn import (
            _key_layout,
            _value_layout,
            split_kv_decode_attention,
        )

        key_layout = _key_layout(key_cache)
        if key_layout[-1] != int(block_size):
            raise ValueError("key_cache block_size mismatch")
        return split_kv_decode_attention(
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            scale,
            num_kv_heads=int(num_kv_heads),
            key_layout=key_layout,
            value_layout=_value_layout(value_cache),
            key_scale=key_scale,
            value_scale=value_scale,
            max_seq_len=max_seq_len,
        )

    return paged_attention_reference(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        scale=scale,
        num_kv_heads=num_kv_heads,
        key_scale=key_scale,
        value_scale=value_scale,
    )


def _use_kernel(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> bool:
    if not (
        query.is_cuda
        and key_cache.is_cuda
        and value_cache.is_cuda
        and block_tables.is_cuda
        and seq_lens.is_cuda
    ):
        return False
    if key_cache.dtype not in _KERNEL_KV_DTYPES:
        return False
    if key_cache.dtype != value_cache.dtype:
        return False
    try:
        from .fused_decode_attn import supports_head_dim
    except ImportError:  # Triton unavailable
        return False
    return supports_head_dim(int(query.shape[-1]))


def paged_attention_reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    num_kv_heads: int,
    key_scale: torch.Tensor | None = None,
    value_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Vectorised fp32 torch reference of :func:`paged_attention_fwd`.

    Gathers every block-table slot (out-of-range ids are clamped and masked
    by ``seq_lens``), so it costs ``O(num_seqs * max_blocks)`` memory. Used
    for CPU tensors and tests; no host synchronisation.
    """
    batch, num_heads, head_dim = query.shape
    group = num_heads // num_kv_heads
    num_blocks = key_cache.shape[0]
    block_size = value_cache.shape[-1]
    bt = block_tables.to(key_cache.device, torch.long).clamp(0, num_blocks - 1)
    max_blocks = bt.shape[1]
    tokens = max_blocks * block_size

    # [B, M, KVH, D/x, bs, x] -> [B, KVH, M*bs, D]
    k = key_cache[bt].permute(0, 2, 1, 4, 3, 5).float()
    k = k.reshape(batch, num_kv_heads, tokens, head_dim)
    # [B, M, KVH, D, bs] -> [B, KVH, M*bs, D]
    v = value_cache[bt].permute(0, 2, 1, 4, 3).float()
    v = v.reshape(batch, num_kv_heads, tokens, head_dim)
    if key_scale is not None and value_scale is not None:
        ks = key_scale[bt].permute(0, 2, 1, 3).float()
        vs = value_scale[bt].permute(0, 2, 1, 3).float()
        k = k * ks.reshape(batch, num_kv_heads, tokens, 1)
        v = v * vs.reshape(batch, num_kv_heads, tokens, 1)

    q = query.to(k.device).float().reshape(batch, num_kv_heads, group, head_dim)
    logits = torch.einsum("bkgd,bkld->bkgl", q, k) * float(scale)
    positions = torch.arange(tokens, device=k.device)
    valid = positions[None, :] < seq_lens.to(k.device)[:, None]
    logits = logits.masked_fill(~valid[:, None, None, :], float("-inf"))
    probs = torch.softmax(logits, dim=-1).nan_to_num(0.0)
    out = torch.einsum("bkgl,bkld->bkgd", probs, v)
    return out.reshape(batch, num_heads, head_dim).to(query.dtype)
