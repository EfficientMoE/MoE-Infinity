"""Paged decode attention dispatch.

Hardware: on the RTX PRO 6000 (``use_pro6000_decode_attention``) every CUDA
call runs the split-KV Triton kernel tuned on that card
(``fused_decode_attn.split_kv_decode_attention_pro6000``). Every other CUDA
device runs the previous paths: ``paged_attention_v1`` (head_dim <= 128),
``paged_attention_int8_v1`` (int8 KV) and the per-sequence SDPA fallback
(head_dim 256 or no extension). The generic changes stay on everywhere: no
``seq_lens.max().item()`` when ``max_seq_len`` is omitted, and CPU tensors use
the vectorised ``paged_attention_reference``.
"""

from __future__ import annotations

import importlib
import logging
from typing import Optional, Protocol, cast

import torch
import torch.nn.functional as F

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


_paged_attn_ops, HAS_PAGED_ATTN = _load_paged_attn_module()

# Head sizes the previous paged_attention_v1 kernel handles (other devices
# than the RTX PRO 6000); head_dim 256 uses the SDPA fallback there.
_CUDA_PAGED_ATTN_HEAD_SIZES = frozenset({64, 80, 96, 112, 128})

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

    on_cuda = (
        query.is_cuda
        and key_cache.is_cuda
        and value_cache.is_cuda
        and block_tables.is_cuda
        and seq_lens.is_cuda
    )
    if on_cuda and not _use_pro6000_kernel(query, key_cache, value_cache):
        return _paged_attention_previous(
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            scale,
            num_kv_heads,
            block_size,
            max_seq_len,
            key_scale,
            value_scale,
        )
    if on_cuda:
        if key_scale is not None and not (
            key_scale.is_cuda
            and value_scale is not None
            and value_scale.is_cuda
        ):
            raise ValueError("key_scale/value_scale must be CUDA tensors")
        from .fused_decode_attn import (
            _key_layout,
            _value_layout,
            split_kv_decode_attention_pro6000,
        )

        key_layout = _key_layout(key_cache)
        if key_layout[-1] != int(block_size):
            raise ValueError("key_cache block_size mismatch")
        return split_kv_decode_attention_pro6000(
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


def _use_pro6000_kernel(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> bool:
    if key_cache.dtype not in _KERNEL_KV_DTYPES:
        return False
    if key_cache.dtype != value_cache.dtype:
        return False
    try:
        from .fused_decode_attn import (
            supports_head_dim,
            use_pro6000_decode_attention,
        )
    except ImportError:  # Triton unavailable
        return False
    return use_pro6000_decode_attention(query.device) and supports_head_dim(
        int(query.shape[-1])
    )


def _paged_attention_previous(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    num_kv_heads: int,
    block_size: int,
    max_seq_len: Optional[int],
    key_scale: torch.Tensor | None,
    value_scale: torch.Tensor | None,
) -> torch.Tensor:
    """Dispatch used before the split-KV kernel, on every device except the
    RTX PRO 6000. The CUDA kernels index pages with 64-bit offsets."""
    if max_seq_len is None:
        # Upper bound from the block table: the kernels only validate it,
        # and reading seq_lens.max() would synchronise with the host.
        max_seq_len = block_tables.shape[1] * int(block_size)
    if key_scale is not None and value_scale is not None:
        native_int8 = getattr(_paged_attn_ops, "paged_attention_int8_v1", None)
        if HAS_PAGED_ATTN and callable(native_int8):
            out = torch.zeros_like(query)
            native_int8(
                out,
                query,
                key_cache,
                value_cache,
                key_scale,
                value_scale,
                int(num_kv_heads),
                float(scale),
                block_tables.int(),
                seq_lens.int(),
                int(block_size),
                int(max_seq_len),
            )
            return out
        return _paged_attention_sdpa_fallback(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            scale=scale,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            key_scale=key_scale,
            value_scale=value_scale,
        )
    if (
        HAS_PAGED_ATTN
        and _paged_attn_ops is not None
        and query.shape[-1] in _CUDA_PAGED_ATTN_HEAD_SIZES
    ):
        out = torch.zeros_like(query)
        _paged_attn_ops.paged_attention_v1(
            out,
            query,
            key_cache,
            value_cache,
            int(num_kv_heads),
            float(scale),
            block_tables.int(),
            seq_lens.int(),
            int(block_size),
            int(max_seq_len),
        )
        return out
    return _paged_attention_sdpa_fallback(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        scale=scale,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
    )


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


def _run_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    try:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=scale,
            is_causal=False,
        )
    except TypeError:
        return F.scaled_dot_product_attention(
            query * scale,
            key,
            value,
            is_causal=False,
        )


def _paged_attention_sdpa_fallback(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    num_kv_heads: int,
    block_size: int,
    key_scale: torch.Tensor | None = None,
    value_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size, num_heads, _ = query.shape
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")

    seq_lens_list = seq_lens.tolist()
    block_tables_list = block_tables.tolist()

    is_int8 = key_scale is not None and value_scale is not None
    query_dtype = query.dtype
    head_ratio = num_heads // num_kv_heads
    outputs: list[torch.Tensor] = []
    for seq_idx in range(batch_size):
        seq_len = int(seq_lens_list[seq_idx])
        q = query[seq_idx : seq_idx + 1]

        if seq_len <= 0:
            outputs.append(torch.zeros_like(q))
            continue

        num_blocks = (seq_len + block_size - 1) // block_size
        k_list: list[torch.Tensor] = []
        v_list: list[torch.Tensor] = []

        for b in range(num_blocks):
            phys_block = int(block_tables_list[seq_idx][b])
            tokens_in_block = min(block_size, seq_len - b * block_size)

            k_block = key_cache[phys_block]
            v_block = value_cache[phys_block]
            nkv, hd_x, _, x = k_block.shape
            k_block = k_block.permute(0, 2, 1, 3).reshape(
                nkv, block_size, hd_x * x
            )
            k_block = k_block[:, :tokens_in_block, :]
            v_block = v_block[:, :, :tokens_in_block].permute(0, 2, 1)

            if is_int8:
                k_block = k_block.to(torch.float32)
                v_block = v_block.to(torch.float32)
                ks = (
                    key_scale[phys_block, :, :tokens_in_block]
                    .to(torch.float32)
                    .unsqueeze(-1)
                )
                vs = (
                    value_scale[phys_block, :, :tokens_in_block]
                    .to(torch.float32)
                    .unsqueeze(-1)
                )
                k_block = k_block * ks
                v_block = v_block * vs

            k_list.append(k_block)
            v_list.append(v_block)

        k = torch.cat(k_list, dim=1)
        v = torch.cat(v_list, dim=1)
        if head_ratio > 1:
            k = k.repeat_interleave(head_ratio, dim=0)
            v = v.repeat_interleave(head_ratio, dim=0)

        if is_int8:
            q_sdpa = q.squeeze(0).to(torch.float32).unsqueeze(0).unsqueeze(2)
            k_sdpa = k.unsqueeze(0)
            v_sdpa = v.unsqueeze(0)
            out_fp32 = _run_sdpa(q_sdpa, k_sdpa, v_sdpa, float(scale))
            outputs.append(out_fp32[0, :, 0, :].to(query_dtype).unsqueeze(0))
        else:
            q_sdpa = q.squeeze(0).unsqueeze(1)
            out = _run_sdpa(q_sdpa, k, v, float(scale))
            outputs.append(out.squeeze(1).unsqueeze(0))

    return torch.cat(outputs, dim=0)
