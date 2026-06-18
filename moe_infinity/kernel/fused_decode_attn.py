"""Decode-phase fused paged attention Triton kernel.

Specialized for decode where each sequence contributes exactly one new query
token attending over its paged KV cache.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_KV": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_KV": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_KV": 128}, num_stages=2, num_warps=8),
    ],
    key=["head_dim", "block_size", "max_blocks_per_seq"],
)
@triton.jit
def _fused_decode_attention_kernel(
    query_ptr,  # [batch, 1, num_heads, head_dim]
    key_ptr,  # [num_blocks, block_size, num_kv_heads, head_dim]
    value_ptr,  # [num_blocks, block_size, num_kv_heads, head_dim]
    block_tables_ptr,  # [batch, max_blocks_per_seq]
    seq_lens_ptr,  # [batch]
    out_ptr,  # [batch, num_heads, head_dim]
    scale,
    head_dim,
    block_size,
    max_blocks_per_seq,
    q_stride_b,
    q_stride_t,
    q_stride_h,
    q_stride_d,
    k_stride_block,
    k_stride_token,
    k_stride_head,
    k_stride_d,
    v_stride_block,
    v_stride_token,
    v_stride_head,
    v_stride_d,
    bt_stride_b,
    bt_stride_block,
    out_stride_b,
    out_stride_h,
    out_stride_d,
    QUERY_GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    offs_d = tl.arange(0, HEAD_DIM)
    out_ptrs = (
        out_ptr
        + batch_idx * out_stride_b
        + head_idx * out_stride_h
        + offs_d * out_stride_d
    )

    seq_len = tl.load(seq_lens_ptr + batch_idx).to(tl.int32)
    if seq_len <= 0:
        tl.store(
            out_ptrs,
            tl.zeros((HEAD_DIM,), dtype=tl.bfloat16),
            mask=offs_d < head_dim,
        )
        return

    kv_head_idx = head_idx // QUERY_GROUP_SIZE
    q_ptrs = (
        query_ptr
        + batch_idx * q_stride_b
        + 0 * q_stride_t
        + head_idx * q_stride_h
        + offs_d * q_stride_d
    )
    q = tl.load(q_ptrs, mask=offs_d < head_dim, other=0.0).to(tl.float32)

    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    m_i = tl.full((), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((), dtype=tl.float32)

    for token_start in range(0, tl.cdiv(seq_len, BLOCK_KV)):
        offs_t = token_start * BLOCK_KV + tl.arange(0, BLOCK_KV)
        token_mask = offs_t < seq_len

        logical_block_idx = offs_t // block_size
        token_in_block = offs_t % block_size
        block_table_ptrs = (
            block_tables_ptr
            + batch_idx * bt_stride_b
            + logical_block_idx * bt_stride_block
        )
        physical_block_idx = tl.load(block_table_ptrs, mask=token_mask, other=0)

        k_ptrs = (
            key_ptr
            + physical_block_idx[:, None] * k_stride_block
            + token_in_block[:, None] * k_stride_token
            + kv_head_idx * k_stride_head
            + offs_d[None, :] * k_stride_d
        )
        kv_mask = token_mask[:, None] & (offs_d[None, :] < head_dim)
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
        logits = tl.sum(k * q[None, :], axis=1) * scale
        logits = tl.where(token_mask, logits, float("-inf"))

        m_ij = tl.max(logits, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(logits - m_new)
        p = tl.where(token_mask, p, 0.0)

        v_ptrs = (
            value_ptr
            + physical_block_idx[:, None] * v_stride_block
            + token_in_block[:, None] * v_stride_token
            + kv_head_idx * v_stride_head
            + offs_d[None, :] * v_stride_d
        )
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

        acc = acc * alpha + tl.sum(v * p[:, None], axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    out = tl.where(l_i > 0, acc / l_i, 0.0)
    tl.store(out_ptrs, out.to(tl.bfloat16), mask=offs_d < head_dim)


def fused_decode_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Launch decode-only single-query paged attention.

    Args:
        query: ``[batch, 1, num_heads, head_dim]`` bf16
        key_cache: ``[num_blocks, block_size, num_kv_heads, head_dim]`` bf16
        value_cache: ``[num_blocks, block_size, num_kv_heads, head_dim]`` bf16
        block_tables: ``[batch, max_blocks_per_seq]`` int32
        seq_lens: ``[batch]`` int32
        scale: attention scaling factor

    Returns:
        ``[batch, num_heads, head_dim]`` bf16
    """
    if query.ndim != 4:
        raise ValueError(
            "query must have shape [batch, 1, num_heads, head_dim]"
        )
    if query.shape[1] != 1:
        raise ValueError("decode attention expects exactly one query token")
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError(
            "key_cache and value_cache must have shape [num_blocks, block_size, num_kv_heads, head_dim]"
        )
    if key_cache.shape != value_cache.shape:
        raise ValueError("key_cache and value_cache must have identical shapes")
    if block_tables.ndim != 2:
        raise ValueError(
            "block_tables must have shape [batch, max_blocks_per_seq]"
        )
    if seq_lens.ndim != 1:
        raise ValueError("seq_lens must have shape [batch]")

    if not (
        query.is_cuda
        and key_cache.is_cuda
        and value_cache.is_cuda
        and block_tables.is_cuda
        and seq_lens.is_cuda
    ):
        raise ValueError("fused_decode_attention requires CUDA tensors")

    if query.dtype != torch.bfloat16:
        raise ValueError(f"query must be bf16, got {query.dtype}")
    if key_cache.dtype != torch.bfloat16 or value_cache.dtype != torch.bfloat16:
        raise ValueError("key_cache and value_cache must be bf16")
    if block_tables.dtype != torch.int32:
        raise ValueError("block_tables must be int32")
    if seq_lens.dtype != torch.int32:
        raise ValueError("seq_lens must be int32")

    batch, _, num_heads, head_dim = query.shape
    num_blocks, block_size, num_kv_heads, kv_head_dim = key_cache.shape
    if kv_head_dim != head_dim:
        raise ValueError("query and KV cache head_dim must match")
    if block_tables.shape[0] != batch:
        raise ValueError("block_tables batch dimension must match query")
    if seq_lens.shape[0] != batch:
        raise ValueError("seq_lens batch dimension must match query")
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if head_dim not in (64, 128):
        raise ValueError(f"head_dim must be 64 or 128, got {head_dim}")
    if num_blocks <= 0:
        raise ValueError("key_cache must contain at least one block")
    if block_tables.shape[1] <= 0:
        raise ValueError("block_tables must contain at least one block slot")
    max_supported_seq_len = block_tables.shape[1] * block_size
    if int(seq_lens.max().item()) > max_supported_seq_len:
        raise ValueError(
            "seq_lens exceed block_tables capacity for the provided block_size"
        )

    query = query.contiguous()
    key_cache = key_cache.contiguous()
    value_cache = value_cache.contiguous()
    block_tables = block_tables.contiguous()
    seq_lens = seq_lens.contiguous()

    output = torch.empty(
        (batch, num_heads, head_dim), dtype=query.dtype, device=query.device
    )
    if output.numel() == 0:
        return output

    query_group_size = num_heads // num_kv_heads
    max_blocks_per_seq = block_tables.shape[1]
    grid = (batch, num_heads)

    _fused_decode_attention_kernel[grid](
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        output,
        float(scale),
        head_dim,
        block_size,
        max_blocks_per_seq,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        block_tables.stride(0),
        block_tables.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        QUERY_GROUP_SIZE=query_group_size,
        HEAD_DIM=head_dim,
    )

    return output


__all__ = ["fused_decode_attention"]
