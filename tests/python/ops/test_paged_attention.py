import math

import pytest
import torch

pytest.importorskip(
    "moe_infinity.kernel.paged_attention_ops",
    reason="paged_attention_ops required",
)

from moe_infinity.kernel.paged_attention_ops import paged_attention_fwd


def make_paged_kv(
    batch: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_blocks = (seq_len + block_size - 1) // block_size * batch
    x = max(1, 16 // (head_dim * 2))
    k_cache = torch.randn(num_blocks, num_heads, head_dim // x, block_size, x)
    v_cache = torch.randn(num_blocks, num_heads, head_dim, block_size)
    block_tables = torch.arange(num_blocks).reshape(batch, -1).int()
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32)
    return k_cache, v_cache, block_tables, seq_lens


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_paged_vs_sdpa():
    batch, num_heads, head_dim, seq_len = 2, 8, 64, 32
    block_size = 16
    query = torch.randn(
        batch,
        num_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    k_cache, v_cache, block_tables, seq_lens = make_paged_kv(
        batch, num_heads, head_dim, seq_len, block_size
    )
    k_cache, v_cache = k_cache.cuda().half(), v_cache.cuda().half()
    scale = 1.0 / math.sqrt(float(head_dim))
    out = paged_attention_fwd(
        query,
        k_cache,
        v_cache,
        block_tables.cuda(),
        seq_lens.cuda(),
        scale,
        num_heads,
        block_size,
    )
    assert out.shape == query.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_gqa_config():
    batch, num_heads, num_kv_heads, head_dim, seq_len = 2, 32, 8, 128, 16
    block_size = 16
    query = torch.randn(
        batch,
        num_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    k_cache, v_cache, block_tables, seq_lens = make_paged_kv(
        batch, num_kv_heads, head_dim, seq_len, block_size
    )
    k_cache, v_cache = k_cache.cuda().half(), v_cache.cuda().half()
    scale = 1.0 / math.sqrt(float(head_dim))
    out = paged_attention_fwd(
        query,
        k_cache,
        v_cache,
        block_tables.cuda(),
        seq_lens.cuda(),
        scale,
        num_kv_heads,
        block_size,
    )
    assert out.shape == query.shape


def test_fallback_import():
    from moe_infinity.kernel.paged_attention_ops import (
        HAS_PAGED_ATTN,
        paged_attention_fwd,
    )

    assert isinstance(HAS_PAGED_ATTN, bool)
    assert callable(paged_attention_fwd)


def test_fallback_cpu():
    batch, num_heads, head_dim, seq_len = 1, 4, 64, 8
    block_size = 8
    query = torch.randn(batch, num_heads, head_dim)
    k_cache, v_cache, block_tables, seq_lens = make_paged_kv(
        batch, num_heads, head_dim, seq_len, block_size
    )
    scale = 1.0 / math.sqrt(float(head_dim))
    out = paged_attention_fwd(
        query,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        scale,
        num_heads,
        block_size,
    )
    assert out.shape == query.shape
