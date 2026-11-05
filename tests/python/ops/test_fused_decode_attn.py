import math

import pytest
import torch
import torch.nn.functional as F

from tests.python.ops.conftest import (
    BF16_ATOL,
    BF16_RTOL,
    requires_cuda,
    requires_triton,
)


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


def _make_paged_kv(
    batch: int,
    seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    blocks_per_seq = math.ceil(seq_len / block_size)
    num_blocks = batch * blocks_per_seq
    key_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    value_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.float32,
    ).to(torch.bfloat16)

    physical_blocks = torch.randperm(
        num_blocks, device=device, dtype=torch.int64
    )
    block_tables = physical_blocks.reshape(batch, blocks_per_seq).to(
        torch.int32
    )
    seq_lens = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
    return key_cache, value_cache, block_tables, seq_lens


def _reference_decode_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    batch, _, num_heads, head_dim = query.shape
    _, block_size, num_kv_heads, _ = key_cache.shape
    query_group_size = num_heads // num_kv_heads

    outputs: list[torch.Tensor] = []
    for batch_idx in range(batch):
        seq_len = int(seq_lens[batch_idx].item())
        blocks_per_seq = math.ceil(seq_len / block_size)
        k_blocks: list[torch.Tensor] = []
        v_blocks: list[torch.Tensor] = []

        for block_idx in range(blocks_per_seq):
            physical_block = int(block_tables[batch_idx, block_idx].item())
            valid_tokens = min(block_size, seq_len - block_idx * block_size)
            k_blocks.append(key_cache[physical_block, :valid_tokens])
            v_blocks.append(value_cache[physical_block, :valid_tokens])

        key = torch.cat(k_blocks, dim=0).permute(1, 0, 2)
        value = torch.cat(v_blocks, dim=0).permute(1, 0, 2)
        if query_group_size > 1:
            key = key.repeat_interleave(query_group_size, dim=0)
            value = value.repeat_interleave(query_group_size, dim=0)

        q = query[batch_idx, 0].unsqueeze(1)
        out = _run_sdpa(q, key, value, scale)
        outputs.append(out.squeeze(1))

    return torch.stack(outputs, dim=0).reshape(batch, num_heads, head_dim)


@requires_cuda
@requires_triton
@pytest.mark.parametrize("batch", [1, 4, 8])
@pytest.mark.parametrize("seq_len", [32, 128, 512, 2048])
@pytest.mark.parametrize("num_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("head_dim", [80, 128, 256])
@pytest.mark.parametrize("block_size", [16])
def test_fused_decode_attention_matches_reference(
    batch: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> None:
    from moe_infinity.kernel.fused_decode_attn import fused_decode_attention

    device = torch.device("cuda")
    query = torch.randn(
        batch,
        1,
        num_heads,
        head_dim,
        device=device,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    key_cache, value_cache, block_tables, seq_lens = _make_paged_kv(
        batch=batch,
        seq_len=seq_len,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        device=device,
    )
    scale = 1.0 / math.sqrt(float(head_dim))

    output = fused_decode_attention(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
    )
    reference = _reference_decode_attention(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
    )

    assert output.shape == (batch, num_heads, head_dim)
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(
        output.float(),
        reference.float(),
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
    )


@requires_cuda
@requires_triton
def test_fused_decode_attention_ragged_seq_lens() -> None:
    from moe_infinity.kernel.fused_decode_attn import fused_decode_attention

    torch.manual_seed(5)
    device = torch.device("cuda")
    batch, num_heads, num_kv_heads, head_dim, block_size = 4, 32, 4, 128, 16
    key_cache, value_cache, block_tables, _ = _make_paged_kv(
        batch=batch,
        seq_len=4096,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        device=device,
    )
    seq_lens = torch.tensor([1, 4096, 17, 1000], device=device).int()
    query = torch.randn(batch, 1, num_heads, head_dim, device=device)
    query = query.to(torch.bfloat16)
    scale = head_dim**-0.5
    output = fused_decode_attention(
        query, key_cache, value_cache, block_tables, seq_lens, scale
    )
    reference = _reference_decode_attention(
        query, key_cache, value_cache, block_tables, seq_lens, scale
    )
    torch.testing.assert_close(
        output.float(), reference.float(), atol=BF16_ATOL, rtol=BF16_RTOL
    )


@requires_cuda
@requires_triton
def test_fused_decode_attention_kv_pool_over_2pow31_elements() -> None:
    """#300: offsets into a KV pool of more than 2**31 elements used to be
    computed in 32 bits and fault with an illegal memory access."""
    from moe_infinity.kernel.fused_decode_attn import fused_decode_attention

    num_kv_heads, head_dim, block_size = 8, 128, 16
    per_block = block_size * num_kv_heads * head_dim
    num_blocks = (1 << 31) // per_block + 64
    free, _ = torch.cuda.mem_get_info()
    if free < 2 * num_blocks * per_block * 2 + (2 << 30):
        pytest.skip("not enough GPU memory for a >2**31-element KV pool")
    device = torch.device("cuda")
    shape = (num_blocks, block_size, num_kv_heads, head_dim)
    key_cache = torch.empty(shape, dtype=torch.bfloat16, device=device)
    value_cache = torch.empty(shape, dtype=torch.bfloat16, device=device)
    assert key_cache.numel() > 2**31
    used = torch.arange(num_blocks - 32, num_blocks, device=device)
    key_cache[used] = torch.randn_like(key_cache[used])
    value_cache[used] = torch.randn_like(value_cache[used])
    block_tables = used.to(torch.int32).view(2, 16)
    seq_lens = torch.tensor([256, 250], dtype=torch.int32, device=device)
    query = torch.randn(2, 1, 64, head_dim, device=device).to(torch.bfloat16)
    scale = head_dim**-0.5
    output = fused_decode_attention(
        query, key_cache, value_cache, block_tables, seq_lens, scale
    )
    torch.cuda.synchronize()
    reference = _reference_decode_attention(
        query, key_cache, value_cache, block_tables, seq_lens, scale
    )
    del key_cache, value_cache
    torch.testing.assert_close(
        output.float(), reference.float(), atol=BF16_ATOL, rtol=BF16_RTOL
    )


@pytest.mark.parametrize("num_kv_heads", [8, 32])
def test_fused_decode_attention_falls_back_to_eager_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    num_kv_heads: int,
) -> None:
    """``MOE_DISABLE_FUSED_KERNELS=1`` must still compute decode attention.

    The fallback runs without CUDA or Triton, so this covers the CPU-only
    path that the parametrized kernel test above cannot reach.
    """
    from moe_infinity import kernel

    monkeypatch.setattr(kernel, "_FUSED_KERNELS_DISABLED", True)

    batch, seq_len, num_heads, head_dim, block_size = 2, 40, 32, 128, 16
    device = torch.device("cpu")
    query = torch.randn(
        batch,
        1,
        num_heads,
        head_dim,
        device=device,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    key_cache, value_cache, block_tables, seq_lens = _make_paged_kv(
        batch=batch,
        seq_len=seq_len,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        device=device,
    )
    scale = 1.0 / math.sqrt(float(head_dim))

    output = kernel.fused_decode_attention(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
    )
    reference = _reference_decode_attention(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
    )

    assert output.shape == (batch, num_heads, head_dim)
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(
        output.float(),
        reference.float(),
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
    )


@pytest.fixture
def previous_path(monkeypatch):
    """Dispatch as on a GPU other than the RTX PRO 6000."""
    from moe_infinity.kernel import fused_decode_attn

    monkeypatch.setattr(
        fused_decode_attn, "use_pro6000_decode_attention", lambda device: False
    )


@requires_cuda
@requires_triton
@pytest.mark.parametrize("head_dim", [64, 128])
def test_previous_kernel_on_other_gpus_matches_reference(
    previous_path, head_dim
) -> None:
    from moe_infinity.kernel.fused_decode_attn import fused_decode_attention

    torch.manual_seed(7)
    device = torch.device("cuda")
    key_cache, value_cache, block_tables, _ = _make_paged_kv(
        batch=3,
        seq_len=1024,
        num_kv_heads=4,
        head_dim=head_dim,
        block_size=16,
        device=device,
    )
    seq_lens = torch.tensor([1, 1024, 333], device=device).int()
    query = torch.randn(3, 1, 32, head_dim, device=device).to(torch.bfloat16)
    scale = head_dim**-0.5
    output = fused_decode_attention(
        query, key_cache, value_cache, block_tables, seq_lens, scale
    )
    reference = _reference_decode_attention(
        query, key_cache, value_cache, block_tables, seq_lens, scale
    )
    torch.testing.assert_close(
        output.float(), reference.float(), atol=BF16_ATOL, rtol=BF16_RTOL
    )


@requires_cuda
@requires_triton
def test_previous_kernel_keeps_head_dim_limits(previous_path) -> None:
    from moe_infinity.kernel.fused_decode_attn import fused_decode_attention

    device = torch.device("cuda")
    key_cache, value_cache, block_tables, seq_lens = _make_paged_kv(
        1, 16, 2, 256, 16, device
    )
    query = torch.randn(1, 1, 8, 256, device=device).to(torch.bfloat16)
    with pytest.raises(ValueError, match="head_dim"):
        fused_decode_attention(
            query, key_cache, value_cache, block_tables, seq_lens, 0.1
        )


@requires_cuda
@requires_triton
def test_previous_kernel_kv_pool_over_2pow31_elements(previous_path) -> None:
    """The #300 64-bit offset fix also applies to the kernel other GPUs run."""
    test_fused_decode_attention_kv_pool_over_2pow31_elements()
