import math

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip(
    "moe_infinity.kernel.paged_attention_ops",
    reason="paged_attention_ops required",
)

from moe_infinity.kernel.paged_attention_ops import (
    paged_attention_fwd,
    paged_attention_reference,
)
from moe_infinity.runtime.kv_cache_format import (
    allocate_layered_paged_kv_store,
    quantize_tokenwise_symmetric,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no GPU"
)

HEAD_DIMS = [64, 80, 96, 112, 128, 256]
# GQA ratios 1, 2, 4, 8, 16.
HEAD_CONFIGS = [(8, 8), (16, 8), (16, 4), (32, 4), (64, 4)]


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


def make_vllm_cache(
    seq_lens: list[int],
    num_kv_heads: int,
    head_dim: int,
    block_size: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    extra_slots: int = 2,
):
    """vLLM-layout caches with randomly permuted pages and ragged lengths.

    Block-table slots past each sequence hold out-of-range ids to prove the
    kernel never dereferences them."""
    max_blocks = max(1, -(-max(seq_lens) // block_size)) + extra_slots
    num_blocks = len(seq_lens) * max_blocks + 3
    x = 8
    k_cache = torch.randn(
        num_blocks, num_kv_heads, head_dim // x, block_size, x, device=device
    ).to(dtype)
    v_cache = torch.randn(
        num_blocks, num_kv_heads, head_dim, block_size, device=device
    ).to(dtype)
    perm = torch.randperm(num_blocks, device=device)
    block_tables = torch.full(
        (len(seq_lens), max_blocks),
        num_blocks + 1000,
        dtype=torch.int32,
        device=device,
    )
    cursor = 0
    for i, n in enumerate(seq_lens):
        used = -(-n // block_size)
        block_tables[i, :used] = perm[cursor : cursor + used].int()
        cursor += used
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    return k_cache, v_cache, block_tables, seq_lens_t


def sdpa_reference(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    num_kv_heads: int,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-sequence gather + fp32 ``F.scaled_dot_product_attention``."""
    batch, num_heads, head_dim = query.shape
    block_size = v_cache.shape[-1]
    outs = []
    for b in range(batch):
        n = int(seq_lens[b])
        if n == 0:
            outs.append(torch.zeros_like(query[b]))
            continue
        pages = block_tables[b, : -(-n // block_size)].long()
        k = k_cache[pages].permute(1, 0, 3, 2, 4).float()
        k = k.reshape(num_kv_heads, -1, head_dim)[:, :n]
        v = v_cache[pages].permute(1, 0, 3, 2).float()
        v = v.reshape(num_kv_heads, -1, head_dim)[:, :n]
        if k_scale is not None and v_scale is not None:
            ks = k_scale[pages].permute(1, 0, 2).reshape(num_kv_heads, -1)
            vs = v_scale[pages].permute(1, 0, 2).reshape(num_kv_heads, -1)
            k = k * ks[:, :n, None].float()
            v = v * vs[:, :n, None].float()
        out = F.scaled_dot_product_attention(
            query[b].float()[None, :, None],
            k[None],
            v[None],
            scale=scale,
            enable_gqa=True,
        )
        outs.append(out[0, :, 0].to(query.dtype))
    return torch.stack(outs)


@requires_cuda
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("num_heads,num_kv_heads", HEAD_CONFIGS)
@pytest.mark.parametrize(
    "seq_lens", [[1, 17, 333, 0], [5000], [1024] * 6], ids=str
)
def test_paged_attention_matches_sdpa(
    head_dim, num_heads, num_kv_heads, seq_lens
):
    torch.manual_seed(0)
    k, v, bt, lens = make_vllm_cache(seq_lens, num_kv_heads, head_dim)
    q = torch.randn(
        len(seq_lens), num_heads, head_dim, device="cuda"
    ).bfloat16()
    scale = head_dim**-0.5
    out = paged_attention_fwd(q, k, v, bt, lens, scale, num_kv_heads, 16)
    ref = sdpa_reference(q, k, v, bt, lens, scale, num_kv_heads)
    assert out.shape == q.shape and out.dtype == q.dtype
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)


@requires_cuda
@pytest.mark.parametrize("block_size", [8, 32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paged_attention_block_sizes_and_dtypes(block_size, dtype):
    torch.manual_seed(1)
    seq_lens = [3, 200, 4097]
    k, v, bt, lens = make_vllm_cache(
        seq_lens, 4, 128, block_size=block_size, dtype=dtype
    )
    q = torch.randn(3, 32, 128, device="cuda").to(dtype)
    out = paged_attention_fwd(q, k, v, bt, lens, 128**-0.5, 4, block_size)
    ref = sdpa_reference(q, k, v, bt, lens, 128**-0.5, 4)
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)


@requires_cuda
def test_paged_attention_max_seq_len_hint_and_no_host_sync():
    """The CUDA path must not synchronise with the host (CUDA graphs, #316),
    with or without a ``max_seq_len`` hint."""
    torch.manual_seed(2)
    seq_lens = [700, 9000, 1]
    k, v, bt, lens = make_vllm_cache(seq_lens, 2, 256)
    q = torch.randn(3, 16, 256, device="cuda").bfloat16()
    ref = sdpa_reference(q, k, v, bt, lens, 256**-0.5, 2)
    for hint in (None, 9000, 20000):
        paged_attention_fwd(q, k, v, bt, lens, 256**-0.5, 2, 16, hint)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            out = paged_attention_fwd(q, k, v, bt, lens, 256**-0.5, 2, 16, hint)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.testing.assert_close(
            out.float(), ref.float(), atol=2e-2, rtol=2e-2
        )


@requires_cuda
def test_paged_vs_sdpa():
    batch, num_heads, head_dim, seq_len = 2, 8, 64, 32
    block_size = 16
    query = torch.randn(
        batch, num_heads, head_dim, dtype=torch.float16, device="cuda"
    )
    k_cache, v_cache, block_tables, seq_lens = make_paged_kv(
        batch, num_heads, head_dim, seq_len, block_size
    )
    k_cache, v_cache = k_cache.cuda().half(), v_cache.cuda().half()
    block_tables, seq_lens = block_tables.cuda(), seq_lens.cuda()
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
    ref = sdpa_reference(
        query, k_cache, v_cache, block_tables, seq_lens, scale, num_heads
    )
    assert out.shape == query.shape
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@requires_cuda
def test_gqa_config():
    batch, num_heads, num_kv_heads, head_dim, seq_len = 2, 32, 8, 128, 16
    block_size = 16
    query = torch.randn(
        batch, num_heads, head_dim, dtype=torch.float16, device="cuda"
    )
    k_cache, v_cache, block_tables, seq_lens = make_paged_kv(
        batch, num_kv_heads, head_dim, seq_len, block_size
    )
    k_cache, v_cache = k_cache.cuda().half(), v_cache.cuda().half()
    block_tables, seq_lens = block_tables.cuda(), seq_lens.cuda()
    scale = 1.0 / math.sqrt(float(head_dim))
    out = paged_attention_fwd(
        query,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        scale,
        num_kv_heads,
        block_size,
    )
    ref = sdpa_reference(
        query, k_cache, v_cache, block_tables, seq_lens, scale, num_kv_heads
    )
    assert out.shape == query.shape
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.gpu
@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 8), (32, 8)])
def test_int8_paged_attention_matches_dequantized_sdpa(num_heads, num_kv_heads):
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    torch.manual_seed(19)
    batch, head_dim, seq_len, block_size = 2, 128, 33, 16
    query = torch.randn(
        batch, num_heads, head_dim, device="cuda", dtype=torch.float16
    )
    k, v, block_tables, seq_lens = make_paged_kv(
        batch, num_kv_heads, head_dim, seq_len, block_size
    )
    k = k.cuda().half()
    v = v.cuda().half()
    k_vectors = k.permute(0, 3, 1, 2, 4).reshape(-1, num_kv_heads, head_dim)
    v_vectors = v.permute(0, 3, 1, 2).reshape(-1, num_kv_heads, head_dim)
    k_q_vectors, k_scale_vectors = quantize_tokenwise_symmetric(k_vectors)
    v_q_vectors, v_scale_vectors = quantize_tokenwise_symmetric(v_vectors)
    num_blocks = k.shape[0]
    k_q = (
        k_q_vectors.reshape(
            num_blocks, block_size, num_kv_heads, head_dim // 8, 8
        )
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    v_q = v_q_vectors.reshape(
        num_blocks, block_size, num_kv_heads, head_dim
    ).permute(0, 2, 3, 1)
    k_scale = (
        k_scale_vectors.reshape(num_blocks, block_size, num_kv_heads)
        .permute(0, 2, 1)
        .contiguous()
    )
    v_scale = (
        v_scale_vectors.reshape(num_blocks, block_size, num_kv_heads)
        .permute(0, 2, 1)
        .contiguous()
    )
    block_tables, seq_lens = block_tables.cuda(), seq_lens.cuda()
    scale = 1.0 / math.sqrt(head_dim)
    actual = paged_attention_fwd(
        query,
        k_q,
        v_q,
        block_tables,
        seq_lens,
        scale,
        num_kv_heads,
        block_size,
        key_scale=k_scale,
        value_scale=v_scale,
    )
    reference = sdpa_reference(
        query,
        k_q,
        v_q,
        block_tables,
        seq_lens,
        scale,
        num_kv_heads,
        k_scale,
        v_scale,
    )
    assert reference.shape == actual.shape == (batch, num_heads, head_dim)
    torch.testing.assert_close(actual, reference, atol=2e-2, rtol=2e-2)


@requires_cuda
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(4, 4), (32, 2)])
def test_int8_store_views_match_sdpa(head_dim, num_heads, num_kv_heads):
    """int8 KV read in place through the (non-contiguous) kernel views of a
    real ``FormatLayeredKVStore``, ragged lengths, multi-split."""
    torch.manual_seed(3)
    block_size = 16
    seq_lens = [0, 5, 130, 3000]
    pages_per_seq = [-(-n // block_size) for n in seq_lens]
    num_pages = sum(pages_per_seq) + 4
    store = allocate_layered_paged_kv_store(
        owner_id="test",
        format_name="int8_sym",
        num_layers=2,
        num_blocks=num_pages,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        execution_dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    perm = torch.randperm(num_pages, device="cuda")
    max_blocks = max(pages_per_seq) + 1
    block_tables = torch.zeros(
        len(seq_lens), max_blocks, dtype=torch.int32, device="cuda"
    )
    cursor = 0
    for i, (n, used) in enumerate(zip(seq_lens, pages_per_seq)):
        block_tables[i, :used] = perm[cursor : cursor + used].int()
        cursor += used
        if n == 0:
            continue
        pos = torch.arange(n, device="cuda")
        slots = block_tables[i, pos // block_size].long() * block_size + (
            pos % block_size
        )
        store.write_chunk(
            layer_idx=1,
            key_chunk=torch.randn(n, num_kv_heads, head_dim, device="cuda"),
            value_chunk=torch.randn(n, num_kv_heads, head_dim, device="cuda"),
            slot_mapping=slots,
        )
    k_view, v_view, k_scale, v_scale = store.paged_kernel_layer_view(1)
    assert not v_view.is_contiguous()
    lens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    q = torch.randn(len(seq_lens), num_heads, head_dim, device="cuda")
    q = q.bfloat16()
    scale = head_dim**-0.5
    out = paged_attention_fwd(
        q,
        k_view,
        v_view,
        block_tables,
        lens,
        scale,
        num_kv_heads,
        block_size,
        key_scale=k_scale,
        value_scale=v_scale,
    )
    ref = sdpa_reference(
        q,
        k_view,
        v_view,
        block_tables,
        lens,
        scale,
        num_kv_heads,
        k_scale,
        v_scale,
    )
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)


@requires_cuda
def test_paged_attention_kv_pool_over_2pow31_elements():
    """Pages beyond element 2**31 of the pool must be addressed with 64-bit
    offsets (illegal-address fault before #300)."""
    num_kv_heads, head_dim, block_size = 8, 128, 16
    per_block = num_kv_heads * head_dim * block_size
    num_blocks = (1 << 31) // per_block + 64
    need = 2 * num_blocks * per_block * 2
    free, _ = torch.cuda.mem_get_info()
    if free < need + (2 << 30):
        pytest.skip("not enough GPU memory for a >2**31-element KV pool")
    k = torch.empty(
        num_blocks,
        num_kv_heads,
        head_dim // 8,
        block_size,
        8,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v = torch.empty(
        num_blocks,
        num_kv_heads,
        head_dim,
        block_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    used = torch.arange(num_blocks - 40, num_blocks, device="cuda")
    k[used] = torch.randn_like(k[used])
    v[used] = torch.randn_like(v[used])
    block_tables = used.int().view(2, 20)
    lens = torch.tensor([320, 301], dtype=torch.int32, device="cuda")
    q = torch.randn(2, 32, head_dim, device="cuda").bfloat16()
    out = paged_attention_fwd(q, k, v, block_tables, lens, head_dim**-0.5, 8)
    torch.cuda.synchronize()
    ref = sdpa_reference(q, k, v, block_tables, lens, head_dim**-0.5, 8)
    del k, v
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)


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


@pytest.mark.parametrize("num_heads,num_kv_heads", [(4, 4), (16, 2)])
def test_reference_cpu_matches_sdpa(num_heads, num_kv_heads):
    torch.manual_seed(4)
    k, v, bt, lens = make_vllm_cache(
        [0, 9, 40], num_kv_heads, 64, dtype=torch.float32, device="cpu"
    )
    q = torch.randn(3, num_heads, 64)
    out = paged_attention_reference(q, k, v, bt, lens, 0.125, num_kv_heads)
    ref = sdpa_reference(q, k, v, bt, lens, 0.125, num_kv_heads)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
