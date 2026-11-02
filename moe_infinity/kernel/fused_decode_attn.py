"""Split-KV (flash-decoding) paged attention for the decode phase.

Each sequence contributes exactly one new query token per step. One program
handles one (KV split, KV head, sequence) triple: it loads every query head of
the GQA group at once, so each KV head is read from DRAM once per step for the
whole group, and walks its slice of the context in tiles of whole pages with
``tl.dot`` + online softmax. A second, tiny kernel merges the splits through
their log-sum-exp; with a single split the main kernel writes the output
directly.

The cache layout is described by strides, so the same kernel reads

* the vLLM layout used by ``PagedAttentionBackend``:
  K ``[blocks, kv_heads, D / x, block, x]``, V ``[blocks, kv_heads, D, block]``;
* permuted views of a token-major store (``FormatLayeredKVStore`` int8 views);
* NHD ``[blocks, block, kv_heads, D]`` (``fused_decode_attention``).

Element ``(page, kv_head, token, d)`` lives at
``page * page_stride + kv_head * S_H + token * S_T + (d // X) * S_DO
+ (d % X) * S_DI``. Only the page stride is a runtime (64-bit) value; the
in-page strides are compile-time constants so the compiler can see the
contiguous axis and vectorise the loads.

The launcher never reads device memory: the number of splits is picked on the
host from the batch geometry and an upper bound of the context length, and each
sequence derives its own split length from ``seq_lens`` on the device.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

_LOG2E = 1.4426950408889634
_MAX_SPLITS = 256
_SUPPORTED_HEAD_DIMS = (64, 80, 96, 112, 128, 256)


@triton.jit
def _load_page_tile(
    ptr,
    page_stride,
    page,
    kv_head,
    offs_t,
    offs_d_mem,
    mask,
    S_H: tl.constexpr,
    S_T: tl.constexpr,
    PAGES: tl.constexpr,
):
    """Load ``[PAGES * PAGE, D_PAD]`` elements of one KV head.

    With one page per tile the page id is a scalar and this is a plain
    strided 2-D load; otherwise each page is a slab of a 3-D load (so the
    in-page contiguity stays visible to the compiler) reshaped to 2-D."""
    if PAGES == 1:
        return tl.load(
            ptr
            + page * page_stride
            + kv_head * S_H
            + offs_t[:, None] * S_T
            + offs_d_mem[None, :],
            mask=mask,
            other=0,
        )
    else:
        x = tl.load(
            ptr
            + (page * page_stride + kv_head * S_H)[:, None, None]
            + offs_t[None, :, None] * S_T
            + offs_d_mem[None, None, :],
            mask=mask,
            other=0,
        )
        return tl.reshape(x, (x.shape[0] * x.shape[1], x.shape[2]))


@triton.jit
def _load_page_scales(
    ptr,
    page_stride,
    page,
    kv_head,
    offs_t,
    mask,
    S_H: tl.constexpr,
    S_T: tl.constexpr,
    PAGES: tl.constexpr,
):
    if PAGES == 1:
        x = tl.load(
            ptr + page * page_stride + kv_head * S_H + offs_t * S_T,
            mask=mask,
            other=0.0,
        )
        return x.to(tl.float32)
    else:
        x = tl.load(
            ptr
            + (page * page_stride + kv_head * S_H)[:, None]
            + offs_t[None, :] * S_T,
            mask=mask,
            other=0.0,
        )
        return tl.reshape(x, (x.shape[0] * x.shape[1],)).to(tl.float32)


@triton.jit
def _split_kv_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    ks_ptr,
    vs_ptr,
    bt_ptr,
    lens_ptr,
    out_ptr,
    part_o_ptr,
    part_lse_ptr,
    qk_scale,
    num_splits,
    num_heads,
    stride_q_b,
    stride_q_h,
    k_page_stride,
    v_page_stride,
    ks_page_stride,
    vs_page_stride,
    stride_bt_b,
    stride_o_b,
    stride_o_h,
    GROUP: tl.constexpr,
    G_PAD: tl.constexpr,
    D: tl.constexpr,
    D_PAD: tl.constexpr,
    PAGE: tl.constexpr,
    PAGES: tl.constexpr,
    K_SH: tl.constexpr,
    K_ST: tl.constexpr,
    K_SDO: tl.constexpr,
    K_SDI: tl.constexpr,
    K_X: tl.constexpr,
    V_SH: tl.constexpr,
    V_ST: tl.constexpr,
    V_SDO: tl.constexpr,
    V_SDI: tl.constexpr,
    V_X: tl.constexpr,
    KS_SH: tl.constexpr,
    KS_ST: tl.constexpr,
    VS_SH: tl.constexpr,
    VS_ST: tl.constexpr,
    IS_INT8: tl.constexpr,
    DIRECT: tl.constexpr,
):
    split = tl.program_id(0)
    kv_head = tl.program_id(1)
    seq = tl.program_id(2).to(tl.int64)
    BLOCK_N: tl.constexpr = PAGES * PAGE

    # Each sequence splits its own context evenly into num_splits ranges of
    # whole tiles; programs past the end store an empty partial.
    seq_len = tl.load(lens_ptr + seq)
    num_tiles = tl.cdiv(seq_len, BLOCK_N)
    tiles_per_split = tl.cdiv(num_tiles, num_splits)
    tile_lo = split * tiles_per_split
    tile_hi = tl.minimum(tile_lo + tiles_per_split, num_tiles)

    offs_g = tl.arange(0, G_PAD)
    offs_d = tl.arange(0, D_PAD)
    g_mask = offs_g < GROUP
    d_mask = offs_d < D
    heads = kv_head * GROUP + offs_g

    q = tl.load(
        q_ptr
        + seq * stride_q_b
        + heads[:, None] * stride_q_h
        + offs_d[None, :],
        mask=g_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    dot_dtype = q.dtype

    offs_t = tl.arange(0, PAGE)
    offs_p = tl.arange(0, PAGES)
    k_d = (offs_d // K_X) * K_SDO + (offs_d % K_X) * K_SDI
    v_d = (offs_d // V_X) * V_SDO + (offs_d % V_X) * V_SDI
    bt_row = bt_ptr + seq * stride_bt_b

    m_i = tl.full((G_PAD,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((G_PAD,), dtype=tl.float32)
    acc = tl.zeros((G_PAD, D_PAD), dtype=tl.float32)

    for tile in range(tile_lo, tile_hi):
        if PAGES == 1:
            page = tl.load(bt_row + tile).to(tl.int64)
            tok_mask = tile * PAGE + offs_t < seq_len
            valid = tok_mask
            if D == D_PAD:
                kv_mask = tok_mask[:, None]
            else:
                kv_mask = tok_mask[:, None] & d_mask[None, :]
        else:
            logical_page = tile * PAGES + offs_p
            page = tl.load(
                bt_row + logical_page,
                mask=logical_page * PAGE < seq_len,
                other=0,
            ).to(tl.int64)
            tok_mask = logical_page[:, None] * PAGE + offs_t[None, :] < seq_len
            valid = tl.reshape(tok_mask, (BLOCK_N,))
            if D == D_PAD:
                kv_mask = tok_mask[:, :, None]
            else:
                kv_mask = tok_mask[:, :, None] & d_mask[None, None, :]

        k = _load_page_tile(
            k_ptr,
            k_page_stride,
            page,
            kv_head,
            offs_t,
            k_d,
            kv_mask,
            K_SH,
            K_ST,
            PAGES,
        )
        v = _load_page_tile(
            v_ptr,
            v_page_stride,
            page,
            kv_head,
            offs_t,
            v_d,
            kv_mask,
            V_SH,
            V_ST,
            PAGES,
        )

        qk = tl.dot(q, tl.trans(k.to(dot_dtype)))
        if IS_INT8:
            k_scale = _load_page_scales(
                ks_ptr,
                ks_page_stride,
                page,
                kv_head,
                offs_t,
                tok_mask,
                KS_SH,
                KS_ST,
                PAGES,
            )
            qk = qk * k_scale[None, :]
        qk = qk * qk_scale
        qk = tl.where(valid[None, :], qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new
        if IS_INT8:
            v_scale = _load_page_scales(
                vs_ptr,
                vs_page_stride,
                page,
                kv_head,
                offs_t,
                tok_mask,
                VS_SH,
                VS_ST,
                PAGES,
            )
            p = p * v_scale[None, :]
        acc = acc * alpha[:, None] + tl.dot(p.to(dot_dtype), v.to(dot_dtype))

    has = l_i > 0.0
    o = acc / tl.where(has, l_i, 1.0)[:, None]
    if DIRECT:
        tl.store(
            out_ptr
            + seq * stride_o_b
            + heads[:, None] * stride_o_h
            + offs_d[None, :],
            o.to(out_ptr.dtype.element_ty),
            mask=g_mask[:, None] & d_mask[None, :],
        )
    else:
        row = (seq * num_heads + heads) * num_splits + split
        tl.store(
            part_o_ptr + row[:, None] * D_PAD + offs_d[None, :],
            o,
            mask=g_mask[:, None],
        )
        lse = tl.where(has, m_i + tl.log2(tl.where(has, l_i, 1.0)), -1e30)
        tl.store(part_lse_ptr + row, lse, mask=g_mask)


@triton.jit
def _merge_splits_kernel(
    part_o_ptr,
    part_lse_ptr,
    out_ptr,
    num_splits,
    num_heads,
    stride_o_b,
    stride_o_h,
    D: tl.constexpr,
    D_PAD: tl.constexpr,
    D_BLOCK: tl.constexpr,
    S_CHUNK: tl.constexpr,
):
    """Online log-sum-exp merge of the splits of one (sequence, head, D_BLOCK
    slice of head_dim).

    Empty splits carry lse = -1e30 and o = 0, so a sequence of length 0
    merges to zeros."""
    head = tl.program_id(0)
    seq = tl.program_id(1).to(tl.int64)
    offs_d = tl.program_id(2) * D_BLOCK + tl.arange(0, D_BLOCK)
    offs_c = tl.arange(0, S_CHUNK)
    base = (seq * num_heads + head) * num_splits

    m_run = tl.full((), -1e30, dtype=tl.float32)
    w_sum = tl.zeros((), dtype=tl.float32)
    acc = tl.zeros((D_BLOCK,), dtype=tl.float32)
    for start in range(0, num_splits, S_CHUNK):
        s = start + offs_c
        s_mask = s < num_splits
        lse = tl.load(part_lse_ptr + base + s, mask=s_mask, other=-1e30)
        o = tl.load(
            part_o_ptr + (base + s)[:, None] * D_PAD + offs_d[None, :],
            mask=s_mask[:, None],
            other=0.0,
        )
        m_new = tl.maximum(m_run, tl.max(lse, axis=0))
        alpha = tl.exp2(m_run - m_new)
        w = tl.where(s_mask, tl.exp2(lse - m_new), 0.0)
        acc = acc * alpha + tl.sum(o * w[:, None], axis=0)
        w_sum = w_sum * alpha + tl.sum(w, axis=0)
        m_run = m_new
    acc = acc / tl.where(w_sum > 0.0, w_sum, 1.0)
    tl.store(
        out_ptr + seq * stride_o_b + head * stride_o_h + offs_d,
        acc.to(out_ptr.dtype.element_ty),
        mask=offs_d < D,
    )


_SM_COUNT: dict[int, int] = {}


def _sm_count(device: torch.device) -> int:
    index = device.index if device.index is not None else 0
    count = _SM_COUNT.get(index)
    if count is None:
        count = torch.cuda.get_device_properties(index).multi_processor_count
        _SM_COUNT[index] = count
    return count


# Per padded head_dim: (tokens per tile, num_warps, resident programs per SM
# the split count aims for). Tuned on sm_120 (RTX PRO 6000): one 16-token page
# per tile with 2 warps keeps registers low enough for 3 (D <= 128) or 2
# (D = 256) programs per SM, and a grid of one full wave avoids a tail.
_TILE_CONFIG = {64: (16, 2, 3), 128: (16, 2, 3), 256: (16, 2, 2)}
_MERGE_WARPS = 2


def _num_splits(
    batch: int,
    num_kv_heads: int,
    max_context: int,
    block_n: int,
    ctas_per_sm: int,
    device: torch.device,
) -> int:
    programs = batch * num_kv_heads
    target = ctas_per_sm * _sm_count(device)
    max_tiles = max(1, triton.cdiv(max_context, block_n))
    return max(1, min(_MAX_SPLITS, max_tiles, triton.cdiv(target, programs)))


def _key_layout(cache: torch.Tensor) -> tuple[int, int, int, int, int, int]:
    """(S_H, S_T, S_DO, S_DI, X, block) of a K cache in vLLM or NHD layout."""
    s = cache.stride()
    if cache.ndim == 5:  # [blocks, kv_heads, D / x, block, x]
        return s[1], s[3], s[2], s[4], cache.shape[4], cache.shape[3]
    raise ValueError("vLLM key cache must be 5-D")


def _value_layout(cache: torch.Tensor) -> tuple[int, int, int, int, int, int]:
    s = cache.stride()
    if cache.ndim == 4:  # [blocks, kv_heads, D, block]
        return s[1], s[3], s[2], 0, 1, cache.shape[3]
    raise ValueError("vLLM value cache must be 4-D")


def _nhd_layout(cache: torch.Tensor) -> tuple[int, int, int, int, int, int]:
    s = cache.stride()  # [blocks, block, kv_heads, D]
    return s[2], s[1], s[3], 0, 1, cache.shape[1]


def split_kv_decode_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    *,
    num_kv_heads: int,
    key_layout: tuple[int, int, int, int, int, int],
    value_layout: tuple[int, int, int, int, int, int],
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    max_seq_len: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Launch split-KV paged decode attention. No device -> host transfers.

    Args:
        query: ``[batch, num_heads, head_dim]`` fp16 / bf16, last dim
            contiguous.
        key_cache / value_cache: paged caches; ``key_layout`` /
            ``value_layout`` are ``(S_H, S_T, S_DO, S_DI, X, block_size)``
            element strides inside one page (``_key_layout`` etc.); the page
            index is always dimension 0.
        block_tables: ``[batch, max_blocks]`` int32.
        seq_lens: ``[batch]`` int32 context lengths (0 yields zeros).
        key_scale / value_scale: per-token fp16 scales ``[blocks, kv_heads,
            block]`` (any strides) when the caches are int8.
        max_seq_len: optional host-side upper bound of ``seq_lens``; only
            used to size the split count, defaults to the block-table
            capacity.

    Returns:
        ``[batch, num_heads, head_dim]`` in ``query.dtype``.
    """
    batch, num_heads, head_dim = query.shape
    k_sh, k_st, k_sdo, k_sdi, k_x, page = key_layout
    v_sh, v_st, v_sdo, v_sdi, v_x, v_page = value_layout
    if page != v_page:
        raise ValueError("key and value caches must use the same block size")
    if page <= 0 or page & (page - 1):
        raise ValueError(f"block_size must be a power of two, got {page}")
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(
            f"head_dim must be one of {_SUPPORTED_HEAD_DIMS}, got {head_dim}"
        )
    if num_heads % num_kv_heads:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    group = num_heads // num_kv_heads

    if query.stride(-1) != 1:
        query = query.contiguous()
    if block_tables.dtype != torch.int32:
        block_tables = block_tables.to(torch.int32)
    if block_tables.stride(-1) != 1:
        block_tables = block_tables.contiguous()
    if seq_lens.dtype != torch.int32:
        seq_lens = seq_lens.to(torch.int32)
    seq_lens = seq_lens.contiguous()

    is_int8 = key_scale is not None
    if out is None:
        out = torch.empty(
            (batch, num_heads, head_dim), dtype=query.dtype, device=query.device
        )
    if out.numel() == 0:
        return out
    if not is_int8 and query.dtype != key_cache.dtype:
        query = query.to(key_cache.dtype)
    elif is_int8 and query.dtype not in (torch.float16, torch.bfloat16):
        query = query.to(torch.bfloat16)

    d_pad = triton.next_power_of_2(head_dim)
    g_pad = max(16, triton.next_power_of_2(group))
    tile_tokens, num_warps, ctas_per_sm = _TILE_CONFIG[d_pad]
    pages = max(1, tile_tokens // page)
    block_n = pages * page
    max_context = (
        int(max_seq_len)
        if max_seq_len is not None
        else block_tables.shape[1] * page
    )
    max_context = min(max_context, block_tables.shape[1] * page)
    splits = _num_splits(
        batch, num_kv_heads, max_context, block_n, ctas_per_sm, query.device
    )
    direct = splits == 1

    if direct:
        part_o = out
        part_lse = out
    else:
        part_o = torch.empty(
            (batch, num_heads, splits, d_pad),
            dtype=torch.float32,
            device=query.device,
        )
        part_lse = torch.empty(
            (batch, num_heads, splits),
            dtype=torch.float32,
            device=query.device,
        )

    if is_int8:
        ks, vs = key_scale, value_scale
        ks_page, vs_page = ks.stride(0), vs.stride(0)
        ks_sh, ks_st = ks.stride(1), ks.stride(2)
        vs_sh, vs_st = vs.stride(1), vs.stride(2)
    else:
        ks = vs = key_cache
        ks_page = vs_page = ks_sh = ks_st = vs_sh = vs_st = 0

    _split_kv_decode_kernel[(splits, num_kv_heads, batch)](
        query,
        key_cache,
        value_cache,
        ks,
        vs,
        block_tables,
        seq_lens,
        out,
        part_o,
        part_lse,
        float(scale) * _LOG2E,
        splits,
        num_heads,
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        value_cache.stride(0),
        ks_page,
        vs_page,
        block_tables.stride(0),
        out.stride(0),
        out.stride(1),
        GROUP=group,
        G_PAD=g_pad,
        D=head_dim,
        D_PAD=d_pad,
        PAGE=page,
        PAGES=pages,
        K_SH=k_sh,
        K_ST=k_st,
        K_SDO=k_sdo,
        K_SDI=k_sdi,
        K_X=k_x,
        V_SH=v_sh,
        V_ST=v_st,
        V_SDO=v_sdo,
        V_SDI=v_sdi,
        V_X=v_x,
        KS_SH=ks_sh,
        KS_ST=ks_st,
        VS_SH=vs_sh,
        VS_ST=vs_st,
        IS_INT8=is_int8,
        DIRECT=direct,
        num_warps=num_warps,
        num_stages=1,
    )
    if not direct:
        d_block = min(d_pad, 64)
        s_chunk = min(triton.next_power_of_2(splits), 4096 // d_block)
        _merge_splits_kernel[(num_heads, batch, d_pad // d_block)](
            part_o,
            part_lse,
            out,
            splits,
            num_heads,
            out.stride(0),
            out.stride(1),
            D=head_dim,
            D_PAD=d_pad,
            D_BLOCK=d_block,
            S_CHUNK=s_chunk,
            num_warps=_MERGE_WARPS,
        )
    return out


def fused_decode_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Launch decode-only single-query paged attention (NHD cache layout).

    Args:
        query: ``[batch, 1, num_heads, head_dim]`` bf16
        key_cache: ``[num_blocks, block_size, num_kv_heads, head_dim]`` bf16
        value_cache: ``[num_blocks, block_size, num_kv_heads, head_dim]`` bf16
        block_tables: ``[batch, max_blocks_per_seq]`` int32
        seq_lens: ``[batch]`` int32, each at most
            ``max_blocks_per_seq * block_size`` (not checked: that would need
            a device -> host sync)
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
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(
            f"head_dim must be one of {_SUPPORTED_HEAD_DIMS}, got {head_dim}"
        )
    if num_blocks <= 0:
        raise ValueError("key_cache must contain at least one block")
    if block_tables.shape[1] <= 0:
        raise ValueError("block_tables must contain at least one block slot")
    if key_cache.stride(-1) != 1 or value_cache.stride(-1) != 1:
        raise ValueError("KV cache head_dim must be the contiguous dimension")

    return split_kv_decode_attention(
        query[:, 0],
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
        num_kv_heads=num_kv_heads,
        key_layout=_nhd_layout(key_cache),
        value_layout=_nhd_layout(value_cache),
    )


def supports_head_dim(head_dim: int) -> bool:
    return head_dim in _SUPPORTED_HEAD_DIMS


__all__ = [
    "fused_decode_attention",
    "split_kv_decode_attention",
    "supports_head_dim",
]
