"""Triton kernels for HF-layout GQA attention (runtime/hf_attention.py).

``gqa_decode_attention``: split-KV GQA decode attention over a contiguous
(HF ``DynamicCache``) KV.  ``apply_attention_sinks``: exact sinks from the
log-sum-exp of a prefill kernel, one in-place pass.

Decode details:

One query token per sequence attends over ``K/V [B, KV_H, L, D]`` (any
strides with a contiguous last dim, so cache views need no copy).  Each
program owns one KV head, all of its query heads (the GQA group is the M
dimension of ``tl.dot``, so every KV byte is read exactly once) and one
chunk of the context; a second launch merges the chunks by log-sum-exp.

Fused in the kernel, so they cost no extra pass over the KV:

* sliding window (keys ``> L - 1 - window``),
* an optional per-sequence key mask ``[B, 1, 1, L]`` (bool keep-mask or
  additive float), as HF passes for padding / MiniMax-M3 block-sparse /
  GLM DSA index layers,
* attention sinks (GPT-OSS): ``out *= sigmoid(lse - sink)`` in the merge.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

_LOG2E = 1.4426950408889634
_LN2 = 0.6931471805599453
_MAX_SPLITS = 256
_TARGET_WAVES = 1  # split programs per SM to aim for
_PARTIAL_FRAC = 0.04  # max partial-result bytes / KV bytes


@triton.jit
def _gqa_decode_split_kernel(
    Q,
    K,
    V,
    MASK,
    OUT_P,
    LSE_P,
    stride_qb,
    stride_qh,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_mb,
    stride_mn,
    kv_heads,
    lo,
    hi,
    chunk,
    num_splits,
    scale_log2,
    G: tl.constexpr,
    G_PAD: tl.constexpr,
    D: tl.constexpr,
    D_PAD: tl.constexpr,
    DV: tl.constexpr,
    DV_PAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MASK_KIND: tl.constexpr,  # 0 none, 1 bool keep, 2 additive
):
    split = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // kv_heads
    h = bh % kv_heads
    start = lo + split * chunk
    end = tl.minimum(start + chunk, hi)

    offs_g = tl.arange(0, G_PAD)
    offs_d = tl.arange(0, D_PAD)
    offs_dv = tl.arange(0, DV_PAD)
    offs_n = tl.arange(0, BLOCK_N)
    g_ok = offs_g < G
    d_ok = offs_d < D
    dv_ok = offs_dv < DV

    q_ptrs = (
        Q
        + b.to(tl.int64) * stride_qb
        + (h * G + offs_g)[:, None] * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(q_ptrs, mask=g_ok[:, None] & d_ok[None, :], other=0.0)

    k_base = K + b.to(tl.int64) * stride_kb + h.to(tl.int64) * stride_kh
    v_base = V + b.to(tl.int64) * stride_vb + h.to(tl.int64) * stride_vh
    m_base = MASK + b.to(tl.int64) * stride_mb

    m_i = tl.full([G_PAD], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([G_PAD], dtype=tl.float32)
    acc = tl.zeros([G_PAD, DV_PAD], dtype=tl.float32)

    for n0 in range(start, end, BLOCK_N):
        n = n0 + offs_n
        valid = n < end
        n64 = n.to(tl.int64)
        k = tl.load(
            k_base + n64[:, None] * stride_kn + offs_d[None, :],
            mask=valid[:, None] & d_ok[None, :],
            other=0.0,
        )
        s = tl.dot(q, tl.trans(k)) * scale_log2
        if MASK_KIND == 1:
            keep = tl.load(m_base + n64 * stride_mn, mask=valid, other=0)
            valid = valid & (keep != 0)
        if MASK_KIND == 2:
            bias = tl.load(m_base + n64 * stride_mn, mask=valid, other=0.0)
            s += bias.to(tl.float32)[None, :] * 1.4426950408889634
        s = tl.where(valid[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(
            v_base + n64[:, None] * stride_vn + offs_dv[None, :],
            mask=valid[:, None] & dv_ok[None, :],
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    has = l_i > 0.0
    out = acc / tl.where(has, l_i, 1.0)[:, None]
    lse = tl.where(has, m_i + tl.log2(tl.where(has, l_i, 1.0)), float("-inf"))

    row = bh.to(tl.int64) * num_splits + split
    op = OUT_P + (row * G + offs_g)[:, None] * DV + offs_dv[None, :]
    tl.store(op, out, mask=g_ok[:, None] & dv_ok[None, :])
    tl.store(LSE_P + row * G + offs_g, lse, mask=g_ok)


@triton.jit
def _gqa_decode_merge_kernel(
    OUT_P,
    LSE_P,
    SINKS,
    OUT,
    LSE_OUT,
    stride_ob,
    stride_oh,
    kv_heads,
    num_splits,
    G: tl.constexpr,
    DV: tl.constexpr,
    DV_PAD: tl.constexpr,
    SPLITS_PAD: tl.constexpr,
    SPLIT_BLOCK: tl.constexpr,
    HAS_SINKS: tl.constexpr,
):
    bh = tl.program_id(0)  # over B * KV_H
    g = tl.program_id(1)
    b = bh // kv_heads
    head = (bh % kv_heads) * G + g

    offs_s = tl.arange(0, SPLITS_PAD)
    offs_dv = tl.arange(0, DV_PAD)
    base = bh.to(tl.int64) * num_splits
    lse = tl.load(
        LSE_P + (base + offs_s) * G + g,
        mask=offs_s < num_splits,
        other=float("-inf"),
    )
    m = tl.max(lse, 0)
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    total = tl.sum(tl.exp2(lse - m_safe), 0)

    offs_b = tl.arange(0, SPLIT_BLOCK)
    acc = tl.zeros([DV_PAD], dtype=tl.float32)
    for s0 in range(0, num_splits, SPLIT_BLOCK):
        s_ok = (s0 + offs_b) < num_splits
        rows = base + s0 + offs_b
        w = tl.exp2(
            tl.load(LSE_P + rows * G + g, mask=s_ok, other=float("-inf"))
            - m_safe
        )
        o = tl.load(
            OUT_P + (rows * G + g)[:, None] * DV + offs_dv[None, :],
            mask=s_ok[:, None] & (offs_dv < DV)[None, :],
            other=0.0,
        )
        acc += tl.sum(o * w[:, None], 0)
    out = acc / tl.where(total > 0.0, total, 1.0)
    lse_nat = tl.where(
        total > 0.0,
        (m_safe + tl.log2(tl.where(total > 0.0, total, 1.0)))
        * 0.6931471805599453,
        float("-inf"),
    )
    if HAS_SINKS:
        sink = tl.load(SINKS + head).to(tl.float32)
        out = out * tl.sigmoid(lse_nat - sink)
    dst = OUT + b.to(tl.int64) * stride_ob + head * stride_oh + offs_dv
    tl.store(dst, out.to(OUT.dtype.element_ty), mask=offs_dv < DV)
    tl.store(LSE_OUT + b.to(tl.int64) * kv_heads * G + head, lse_nat)


@triton.jit
def _sink_scale_kernel(
    OUT,
    LSE,
    SINKS,
    n_rows,
    heads,
    qlen,
    stride_ob,
    stride_oq,
    stride_oh,
    stride_lb,
    stride_lh,
    stride_lq,
    D: tl.constexpr,
    D_PAD: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    r_ok = rows < n_rows
    h = rows % heads
    t = rows // heads
    q = t % qlen
    b = (t // qlen).to(tl.int64)
    lse = tl.load(
        LSE + b * stride_lb + h * stride_lh + q * stride_lq,
        mask=r_ok,
        other=0.0,
    )
    sink = tl.load(SINKS + h, mask=r_ok, other=0.0).to(tl.float32)
    f = tl.sigmoid(lse - sink)
    offs_d = tl.arange(0, D_PAD)
    ptrs = (
        OUT
        + (b * stride_ob + q * stride_oq + h * stride_oh)[:, None]
        + offs_d[None, :]
    )
    m = r_ok[:, None] & (offs_d < D)[None, :]
    x = tl.load(ptrs, mask=m, other=0.0)
    tl.store(ptrs, (x.to(tl.float32) * f[:, None]).to(x.dtype), mask=m)


def apply_attention_sinks(
    out: torch.Tensor, lse: torch.Tensor, sinks: torch.Tensor
) -> torch.Tensor:
    """In place: ``out [B, Q, H, D] *= sigmoid(lse [B, H, Q] - sinks [H])``.

    ``softmax([s, sink])[:-1] @ V == (softmax(s) @ V) * Z / (Z + e^sink)``
    with ``lse = log Z``: exact attention sinks on top of any kernel that
    returns the log-sum-exp.  One launch; ``out`` needs a contiguous last dim.
    """
    bsz, qlen, heads, d = out.shape
    n_rows = bsz * qlen * heads
    rows = max(1, min(64, 4096 // _pow2(d)))
    _sink_scale_kernel[(triton.cdiv(n_rows, rows),)](
        out,
        lse,
        sinks,
        n_rows,
        heads,
        qlen,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        D=d,
        D_PAD=max(16, _pow2(d)),
        ROWS=rows,
        num_warps=4,
    )
    return out


def _pow2(x: int) -> int:
    return 1 << max(0, (x - 1).bit_length())


@functools.lru_cache(maxsize=8)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _config(d: int, g: int) -> tuple[int, int, int]:
    """(BLOCK_N, num_warps, num_stages); swept on sm_120 (RTX PRO 6000)."""
    if d <= 64:
        return 64, 4, 3
    if d <= 128:
        return 32, 4, 3
    return 32, 4, 2


def gqa_decode_supported(q, k, v, mask=None) -> bool:
    """Whether :func:`gqa_decode_attention` handles these HF-layout inputs."""
    if not (q.is_cuda and q.dim() == 4 and q.shape[2] == 1):
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    if k.shape[0] != q.shape[0] or v.shape[:3] != k.shape[:3]:
        return False
    if q.shape[1] % k.shape[1] != 0 or q.shape[-1] != k.shape[-1]:
        return False
    if q.shape[-1] > 256 or v.shape[-1] > 256 or k.shape[2] == 0:
        return False
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        return False
    if mask is not None:
        if mask.dim() != 4 or mask.shape[1] != 1 or mask.shape[2] != 1:
            return False
        if mask.shape[0] not in (1, q.shape[0]) or mask.shape[-1] != k.shape[2]:
            return False
    return True


def gqa_decode_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    *,
    window: int | None = None,
    sinks: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    return_lse: bool = False,
):
    """Decode attention in HF layout.

    Args:
        query: ``[B, H, 1, D]``.
        key / value: ``[B, KV_H, L, D]`` / ``[B, KV_H, L, Dv]``, ``H % KV_H == 0``.
        scale: softmax scale.
        window: keep only the last ``window`` keys (sliding-window layers).
        sinks: ``[H]`` per-head sink logits (GPT-OSS).
        mask: ``[B or 1, 1, 1, L]`` bool keep-mask or additive float mask.

    Returns:
        ``[B, 1, H, Dv]`` in ``query.dtype`` (and ``lse [B, H]`` fp32 if
        ``return_lse``).
    """
    bsz, heads, _, d = query.shape
    kv_heads, klen, dv = key.shape[1], key.shape[2], value.shape[-1]
    g = heads // kv_heads
    lo = 0 if window is None else max(0, klen - window)
    span = klen - lo

    block_n, num_warps, num_stages = _config(max(d, dv), g)
    n_bh = bsz * kv_heads
    target = _TARGET_WAVES * _num_sms(query.device.index or 0)
    # Keep the fp32 partial results (one [G, Dv] row per split) a few percent
    # of the KV bytes this (batch, kv-head) streams.
    kv_bytes = span * (d + dv) * query.element_size()
    max_splits = min(
        _MAX_SPLITS,
        triton.cdiv(span, block_n),
        max(1, int(_PARTIAL_FRAC * kv_bytes / (4 * g * dv))),
    )
    splits = max(1, min(triton.cdiv(target, n_bh), max_splits))
    chunk = triton.cdiv(triton.cdiv(span, splits), block_n) * block_n
    splits = triton.cdiv(span, chunk)

    if mask is None:
        mask_kind, m_arg, smb, smn = 0, query, 0, 0
    else:
        if mask.dtype == torch.bool:
            mask_kind, m_arg = 1, mask.view(torch.uint8)
        else:
            mask_kind, m_arg = 2, mask
        smb = m_arg.stride(0) if m_arg.shape[0] > 1 else 0
        smn = m_arg.stride(-1)

    out_p = torch.empty(
        (n_bh * splits * g, dv), device=query.device, dtype=torch.float32
    )
    lse_p = torch.empty(
        (n_bh * splits * g,), device=query.device, dtype=torch.float32
    )
    _gqa_decode_split_kernel[(splits, n_bh)](
        query,
        key,
        value,
        m_arg,
        out_p,
        lse_p,
        query.stride(0),
        query.stride(1),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        smb,
        smn,
        kv_heads,
        lo,
        klen,
        chunk,
        splits,
        float(scale) * _LOG2E,
        G=g,
        G_PAD=max(16, _pow2(g)),
        D=d,
        D_PAD=max(16, _pow2(d)),
        DV=dv,
        DV_PAD=max(16, _pow2(dv)),
        BLOCK_N=block_n,
        MASK_KIND=mask_kind,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    out = torch.empty(
        (bsz, 1, heads, dv), device=query.device, dtype=query.dtype
    )
    lse = torch.empty((bsz, heads), device=query.device, dtype=torch.float32)
    _gqa_decode_merge_kernel[(n_bh, g)](
        out_p,
        lse_p,
        sinks if sinks is not None else lse_p,
        out,
        lse,
        out.stride(0),
        out.stride(2),
        kv_heads,
        splits,
        G=g,
        DV=dv,
        DV_PAD=max(16, _pow2(dv)),
        SPLITS_PAD=max(2, _pow2(splits)),
        SPLIT_BLOCK=min(16, max(2, _pow2(splits))),
        HAS_SINKS=sinks is not None,
        num_warps=1 if dv <= 64 else 2,
    )
    return (out, lse) if return_lse else out


__all__ = [
    "apply_attention_sinks",
    "gqa_decode_attention",
    "gqa_decode_supported",
]
