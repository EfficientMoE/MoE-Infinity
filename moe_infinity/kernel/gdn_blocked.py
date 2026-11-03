"""L2-blocked GDN / KDA prefill (#304, round 4).

The sequence is processed in token blocks small enough that everything one
block's prep writes stays in the 128 MB L2 until the recurrence reads it:

    for each block of ``block_tokens`` tokens:
        prep  (gdn_prep, parallel over chunks x heads): k L2 norm folded into
              kkt + solve + w / u (A stays in registers)
        aux   (gdn_prep, parallel): the h-independent recurrence operands
              qg = scale * exp2(g) * q_n, kd = exp2(g_last - g) * k_n,
              S  = scale * tril(q_n k_n^T * exp2(g_i - g_j))
        recur (one program per head x 32 value columns): per chunk
              v = u - w @ h;  o = qg @ h + S @ v;  h = exp2(g_last) h + kd^T @ v

The recurrence carries no h-independent work. It runs as a CUDA mma.sync
kernel (csrc/gdn_recur.cu, JIT-built) or, without nvcc, as Triton. The block
buffers are reused, so the intermediates stay in L2.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# keep the block buffers resident (Rule F); toggled for measurement
L2_PERSIST = True


@triton.jit(do_not_specialize=["T", "t_base", "TB", "NC"])
def blocked_recur_kernel(
    qg,
    kd,
    w,
    u,
    S,
    g,
    o,
    h,
    T,
    t_base,
    TB,
    NC,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    GATE_VEC: tl.constexpr,
):
    """One program per (sequence, head, BV value slice). ``h`` [B, HV, K, V]
    fp32 is read at the start and written back at the end of the block.
    Block buffers are [B, TB, HV, *]; g / o are the full [B, T, HV, *]."""
    pid = tl.program_id(0).to(tl.int64)
    NV: tl.constexpr = V // BV
    i_v, i_nh = pid % NV, pid // NV
    i_n, i_h = i_nh // HV, i_nh % HV
    lb = i_n * TB
    o_v = i_v * BV + tl.arange(0, BV)
    o_k = tl.arange(0, K)
    o_c = tl.arange(0, BT)

    qg += (lb * HV + i_h) * K
    kd += (lb * HV + i_h) * K
    w += (lb * HV + i_h) * K
    u += (lb * HV + i_h) * V
    S += (lb * HV + i_h) * BT
    o += (i_n * T * HV + i_h) * V
    if GATE_VEC:
        g += (i_n * T * HV + i_h) * K
    else:
        g += i_n * T * HV + i_h
    p_h = h + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
    b_h = tl.load(p_h)

    for i_c in range(NC):
        o_l = i_c * BT + o_c
        o_t = t_base + o_l
        m_t = o_t < T
        last = tl.minimum(t_base + i_c * BT + BT, T) - 1
        p_row = o_l[:, None] * (HV * K) + o_k[None, :]
        b_w = tl.load(w + p_row, mask=m_t[:, None], other=0.0)
        b_q = tl.load(qg + p_row, mask=m_t[:, None], other=0.0)
        b_k = tl.load(
            kd + o_k[:, None] + o_l[None, :] * (HV * K),
            mask=m_t[None, :],
            other=0.0,
        )
        b_u = tl.load(
            u + o_l[:, None] * (HV * V) + o_v[None, :],
            mask=m_t[:, None],
            other=0.0,
        )
        b_S = tl.load(
            S + o_l[:, None] * (HV * BT) + o_c[None, :],
            mask=m_t[:, None],
            other=0.0,
        )
        if GATE_VEC:
            b_dl = tl.exp2(tl.load(g + last * (HV * K) + o_k))[:, None]
        else:
            b_dl = tl.exp2(tl.load(g + last * HV))
        b_hb = b_h.to(b_w.dtype)
        b_v = (b_u.to(tl.float32) - tl.dot(b_w, b_hb)).to(b_w.dtype)
        b_o = tl.dot(b_S, b_v, tl.dot(b_q, b_hb))
        tl.store(
            o + o_t[:, None] * (HV * V) + o_v[None, :],
            b_o.to(o.dtype.element_ty),
            mask=m_t[:, None],
        )
        b_h = tl.dot(b_k, b_v, b_h * b_dl)

    tl.store(p_h, b_h)


def _recur(
    qg, kd, w, u, S, g, o, h, t_base, nb, BV, num_warps, num_stages, use_cuda
):
    B, TB, HV, K = qg.shape
    T = o.shape[1]
    V = u.shape[-1]
    if use_cuda:
        from moe_infinity.kernel import gdn_recur_cuda

        gdn_recur_cuda.recur(qg, kd, w, u, S, g, o, h, t_base, nb)
        return
    blocked_recur_kernel[(B * HV * (V // BV),)](
        qg,
        kd,
        w,
        u,
        S,
        g,
        o,
        h,
        T,
        t_base,
        TB,
        triton.cdiv(nb, 64),
        HV=HV,
        K=K,
        V=V,
        BT=64,
        BV=BV,
        GATE_VEC=g.dim() == 4,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def chunk_gated_delta_rule_blocked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gc: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    norm_qk: bool = True,
    block_tokens: int = 2048,
    BV: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
    use_cuda: bool | None = None,
):
    """GDN forward. ``gc`` is the chunk-local cumsum of the log2 decay
    ([B, T, H] fp32); q / k raw (``norm_qk``) or unit-norm; bf16 [B, T, H, 128].
    """
    from moe_infinity.kernel.gdn_prep import (
        gdn_fwd_aux_kernel,
        gdn_fwd_prep_kernel,
    )

    if use_cuda is None:
        from moe_infinity.kernel import gdn_recur_cuda

        use_cuda = gdn_recur_cuda.available()

    B, T, H, K = k.shape
    V = v.shape[-1]
    TB = min(block_tokens, triton.cdiv(T, 64) * 64)
    o = torch.empty_like(v)
    h = (
        initial_state.clone()
        if initial_state is not None
        else k.new_zeros(B, H, K, V, dtype=torch.float32)
    )
    # one allocation so a single persisting-L2 window can cover the buffers
    n = B * TB * H
    arena = k.new_empty(n * (3 * K + V + 64))
    w, u, qg, kd, S = (
        t.view(B, TB, H, -1)
        for t in arena.split([n * K, n * V, n * K, n * K, n * 64])
    )
    if use_cuda and L2_PERSIST:
        from moe_infinity.kernel import gdn_recur_cuda

        gdn_recur_cuda.l2_window(arena, arena.numel() * arena.element_size())
    for t_base in range(0, T, TB):
        nb = min(TB, T - t_base)
        gdn_fwd_prep_kernel[(triton.cdiv(nb, 64), B * H)](
            k,
            v,
            gc,
            beta,
            w,
            u,
            w,
            T,
            t_base,
            TB,
            HV=H,
            K=K,
            V=V,
            BT=64,
            BC=16,
            BK=K,
            NORM_K=norm_qk,
            BLOCK_OUT=True,
            num_warps=2,
            num_stages=1,
        )
        gdn_fwd_aux_kernel[(triton.cdiv(nb, 64), B * H)](
            q,
            k,
            gc,
            qg,
            kd,
            S,
            scale,
            T,
            t_base,
            TB,
            HV=H,
            K=K,
            BT=64,
            NORM=norm_qk,
            num_warps=4,
        )
        _recur(
            qg,
            kd,
            w,
            u,
            S,
            gc,
            o,
            h,
            t_base,
            nb,
            BV,
            num_warps,
            num_stages,
            use_cuda,
        )
    if use_cuda and L2_PERSIST:
        gdn_recur_cuda.l2_window(arena, 0)
    return o, (h if output_final_state else None)
