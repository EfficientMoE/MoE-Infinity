# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# EXPERIMENT (#304 round 3, not wired into MoE-Infinity): sequence-split
# ("intra-card context parallel") version of moe_infinity/kernel/gdn_fused.py.
# Measured slower than the sequential fused kernel on sm_120; kept for the
# record. See PR #336 Status.
"""Segmented delta-rule recurrence + output (forward, inference).

The chunk update is affine in the state: h' = M_c h + k^T u with
M_c = exp2(g_last) I - (k * exp2(g_last - g))^T w (GDN; KDA: diag decay).
Pass 1 runs every segment (L chunks) from zero over V + K columns:
columns < V give Q_s (state reached from 0), columns >= V start from I with
u = 0 and give P_s = prod M_c. A fp32 scan gives each segment's initial
state hs_s = P_{s-1} hs_{s-1} + Q_{s-1}; pass 2 recomputes every segment's
outputs from its hs in parallel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T", "L"])
def seg_h_o_kernel(
    q,
    k,
    w,
    u,
    g,
    Aqk,
    o,
    h0,
    ht,
    scale,
    T,
    L,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    GATE_VEC: tl.constexpr,
    STATE_ONLY: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    VS: tl.constexpr = V + K if STATE_ONLY else V
    NV: tl.constexpr = VS // BV
    i_v, i_nh = pid % NV, pid // NV
    i_sn, i_h = i_nh // HV, i_nh % HV
    B = tl.num_programs(0) // (NV * HV)
    NT = tl.cdiv(T, BT)
    NS = tl.cdiv(NT, L)
    if STATE_ONLY:
        NS = NS - 1
    B = B // NS
    i_s, i_n = i_sn // B, i_sn % B
    bos = i_n * T
    c0 = i_s * L
    c1 = tl.minimum(c0 + L, NT)
    i_nh_seq = i_n * HV + i_h

    q += (bos * HV + i_h) * K
    k += (bos * HV + i_h) * K
    w += (bos * HV + i_h) * K
    u += (bos * HV + i_h) * V
    o += (bos * HV + i_h) * V
    if GATE_VEC:
        g += (bos * HV + i_h) * K
        Aqk += (bos * HV + i_h) * BT
    else:
        g += bos * HV + i_h

    o_v = i_v * BV + tl.arange(0, BV)
    o_k = tl.arange(0, K)
    o_c = tl.arange(0, BT)
    m_c = o_c[:, None] >= o_c[None, :]
    m_u = o_v < V
    o_u = tl.where(m_u, o_v, 0)

    b_h = tl.zeros([K, BV], dtype=tl.float32)
    if STATE_ONLY:
        b_h = tl.where((o_k[:, None] + V) == o_v[None, :], 1.0, b_h)
    elif USE_INITIAL_STATE:
        b_h += tl.load(
            h0
            + (i_s * B * HV + i_nh_seq) * K * V
            + o_k[:, None] * V
            + o_v[None, :]
        )

    for i_t in range(c0, c1):
        o_t = i_t * BT + o_c
        m_t = o_t < T
        last = tl.minimum((i_t + 1) * BT, T) - 1
        p_row = o_t[:, None] * (HV * K) + o_k[None, :]
        b_w = tl.load(w + p_row, mask=m_t[:, None], other=0.0)
        b_k = tl.load(
            k + o_k[:, None] + o_t[None, :] * (HV * K),
            mask=m_t[None, :],
            other=0.0,
        )
        b_u = tl.load(
            u + o_t[:, None] * (HV * V) + o_u[None, :],
            mask=m_t[:, None] & m_u[None, :],
            other=0.0,
        )
        if GATE_VEC:
            b_gl = tl.load(g + last * (HV * K) + o_k)
        else:
            b_g = tl.load(g + o_t * HV, mask=m_t, other=0.0)
            b_gl = tl.load(g + last * HV)
        b_hb = b_h.to(b_w.dtype)
        b_v = (b_u.to(tl.float32) - tl.dot(b_w, b_hb)).to(b_w.dtype)
        if not STATE_ONLY:
            b_q = tl.load(q + p_row, mask=m_t[:, None], other=0.0)
            m_S = m_c & m_t[:, None] & m_t[None, :]
            if GATE_VEC:
                b_g = tl.load(g + p_row, mask=m_t[:, None], other=0.0)
                b_q = (b_q * tl.exp2(b_g) * scale).to(b_w.dtype)
                b_S = tl.load(
                    Aqk + o_t[:, None] * (HV * BT) + o_c[None, :],
                    mask=m_t[:, None],
                    other=0.0,
                )
                b_S = tl.where(m_S, b_S, 0.0)
                b_o = tl.dot(b_S.to(b_v.dtype), b_v, tl.dot(b_q, b_hb))
            else:
                b_S = tl.dot(b_q, b_k)
                b_S = tl.where(
                    m_S, b_S * tl.exp2(b_g[:, None] - b_g[None, :]), 0.0
                )
                b_o = tl.dot(b_q, b_hb) * tl.exp2(b_g)[:, None]
                b_o = tl.dot(b_S.to(b_v.dtype), b_v, b_o) * scale
            tl.store(
                o + o_t[:, None] * (HV * V) + o_v[None, :],
                b_o.to(o.dtype.element_ty),
                mask=m_t[:, None],
            )
        if GATE_VEC:
            b_h = tl.dot(b_k, b_v, b_h * tl.exp2(b_gl)[:, None])
        else:
            b_vg = (
                b_v.to(tl.float32)
                * tl.where(m_t, tl.exp2(b_gl - b_g), 0.0)[:, None]
            )
            b_h = tl.dot(b_k, b_vg.to(b_w.dtype), b_h * tl.exp2(b_gl))

    if STATE_ONLY:
        p = ht + (i_s * B * HV + i_nh_seq) * K * VS + o_k[:, None] * VS
        tl.store(p + o_v[None, :], b_h)
    elif STORE_FINAL_STATE:
        if c1 == NT:
            p = ht + i_nh_seq * K * V + o_k[:, None] * V + o_v[None, :]
            tl.store(p, b_h.to(ht.dtype.element_ty))


@triton.jit
def seg_combine_kernel(
    pq,
    h0,
    hs,
    NS,
    BH,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
):
    """hs_0 = h0, hs_s = P_{s-1} hs_{s-1} + Q_{s-1} (fp32, ieee dot)."""
    pid = tl.program_id(0).to(tl.int64)
    NV: tl.constexpr = V // BV
    i_v, i_bh = pid % NV, pid // NV
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    VS: tl.constexpr = V + K
    KB: tl.constexpr = 32
    o_kb = tl.arange(0, KB)
    b_h = tl.zeros([K, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        b_h += tl.load(h0 + i_bh * K * V + o_k[:, None] * V + o_v[None, :])
    tl.store(hs + i_bh * K * V + o_k[:, None] * V + o_v[None, :], b_h)
    for i_s in range(1, NS):
        tl.debug_barrier()
        base = pq + ((i_s - 1) * BH + i_bh) * K * VS + o_k[:, None] * VS
        prev = hs + ((i_s - 1) * BH + i_bh) * K * V + o_v[None, :]
        b_h = tl.load(base + o_v[None, :])
        for i_k in range(K // KB):
            b_p = tl.load(base + V + i_k * KB + o_kb[None, :])
            b_x = tl.load(prev + (i_k * KB + o_kb)[:, None] * V)
            b_h = tl.dot(b_p, b_x, b_h, input_precision="ieee")
        tl.store(
            hs + (i_s * BH + i_bh) * K * V + o_k[:, None] * V + o_v[None, :],
            b_h,
        )


def seg_fwd_h_o(
    q,
    k,
    w,
    u,
    g,
    scale,
    Aqk=None,
    initial_state=None,
    output_final_state=False,
    seg_chunks=12,
    BV=None,
    num_warps=4,
    num_stages=None,
):
    B, T, HV, K = k.shape
    V = u.shape[-1]
    gate_vec = g.dim() == 4
    NT = triton.cdiv(T, 64)
    L = min(seg_chunks, NT)
    NS = triton.cdiv(NT, L)
    if BV is None:
        BV = 32 if NS * B * HV * (V // 32) <= 192 else 64
    if num_stages is None:
        num_stages = 1 if gate_vec else 2
    o = torch.empty_like(u)
    ht = (
        k.new_empty(B, HV, K, V, dtype=torch.float32)
        if output_final_state
        else None
    )
    common = dict(
        HV=HV,
        K=K,
        V=V,
        BT=64,
        GATE_VEC=gate_vec,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    hs = initial_state
    A = Aqk if gate_vec else u
    if NS > 1:
        pq = k.new_empty(NS - 1, B, HV, K, V + K, dtype=torch.float32)
        seg_h_o_kernel[((NS - 1) * B * HV * (V + K) // 32,)](
            q,
            k,
            w,
            u,
            g,
            A,
            o,
            u,
            pq,
            scale,
            T,
            L,
            BV=32,
            STATE_ONLY=True,
            USE_INITIAL_STATE=False,
            STORE_FINAL_STATE=False,
            **common,
        )
        hs = k.new_empty(NS, B, HV, K, V, dtype=torch.float32)
        seg_combine_kernel[(B * HV * (V // 32),)](
            pq,
            initial_state,
            hs,
            NS,
            B * HV,
            K=K,
            V=V,
            BV=32,
            USE_INITIAL_STATE=initial_state is not None,
            num_warps=4,
        )
    seg_h_o_kernel[(NS * B * HV * (V // BV),)](
        q,
        k,
        w,
        u,
        g,
        A,
        o,
        hs,
        ht,
        scale,
        T,
        L,
        BV=BV,
        STATE_ONLY=False,
        USE_INITIAL_STATE=hs is not None,
        STORE_FINAL_STATE=output_final_state,
        **common,
    )
    return o, ht
