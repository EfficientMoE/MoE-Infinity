# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Modified for MoE-Infinity (#304): flash-linear-attention 0.6.0
# `chunk_gated_delta_rule_fwd_kkt_solve_kernel` extended with the
# `recompute_w_u_fwd_kernel` tail (w = A @ (beta * exp(g) * k),
# u = A @ (beta * v)) so the solved 64x64 A block never leaves registers,
# and with the k L2 norm folded in (kkt scaled by the row norms).
"""Fused kkt + solve_tril + recompute_w_u (+ k L2 norm), forward only.

Hardware: RTX_PRO_6000 (tuned on the RTX PRO 6000 Blackwell Max-Q: 188 SMs,
128 MiB L2, 99 KiB shared memory per block). Reached only through
``install_pro6000_linear_attention``, which checks
``hw_profile.supports(RTX_PRO_6000)``; every other GPU runs fla's kernels.
Tuned here: 2 warps / BK = 128 (register budget), block-buffer outputs
sized for the 128 MiB L2.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _wu_row(A0, A1, A2, A3, x0, x1, x2, x3, R: tl.constexpr):
    """Row block R of A @ X for block-lower-triangular A ([16,16] blocks)."""
    acc = tl.dot(A0.to(x0.dtype), x0)
    if R >= 1:
        acc = tl.dot(A1.to(x1.dtype), x1, acc)
    if R >= 2:
        acc = tl.dot(A2.to(x2.dtype), x2, acc)
    if R >= 3:
        acc = tl.dot(A3.to(x3.dtype), x3, acc)
    return acc


@triton.jit
def _solve_diag(b_A, o_i, n, BC: tl.constexpr):
    """(I + A)^{-1} - I of a strictly lower [BC, BC] block (fla solve_tril)."""
    b_Ai = -b_A
    for i in range(2, n):
        b_a = tl.sum(tl.where((o_i == i)[:, None], -b_A, 0.0), 0)
        b_a = tl.where(o_i < i, b_a, 0.0)
        b_a = b_a + tl.sum(b_a[:, None] * b_Ai, 0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
    return b_Ai


@triton.jit
def _solve_diag_neumann(b_A, o_i, BC: tl.constexpr):
    """(I + A)^{-1} - I of a strictly lower [BC, BC] block by the Neumann
    product: with M = -A (nilpotent, M^BC = 0),
    (I + A)^{-1} = (I + M)(I + M^2)(I + M^4)(I + M^8) for BC = 16. Six small
    fp32 matmuls replace BC - 2 dependent row-reduction steps."""
    m_I = (o_i[:, None] == o_i[None, :]).to(tl.float32)
    b_P = -b_A
    b_X = m_I + b_P
    for _ in tl.static_range(3):
        b_P = tl.dot(b_P, b_P, input_precision="ieee")
        b_X = tl.dot(b_X, m_I + b_P, input_precision="ieee")
    return b_X - m_I


@triton.jit
def _gate(b_A, gi, gj, mi, mj, STRICT: tl.constexpr, o_i):
    m = mi[:, None] & mj[None, :]
    if STRICT:
        m = m & (o_i[:, None] > o_i[None, :])
    return b_A * tl.where(m, tl.exp2(gi[:, None] - gj[None, :]), 0.0)


@triton.jit(do_not_specialize=["T", "t_base", "TB"])
def gdn_fwd_prep_pro6000_kernel(
    k,
    v,
    g,
    beta,
    w,
    u,
    kn,
    T,
    t_base,
    TB,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NORM_K: tl.constexpr,
    BLOCK_OUT: tl.constexpr = False,
    NEUMANN: tl.constexpr = False,
    CUMSUM_G: tl.constexpr = False,
):
    """BLOCK_OUT: chunk i_t of the token block starting at ``t_base``; w / u
    go to block-local buffers ([B, TB, H, *]) and k_n is not written."""
    tl.static_assert(BT == 4 * BC)
    RCP_LN2: tl.constexpr = 1.4426950408889634
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV
    bos = i_b * T
    # output rows are indexed with global token ids; shift the base so that
    # global id t_base lands on local row 0 of the block buffers
    if BLOCK_OUT:
        obos = i_b * TB - t_base
    else:
        obos = bos

    t0 = t_base + i_t * BT
    o_i = tl.arange(0, BC)
    r0 = t0 + o_i
    r1 = r0 + BC
    r2 = r1 + BC
    r3 = r2 + BC
    m0, m1, m2, m3 = r0 < T, r1 < T, r2 < T, r3 < T

    k += (bos * HV + i_h) * K
    v += (bos * HV + i_h) * V
    w += (obos * HV + i_h) * K
    u += (obos * HV + i_h) * V
    kn += (obos * HV + i_h) * K
    pb = beta + bos * HV + i_h
    pg = g + bos * HV + i_h
    b0 = tl.load(pb + r0 * HV, mask=m0, other=0.0).to(tl.float32)
    b1 = tl.load(pb + r1 * HV, mask=m1, other=0.0).to(tl.float32)
    b2 = tl.load(pb + r2 * HV, mask=m2, other=0.0).to(tl.float32)
    b3 = tl.load(pb + r3 * HV, mask=m3, other=0.0).to(tl.float32)
    g0 = tl.load(pg + r0 * HV, mask=m0, other=0.0).to(tl.float32)
    g1 = tl.load(pg + r1 * HV, mask=m1, other=0.0).to(tl.float32)
    g2 = tl.load(pg + r2 * HV, mask=m2, other=0.0).to(tl.float32)
    g3 = tl.load(pg + r3 * HV, mask=m3, other=0.0).to(tl.float32)
    if CUMSUM_G:
        d0 = tl.sum(g0, 0) * RCP_LN2
        s1 = d0 + tl.sum(g1, 0) * RCP_LN2
        s2 = s1 + tl.sum(g2, 0) * RCP_LN2
        g0 = tl.cumsum(g0, 0) * RCP_LN2
        g1 = d0 + tl.cumsum(g1, 0) * RCP_LN2
        g2 = s1 + tl.cumsum(g2, 0) * RCP_LN2
        g3 = s2 + tl.cumsum(g3, 0) * RCP_LN2

    # ---- k k^T: 10 lower-triangular [BC, BC] blocks, plus row sums of k^2
    A00 = tl.zeros([BC, BC], dtype=tl.float32)
    A11 = tl.zeros([BC, BC], dtype=tl.float32)
    A22 = tl.zeros([BC, BC], dtype=tl.float32)
    A33 = tl.zeros([BC, BC], dtype=tl.float32)
    A10 = tl.zeros([BC, BC], dtype=tl.float32)
    A20 = tl.zeros([BC, BC], dtype=tl.float32)
    A21 = tl.zeros([BC, BC], dtype=tl.float32)
    A30 = tl.zeros([BC, BC], dtype=tl.float32)
    A31 = tl.zeros([BC, BC], dtype=tl.float32)
    A32 = tl.zeros([BC, BC], dtype=tl.float32)
    s0 = tl.zeros([BC], dtype=tl.float32)
    s1 = tl.zeros([BC], dtype=tl.float32)
    s2 = tl.zeros([BC], dtype=tl.float32)
    s3 = tl.zeros([BC], dtype=tl.float32)
    for i_k in range(K // BK):
        o_k = i_k * BK + tl.arange(0, BK)
        k0 = tl.load(
            k + r0[:, None] * (HV * K) + o_k[None, :],
            mask=m0[:, None],
            other=0.0,
        )
        k1 = tl.load(
            k + r1[:, None] * (HV * K) + o_k[None, :],
            mask=m1[:, None],
            other=0.0,
        )
        k2 = tl.load(
            k + r2[:, None] * (HV * K) + o_k[None, :],
            mask=m2[:, None],
            other=0.0,
        )
        k3 = tl.load(
            k + r3[:, None] * (HV * K) + o_k[None, :],
            mask=m3[:, None],
            other=0.0,
        )
        if NORM_K:
            f0, f1 = k0.to(tl.float32), k1.to(tl.float32)
            f2, f3 = k2.to(tl.float32), k3.to(tl.float32)
            s0 += tl.sum(f0 * f0, 1)
            s1 += tl.sum(f1 * f1, 1)
            s2 += tl.sum(f2 * f2, 1)
            s3 += tl.sum(f3 * f3, 1)
        A00 = tl.dot(k0, tl.trans(k0), A00)
        A11 = tl.dot(k1, tl.trans(k1), A11)
        A22 = tl.dot(k2, tl.trans(k2), A22)
        A33 = tl.dot(k3, tl.trans(k3), A33)
        A10 = tl.dot(k1, tl.trans(k0), A10)
        A20 = tl.dot(k2, tl.trans(k0), A20)
        A21 = tl.dot(k2, tl.trans(k1), A21)
        A30 = tl.dot(k3, tl.trans(k0), A30)
        A31 = tl.dot(k3, tl.trans(k1), A31)
        A32 = tl.dot(k3, tl.trans(k2), A32)

    if NORM_K:
        n0 = tl.rsqrt(s0 + 1e-6)
        n1 = tl.rsqrt(s1 + 1e-6)
        n2 = tl.rsqrt(s2 + 1e-6)
        n3 = tl.rsqrt(s3 + 1e-6)
        A00 *= n0[:, None] * n0[None, :]
        A11 *= n1[:, None] * n1[None, :]
        A22 *= n2[:, None] * n2[None, :]
        A33 *= n3[:, None] * n3[None, :]
        A10 *= n1[:, None] * n0[None, :]
        A20 *= n2[:, None] * n0[None, :]
        A21 *= n2[:, None] * n1[None, :]
        A30 *= n3[:, None] * n0[None, :]
        A31 *= n3[:, None] * n1[None, :]
        A32 *= n3[:, None] * n2[None, :]
    else:
        n0 = tl.full([BC], 1.0, tl.float32)
        n1 = n0
        n2 = n0
        n3 = n0

    # ---- gate, beta, strict-lower mask
    A00 = _gate(A00, g0, g0, m0, m0, True, o_i) * b0[:, None]
    A11 = _gate(A11, g1, g1, m1, m1, True, o_i) * b1[:, None]
    A22 = _gate(A22, g2, g2, m2, m2, True, o_i) * b2[:, None]
    A33 = _gate(A33, g3, g3, m3, m3, True, o_i) * b3[:, None]
    A10 = _gate(A10, g1, g0, m1, m0, False, o_i) * b1[:, None]
    A20 = _gate(A20, g2, g0, m2, m0, False, o_i) * b2[:, None]
    A21 = _gate(A21, g2, g1, m2, m1, False, o_i) * b2[:, None]
    A30 = _gate(A30, g3, g0, m3, m0, False, o_i) * b3[:, None]
    A31 = _gate(A31, g3, g1, m3, m1, False, o_i) * b3[:, None]
    A32 = _gate(A32, g3, g2, m3, m2, False, o_i) * b3[:, None]

    # ---- (I + A)^{-1}: diagonal forward substitution + block merge
    m_I = o_i[:, None] == o_i[None, :]
    if NEUMANN:
        I00 = _solve_diag_neumann(A00, o_i, BC) + m_I
    else:
        I00 = _solve_diag(A00, o_i, tl.minimum(BC, T - t0), BC) + m_I
    if NEUMANN:
        I11 = _solve_diag_neumann(A11, o_i, BC) + m_I
    else:
        I11 = _solve_diag(A11, o_i, tl.minimum(BC, T - t0 - BC), BC) + m_I
    if NEUMANN:
        I22 = _solve_diag_neumann(A22, o_i, BC) + m_I
    else:
        I22 = _solve_diag(A22, o_i, tl.minimum(BC, T - t0 - 2 * BC), BC) + m_I
    if NEUMANN:
        I33 = _solve_diag_neumann(A33, o_i, BC) + m_I
    else:
        I33 = _solve_diag(A33, o_i, tl.minimum(BC, T - t0 - 3 * BC), BC) + m_I
    P: tl.constexpr = "tf32"
    I10 = -tl.dot(tl.dot(I11, A10, input_precision=P), I00, input_precision=P)
    I21 = -tl.dot(tl.dot(I22, A21, input_precision=P), I11, input_precision=P)
    I32 = -tl.dot(tl.dot(I33, A32, input_precision=P), I22, input_precision=P)
    I20 = -tl.dot(
        I22,
        tl.dot(A20, I00, input_precision=P)
        + tl.dot(A21, I10, input_precision=P),
        input_precision=P,
    )
    I31 = -tl.dot(
        I33,
        tl.dot(A31, I11, input_precision=P)
        + tl.dot(A32, I21, input_precision=P),
        input_precision=P,
    )
    I30 = -tl.dot(
        I33,
        tl.dot(A30, I00, input_precision=P)
        + tl.dot(A31, I10, input_precision=P)
        + tl.dot(A32, I20, input_precision=P),
        input_precision=P,
    )
    # fla stores A in the input dtype before recompute_w_u; round the same way
    dt = k.dtype.element_ty
    I00, I11, I22, I33 = I00.to(dt), I11.to(dt), I22.to(dt), I33.to(dt)
    I10, I20, I21 = I10.to(dt), I20.to(dt), I21.to(dt)
    I30, I31, I32 = I30.to(dt), I31.to(dt), I32.to(dt)

    # ---- w = A @ (beta * exp2(g) * k_n),  u = A @ (beta * v)
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    e0 = (b0 * tl.exp2(g0) * n0)[:, None]
    e1 = (b1 * tl.exp2(g1) * n1)[:, None]
    e2 = (b2 * tl.exp2(g2) * n2)[:, None]
    e3 = (b3 * tl.exp2(g3) * n3)[:, None]
    pk = k + o_k[None, :]
    x0 = tl.load(pk + r0[:, None] * (HV * K), mask=m0[:, None], other=0.0)
    x1 = tl.load(pk + r1[:, None] * (HV * K), mask=m1[:, None], other=0.0)
    x2 = tl.load(pk + r2[:, None] * (HV * K), mask=m2[:, None], other=0.0)
    x3 = tl.load(pk + r3[:, None] * (HV * K), mask=m3[:, None], other=0.0)
    if NORM_K and not BLOCK_OUT:
        pn = kn + o_k[None, :]
        tl.store(
            pn + r0[:, None] * (HV * K),
            (x0 * n0[:, None]).to(dt),
            mask=m0[:, None],
        )
        tl.store(
            pn + r1[:, None] * (HV * K),
            (x1 * n1[:, None]).to(dt),
            mask=m1[:, None],
        )
        tl.store(
            pn + r2[:, None] * (HV * K),
            (x2 * n2[:, None]).to(dt),
            mask=m2[:, None],
        )
        tl.store(
            pn + r3[:, None] * (HV * K),
            (x3 * n3[:, None]).to(dt),
            mask=m3[:, None],
        )
    x0 = (x0 * e0).to(dt)
    x1 = (x1 * e1).to(dt)
    x2 = (x2 * e2).to(dt)
    x3 = (x3 * e3).to(dt)
    pw = w + o_k[None, :]
    tl.store(
        pw + r0[:, None] * (HV * K),
        _wu_row(I00, I00, I00, I00, x0, x0, x0, x0, 0).to(dt),
        mask=m0[:, None],
    )
    tl.store(
        pw + r1[:, None] * (HV * K),
        _wu_row(I10, I11, I11, I11, x0, x1, x1, x1, 1).to(dt),
        mask=m1[:, None],
    )
    tl.store(
        pw + r2[:, None] * (HV * K),
        _wu_row(I20, I21, I22, I22, x0, x1, x2, x2, 2).to(dt),
        mask=m2[:, None],
    )
    tl.store(
        pw + r3[:, None] * (HV * K),
        _wu_row(I30, I31, I32, I33, x0, x1, x2, x3, 3).to(dt),
        mask=m3[:, None],
    )

    pv = v + o_v[None, :]
    y0 = (
        tl.load(pv + r0[:, None] * (HV * V), mask=m0[:, None], other=0.0)
        * b0[:, None]
    ).to(dt)
    y1 = (
        tl.load(pv + r1[:, None] * (HV * V), mask=m1[:, None], other=0.0)
        * b1[:, None]
    ).to(dt)
    y2 = (
        tl.load(pv + r2[:, None] * (HV * V), mask=m2[:, None], other=0.0)
        * b2[:, None]
    ).to(dt)
    y3 = (
        tl.load(pv + r3[:, None] * (HV * V), mask=m3[:, None], other=0.0)
        * b3[:, None]
    ).to(dt)
    pu = u + o_v[None, :]
    tl.store(
        pu + r0[:, None] * (HV * V),
        _wu_row(I00, I00, I00, I00, y0, y0, y0, y0, 0).to(dt),
        mask=m0[:, None],
    )
    tl.store(
        pu + r1[:, None] * (HV * V),
        _wu_row(I10, I11, I11, I11, y0, y1, y1, y1, 1).to(dt),
        mask=m1[:, None],
    )
    tl.store(
        pu + r2[:, None] * (HV * V),
        _wu_row(I20, I21, I22, I22, y0, y1, y2, y2, 2).to(dt),
        mask=m2[:, None],
    )
    tl.store(
        pu + r3[:, None] * (HV * V),
        _wu_row(I30, I31, I32, I33, y0, y1, y2, y3, 3).to(dt),
        mask=m3[:, None],
    )


@triton.jit(do_not_specialize=["T", "t_base", "TB"])
def gdn_fwd_aux_pro6000_kernel(
    q,
    k,
    g,
    gc,
    qg,
    kd,
    S,
    scale,
    T,
    t_base,
    TB,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    NORM: tl.constexpr,
    CUMSUM_G: tl.constexpr = False,
):
    """h-independent recurrence operands of one chunk into block buffers:
    qg = scale * exp2(g) * q_n, kd = exp2(g_last - g) * k_n,
    S = scale * tril(q_n k_n^T * exp2(g_i - g_j))."""
    RCP_LN2: tl.constexpr = 1.4426950408889634
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV
    bos = i_b * T
    obos = i_b * TB - t_base
    t0 = t_base + i_t * BT
    o_c = tl.arange(0, BT)
    o_k = tl.arange(0, K)
    o_t = t0 + o_c
    m_t = o_t < T
    p_row = o_t[:, None] * (HV * K) + o_k[None, :]
    base = (bos * HV + i_h) * K
    b_k = tl.load(k + base + p_row, mask=m_t[:, None], other=0.0).to(tl.float32)
    b_q = tl.load(q + base + p_row, mask=m_t[:, None], other=0.0).to(tl.float32)
    pg = g + bos * HV + i_h
    if CUMSUM_G:
        b_raw = (
            tl.load(pg + o_t * HV, mask=m_t, other=0.0).to(tl.float32) * RCP_LN2
        )
        b_gc = tl.cumsum(b_raw, 0)
        b_gl = tl.sum(b_raw, 0)
        tl.store(gc + bos * HV + i_h + o_t * HV, b_gc, mask=m_t)
    else:
        b_gc = tl.load(pg + o_t * HV, mask=m_t, other=0.0).to(tl.float32)
        b_gl = tl.load(pg + (tl.minimum(t0 + BT, T) - 1) * HV).to(tl.float32)
    if NORM:
        b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k, 1) + 1e-6)[:, None]
        b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q, 1) + 1e-6)[:, None]
    dt = qg.dtype.element_ty
    b_S = tl.dot(b_q.to(dt), tl.trans(b_k.to(dt)))
    m_S = (o_c[:, None] >= o_c[None, :]) & m_t[:, None] & m_t[None, :]
    b_S = tl.where(m_S, b_S * tl.exp2(b_gc[:, None] - b_gc[None, :]), 0.0)
    o_row = (obos * HV + i_h) * K + p_row
    tl.store(
        qg + o_row,
        (b_q * (tl.exp2(b_gc) * scale)[:, None]).to(dt),
        mask=m_t[:, None],
    )
    tl.store(
        kd + o_row,
        (b_k * tl.where(m_t, tl.exp2(b_gl - b_gc), 0.0)[:, None]).to(dt),
        mask=m_t[:, None],
    )
    tl.store(
        S + (obos * HV + i_h) * BT + o_t[:, None] * (HV * BT) + o_c[None, :],
        (b_S * scale).to(dt),
        mask=m_t[:, None],
    )


@triton.jit(do_not_specialize=["T", "t_base", "TB"])
def kda_fwd_prep_pro6000_kernel(
    k,
    v,
    g,
    beta,
    w,
    u,
    T,
    t_base,
    TB,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    NORM_K: tl.constexpr,
    NEUMANN: tl.constexpr = True,
):
    """KDA WY prep for chunk i_t of the token block starting at ``t_base``:
    chunk-local cumsum(g), k L2 norm, per-channel gated kkt (diagonal +
    off-diagonal 16x16 blocks), (I + A)^{-1} solve, and w / u into block
    buffers ([B, TB, H, *]). A and gc never touch DRAM."""
    tl.static_assert(BT == 4 * BC)
    RCP_LN2: tl.constexpr = 1.4426950408889634
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV
    bos = i_b * T
    obos = i_b * TB - t_base

    t0 = t_base + i_t * BT
    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, K)
    r0 = t0 + o_i
    r1 = r0 + BC
    r2 = r1 + BC
    r3 = r2 + BC
    m0, m1, m2, m3 = r0 < T, r1 < T, r2 < T, r3 < T

    k += (bos * HV + i_h) * K
    v += (bos * HV + i_h) * V
    g += (bos * HV + i_h) * K
    w += (obos * HV + i_h) * K
    u += (obos * HV + i_h) * V
    pb = beta + bos * HV + i_h
    b0 = tl.load(pb + r0 * HV, mask=m0, other=0.0).to(tl.float32)
    b1 = tl.load(pb + r1 * HV, mask=m1, other=0.0).to(tl.float32)
    b2 = tl.load(pb + r2 * HV, mask=m2, other=0.0).to(tl.float32)
    b3 = tl.load(pb + r3 * HV, mask=m3, other=0.0).to(tl.float32)

    dt = k.dtype.element_ty
    pk = k + o_k[None, :]
    pg = g + o_k[None, :]
    m_str = o_i[:, None] > o_i[None, :]

    # ---- subchunk 0
    k0 = tl.load(pk + r0[:, None] * (HV * K), mask=m0[:, None], other=0.0).to(
        tl.float32
    )
    if NORM_K:
        k0 = (
            (k0 * tl.rsqrt(tl.sum(k0 * k0, 1) + 1e-6)[:, None])
            .to(dt)
            .to(tl.float32)
        )
    g0 = tl.load(pg + r0[:, None] * (HV * K), mask=m0[:, None], other=0.0).to(
        tl.float32
    )
    d0 = tl.sum(g0, 0) * RCP_LN2
    c0 = tl.cumsum(g0, 0) * RCP_LN2
    gm0 = tl.where(
        m0[:, None],
        tl.maximum(tl.minimum(c0 - 0.5 * d0[None, :], 60.0), -60.0),
        0.0,
    )
    A00 = tl.where(
        m_str & m0[:, None] & m0[None, :],
        tl.dot(
            (k0 * tl.exp2(gm0)).to(dt), tl.trans((k0 * tl.exp2(-gm0)).to(dt))
        )
        * b0[:, None],
        0.0,
    )
    x0 = (k0 * (b0[:, None] * tl.where(m0[:, None], tl.exp2(c0), 0.0))).to(dt)
    k0_end = k0 * tl.where(m0[:, None], tl.exp2(d0[None, :] - c0), 0.0)

    # ---- subchunk 1
    k1 = tl.load(pk + r1[:, None] * (HV * K), mask=m1[:, None], other=0.0).to(
        tl.float32
    )
    if NORM_K:
        k1 = (
            (k1 * tl.rsqrt(tl.sum(k1 * k1, 1) + 1e-6)[:, None])
            .to(dt)
            .to(tl.float32)
        )
    g1 = tl.load(pg + r1[:, None] * (HV * K), mask=m1[:, None], other=0.0).to(
        tl.float32
    )
    d1 = tl.sum(g1, 0) * RCP_LN2
    c1 = tl.cumsum(g1, 0) * RCP_LN2
    gm1 = tl.where(
        m1[:, None],
        tl.maximum(tl.minimum(c1 - 0.5 * d1[None, :], 60.0), -60.0),
        0.0,
    )
    A11 = tl.where(
        m_str & m1[:, None] & m1[None, :],
        tl.dot(
            (k1 * tl.exp2(gm1)).to(dt), tl.trans((k1 * tl.exp2(-gm1)).to(dt))
        )
        * b1[:, None],
        0.0,
    )
    kr1 = (k1 * tl.where(m1[:, None], tl.exp2(c1), 0.0)).to(dt)
    A10 = tl.dot(kr1, tl.trans(k0_end.to(dt))) * b1[:, None]
    s1 = d0 + d1
    x1 = (
        k1
        * (b1[:, None] * tl.where(m1[:, None], tl.exp2(d0[None, :] + c1), 0.0))
    ).to(dt)
    k1_end = k1 * tl.where(m1[:, None], tl.exp2(d1[None, :] - c1), 0.0)

    # ---- subchunk 2
    k2 = tl.load(pk + r2[:, None] * (HV * K), mask=m2[:, None], other=0.0).to(
        tl.float32
    )
    if NORM_K:
        k2 = (
            (k2 * tl.rsqrt(tl.sum(k2 * k2, 1) + 1e-6)[:, None])
            .to(dt)
            .to(tl.float32)
        )
    g2 = tl.load(pg + r2[:, None] * (HV * K), mask=m2[:, None], other=0.0).to(
        tl.float32
    )
    d2 = tl.sum(g2, 0) * RCP_LN2
    c2 = tl.cumsum(g2, 0) * RCP_LN2
    gm2 = tl.where(
        m2[:, None],
        tl.maximum(tl.minimum(c2 - 0.5 * d2[None, :], 60.0), -60.0),
        0.0,
    )
    A22 = tl.where(
        m_str & m2[:, None] & m2[None, :],
        tl.dot(
            (k2 * tl.exp2(gm2)).to(dt), tl.trans((k2 * tl.exp2(-gm2)).to(dt))
        )
        * b2[:, None],
        0.0,
    )
    kr2 = (k2 * tl.where(m2[:, None], tl.exp2(c2), 0.0)).to(dt)
    ed1 = tl.exp2(d1)[None, :]
    A20 = tl.dot(kr2, tl.trans((k0_end * ed1).to(dt))) * b2[:, None]
    A21 = tl.dot(kr2, tl.trans(k1_end.to(dt))) * b2[:, None]
    s2 = s1 + d2
    x2 = (
        k2
        * (b2[:, None] * tl.where(m2[:, None], tl.exp2(s1[None, :] + c2), 0.0))
    ).to(dt)
    k2_end = k2 * tl.where(m2[:, None], tl.exp2(d2[None, :] - c2), 0.0)

    # ---- subchunk 3
    k3 = tl.load(pk + r3[:, None] * (HV * K), mask=m3[:, None], other=0.0).to(
        tl.float32
    )
    if NORM_K:
        k3 = (
            (k3 * tl.rsqrt(tl.sum(k3 * k3, 1) + 1e-6)[:, None])
            .to(dt)
            .to(tl.float32)
        )
    g3 = tl.load(pg + r3[:, None] * (HV * K), mask=m3[:, None], other=0.0).to(
        tl.float32
    )
    d3 = tl.sum(g3, 0) * RCP_LN2
    c3 = tl.cumsum(g3, 0) * RCP_LN2
    gm3 = tl.where(
        m3[:, None],
        tl.maximum(tl.minimum(c3 - 0.5 * d3[None, :], 60.0), -60.0),
        0.0,
    )
    A33 = tl.where(
        m_str & m3[:, None] & m3[None, :],
        tl.dot(
            (k3 * tl.exp2(gm3)).to(dt), tl.trans((k3 * tl.exp2(-gm3)).to(dt))
        )
        * b3[:, None],
        0.0,
    )
    kr3 = (k3 * tl.where(m3[:, None], tl.exp2(c3), 0.0)).to(dt)
    ed2 = tl.exp2(d2)[None, :]
    ed12 = tl.exp2(d1 + d2)[None, :]
    A30 = tl.dot(kr3, tl.trans((k0_end * ed12).to(dt))) * b3[:, None]
    A31 = tl.dot(kr3, tl.trans((k1_end * ed2).to(dt))) * b3[:, None]
    A32 = tl.dot(kr3, tl.trans(k2_end.to(dt))) * b3[:, None]
    x3 = (
        k3
        * (b3[:, None] * tl.where(m3[:, None], tl.exp2(s2[None, :] + c3), 0.0))
    ).to(dt)

    # ---- (I + A)^{-1}: diagonal solve + block merge
    m_I = o_i[:, None] == o_i[None, :]
    if NEUMANN:
        I00 = _solve_diag_neumann(A00, o_i, BC) + m_I
        I11 = _solve_diag_neumann(A11, o_i, BC) + m_I
        I22 = _solve_diag_neumann(A22, o_i, BC) + m_I
        I33 = _solve_diag_neumann(A33, o_i, BC) + m_I
    else:
        I00 = _solve_diag(A00, o_i, tl.minimum(BC, T - t0), BC) + m_I
        I11 = _solve_diag(A11, o_i, tl.minimum(BC, T - t0 - BC), BC) + m_I
        I22 = _solve_diag(A22, o_i, tl.minimum(BC, T - t0 - 2 * BC), BC) + m_I
        I33 = _solve_diag(A33, o_i, tl.minimum(BC, T - t0 - 3 * BC), BC) + m_I
    P: tl.constexpr = "tf32"
    I10 = -tl.dot(tl.dot(I11, A10, input_precision=P), I00, input_precision=P)
    I21 = -tl.dot(tl.dot(I22, A21, input_precision=P), I11, input_precision=P)
    I32 = -tl.dot(tl.dot(I33, A32, input_precision=P), I22, input_precision=P)
    I20 = -tl.dot(
        I22,
        tl.dot(A20, I00, input_precision=P)
        + tl.dot(A21, I10, input_precision=P),
        input_precision=P,
    )
    I31 = -tl.dot(
        I33,
        tl.dot(A31, I11, input_precision=P)
        + tl.dot(A32, I21, input_precision=P),
        input_precision=P,
    )
    I30 = -tl.dot(
        I33,
        tl.dot(A30, I00, input_precision=P)
        + tl.dot(A31, I10, input_precision=P)
        + tl.dot(A32, I20, input_precision=P),
        input_precision=P,
    )
    I00, I11, I22, I33 = I00.to(dt), I11.to(dt), I22.to(dt), I33.to(dt)
    I10, I20, I21 = I10.to(dt), I20.to(dt), I21.to(dt)
    I30, I31, I32 = I30.to(dt), I31.to(dt), I32.to(dt)

    # ---- w = A @ (beta * exp2(g) * k_n),  u = A @ (beta * v)
    pw = w + o_k[None, :]
    tl.store(
        pw + r0[:, None] * (HV * K),
        _wu_row(I00, I00, I00, I00, x0, x0, x0, x0, 0).to(dt),
        mask=m0[:, None],
    )
    tl.store(
        pw + r1[:, None] * (HV * K),
        _wu_row(I10, I11, I11, I11, x0, x1, x1, x1, 1).to(dt),
        mask=m1[:, None],
    )
    tl.store(
        pw + r2[:, None] * (HV * K),
        _wu_row(I20, I21, I22, I22, x0, x1, x2, x2, 2).to(dt),
        mask=m2[:, None],
    )
    tl.store(
        pw + r3[:, None] * (HV * K),
        _wu_row(I30, I31, I32, I33, x0, x1, x2, x3, 3).to(dt),
        mask=m3[:, None],
    )

    o_v = tl.arange(0, V)
    pv = v + o_v[None, :]
    y0 = (
        tl.load(pv + r0[:, None] * (HV * V), mask=m0[:, None], other=0.0)
        * b0[:, None]
    ).to(dt)
    y1 = (
        tl.load(pv + r1[:, None] * (HV * V), mask=m1[:, None], other=0.0)
        * b1[:, None]
    ).to(dt)
    y2 = (
        tl.load(pv + r2[:, None] * (HV * V), mask=m2[:, None], other=0.0)
        * b2[:, None]
    ).to(dt)
    y3 = (
        tl.load(pv + r3[:, None] * (HV * V), mask=m3[:, None], other=0.0)
        * b3[:, None]
    ).to(dt)
    pu = u + o_v[None, :]
    tl.store(
        pu + r0[:, None] * (HV * V),
        _wu_row(I00, I00, I00, I00, y0, y0, y0, y0, 0).to(dt),
        mask=m0[:, None],
    )
    tl.store(
        pu + r1[:, None] * (HV * V),
        _wu_row(I10, I11, I11, I11, y0, y1, y1, y1, 1).to(dt),
        mask=m1[:, None],
    )
    tl.store(
        pu + r2[:, None] * (HV * V),
        _wu_row(I20, I21, I22, I22, y0, y1, y2, y2, 2).to(dt),
        mask=m2[:, None],
    )
    tl.store(
        pu + r3[:, None] * (HV * V),
        _wu_row(I30, I31, I32, I33, y0, y1, y2, y3, 3).to(dt),
        mask=m3[:, None],
    )


@triton.jit(do_not_specialize=["T", "t_base", "TB"])
def kda_fwd_aux_pro6000_kernel(
    q,
    k,
    g,
    qg,
    kd,
    S,
    gl,
    scale,
    T,
    t_base,
    TB,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    NORM: tl.constexpr,
):
    """KDA h-independent recurrence operands of one chunk into block buffers:
    qg = scale * exp2(g) * q_n, kd = exp2(g_last - g) * k_n,
    S = scale * tril(q_n k_n^T * exp2(g_i - g_j)), gl = g_last [K]."""
    tl.static_assert(BT == 4 * BC)
    RCP_LN2: tl.constexpr = 1.4426950408889634
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV
    bos = i_b * T
    obos = i_b * TB - t_base

    t0 = t_base + i_t * BT
    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, K)
    r0 = t0 + o_i
    r1 = r0 + BC
    r2 = r1 + BC
    r3 = r2 + BC
    m0, m1, m2, m3 = r0 < T, r1 < T, r2 < T, r3 < T

    base = (bos * HV + i_h) * K
    obase = (obos * HV + i_h) * K
    pq = q + base + o_k[None, :]
    pk = k + base + o_k[None, :]
    pg = g + base + o_k[None, :]
    pqg = qg + obase + o_k[None, :]
    pkd = kd + obase + o_k[None, :]
    pS = S + (obos * HV + i_h) * BT + o_i[None, :]

    dt = qg.dtype.element_ty
    m_inc = o_i[:, None] >= o_i[None, :]
    z = tl.zeros([BC, BC], dtype=dt)

    # ---- subchunk 0
    q0 = tl.load(pq + r0[:, None] * (HV * K), mask=m0[:, None], other=0.0).to(
        tl.float32
    )
    k0 = tl.load(pk + r0[:, None] * (HV * K), mask=m0[:, None], other=0.0).to(
        tl.float32
    )
    if NORM:
        q0 = (
            (q0 * tl.rsqrt(tl.sum(q0 * q0, 1) + 1e-6)[:, None])
            .to(dt)
            .to(tl.float32)
        )
        k0 = (
            (k0 * tl.rsqrt(tl.sum(k0 * k0, 1) + 1e-6)[:, None])
            .to(dt)
            .to(tl.float32)
        )
    g0 = tl.load(pg + r0[:, None] * (HV * K), mask=m0[:, None], other=0.0).to(
        tl.float32
    )
    d0 = tl.sum(g0, 0) * RCP_LN2
    c0 = tl.cumsum(g0, 0) * RCP_LN2
    gm0 = tl.where(
        m0[:, None],
        tl.maximum(tl.minimum(c0 - 0.5 * d0[None, :], 60.0), -60.0),
        0.0,
    )
    S00 = tl.where(
        m_inc & m0[:, None] & m0[None, :],
        tl.dot(
            (q0 * tl.exp2(gm0)).to(dt), tl.trans((k0 * tl.exp2(-gm0)).to(dt))
        )
        * scale,
        0.0,
    )
    tl.store(
        pqg + r0[:, None] * (HV * K),
        (q0 * (tl.exp2(c0) * scale)).to(dt),
        mask=m0[:, None],
    )
    tl.store(pS + r0[:, None] * (HV * BT), S00.to(dt), mask=m0[:, None])
    tl.store(pS + r0[:, None] * (HV * BT) + BC, z, mask=m0[:, None])
    tl.store(pS + r0[:, None] * (HV * BT) + 2 * BC, z, mask=m0[:, None])
    tl.store(pS + r0[:, None] * (HV * BT) + 3 * BC, z, mask=m0[:, None])
    k0_end = k0 * tl.where(m0[:, None], tl.exp2(d0[None, :] - c0), 0.0)

    # ---- subchunk 1
    q1 = tl.load(pq + r1[:, None] * (HV * K), mask=m1[:, None], other=0.0).to(
        tl.float32
    )
    k1 = tl.load(pk + r1[:, None] * (HV * K), mask=m1[:, None], other=0.0).to(
        tl.float32
    )
    if NORM:
        q1 = (
            (q1 * tl.rsqrt(tl.sum(q1 * q1, 1) + 1e-6)[:, None])
            .to(dt)
            .to(tl.float32)
        )
        k1 = (
            (k1 * tl.rsqrt(tl.sum(k1 * k1, 1) + 1e-6)[:, None])
            .to(dt)
            .to(tl.float32)
        )
    g1 = tl.load(pg + r1[:, None] * (HV * K), mask=m1[:, None], other=0.0).to(
        tl.float32
    )
    d1 = tl.sum(g1, 0) * RCP_LN2
    c1 = tl.cumsum(g1, 0) * RCP_LN2
    gm1 = tl.where(
        m1[:, None],
        tl.maximum(tl.minimum(c1 - 0.5 * d1[None, :], 60.0), -60.0),
        0.0,
    )
    S11 = tl.where(
        m_inc & m1[:, None] & m1[None, :],
        tl.dot(
            (q1 * tl.exp2(gm1)).to(dt), tl.trans((k1 * tl.exp2(-gm1)).to(dt))
        )
        * scale,
        0.0,
    )
    qr1 = (q1 * tl.where(m1[:, None], tl.exp2(c1), 0.0)).to(dt)
    S10 = tl.dot(qr1, tl.trans(k0_end.to(dt))) * scale
    s1 = d0 + d1
    tl.store(
        pqg + r1[:, None] * (HV * K),
        (q1 * (tl.exp2(d0[None, :] + c1) * scale)).to(dt),
        mask=m1[:, None],
    )
    tl.store(pS + r1[:, None] * (HV * BT), S10.to(dt), mask=m1[:, None])
    tl.store(pS + r1[:, None] * (HV * BT) + BC, S11.to(dt), mask=m1[:, None])
    tl.store(pS + r1[:, None] * (HV * BT) + 2 * BC, z, mask=m1[:, None])
    tl.store(pS + r1[:, None] * (HV * BT) + 3 * BC, z, mask=m1[:, None])
    k1_end = k1 * tl.where(m1[:, None], tl.exp2(d1[None, :] - c1), 0.0)

    # ---- subchunk 2
    q2 = tl.load(pq + r2[:, None] * (HV * K), mask=m2[:, None], other=0.0).to(
        tl.float32
    )
    k2 = tl.load(pk + r2[:, None] * (HV * K), mask=m2[:, None], other=0.0).to(
        tl.float32
    )
    if NORM:
        q2 = (
            (q2 * tl.rsqrt(tl.sum(q2 * q2, 1) + 1e-6)[:, None])
            .to(dt)
            .to(tl.float32)
        )
        k2 = (
            (k2 * tl.rsqrt(tl.sum(k2 * k2, 1) + 1e-6)[:, None])
            .to(dt)
            .to(tl.float32)
        )
    g2 = tl.load(pg + r2[:, None] * (HV * K), mask=m2[:, None], other=0.0).to(
        tl.float32
    )
    d2 = tl.sum(g2, 0) * RCP_LN2
    c2 = tl.cumsum(g2, 0) * RCP_LN2
    gm2 = tl.where(
        m2[:, None],
        tl.maximum(tl.minimum(c2 - 0.5 * d2[None, :], 60.0), -60.0),
        0.0,
    )
    S22 = tl.where(
        m_inc & m2[:, None] & m2[None, :],
        tl.dot(
            (q2 * tl.exp2(gm2)).to(dt), tl.trans((k2 * tl.exp2(-gm2)).to(dt))
        )
        * scale,
        0.0,
    )
    qr2 = (q2 * tl.where(m2[:, None], tl.exp2(c2), 0.0)).to(dt)
    ed1 = tl.exp2(d1)[None, :]
    S20 = tl.dot(qr2, tl.trans((k0_end * ed1).to(dt))) * scale
    S21 = tl.dot(qr2, tl.trans(k1_end.to(dt))) * scale
    s2 = s1 + d2
    tl.store(
        pqg + r2[:, None] * (HV * K),
        (q2 * (tl.exp2(s1[None, :] + c2) * scale)).to(dt),
        mask=m2[:, None],
    )
    tl.store(pS + r2[:, None] * (HV * BT), S20.to(dt), mask=m2[:, None])
    tl.store(pS + r2[:, None] * (HV * BT) + BC, S21.to(dt), mask=m2[:, None])
    tl.store(
        pS + r2[:, None] * (HV * BT) + 2 * BC, S22.to(dt), mask=m2[:, None]
    )
    tl.store(pS + r2[:, None] * (HV * BT) + 3 * BC, z, mask=m2[:, None])
    k2_end = k2 * tl.where(m2[:, None], tl.exp2(d2[None, :] - c2), 0.0)

    # ---- subchunk 3
    q3 = tl.load(pq + r3[:, None] * (HV * K), mask=m3[:, None], other=0.0).to(
        tl.float32
    )
    k3 = tl.load(pk + r3[:, None] * (HV * K), mask=m3[:, None], other=0.0).to(
        tl.float32
    )
    if NORM:
        q3 = (
            (q3 * tl.rsqrt(tl.sum(q3 * q3, 1) + 1e-6)[:, None])
            .to(dt)
            .to(tl.float32)
        )
        k3 = (
            (k3 * tl.rsqrt(tl.sum(k3 * k3, 1) + 1e-6)[:, None])
            .to(dt)
            .to(tl.float32)
        )
    g3 = tl.load(pg + r3[:, None] * (HV * K), mask=m3[:, None], other=0.0).to(
        tl.float32
    )
    d3 = tl.sum(g3, 0) * RCP_LN2
    c3 = tl.cumsum(g3, 0) * RCP_LN2
    gm3 = tl.where(
        m3[:, None],
        tl.maximum(tl.minimum(c3 - 0.5 * d3[None, :], 60.0), -60.0),
        0.0,
    )
    S33 = tl.where(
        m_inc & m3[:, None] & m3[None, :],
        tl.dot(
            (q3 * tl.exp2(gm3)).to(dt), tl.trans((k3 * tl.exp2(-gm3)).to(dt))
        )
        * scale,
        0.0,
    )
    qr3 = (q3 * tl.where(m3[:, None], tl.exp2(c3), 0.0)).to(dt)
    ed2 = tl.exp2(d2)[None, :]
    ed12 = tl.exp2(d1 + d2)[None, :]
    S30 = tl.dot(qr3, tl.trans((k0_end * ed12).to(dt))) * scale
    S31 = tl.dot(qr3, tl.trans((k1_end * ed2).to(dt))) * scale
    S32 = tl.dot(qr3, tl.trans(k2_end.to(dt))) * scale
    tl.store(
        pqg + r3[:, None] * (HV * K),
        (q3 * (tl.exp2(s2[None, :] + c3) * scale)).to(dt),
        mask=m3[:, None],
    )
    tl.store(pS + r3[:, None] * (HV * BT), S30.to(dt), mask=m3[:, None])
    tl.store(pS + r3[:, None] * (HV * BT) + BC, S31.to(dt), mask=m3[:, None])
    tl.store(
        pS + r3[:, None] * (HV * BT) + 2 * BC, S32.to(dt), mask=m3[:, None]
    )
    tl.store(
        pS + r3[:, None] * (HV * BT) + 3 * BC, S33.to(dt), mask=m3[:, None]
    )
    k3_end = k3 * tl.where(m3[:, None], tl.exp2(d3[None, :] - c3), 0.0)

    # ---- kd = exp2(g_last - g) * k_n,  gl = g_last
    ed3 = tl.exp2(d3)[None, :]
    ed23 = tl.exp2(d2 + d3)[None, :]
    ed123 = tl.exp2(d1 + d2 + d3)[None, :]
    tl.store(
        pkd + r0[:, None] * (HV * K), (k0_end * ed123).to(dt), mask=m0[:, None]
    )
    tl.store(
        pkd + r1[:, None] * (HV * K), (k1_end * ed23).to(dt), mask=m1[:, None]
    )
    tl.store(
        pkd + r2[:, None] * (HV * K), (k2_end * ed3).to(dt), mask=m2[:, None]
    )
    tl.store(pkd + r3[:, None] * (HV * K), k3_end.to(dt), mask=m3[:, None])
    tl.store(gl + ((i_b * (TB // BT) + i_t) * HV + i_h) * K + o_k, s2 + d3)


def gdn_fwd_prep(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    norm_k: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """WY representation of one GDN chunk pass.

    k, v: [B, T, H, 128] bf16 (k raw when ``norm_k``), g: chunk-local
    cumsum of the log2 decay [B, T, H] fp32, beta: [B, T, H].
    Returns (w, u, k_normalised); k_normalised is ``k`` when not norm_k.
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    w = torch.empty_like(k)
    u = torch.empty_like(v)
    kn = torch.empty_like(k) if norm_k else k
    gdn_fwd_prep_pro6000_kernel[(triton.cdiv(T, 64), B * H)](
        k,
        v,
        g,
        beta,
        w,
        u,
        kn,
        T,
        0,
        T,
        HV=H,
        K=K,
        V=V,
        BT=64,
        BC=16,
        BK=K,
        NORM_K=norm_k,
        NEUMANN=True,
        num_warps=2,
        num_stages=1,
    )
    return w, u, kn
