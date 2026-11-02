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
"""Fused kkt + solve_tril + recompute_w_u (+ k L2 norm), forward only."""

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
def _gate(b_A, gi, gj, mi, mj, STRICT: tl.constexpr, o_i):
    m = mi[:, None] & mj[None, :]
    if STRICT:
        m = m & (o_i[:, None] > o_i[None, :])
    return b_A * tl.where(m, tl.exp2(gi[:, None] - gj[None, :]), 0.0)


@triton.jit(do_not_specialize=["T"])
def gdn_fwd_prep_kernel(
    k,
    v,
    g,
    beta,
    w,
    u,
    kn,
    T,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NORM_K: tl.constexpr,
):
    tl.static_assert(BT == 4 * BC)
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV
    bos = i_b * T

    t0 = i_t * BT
    o_i = tl.arange(0, BC)
    r0 = t0 + o_i
    r1 = r0 + BC
    r2 = r1 + BC
    r3 = r2 + BC
    m0, m1, m2, m3 = r0 < T, r1 < T, r2 < T, r3 < T

    k += (bos * HV + i_h) * K
    v += (bos * HV + i_h) * V
    w += (bos * HV + i_h) * K
    u += (bos * HV + i_h) * V
    kn += (bos * HV + i_h) * K
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
    if NORM_K:
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
    gdn_fwd_prep_kernel[(triton.cdiv(T, 64), B * H)](
        k,
        v,
        g,
        beta,
        w,
        u,
        kn,
        T,
        HV=H,
        K=K,
        V=V,
        BT=64,
        BC=16,
        BK=K,
        NORM_K=norm_k,
        num_warps=2,
        num_stages=1,
    )
    return w, u, kn
