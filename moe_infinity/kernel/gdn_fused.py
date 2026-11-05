# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Modified for MoE-Infinity (sm_120 audit, #304): flash-linear-attention
# 0.6.0 `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` fused with
# `chunk_fwd_kernel_o` (GDN, scalar gate) / `chunk_gla_fwd_kernel_o` (KDA,
# per-channel gate), so the per-chunk states `h` and `v_new` never reach
# DRAM. sm_120 has no TMEM / tcgen05, so the Blackwell `tl.dot` recurrence
# race that keeps fla at num_warps=2 (triton-lang/triton#10590, sm_100/103)
# does not apply; tests check bitwise determinism at the 4-warp config.
"""Fused chunked delta-rule recurrence + output (forward, inference).

Hardware: RTX_PRO_6000 (tuned on the RTX PRO 6000 Blackwell Max-Q: 188 SMs,
128 MiB L2, 99 KiB shared memory per block). Reached only through
``install_pro6000_linear_attention``, which checks
``hw_profile.supports(RTX_PRO_6000)``; every other GPU runs fla's kernels.
Tuned here: BV giving one wave of 128 CTAs on 188 SMs, 4 warps (safe on
sm_120, which has no TMEM), 2 stages for GDN / 1 for KDA to fit 99 KiB.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def delta_rule_fwd_h_o_pro6000_kernel(
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
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    GATE_VEC: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    """One program per (sequence, head, BV slice of V); loops over chunks.

    v_new = u - w @ h, then
      scalar gate g [T] (GDN, g = chunk-local cumsum, log2):
        o = scale * (exp2(g) * (q @ h) + tril(q k^T * exp2(g_i - g_j)) @ v_new)
        h = exp2(g_last) * h + k^T @ (exp2(g_last - g) * v_new)
      per-channel gate g [T, K] (KDA; k is fla's decayed kg, Aqk its scaled
      intra-chunk scores):
        o = (scale * q * exp2(g)) @ h + tril(Aqk) @ v_new
        h = exp2(g_last)[:, None] * h + k^T @ v_new
    q / k / w / u / Aqk are bf16, g fp32; h is fp32 in registers.
    """
    pid = tl.program_id(0).to(tl.int64)
    NV: tl.constexpr = V // BV
    i_v, i_nh = pid % NV, pid // NV
    i_n, i_h = i_nh // HV, i_nh % HV
    bos = i_n * T
    NT = tl.cdiv(T, BT)

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

    b_h = tl.zeros([K, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        b_h += tl.load(h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :])

    for i_t in range(NT):
        o_t = i_t * BT + o_c
        m_t = o_t < T
        last = tl.minimum((i_t + 1) * BT, T) - 1
        p_row = o_t[:, None] * (HV * K) + o_k[None, :]
        b_w = tl.load(w + p_row, mask=m_t[:, None], other=0.0)
        b_q = tl.load(q + p_row, mask=m_t[:, None], other=0.0)
        b_k = tl.load(
            k + o_k[:, None] + o_t[None, :] * (HV * K),
            mask=m_t[None, :],
            other=0.0,
        )
        b_u = tl.load(
            u + o_t[:, None] * (HV * V) + o_v[None, :],
            mask=m_t[:, None],
            other=0.0,
        )
        m_S = m_c & m_t[:, None] & m_t[None, :]
        if GATE_VEC:
            b_g = tl.load(g + p_row, mask=m_t[:, None], other=0.0)
            b_gl = tl.load(g + last * (HV * K) + o_k)
            b_q = (b_q * tl.exp2(b_g) * scale).to(b_w.dtype)
            b_S = tl.load(
                Aqk + o_t[:, None] * (HV * BT) + o_c[None, :],
                mask=m_t[:, None],
                other=0.0,
            )
            b_S = tl.where(m_S, b_S, 0.0)
        else:
            b_g = tl.load(g + o_t * HV, mask=m_t, other=0.0)
            b_gl = tl.load(g + last * HV)
            b_S = tl.dot(b_q, b_k)
            b_S = tl.where(m_S, b_S * tl.exp2(b_g[:, None] - b_g[None, :]), 0.0)

        b_hb = b_h.to(b_w.dtype)
        b_v = (b_u.to(tl.float32) - tl.dot(b_w, b_hb)).to(b_w.dtype)
        b_o = tl.dot(b_q, b_hb)
        if GATE_VEC:
            b_o = tl.dot(b_S.to(b_v.dtype), b_v, b_o)
        else:
            b_o = tl.dot(b_S.to(b_v.dtype), b_v, b_o * tl.exp2(b_g)[:, None])
            b_o = b_o * scale
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

    if STORE_FINAL_STATE:
        p = ht + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p, b_h.to(ht.dtype.element_ty))


def delta_rule_fwd_h_o(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    Aqk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    BV: int | None = None,
    num_warps: int = 4,
    num_stages: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fused recurrence + output. ``g`` is [B, T, H] (GDN; requires Aqk None)
    or [B, T, H, K] with ``Aqk`` [B, T, H, 64] (KDA). Returns (o, final h)."""
    B, T, HV, K = k.shape
    V = u.shape[-1]
    gate_vec = g.dim() == 4
    assert gate_vec == (Aqk is not None)
    if BV is None:
        # one wave on 188 SMs: 32 heads x 4 slices (GDN), 64 heads x 2 (KDA)
        BV = 32 if B * HV * (V // 32) <= 192 else 64
    if num_stages is None:
        # the per-channel gate tile does not fit two stages in 100 KB smem
        num_stages = 1 if gate_vec else 2
    o = torch.empty_like(u)
    ht = (
        k.new_empty(B, HV, K, V, dtype=torch.float32)
        if output_final_state
        else None
    )
    delta_rule_fwd_h_o_pro6000_kernel[(V // BV * B * HV,)](
        q,
        k,
        w,
        u,
        g,
        Aqk if gate_vec else u,
        o,
        initial_state,
        ht,
        scale,
        T,
        HV=HV,
        K=K,
        V=V,
        BT=64,
        BV=BV,
        GATE_VEC=gate_vec,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=output_final_state,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o, ht
