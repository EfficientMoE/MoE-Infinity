# Fused MoE routing kernels (#315).
#
# Hardware: the routing kernels below (Triton names carry ``pro6000``) and the
# moe_infinity._router extension are gated RTX_PRO_6000: their tile / stage
# tables, split-K counts and dispatch thresholds were tuned on the RTX PRO 6000
# Blackwell. Other devices run ``_route_reference`` (the previous torch
# routing); see ``_pro6000_routing``.

from typing import NamedTuple

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def launch_softmax(input_tensor):
    n_rows, n_cols = input_tensor.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_stages = 4 if BLOCK_SIZE <= 2048 else 2
    output = torch.empty_like(input_tensor)
    softmax_kernel[(n_rows,)](
        output,
        input_tensor,
        input_tensor.stride(0),
        output.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
    )
    return output


@triton.jit
def fused_softmax_topk_kernel_nobias(
    hidden_ptr,
    weight_ptr,
    routing_mask_ptr,
    routing_weight_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_E: tl.constexpr,
    normalize_topk: tl.constexpr,
):
    batch_id = tl.program_id(0)
    off_e = tl.arange(0, BLOCK_E)

    logits = tl.zeros([BLOCK_E], dtype=tl.float32)

    for h in range(H):
        h_val = tl.load(hidden_ptr + batch_id * H + h)
        w_ptr = weight_ptr + off_e * H + h
        valid = off_e < E
        w_val = tl.load(w_ptr, mask=valid, other=0.0)
        logits = tl.where(valid, logits + h_val * w_val, logits)

    logits = tl.where(off_e < E, logits, -float("inf"))

    max_logit = tl.max(logits, axis=0)
    logits = logits - max_logit
    exp_logits = tl.exp(logits)
    sum_exp = tl.sum(exp_logits, axis=0)
    probs = exp_logits / sum_exp

    remaining = tl.where(off_e < E, probs, -float("inf"))
    top_mask = tl.zeros([BLOCK_E], dtype=tl.int1)

    for _k in range(TOPK):
        cur_max = tl.max(remaining, axis=0)
        is_max = remaining == cur_max
        first = is_max & (tl.cumsum(is_max.to(tl.int32), axis=0) == 1)
        top_mask = top_mask | first
        remaining = tl.where(first, -float("inf"), remaining)

    top_probs = tl.where(top_mask, probs, tl.zeros([BLOCK_E], dtype=tl.float32))

    if normalize_topk:
        top_sum = tl.sum(top_probs, axis=0)
        top_probs = top_probs / top_sum

    tl.store(routing_mask_ptr + batch_id * E + off_e, top_mask, mask=off_e < E)
    tl.store(
        routing_weight_ptr + batch_id * E + off_e,
        top_probs.to(tl.bfloat16),
        mask=off_e < E,
    )


def launch_fused_softmax_topk_nobias(
    hidden_states, weight, top_k, normalize_topk=True
):
    B, H = hidden_states.shape
    E = weight.shape[0]
    dtype = hidden_states.dtype

    routing_mask = torch.zeros(
        (B, E), dtype=torch.bool, device=hidden_states.device
    )
    routing_weight = torch.zeros(
        (B, E), dtype=dtype, device=hidden_states.device
    )

    BLOCK_E = triton.next_power_of_2(E)

    fused_softmax_topk_kernel_nobias[(B,)](
        hidden_states,
        weight,
        routing_mask,
        routing_weight,
        B=B,
        H=H,
        E=E,
        TOPK=top_k,
        BLOCK_E=BLOCK_E,
        normalize_topk=normalize_topk,
    )

    return routing_mask, routing_weight


# --------------------------------------------------------------------------
# Fused routing (#315): gate logits on tensor cores + scoring + (grouped)
# top-k + renorm, written straight to compact ids/weights and the dense
# [T, E] mask/weights the expert dispatcher consumes. One launch at decode
# (split-K over H, deterministic last-CTA reduction); at most two at prefill.
# --------------------------------------------------------------------------

SCORING_SOFTMAX = 0
SCORING_SIGMOID = 1
SCORING_SQRTSOFTPLUS = 2
_SCORING = {
    "softmax": SCORING_SOFTMAX,
    "sigmoid": SCORING_SIGMOID,
    "sqrtsoftplus": SCORING_SQRTSOFTPLUS,
}


@triton.jit
def _route_epilogue(
    logits,
    offs_t,
    offs_e,
    T,
    bias_ptr,
    ids_ptr,
    tw_ptr,
    mask_ptr,
    dw_ptr,
    logits_out_ptr,
    scale,
    eps,
    E: tl.constexpr,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
    BLOCK_E: tl.constexpr,
    SCORING: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    N_GROUP: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
    RENORM: tl.constexpr,
    ROUND_LOGITS: tl.constexpr,
    WRITE_LOGITS: tl.constexpr,
    WRITE_DENSE: tl.constexpr,
):
    valid_t = offs_t < T
    valid_e = offs_e < E
    io_mask = valid_t[:, None] & valid_e[None, :]
    if ROUND_LOGITS:
        # Softmax routers see bf16 logits today (F.linear in model dtype).
        logits = logits.to(tl.bfloat16).to(tl.float32)
    if WRITE_LOGITS:
        tl.store(
            logits_out_ptr + offs_t[:, None] * E + offs_e[None, :],
            logits.to(logits_out_ptr.dtype.element_ty),
            mask=io_mask,
        )
    neg_inf = float("-inf")
    if SCORING == 0:
        x = tl.where(valid_e[None, :], logits, neg_inf)
        x = x - tl.max(x, axis=1)[:, None]
        num = tl.exp(x)
        scores = num / tl.sum(num, axis=1)[:, None]
        sel = scores
    else:
        # Same formulas and IEEE-rounded ops as ATen's CUDA kernels, so
        # mathematically tied scores (e.g. s(a) + s(-a) == 1) round the same
        # way and tie-breaking matches the torch routers bit for bit.
        if SCORING == 1:
            scores = tl.div_rn(1.0, 1.0 + libdevice.exp(-logits))
        else:
            # sqrt(softplus(x)), torch softplus threshold 20.
            sp = tl.where(
                logits > 20.0, logits, libdevice.log1p(libdevice.exp(logits))
            )
            scores = tl.sqrt_rn(sp)
        sel = scores
        if HAS_BIAS:
            b = tl.load(bias_ptr + offs_e, mask=valid_e, other=0.0)
            sel = scores + b[None, :].to(tl.float32)
    sel = tl.where(valid_e[None, :], sel, neg_inf)

    if N_GROUP > 1:
        # noaux_tc: group score = sum of the group's top-2 selection scores;
        # keep the TOPK_GROUP best groups, mask the rest with -inf.
        GS: tl.constexpr = E // N_GROUP
        gid = offs_e // GS
        gexp = tl.full(sel.shape, neg_inf, tl.float32)
        for g in tl.static_range(N_GROUP):
            in_g = (gid == g) & valid_e
            v = tl.where(in_g[None, :], sel, neg_inf)
            m1 = tl.max(v, axis=1)
            i1 = tl.min(
                tl.where(v == m1[:, None], offs_e[None, :], BLOCK_E), axis=1
            )
            v2 = tl.where(offs_e[None, :] == i1[:, None], neg_inf, v)
            m2 = tl.max(v2, axis=1)
            gexp = tl.where(in_g[None, :], (m1 + m2)[:, None], gexp)
        keep = tl.zeros(sel.shape, tl.int1)
        for _j in tl.static_range(TOPK_GROUP):
            gm = tl.max(gexp, axis=1)
            gsel = tl.min(
                tl.where(gexp == gm[:, None], gid[None, :], N_GROUP), axis=1
            )
            chosen_g = gid[None, :] == gsel[:, None]
            keep = keep | chosen_g
            gexp = tl.where(chosen_g, neg_inf, gexp)
        sel = tl.where(keep, sel, neg_inf)

    offs_k = tl.arange(0, K_PAD)
    ids_tile = tl.zeros([sel.shape[0], K_PAD], tl.int32)
    w_tile = tl.zeros([sel.shape[0], K_PAD], tl.float32)
    selmask = tl.zeros(sel.shape, tl.int1)
    for i in tl.static_range(K):
        m = tl.max(sel, axis=1)
        idx = tl.min(
            tl.where(sel == m[:, None], offs_e[None, :], BLOCK_E), axis=1
        )
        chosen = offs_e[None, :] == idx[:, None]
        w_i = tl.sum(tl.where(chosen, scores, 0.0), axis=1)
        ids_tile = tl.where(offs_k[None, :] == i, idx[:, None], ids_tile)
        w_tile = tl.where(offs_k[None, :] == i, w_i[:, None], w_tile)
        selmask = selmask | chosen
        sel = tl.where(chosen, neg_inf, sel)

    if RENORM:
        inv = 1.0 / (tl.sum(w_tile, axis=1) + eps)
    else:
        inv = tl.full([sel.shape[0]], 1.0, tl.float32)
    k_mask = valid_t[:, None] & (offs_k[None, :] < K)
    tw = w_tile * inv[:, None] * scale
    tl.store(ids_ptr + offs_t[:, None] * K + offs_k[None, :], ids_tile, k_mask)
    tl.store(tw_ptr + offs_t[:, None] * K + offs_k[None, :], tw, k_mask)
    if WRITE_DENSE:
        dw = tl.where(selmask, scores * inv[:, None] * scale, 0.0)
        tl.store(
            mask_ptr + offs_t[:, None] * E + offs_e[None, :],
            selmask,
            mask=io_mask,
        )
        tl.store(
            dw_ptr + offs_t[:, None] * E + offs_e[None, :],
            dw.to(dw_ptr.dtype.element_ty),
            mask=io_mask,
        )


@triton.jit
def _fused_route_pro6000_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    logits_in_ptr,
    partial_ptr,
    counter_ptr,
    ids_ptr,
    tw_ptr,
    mask_ptr,
    dw_ptr,
    logits_out_ptr,
    T,
    stride_xt,
    stride_we,
    scale,
    eps,
    H: tl.constexpr,
    E: tl.constexpr,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_H: tl.constexpr,
    SPLIT_K: tl.constexpr,
    H_PER_SPLIT: tl.constexpr,
    SCORING: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    N_GROUP: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
    RENORM: tl.constexpr,
    ROUND_LOGITS: tl.constexpr,
    LOGITS_IN: tl.constexpr,
    WRITE_LOGITS: tl.constexpr,
    WRITE_DENSE: tl.constexpr,
    PRECISION: tl.constexpr,
):
    pid_t = tl.program_id(0)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_e = tl.arange(0, BLOCK_E)
    valid_t = offs_t < T
    valid_e = offs_e < E

    if LOGITS_IN:
        acc = tl.load(
            logits_in_ptr + offs_t[:, None] * E + offs_e[None, :],
            mask=valid_t[:, None] & valid_e[None, :],
            other=0.0,
        ).to(tl.float32)
        _route_epilogue(
            acc, offs_t, offs_e, T, bias_ptr, ids_ptr, tw_ptr, mask_ptr,
            dw_ptr, logits_out_ptr, scale, eps, E, K, K_PAD, BLOCK_E,
            SCORING, HAS_BIAS, N_GROUP, TOPK_GROUP, RENORM, ROUND_LOGITS,
            WRITE_LOGITS, WRITE_DENSE,
        )  # fmt: skip
    else:
        pid_k = tl.program_id(1)
        offs_h = tl.arange(0, BLOCK_H)
        acc = tl.zeros([BLOCK_T, BLOCK_E], dtype=tl.float32)
        x_rows = x_ptr + offs_t[:, None] * stride_xt
        w_rows = w_ptr + offs_e[:, None] * stride_we
        h_lo = pid_k * H_PER_SPLIT
        for h0 in range(0, H_PER_SPLIT, BLOCK_H):
            hh = h_lo + h0 + offs_h
            hm = hh < H
            xa = tl.load(
                x_rows + hh[None, :], mask=valid_t[:, None] & hm[None, :],
                other=0.0,
            )  # fmt: skip
            wb = tl.load(
                w_rows + hh[None, :], mask=valid_e[:, None] & hm[None, :],
                other=0.0,
            )  # fmt: skip
            if xa.dtype != wb.dtype:
                xa = xa.to(wb.dtype)
            acc = tl.dot(xa, tl.trans(wb), acc, input_precision=PRECISION)
        if SPLIT_K == 1:
            _route_epilogue(
                acc, offs_t, offs_e, T, bias_ptr, ids_ptr, tw_ptr, mask_ptr,
                dw_ptr, logits_out_ptr, scale, eps, E, K, K_PAD, BLOCK_E,
                SCORING, HAS_BIAS, N_GROUP, TOPK_GROUP, RENORM, ROUND_LOGITS,
                WRITE_LOGITS, WRITE_DENSE,
            )  # fmt: skip
        else:
            # Partials of valid rows only: [SPLIT_K, T, BLOCK_E] fp32.
            p_rows = partial_ptr + offs_t[:, None] * BLOCK_E + offs_e[None, :]
            tl.store(p_rows + pid_k * T * BLOCK_E, acc, mask=valid_t[:, None])
            tl.debug_barrier()
            done = tl.atomic_add(
                counter_ptr + pid_t, 1, sem="acq_rel", scope="gpu"
            )
            if done == SPLIT_K - 1:
                # Last CTA of this token tile: reduce in fixed split order
                # (deterministic), reset the counter, run the epilogue.
                tl.debug_barrier()
                total = tl.zeros([BLOCK_T, BLOCK_E], dtype=tl.float32)
                for s in range(SPLIT_K):
                    total += tl.load(
                        p_rows + s * T * BLOCK_E,
                        mask=valid_t[:, None],
                        other=0.0,
                        cache_modifier=".cg",
                    )
                tl.store(counter_ptr + pid_t, 0)
                _route_epilogue(
                    total, offs_t, offs_e, T, bias_ptr, ids_ptr, tw_ptr,
                    mask_ptr, dw_ptr, logits_out_ptr, scale, eps, E, K,
                    K_PAD, BLOCK_E, SCORING, HAS_BIAS, N_GROUP, TOPK_GROUP,
                    RENORM, ROUND_LOGITS, WRITE_LOGITS, WRITE_DENSE,
                )  # fmt: skip


class _RouteScratch:
    """Per (device, stream) split-K scratch: partials + self-resetting counters."""

    def __init__(self, device):
        self.device = device
        self.partial = torch.empty(0, dtype=torch.float32, device=device)
        self.counter = torch.zeros(0, dtype=torch.int32, device=device)

    def get(self, n_partial: int, n_counter: int):
        if self.partial.numel() < n_partial:
            self.partial = torch.empty(
                n_partial, dtype=torch.float32, device=self.device
            )
        if self.counter.numel() < n_counter:
            self.counter = torch.zeros(
                max(n_counter, 1024), dtype=torch.int32, device=self.device
            )
        return self.partial, self.counter


_SCRATCH: dict = {}
_SM_COUNT: dict = {}


def _scratch(device) -> _RouteScratch:
    key = (device, torch.cuda.current_stream(device).stream_id)
    s = _SCRATCH.get(key)
    if s is None:
        s = _SCRATCH[key] = _RouteScratch(device)
    return s


def _sm_count(device) -> int:
    idx = device.index if device.index is not None else 0
    n = _SM_COUNT.get(idx)
    if n is None:
        n = _SM_COUNT[idx] = torch.cuda.get_device_properties(
            idx
        ).multi_processor_count
    return n


# Token count above which logits come from one cuBLAS tensor-core GEMM and a
# second launch fuses scoring/top-k (the fused-dot kernel is used below it).
PREFILL_GEMM_MIN_TOKENS_PRO6000 = 4096


# --------------------------------------------------------------------------
# Prefill: one fused tensor-core GEMM + top-k launch. Each program owns a
# [BM, E] logits tile (E <= 512 fits in one tile row), accumulates it over H
# with tl.dot (bf16 operands, fp32 accumulate: exact products), and the
# epilogue scores it, selects the top-k and writes the routing outputs. X is
# read once, logits never leave the SM.
# --------------------------------------------------------------------------

_I64_MIN = -9223372036854775808
# The Triton GEMM + top-k kernel replaces cuBLASLt logits + routing (one
# launch instead of two) where it measured faster on sm_120: few experts
# (E <= 128: the op streams X) from 2K tokens. For E >= 256 the [BM, E]
# full-row tile re-reads W per tile and the cuBLASLt path stays faster.
PREFILL_FUSED_MIN_TOKENS_PRO6000 = 2048
PREFILL_FUSED_MAX_EXPERTS_PRO6000 = 128


def _use_prefill_kernel(x, weight, gemm_min_tokens) -> bool:
    return (
        not gemm_min_tokens
        and x.is_cuda
        and x.shape[0] >= PREFILL_FUSED_MIN_TOKENS_PRO6000
        and weight.shape[0] <= PREFILL_FUSED_MAX_EXPERTS_PRO6000
        and weight.dtype == x.dtype
        and x.dtype in _LOW_PRECISION
    )


@triton.jit
def _sel_keys(v, offs_e, valid, BLOCK_E: tl.constexpr):
    # Monotone int64 key: (ordered float bits) << 32 | (BLOCK_E - 1 - e), so
    # one max picks the largest value and, among equal values, the lowest
    # expert index (torch's tie order).
    u = v.to(tl.int32, bitcast=True)
    o = u ^ ((u >> 31) & 0x7FFFFFFF)
    key = (o.to(tl.int64) << 32) | (BLOCK_E - 1 - offs_e).to(tl.int64)[None, :]
    return tl.where(valid, key, -9223372036854775808)


# seq / T change every call: specialising on them would recompile.
@triton.jit(do_not_specialize=["seq", "T"])
def _gemm_route_pro6000_kernel(
    x_ptr, w_ptr, bias_ptr, mask_ptr, dw_ptr, ids_ptr, tw_ptr, logits_ptr,
    flags_ptr, done_ptr, state_ptr, seq,
    T, stride_xt, stride_we, scale, eps,
    H: tl.constexpr, E: tl.constexpr, K: tl.constexpr, K_PAD: tl.constexpr,
    BLOCK_E: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
    SCORING: tl.constexpr, HAS_BIAS: tl.constexpr, N_GROUP: tl.constexpr,
    TOPK_GROUP: tl.constexpr, RENORM: tl.constexpr,
    ROUND_LOGITS: tl.constexpr, WRITE_DENSE: tl.constexpr,
    WRITE_IDS: tl.constexpr, WRITE_LOGITS: tl.constexpr,
    WRITE_FLAGS: tl.constexpr,
):  # fmt: skip
    pid = tl.program_id(0)
    offs_t = pid * BM + tl.arange(0, BM)
    offs_e = tl.arange(0, BLOCK_E)
    offs_k = tl.arange(0, BK)
    valid_t = offs_t < T
    valid_e = offs_e < E
    x_ptrs = x_ptr + offs_t[:, None] * stride_xt + offs_k[None, :]
    w_ptrs = w_ptr + offs_e[:, None] * stride_we + offs_k[None, :]
    acc = tl.zeros([BM, BLOCK_E], dtype=tl.float32)
    for k0 in range(0, H, BK):
        if H % BK == 0:
            a = tl.load(x_ptrs, mask=valid_t[:, None], other=0.0)
            b = tl.load(w_ptrs, mask=valid_e[:, None], other=0.0)
        else:
            km = (k0 + offs_k) < H
            a = tl.load(x_ptrs, mask=valid_t[:, None] & km[None, :], other=0.0)
            b = tl.load(w_ptrs, mask=valid_e[:, None] & km[None, :], other=0.0)
        acc = tl.dot(a, tl.trans(b), acc)
        x_ptrs += BK
        w_ptrs += BK

    logits = acc
    if ROUND_LOGITS:
        logits = logits.to(tl.bfloat16).to(tl.float32)
    io_mask = valid_t[:, None] & valid_e[None, :]
    if WRITE_LOGITS:
        tl.store(
            logits_ptr + offs_t[:, None] * E + offs_e[None, :],
            logits.to(logits_ptr.dtype.element_ty), mask=io_mask,
        )  # fmt: skip
    if SCORING == 0:
        scores = logits  # softmax is monotone: select on the logits
        sel = logits
    else:
        # IEEE-rounded ops in ATen's order, so selection matches torch.
        if SCORING == 1:
            scores = tl.div_rn(1.0, 1.0 + libdevice.exp(-logits))
        else:
            sp = tl.where(
                logits > 20.0, logits, libdevice.log1p(libdevice.exp(logits))
            )
            scores = tl.sqrt_rn(sp)
        sel = scores
        if HAS_BIAS:
            b = tl.load(bias_ptr + offs_e, mask=valid_e, other=0.0)
            sel = scores + b[None, :]
    keys = _sel_keys(sel, offs_e, valid_e[None, :], BLOCK_E)

    if N_GROUP > 1:
        # noaux_tc: group score = sum of its top-2 selection values; keep the
        # TOPK_GROUP best groups (lowest index on ties).
        GS: tl.constexpr = E // N_GROUP
        gid = offs_e // GS
        neg_inf = float("-inf")
        gexp = tl.full(sel.shape, neg_inf, tl.float32)
        for g in tl.static_range(N_GROUP):
            in_g = (gid == g)[None, :] & valid_e[None, :]
            kg = tl.where(in_g, keys, -9223372036854775808)
            k1 = tl.max(kg, axis=1)
            m1 = tl.max(tl.where(kg == k1[:, None], sel, neg_inf), axis=1)
            kg2 = tl.where(kg == k1[:, None], -9223372036854775808, kg)
            k2 = tl.max(kg2, axis=1)
            m2 = tl.max(tl.where(kg2 == k2[:, None], sel, neg_inf), axis=1)
            gexp = tl.where(in_g, (m1 + m2)[:, None], gexp)
        keep = tl.zeros(sel.shape, tl.int1)
        for _j in tl.static_range(TOPK_GROUP):
            gk = _sel_keys(gexp, gid, valid_e[None, :], N_GROUP)
            gm = tl.max(gk, axis=1)
            gsel = N_GROUP - 1 - (gm & 0xFFFFFFFF).to(tl.int32)
            chosen_g = gid[None, :] == gsel[:, None]
            keep = keep | chosen_g
            gexp = tl.where(chosen_g, neg_inf, gexp)
        masked = _sel_keys(
            tl.full(sel.shape, neg_inf, tl.float32), offs_e,
            valid_e[None, :], BLOCK_E,
        )  # fmt: skip
        keys = tl.where(keep, keys, masked)

    # int32 keys (value only) + a lowest-index pass per step: half the
    # registers of a packed int64 key, so the [BM, E] epilogue doesn't spill.
    k32 = (keys >> 32).to(tl.int32)
    k32 = tl.where(valid_e[None, :], k32, -2147483648)
    selmask = tl.zeros([BM, BLOCK_E], tl.int1)
    offs_kk = tl.arange(0, K_PAD)
    ids_tile = tl.zeros([BM, K_PAD], tl.int32)
    for i in tl.static_range(K):
        m = tl.max(k32, axis=1)
        idx = tl.min(
            tl.where(k32 == m[:, None], offs_e[None, :], BLOCK_E), axis=1
        )
        chosen = offs_e[None, :] == idx[:, None]
        selmask = selmask | chosen
        k32 = tl.where(chosen, -2147483648, k32)
        if WRITE_IDS:
            ids_tile = tl.where(offs_kk[None, :] == i, idx[:, None], ids_tile)

    if SCORING == 0:
        top = tl.max(tl.where(selmask, logits, float("-inf")), axis=1)
        num = tl.where(selmask, tl.exp(logits - top[:, None]), 0.0)
        if RENORM:
            inv = 1.0 / (tl.sum(num, axis=1) + eps)
        else:
            full = tl.where(
                valid_e[None, :], tl.exp(logits - top[:, None]), 0.0
            )
            inv = 1.0 / tl.sum(full, axis=1)
    else:
        num = tl.where(selmask, scores, 0.0)
        if RENORM:
            inv = 1.0 / (tl.sum(num, axis=1) + eps)
        else:
            inv = tl.full([BM], 1.0, tl.float32)
    dw = num * inv[:, None] * scale
    if WRITE_DENSE:
        tl.store(
            mask_ptr + offs_t[:, None] * E + offs_e[None, :], selmask,
            mask=io_mask,
        )  # fmt: skip
        tl.store(
            dw_ptr + offs_t[:, None] * E + offs_e[None, :],
            dw.to(dw_ptr.dtype.element_ty), mask=io_mask,
        )  # fmt: skip
    if WRITE_IDS:
        tw_tile = tl.zeros([BM, K_PAD], tl.float32)
        for i in tl.static_range(K):
            idx_i = tl.sum(tl.where(offs_kk[None, :] == i, ids_tile, 0), axis=1)
            w_i = tl.sum(
                tl.where(offs_e[None, :] == idx_i[:, None], dw, 0.0), axis=1
            )
            tw_tile = tl.where(offs_kk[None, :] == i, w_i[:, None], tw_tile)
        k_mask = valid_t[:, None] & (offs_kk[None, :] < K)
        tl.store(
            ids_ptr + offs_t[:, None] * K + offs_kk[None, :], ids_tile, k_mask
        )
        tl.store(
            tw_ptr + offs_t[:, None] * K + offs_kk[None, :], tw_tile, k_mask
        )
    if WRITE_FLAGS:
        # OR this tile's active experts into a device-side [E] buffer; the
        # last program publishes it to pinned host memory (pre-zeroed by the
        # host), re-zeroes it and writes the ring's completion word. Writing
        # host memory from every program costs ~25 % at 200K tokens.
        act = tl.max((selmask & valid_t[:, None]).to(tl.int32), axis=0)
        tl.atomic_or(state_ptr + offs_e, act, mask=(act > 0) & valid_e)
        tl.debug_barrier()
        n_done = tl.atomic_add(state_ptr + 512, 1, sem="acq_rel", scope="gpu")
        if n_done == tl.num_programs(0) - 1:
            agg = tl.atomic_xchg(state_ptr + offs_e, 0, mask=valid_e)
            tl.store(flags_ptr + offs_e, (agg > 0).to(tl.int8), mask=valid_e)
            tl.store(state_ptr + 512, 0)
            tl.store(done_ptr, seq)


# Measured best (BM, BK, num_warps, num_stages) per padded E on sm_120
# (50K-token sweep); other sizes fall back to the formula below.
_FLAG_STATE: dict = {}


def _flag_state(dev):
    # [512] OR-buffer + arrival counter, zero between launches (the last
    # program of each launch restores it). One per (device, stream).
    key = (dev, torch.cuda.current_stream(dev).stream_id)
    st = _FLAG_STATE.get(key)
    if st is None:
        st = _FLAG_STATE[key] = torch.zeros(513, dtype=torch.int32, device=dev)
    return st


_GEMM_CFG_PRO6000: dict = {
    16: (128, 64, 8, 2),
    32: (32, 128, 4, 2),
    128: (64, 64, 8, 4),
    256: (64, 64, 16, 3),
    512: (64, 32, 8, 2),
}


# T <= 16K (pro6000 sweep, round 6): smaller BM / longer BK so short prompts
# put more CTAs on the 188 SMs (2K: 16-32 CTAs with the large-T table).
def _small_t_config_pro6000(T, block_e):
    if T > 16384:
        return None
    if block_e == 16:
        return (64, 128, 8, 4)
    if block_e == 32:
        return (32, 128, 4, 4)
    if block_e == 128:
        if T <= 4096:
            return (32, 128, 8, 3)
        return (64, 128, 8, 3) if T <= 10240 else (128, 64, 8, 4)
    return None


def _gemm_config(T, H, E, block_e):
    """(BM, BK, num_warps, num_stages) for the fused prefill kernel."""
    small = _small_t_config_pro6000(T, block_e)
    if small is not None:
        return small
    if block_e == 128 and H >= 6144:
        return (64, 64, 4, 3)  # MiniMax-M3 (H = 6144) sweep
    cfg = _GEMM_CFG_PRO6000.get(block_e)
    if cfg is not None:
        return cfg
    bm = 128 if block_e <= 128 else (64 if block_e <= 256 else 32)
    bk = 64 if block_e <= 256 else 32
    stage = (bm + block_e) * bk * 2
    return (bm, bk, 8, max(2, min(4, (80 * 1024) // stage)))


def _route_prefill(
    x, w, top_k, sc, bias, n_group, topk_group, renormalize, scaling, eps,
    round_logits, weights_dtype, logits_dtype, return_logits, dense,
    active_flags, compact, cfg=None,
):  # fmt: skip
    T, H = x.shape
    E = w.shape[0]
    dev = x.device
    block_e = max(16, triton.next_power_of_2(E))
    BM, BK, nw, ns = cfg or _gemm_config(T, H, E, block_e)
    mask = dw = ids = tw = logits = flags = None
    if dense:
        mask = torch.empty((T, E), dtype=torch.bool, device=dev)
        dw = torch.empty((T, E), dtype=weights_dtype, device=dev)
    if compact:
        ids = torch.empty((T, top_k), dtype=torch.int32, device=dev)
        tw = torch.empty((T, top_k), dtype=torch.float32, device=dev)
    if return_logits:
        logits = torch.empty((T, E), dtype=logits_dtype, device=dev)
    done = seq = None
    if active_flags and _CUDA_EXT is not None:
        flags, done, seq = _CUDA_EXT.acquire_flags(dev.index or 0, E)
    if bias is not None and (bias.dtype != torch.float32 or not bias.is_cuda):
        bias = bias.to(device=dev, dtype=torch.float32)
    dummy = x
    _gemm_route_pro6000_kernel[(triton.cdiv(T, BM),)](
        x, w, bias if bias is not None else dummy,
        mask if dense else dummy, dw if dense else dummy,
        ids if compact else dummy, tw if compact else dummy,
        logits if return_logits else dummy,
        flags if flags is not None else dummy,
        done if done is not None else dummy,
        _flag_state(dev) if flags is not None else dummy, seq or 0,
        T, x.stride(0), w.stride(0), float(scaling), float(eps),
        H=H, E=E, K=top_k, K_PAD=triton.next_power_of_2(top_k),
        BLOCK_E=block_e, BM=BM, BK=BK, SCORING=sc,
        HAS_BIAS=bias is not None, N_GROUP=n_group, TOPK_GROUP=topk_group,
        RENORM=bool(renormalize), ROUND_LOGITS=bool(round_logits),
        WRITE_DENSE=bool(dense), WRITE_IDS=bool(compact),
        WRITE_LOGITS=bool(return_logits), WRITE_FLAGS=flags is not None,
        num_warps=nw, num_stages=ns,
    )  # fmt: skip
    if flags is not None and dense:
        _CUDA_EXT.remember_flags(mask, flags)
    return (ids, tw, mask, dw, logits, flags)


class RouteOutput(NamedTuple):
    topk_ids: torch.Tensor | None
    topk_weights: torch.Tensor | None
    router_mask: torch.Tensor | None
    routing_weights: torch.Tensor | None
    logits: torch.Tensor | None
    active_flags: torch.Tensor | None


def take_active_flags(router_mask):
    """Pinned ``[E]`` active-expert flags produced together with
    ``router_mask`` by the last fused routing call on this thread, or
    ``None``. Consumes the entry."""
    if _CUDA_EXT is None:
        return None
    return _CUDA_EXT.take_active_flags(router_mask)


# CUDA backend: token count from which logits come from cuBLAS + one
# epilogue launch instead of the single split-K launch.
CUDA_GEMM_MIN_TOKENS = 0  # 0: automatic choice in the extension
try:
    from moe_infinity import _router as _CUDA_EXT
except ImportError:  # extension not built: Triton / torch fallbacks
    _CUDA_EXT = None

_LOW_PRECISION = (torch.bfloat16, torch.float16)


_PLAN_OVERRIDE: dict = {}


def _plan_launch_pro6000(T: int, H: int, E: int, block_e: int, device):
    """Pick (BLOCK_T, BLOCK_H, SPLIT_K, num_warps, num_stages)."""
    if _PLAN_OVERRIDE:
        return _PLAN_OVERRIDE["plan"]
    sms = _sm_count(device)
    block_h = 32 if block_e > 256 else 64
    block_t = 16 if T <= 64 else (32 if block_e > 256 else 64)
    # ~64 KB of pipelined operand tiles (sm_120: 99 KB smem per CTA).
    stage_bytes = (block_t + block_e) * block_h * 2
    num_stages = max(2, min(4, (64 * 1024) // stage_bytes))
    n_t = triton.cdiv(T, block_t)
    max_split = max(1, H // block_h)
    split = max(1, min(max_split, triton.cdiv(2 * sms, n_t)))
    # Bound the last CTA's reduction volume (valid rows only) to ~256 KB.
    rows = min(T, block_t)
    split = max(1, min(split, (256 * 1024) // (4 * rows * block_e)))
    num_warps = 4 if block_t <= 16 else 8
    return block_t, block_h, split, num_warps, num_stages


# --------------------------------------------------------------------------
# Hardware gate. Every routing kernel in this module and in
# moe_infinity._router (split-K decode, cuBLASLt + routing, Triton fused
# prefill, CUTLASS GEMM + top-k) was tuned on the RTX PRO 6000 Blackwell
# (188 SMs, 128 MiB L2, 99 KiB shared memory per block): split counts from the
# SM count, tile / stage tables, one-wave thresholds. On any other device the
# routing falls back to the previous torch implementation (_route_reference:
# F.linear + softmax / sigmoid / sqrt-softplus + top-k, the models' own code).
# The sync-free expert dispatch around it is generic and stays on.
# --------------------------------------------------------------------------

_PRO6000_ROUTING: dict = {}


def _pro6000_routing(device) -> bool:
    """True if the RTX PRO 6000-tuned routing kernels may run on ``device``
    (evaluated once per device)."""
    if device is None or torch.device(device).type != "cuda":
        return False
    idx = torch.device(device).index
    if idx is None:
        idx = torch.cuda.current_device()
    ok = _PRO6000_ROUTING.get(idx)
    if ok is None:
        from moe_infinity.utils.hw_profile import RTX_PRO_6000, supports

        ok = _PRO6000_ROUTING[idx] = supports(RTX_PRO_6000, idx)
    return ok


def router_supported(device) -> bool:
    """Public form of the gate: callers with their own torch routing (the
    moe-store wrappers) keep it when this is ``False``."""
    return _pro6000_routing(device)


def _route_reference(
    x, w, top_k, sc, bias, n_group, topk_group, renormalize, scaling, eps,
    round_logits, weights_dtype, logits_dtype, return_logits, dense,
):  # fmt: skip
    """Previous (torch) routing: what the models ran before the fused router."""
    T, E = x.shape[0], w.shape[0]
    if not 1 <= top_k <= E:
        raise ValueError("top_k must be in [1, E]")
    if round_logits or (sc == SCORING_SOFTMAX and x.dtype in _LOW_PRECISION):
        logits = torch.nn.functional.linear(x, w.to(x.dtype))
        lg = logits.float()
    else:
        lg = torch.nn.functional.linear(x.float(), w.float())
        logits = lg
    if sc == SCORING_SOFTMAX:
        scores = torch.softmax(lg, dim=-1)
        sel = scores
    else:
        if sc == SCORING_SIGMOID:
            scores = torch.sigmoid(lg)
        else:
            scores = torch.nn.functional.softplus(lg).sqrt()
        sel = (
            scores + bias.to(lg.device, torch.float32)
            if bias is not None
            else scores
        )
    if n_group > 1:
        if E % n_group or not 1 <= topk_group <= n_group:
            raise ValueError("invalid n_group / topk_group")
        gs = sel.view(T, n_group, E // n_group).topk(2, dim=-1)[0].sum(-1)
        gi = torch.topk(gs, k=topk_group, dim=-1, sorted=False)[1]
        gm = torch.zeros_like(gs).scatter_(1, gi, 1.0)
        keep = gm.unsqueeze(-1).expand(T, n_group, E // n_group).reshape(T, E)
        sel = sel.masked_fill(keep == 0, float("-inf"))
    ids = torch.topk(sel, top_k, dim=-1)[1]
    tw = scores.gather(1, ids)
    if renormalize:
        tw = tw / (tw.sum(dim=-1, keepdim=True) + eps)
    tw = (tw * scaling).float()
    ids = ids.to(torch.int32)
    mask = dw = None
    if dense:
        mask = torch.zeros((T, E), dtype=torch.bool, device=x.device)
        mask.scatter_(1, ids.long(), True)
        dw = torch.zeros((T, E), dtype=weights_dtype, device=x.device)
        dw.scatter_(1, ids.long(), tw.to(weights_dtype))
    out_logits = logits.to(logits_dtype) if return_logits else None
    return RouteOutput(ids, tw, mask, dw, out_logits, None)


def fused_route(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    top_k: int,
    *,
    scoring: str = "softmax",
    bias: torch.Tensor | None = None,
    n_group: int = 1,
    topk_group: int = 1,
    renormalize: bool = True,
    scaling: float = 1.0,
    eps: float = 0.0,
    round_logits: bool | None = None,
    weights_dtype: torch.dtype | None = None,
    return_logits: bool = False,
    dense: bool = True,
    gemm_min_tokens: int | None = None,
    backend: str = "auto",
    active_flags: bool = False,
    compact: bool = True,
):
    """Fused MoE routing.

    Args:
        hidden_states: ``[T, H]`` activations.
        weight: ``[E, H]`` gate weight.
        top_k: experts per token.
        scoring: ``"softmax"`` (select top-k of softmax probs), ``"sigmoid"`` or
            ``"sqrtsoftplus"`` (select top-k of ``score + bias``, weights are the
            unbiased scores).
        bias: optional ``[E]`` selection bias (``e_score_correction_bias``).
        n_group, topk_group: noaux_tc grouped selection (group score = sum of
            the group's top-2 selection scores).
        renormalize: divide the selected weights by ``sum + eps``.
        scaling: multiply the final weights (``routed_scaling_factor``).
        round_logits: round logits to bf16 before scoring (default: on for
            softmax with bf16 inputs, matching ``F.linear`` in model dtype).
        weights_dtype: dtype of the dense weights (default: activations dtype
            for softmax, fp32 otherwise).
        return_logits: also return ``[T, E]`` logits (activations dtype for
            softmax, fp32 otherwise).
        dense: also write the dense ``[T, E]`` mask / weights.
        compact: also return ``topk_ids`` / ``topk_weights`` (CUDA backend may
            skip them when ``False``: one allocation less per call).
        backend: ``"cuda"`` (``moe_infinity._router``), ``"triton"``,
            ``"reference"`` (torch) or ``"auto"`` (CUDA when built and
            applicable, else Triton). Off the RTX PRO 6000 every backend
            runs the torch reference (see ``_pro6000_routing``).
        active_flags: also produce the ``[E]`` active-expert flags directly
            in pinned host memory (CUDA backend; written by the routing kernel,
            valid once the stream reaches this point). They are remembered for
            :func:`take_active_flags` so the expert dispatcher skips its own
            mask reduction + device-to-host copy.

    Returns:
        :class:`RouteOutput` ``(topk_ids [T,k] int32, topk_weights [T,k] fp32,
        router_mask [T,E] bool | None, routing_weights [T,E] | None,
        logits | None, active_flags [E] bool (pinned CPU) | None)``.

    Logits accumulate in fp32 on tensor cores (bf16 MMA for bf16 operands --
    products are exact, so this equals fp32 logits up to summation order;
    ``tf32x3`` for fp32 operands). No host synchronisation.
    """
    sc = _SCORING.get(scoring)
    if sc is None:
        raise ValueError(f"unknown scoring {scoring!r}")
    x = hidden_states
    if x.dim() != 2:
        x = x.reshape(-1, x.shape[-1])
    xdt = x.dtype
    softmax_lp = sc == SCORING_SOFTMAX and xdt in _LOW_PRECISION
    if round_logits is None:
        round_logits = sc == SCORING_SOFTMAX and xdt == torch.bfloat16
    if weights_dtype is None:
        weights_dtype = xdt if sc == SCORING_SOFTMAX else torch.float32
    logits_dtype = xdt if softmax_lp else torch.float32

    if backend == "reference" or not (x.is_cuda and _pro6000_routing(x.device)):
        return _route_reference(
            x, weight, top_k, sc, bias, n_group, topk_group, renormalize,
            scaling, eps, round_logits, weights_dtype, logits_dtype,
            return_logits, dense,
        )  # fmt: skip
    if backend == "prefill" or (
        backend == "auto" and _use_prefill_kernel(x, weight, gemm_min_tokens)
    ):
        return RouteOutput(*_route_prefill(
            x, weight, top_k, sc, bias, n_group, topk_group, renormalize,
            scaling, eps, round_logits, weights_dtype, logits_dtype,
            return_logits, dense, active_flags, compact,
        ))  # fmt: skip
    if (
        _CUDA_EXT is not None
        and backend not in ("triton", "prefill")
        and x.is_cuda
        and weight.dtype == xdt
        and xdt in _LOW_PRECISION
    ):
        # Kept lean: this runs once per MoE layer per decode step.
        res = _CUDA_EXT.fused_route(
            x, weight, top_k, sc, bias, n_group, topk_group, renormalize,
            scaling, eps, round_logits, weights_dtype, logits_dtype,
            return_logits, dense, active_flags, compact,
            CUDA_GEMM_MIN_TOKENS if gemm_min_tokens is None
            else gemm_min_tokens,
        )  # fmt: skip
        if res:
            return RouteOutput._make(res)
    if backend == "cuda":
        raise RuntimeError(
            "moe_infinity._router is unavailable for these inputs"
        )
    return _fused_route_triton(
        x, weight, top_k, sc, bias, n_group, topk_group, renormalize,
        scaling, eps, round_logits, weights_dtype, logits_dtype,
        return_logits, dense, gemm_min_tokens,
    )  # fmt: skip


def make_router(
    weight: torch.Tensor,
    top_k: int,
    *,
    scoring: str = "softmax",
    bias: torch.Tensor | None = None,
    n_group: int = 1,
    topk_group: int = 1,
    renormalize: bool = True,
    scaling: float = 1.0,
    eps: float = 0.0,
    round_logits: bool | None = None,
    weights_dtype: torch.dtype | None = None,
    return_logits: bool = False,
    dense: bool = True,
    active_flags: bool = False,
    compact: bool = True,
    gemm_min_tokens: int | None = None,
):
    """A router bound to one gate: ``router(hidden_states)`` returns the same
    six outputs as :func:`fused_route` (as a plain tuple, in
    :class:`RouteOutput` order). With the CUDA extension the static
    arguments are converted once, so a call costs one native call -- build
    it once per MoE layer. Falls back to :func:`fused_route`."""
    sc = _SCORING.get(scoring)
    if sc is None:
        raise ValueError(f"unknown scoring {scoring!r}")
    wdt = weight.dtype
    if round_logits is None:
        round_logits = sc == SCORING_SOFTMAX and wdt == torch.bfloat16
    if weights_dtype is None:
        weights_dtype = wdt if sc == SCORING_SOFTMAX else torch.float32
    logits_dtype = (
        wdt
        if sc == SCORING_SOFTMAX and wdt in _LOW_PRECISION
        else torch.float32
    )
    gmt = CUDA_GEMM_MIN_TOKENS if gemm_min_tokens is None else gemm_min_tokens
    if (
        _CUDA_EXT is not None
        and weight.is_cuda
        and wdt in _LOW_PRECISION
        and _pro6000_routing(weight.device)
    ):
        native = _CUDA_EXT.Router(
            weight, top_k, sc, bias, n_group, topk_group, renormalize,
            scaling, eps, round_logits, weights_dtype, logits_dtype,
            return_logits, dense, active_flags, compact, gmt,
        )  # fmt: skip
        kwargs = dict(
            scoring=scoring, bias=bias, n_group=n_group,
            topk_group=topk_group, renormalize=renormalize, scaling=scaling,
            eps=eps, round_logits=round_logits, weights_dtype=weights_dtype,
            return_logits=return_logits, dense=dense,
            active_flags=active_flags, compact=compact, gemm_min_tokens=gmt,
        )  # fmt: skip

        def route(hidden_states):
            if _use_prefill_kernel(hidden_states, weight, gemm_min_tokens):
                return tuple(
                    fused_route(hidden_states, weight, top_k, **kwargs)
                )
            res = native(hidden_states)
            # empty: shape / alignment outside the CUDA kernel's envelope
            return (
                res
                if res
                else tuple(fused_route(hidden_states, weight, top_k, **kwargs))
            )

        return route

    def route(hidden_states):
        return tuple(
            fused_route(
                hidden_states, weight, top_k, scoring=scoring, bias=bias,
                n_group=n_group, topk_group=topk_group,
                renormalize=renormalize, scaling=scaling, eps=eps,
                round_logits=round_logits, weights_dtype=weights_dtype,
                return_logits=return_logits, dense=dense,
                active_flags=active_flags, compact=compact,
                gemm_min_tokens=gmt,
            )
        )  # fmt: skip

    return route


def _fused_route_triton(
    x, weight, top_k, sc, bias, n_group, topk_group, renormalize, scaling,
    eps, round_logits, weights_dtype, logits_dtype, return_logits, dense,
    gemm_min_tokens,
):  # fmt: skip
    if x.stride(-1) != 1:
        x = x.contiguous()
    w = weight if weight.stride(-1) == 1 else weight.contiguous()
    T, H = x.shape
    E = w.shape[0]
    if w.shape[1] != H:
        raise ValueError(
            "hidden size mismatch between hidden_states and weight"
        )
    if not 1 <= top_k <= E:
        raise ValueError("top_k must be in [1, E]")
    if n_group > 1 and (E % n_group or topk_group < 1 or topk_group > n_group):
        raise ValueError("invalid n_group / topk_group")
    device = x.device
    low_precision = x.dtype in _LOW_PRECISION

    block_e = max(16, triton.next_power_of_2(E))
    k_pad = triton.next_power_of_2(top_k)
    ids = torch.empty((T, top_k), dtype=torch.int32, device=device)
    tw = torch.empty((T, top_k), dtype=torch.float32, device=device)
    if dense:
        mask = torch.empty((T, E), dtype=torch.bool, device=device)
        dw = torch.empty((T, E), dtype=weights_dtype, device=device)
    else:
        mask = dw = ids  # unused pointers
    logits_out = (
        torch.empty((T, E), dtype=logits_dtype, device=device)
        if return_logits
        else ids
    )
    if T == 0:
        return RouteOutput(
            ids,
            tw,
            mask if dense else None,
            dw if dense else None,
            logits_out if return_logits else None,
            None,
        )
    if bias is not None:
        bias = bias.to(device=device, dtype=torch.float32).contiguous()
    bias_arg = bias if bias is not None else tw

    both_low = low_precision and w.dtype == x.dtype
    precision = "ieee" if both_low else "tf32x3"
    if not gemm_min_tokens:  # None / 0: default
        gemm_min_tokens = PREFILL_GEMM_MIN_TOKENS_PRO6000
    common = dict(
        H=H,
        E=E,
        K=top_k,
        K_PAD=k_pad,
        BLOCK_E=block_e,
        SCORING=sc,
        HAS_BIAS=bias is not None,
        N_GROUP=n_group,
        TOPK_GROUP=topk_group,
        RENORM=renormalize,
        ROUND_LOGITS=bool(round_logits),
        WRITE_LOGITS=return_logits,
        WRITE_DENSE=dense,
    )
    if T >= gemm_min_tokens and both_low:
        # Tensor-core GEMM (cuBLAS) for the logits, then one fused
        # score/top-k launch over the [T, E] logits.
        if logits_dtype == torch.float32:
            logits = torch.mm(x, w.t(), out_dtype=torch.float32)
        else:
            logits = torch.mm(x, w.t())
        block_t = max(1, min(64, 8192 // block_e))
        _fused_route_pro6000_kernel[(triton.cdiv(T, block_t), 1)](
            x, w, bias_arg, logits, tw, tw, ids, tw, mask, dw, logits_out,
            T, x.stride(0), w.stride(0), float(scaling), float(eps),
            BLOCK_T=block_t, BLOCK_H=16, SPLIT_K=1, H_PER_SPLIT=16,
            LOGITS_IN=True, PRECISION=precision, num_warps=4, **common,
        )  # fmt: skip
    else:
        block_t, block_h, split, num_warps, num_stages = _plan_launch_pro6000(
            T, H, E, block_e, device
        )
        h_per_split = triton.cdiv(triton.cdiv(H, split), block_h) * block_h
        split = triton.cdiv(H, h_per_split)
        n_t = triton.cdiv(T, block_t)
        if split > 1:
            partial, counter = _scratch(device).get(split * T * block_e, n_t)
        else:
            partial, counter = tw, ids
        _fused_route_pro6000_kernel[(n_t, split)](
            x, w, bias_arg, tw, partial, counter, ids, tw, mask, dw,
            logits_out, T, x.stride(0), w.stride(0), float(scaling),
            float(eps), BLOCK_T=block_t, BLOCK_H=block_h, SPLIT_K=split,
            H_PER_SPLIT=h_per_split, LOGITS_IN=False, PRECISION=precision,
            num_warps=num_warps, num_stages=num_stages, **common,
        )  # fmt: skip
    return RouteOutput(
        ids,
        tw,
        mask if dense else None,
        dw if dense else None,
        logits_out if return_logits else None,
        None,
    )
