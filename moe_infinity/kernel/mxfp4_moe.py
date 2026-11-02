"""Grouped MXFP4 MoE for GPT-OSS resident experts (device-side dispatch).

Replaces the per-expert Python loop (``token_mask.any()`` host sync, boolean
gather/scatter per expert) with three launches and no host synchronisation:

1. ``_moe_align_kernel``: counting sort of the ``T * top_k`` (token, slot)
   assignments by expert, each expert segment padded to ``BLOCK_M`` rows,
   plus a per-block expert id; the remaining programs zero the output
   accumulator.
2. ``_moe_gemm1_kernel``: per expert-homogeneous row block, gathers the
   token rows, multiplies by the MXFP4 ``gate_up`` weights (``tl.dot_scaled``
   bf16 x e2m1), adds the bias and applies the GPT-OSS clamped SwiGLU to the
   interleaved gate/up columns.
3. ``_moe_gemm2_kernel``: MXFP4 ``down`` projection, bias, routing weight and
   an atomic scatter-add into ``out[token]``. Small batches split K and
   accumulate in an fp32 workspace that the last program converts to bf16.

Launch grids are static upper bounds derived from shapes only; blocks past
the real padded length exit immediately, so nothing is read back to the host
and the whole forward is CUDA-graph capturable.

Weight layout (as loaded by ``_load_resident_gpt_oss``): packed blocks
``[E, N, K // 2]`` uint8 (low nibble = even k), scales ``[E, N, K // 32]``
uint8 (E8M0, bias 127), biases ``[E, N]``, ``gate_up`` rows interleaved
(gate = even rows, up = odd rows).
"""

import functools

import torch
import triton
import triton.language as tl

try:
    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

    _HAS_GDC = True
except ImportError:  # Triton < 3.3: no programmatic dependent launch
    _HAS_GDC = False

    @triton.jit
    def gdc_wait():
        pass

    @triton.jit
    def gdc_launch_dependents():
        pass


@triton.jit
def _moe_align_kernel(
    topk_ids_ptr,
    sorted_ids_ptr,
    block_expert_ptr,
    zero_ptr,
    TK,
    MAX_BLOCKS,
    ZERO_NUMEL,
    BLOCK_M: tl.constexpr,
    E_P2: tl.constexpr,
    CHUNK: tl.constexpr,
    ZERO_BLOCK: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        gdc_wait()
        gdc_launch_dependents()
    pid = tl.program_id(0)
    if pid > 0:
        # Programs 1..P zero the scatter-add target.
        nprog = tl.num_programs(0) - 1
        offs = (pid - 1) * ZERO_BLOCK + tl.arange(0, ZERO_BLOCK)
        zeros = tl.zeros((ZERO_BLOCK,), dtype=zero_ptr.dtype.element_ty)
        for start in range(0, ZERO_NUMEL, nprog * ZERO_BLOCK):
            o = start + offs
            tl.store(zero_ptr + o, zeros, mask=o < ZERO_NUMEL)
        return

    experts = tl.arange(0, E_P2)
    lanes = tl.arange(0, CHUNK)

    # Pass 1: assignments per expert.
    counts = tl.zeros((E_P2,), dtype=tl.int32)
    for start in range(0, TK, CHUNK):
        idx = start + lanes
        ids = tl.load(topk_ids_ptr + idx, mask=idx < TK, other=-1).to(tl.int32)
        counts += tl.sum((ids[:, None] == experts[None, :]).to(tl.int32), 0)

    padded = tl.cdiv(counts, BLOCK_M) * BLOCK_M
    seg_end = tl.cumsum(padded, axis=0)
    total = tl.sum(padded, axis=0)

    # Block -> expert map (-1 past the padded total) and sentinel fill.
    for start in range(0, MAX_BLOCKS, CHUNK):
        b = start + lanes
        row0 = b * BLOCK_M
        e_of_b = tl.sum((seg_end[None, :] <= row0[:, None]).to(tl.int32), 1)
        e_of_b = tl.where(row0 < total, e_of_b, -1)
        tl.store(block_expert_ptr + b, e_of_b, mask=b < MAX_BLOCKS)
    for start in range(0, MAX_BLOCKS * BLOCK_M, CHUNK):
        r = start + lanes
        tl.store(
            sorted_ids_ptr + r,
            tl.zeros((CHUNK,), dtype=tl.int32) + TK,
            mask=r < MAX_BLOCKS * BLOCK_M,
        )
    tl.debug_barrier()

    # Pass 2: stable scatter of assignment ids into their expert segment.
    running = seg_end - padded
    for start in range(0, TK, CHUNK):
        idx = start + lanes
        ids = tl.load(topk_ids_ptr + idx, mask=idx < TK, other=-1).to(tl.int32)
        onehot = (ids[:, None] == experts[None, :]).to(tl.int32)
        excl = tl.cumsum(onehot, axis=0) - onehot + running[None, :]
        pos = tl.sum(onehot * excl, axis=1)
        tl.store(sorted_ids_ptr + pos, idx, mask=idx < TK)
        running += tl.sum(onehot, axis=0)


@triton.jit
def _mxfp4_dot_k(
    acc,
    a_ptrs,
    row_mask,
    w_ptrs,
    s_ptrs,
    n_mask,
    k_lo,
    k_hi,
    BLOCK_K: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    """acc += A[:, k_lo:k_hi] @ dequant(W[:, k_lo:k_hi]).T (k_lo % BLOCK_K == 0).

    ``a_ptrs`` [BM, BLOCK_K] bf16, ``w_ptrs`` [BN, BLOCK_K // 2] packed e2m1,
    ``s_ptrs`` [BN, BLOCK_K // 32] e8m0, all pointing at k = 0.
    """
    offs_k = tl.arange(0, BLOCK_K)
    offs_kp = tl.arange(0, BLOCK_K // 2)
    offs_ks = tl.arange(0, BLOCK_K // 32)
    a_ptrs += k_lo
    w_ptrs += k_lo // 2
    s_ptrs += k_lo // 32
    k_main = k_lo + (k_hi - k_lo) // BLOCK_K * BLOCK_K
    for _ in range(k_lo, k_main, BLOCK_K):
        a = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0)
        if EVEN_N:
            w = tl.load(w_ptrs)
            s = tl.load(s_ptrs)
        else:
            w = tl.load(w_ptrs, mask=n_mask[:, None], other=0)
            s = tl.load(s_ptrs, mask=n_mask[:, None], other=127)
        acc = tl.dot_scaled(
            a, None, "bf16", w.T, s, "e2m1", acc, fast_math=True
        )
        a_ptrs += BLOCK_K
        w_ptrs += BLOCK_K // 2
        s_ptrs += BLOCK_K // 32
    if k_main < k_hi:
        rem = k_hi - k_main
        a = tl.load(
            a_ptrs, mask=row_mask[:, None] & (offs_k < rem)[None, :], other=0.0
        )
        w = tl.load(
            w_ptrs,
            mask=n_mask[:, None] & (offs_kp < rem // 2)[None, :],
            other=0,
        )
        s = tl.load(
            s_ptrs,
            mask=n_mask[:, None] & (offs_ks < rem // 32)[None, :],
            other=127,
        )
        acc = tl.dot_scaled(
            a, None, "bf16", w.T, s, "e2m1", acc, fast_math=True
        )
    return acc


@triton.jit
def _moe_gemm1_kernel(
    x_ptr,
    sorted_ids_ptr,
    block_expert_ptr,
    w_ptr,
    s_ptr,
    b_ptr,
    inter_ptr,
    TK,
    K,
    N_HALF,
    stride_xm,
    stride_we,
    stride_wn,
    stride_se,
    stride_sn,
    stride_be,
    stride_im,
    alpha,
    limit,
    TOP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        gdc_wait()
        gdc_launch_dependents()
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    e32 = tl.load(block_expert_ptr + pid_m)
    aid = tl.load(sorted_ids_ptr + rows)
    valid = aid < TK
    if e32 < 0:
        return
    e = e32.to(tl.int64)
    tok = (aid // TOP_K).to(tl.int64)

    # 2*BLOCK_N interleaved gate/up weight rows -> BLOCK_N output columns.
    offs_wn = pid_n * (2 * BLOCK_N) + tl.arange(0, 2 * BLOCK_N)
    wn_mask = offs_wn < 2 * N_HALF
    wn = offs_wn[:, None].to(tl.int64)
    acc = _mxfp4_dot_k(
        tl.zeros((BLOCK_M, 2 * BLOCK_N), dtype=tl.float32),
        x_ptr + tok[:, None] * stride_xm + tl.arange(0, BLOCK_K)[None, :],
        valid,
        w_ptr
        + e * stride_we
        + wn * stride_wn
        + tl.arange(0, BLOCK_K // 2)[None, :],
        s_ptr
        + e * stride_se
        + wn * stride_sn
        + tl.arange(0, BLOCK_K // 32)[None, :],
        wn_mask,
        0,
        K,
        BLOCK_K,
        EVEN_N,
    )

    bias = tl.load(b_ptr + e * stride_be + offs_wn, mask=wn_mask, other=0.0).to(
        tl.float32
    )
    # Match the reference: GEMM output is rounded to bf16 before SwiGLU.
    gu = (acc + bias[None, :]).to(tl.bfloat16).to(tl.float32)
    gate, up = tl.split(tl.reshape(gu, (BLOCK_M, BLOCK_N, 2)))
    gate = tl.minimum(gate, limit)
    up = tl.minimum(tl.maximum(up, -limit), limit)
    act = (up + 1.0) * (gate * tl.sigmoid(gate * alpha))

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    i_ptrs = (
        inter_ptr + rows[:, None].to(tl.int64) * stride_im + offs_n[None, :]
    )
    tl.store(
        i_ptrs,
        act.to(tl.bfloat16),
        mask=valid[:, None] & (offs_n < N_HALF)[None, :],
    )


@triton.jit
def _moe_gemm2_kernel(
    inter_ptr,
    sorted_ids_ptr,
    block_expert_ptr,
    topk_w_ptr,
    w_ptr,
    s_ptr,
    b_ptr,
    out_ptr,
    ws_ptr,
    counter_ptr,
    TK,
    N,
    K,
    stride_im,
    stride_we,
    stride_wn,
    stride_se,
    stride_sn,
    stride_be,
    TOP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    if USE_PDL:
        gdc_wait()
        gdc_launch_dependents()
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    e32 = tl.load(block_expert_ptr + pid_m)
    aid = tl.load(sorted_ids_ptr + rows)
    valid = aid < TK
    if e32 >= 0:
        e = e32.to(tl.int64)
        tok = (aid // TOP_K).to(tl.int64)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        wn = offs_n[:, None].to(tl.int64)
        k_per = tl.cdiv(tl.cdiv(K, BLOCK_K), SPLIT_K) * BLOCK_K
        k_lo = pid_k * k_per
        acc = _mxfp4_dot_k(
            tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32),
            inter_ptr
            + rows[:, None].to(tl.int64) * stride_im
            + tl.arange(0, BLOCK_K)[None, :],
            valid,
            w_ptr
            + e * stride_we
            + wn * stride_wn
            + tl.arange(0, BLOCK_K // 2)[None, :],
            s_ptr
            + e * stride_se
            + wn * stride_sn
            + tl.arange(0, BLOCK_K // 32)[None, :],
            n_mask,
            k_lo,
            tl.minimum(k_lo + k_per, K),
            BLOCK_K,
            EVEN_N,
        )

        rw = tl.load(topk_w_ptr + aid, mask=valid, other=0.0).to(tl.float32)
        o_off = tok[:, None] * N + offs_n[None, :]
        o_mask = valid[:, None] & n_mask[None, :]
        if pid_k == 0:
            bias = tl.load(
                b_ptr + e * stride_be + offs_n, mask=n_mask, other=0.0
            )
            acc += bias.to(tl.float32)[None, :]
        if SPLIT_K == 1:
            # Same rounding as the reference loop: bf16(w * bf16(y)).
            y = acc.to(tl.bfloat16).to(tl.float32)
            contrib = (rw[:, None] * y).to(tl.bfloat16)
            tl.atomic_add(out_ptr + o_off, contrib, mask=o_mask, sem="relaxed")
        else:
            tl.atomic_add(
                ws_ptr + o_off, rw[:, None] * acc, mask=o_mask, sem="relaxed"
            )

    if SPLIT_K > 1:
        # The last program to finish converts the fp32 workspace to bf16.
        tl.debug_barrier()
        total = tl.num_programs(0) * tl.num_programs(1) * SPLIT_K
        done = tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="gpu")
        if done == total - 1:
            n = (TK // TOP_K) * N
            offs = tl.arange(0, 1024)
            for start in range(0, n, 1024):
                o = start + offs
                v = tl.load(
                    ws_ptr + o, mask=o < n, other=0.0, cache_modifier=".cg"
                )
                tl.store(out_ptr + o, v.to(tl.bfloat16), mask=o < n)


@functools.lru_cache(maxsize=None)
def _use_pdl(device_index: int) -> bool:
    # Programmatic dependent launch (griddepcontrol) needs sm_90+: each
    # kernel's prologue/launch overlaps the previous kernel's tail.
    return _HAS_GDC and torch.cuda.get_device_capability(device_index)[0] >= 9


def _block_m(tk: int, num_experts: int) -> int:
    rows_per_expert = tk / max(1, min(num_experts, tk))
    if rows_per_expert <= 16:
        return 16
    if rows_per_expert <= 32:
        return 32
    return 64


# Small batches split K of the down projection (fp32 workspace) up to this
# many routed (token, slot) assignments.
SPLIT_MAX_TK = 8


def _configs(tk: int, block_m: int):
    """((BN, BK, warps, stages) gemm1, (BN, BK, warps, stages, split) gemm2).

    Tuned on RTX PRO 6000 Blackwell (sm_120) with GPT-OSS shapes
    (H = I = 2880) under CUDA-graph replay with a cold L2.
    """
    if block_m <= 16:
        g1 = (32, 256, 4, 3)
        g2 = (64, 256, 4, 2, 3 if tk <= SPLIT_MAX_TK else 1)
    elif block_m <= 32:
        g1 = (64, 128, 4, 2)
        g2 = (128, 128, 4, 2, 1)
    else:
        g1 = (64, 128, 8, 3)
        g2 = (128, 128, 8, 3, 1)
    return g1, g2


def moe_align_mxfp4(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_m: int,
    zero: torch.Tensor | None = None,
):
    """Sort (token, slot) assignments by expert without a host sync.

    Returns ``(sorted_ids, block_expert)``: ``sorted_ids[max_blocks * block_m]``
    holds flat assignment indices ``t * top_k + j`` grouped by expert (each
    group padded to ``block_m`` with the sentinel ``T * top_k``);
    ``block_expert[max_blocks]`` gives the expert of each row block (-1 when
    unused). If ``zero`` is given it is zero-filled in the same launch.
    """
    tk = topk_ids.numel()
    max_blocks = triton.cdiv(tk, block_m) + min(num_experts, tk)
    dev = topk_ids.device
    sorted_ids = torch.empty(
        max_blocks * block_m, dtype=torch.int32, device=dev
    )
    block_expert = torch.empty(max_blocks, dtype=torch.int32, device=dev)
    zero_block = 1024
    n_zero = 0
    if zero is not None:
        n_zero = min(256, triton.cdiv(zero.numel(), zero_block))
    pdl = _use_pdl(torch.device(dev).index or 0)
    e_p2 = max(16, triton.next_power_of_2(num_experts))
    chunk = min(max(16, triton.next_power_of_2(max(tk, max_blocks))), 128)
    if e_p2 > 128:
        chunk = min(chunk, 32)
    _moe_align_kernel[(1 + n_zero,)](
        topk_ids.reshape(-1),
        sorted_ids,
        block_expert,
        zero if zero is not None else sorted_ids,
        tk,
        max_blocks,
        zero.numel() if zero is not None else 0,
        BLOCK_M=block_m,
        E_P2=e_p2,
        CHUNK=chunk,
        ZERO_BLOCK=zero_block,
        USE_PDL=pdl,
        num_warps=4,
        launch_pdl=pdl,
    )
    return sorted_ids, block_expert


def fused_moe_mxfp4(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_blocks: torch.Tensor,
    gate_up_scales: torch.Tensor,
    gate_up_bias: torch.Tensor,
    down_blocks: torch.Tensor,
    down_scales: torch.Tensor,
    down_bias: torch.Tensor,
    alpha: float = 1.702,
    limit: float = 7.0,
    configs=None,
) -> torch.Tensor:
    """GPT-OSS MoE forward over resident MXFP4 experts in three launches.

    Args:
        x:              ``[T, H]`` bf16 token activations
        topk_ids:       ``[T, top_k]`` selected expert ids (any int dtype)
        topk_weights:   ``[T, top_k]`` routing weights (any float dtype)
        gate_up_blocks: ``[E, 2I, H // 2]`` uint8 (``[E, 2I, H//32, 16]`` ok)
        gate_up_scales: ``[E, 2I, H // 32]`` uint8
        gate_up_bias:   ``[E, 2I]``
        down_blocks:    ``[E, H, I // 2]`` uint8 (``[E, H, I//32, 16]`` ok)
        down_scales:    ``[E, H, I // 32]`` uint8
        down_bias:      ``[E, H]``
        configs:        optional override of ``_configs`` (tuning)

    Returns:
        ``[T, H]`` bf16, ``sum_j w[t, j] * expert_{ids[t, j]}(x[t])``.
    """
    assert x.dim() == 2 and x.dtype == torch.bfloat16
    T, H = x.shape
    E = gate_up_blocks.shape[0]
    top_k = topk_ids.shape[-1]
    gate_up_blocks = gate_up_blocks.reshape(E, gate_up_blocks.shape[1], -1)
    down_blocks = down_blocks.reshape(E, down_blocks.shape[1], -1)
    inter = gate_up_blocks.shape[1] // 2
    assert gate_up_blocks.shape[2] * 2 == H, "gate_up K mismatch"
    assert down_blocks.shape[1] == H and down_blocks.shape[2] * 2 == inter
    assert H % 32 == 0 and inter % 32 == 0
    for t in (gate_up_blocks, gate_up_scales, down_blocks, down_scales):
        assert t.dtype == torch.uint8 and t.stride(-1) == 1
    x = x.contiguous()
    topk_ids = topk_ids.reshape(-1).contiguous()
    topk_weights = topk_weights.reshape(-1).contiguous()
    gate_up_bias = gate_up_bias.contiguous()
    down_bias = down_bias.contiguous()

    tk = T * top_k
    out = torch.empty_like(x)
    if tk == 0:
        return out.zero_()
    block_m = _block_m(tk, E)
    (bn1, bk1, w1, st1), (bn2, bk2, w2, st2, split) = configs or _configs(
        tk, block_m
    )
    if split > 1:
        # fp32 accumulator + int32 arrival counter (zero bits == 0).
        ws = torch.empty(T * H + 1, dtype=torch.float32, device=x.device)
        counter = ws[T * H :].view(torch.int32)
        zero = ws
    else:
        ws = counter = zero = out
    sorted_ids, block_expert = moe_align_mxfp4(topk_ids, E, block_m, zero)
    max_blocks = block_expert.numel()
    pdl = _use_pdl(x.device.index or 0)
    inter_buf = torch.empty(
        sorted_ids.numel(), inter, dtype=torch.bfloat16, device=x.device
    )

    _moe_gemm1_kernel[(max_blocks, triton.cdiv(inter, bn1))](
        x,
        sorted_ids,
        block_expert,
        gate_up_blocks,
        gate_up_scales,
        gate_up_bias,
        inter_buf,
        tk,
        H,
        inter,
        x.stride(0),
        gate_up_blocks.stride(0),
        gate_up_blocks.stride(1),
        gate_up_scales.stride(0),
        gate_up_scales.stride(1),
        gate_up_bias.stride(0),
        inter_buf.stride(0),
        float(alpha),
        float(limit),
        TOP_K=top_k,
        BLOCK_M=block_m,
        BLOCK_N=bn1,
        BLOCK_K=bk1,
        EVEN_N=inter % bn1 == 0,
        USE_PDL=pdl,
        num_warps=w1,
        num_stages=st1,
        launch_pdl=pdl,
    )

    _moe_gemm2_kernel[(max_blocks, triton.cdiv(H, bn2), split)](
        inter_buf,
        sorted_ids,
        block_expert,
        topk_weights,
        down_blocks,
        down_scales,
        down_bias,
        out,
        ws,
        counter,
        tk,
        H,
        inter,
        inter_buf.stride(0),
        down_blocks.stride(0),
        down_blocks.stride(1),
        down_scales.stride(0),
        down_scales.stride(1),
        down_bias.stride(0),
        TOP_K=top_k,
        BLOCK_M=block_m,
        BLOCK_N=bn2,
        BLOCK_K=bk2,
        SPLIT_K=split,
        EVEN_N=H % bn2 == 0,
        USE_PDL=pdl,
        num_warps=w2,
        num_stages=st2,
        launch_pdl=pdl,
    )
    return out
