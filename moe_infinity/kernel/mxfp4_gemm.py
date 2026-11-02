"""Fused MXFP4 dequantization + GEMM Triton kernels.

Ported from BatchGen (EfficientMoE/BatchGen) for MoE-Infinity expert offloading,
reworked for sm_120 (EfficientMoE/MoE-Infinity#312).

MXFP4 Format (E2M1):
    - Block size: 32 FP4 values share one uint8 scale
    - Packing: 2 FP4 values per uint8 (low nibble = even idx, high nibble = odd idx)
    - Scale: uint8, exponent = scale_byte - 127
    - Values: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] (+ negatives)
    - Dequant: fp4_value * 2^exponent

Two kernels, picked per call from a static table keyed on a bucketed M (no
autotuning on the request path; ``warmup_mxfp4_gemm`` pre-compiles every
bucket):

    * ``_mxfp4_gemv_kernel`` (M == 1): register-only weight streaming. Each
      thread owns 4-byte slices of 16-byte scale groups, decodes E2M1 with the
      native ``cvt.rn.f16x2.e2m1x2`` into fp32 and FMAs against the bf16
      activation pairs read in the same per-thread layout.
    * ``_mxfp4_mma_kernel`` (M > 1): tensor-core GEMM with BLOCK_K >= 64 (one
      scale per 32 broadcast in registers), bit-trick E2M1 -> BF16 decode and an
      N-fastest program order so every M band of activations is read from DRAM
      once while the (L2-resident) weights are swept.

Small M uses split-K when there are too few output tiles to fill the GPU.
Partial sums go to an fp32 scratch buffer and the last CTA of each output tile
(per-tile arrival counter) reduces them in a fixed order, so results are
deterministic and no zero-fill launch is needed.
"""

from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _e2m1_to_f32(w):
    """4 packed bytes -> (lo, hi) nibbles as exact fp32 E2M1 values.

    Uses the sm_100a/sm_120a ``cvt.rn.f16x2.e2m1x2`` conversion (one
    instruction per byte) and widens the halves to fp32.
    """
    return tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b8 b0, b1, b2, b3;
        .reg .b16 l0, l1, l2, l3, h0, h1, h2, h3;
        .reg .b32 p0, p1, p2, p3;
        mov.b32 {b0, b1, b2, b3}, $8;
        cvt.rn.f16x2.e2m1x2 p0, b0;
        cvt.rn.f16x2.e2m1x2 p1, b1;
        cvt.rn.f16x2.e2m1x2 p2, b2;
        cvt.rn.f16x2.e2m1x2 p3, b3;
        mov.b32 {l0, h0}, p0;
        mov.b32 {l1, h1}, p1;
        mov.b32 {l2, h2}, p2;
        mov.b32 {l3, h3}, p3;
        cvt.f32.f16 $0, l0;
        cvt.f32.f16 $1, l1;
        cvt.f32.f16 $2, l2;
        cvt.f32.f16 $3, l3;
        cvt.f32.f16 $4, h0;
        cvt.f32.f16 $5, h1;
        cvt.f32.f16 $6, h2;
        cvt.f32.f16 $7, h3;
        }
        """,
        constraints="=r,=r,=r,=r,=r,=r,=r,=r,r",
        args=[w],
        dtype=(tl.float32, tl.float32),
        is_pure=True,
        pack=4,
    )


@triton.jit
def _e2m1_to_bf16(w, s):
    """4 packed bytes + their 4 e8m0 scales -> (lo, hi) scaled BF16 values.

    Nibble bits go to BF16 bits 6-8 / 15 (value * 2^-126), then two packed
    BF16 multiplies apply 2^126 and 2^(s-127). Both are exact (powers of two)
    unless the result leaves the BF16 range, matching the reference.
    """
    return tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b32 t0, t1, s0, s1, a, b, c;
        prmt.b32 t0, $4, 0, 0x4140;
        prmt.b32 t1, $4, 0, 0x4342;
        prmt.b32 s0, $5, 0, 0x4140;
        prmt.b32 s1, $5, 0, 0x4342;
        shl.b32 s0, s0, 7;
        shl.b32 s1, s1, 7;
        mov.b32 c, 0x7E807E80;
        shl.b32 a, t0, 6; shl.b32 b, t0, 12;
        and.b32 a, a, 0x01C001C0; and.b32 b, b, 0x80008000; or.b32 a, a, b;
        mul.rn.bf16x2 a, a, c; mul.rn.bf16x2 $0, a, s0;
        shl.b32 a, t1, 6; shl.b32 b, t1, 12;
        and.b32 a, a, 0x01C001C0; and.b32 b, b, 0x80008000; or.b32 a, a, b;
        mul.rn.bf16x2 a, a, c; mul.rn.bf16x2 $1, a, s1;
        shl.b32 a, t0, 2; shl.b32 b, t0, 8;
        and.b32 a, a, 0x01C001C0; and.b32 b, b, 0x80008000; or.b32 a, a, b;
        mul.rn.bf16x2 a, a, c; mul.rn.bf16x2 $2, a, s0;
        shl.b32 a, t1, 2; shl.b32 b, t1, 8;
        and.b32 a, a, 0x01C001C0; and.b32 b, b, 0x80008000; or.b32 a, a, b;
        mul.rn.bf16x2 a, a, c; mul.rn.bf16x2 $3, a, s1;
        }
        """,
        constraints="=r,=r,=r,=r,r,r",
        args=[w, s],
        dtype=(tl.bfloat16, tl.bfloat16),
        is_pure=True,
        pack=4,
    )


@triton.jit
def _epilogue(
    acc,
    offs_m,
    offs_n,
    M,
    N,
    bias_ptr,
    out_ptr,
    stride_om,
    part_ptr,
    cnt_ptr,
    tile_id,
    pid_k,
    HAS_BIAS: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """Store ``acc + bias`` as BF16; with split-K the last CTA of the tile
    reduces the fp32 partials of all K slices in a fixed order."""
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if SPLIT_K > 1:
        part = part_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.store(part + pid_k * M * N, acc, mask=mask)
        tl.debug_barrier()
        arrived = tl.atomic_add(cnt_ptr + tile_id, 1, sem="acq_rel")
        last = arrived == SPLIT_K - 1
        mask = mask & last
        acc = tl.load(part, mask=mask, other=0.0, cache_modifier=".cg")
        for k in tl.static_range(1, SPLIT_K):
            acc += tl.load(
                part + k * M * N, mask=mask, other=0.0, cache_modifier=".cg"
            )
        tl.atomic_xchg(cnt_ptr + tile_id, 0, mask=last)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :].to(tl.float32)
    out = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(out, acc.to(tl.bfloat16), mask=mask)


@triton.jit(do_not_specialize=["M"])
def _mxfp4_gemv_kernel(
    x_ptr,  # [1, K//2] int32 view of the bf16 activations (pairs)
    w_ptr,  # [N, K//2] uint8
    s_ptr,  # [N, K//32] uint8
    bias_ptr,
    out_ptr,
    part_ptr,
    cnt_ptr,
    M,
    N,
    K,
    stride_wn,
    stride_sn,
    stride_om,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_G: tl.constexpr,  # scale groups (32 values / 16 bytes) per step
    SPLIT_K: tl.constexpr,
    G_PER_SPLIT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    R: tl.constexpr = BLOCK_N * BLOCK_G
    r = tl.arange(0, R)
    rn = (pid_n * BLOCK_N + r // BLOCK_G) % N
    rg = r % BLOCK_G
    num_g = K // 32
    g_lo = pid_k * G_PER_SPLIT
    g_hi = tl.minimum(g_lo + G_PER_SPLIT, num_g)
    j = tl.arange(0, 16)
    # Contiguity 4 keeps the uint8 load in the same [1, 4]-per-thread layout
    # as the int32 activation pairs, so no layout shuffles are needed.
    j4 = (j // 4) * 4 + j % 4
    acc = tl.zeros((R, 16), dtype=tl.float32)
    for g0 in range(g_lo, g_hi, BLOCK_G):
        g = g0 + rg
        gm = g < g_hi
        w = tl.load(
            w_ptr + (rn * stride_wn + g * 16)[:, None] + j4[None, :],
            mask=gm[:, None],
            other=0,
        )
        s = tl.load(s_ptr + rn * stride_sn + g, mask=gm, other=0)
        scale = (s.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        lo, hi = _e2m1_to_f32(w)
        xp = tl.load(
            x_ptr + (g * 16)[:, None] + j[None, :], mask=gm[:, None], other=0
        )
        x_even = (xp << 16).to(tl.float32, bitcast=True)
        x_odd = (xp & -65536).to(tl.float32, bitcast=True)
        acc += (lo * x_even + hi * x_odd) * scale[:, None]
    acc_n = tl.sum(tl.sum(acc, axis=1).reshape(BLOCK_N, BLOCK_G), axis=1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, 1)
    _epilogue(
        acc_n[None, :],
        offs_m,
        offs_n,
        M,
        N,
        bias_ptr,
        out_ptr,
        stride_om,
        part_ptr,
        cnt_ptr,
        pid_n,
        pid_k,
        HAS_BIAS,
        SPLIT_K,
    )


@triton.jit(do_not_specialize=["M"])
def _mxfp4_mma_kernel(
    x_ptr,  # [M, K]    bf16
    w_ptr,  # [N, K//2] uint8
    s_ptr,  # [N, K//32] uint8
    bias_ptr,
    out_ptr,  # [M, N]  bf16
    part_ptr,
    cnt_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    stride_sn,
    stride_om,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    K_PER_SPLIT: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_n = tl.cdiv(N, BLOCK_N)
    # N-fastest: co-resident CTAs share one M band of activations.
    pid_m = pid // num_n
    pid_n = pid % num_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = offs_m % M
    rn = offs_n % N
    BKH: tl.constexpr = BLOCK_K // 2
    BKS: tl.constexpr = BLOCK_K // 32
    k_lo = pid_k * K_PER_SPLIT
    k_hi = tl.minimum(k_lo + K_PER_SPLIT, K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(k_lo, k_hi, BLOCK_K):
        kh = k0 // 2 + tl.arange(0, BKH)
        ks = k0 // 32 + tl.arange(0, BKS)
        kk = k0 + tl.arange(0, BLOCK_K)
        w = tl.load(
            w_ptr + rn[:, None] * stride_wn + kh[None, :],
            mask=kh[None, :] < k_hi // 2,
            other=0,
        )
        s = tl.load(
            s_ptr + rn[:, None] * stride_sn + ks[None, :],
            mask=ks[None, :] < k_hi // 32,
            other=0,
        )
        s = tl.broadcast_to(s[:, :, None], (BLOCK_N, BKS, 16)).reshape(
            BLOCK_N, BKH
        )
        x = tl.load(
            x_ptr + rm[:, None] * stride_xm + kk[None, :],
            mask=kk[None, :] < k_hi,
            other=0.0,
        )
        lo, hi = _e2m1_to_bf16(w, s)
        wb = tl.join(lo, hi).reshape(BLOCK_N, BLOCK_K)
        acc = tl.dot(x, tl.trans(wb), acc)
    _epilogue(
        acc,
        offs_m,
        offs_n,
        M,
        N,
        bias_ptr,
        out_ptr,
        stride_om,
        part_ptr,
        cnt_ptr,
        pid,
        pid_k,
        HAS_BIAS,
        SPLIT_K,
    )


# ---------------------------------------------------------------------------
# Launch configuration (static, tuned on RTX PRO 6000 Blackwell / sm_120)
# ---------------------------------------------------------------------------


def _m_bucket(M: int) -> int:
    return triton.next_power_of_2(min(M, 4096))


def _num_sms(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _split_for(tiles: int, k_steps: int, target_ctas: int) -> int:
    """K slices so that ``tiles * split`` approaches ``target_ctas`` while
    every slice keeps at least two K steps."""
    if tiles >= target_ctas or k_steps < 4:
        return 1
    return max(1, min(round(target_ctas / tiles), k_steps // 2, 16))


def _config(bucket: int, N: int, K: int, num_sms: int) -> dict:
    """Launch config for an M bucket. Tuned on GPT-OSS shapes (N in {2880,
    5760}, K = 2880) on a 188-SM RTX PRO 6000 Blackwell; the rules scale the
    split-K factor with the SM count for other GPUs / shapes."""
    if bucket == 1:
        # Weight streaming: many small CTAs, one K pass each when it fits.
        many = triton.cdiv(N, 2) >= 2048
        cfg = dict(
            kind="gemv",
            BLOCK_N=2,
            BLOCK_G=32 if many else 128,
            num_warps=2 if many else 4,
        )
        tiles = triton.cdiv(N, cfg["BLOCK_N"])
        if tiles < 2 * num_sms:
            cfg.update(BLOCK_G=32, num_warps=2)
        k_steps = triton.cdiv(K // 32, cfg["BLOCK_G"])
        cfg["SPLIT_K"] = (
            1
            if tiles >= 2 * num_sms
            else max(1, min(triton.cdiv(4 * num_sms, tiles), k_steps))
        )
        return cfg
    wide_n = N >= 4096
    if bucket <= 64:
        cfg = dict(
            BLOCK_M=max(16, bucket),
            BLOCK_N=64 if wide_n else 32,
            BLOCK_K=256 if bucket <= 32 else 128,
            num_warps=4,
            num_stages=3,
        )
        target = (2 if wide_n or bucket > 16 else 3) * num_sms
    elif bucket <= 256:
        cfg = dict(
            BLOCK_M=64,
            BLOCK_N=128 if wide_n and bucket == 256 else 64,
            BLOCK_K=128,
            num_warps=4,
            num_stages=3,
        )
        target = 2 * num_sms
    else:
        cfg = dict(
            BLOCK_M=128 if bucket >= 4096 and not wide_n else 64,
            BLOCK_N=128 if wide_n else 64,
            BLOCK_K=64,
            num_warps=4,
            num_stages=3,
        )
        target = num_sms
    cfg["kind"] = "mma"
    tiles = triton.cdiv(bucket, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"])
    cfg["SPLIT_K"] = _split_for(tiles, triton.cdiv(K, cfg["BLOCK_K"]), target)
    return cfg


_CONFIG_CACHE: Dict[Tuple[int, int, int, int], dict] = {}
# Per (device, stream) split-K arrival counters; the kernels reset them to 0.
_COUNTERS: Dict[Tuple[int, int], torch.Tensor] = {}
_WARM: set = set()


def _get_config(M: int, N: int, K: int, device: torch.device) -> dict:
    key = (_m_bucket(M), N, K, device.index or 0)
    cfg = _CONFIG_CACHE.get(key)
    if cfg is None:
        cfg = _config(key[0], N, K, _num_sms(device))
        _CONFIG_CACHE[key] = cfg
    return cfg


def _counters(device: torch.device, n: int) -> torch.Tensor:
    key = (device.index or 0, torch.cuda.current_stream(device).cuda_stream)
    buf = _COUNTERS.get(key)
    if buf is None or buf.numel() < n:
        size = max(n, 4096 if buf is None else 2 * buf.numel())
        buf = torch.zeros(size, dtype=torch.int32, device=device)
        _COUNTERS[key] = buf
    return buf


def _launch(
    x_2d: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scales: torch.Tensor,
    bias: Optional[torch.Tensor],
    output: torch.Tensor,
    cfg: dict,
) -> None:
    M, K = x_2d.shape
    N = weight_packed.shape[0]
    split = cfg["SPLIT_K"]
    has_bias = bias is not None
    if cfg["kind"] == "gemv":
        tiles = triton.cdiv(N, cfg["BLOCK_N"])
    else:
        tiles = triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"])
    if split > 1:
        part = torch.empty(
            (split, M, N), dtype=torch.float32, device=x_2d.device
        )
        cnt = _counters(x_2d.device, tiles)
    else:
        part = cnt = output
    bias_arg = bias if has_bias else output
    if cfg["kind"] == "gemv":
        num_g = K // 32
        g_per_split = (
            triton.cdiv(triton.cdiv(num_g, cfg["BLOCK_G"]), split)
            * cfg["BLOCK_G"]
        )
        _mxfp4_gemv_kernel[(tiles, split)](
            x_2d.view(torch.int32),
            weight_packed,
            weight_scales,
            bias_arg,
            output,
            part,
            cnt,
            M,
            N,
            K,
            weight_packed.stride(0),
            weight_scales.stride(0),
            output.stride(0),
            HAS_BIAS=has_bias,
            BLOCK_N=cfg["BLOCK_N"],
            BLOCK_G=cfg["BLOCK_G"],
            SPLIT_K=split,
            G_PER_SPLIT=g_per_split,
            num_warps=cfg["num_warps"],
            num_stages=1,
        )
    else:
        k_per_split = (
            triton.cdiv(triton.cdiv(K, cfg["BLOCK_K"]), split) * cfg["BLOCK_K"]
        )
        _mxfp4_mma_kernel[(tiles, split)](
            x_2d,
            weight_packed,
            weight_scales,
            bias_arg,
            output,
            part,
            cnt,
            M,
            N,
            K,
            x_2d.stride(0),
            weight_packed.stride(0),
            weight_scales.stride(0),
            output.stride(0),
            HAS_BIAS=has_bias,
            BLOCK_M=cfg["BLOCK_M"],
            BLOCK_N=cfg["BLOCK_N"],
            BLOCK_K=cfg["BLOCK_K"],
            SPLIT_K=split,
            K_PER_SPLIT=k_per_split,
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )


def warmup_mxfp4_gemm(
    N: int,
    K: int,
    device: Optional[torch.device] = None,
    has_bias: bool = True,
) -> None:
    """Compile every M bucket for an ``[N, K]`` MXFP4 weight.

    Call at model load; afterwards a never-seen token count only selects an
    already compiled kernel. ``fused_mxfp4_gemm`` also runs this on the first
    call for each ``(N, K, device, has_bias)``.
    """
    device = torch.device(device or "cuda")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = (N, K, device.index, has_bias)
    if key in _WARM:
        return
    _WARM.add(key)
    w = torch.zeros((N, K // 2), dtype=torch.uint8, device=device)
    s = torch.full((N, K // 32), 127, dtype=torch.uint8, device=device)
    b = (
        torch.zeros(N, dtype=torch.bfloat16, device=device)
        if has_bias
        else None
    )
    seen = set()
    for bucket in (1 << i for i in range(13)):
        cfg = _get_config(bucket, N, K, device)
        sig = tuple(sorted(cfg.items()))
        if sig in seen:
            continue
        seen.add(sig)
        x = torch.zeros((bucket, K), dtype=torch.bfloat16, device=device)
        out = torch.empty((bucket, N), dtype=torch.bfloat16, device=device)
        _launch(x, w, s, b, out, cfg)


def fused_mxfp4_gemm(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scales: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused MXFP4 dequant + GEMM for a single linear layer.

    Computes ``out = x @ dequant(weight).T + bias`` without materialising
    the full BF16 weight matrix.

    Args:
        x:              Input activations  ``[*, K]``  bf16
        weight_packed:  Packed FP4 weights ``[N, K//2]`` uint8
        weight_scales:  Block scales       ``[N, K//32]`` uint8
        bias:           Optional bias      ``[N]``      bf16

    Returns:
        Output ``[*, N]`` bf16
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    M, K = x_2d.shape

    if weight_packed.dim() == 3:
        weight_packed = weight_packed.reshape(weight_packed.shape[0], -1)

    weight_packed = weight_packed.contiguous()
    weight_scales = weight_scales.contiguous()

    N = weight_packed.shape[0]
    assert (
        K == weight_packed.shape[1] * 2
    ), f"K mismatch: activations K={K}, packed weight implies K={weight_packed.shape[1] * 2}"
    assert K % 32 == 0, f"K={K} must be a multiple of the MXFP4 block (32)"
    assert (
        x_2d.dtype == torch.bfloat16
    ), f"Activations must be bf16, got {x_2d.dtype}"
    assert weight_packed.dtype == torch.uint8
    assert weight_scales.dtype == torch.uint8
    if bias is not None:
        bias = bias.contiguous()

    if x_2d.data_ptr() % 16:
        x_2d = x_2d.clone()  # the M == 1 path reads bf16 pairs as int32
    output = torch.empty((M, N), dtype=torch.bfloat16, device=x_2d.device)
    if M == 0:
        return output.reshape(*orig_shape[:-1], N)

    warmup_mxfp4_gemm(N, K, x_2d.device, bias is not None)
    cfg = _get_config(M, N, K, x_2d.device)
    _launch(x_2d, weight_packed, weight_scales, bias, output, cfg)

    return output.reshape(*orig_shape[:-1], N)


_FP4_TABLE = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def mxfp4_dequantize(
    packed: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    block_size: int = 0,
) -> torch.Tensor:
    """Reference MXFP4 dequantisation (pure PyTorch, any device).

    Args:
        packed: ``[*, K//2]`` uint8
        scales: ``[*, K//BS]`` uint8 where BS = block_size
        dtype:  output dtype
        block_size: FP4 values per scale (0 = auto-detect from shapes)

    Returns:
        ``[*, K]`` dequantised tensor
    """
    table = _FP4_TABLE.to(packed.device)

    idx_lo = (packed & 0x0F).long()
    idx_hi = (packed >> 4).long()

    val_lo = table[idx_lo]
    val_hi = table[idx_hi]

    out_shape = packed.shape[:-1] + (packed.shape[-1] * 2,)
    unpacked = torch.empty(out_shape, dtype=torch.float32, device=packed.device)
    unpacked[..., 0::2] = val_lo
    unpacked[..., 1::2] = val_hi

    exponents = scales.to(torch.int32) - 127
    exponents = exponents.clamp(-126, 127)

    n_blocks = scales.shape[-1]
    if block_size <= 0:
        block_size = unpacked.shape[-1] // n_blocks
    expanded = (
        exponents.unsqueeze(-1)
        .expand(*exponents.shape, block_size)
        .reshape(*exponents.shape[:-1], n_blocks * block_size)
    )
    if expanded.shape[-1] > unpacked.shape[-1]:
        expanded = expanded[..., : unpacked.shape[-1]]

    return torch.ldexp(unpacked, expanded).to(dtype)
