"""Fused MXFP4 dequantization + GEMM Triton kernel.

Ported from BatchGen (EfficientMoE/BatchGen) for MoE-Infinity expert offloading.

MXFP4 Format (E2M1):
    - Block size: 32 FP4 values share one uint8 scale
    - Packing: 2 FP4 values per uint8 (low nibble = even idx, high nibble = odd idx)
    - Scale: uint8, exponent = scale_byte - 127
    - Values: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] (+ negatives)
    - Dequant: fp4_value * 2^exponent
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _fp4_lookup(idx):
    """Decode a 4-bit index into its FP4 E2M1 float value.

    Table (sign in bit 3, magnitude in bits 0-2):
        0→0.0  1→0.5  2→1.0  3→1.5  4→2.0  5→3.0  6→4.0  7→6.0
        8→-0.0 9→-0.5 10→-1.0 11→-1.5 12→-2.0 13→-3.0 14→-4.0 15→-6.0
    """
    sign = tl.where(idx >= 8, -1.0, 1.0)
    mag_idx = idx & 0x07

    mag = tl.where(mag_idx == 0, 0.0, 0.0)
    mag = tl.where(mag_idx == 1, 0.5, mag)
    mag = tl.where(mag_idx == 2, 1.0, mag)
    mag = tl.where(mag_idx == 3, 1.5, mag)
    mag = tl.where(mag_idx == 4, 2.0, mag)
    mag = tl.where(mag_idx == 5, 3.0, mag)
    mag = tl.where(mag_idx == 6, 4.0, mag)
    mag = tl.where(mag_idx == 7, 6.0, mag)

    return (sign * mag).to(tl.float32)


@triton.jit
def _ldexp(mantissa, exponent):
    """Compute mantissa * 2^exponent via IEEE-754 bit manipulation."""
    exp_clamped = tl.minimum(tl.maximum(exponent, -126), 127)
    exp_bits = (exp_clamped + 127).to(tl.int32) << 23
    power_of_2 = exp_bits.to(tl.float32, bitcast=True)
    return mantissa * power_of_2


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32},
            num_stages=3,
            num_warps=4,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _fused_mxfp4_gemm_kernel(
    # Pointers
    lhs_ptr,  # [M, K]   bf16  — input activations
    rhs_packed_ptr,  # [N, K//2] uint8 — packed FP4 weights
    rhs_scales_ptr,  # [N, K//32] uint8 — block scales
    bias_ptr,  # [N]       bf16  — optional bias
    out_ptr,  # [M, N]    bf16  — output
    # Dimensions
    M,
    N,
    K,
    # Strides
    stride_lhs_m,
    stride_lhs_k,
    stride_rhs_n,
    stride_rhs_k,
    stride_scales_n,
    stride_scales_k,
    stride_out_m,
    stride_out_n,
    # Flags
    HAS_BIAS: tl.constexpr,
    # Tile sizes (BLOCK_K=32 matches MXFP4 scale granularity)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused on-the-fly MXFP4 dequant + matmul.

    Computes: out[m, n] = sum_k( lhs[m, k] * dequant(rhs)[n, k] ) + bias[n]

    BLOCK_K is fixed at 32 so each K-tile uses exactly one scale per N row.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K_packed = K // 2
    BLOCK_K_HALF: tl.constexpr = BLOCK_K // 2

    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k_block * BLOCK_K

        scale_idx = k_start // 32
        scale_ptrs = (
            rhs_scales_ptr
            + offs_n * stride_scales_n
            + scale_idx * stride_scales_k
        )
        scales = tl.load(scale_ptrs, mask=offs_n < N, other=127)
        exponents = scales.to(tl.int32) - 127

        offs_k_packed = k_start // 2 + tl.arange(0, BLOCK_K_HALF)
        rhs_ptrs = (
            rhs_packed_ptr
            + offs_n[:, None] * stride_rhs_n
            + offs_k_packed[None, :] * stride_rhs_k
        )
        rhs_mask = (offs_n[:, None] < N) & (offs_k_packed[None, :] < K_packed)
        rhs_packed = tl.load(rhs_ptrs, mask=rhs_mask, other=0)

        idx_lo = (rhs_packed & 0x0F).to(tl.int32)
        idx_hi = ((rhs_packed >> 4) & 0x0F).to(tl.int32)

        val_lo = _fp4_lookup(idx_lo)
        val_hi = _fp4_lookup(idx_hi)

        exp_bc = exponents[:, None] + tl.zeros(
            (1, BLOCK_K_HALF), dtype=tl.int32
        )
        val_lo = _ldexp(val_lo, exp_bc).to(tl.bfloat16)
        val_hi = _ldexp(val_hi, exp_bc).to(tl.bfloat16)

        offs_k_even = k_start + tl.arange(0, BLOCK_K_HALF) * 2
        offs_k_odd = offs_k_even + 1

        lhs_even_ptrs = (
            lhs_ptr
            + offs_m[:, None] * stride_lhs_m
            + offs_k_even[None, :] * stride_lhs_k
        )
        lhs_odd_ptrs = (
            lhs_ptr
            + offs_m[:, None] * stride_lhs_m
            + offs_k_odd[None, :] * stride_lhs_k
        )

        lhs_even_mask = (offs_m[:, None] < M) & (offs_k_even[None, :] < K)
        lhs_odd_mask = (offs_m[:, None] < M) & (offs_k_odd[None, :] < K)

        lhs_even = tl.load(lhs_even_ptrs, mask=lhs_even_mask, other=0.0)
        lhs_odd = tl.load(lhs_odd_ptrs, mask=lhs_odd_mask, other=0.0)

        acc += tl.dot(
            lhs_even.to(tl.bfloat16), tl.trans(val_lo), allow_tf32=False
        )
        acc += tl.dot(
            lhs_odd.to(tl.bfloat16), tl.trans(val_hi), allow_tf32=False
        )

    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias_vals[None, :].to(tl.float32)

    out_ptrs = (
        out_ptr
        + offs_m[:, None] * stride_out_m
        + offs_n[None, :] * stride_out_n
    )
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


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
    assert K == weight_packed.shape[1] * 2, (
        f"K mismatch: activations K={K}, packed weight implies K={weight_packed.shape[1] * 2}"
    )
    assert x_2d.dtype == torch.bfloat16, (
        f"Activations must be bf16, got {x_2d.dtype}"
    )
    assert weight_packed.dtype == torch.uint8
    assert weight_scales.dtype == torch.uint8

    output = torch.empty((M, N), dtype=torch.bfloat16, device=x_2d.device)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    _fused_mxfp4_gemm_kernel[grid](
        x_2d,
        weight_packed,
        weight_scales,
        bias if bias is not None else x_2d,
        output,
        M,
        N,
        K,
        x_2d.stride(0),
        x_2d.stride(1),
        weight_packed.stride(0),
        weight_packed.stride(1),
        weight_scales.stride(0),
        weight_scales.stride(1),
        output.stride(0),
        output.stride(1),
        HAS_BIAS=(bias is not None),
    )

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
