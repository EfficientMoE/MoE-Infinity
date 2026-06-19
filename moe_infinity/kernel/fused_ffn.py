"""Fused Gate+Up+SiLU FFN Triton kernel.

Computes the gated intermediate ``silu(x @ gate_weight.T) * (x @ up_weight.T)``
in a single Triton pass so ``x`` is read once per tile, then finishes the FFN
with a second matmul against ``down_weight.T``.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _silu(x):
    """Compute the SiLU activation in Triton."""
    return x * tl.sigmoid(x)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
            num_stages=3,
            num_warps=8,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _fused_gate_up_silu_kernel(
    x_ptr,  # [M, K] bf16  — input activations
    gate_weight_ptr,  # [N, K] bf16  — gate projection weights
    up_weight_ptr,  # [N, K] bf16  — up projection weights
    intermediate_ptr,  # [M, N] bf16 — fused intermediate activations
    M,
    N,
    K,
    stride_x_m,
    stride_x_k,
    stride_gate_n,
    stride_gate_k,
    stride_up_n,
    stride_up_k,
    stride_intermediate_m,
    stride_intermediate_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fuse gate/up GEMMs with SiLU and elementwise multiply."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k_block * BLOCK_K
        k_offsets = k_start + offs_k

        x_ptrs = (
            x_ptr
            + offs_m[:, None] * stride_x_m
            + k_offsets[None, :] * stride_x_k
        )
        gate_ptrs = (
            gate_weight_ptr
            + offs_n[:, None] * stride_gate_n
            + k_offsets[None, :] * stride_gate_k
        )
        up_ptrs = (
            up_weight_ptr
            + offs_n[:, None] * stride_up_n
            + k_offsets[None, :] * stride_up_k
        )

        x_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
        weight_mask = (offs_n[:, None] < N) & (k_offsets[None, :] < K)

        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        gate_tile = tl.load(gate_ptrs, mask=weight_mask, other=0.0)
        up_tile = tl.load(up_ptrs, mask=weight_mask, other=0.0)

        gate_acc += tl.dot(
            x_tile.to(tl.bfloat16),
            tl.trans(gate_tile.to(tl.bfloat16)),
            allow_tf32=False,
        )
        up_acc += tl.dot(
            x_tile.to(tl.bfloat16),
            tl.trans(up_tile.to(tl.bfloat16)),
            allow_tf32=False,
        )

    fused = _silu(gate_acc) * up_acc

    intermediate_ptrs = (
        intermediate_ptr
        + offs_m[:, None] * stride_intermediate_m
        + offs_n[None, :] * stride_intermediate_n
    )
    intermediate_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(intermediate_ptrs, fused.to(tl.bfloat16), mask=intermediate_mask)


def reference_ffn(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
) -> torch.Tensor:
    """Reference FFN implementation for correctness checks."""
    return (F.silu(x @ gate_w.T) * (x @ up_w.T)) @ down_w.T


def fused_ffn(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """Run a two-pass fused FFN.

    Pass 1 computes ``silu(x @ gate_weight.T) * (x @ up_weight.T)`` in Triton
    without materialising the gate and up projections separately. Pass 2 applies
    the down projection with a standard matmul.

    Args:
        x: Input activations ``[M, K]`` in bf16.
        gate_weight: Gate projection weights ``[I, K]`` in bf16.
        up_weight: Up projection weights ``[I, K]`` in bf16.
        down_weight: Down projection weights ``[K_out, I]`` in bf16.

    Returns:
        Output activations ``[M, K_out]``.
    """
    if x.dim() != 2:
        raise ValueError(f"x must be 2D [M, K], got shape {tuple(x.shape)}")
    if gate_weight.dim() != 2 or up_weight.dim() != 2 or down_weight.dim() != 2:
        raise ValueError("gate_weight, up_weight, and down_weight must be 2D")

    x = x.contiguous()
    gate_weight = gate_weight.contiguous()
    up_weight = up_weight.contiguous()
    down_weight = down_weight.contiguous()

    if x.dtype != torch.bfloat16:
        raise TypeError(f"x must be bf16, got {x.dtype}")
    if gate_weight.dtype != torch.bfloat16:
        raise TypeError(f"gate_weight must be bf16, got {gate_weight.dtype}")
    if up_weight.dtype != torch.bfloat16:
        raise TypeError(f"up_weight must be bf16, got {up_weight.dtype}")
    if down_weight.dtype != torch.bfloat16:
        raise TypeError(f"down_weight must be bf16, got {down_weight.dtype}")
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if (
        not gate_weight.is_cuda
        or not up_weight.is_cuda
        or not down_weight.is_cuda
    ):
        raise ValueError("all weights must be CUDA tensors")
    if (
        gate_weight.device != x.device
        or up_weight.device != x.device
        or down_weight.device != x.device
    ):
        raise ValueError("x and all weights must be on the same CUDA device")

    M, K = x.shape
    I, gate_k = gate_weight.shape
    up_i, up_k = up_weight.shape
    _, down_i = down_weight.shape

    if gate_k != K or up_k != K:
        raise ValueError(
            f"projection K mismatch: x has K={K}, gate has K={gate_k}, up has K={up_k}"
        )
    if up_i != I:
        raise ValueError(
            f"intermediate size mismatch: gate has I={I}, up has I={up_i}"
        )
    if down_i != I:
        raise ValueError(
            f"down projection mismatch: intermediate I={I}, down expects I={down_i}"
        )

    intermediate = torch.empty((M, I), dtype=torch.bfloat16, device=x.device)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(I, META["BLOCK_N"]),
    )

    _fused_gate_up_silu_kernel[grid](
        x,
        gate_weight,
        up_weight,
        intermediate,
        M,
        I,
        K,
        x.stride(0),
        x.stride(1),
        gate_weight.stride(0),
        gate_weight.stride(1),
        up_weight.stride(0),
        up_weight.stride(1),
        intermediate.stride(0),
        intermediate.stride(1),
    )

    return torch.matmul(intermediate, down_weight.t())
