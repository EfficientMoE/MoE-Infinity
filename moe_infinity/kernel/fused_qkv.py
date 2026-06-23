"""Fused QKV projection Triton kernel.

Computes Q, K, V = X @ [W_q | W_k | W_v] in a single GEMM pass."""

# ruff: noqa: I001

import triton
import triton.language as tl
import torch


_AUTOTUNE_CONFIGS = [
    triton.Config(
        {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k},
        num_stages=3 if block_k == 32 else 4,
        num_warps=4 if block_m * block_n <= 8192 else 8,
    )
    for block_m in (32, 64, 128)
    for block_n in (64, 128, 256)
    for block_k in (32, 64)
]


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["M", "N", "K"],
)
@triton.jit
def _fused_qkv_kernel(
    hidden_ptr,
    weight_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    M,
    N,
    K,
    Q_DIM,
    KV_DIM,
    stride_hidden_m,
    stride_hidden_k,
    stride_weight_k,
    stride_weight_n,
    stride_q_m,
    stride_q_n,
    stride_k_m,
    stride_k_n,
    stride_v_m,
    stride_v_n,
    OUTPUT_IS_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        k_block = k_start * BLOCK_K + offs_k
        hidden_ptrs = (
            hidden_ptr
            + offs_m[:, None] * stride_hidden_m
            + k_block[None, :] * stride_hidden_k
        )
        weight_ptrs = (
            weight_ptr
            + k_block[:, None] * stride_weight_k
            + offs_n[None, :] * stride_weight_n
        )

        hidden_mask = (offs_m[:, None] < M) & (k_block[None, :] < K)
        weight_mask = (k_block[:, None] < K) & (offs_n[None, :] < N)

        hidden = tl.load(hidden_ptrs, mask=hidden_mask, other=0.0)
        weight = tl.load(weight_ptrs, mask=weight_mask, other=0.0)
        acc += tl.dot(hidden, weight, allow_tf32=False)

    if OUTPUT_IS_BF16:
        out = acc.to(tl.bfloat16)
    else:
        out = acc.to(tl.float16)

    q_mask = (offs_m[:, None] < M) & (offs_n[None, :] < Q_DIM)
    q_ptrs = q_ptr + offs_m[:, None] * stride_q_m + offs_n[None, :] * stride_q_n
    tl.store(q_ptrs, out, mask=q_mask)

    k_offsets = offs_n - Q_DIM
    k_mask = (
        (offs_m[:, None] < M)
        & (offs_n[None, :] >= Q_DIM)
        & (offs_n[None, :] < Q_DIM + KV_DIM)
    )
    k_ptrs = (
        k_ptr + offs_m[:, None] * stride_k_m + k_offsets[None, :] * stride_k_n
    )
    tl.store(k_ptrs, out, mask=k_mask)

    v_offsets = offs_n - Q_DIM - KV_DIM
    v_mask = (
        (offs_m[:, None] < M)
        & (offs_n[None, :] >= Q_DIM + KV_DIM)
        & (offs_n[None, :] < N)
    )
    v_ptrs = (
        v_ptr + offs_m[:, None] * stride_v_m + v_offsets[None, :] * stride_v_n
    )
    tl.store(v_ptrs, out, mask=v_mask)


def fused_qkv_proj(
    hidden_states: torch.Tensor,
    weight_qkv: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project hidden states into Q, K, V with one Triton GEMM pass.

    Args:
        hidden_states: Input activations with shape ``[*, hidden_dim]``.
        weight_qkv: Concatenated projection weights with shape
            ``[hidden_dim, (num_q_heads + 2 * num_kv_heads) * head_dim]``.
        num_q_heads: Number of query heads.
        num_kv_heads: Number of key/value heads.
        head_dim: Per-head hidden size.

    Returns:
        Tuple of ``(q, k, v)`` with shapes
        ``[*, num_q_heads, head_dim]``,
        ``[*, num_kv_heads, head_dim]``,
        ``[*, num_kv_heads, head_dim]``.
    """
    if hidden_states.dim() < 2:
        raise ValueError(
            f"hidden_states must be at least 2D, got shape {tuple(hidden_states.shape)}"
        )
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(
            f"hidden_states must be fp16 or bf16, got {hidden_states.dtype}"
        )
    if weight_qkv.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(
            f"weight_qkv must be fp16 or bf16, got {weight_qkv.dtype}"
        )
    if hidden_states.dtype != weight_qkv.dtype:
        raise TypeError(
            "hidden_states and weight_qkv must share the same dtype, got "
            f"{hidden_states.dtype} and {weight_qkv.dtype}"
        )
    if not hidden_states.is_cuda or not weight_qkv.is_cuda:
        raise ValueError("hidden_states and weight_qkv must be CUDA tensors")

    orig_shape = hidden_states.shape
    hidden_states_2d = hidden_states.reshape(
        -1, hidden_states.shape[-1]
    ).contiguous()
    weight_qkv = weight_qkv.contiguous()

    M, K = hidden_states_2d.shape
    q_dim = num_q_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    total_dim = q_dim + 2 * kv_dim

    if weight_qkv.dim() != 2:
        raise ValueError(
            f"weight_qkv must be 2D, got shape {tuple(weight_qkv.shape)}"
        )
    if weight_qkv.shape[0] != K:
        raise ValueError(
            f"hidden dimension mismatch: hidden_states={K}, weight_qkv={weight_qkv.shape[0]}"
        )
    if weight_qkv.shape[1] != total_dim:
        raise ValueError(
            "output dimension mismatch: expected "
            f"{total_dim}, got {weight_qkv.shape[1]}"
        )

    q = torch.empty(
        (M, q_dim), dtype=hidden_states_2d.dtype, device=hidden_states_2d.device
    )
    k = torch.empty(
        (M, kv_dim),
        dtype=hidden_states_2d.dtype,
        device=hidden_states_2d.device,
    )
    v = torch.empty(
        (M, kv_dim),
        dtype=hidden_states_2d.dtype,
        device=hidden_states_2d.device,
    )

    grid = lambda meta: (  # noqa: E731
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(total_dim, meta["BLOCK_N"]),
    )

    _fused_qkv_kernel[grid](
        hidden_states_2d,
        weight_qkv,
        q,
        k,
        v,
        M,
        total_dim,
        K,
        q_dim,
        kv_dim,
        hidden_states_2d.stride(0),
        hidden_states_2d.stride(1),
        weight_qkv.stride(0),
        weight_qkv.stride(1),
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        OUTPUT_IS_BF16=(hidden_states_2d.dtype == torch.bfloat16),
    )

    prefix_shape = orig_shape[:-1]
    return (
        q.reshape(*prefix_shape, num_q_heads, head_dim),
        k.reshape(*prefix_shape, num_kv_heads, head_dim),
        v.reshape(*prefix_shape, num_kv_heads, head_dim),
    )
