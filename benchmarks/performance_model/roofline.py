from __future__ import annotations

from typing import Literal

from benchmarks.performance_model.types import (
    DemandResult,
    ModelParams,
    WorkloadPoint,
)


def dtype_bytes(dtype: str) -> float:
    mapping = {"fp8": 1.0, "fp4": 0.5, "bf16": 2.0, "fp16": 2.0, "fp32": 4.0}
    return mapping[dtype]


def decode_flops_per_token(p: ModelParams) -> int:
    moe_layers = p.num_layers - p.first_k_dense

    # Attention: per token, per layer — QKV proj + O proj
    # Q: hidden -> num_attn_heads * head_dim  (2 * MAC)
    # K: hidden -> num_kv_heads * head_dim
    # V: hidden -> num_kv_heads * head_dim
    # O: num_attn_heads * head_dim -> hidden
    attn_qkv_o = (
        2
        * p.hidden_size
        * (
            p.num_attn_heads * p.head_dim  # Q
            + p.num_kv_heads * p.head_dim  # K
            + p.num_kv_heads * p.head_dim  # V
            + p.num_attn_heads * p.head_dim  # O
        )
    )
    attn_flops = p.num_layers * attn_qkv_o

    # Dense FFN (first_k_dense layers): gate + up + down (3 matmuls, each 2*MAC)
    # Assume intermediate_size = expert_intermediate_size for dense layers
    dense_ffn_per_layer = 2 * (
        p.hidden_size * p.expert_intermediate_size  # gate
        + p.hidden_size * p.expert_intermediate_size  # up
        + p.expert_intermediate_size * p.hidden_size  # down
    )
    dense_flops = p.first_k_dense * dense_ffn_per_layer

    # MoE FFN: top_k active routed experts + shared experts, per MoE layer
    # Each expert: gate_proj + up_proj + down_proj (3 matmuls, each 2*MAC)
    expert_ffn_per_expert = 2 * (
        p.hidden_size * p.expert_intermediate_size  # gate
        + p.hidden_size * p.expert_intermediate_size  # up
        + p.expert_intermediate_size * p.hidden_size  # down
    )
    moe_flops_per_layer = (p.top_k + p.shared_experts) * expert_ffn_per_expert
    moe_flops = moe_layers * moe_flops_per_layer

    return attn_flops + dense_flops + moe_flops


def decode_hbm_bytes_per_token(
    p: ModelParams, dtype_override: str | None = None
) -> int:
    expert_bw = dtype_bytes(
        dtype_override if dtype_override else p.expert_dtype
    )
    attn_bw = dtype_bytes(p.attn_dtype)

    # Attention weight bytes per layer (Q, K, V, O projections)
    attn_weight_bytes_per_layer = attn_bw * (
        p.hidden_size * p.num_attn_heads * p.head_dim  # Q
        + p.hidden_size * p.num_kv_heads * p.head_dim  # K
        + p.hidden_size * p.num_kv_heads * p.head_dim  # V
        + p.num_attn_heads * p.head_dim * p.hidden_size  # O
    )
    attn_bytes = p.num_layers * attn_weight_bytes_per_layer

    # Dense FFN weight bytes
    dense_ffn_bytes_per_layer = attn_bw * (
        p.hidden_size * p.expert_intermediate_size  # gate
        + p.hidden_size * p.expert_intermediate_size  # up
        + p.expert_intermediate_size * p.hidden_size  # down
    )
    dense_bytes = p.first_k_dense * dense_ffn_bytes_per_layer

    moe_layers = p.num_layers - p.first_k_dense

    # Routed expert weight bytes: top_k active experts per MoE layer
    routed_expert_bytes_per_layer = (
        expert_bw
        * p.top_k
        * (
            p.hidden_size * p.expert_intermediate_size  # gate
            + p.hidden_size * p.expert_intermediate_size  # up
            + p.expert_intermediate_size * p.hidden_size  # down
        )
    )

    # Shared expert weight bytes (always bf16 / attn_bw)
    shared_expert_bytes_per_layer = (
        attn_bw
        * p.shared_experts
        * (
            p.hidden_size * p.expert_intermediate_size
            + p.hidden_size * p.expert_intermediate_size
            + p.expert_intermediate_size * p.hidden_size
        )
    )

    moe_bytes = moe_layers * (
        routed_expert_bytes_per_layer + shared_expert_bytes_per_layer
    )

    return int(attn_bytes + dense_bytes + moe_bytes)


def arithmetic_intensity(flops: int, hbm_bytes: int) -> float:
    return flops / hbm_bytes


def classify_bound(
    flops: int,
    hbm_bytes: int,
    pcie_bytes: int,
    peak_flops: float,
    hbm_gbps: float,
    pcie_gbps: float,
) -> Literal["compute", "hbm", "pcie"]:
    t_compute = flops / peak_flops
    t_hbm = hbm_bytes / (hbm_gbps * 1e9)
    t_pcie = (
        pcie_bytes / (pcie_gbps * 1e9)
        if pcie_gbps > 0 and pcie_bytes > 0
        else 0.0
    )

    bottleneck = max(t_compute, t_hbm, t_pcie)
    if bottleneck == t_pcie and t_pcie > 0:
        return "pcie"
    if bottleneck == t_hbm:
        return "hbm"
    return "compute"


def predict_decode(
    p: ModelParams,
    wp: WorkloadPoint,
    peak_flops: float = 312e12,
    hbm_gbps: float = 3350.0,
    pcie_gbps: float = 0.0,
) -> DemandResult:
    flops = decode_flops_per_token(p)
    hbm_bytes = decode_hbm_bytes_per_token(p)
    pcie_bytes = 0

    ai = arithmetic_intensity(flops, hbm_bytes)
    bound = classify_bound(
        flops, hbm_bytes, pcie_bytes, peak_flops, hbm_gbps, pcie_gbps
    )

    return DemandResult(
        flops_per_token=flops,
        hbm_bytes_per_token=hbm_bytes,
        pcie_bytes_per_token=pcie_bytes,
        arithmetic_intensity=ai,
        bound=bound,
    )
