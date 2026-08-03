from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class ModelParams:
    name: str
    num_layers: int
    num_attn_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    num_experts: int
    top_k: int
    shared_experts: int
    expert_intermediate_size: int
    first_k_dense: int
    expert_dtype: str  # "fp8" | "bf16"
    attn_dtype: str    # "bf16" | "fp16" | "fp32"
    kv_lora_rank: Optional[int] = None
    q_lora_rank: Optional[int] = None


@dataclass(frozen=True)
class WorkloadPoint:
    batch: int
    seq_len: int
    gen_len: int


@dataclass(frozen=True)
class DemandResult:
    flops_per_token: int
    hbm_bytes_per_token: int
    pcie_bytes_per_token: int
    arithmetic_intensity: float
    bound: Literal["compute", "hbm", "pcie"]
