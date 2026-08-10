import os

import pytest

from benchmarks.performance_model.roofline import (
    classify_bound,
    decode_flops_per_token,
    decode_hbm_bytes_per_token,
    dtype_bytes,
    predict_decode,
)
from benchmarks.performance_model.types import ModelParams, WorkloadPoint

GLM_SYNTHETIC = ModelParams(
    name="glm-synthetic",
    num_layers=78,
    num_attn_heads=64,
    num_kv_heads=64,
    head_dim=128,
    hidden_size=7168,
    vocab_size=151552,
    num_experts=256,
    top_k=8,
    shared_experts=1,
    expert_intermediate_size=2048,
    first_k_dense=3,
    expert_dtype="fp8",
    attn_dtype="bf16",
    kv_lora_rank=512,
    q_lora_rank=2048,
)

TINY = ModelParams(
    name="tiny",
    num_layers=4,
    num_attn_heads=8,
    num_kv_heads=8,
    head_dim=64,
    hidden_size=512,
    vocab_size=1000,
    num_experts=8,
    top_k=2,
    shared_experts=0,
    expert_intermediate_size=256,
    first_k_dense=1,
    expert_dtype="bf16",
    attn_dtype="bf16",
)

TINY_TOPK4 = ModelParams(
    name="tiny-topk4",
    num_layers=4,
    num_attn_heads=8,
    num_kv_heads=8,
    head_dim=64,
    hidden_size=512,
    vocab_size=1000,
    num_experts=8,
    top_k=4,
    shared_experts=0,
    expert_intermediate_size=256,
    first_k_dense=1,
    expert_dtype="bf16",
    attn_dtype="bf16",
)


def test_decode_flops_positive():
    flops = decode_flops_per_token(GLM_SYNTHETIC)
    assert flops > 0


def test_decode_flops_scales_with_topk():
    f2 = decode_flops_per_token(TINY)
    f4 = decode_flops_per_token(TINY_TOPK4)
    assert f4 > f2


def test_fp8_expert_bytes_half_of_bf16():
    assert dtype_bytes("fp8") == 1.0
    assert dtype_bytes("bf16") == 2.0
    assert dtype_bytes("bf16") / dtype_bytes("fp8") == 2.0

    base = ModelParams(
        name="base",
        num_layers=4,
        num_attn_heads=8,
        num_kv_heads=8,
        head_dim=64,
        hidden_size=512,
        vocab_size=1000,
        num_experts=8,
        top_k=2,
        shared_experts=0,
        expert_intermediate_size=256,
        first_k_dense=0,
        expert_dtype="fp8",
        attn_dtype="bf16",
    )
    bytes_fp8 = decode_hbm_bytes_per_token(base, dtype_override="fp8")
    bytes_bf16 = decode_hbm_bytes_per_token(base, dtype_override="bf16")
    assert bytes_fp8 < bytes_bf16


def test_classify_bound_memory_heavy():
    result = classify_bound(
        flops=1,
        hbm_bytes=10**12,
        pcie_bytes=0,
        peak_flops=312e12,
        hbm_gbps=3350.0,
        pcie_gbps=0.0,
    )
    assert result == "hbm"


def test_classify_bound_compute_heavy():
    result = classify_bound(
        flops=10**18,
        hbm_bytes=1,
        pcie_bytes=0,
        peak_flops=312e12,
        hbm_gbps=3350.0,
        pcie_gbps=0.0,
    )
    assert result == "compute"


def test_predict_decode_returns_demand_result():
    wp = WorkloadPoint(batch=1, seq_len=512, gen_len=128)
    result = predict_decode(GLM_SYNTHETIC, wp)
    assert result.flops_per_token > 0
    assert result.hbm_bytes_per_token > 0
    assert result.pcie_bytes_per_token == 0
    assert result.arithmetic_intensity > 0
    assert result.bound in ("compute", "hbm", "pcie")


@pytest.mark.skipif(
    not os.path.exists(
        os.path.join(
            os.environ.get("HF_HUB_CACHE", ""),
            "models--zai-org--GLM-5.2-FP8",
        )
    ),
    reason="GLM-5.2-FP8 not in HF cache",
)
def test_extract_glm_params_from_real_config():
    from benchmarks.performance_model.model_config import extract_model_params

    p = extract_model_params("zai-org/GLM-5.2-FP8")
    assert p.num_experts == 256
    assert p.top_k == 8
    assert p.expert_dtype == "fp8"
