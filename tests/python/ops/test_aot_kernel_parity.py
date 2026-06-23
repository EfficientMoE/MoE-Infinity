import importlib

import pytest

torch = importlib.import_module("torch")

_LOADER = importlib.import_module("moe_infinity.kernel._loader")
_ROUTER = importlib.import_module("moe_infinity.kernel.router")
_FUSED_QKV = importlib.import_module("moe_infinity.kernel.fused_qkv")

from tests.python.ops.conftest import (
    BF16_ATOL,
    BF16_RTOL,
    requires_cuda,
    requires_triton,
    seed_everything,
)


def _load_first_available(*kernel_names):
    for kernel_name in kernel_names:
        kernel = _LOADER.load_compiled_kernel(kernel_name)
        if kernel is not None:
            return kernel
    return None


def test_loader_returns_none_without_compiled():
    assert _LOADER.load_compiled_kernel("nonexistent") is None


def test_loader_cache_is_dict():
    assert isinstance(_LOADER._KERNEL_CACHE, dict)


@requires_cuda
@requires_triton
def test_aot_vs_jit_router_kernel(seed_everything):
    aot_kernel = _load_first_available(
        "fused_softmax_topk_kernel_nobias",
        "router",
    )
    if aot_kernel is None:
        pytest.skip("AOT not compiled")

    batch, hidden_dim, num_experts, top_k = 16, 128, 128, 2
    hidden_states = torch.randn(
        batch,
        hidden_dim,
        dtype=torch.bfloat16,
        device="cuda",
    ).contiguous()
    weight = torch.randn(
        num_experts,
        hidden_dim,
        dtype=torch.bfloat16,
        device="cuda",
    ).contiguous()

    jit_mask, jit_weight = _ROUTER.launch_fused_softmax_topk_nobias(
        hidden_states,
        weight,
        top_k,
    )
    aot_mask = torch.empty_like(jit_mask)
    aot_weight = torch.empty_like(jit_weight)
    aot_kernel[(batch,)](
        hidden_states,
        weight,
        aot_mask,
        aot_weight,
        batch,
        hidden_dim,
        num_experts,
    )

    assert torch.equal(aot_mask, jit_mask)
    torch.testing.assert_close(
        aot_weight, jit_weight, rtol=BF16_RTOL, atol=BF16_ATOL
    )


@requires_cuda
@requires_triton
def test_aot_vs_jit_fused_qkv(seed_everything):
    aot_kernel = _load_first_available("_fused_qkv_kernel", "fused_qkv_kernel")
    if aot_kernel is None:
        pytest.skip("AOT not compiled")

    tokens, hidden_dim = 16, 256
    num_q_heads, num_kv_heads, head_dim = 8, 4, 16
    total_dim = (num_q_heads + 2 * num_kv_heads) * head_dim

    hidden_states = torch.randn(
        tokens,
        hidden_dim,
        dtype=torch.bfloat16,
        device="cuda",
    ).contiguous()
    weight_qkv = torch.randn(
        hidden_dim,
        total_dim,
        dtype=torch.bfloat16,
        device="cuda",
    ).contiguous()

    jit_q, jit_k, jit_v = _FUSED_QKV.fused_qkv_proj(
        hidden_states,
        weight_qkv,
        num_q_heads,
        num_kv_heads,
        head_dim,
    )
    aot_q = torch.empty_like(jit_q)
    aot_k = torch.empty_like(jit_k)
    aot_v = torch.empty_like(jit_v)
    hidden_2d = hidden_states.reshape(-1, hidden_dim).contiguous()
    q_dim = num_q_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    aot_kernel[(1, 2)](
        hidden_2d,
        weight_qkv,
        aot_q.reshape(tokens, q_dim),
        aot_k.reshape(tokens, kv_dim),
        aot_v.reshape(tokens, kv_dim),
        tokens,
        total_dim,
        hidden_dim,
        q_dim,
        kv_dim,
        hidden_2d.stride(0),
        hidden_2d.stride(1),
        weight_qkv.stride(0),
        weight_qkv.stride(1),
        aot_q.reshape(tokens, q_dim).stride(0),
        aot_q.reshape(tokens, q_dim).stride(1),
        aot_k.reshape(tokens, kv_dim).stride(0),
        aot_k.reshape(tokens, kv_dim).stride(1),
        aot_v.reshape(tokens, kv_dim).stride(0),
        aot_v.reshape(tokens, kv_dim).stride(1),
    )

    torch.testing.assert_close(aot_q, jit_q, rtol=BF16_RTOL, atol=BF16_ATOL)
    torch.testing.assert_close(aot_k, jit_k, rtol=BF16_RTOL, atol=BF16_ATOL)
    torch.testing.assert_close(aot_v, jit_v, rtol=BF16_RTOL, atol=BF16_ATOL)
