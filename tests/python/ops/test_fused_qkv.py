import importlib

import pytest

from tests.python.ops.conftest import (
    BF16_ATOL,
    BF16_RTOL,
    requires_cuda,
    requires_triton,
)

torch = importlib.import_module("torch")

_FUSED_QKV = importlib.import_module("moe_infinity.kernel.fused_qkv")


def _reference_fused_qkv(
    hidden_states,
    weight_qkv,
    num_q_heads,
    num_kv_heads,
    head_dim,
):
    q_dim = num_q_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    weight_q = weight_qkv[:, :q_dim]
    weight_k = weight_qkv[:, q_dim : q_dim + kv_dim]
    weight_v = weight_qkv[:, q_dim + kv_dim :]

    q = hidden_states @ weight_q
    k = hidden_states @ weight_k
    v = hidden_states @ weight_v

    return (
        q.reshape(hidden_states.size(0), num_q_heads, head_dim),
        k.reshape(hidden_states.size(0), num_kv_heads, head_dim),
        v.reshape(hidden_states.size(0), num_kv_heads, head_dim),
    )


@requires_cuda
@requires_triton
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "tokens,hidden_dim",
    [(1, 2048), (4, 2048), (16, 4096), (32, 2048)],
)
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,head_dim",
    [(32, 8, 128), (16, 16, 64), (64, 8, 128)],
)
def test_fused_qkv_matches_reference(
    seed_everything,
    dtype,
    tokens,
    hidden_dim,
    num_q_heads,
    num_kv_heads,
    head_dim,
):
    total_dim = (num_q_heads + 2 * num_kv_heads) * head_dim

    hidden_states = torch.randn(
        tokens,
        hidden_dim,
        dtype=dtype,
        device="cuda",
    ).contiguous()
    weight_qkv = torch.randn(
        hidden_dim,
        total_dim,
        dtype=dtype,
        device="cuda",
    ).contiguous()

    fused_q, fused_k, fused_v = _FUSED_QKV.fused_qkv_proj(
        hidden_states,
        weight_qkv,
        num_q_heads,
        num_kv_heads,
        head_dim,
    )
    ref_q, ref_k, ref_v = _reference_fused_qkv(
        hidden_states,
        weight_qkv,
        num_q_heads,
        num_kv_heads,
        head_dim,
    )

    torch.testing.assert_close(
        fused_q,
        ref_q,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )
    torch.testing.assert_close(
        fused_k,
        ref_k,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )
    torch.testing.assert_close(
        fused_v,
        ref_v,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )
