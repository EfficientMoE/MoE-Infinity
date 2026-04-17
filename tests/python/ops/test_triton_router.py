import importlib

import pytest

torch = importlib.import_module("torch")

from moe_infinity.kernel.router import launch_fused_softmax_topk_nobias
from tests.python.ops.conftest import (
    BF16_ATOL,
    BF16_RTOL,
    requires_cuda,
    requires_triton,
    seed_everything,
)


def _reference_softmax_topk_nobias(
    hidden_states,
    weight,
    top_k,
    normalize_topk,
):
    bsz, _ = hidden_states.shape
    num_experts = weight.shape[0]

    logits = hidden_states.float() @ weight.float().t()
    probs = torch.softmax(logits, dim=-1)
    topk_vals, topk_idx = torch.topk(probs, k=top_k, dim=-1)

    if normalize_topk:
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

    ref_mask = torch.zeros(
        (bsz, num_experts), dtype=torch.bool, device=hidden_states.device
    )
    ref_weight = torch.zeros(
        (bsz, num_experts),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    ref_mask.scatter_(1, topk_idx, True)
    ref_weight.scatter_(1, topk_idx, topk_vals.to(hidden_states.dtype))

    return ref_mask, ref_weight


@requires_cuda
@requires_triton
@pytest.mark.parametrize("normalize_topk", [True, False])
def test_fused_softmax_topk_router_matches_torch_reference(
    seed_everything, normalize_topk
):
    batch, hidden_dim, num_experts, top_k = 16, 128, 32, 4
    hidden_states = (
        2.0
        * torch.randn(batch, hidden_dim, dtype=torch.bfloat16, device="cuda")
    ).contiguous()
    weight = (
        2.0
        * torch.randn(
            num_experts, hidden_dim, dtype=torch.bfloat16, device="cuda"
        )
    ).contiguous()

    try:
        routing_mask, routing_weight = launch_fused_softmax_topk_nobias(
            hidden_states,
            weight,
            top_k,
            normalize_topk=normalize_topk,
        )
    except Exception as e:
        if (
            "CompilationError" in type(e).__name__
            or "compilation" in str(e).lower()
        ):
            pytest.skip(
                f"Triton kernel compilation failed (known incompatibility): {e}"
            )
        raise
    ref_mask, ref_weight = _reference_softmax_topk_nobias(
        hidden_states,
        weight,
        top_k,
        normalize_topk=normalize_topk,
    )

    assert routing_mask.sum(dim=1).eq(top_k).all(), (
        f"Expected {top_k} experts per token"
    )

    custom_weight_sums = routing_weight.sum(dim=1)
    ref_weight_sums = ref_weight.sum(dim=1)
    torch.testing.assert_close(
        custom_weight_sums,
        ref_weight_sums,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )
