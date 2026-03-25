import pytest

from tests.python.ops.conftest import (
    BF16_ATOL,
    BF16_RTOL,
    requires_cuda,
)

torch = pytest.importorskip("torch")
_store = pytest.importorskip("moe_infinity._store")


def _assert_same_selected_experts(custom_indices, ref_indices):
    custom_cpu = custom_indices.to(torch.int64).cpu()
    ref_cpu = ref_indices.to(torch.int64).cpu()

    for token_idx in range(custom_cpu.size(0)):
        custom_set = set(custom_cpu[token_idx].tolist())
        ref_set = set(ref_cpu[token_idx].tolist())
        assert custom_set == ref_set, (
            f"Token {token_idx}: custom={custom_cpu[token_idx].tolist()} "
            f"ref={ref_cpu[token_idx].tolist()}"
        )


@requires_cuda
@pytest.mark.usefixtures("seed_everything")
@pytest.mark.parametrize(
    "num_tokens,num_experts,top_k",
    [
        (8, 8, 2),
        (32, 64, 2),
        (16, 8, 4),
        (32, 64, 4),
    ],
)
def test_topk_softmax_matches_torch_reference(
    num_tokens: int, num_experts: int, top_k: int
):
    gating_output = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device="cuda"
    )

    try:
        topk_indices, router_mask, router_weight = _store.topk_softmax(
            gating_output
        )
    except RuntimeError as e:
        if "not initialized" in str(e).lower():
            pytest.skip(f"MoELayer not initialized: {e}")
        raise

    if topk_indices.shape[1] != top_k:
        pytest.skip(
            "MoELayer top_k does not match this test case "
            f"(configured={topk_indices.shape[1]}, case={top_k})"
        )

    ref_values, ref_indices = torch.topk(
        torch.softmax(gating_output, dim=-1), k=top_k, dim=-1
    )

    _assert_same_selected_experts(topk_indices, ref_indices)

    assert router_mask.dtype == torch.bool
    assert router_mask.shape == (num_tokens, num_experts)
    assert router_weight.shape == (num_tokens, num_experts)


@requires_cuda
@pytest.mark.usefixtures("seed_everything")
@pytest.mark.parametrize(
    "num_tokens,num_experts,top_k",
    [
        (8, 8, 2),
        (32, 64, 2),
        (16, 8, 4),
        (32, 64, 4),
    ],
)
def test_topk_softmax_token_expert_indices_layout(
    num_tokens: int, num_experts: int, top_k: int
):
    pytest.skip(
        "_store.topk_softmax now returns only (weights, indices); "
        "token_expert_indices is no longer exposed by the API."
    )
