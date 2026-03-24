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

    topk_weights = torch.empty(
        num_tokens, top_k, dtype=torch.float32, device="cuda"
    )
    topk_indices = torch.empty(
        num_tokens, top_k, dtype=torch.int32, device="cuda"
    )
    token_expert_indices = torch.empty(
        num_tokens, top_k, dtype=torch.int32, device="cuda"
    )

    _store.topk_softmax(
        topk_weights, topk_indices, token_expert_indices, gating_output
    )

    ref_values, ref_indices = torch.topk(
        torch.softmax(gating_output, dim=-1), k=top_k, dim=-1
    )

    torch.testing.assert_close(
        topk_weights, ref_values, rtol=BF16_RTOL, atol=BF16_ATOL
    )
    _assert_same_selected_experts(topk_indices, ref_indices)


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
    gating_output = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device="cuda"
    )

    topk_weights = torch.empty(
        num_tokens, top_k, dtype=torch.float32, device="cuda"
    )
    topk_indices = torch.empty(
        num_tokens, top_k, dtype=torch.int32, device="cuda"
    )
    token_expert_indices = torch.empty(
        num_tokens, top_k, dtype=torch.int32, device="cuda"
    )

    _store.topk_softmax(
        topk_weights, topk_indices, token_expert_indices, gating_output
    )

    expected = (
        torch.arange(top_k, dtype=torch.int32, device="cuda").unsqueeze(1)
        * num_tokens
        + torch.arange(num_tokens, dtype=torch.int32, device="cuda").unsqueeze(
            0
        )
    ).transpose(0, 1)

    assert torch.equal(token_expert_indices, expected)
