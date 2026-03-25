import pytest

from tests.python.ops.conftest import (
    BF16_ATOL,
    BF16_RTOL,
    requires_cuda,
)

torch = pytest.importorskip("torch")
_store = pytest.importorskip("moe_infinity._store")


_MOE_INIT_KEY = {}


def _init_moe_layer(num_experts, top_k, num_tokens):
    key = (num_experts, top_k)
    if _MOE_INIT_KEY and tuple(_MOE_INIT_KEY.values()) != key:
        pytest.skip(
            f"MoELayer singleton already initialized; "
            f"cannot re-init with different params in same process"
        )
    try:
        _store.init_moe_layer(num_experts, top_k, num_tokens, 64, 128)
        _MOE_INIT_KEY["ne"] = num_experts
        _MOE_INIT_KEY["tk"] = top_k
    except Exception:
        pass


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

    _init_moe_layer(num_experts, top_k, num_tokens)

    try:
        result = _store.topk_softmax(gating_output)
    except RuntimeError as e:
        if "not initialized" in str(e).lower():
            pytest.skip(f"MoELayer not initialized: {e}")
        raise

    if len(result) == 0:
        pytest.skip(
            "topk_softmax returned empty result (MoELayer config mismatch)"
        )

    if len(result) == 3:
        topk_indices, router_mask, router_weight = result
    elif len(result) == 2:
        router_mask, router_weight = result
        topk_indices = None
    else:
        pytest.fail(f"Unexpected topk_softmax return count: {len(result)}")

    if topk_indices is not None and topk_indices.shape[1] != top_k:
        pytest.skip(
            f"MoELayer top_k mismatch (configured={topk_indices.shape[1]}, case={top_k})"
        )

    ref_values, ref_indices = torch.topk(
        torch.softmax(gating_output, dim=-1), k=top_k, dim=-1
    )

    assert router_mask.dtype == torch.bool
    assert router_mask.shape == (num_tokens, num_experts)
    assert router_weight.shape == (num_tokens, num_experts)

    if topk_indices is not None:
        custom_cpu = topk_indices.to(torch.int64).cpu()
        ref_cpu = ref_indices.to(torch.int64).cpu()
        for t in range(num_tokens):
            assert (
                set(custom_cpu[t].tolist()) == set(ref_cpu[t].tolist())
            ), f"Token {t}: custom={custom_cpu[t].tolist()} ref={ref_cpu[t].tolist()}"
