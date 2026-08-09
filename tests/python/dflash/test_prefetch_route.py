"""Hand-checked unit tests for the DFlash route-ahead prefetch pure helpers.

Autonomous correctness gate for Track A1 of
``.sisyphus/plans/dflash-deferred-tracks-plan.md``. Pure, CPU-only set math
over router outputs -- no model loading, no prefetcher/executor wiring
(that is A2/A3).

Route-ahead prefetch uses the ACTUAL routed union (model-exact), matching how
``ExpertExecutor.dispatch_local`` derives ``expert_list`` from ``router_mask``
(``distributed/expert_executor.py:101-111``) and how the MoE block builds the
mask from ``topk(softmax(logits))`` (``models/gpt_oss.py:130-147``). Softmax is
monotonic, so top-k on raw logits selects the same experts as top-k on the
probabilities -- hence ``union_experts_from_logits`` must agree with
``union_experts_from_mask`` when ``top_k`` matches the routing.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from moe_infinity.spec_decode._prefetch_route import (
    prefetch_coverage,
    rejected_expert_ids,
    union_experts_from_logits,
    union_experts_from_mask,
)

NUM_EXPERTS = 8
TOP_K = 2

# Hand-checked logits [3 tokens, 8 experts]; per-token top-2:
#   token 0: experts {0, 3}   (logits 9.0, 8.0)
#   token 1: experts {3, 5}   (logits 9.0, 8.0)
#   token 2: experts {0, 7}   (logits 9.0, 8.0)
LOGITS = torch.tensor(
    [
        [9.0, 1.0, 2.0, 8.0, 0.0, 3.0, 4.0, 5.0],
        [1.0, 0.0, 2.0, 9.0, 3.0, 8.0, 4.0, 5.0],
        [9.0, 2.0, 0.0, 1.0, 3.0, 4.0, 5.0, 8.0],
    ]
)
TOKEN_TOP2 = [{0, 3}, {3, 5}, {0, 7}]
UNION_TOP2 = [0, 3, 5, 7]


def _mask_from_topk(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Replicate the gpt_oss.py:132-147 routing: softmax -> topk -> scatter bool mask."""
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    selected = torch.topk(probs, top_k, dim=-1).indices
    mask = torch.zeros(logits.shape[0], logits.shape[1], dtype=torch.bool)
    mask.scatter_(1, selected, True)
    return mask


# ---------------------------------------------------------------------------
# union_experts_from_mask
# ---------------------------------------------------------------------------


def test_mask_union_overlapping_tokens():
    mask = torch.tensor(
        [
            [True, False, False, True, False, False, False, False],
            [False, False, False, True, False, True, False, False],
            [True, False, False, False, False, False, False, True],
        ]
    )
    assert union_experts_from_mask(mask) == UNION_TOP2


def test_mask_union_disjoint_tokens():
    mask = torch.zeros(3, NUM_EXPERTS, dtype=torch.bool)
    mask[0, 1] = True
    mask[1, 4] = True
    mask[2, 6] = True
    assert union_experts_from_mask(mask) == [1, 4, 6]


def test_mask_union_single_token():
    mask = torch.zeros(1, NUM_EXPERTS, dtype=torch.bool)
    mask[0, 5] = True
    mask[0, 2] = True
    assert union_experts_from_mask(mask) == [2, 5]


def test_mask_union_all_false_is_empty():
    mask = torch.zeros(4, NUM_EXPERTS, dtype=torch.bool)
    assert union_experts_from_mask(mask) == []


def test_mask_union_zero_tokens_is_empty():
    mask = torch.zeros(0, NUM_EXPERTS, dtype=torch.bool)
    assert union_experts_from_mask(mask) == []


def test_mask_union_all_true_is_all_experts():
    mask = torch.ones(2, NUM_EXPERTS, dtype=torch.bool)
    assert union_experts_from_mask(mask) == list(range(NUM_EXPERTS))


def test_mask_union_int_mask_treats_nonzero_as_routed():
    mask = torch.tensor([[0, 1, 0, 2], [0, 0, 0, 0]], dtype=torch.int64)
    assert union_experts_from_mask(mask) == [1, 3]


@pytest.mark.parametrize(
    "mask",
    [
        [[True, False, True], [False, True, False]],  # python nested list
        np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int64),  # numpy array
    ],
)
def test_mask_union_accepts_list_and_numpy(mask):
    assert union_experts_from_mask(mask) == [0, 1, 2]


def test_mask_union_matches_executor_derivation():
    # Same input, computed the way dispatch_local does it (expert_executor.py:101-111).
    mask = _mask_from_topk(LOGITS, TOP_K)
    num_expert = mask.shape[-1]
    expert_count = (
        torch.sum(mask.view((-1, num_expert)), dim=0).cpu().numpy().flatten()
    )
    executor_list = np.arange(num_expert).astype(int)[expert_count > 0].tolist()
    assert union_experts_from_mask(mask) == executor_list == UNION_TOP2


def test_mask_union_returns_sorted_python_ints():
    mask = torch.zeros(1, NUM_EXPERTS, dtype=torch.bool)
    mask[0, 7] = True
    mask[0, 1] = True
    out = union_experts_from_mask(mask)
    assert out == [1, 7]
    assert all(isinstance(i, int) for i in out)


def test_mask_union_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        union_experts_from_mask(torch.zeros(NUM_EXPERTS, dtype=torch.bool))


# ---------------------------------------------------------------------------
# union_experts_from_logits
# ---------------------------------------------------------------------------


def test_logits_union_topk2_handchecked():
    assert union_experts_from_logits(LOGITS, TOP_K) == UNION_TOP2


def test_logits_union_topk1_is_per_token_argmax():
    # token argmaxes: 0, 3, 0 -> union {0, 3}
    assert union_experts_from_logits(LOGITS, 1) == [0, 3]


def test_logits_union_full_k_is_all_experts():
    assert union_experts_from_logits(LOGITS, NUM_EXPERTS) == list(
        range(NUM_EXPERTS)
    )


def test_logits_union_single_token():
    assert union_experts_from_logits(LOGITS[:1], TOP_K) == sorted(TOKEN_TOP2[0])


def test_logits_union_zero_tokens_is_empty():
    assert union_experts_from_logits(LOGITS[:0], TOP_K) == []


def test_logits_union_matches_mask_union_when_topk_matches_routing():
    # The plan's core invariant: logits-derived union == mask-derived union.
    mask = _mask_from_topk(LOGITS, TOP_K)
    assert union_experts_from_logits(LOGITS, TOP_K) == union_experts_from_mask(
        mask
    )


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.float64]
)
def test_logits_union_dtype_robust(dtype):
    out = union_experts_from_logits(LOGITS.to(dtype), TOP_K)
    assert out == UNION_TOP2


@pytest.mark.parametrize(
    "logits",
    [
        LOGITS.tolist(),  # python nested list
        LOGITS.numpy(),  # numpy array
    ],
)
def test_logits_union_accepts_list_and_numpy(logits):
    assert union_experts_from_logits(logits, TOP_K) == UNION_TOP2


def test_logits_union_rejects_bad_topk():
    with pytest.raises(ValueError, match="top_k"):
        union_experts_from_logits(LOGITS, 0)
    with pytest.raises(ValueError, match="top_k"):
        union_experts_from_logits(LOGITS, NUM_EXPERTS + 1)


def test_logits_union_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        union_experts_from_logits(LOGITS.flatten(), TOP_K)


# ---------------------------------------------------------------------------
# prefetch_coverage
# ---------------------------------------------------------------------------


def test_coverage_full_when_prediction_is_superset():
    assert prefetch_coverage([0, 3, 5, 7, 99], [0, 3, 5, 7]) == 1.0


def test_coverage_full_when_identical():
    assert prefetch_coverage([0, 3, 5, 7], [0, 3, 5, 7]) == 1.0


def test_coverage_partial():
    # predicted {0, 3, 4} vs actual {0, 3, 5, 7} -> 2/4
    assert prefetch_coverage([0, 3, 4], [0, 3, 5, 7]) == 0.5


def test_coverage_zero_on_disjoint():
    assert prefetch_coverage([1, 2], [0, 3, 5, 7]) == 0.0


def test_coverage_empty_actual_is_one():
    assert prefetch_coverage([0, 3], []) == 1.0


def test_coverage_both_empty_is_one():
    assert prefetch_coverage([], []) == 1.0


def test_coverage_empty_prediction_nonempty_actual_is_zero():
    assert prefetch_coverage([], [0, 3, 5, 7]) == 0.0


def test_coverage_uses_set_semantics_not_multiset():
    # duplicates in either side must not change the ratio
    assert prefetch_coverage([0, 0, 3, 3], [0, 3, 3, 5, 7, 7]) == 0.5


def test_coverage_accepts_tensors_and_numpy():
    predicted = torch.tensor([0, 3, 5])
    actual = np.array([0, 3, 5, 7], dtype=np.int64)
    assert prefetch_coverage(predicted, actual) == 0.75
    assert isinstance(prefetch_coverage(predicted, actual), float)


# ---------------------------------------------------------------------------
# rejected_expert_ids
# ---------------------------------------------------------------------------


def test_rejected_partial_waste():
    # full block union {0,3,5,7}; kept prefix only routed {0,3} -> waste {5,7}
    assert rejected_expert_ids(UNION_TOP2, [0, 3]) == [5, 7]


def test_rejected_none_when_all_kept():
    assert rejected_expert_ids(UNION_TOP2, UNION_TOP2) == []


def test_rejected_all_when_nothing_kept():
    assert rejected_expert_ids(UNION_TOP2, []) == UNION_TOP2


def test_rejected_empty_full_is_empty():
    assert rejected_expert_ids([], [0, 3]) == []


def test_rejected_both_empty_is_empty():
    assert rejected_expert_ids([], []) == []


def test_rejected_kept_superset_of_full_is_empty():
    assert rejected_expert_ids([0, 3], [0, 3, 5, 7]) == []


def test_rejected_returns_sorted_python_ints():
    out = rejected_expert_ids(torch.tensor([7, 5, 3, 0]), np.array([3, 0]))
    assert out == [5, 7]
    assert all(isinstance(i, int) for i in out)


# ---------------------------------------------------------------------------
# end-to-end purity: mask -> union -> coverage/waste pipeline
# ---------------------------------------------------------------------------


def test_block_union_vs_kept_prefix_pipeline():
    # Simulate a 3-token block where only the first 2 tokens survive verify:
    # full union over all 3 tokens, kept union over tokens 0..1.
    mask = _mask_from_topk(LOGITS, TOP_K)
    full = union_experts_from_mask(mask)
    kept = union_experts_from_mask(mask[:2])
    assert full == UNION_TOP2
    assert kept == [0, 3, 5]
    assert prefetch_coverage(kept, full) == 0.75
    assert rejected_expert_ids(full, kept) == [7]
