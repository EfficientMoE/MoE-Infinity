import pytest
import torch

from moe_infinity.runtime.expert_drop import select_expert_drops


def test_drops_lightest_nonresident_inside_budget():
    mask = torch.tensor([[True, True, True, False]])
    weights = torch.tensor([[0.5, 0.25, 0.125, 0.0]])
    resident = torch.tensor([True, False, False, True])
    out_mask, out_w, counts = select_expert_drops(
        mask, weights, resident, min_k=1, mass_budget=0.25, flatness_floor=0.5
    )
    assert out_mask.tolist() == [[True, True, False, False]]
    assert torch.allclose(
        out_w, torch.tensor([[2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0]])
    )
    assert counts.tokens_seen == 1
    assert counts.tokens_changed == 1
    assert counts.experts_dropped == 1
    assert counts.flatness_bypasses == 0


def test_tie_drops_higher_expert_id_first():
    mask = torch.tensor([[True, True, True, False]])
    weights = torch.tensor([[0.5, 0.25, 0.25, 0.0]])
    resident = torch.tensor([True, False, False, True])
    out_mask, out_w, counts = select_expert_drops(
        mask, weights, resident, min_k=1, mass_budget=0.25, flatness_floor=0.5
    )
    assert out_mask.tolist() == [[True, True, False, False]]
    assert counts.experts_dropped == 1
    assert torch.allclose(
        out_w, torch.tensor([[2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0]])
    )


def test_min_k_blocks_a_drop_the_budget_would_allow():
    mask = torch.tensor([[True, True, True]])
    weights = torch.tensor([[0.5, 0.25, 0.25]])
    resident = torch.tensor([False, False, False])
    out_mask, out_w, counts = select_expert_drops(
        mask, weights, resident, min_k=3, mass_budget=1.0, flatness_floor=1.0
    )
    assert torch.equal(out_mask, mask)
    assert torch.equal(out_w, weights)
    assert counts.experts_dropped == 0


@pytest.mark.parametrize("min_k", [1.5, True])
def test_min_k_rejects_non_integer_values(min_k):
    mask = torch.tensor([[True, True]])
    weights = torch.tensor([[0.75, 0.25]])
    resident = torch.tensor([True, False])

    with pytest.raises(ValueError, match="min_k must be an integer"):
        select_expert_drops(
            mask,
            weights,
            resident,
            min_k=min_k,
            mass_budget=1.0,
            flatness_floor=1.0,
        )


def test_flatness_equal_to_floor_still_drops():
    mask = torch.tensor([[True, True]])
    weights = torch.tensor([[0.5, 0.25]])
    resident = torch.tensor([True, False])
    _, _, counts = select_expert_drops(
        mask, weights, resident, min_k=1, mass_budget=0.25, flatness_floor=0.5
    )
    assert counts.experts_dropped == 1
    assert counts.flatness_bypasses == 0


def test_flatness_above_floor_bypasses():
    mask = torch.tensor([[True, True]])
    weights = torch.tensor([[0.5, 0.25 + 2**-10]])
    resident = torch.tensor([True, False])
    out_mask, out_w, counts = select_expert_drops(
        mask, weights, resident, min_k=1, mass_budget=1.0, flatness_floor=0.5
    )
    assert torch.equal(out_mask, mask)
    assert torch.equal(out_w, weights)
    assert counts.flatness_bypasses == 1
    assert counts.experts_dropped == 0


def test_budget_zero_returns_the_same_objects():
    mask = torch.tensor([[True, False]])
    weights = torch.tensor([[0.5, 0.5]])
    resident = torch.tensor([False, False])
    out_mask, out_w, counts = select_expert_drops(
        mask, weights, resident, min_k=1, mass_budget=0.0, flatness_floor=0.5
    )
    assert out_mask is mask
    assert out_w is weights
    assert counts.tokens_seen == 0
    assert counts.experts_dropped == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.25])
def test_nonfinite_or_negative_selected_weight_is_unchanged(bad):
    mask = torch.tensor([[True, True]])
    weights = torch.tensor([[0.5, bad]])
    resident = torch.tensor([False, False])
    out_mask, out_w, counts = select_expert_drops(
        mask, weights, resident, min_k=1, mass_budget=1.0, flatness_floor=1.0
    )
    assert torch.equal(out_mask, mask)
    assert out_w[0, 0].item() == 0.5
    if bad != bad:  # NaN
        assert out_w[0, 1].item() != out_w[0, 1].item()
    else:
        assert out_w[0, 1].item() == bad
    assert counts.experts_dropped == 0
    assert counts.tokens_changed == 0


def test_preserves_weight_dtype_and_device():
    mask = torch.tensor([[True, True]], dtype=torch.bool)
    weights = torch.tensor([[0.75, 0.25]], dtype=torch.float64)
    resident = torch.tensor([True, False])
    out_mask, out_w, _ = select_expert_drops(
        mask, weights, resident, min_k=1, mass_budget=0.5, flatness_floor=1.0
    )
    assert out_mask.dtype == torch.bool
    assert out_mask.device == mask.device
    assert out_w.dtype == torch.float64
    assert out_w.device == weights.device
    assert torch.allclose(
        out_w, torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    )


def test_never_sets_a_false_mask_bit():
    mask = torch.tensor([[False, True, False]])
    weights = torch.tensor([[0.4, 0.6, 0.2]])
    resident = torch.tensor([False, False, False])
    out_mask, _, _ = select_expert_drops(
        mask, weights, resident, min_k=1, mass_budget=1.0, flatness_floor=1.0
    )
    assert out_mask.tolist() == [[False, True, False]]
