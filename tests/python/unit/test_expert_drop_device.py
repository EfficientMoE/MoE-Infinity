import contextlib

import pytest
import torch

from moe_infinity.runtime.expert_drop import (
    select_expert_drops,
    select_expert_drops_device,
)

CUDA = torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])


def _random_case(seed, device):
    gen = torch.Generator().manual_seed(seed)
    rows = int(torch.randint(1, 65, (1,), generator=gen))
    experts = int(torch.randint(8, 257, (1,), generator=gen))
    topk = int(torch.randint(2, min(17, experts + 1), (1,), generator=gen))
    weights = torch.rand(rows, experts, generator=gen)
    mask = torch.zeros(rows, experts, dtype=torch.bool)
    top_idx = weights.topk(topk, dim=-1).indices
    mask.scatter_(-1, top_idx, True)
    weights = torch.where(mask, weights, torch.zeros(()))
    weights = weights / weights.sum(-1, keepdim=True)
    resident = torch.rand(experts, generator=gen) < 0.4
    min_k = int(torch.randint(1, topk + 1, (1,), generator=gen))
    mass_budget = float(torch.rand(1, generator=gen)) * 0.3
    flatness_floor = float(torch.rand(1, generator=gen))
    return (
        mask.to(device),
        weights.to(device),
        resident.to(device),
        dict(
            min_k=min_k,
            mass_budget=mass_budget,
            flatness_floor=flatness_floor,
        ),
    )


def _assert_parity(mask, weights, resident, kw):
    ref_mask, ref_w, ref_counts = select_expert_drops(
        mask.cpu(), weights.cpu(), resident.cpu(), **kw
    )
    dev_mask, dev_w, dev_counts = select_expert_drops_device(
        mask, weights, resident, **kw
    )
    assert torch.equal(dev_mask.cpu().bool(), ref_mask.cpu().bool())
    assert torch.allclose(
        dev_w.cpu(), ref_w.cpu(), rtol=1e-5, atol=1e-7, equal_nan=True
    )
    assert int(dev_counts.tokens_seen) == ref_counts.tokens_seen
    assert int(dev_counts.tokens_changed) == ref_counts.tokens_changed
    assert int(dev_counts.experts_dropped) == ref_counts.experts_dropped
    assert int(dev_counts.flatness_bypasses) == ref_counts.flatness_bypasses


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("chunk", range(20))
def test_random_parity(device, chunk):
    for seed in range(chunk * 10, chunk * 10 + 10):
        mask, weights, resident, kw = _random_case(seed, device)
        _assert_parity(mask, weights, resident, kw)


@pytest.mark.parametrize("device", DEVICES)
def test_equal_weight_tie_drops_higher_id_first(device):
    mask = torch.tensor([[True, True, True, False]], device=device)
    weights = torch.tensor([[0.5, 0.25, 0.25, 0.0]], device=device)
    resident = torch.tensor([True, False, False, True], device=device)
    out_mask, _, counts = select_expert_drops_device(
        mask, weights, resident, min_k=1, mass_budget=0.25, flatness_floor=0.5
    )
    assert out_mask.cpu().tolist() == [[True, True, False, False]]
    assert int(counts.experts_dropped) == 1
    _assert_parity(
        mask,
        weights,
        resident,
        dict(min_k=1, mass_budget=0.25, flatness_floor=0.5),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "case",
    [
        "all_resident",
        "non_finite",
        "negative",
        "zero_max",
        "budget_zero",
        "budget_one",
        "min_k_floor",
        "flatness_equal_floor",
        "flatness_above_floor",
        "renorm_zero_sum",
    ],
)
def test_targeted_parity(device, case):
    mask = torch.tensor([[True, True, True, False]], device=device)
    weights = torch.tensor([[0.5, 0.3, 0.2, 0.0]], device=device)
    resident = torch.tensor([True, False, False, True], device=device)
    kw = dict(min_k=1, mass_budget=0.25, flatness_floor=0.5)
    if case == "all_resident":
        resident = torch.ones(4, dtype=torch.bool, device=device)
    elif case == "non_finite":
        weights = torch.tensor([[0.5, float("nan"), 0.2, 0.0]], device=device)
    elif case == "negative":
        weights = torch.tensor([[0.5, -0.3, 0.2, 0.0]], device=device)
    elif case == "zero_max":
        weights = torch.zeros(1, 4, device=device)
    elif case == "budget_zero":
        kw["mass_budget"] = 0.0
    elif case == "budget_one":
        kw["mass_budget"] = 1.0
    elif case == "min_k_floor":
        kw["min_k"] = 3
    elif case == "flatness_equal_floor":
        weights = torch.tensor([[0.4, 0.4, 0.2, 0.0]], device=device)
        kw["flatness_floor"] = 0.5
    elif case == "flatness_above_floor":
        weights = torch.tensor([[0.4, 0.35, 0.25, 0.0]], device=device)
        kw["flatness_floor"] = 0.5
    elif case == "renorm_zero_sum":
        weights = torch.tensor([[0.0, 0.6, 0.4, 0.0]], device=device)
        resident = torch.tensor([True, False, False, True], device=device)
        kw["mass_budget"] = 1.0
        kw["min_k"] = 1
    _assert_parity(mask, weights, resident, kw)


@pytest.mark.parametrize("device", DEVICES)
def test_budget_zero_aliases_inputs(device):
    mask = torch.tensor([[True, True]], device=device)
    weights = torch.tensor([[0.6, 0.4]], device=device)
    resident = torch.tensor([False, False], device=device)
    out_mask, out_w, counts = select_expert_drops_device(
        mask, weights, resident, min_k=1, mass_budget=0.0, flatness_floor=0.5
    )
    assert out_mask is mask
    assert out_w is weights
    assert int(counts.tokens_seen) == 0


def test_validation_matches_reference():
    mask = torch.tensor([[True, True]])
    weights = torch.tensor([[0.6, 0.4]])
    resident = torch.tensor([False, False])
    for bad_kw in (
        dict(min_k=0, mass_budget=0.1, flatness_floor=0.5),
        dict(min_k=1.5, mass_budget=0.1, flatness_floor=0.5),
        dict(min_k=True, mass_budget=0.1, flatness_floor=0.5),
        dict(min_k=1, mass_budget=-0.1, flatness_floor=0.5),
        dict(min_k=1, mass_budget=0.1, flatness_floor=1.5),
    ):
        with pytest.raises(ValueError):
            select_expert_drops_device(mask, weights, resident, **bad_kw)
    with pytest.raises(ValueError):
        select_expert_drops_device(
            mask[0],
            weights[0],
            resident,
            min_k=1,
            mass_budget=0.1,
            flatness_floor=0.5,
        )


@pytest.mark.skipif(not CUDA, reason="requires CUDA")
def test_device_selection_never_synchronizes():
    mask, weights, resident, kw = _random_case(12345, "cuda")
    torch.cuda.synchronize()
    with contextlib.ExitStack() as stack:
        torch.cuda.set_sync_debug_mode("error")
        stack.callback(torch.cuda.set_sync_debug_mode, "default")
        out_mask, out_w, counts = select_expert_drops_device(
            mask, weights, resident, **kw
        )
        assert out_mask.is_cuda and out_w.is_cuda
        assert counts.experts_dropped.is_cuda
    torch.cuda.synchronize()
    _assert_parity(mask, weights, resident, kw)
