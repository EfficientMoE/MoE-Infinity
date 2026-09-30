from __future__ import annotations

import math
import random
import time
from types import SimpleNamespace

import pytest
import torch

from moe_infinity.distributed.expert_executor import DistributedExpertExecutor
from moe_infinity.runtime.expert_drop import (
    ExpertDropCounts,
    select_expert_drops,
)


def _validate_fraction(value: float, message: str) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(message)


def _reference_select_expert_drops(
    router_mask: torch.Tensor,
    router_weights: torch.Tensor,
    resident: torch.Tensor,
    *,
    min_k: int,
    mass_budget: float,
    flatness_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, ExpertDropCounts]:
    if router_mask.dim() != 2:
        raise ValueError("router_mask must be a 2-D tensor")
    if router_weights.shape != router_mask.shape:
        raise ValueError("router_weights must match router_mask shape")
    if resident.shape != (router_mask.shape[-1],):
        raise ValueError("resident must have shape (num_experts,)")
    if min_k < 1:
        raise ValueError("min_k must be >= 1")
    _validate_fraction(
        mass_budget, "mass_budget must be a finite float in [0, 1]"
    )
    _validate_fraction(
        flatness_floor, "flatness_floor must be a finite float in [0, 1]"
    )

    if mass_budget == 0:
        return router_mask, router_weights, ExpertDropCounts(0, 0, 0, 0)

    out_mask = router_mask.to(dtype=torch.bool, copy=True)
    out_w = router_weights.clone()

    tokens_seen = 0
    tokens_changed = 0
    experts_dropped = 0
    flatness_bypasses = 0

    for r in range(out_mask.shape[0]):
        tokens_seen += 1
        mask_row = out_mask[r]
        w_row = out_w[r]
        selected = torch.nonzero(mask_row, as_tuple=False).flatten().tolist()
        if len(selected) <= min_k:
            continue
        selected_weights = w_row[mask_row]
        finite = bool(torch.isfinite(selected_weights).all())
        if not finite or bool((selected_weights < 0).any()):
            continue
        max_weight = float(selected_weights.max().item())
        if max_weight == 0.0:
            continue
        min_weight = float(selected_weights.min().item())
        if min_weight / max_weight > flatness_floor:
            flatness_bypasses += 1
            continue

        candidates = [i for i in selected if not bool(resident[i])]
        candidates.sort(key=lambda i: (float(w_row[i].item()), -i))

        remaining = len(selected)
        dropped_mass = 0.0
        dropped_here = 0
        for i in candidates:
            if remaining <= min_k:
                break
            weight = float(w_row[i].item())
            if dropped_mass + weight > mass_budget:
                break
            out_mask[r, i] = False
            out_w[r, i] = 0
            dropped_mass += weight
            remaining -= 1
            dropped_here += 1

        if dropped_here == 0:
            continue
        tokens_changed += 1
        experts_dropped += dropped_here

        surviving = out_mask[r]
        surviving_sum = out_w[r][surviving].sum()
        if float(surviving_sum.item()) != 0.0:
            out_w[r][surviving] = out_w[r][surviving] / surviving_sum

    return (
        out_mask,
        out_w,
        ExpertDropCounts(
            tokens_seen, tokens_changed, experts_dropped, flatness_bypasses
        ),
    )


def _assert_weights_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.allclose(actual, expected, equal_nan=True)


def test_randomized_equivalence_with_reference() -> None:
    rng = random.Random(20260930)
    budgets = (0.0, 0.05, 0.1, 0.2, 1.0)
    floors = (0.0, 0.5, 1.0)
    special_weights = (float("nan"), float("inf"), -0.125, 0.0)

    for case in range(200):
        rows = rng.randint(1, 24)
        experts = rng.choice((8, 64))
        dtype = rng.choice((torch.float32, torch.float64))
        mask = torch.zeros((rows, experts), dtype=torch.bool)
        weights = torch.rand((rows, experts), dtype=dtype)

        selected_counts = []
        for row in range(rows):
            selected_count = rng.randint(1, min(experts, 8))
            selected_counts.append(selected_count)
            selected = rng.sample(range(experts), selected_count)
            mask[row, selected] = True
            if case % 7 == 0:
                weights[row, selected] = rng.choice((0.0, 0.125, 0.25, 0.5))
            if case % 11 == 0:
                weights[row, rng.choice(selected)] = rng.choice(special_weights)

        resident = torch.tensor(
            [rng.random() < 0.7 for _ in range(experts)], dtype=torch.bool
        )
        min_k = rng.randint(1, max(selected_counts))
        kwargs = {
            "min_k": min_k,
            "mass_budget": rng.choice(budgets),
            "flatness_floor": rng.choice(floors),
        }

        actual_mask, actual_weights, actual_counts = select_expert_drops(
            mask, weights, resident, **kwargs
        )
        expected_mask, expected_weights, expected_counts = (
            _reference_select_expert_drops(mask, weights, resident, **kwargs)
        )

        assert torch.equal(actual_mask, expected_mask), f"case {case}"
        _assert_weights_equal(actual_weights, expected_weights)
        assert actual_counts == expected_counts, f"case {case}"


def test_no_change_returns_original_tensors() -> None:
    mask = torch.tensor([[True, True, False]])
    weights = torch.tensor([[0.75, 0.25, 0.0]])
    resident = torch.ones(3, dtype=torch.bool)

    out_mask, out_weights, counts = select_expert_drops(
        mask,
        weights,
        resident,
        min_k=1,
        mass_budget=0.2,
        flatness_floor=1.0,
    )

    assert out_mask is mask
    assert out_weights is weights
    assert counts == ExpertDropCounts(1, 0, 0, 0)


def test_resident_probe_is_memoized_per_layer_and_reset() -> None:
    class Dispatcher:
        def __init__(self) -> None:
            self.calls = 0

        def resident_on_gpu(self, _layer_id: int) -> list[int]:
            self.calls += 1
            return [1, 1]

    config = SimpleNamespace(
        expert_drop_policy="on_miss",
        expert_drop_mass_budget=0.2,
        expert_drop_min_k=1,
        expert_drop_flatness_floor=1.0,
        gpu_only_expert_routing=False,
        speculative_prefetch_overlap=False,
    )
    executor = DistributedExpertExecutor(config)
    dispatcher = Dispatcher()
    executor.set_expert_dispatcher(dispatcher)
    mask = torch.tensor([[True, True]])
    weights = torch.tensor([[0.75, 0.25]])

    executor._apply_expert_drop(3, mask, weights)
    executor._apply_expert_drop(3, mask, weights)

    assert dispatcher.calls == 1
    assert executor._resident_probe_cache[0] == 3
    assert executor._resident_probe_cache[1].device.type == "cpu"

    executor.reset_expert_drop_stats()
    executor._apply_expert_drop(3, mask, weights)
    assert dispatcher.calls == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_drop_decision_microbenchmark() -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260930)
    weights = torch.rand((1, 64), device="cuda", generator=generator)
    mask = torch.zeros((1, 64), dtype=torch.bool, device="cuda")
    selected = torch.randperm(64, device="cuda", generator=generator)[:6]
    mask[0, selected] = True
    selected_cpu = selected.cpu()
    resident = torch.rand(64, generator=torch.Generator().manual_seed(7)) >= 0.3
    resident[selected_cpu] = True
    resident[selected_cpu[-1]] = False
    weights[0, selected[-1]] = 0.01
    kwargs = {"min_k": 1, "mass_budget": 0.2, "flatness_floor": 1.0}
    layers_per_token = 26
    warmup_tokens = 5
    timed_tokens = 50

    _, _, changed_counts = select_expert_drops(
        mask, weights, resident, **kwargs
    )
    assert changed_counts.tokens_changed == 1

    def measure(selector) -> float:
        for _ in range(warmup_tokens * layers_per_token):
            selector(mask, weights, resident, **kwargs)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(timed_tokens * layers_per_token):
            selector(mask, weights, resident, **kwargs)
        torch.cuda.synchronize()
        return time.perf_counter() - started

    new_time = measure(select_expert_drops)
    reference_time = measure(_reference_select_expert_drops)
    calls = timed_tokens * layers_per_token
    new_us = new_time * 1e6 / calls
    reference_us = reference_time * 1e6 / calls
    result = (
        f"new={new_time:.6f}s ({new_us:.2f} us/call), "
        f"reference={reference_time:.6f}s ({reference_us:.2f} us/call)"
    )
    print(result)

    # Relative speedup is the hard gate; absolute bounds are loose sanity ceilings.
    assert new_time < reference_time * 0.5, result
    assert new_us < 250.0, result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_no_change_microbenchmark() -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260930)
    weights = torch.rand((1, 64), device="cuda", generator=generator)
    mask = torch.zeros((1, 64), dtype=torch.bool, device="cuda")
    selected = torch.randperm(64, device="cuda", generator=generator)[:6]
    mask[0, selected] = True
    resident = torch.ones(64, dtype=torch.bool)
    kwargs = {"min_k": 1, "mass_budget": 0.2, "flatness_floor": 1.0}
    layers_per_token = 26
    warmup_tokens = 5
    timed_tokens = 50

    for _ in range(warmup_tokens * layers_per_token):
        select_expert_drops(mask, weights, resident, **kwargs)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(timed_tokens * layers_per_token):
        select_expert_drops(mask, weights, resident, **kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    calls = timed_tokens * layers_per_token
    per_call_us = elapsed * 1e6 / calls
    result = f"no-change={elapsed:.6f}s ({per_call_us:.2f} us/call)"
    print(result)

    # This absolute bound is only a loose sanity ceiling.
    assert per_call_us < 150.0, result
