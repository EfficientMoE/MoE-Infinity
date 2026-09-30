# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ExpertDropCounts:
    tokens_seen: int
    tokens_changed: int
    experts_dropped: int
    flatness_bypasses: int


def _validate_fraction(value: float, message: str) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(message)


def select_expert_drops(
    router_mask: torch.Tensor,
    router_weights: torch.Tensor,
    resident: torch.Tensor,
    *,
    min_k: int,
    mass_budget: float,
    flatness_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, ExpertDropCounts]:
    """Select non-resident routed experts to drop under a mass budget.

    The input ``router_mask`` and ``router_weights`` objects are returned
    unchanged (aliased) when the budget is zero or no row changes. Changed
    decisions return fresh mask and weight tensors.
    """
    if router_mask.dim() != 2:
        raise ValueError("router_mask must be a 2-D tensor")
    if router_weights.shape != router_mask.shape:
        raise ValueError("router_weights must match router_mask shape")
    if resident.shape != (router_mask.shape[-1],):
        raise ValueError("resident must have shape (num_experts,)")
    if type(min_k) is not int or min_k < 1:
        raise ValueError("min_k must be an integer >= 1")
    _validate_fraction(
        mass_budget, "mass_budget must be a finite float in [0, 1]"
    )
    _validate_fraction(
        flatness_floor, "flatness_floor must be a finite float in [0, 1]"
    )

    if mass_budget == 0:
        return router_mask, router_weights, ExpertDropCounts(0, 0, 0, 0)

    original_mask_device = router_mask.device
    original_weight_device = router_weights.device
    if router_mask.is_cuda:
        packed_parts = (
            router_weights.reshape(-1),
            router_mask.reshape(-1),
        )
        resident_is_packed = resident.is_cuda
        if resident_is_packed:
            packed_parts += (resident,)
        packed = torch.cat(packed_parts).cpu()
        weights_end = router_weights.numel()
        mask_end = weights_end + router_mask.numel()
        out_w = packed[:weights_end].reshape(router_weights.shape)
        out_mask = (
            packed[weights_end:mask_end].reshape(router_mask.shape).bool()
        )
        resident_cpu = (
            packed[mask_end:].bool()
            if resident_is_packed
            else resident.to(device="cpu", dtype=torch.bool)
        )
    else:
        out_mask = router_mask.to(device="cpu", dtype=torch.bool, copy=True)
        out_w = router_weights.to(device="cpu", copy=True)
        resident_cpu = resident.to(device="cpu", dtype=torch.bool)

    tokens_seen = 0
    tokens_changed = 0
    experts_dropped = 0
    flatness_bypasses = 0

    for r in range(out_mask.shape[0]):
        tokens_seen += 1
        mask_row = out_mask[r]
        w_row = out_w[r]
        selected_count = int(mask_row.sum().item())
        if selected_count <= min_k:
            continue
        selected_weights = w_row[mask_row]
        finite = bool(torch.isfinite(selected_weights).all())
        if not finite or bool((selected_weights < 0).any()):
            continue
        max_weight = float(selected_weights.max().item())
        if max_weight == 0.0:
            continue
        min_weight = float(selected_weights.min().item())
        # Strict ">" bypasses only flatter rows; equality still drops.
        if min_weight / max_weight > flatness_floor:
            flatness_bypasses += 1
            continue

        candidate_ids = (
            torch.nonzero(mask_row & ~resident_cpu, as_tuple=False)
            .flatten()
            .flip(0)
        )
        if candidate_ids.numel() == 0:
            continue
        if candidate_ids.numel() > 1:
            candidate_order = torch.argsort(w_row[candidate_ids], stable=True)
            candidate_ids = candidate_ids[candidate_order]
        cumulative_mass = torch.cumsum(
            w_row[candidate_ids], dim=0, dtype=torch.float64
        )
        budget_count = int((cumulative_mass <= mass_budget).sum().item())
        dropped_here = min(budget_count, selected_count - min_k)

        if dropped_here == 0:
            continue
        dropped_ids = candidate_ids[:dropped_here]
        out_mask[r, dropped_ids] = False
        out_w[r, dropped_ids] = 0
        tokens_changed += 1
        experts_dropped += dropped_here

        surviving = out_mask[r]
        surviving_sum = out_w[r][surviving].sum()
        if float(surviving_sum.item()) != 0.0:
            out_w[r][surviving] = out_w[r][surviving] / surviving_sum

    counts = ExpertDropCounts(
        tokens_seen, tokens_changed, experts_dropped, flatness_bypasses
    )
    if tokens_changed == 0:
        return router_mask, router_weights, counts

    device_mask = out_mask.to(device=original_mask_device)
    device_w = out_w.to(device=original_weight_device)

    return (
        device_mask,
        device_w,
        counts,
    )


@dataclass(frozen=True)
class ExpertDropCountTensors:
    """Device-resident drop statistics.

    Every field is a 0-dim integer tensor on the selection device. Callers
    accumulate these lazily (e.g. into a device-side stats buffer) and only
    synchronize when stats are actually read, keeping the decode hot path
    free of host round-trips.
    """

    tokens_seen: torch.Tensor
    tokens_changed: torch.Tensor
    experts_dropped: torch.Tensor
    flatness_bypasses: torch.Tensor


def select_expert_drops_device(
    router_mask: torch.Tensor,
    router_weights: torch.Tensor,
    resident: torch.Tensor,
    *,
    min_k: int,
    mass_budget: float,
    flatness_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, ExpertDropCountTensors]:
    """Device-side, fully vectorized variant of :func:`select_expert_drops`.

    Semantics are identical to the CPU reference, but the selection is
    expressed entirely as tensor ops on ``router_mask.device``: no ``.cpu()``,
    no ``.item()``, and no data-dependent Python branching, so the call
    enqueues work without forcing a host synchronization (PROTOTYPE for
    issue #234 Phase 2, option (a) of the decision checkpoint).

    Differences from the CPU reference, by design:

    - ``counts`` are 0-dim device tensors (:class:`ExpertDropCountTensors`).
    - Fresh mask/weight tensors are returned whenever ``mass_budget > 0``,
      even if no row changed (detecting "no change" would require a sync).
      ``mass_budget == 0`` still aliases the inputs.
    """
    if router_mask.dim() != 2:
        raise ValueError("router_mask must be a 2-D tensor")
    if router_weights.shape != router_mask.shape:
        raise ValueError("router_weights must match router_mask shape")
    if resident.shape != (router_mask.shape[-1],):
        raise ValueError("resident must have shape (num_experts,)")
    if type(min_k) is not int or min_k < 1:
        raise ValueError("min_k must be an integer >= 1")
    _validate_fraction(
        mass_budget, "mass_budget must be a finite float in [0, 1]"
    )
    _validate_fraction(
        flatness_floor, "flatness_floor must be a finite float in [0, 1]"
    )

    device = router_mask.device
    num_rows, num_experts = router_mask.shape

    def _zero() -> torch.Tensor:
        return torch.zeros((), dtype=torch.int64, device=device)

    if mass_budget == 0:
        zero = _zero()
        return (
            router_mask,
            router_weights,
            ExpertDropCountTensors(zero, zero, zero, zero),
        )

    mask = router_mask.bool()
    w = router_weights
    resident_dev = resident.to(device=device, dtype=torch.bool)

    zero_w = torch.zeros((), dtype=w.dtype, device=device)
    neg_inf = torch.full((), float("-inf"), dtype=w.dtype, device=device)
    pos_inf = torch.full((), float("inf"), dtype=w.dtype, device=device)

    selected_count = mask.sum(-1)
    w_sel = torch.where(mask, w, zero_w)
    finite_ok = torch.isfinite(w_sel).all(-1)
    nonneg_ok = ~(w_sel < 0).any(-1)
    max_w = torch.where(mask, w, neg_inf).amax(-1)
    min_w = torch.where(mask, w, pos_inf).amin(-1)

    # Guard order mirrors the CPU loop: count floor, finite/negative,
    # zero max weight, then flatness. Only rows passing the earlier guards
    # count as flatness bypasses.
    base_ok = (selected_count > min_k) & finite_ok & nonneg_ok & (max_w > 0)
    ratio = min_w.double() / max_w.double()
    flat_bypass = base_ok & (ratio > flatness_floor)
    eligible = base_ok & ~flat_bypass

    candidates = mask & ~resident_dev.unsqueeze(0)

    # Drop order: ascending weight, ties broken by higher expert id first.
    # Reversing the expert axis makes a stable ascending argsort resolve
    # ties toward higher original ids; non-candidates sort last via +inf.
    sort_key = torch.where(
        candidates,
        w.double(),
        torch.full((), float("inf"), dtype=torch.float64, device=device),
    )
    order_rev = torch.argsort(sort_key.flip(-1), dim=-1, stable=True)
    sorted_ids = (num_experts - 1) - order_rev
    sorted_is_cand = torch.gather(candidates, -1, sorted_ids)
    sorted_w = torch.where(
        sorted_is_cand,
        torch.gather(w.double(), -1, sorted_ids),
        torch.zeros((), dtype=torch.float64, device=device),
    )
    cumulative = torch.cumsum(sorted_w, dim=-1)
    budget_count = (sorted_is_cand & (cumulative <= mass_budget)).sum(-1)

    allowed = (selected_count - min_k).clamp_min(0)
    dropped_here = torch.minimum(budget_count, allowed)
    dropped_here = torch.where(
        eligible, dropped_here, torch.zeros_like(dropped_here)
    )

    position = torch.arange(num_experts, device=device).unsqueeze(0)
    drop_sorted = position < dropped_here.unsqueeze(-1)
    drop_mask = torch.zeros_like(mask)
    drop_mask.scatter_(-1, sorted_ids, drop_sorted)

    changed = dropped_here > 0
    new_mask = mask & ~drop_mask
    stripped_w = torch.where(drop_mask, zero_w, w)
    surviving_sum = torch.where(new_mask, stripped_w, zero_w).sum(
        -1, keepdim=True
    )
    safe_sum = torch.where(
        surviving_sum == 0, torch.ones_like(surviving_sum), surviving_sum
    )
    renormed = torch.where(
        new_mask & (surviving_sum != 0), stripped_w / safe_sum, stripped_w
    )

    changed_col = changed.unsqueeze(-1)
    out_mask = torch.where(changed_col, new_mask, mask)
    out_w = torch.where(changed_col, renormed, w)

    counts = ExpertDropCountTensors(
        tokens_seen=torch.full((), num_rows, dtype=torch.int64, device=device),
        tokens_changed=changed.sum(),
        experts_dropped=dropped_here.sum(),
        flatness_bypasses=flat_bypass.sum(),
    )
    return out_mask, out_w, counts
