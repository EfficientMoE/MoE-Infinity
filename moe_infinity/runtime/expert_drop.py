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
