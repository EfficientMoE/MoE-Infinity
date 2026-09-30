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


def _validate_fraction(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(name)


def select_expert_drops(
    router_mask: torch.Tensor,
    router_weights: torch.Tensor,
    resident: torch.Tensor,
    *,
    min_k: int,
    mass_budget: float,
    flatness_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, ExpertDropCounts]:
    if router_mask.dim() != 2:
        raise ValueError("router_mask")
    if router_weights.shape != router_mask.shape:
        raise ValueError("router_weights")
    if resident.shape != (router_mask.shape[-1],):
        raise ValueError("resident")
    if min_k < 1:
        raise ValueError("min_k")
    _validate_fraction(mass_budget, "mass_budget")
    _validate_fraction(flatness_floor, "flatness_floor")

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
