# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from __future__ import annotations

import re
from typing import Callable, Dict, Optional

import torch

FP8_BLOCK = 128

EXPERT_SCALE_KEY_RE = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)


def dequant_fp8_blockwise(
    weight: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    block_size: int = FP8_BLOCK,
) -> torch.Tensor:
    n, k = weight.shape
    w = weight.to(torch.float32)
    s = scale.to(torch.float32)
    s_full = s.repeat_interleave(block_size, dim=0).repeat_interleave(
        block_size, dim=1
    )[:n, :k]
    return (w * s_full).to(dtype)


def dequant_fp8_state_dict(
    state_dict: dict,
    dtype: torch.dtype = torch.bfloat16,
    block_size: int = FP8_BLOCK,
    keep_fp8: Optional[Callable[[str], object]] = None,
) -> Dict[str, torch.Tensor]:
    kept_scales: Dict[str, torch.Tensor] = {}
    for scale_key in [
        k for k in list(state_dict) if k.endswith("weight_scale_inv")
    ]:
        weight_key = scale_key[: -len("_scale_inv")]
        weight = state_dict.get(weight_key)
        scale = state_dict.get(scale_key)
        if weight is None or scale is None:
            continue
        if weight.numel() == 0 or scale.numel() == 0:
            continue
        if keep_fp8 is not None and keep_fp8(weight_key):
            kept_scales[weight_key] = scale.to(torch.float32)
            del state_dict[scale_key]
            continue
        state_dict[weight_key] = dequant_fp8_blockwise(
            weight, scale, dtype, block_size
        )
        del state_dict[scale_key]
    return kept_scales


def stack_expert_scales(
    flat_scales: Dict[str, torch.Tensor],
) -> Dict[int, Dict[str, torch.Tensor]]:
    per_layer: Dict[int, Dict[str, Dict[int, torch.Tensor]]] = {}
    for key, scale in flat_scales.items():
        match = EXPERT_SCALE_KEY_RE.match(key)
        if not match:
            continue
        layer_id = int(match.group(1))
        expert_id = int(match.group(2))
        proj = match.group(3).split("_")[0]
        per_layer.setdefault(layer_id, {}).setdefault(proj, {})[expert_id] = (
            scale
        )

    stacked: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer_id, projs in per_layer.items():
        stacked[layer_id] = {
            proj: torch.stack(
                [experts[i] for i in sorted(experts)]
            ).contiguous()
            for proj, experts in projs.items()
        }
    return stacked
