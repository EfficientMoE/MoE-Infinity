# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .expert_bundle import FP4_SCALE_BLOCK

_FP4_E2M1_TABLE = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _to_packed_bytes(weight: torch.Tensor) -> torch.Tensor:
    if weight.element_size() == 1:
        return weight.contiguous().view(torch.uint8)
    return weight.contiguous().to(torch.uint8)


def dequant_fp4_e2m1(
    weight: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    block_size: int = FP4_SCALE_BLOCK,
) -> torch.Tensor:
    packed = _to_packed_bytes(weight)
    table = torch.tensor(
        _FP4_E2M1_TABLE, dtype=torch.float32, device=packed.device
    )

    low = packed & 0x0F
    high = (packed >> 4) & 0x0F

    unpacked_shape = packed.shape[:-1] + (packed.shape[-1] * 2,)
    unpacked = torch.empty(
        unpacked_shape, dtype=torch.float32, device=packed.device
    )
    unpacked[..., 0::2] = table[low.long()]
    unpacked[..., 1::2] = table[high.long()]

    scale_f32 = scale.to(torch.float32)
    expanded_scale = (
        scale_f32.unsqueeze(-1)
        .expand(*scale_f32.shape, block_size)
        .reshape(*scale_f32.shape[:-1], scale_f32.shape[-1] * block_size)
    )
    expanded_scale = expanded_scale[..., : unpacked.shape[-1]]

    return (unpacked * expanded_scale).to(dtype)


def fp4_expert_forward(
    hidden_states: torch.Tensor,
    w1_weight: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    w3_weight: torch.Tensor,
    w3_scale: torch.Tensor,
    swiglu_limit: float = 0.0,
    routing_weight: Optional[torch.Tensor] = None,
    block_size: int = FP4_SCALE_BLOCK,
) -> torch.Tensor:
    compute_dtype = hidden_states.dtype
    w1 = dequant_fp4_e2m1(w1_weight, w1_scale, compute_dtype, block_size)
    w3 = dequant_fp4_e2m1(w3_weight, w3_scale, compute_dtype, block_size)
    w2 = dequant_fp4_e2m1(w2_weight, w2_scale, compute_dtype, block_size)

    gate = F.linear(hidden_states, w1)
    up = F.linear(hidden_states, w3)

    if swiglu_limit and swiglu_limit > 0:
        gate = torch.clamp(gate.float(), max=swiglu_limit).to(gate.dtype)
        up = torch.clamp(up.float(), min=-swiglu_limit, max=swiglu_limit).to(
            up.dtype
        )

    activated = F.silu(gate.float()) * up.float()
    if routing_weight is not None:
        activated = activated * routing_weight

    out = F.linear(activated.to(compute_dtype), w2)
    return out
