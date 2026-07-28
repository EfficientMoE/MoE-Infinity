# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from __future__ import annotations

import torch
import torch.nn.functional as F

from moe_infinity.utils.fp8 import FP8_BLOCK, dequant_fp8_blockwise


def fp8_shared_expert_forward(
    hidden_states: torch.Tensor,
    w1_weight: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    w3_weight: torch.Tensor,
    w3_scale: torch.Tensor,
    swiglu_limit: float = 0.0,
    block_size: int = FP8_BLOCK,
) -> torch.Tensor:
    compute_dtype = hidden_states.dtype
    w1 = dequant_fp8_blockwise(w1_weight, w1_scale, compute_dtype, block_size)
    w2 = dequant_fp8_blockwise(w2_weight, w2_scale, compute_dtype, block_size)
    w3 = dequant_fp8_blockwise(w3_weight, w3_scale, compute_dtype, block_size)

    gate = F.linear(hidden_states, w1).float()
    up = F.linear(hidden_states, w3).float()
    if swiglu_limit and swiglu_limit > 0:
        gate = torch.clamp(gate, max=swiglu_limit)
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
    activated = F.silu(gate) * up
    return F.linear(activated.to(compute_dtype), w2)
