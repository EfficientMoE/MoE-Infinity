# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

FP8_BLOCK = 128


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
