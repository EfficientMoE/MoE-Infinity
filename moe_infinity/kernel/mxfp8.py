# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Reference MXFP8 dequantisation for checkpoint-native routed experts.

MXFP8 format (MiniMax-M3-MXFP8, issue #221 phase 5), OCP microscaling:
    - Payload: F8_E4M3 (``torch.float8_e4m3fn``), one value per element.
    - Scale:   U8 encoding E8M0; ``weight_block_size = [1, 32]`` means one
      scale per 32 payload elements along K (the last dim), shared across a
      single row.
    - E8M0 byte ``e`` encodes the power-of-two scale ``2**(e - 127)``.
    - Dequant: ``value * 2**(e - 127)``.

Because every scale is an exact power of two, the decode is bit-exact into any
float wider than E4M3 (e.g. bf16) absent overflow -- the model consumes the
bf16 result through the existing bf16 expert store.
"""

import torch


def mxfp8_dequantize(
    payload: torch.Tensor,
    scale_inv: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    block_size: int = 0,
) -> torch.Tensor:
    """Dequantise an MXFP8 payload/scale pair to ``dtype``.

    Args:
        payload:    ``[*, K]`` ``float8_e4m3fn`` weights.
        scale_inv:  ``[*, K // block_size]`` ``uint8`` E8M0 scales.
        dtype:      output dtype (bf16 for the routed-expert store).
        block_size: payload elements per scale (0 = auto-detect from shapes).

    Returns:
        ``[*, K]`` dequantised tensor in ``dtype``.
    """
    values = payload.to(torch.float32)
    exponents = scale_inv.to(torch.int32) - 127

    n_blocks = scale_inv.shape[-1]
    if block_size <= 0:
        block_size = values.shape[-1] // max(n_blocks, 1)

    expanded = (
        exponents.unsqueeze(-1)
        .expand(*exponents.shape, block_size)
        .reshape(*exponents.shape[:-1], n_blocks * block_size)
    )
    if expanded.shape[-1] > values.shape[-1]:
        expanded = expanded[..., : values.shape[-1]]

    return torch.ldexp(values, expanded).to(dtype)
