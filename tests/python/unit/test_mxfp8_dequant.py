# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Recorded-group MXFP8 dequant round-trip (MiniMax-M3-MXFP8, [1,32] groups).

F8_E4M3 payload plus U8 E8M0 inverse scale per 32 K-elements. The engine
dequantises routed experts to bf16 at onboarding; this compares the bf16
output (promoted to float32) against an independent float32 decode within
rtol=1e-2, atol=1e-2.
"""

import torch

from moe_infinity.kernel.mxfp8 import mxfp8_dequantize

BLOCK = 32


def _reference_decode(payload, scale_inv, block):
    payload_f32 = payload.to(torch.float32)
    rows, cols = payload_f32.shape
    out = torch.empty_like(payload_f32)
    for r in range(rows):
        for c in range(cols):
            exponent = int(scale_inv[r, c // block].item()) - 127
            out[r, c] = payload_f32[r, c] * (2.0**exponent)
    return out


def test_recorded_group_roundtrip_to_bf16():
    rows, k = 4, 64  # two [1, 32] groups along K per row
    base = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    payload = base.repeat(rows, k // base.numel()).to(torch.float8_e4m3fn)
    # Distinct E8M0 bytes per group prove the per-32 grouping along K.
    scale_inv = torch.tensor(
        [[128, 126], [127, 129], [125, 127], [130, 124]],
        dtype=torch.uint8,
    )

    got = mxfp8_dequantize(
        payload, scale_inv, dtype=torch.bfloat16, block_size=BLOCK
    )

    assert got.dtype == torch.bfloat16
    assert got.shape == (rows, k)
    ref = _reference_decode(payload, scale_inv, BLOCK)
    assert torch.allclose(got.float(), ref, rtol=1e-2, atol=1e-2)
