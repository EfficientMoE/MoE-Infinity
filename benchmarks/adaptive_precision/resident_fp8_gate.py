"""Validation gates for uniform FP8 expert residency measurements."""

from __future__ import annotations

import math
from numbers import Real

KERNEL_MATRIX = {
    "format": "fp8_e4m3_block128",
    "execution": "fp8_dequant_bf16_gemm",
    "block_size": 128,
    "output_dtype": "bfloat16",
    "architecture_floor": None,
    "online_tier_changes": False,
    "uniform_per_dispatch": True,
}


def resident_ratio_passes(resident_fp8: int, resident_bf16: int) -> bool:
    """Return whether uniform FP8 holds at least 1.8x BF16 residents."""
    if (
        type(resident_fp8) is not int
        or type(resident_bf16) is not int
        or resident_fp8 < 0
        or resident_bf16 <= 0
    ):
        raise ValueError(
            "resident counts must be integers with BF16 count above zero"
        )
    return resident_fp8 >= 1.8 * resident_bf16


def nll_gate_passes(nll_bf16: float, nll_fp8: float) -> bool:
    """Return whether FP8 NLL is at most one percent worse than BF16."""
    if (
        isinstance(nll_bf16, bool)
        or isinstance(nll_fp8, bool)
        or not isinstance(nll_bf16, Real)
        or not isinstance(nll_fp8, Real)
        or not math.isfinite(nll_bf16)
        or not math.isfinite(nll_fp8)
        or nll_bf16 <= 0
    ):
        raise ValueError("NLL values must be finite with BF16 NLL above zero")
    return (nll_fp8 - nll_bf16) / nll_bf16 <= 0.01
