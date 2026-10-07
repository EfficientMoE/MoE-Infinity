# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Pure speedup-gate helpers for the MiniMax-M3 indexer microbenchmark.

Kept free of torch/CUDA so the gate logic is unit-testable on CPU. The kernel
lands only when prefill is at least 20% faster than eager at the enforced
sequence length (``t_kernel_ms <= 0.8 * t_eager_ms``).
"""

from __future__ import annotations

import math

SPEEDUP_THRESHOLD = 0.8


def _require_positive_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"{name} must be a finite positive number, got {value!r}"
        )


def kernel_speedup_passes(t_eager_ms: float, t_kernel_ms: float) -> bool:
    _require_positive_finite("t_eager_ms", t_eager_ms)
    _require_positive_finite("t_kernel_ms", t_kernel_ms)
    return t_kernel_ms <= SPEEDUP_THRESHOLD * t_eager_ms


def speedup_row(seq_len: int, t_eager_ms: float, t_kernel_ms: float) -> dict:
    _require_positive_finite("seq_len", seq_len)
    _require_positive_finite("t_eager_ms", t_eager_ms)
    _require_positive_finite("t_kernel_ms", t_kernel_ms)
    return {
        "seq_len": seq_len,
        "t_eager_ms": t_eager_ms,
        "t_kernel_ms": t_kernel_ms,
        "speedup": t_eager_ms / t_kernel_ms,
    }
