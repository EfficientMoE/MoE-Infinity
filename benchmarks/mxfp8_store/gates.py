# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Host-store gates for the MXFP8 expert store (#221 phase 5).

``host_ratio_passes`` is the single hard gate: the MXFP8 routed host-store
must be at most 0.6x the BF16 baseline. The F8_E4M3 payload plus 1/32 E8M0
scales is about 0.53x BF16; 0.6 leaves alignment/headroom. ``onboarding_row``
records per-expert onboarding latency for the PR but is never gated, because
dequant overhead is expected in this phase.
"""

import math

HOST_RATIO_GATE = 0.6


def _positive_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return float(value)


def host_ratio_passes(host_bytes_mxfp8: float, host_bytes_bf16: float) -> bool:
    mxfp8 = _positive_finite(host_bytes_mxfp8, "host_bytes_mxfp8")
    bf16 = _positive_finite(host_bytes_bf16, "host_bytes_bf16")
    return mxfp8 <= HOST_RATIO_GATE * bf16


def onboarding_row(
    n_experts: int, seconds_mxfp8: float, seconds_bf16: float
) -> dict:
    if isinstance(n_experts, bool) or not isinstance(n_experts, int):
        raise ValueError(f"n_experts must be an int, got {n_experts!r}")
    if n_experts <= 0:
        raise ValueError(f"n_experts must be > 0, got {n_experts!r}")
    mxfp8 = _positive_finite(seconds_mxfp8, "seconds_mxfp8")
    bf16 = _positive_finite(seconds_bf16, "seconds_bf16")
    return {
        "n_experts": n_experts,
        "seconds_mxfp8": mxfp8,
        "seconds_bf16": bf16,
        "seconds_per_expert_mxfp8": mxfp8 / n_experts,
        "seconds_per_expert_bf16": bf16 / n_experts,
        "relative_delta": (mxfp8 - bf16) / bf16,
    }
