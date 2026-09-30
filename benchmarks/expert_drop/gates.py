"""Drop-on-miss latency and accuracy reporting helpers."""

from __future__ import annotations

import math

EVAL_MASS_BUDGETS = (0.05, 0.10, 0.20)


def latency_gate_passes(p99_off: float, p99_on: float) -> bool:
    """Return whether drop-on-miss cuts decode p99 latency by 20 percent."""
    if not all(
        math.isfinite(value) and value > 0 for value in (p99_off, p99_on)
    ):
        raise ValueError(
            "p99 latency values must be finite and greater than zero"
        )
    return p99_on <= 0.8 * p99_off


def accuracy_row(
    budget: float, nll_off: float, nll_on: float
) -> dict[str, float]:
    """Build an NLL comparison row without applying an accuracy threshold."""
    if not all(math.isfinite(value) for value in (budget, nll_off, nll_on)):
        raise ValueError("budget and NLL values must be finite")
    if nll_off <= 0:
        raise ValueError("nll_off must be greater than zero")
    return {
        "budget": budget,
        "nll_off": nll_off,
        "nll_on": nll_on,
        "relative_delta": (nll_on - nll_off) / nll_off,
    }
