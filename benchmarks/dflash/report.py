"""Immutable result + cost-model contract for the PD-DFlash serving gate.

Task 1 of ``docs/superpowers/plans/2026-08-14-pd-dflash-serving-scheduler.md``.
Pure Python, imported by both the (later) GPU runner and the aggregator so the
schema is defined once. Two public entry points:

* ``evaluate_hide_inequality`` -- the design's §7 route-ahead hiding inequality
  ``(1 - r) * s * M / BW <= t_draft + t_router + overlap`` evaluated from
  *measured* terms only (never a theoretical PCIe bandwidth);
* ``validate_result_matrix`` -- enforces that a §8 result matrix carries every
  baseline (B0-B3) and every metric, permitting B3's explicit
  ``UNAVAILABLE_CAPACITY`` status when a resident upper bound does not fit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

REQUIRED_METRICS = (
    "output_tokens_per_second",
    "acceptance_length_a",
    "ttft_seconds",
    "per_round_latency_seconds",
    "goodput_at_slo",
    "expert_cache_hit_rate",
    "route_ahead_prefetch_coverage",
    "wasted_prefetch_bytes",
    "expert_occupancy_bytes",
    "kv_occupancy_bytes",
)

REQUIRED_BASELINES = ("B0", "B1", "B2", "B3")

UNAVAILABLE_CAPACITY = "UNAVAILABLE_CAPACITY"


@dataclass(frozen=True)
class HideInequality:
    """Both sides of the §7 route-ahead hiding inequality and its verdict."""

    resident_fraction: float
    saturation: float
    total_expert_bytes: float
    measured_h2d_bytes_per_second: float
    draft_seconds: float
    router_seconds: float
    overlap_seconds: float
    fetch_seconds: float
    hide_window_seconds: float
    hidden: bool


def _require_finite_non_negative(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and >= 0; got {value!r}")
    return number


def evaluate_hide_inequality(
    *,
    resident_fraction: float,
    saturation: float,
    total_expert_bytes: float,
    measured_h2d_bytes_per_second: float,
    draft_seconds: float,
    router_seconds: float,
    overlap_seconds: float,
) -> HideInequality:
    """Evaluate ``(1 - r) * s * M / BW <= t_draft + t_router + overlap``.

    Raises ``ValueError`` on a resident fraction outside ``[0, 1]``, a
    non-positive measured bandwidth, or any negative time/byte term.
    """
    r = float(resident_fraction)
    if not math.isfinite(r) or not 0.0 <= r <= 1.0:
        raise ValueError(
            f"resident_fraction must be in [0, 1]; got {resident_fraction!r}"
        )
    s = _require_finite_non_negative("saturation", saturation)
    if s > 1.0:
        raise ValueError(f"saturation must be in [0, 1]; got {saturation!r}")
    total = _require_finite_non_negative(
        "total_expert_bytes", total_expert_bytes
    )
    bandwidth = float(measured_h2d_bytes_per_second)
    if not math.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError(
            "measured_h2d_bytes_per_second must be > 0; "
            f"got {measured_h2d_bytes_per_second!r}"
        )
    draft = _require_finite_non_negative("draft_seconds", draft_seconds)
    router = _require_finite_non_negative("router_seconds", router_seconds)
    overlap = _require_finite_non_negative("overlap_seconds", overlap_seconds)

    fetch_seconds = (1.0 - r) * s * total / bandwidth
    hide_window_seconds = draft + router + overlap
    return HideInequality(
        resident_fraction=r,
        saturation=s,
        total_expert_bytes=total,
        measured_h2d_bytes_per_second=bandwidth,
        draft_seconds=draft,
        router_seconds=router,
        overlap_seconds=overlap,
        fetch_seconds=fetch_seconds,
        hide_window_seconds=hide_window_seconds,
        hidden=fetch_seconds <= hide_window_seconds,
    )


def validate_result_matrix(
    rows: Mapping[str, Mapping[str, object]],
) -> None:
    """Assert a §8 matrix has every baseline and every well-formed metric.

    B0-B3 must all be present. Each row must supply every ``REQUIRED_METRICS``
    entry as a finite, non-negative number, with one exception: a B3 row whose
    ``status`` equals ``UNAVAILABLE_CAPACITY`` (resident upper bound did not
    fit) is accepted without metrics. Raises ``ValueError`` otherwise.
    """
    missing = [b for b in REQUIRED_BASELINES if b not in rows]
    if missing:
        raise ValueError(f"missing baselines: {', '.join(missing)}")

    for baseline in REQUIRED_BASELINES:
        row = rows[baseline]
        if baseline == "B3" and row.get("status") == UNAVAILABLE_CAPACITY:
            continue
        for metric in REQUIRED_METRICS:
            if metric not in row:
                raise ValueError(f"{baseline} missing metric: {metric}")
            value = row[metric]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"{baseline}.{metric} must be a finite, non-negative "
                    f"number; got {value!r}"
                )
            number = float(value)
            if not math.isfinite(number) or number < 0.0:
                raise ValueError(
                    f"{baseline}.{metric} must be a finite, non-negative "
                    f"number; got {value!r}"
                )


__all__ = [
    "REQUIRED_METRICS",
    "REQUIRED_BASELINES",
    "UNAVAILABLE_CAPACITY",
    "HideInequality",
    "evaluate_hide_inequality",
    "validate_result_matrix",
]
