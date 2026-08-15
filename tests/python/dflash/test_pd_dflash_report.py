"""CPU-only contract tests for the PD-DFlash serving experiment report.

Task 1 of ``docs/superpowers/plans/2026-08-14-pd-dflash-serving-scheduler.md``
("Freeze the experiment schema and cost-model decision"). These tests pin the
immutable result contract *before* any GPU runner exists:

* ``evaluate_hide_inequality`` computes the route-ahead hiding inequality from
  the design's §7 terms only -- never a theoretical PCIe number -- and reports
  both sides plus the boolean verdict;
* ``validate_result_matrix`` requires the full B0-B3 baseline set and every §8
  metric (permitting B3's explicit ``UNAVAILABLE_CAPACITY`` status), and treats
  ``wasted_prefetch_bytes`` as a byte quantity rather than an expert count.

All pure Python; no CUDA, no checkpoint, no network.
"""

from __future__ import annotations

import math

import pytest

from benchmarks.dflash.report import (
    REQUIRED_METRICS,
    evaluate_hide_inequality,
    validate_result_matrix,
)


def _full_matrix() -> dict[str, dict[str, float]]:
    return {
        baseline: {metric: 1.0 for metric in REQUIRED_METRICS}
        for baseline in ("B0", "B1", "B2", "B3")
    }


# ---------------------------------------------------------------------------
# hide inequality: measured terms only (design §7)
# ---------------------------------------------------------------------------


def test_hide_inequality_uses_measured_terms():
    result = evaluate_hide_inequality(
        resident_fraction=0.5,
        saturation=1.0,
        total_expert_bytes=14_000_000_000,
        measured_h2d_bytes_per_second=50_000_000_000,
        draft_seconds=0.04,
        router_seconds=0.01,
        overlap_seconds=0.10,
    )
    assert result.fetch_seconds == 0.14
    # 0.04 + 0.01 + 0.10 is 0.15000000000000002 in IEEE-754 double, so the
    # plan's literal ``== 0.15`` is asserted via approx (documented deviation).
    assert result.hide_window_seconds == pytest.approx(0.15)
    assert result.hidden is True


def test_hide_inequality_false_when_fetch_exceeds_window():
    result = evaluate_hide_inequality(
        resident_fraction=0.0,
        saturation=1.0,
        total_expert_bytes=14_000_000_000,
        measured_h2d_bytes_per_second=50_000_000_000,
        draft_seconds=0.04,
        router_seconds=0.01,
        overlap_seconds=0.10,
    )
    # fetch = 14e9 / 50e9 = 0.28 s > 0.15 s window -> exposed, not hidden.
    assert result.fetch_seconds == pytest.approx(0.28)
    assert result.hidden is False


def test_hide_inequality_rejects_impossible_terms():
    base = dict(
        resident_fraction=0.5,
        saturation=1.0,
        total_expert_bytes=14_000_000_000,
        measured_h2d_bytes_per_second=50_000_000_000,
        draft_seconds=0.04,
        router_seconds=0.01,
        overlap_seconds=0.10,
    )
    with pytest.raises(ValueError, match="resident_fraction"):
        evaluate_hide_inequality(**{**base, "resident_fraction": 1.5})
    with pytest.raises(ValueError, match="resident_fraction"):
        evaluate_hide_inequality(**{**base, "resident_fraction": -0.1})
    with pytest.raises(ValueError, match="measured_h2d_bytes_per_second"):
        evaluate_hide_inequality(**{**base, "measured_h2d_bytes_per_second": 0})
    with pytest.raises(ValueError, match="overlap_seconds"):
        evaluate_hide_inequality(**{**base, "overlap_seconds": -0.01})


# ---------------------------------------------------------------------------
# result matrix: B0-B3 present, every §8 metric present and well-formed
# ---------------------------------------------------------------------------


def test_result_matrix_requires_b0_through_b3_and_every_section8_metric():
    rows = _full_matrix()
    validate_result_matrix(rows)
    del rows["B2"]
    with pytest.raises(ValueError, match="missing baselines: B2"):
        validate_result_matrix(rows)


def test_result_matrix_requires_each_metric_present():
    rows = _full_matrix()
    del rows["B1"]["route_ahead_prefetch_coverage"]
    with pytest.raises(ValueError, match="route_ahead_prefetch_coverage"):
        validate_result_matrix(rows)


def test_result_matrix_rejects_non_finite_or_negative_metric():
    rows = _full_matrix()
    rows["B0"]["ttft_seconds"] = float("nan")
    with pytest.raises(ValueError, match="ttft_seconds"):
        validate_result_matrix(rows)

    rows = _full_matrix()
    rows["B0"]["wasted_prefetch_bytes"] = -1
    with pytest.raises(ValueError, match="wasted_prefetch_bytes"):
        validate_result_matrix(rows)


def test_b3_may_be_reported_unavailable_capacity():
    rows = _full_matrix()
    rows["B3"] = {"status": "UNAVAILABLE_CAPACITY"}
    validate_result_matrix(rows)


def test_b3_without_status_still_requires_every_metric():
    rows = _full_matrix()
    rows["B3"] = {"status": "OK"}
    with pytest.raises(ValueError, match="B3"):
        validate_result_matrix(rows)


def test_wasted_prefetch_is_bytes_not_expert_count():
    rows = _full_matrix()
    rows["B1"]["wasted_prefetch_bytes"] = 12_582_912
    validate_result_matrix(rows)
    assert rows["B1"]["wasted_prefetch_bytes"] == 12_582_912
    assert "wasted_prefetch_bytes" in REQUIRED_METRICS


def test_required_metrics_are_frozen_and_complete():
    assert isinstance(REQUIRED_METRICS, tuple)
    assert REQUIRED_METRICS == (
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
    assert len(set(REQUIRED_METRICS)) == len(REQUIRED_METRICS)
    assert math.isfinite(1.0)
