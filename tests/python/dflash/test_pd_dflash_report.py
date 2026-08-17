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
    aggregate_result_matrices,
    evaluate_hide_inequality,
    evaluate_matrix,
    summarise_row,
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


# ---------------------------------------------------------------------------
# BM1 router-ahead cost aggregation (design §10)
# ---------------------------------------------------------------------------


def complete_row(**overrides) -> dict[str, float]:
    row: dict[str, float] = {metric: 1.0 for metric in REQUIRED_METRICS}
    row["t_router_seconds"] = 0.002
    row["t_verify_seconds"] = 0.020
    row.update(overrides)
    return row


def test_bm1_reports_router_cost_against_verify():
    row = complete_row(t_router_seconds=0.002, t_verify_seconds=0.020)
    summary = summarise_row(row)
    assert summary["bm1_router_to_verify_ratio"] == 0.1
    assert summary["bm1_pass"] is True


def test_bm1_fails_when_router_not_cheaper_than_verify():
    summary = summarise_row(
        complete_row(t_router_seconds=0.03, t_verify_seconds=0.02)
    )
    assert summary["bm1_pass"] is False
    assert summary["bm1_router_to_verify_ratio"] == 1.5


def test_bm1_retains_raw_terms_and_rejects_bad_inputs():
    summary = summarise_row(complete_row())
    assert summary["t_router_seconds"] == 0.002
    assert summary["t_verify_seconds"] == 0.020
    with pytest.raises(ValueError, match="t_verify_seconds"):
        summarise_row(complete_row(t_verify_seconds=0.0))
    with pytest.raises(ValueError, match="t_router_seconds"):
        summarise_row(complete_row(t_router_seconds=-0.1))
    with pytest.raises(ValueError, match="t_router_seconds"):
        summarise_row({"t_verify_seconds": 0.02})


# ---------------------------------------------------------------------------
# aggregation: group raw rows into §8 matrices, allow blocked B2
# ---------------------------------------------------------------------------


def _obs_row(baseline: str, **overrides) -> dict[str, object]:
    row: dict[str, object] = {
        "model": "M",
        "draft": "d",
        "baseline": baseline,
        "block_size": 16,
        "concurrency": 8,
        "repeat": 0,
    }
    row.update({metric: 1.0 for metric in REQUIRED_METRICS})
    row.update(overrides)
    return row


def _blocked_b2() -> dict[str, object]:
    return {
        "model": "M",
        "draft": "d",
        "baseline": "B2",
        "block_size": 16,
        "concurrency": 8,
        "repeat": 0,
        "status": "BLOCKED_UNTIL_2D_SCHEDULER",
    }


def test_aggregate_groups_rows_and_rejects_duplicate_baseline():
    matrices = aggregate_result_matrices(
        [_obs_row(b) for b in ("B0", "B1", "B3")]
    )
    assert set(matrices) == {("M", 16, 8)}
    assert set(matrices[("M", 16, 8)]) == {"B0", "B1", "B3"}
    with pytest.raises(ValueError, match="duplicate baseline"):
        aggregate_result_matrices([_obs_row("B0"), _obs_row("B0")])


def test_evaluate_matrix_allows_blocked_b2_only_when_permitted():
    rows = {b: _obs_row(b) for b in ("B0", "B1", "B3")}
    rows["B2"] = _blocked_b2()
    ok, detail = evaluate_matrix(rows, allow_blocked=["B2"])
    assert ok is True and detail["blocked"] == ["B2"]
    not_ok, detail2 = evaluate_matrix(rows, allow_blocked=[])
    assert not_ok is False and "B2" in detail2["blocked"]


def test_evaluate_matrix_flags_missing_baseline_and_attaches_bm1():
    partial, missing = evaluate_matrix(
        {b: _obs_row(b) for b in ("B0", "B1")}, []
    )
    assert partial is False and set(missing["missing"]) == {"B2", "B3"}

    full = {
        b: _obs_row(b, t_router_seconds=0.002, t_verify_seconds=0.020)
        for b in ("B0", "B1", "B2", "B3")
    }
    ok, detail = evaluate_matrix(full, [])
    assert ok is True
    assert detail["bm1"]["B1"]["bm1_pass"] is True


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


# ===========================================================================
# BM5 equivalence + final C++-hop keep/remove verdicts (design §10, Task 10).
#
# BM5: switching from Python per-expert issue to shipped C++ batched issue must
# not alter acceptance, route-ahead coverage, wasted bytes, or output tokens --
# only tokens/s may move. The final gate reports keep/remove per C++ hop from
# each hop's paired BM ship flag; a hop lacking a passing BM is removed.
# ===========================================================================

from benchmarks.dflash.report import (  # noqa: E402
    cpp_hop_verdicts,
    summarise_bm5_equivalence,
)


def _bm5_row(tokens_per_second):
    return {
        "acceptance_length_a": 3.0,
        "route_ahead_prefetch_coverage": 0.75,
        "wasted_prefetch_bytes": 12_582_912.0,
        "output_tokens_per_second": tokens_per_second,
    }


def test_bm5_equivalence_holds_when_only_throughput_moves():
    summary = summarise_bm5_equivalence(
        python_row=_bm5_row(100.0), cpp_row=_bm5_row(140.0)
    )
    assert summary["equivalent"] is True
    assert summary["python_tokens_per_second"] == 100.0
    assert summary["cpp_tokens_per_second"] == 140.0


def test_bm5_equivalence_fails_when_coverage_changes():
    cpp = _bm5_row(140.0)
    cpp["route_ahead_prefetch_coverage"] = 0.50
    summary = summarise_bm5_equivalence(python_row=_bm5_row(100.0), cpp_row=cpp)
    assert summary["equivalent"] is False
    assert "route_ahead_prefetch_coverage" in summary["mismatched"]


def test_bm5_equivalence_fails_when_waste_bytes_change():
    cpp = _bm5_row(140.0)
    cpp["wasted_prefetch_bytes"] = 0.0
    summary = summarise_bm5_equivalence(python_row=_bm5_row(100.0), cpp_row=cpp)
    assert summary["equivalent"] is False
    assert "wasted_prefetch_bytes" in summary["mismatched"]


def test_cpp_hop_verdicts_keep_only_hops_with_passing_bm():
    verdicts = cpp_hop_verdicts(
        bm2={"ship_batched": True},
        bm3={"ship_priority_band": False},
    )
    assert verdicts["batched_issuance"]["keep"] is True
    assert verdicts["priority_band"]["keep"] is False


def test_cpp_hop_verdicts_missing_bm_removes_hop():
    verdicts = cpp_hop_verdicts(bm2=None, bm3=None)
    assert verdicts["batched_issuance"]["keep"] is False
    assert verdicts["priority_band"]["keep"] is False
    assert verdicts["batched_issuance"]["reason"]
