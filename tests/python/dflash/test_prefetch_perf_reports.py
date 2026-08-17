# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""CPU-only decision-rule tests for the benchmark-gated prefetch reports.

Task 7 of ``docs/superpowers/plans/2026-08-14-pd-dflash-serving-scheduler.md``
(the BM2 issuance decision rule, design §10). These tests are pure: they never
import torch, load a checkpoint, or touch CUDA. They exercise only the
``bm2_decision`` rule, the percentile helper, and the report schema that the
GPU issuance micro-bench (``benchmarks.dflash.bench_prefetch_issuance``) emits,
so the ship gate for the batched ``prefetch_tensors`` C++ hop is verifiable
entirely off-hardware.

The gate rule (design §10 / plan Task 7 Step 1):

* ``candidate_required`` iff the current Python per-expert issuance median
  exceeds the route-ahead window ``t_draft + t_router``;
* ``ship_batched`` iff a candidate is required *and* the batched-pybind median
  is at or below that same window.

Unavailable candidate medians are reported as ``None`` (JSON ``null``), never
zero, so a missing mode can never masquerade as an infinitely fast candidate.
"""

from __future__ import annotations

import pytest

from benchmarks.dflash.bench_prefetch_issuance import (
    BATCHED_PYBIND,
    CPP_INTERNAL,
    PYTHON_PER_EXPERT,
    Bm2Decision,
    bm2_decision,
    build_bm2_report,
    percentiles_us,
)
from benchmarks.dflash.bench_prefetch_priority import (
    BACKGROUND,
    ON_DEMAND,
    PRIORITY_BANDS,
    ROUTE_AHEAD,
    Bm3Decision,
    PriorityArm,
    bm3_decision,
    build_bm3_report,
    median,
    priority_arm,
)


def test_bm2_candidate_required_when_per_expert_exceeds_window():
    assert bm2_decision(900.0, None, None, 500.0).candidate_required is True


def test_bm2_no_candidate_when_per_expert_within_window():
    assert bm2_decision(400.0, None, None, 500.0).candidate_required is False


def test_bm2_ship_batched_true_when_batched_within_window():
    assert bm2_decision(900.0, 300.0, 250.0, 500.0).ship_batched is True


def test_bm2_ship_batched_false_when_batched_exceeds_window():
    assert bm2_decision(900.0, 700.0, 650.0, 500.0).ship_batched is False


def test_bm2_decision_is_frozen_and_reports_all_medians():
    decision = bm2_decision(900.0, 300.0, 250.0, 500.0)
    assert isinstance(decision, Bm2Decision)
    assert decision.per_expert_us == 900.0
    assert decision.batched_us == 300.0
    assert decision.cpp_internal_us == 250.0
    assert decision.window_us == 500.0
    with pytest.raises(Exception):
        decision.per_expert_us = 1.0  # type: ignore[misc]


def test_bm2_ship_requires_a_measured_batched_median():
    # A candidate is required but no batched candidate exists yet: cannot ship.
    decision = bm2_decision(900.0, None, None, 500.0)
    assert decision.candidate_required is True
    assert decision.ship_batched is False


def test_bm2_never_ships_without_a_candidate_even_if_batched_is_fast():
    # Per-expert already within the window -> no candidate -> never ship,
    # even when a batched median would trivially satisfy the window.
    decision = bm2_decision(400.0, 100.0, 90.0, 500.0)
    assert decision.candidate_required is False
    assert decision.ship_batched is False


def test_bm2_missing_per_expert_median_is_not_a_candidate():
    decision = bm2_decision(None, 100.0, 90.0, 500.0)
    assert decision.candidate_required is False
    assert decision.ship_batched is False


def test_bm2_batched_exactly_at_window_ships():
    # "<= window" is inclusive at the boundary.
    assert bm2_decision(900.0, 500.0, None, 500.0).ship_batched is True


@pytest.mark.parametrize("bad_window", [0.0, -1.0, float("nan"), float("inf")])
def test_bm2_window_must_be_finite_and_positive(bad_window):
    with pytest.raises(ValueError):
        bm2_decision(900.0, 300.0, 250.0, bad_window)


@pytest.mark.parametrize("bad_value", [-1.0, float("nan"), float("inf")])
def test_bm2_negative_or_nonfinite_medians_rejected(bad_value):
    with pytest.raises(ValueError):
        bm2_decision(bad_value, None, None, 500.0)


def test_bm2_percentiles_us_from_nanoseconds_nearest_rank():
    # 1..100 microseconds expressed in nanoseconds.
    samples_ns = [i * 1000 for i in range(1, 101)]
    percentiles = percentiles_us(samples_ns)
    assert percentiles["p50"] == pytest.approx(50.0)
    assert percentiles["p90"] == pytest.approx(90.0)
    assert percentiles["p99"] == pytest.approx(99.0)
    assert percentiles["count"] == 100


def test_bm2_percentiles_us_requires_samples():
    with pytest.raises(ValueError):
        percentiles_us([])


def test_bm2_report_marks_unavailable_candidate_modes_null():
    report = build_bm2_report(
        model="tiny/fixture",
        saturated_tensor_count=6144,
        window_us=500.0,
        per_expert_samples_ns=[900_000] * 32,
        batched_samples_ns=None,
        cpp_internal_samples_ns=None,
        warmup=20,
        iterations=200,
    )
    assert report["benchmark"] == "BM2"
    assert report["saturated_tensor_count"] == 6144
    assert report["window_us"] == 500.0
    assert report["warmup"] == 20
    assert report["iterations"] == 200
    assert report["modes"][PYTHON_PER_EXPERT]["p50"] == pytest.approx(900.0)
    # Unavailable candidate modes are null, never zero.
    assert report["modes"][BATCHED_PYBIND] is None
    assert report["modes"][CPP_INTERNAL] is None
    assert report["medians_us"][BATCHED_PYBIND] is None
    assert report["candidate_required"] is True
    assert report["ship_batched"] is False


def test_bm2_report_ships_when_batched_mode_present_and_fast():
    report = build_bm2_report(
        model="tiny/fixture",
        saturated_tensor_count=6144,
        window_us=500.0,
        per_expert_samples_ns=[900_000] * 32,
        batched_samples_ns=[300_000] * 32,
        cpp_internal_samples_ns=[250_000] * 32,
        warmup=20,
        iterations=200,
    )
    assert report["modes"][BATCHED_PYBIND]["p50"] == pytest.approx(300.0)
    assert report["medians_us"][PYTHON_PER_EXPERT] == pytest.approx(900.0)
    assert report["candidate_required"] is True
    assert report["ship_batched"] is True


# ===========================================================================
# BM3 -- route-ahead priority-band ablation decision rule (design §10, Task 9).
#
# A three-way ablation over median *exposed-fetch seconds* and *tokens/s*:
#   default background (prio 2), dedicated route-ahead band (prio 1), and
#   on-demand (prio 0). The dedicated route-ahead band ships iff all three
#   hold:
#     (1) route-ahead has *lower* exposed fetch than default background;
#     (2) route-ahead does not *reduce* tokens/s vs default background; and
#     (3) on-demand remains the fastest service class (no priority inversion).
# All arms are pure medians, so the ship gate is verifiable off-hardware.
# ===========================================================================


def _arm(exposed_fetch_seconds: float, tokens_per_second: float) -> PriorityArm:
    return priority_arm(
        exposed_fetch_seconds=exposed_fetch_seconds,
        tokens_per_second=tokens_per_second,
    )


def test_bm3_ships_when_route_ahead_helps_without_inversion_or_regression():
    decision = bm3_decision(
        default=_arm(0.020, 100.0),
        route_ahead=_arm(0.012, 105.0),
        on_demand=_arm(0.008, 106.0),
    )
    assert isinstance(decision, Bm3Decision)
    assert decision.exposed_fetch_improved is True
    assert decision.throughput_preserved is True
    assert decision.on_demand_fastest is True
    assert decision.ship_priority_band is True


def test_bm3_no_improvement_does_not_ship():
    # Route-ahead exposed fetch is not lower than default background.
    decision = bm3_decision(
        default=_arm(0.012, 100.0),
        route_ahead=_arm(0.012, 100.0),
        on_demand=_arm(0.008, 101.0),
    )
    assert decision.exposed_fetch_improved is False
    assert decision.ship_priority_band is False


def test_bm3_throughput_regression_does_not_ship():
    # Route-ahead lowers exposed fetch but *reduces* tokens/s vs background.
    decision = bm3_decision(
        default=_arm(0.020, 100.0),
        route_ahead=_arm(0.012, 97.0),
        on_demand=_arm(0.008, 101.0),
    )
    assert decision.exposed_fetch_improved is True
    assert decision.throughput_preserved is False
    assert decision.ship_priority_band is False


def test_bm3_priority_inversion_does_not_ship():
    # On-demand is no longer the fastest class (route-ahead starves it).
    decision = bm3_decision(
        default=_arm(0.020, 100.0),
        route_ahead=_arm(0.010, 105.0),
        on_demand=_arm(0.014, 103.0),
    )
    assert decision.exposed_fetch_improved is True
    assert decision.throughput_preserved is True
    assert decision.on_demand_fastest is False
    assert decision.ship_priority_band is False


def test_bm3_throughput_preserved_is_inclusive_at_equality():
    decision = bm3_decision(
        default=_arm(0.020, 100.0),
        route_ahead=_arm(0.012, 100.0),
        on_demand=_arm(0.008, 100.0),
    )
    assert decision.throughput_preserved is True
    assert decision.on_demand_fastest is True
    assert decision.ship_priority_band is True


def test_bm3_decision_is_frozen():
    decision = bm3_decision(
        default=_arm(0.020, 100.0),
        route_ahead=_arm(0.012, 105.0),
        on_demand=_arm(0.008, 106.0),
    )
    with pytest.raises(Exception):
        decision.ship_priority_band = False  # type: ignore[misc]


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_bm3_arm_rejects_nonfinite_or_negative_exposed_fetch(bad):
    with pytest.raises(ValueError):
        priority_arm(exposed_fetch_seconds=bad, tokens_per_second=100.0)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_bm3_arm_rejects_nonfinite_or_negative_tokens_per_second(bad):
    with pytest.raises(ValueError):
        priority_arm(exposed_fetch_seconds=0.01, tokens_per_second=bad)


def test_bm3_median_nearest_rank_odd_and_even():
    assert median([0.01, 0.03, 0.02]) == pytest.approx(0.02)
    # even count -> mean of the two central order statistics
    assert median([0.01, 0.02, 0.03, 0.06]) == pytest.approx(0.025)


def test_bm3_median_requires_samples():
    with pytest.raises(ValueError):
        median([])


def test_bm3_priority_bands_are_named_and_ordered_high_to_low_service():
    # On-demand (0) is serviced first, then route-ahead (1), then background (2).
    assert ON_DEMAND == 0
    assert ROUTE_AHEAD == 1
    assert BACKGROUND == 2
    assert PRIORITY_BANDS == ("background", "route-ahead", "on-demand")


def test_bm3_report_assembles_medians_and_ship_verdict():
    report = build_bm3_report(
        models=["tiny/fixture"],
        arms={
            "background": {
                "exposed_fetch_seconds": [0.021, 0.020, 0.019],
                "tokens_per_second": [99.0, 100.0, 101.0],
            },
            "route-ahead": {
                "exposed_fetch_seconds": [0.013, 0.012, 0.011],
                "tokens_per_second": [104.0, 105.0, 106.0],
            },
            "on-demand": {
                "exposed_fetch_seconds": [0.009, 0.008, 0.007],
                "tokens_per_second": [105.0, 106.0, 107.0],
            },
        },
    )
    assert report["benchmark"] == "BM3"
    assert report["medians"]["route-ahead"]["exposed_fetch_seconds"] == (
        pytest.approx(0.012)
    )
    assert report["exposed_fetch_improved"] is True
    assert report["throughput_preserved"] is True
    assert report["on_demand_fastest"] is True
    assert report["ship_priority_band"] is True


def test_bm3_report_requires_all_three_arms():
    with pytest.raises(ValueError):
        build_bm3_report(
            models=["tiny/fixture"],
            arms={
                "background": {
                    "exposed_fetch_seconds": [0.02],
                    "tokens_per_second": [100.0],
                },
                "route-ahead": {
                    "exposed_fetch_seconds": [0.01],
                    "tokens_per_second": [100.0],
                },
            },
        )


# ===========================================================================
# BM4 -- H2D/compute overlap from nsys/CUPTI intervals (design §10, Task 10).
#
# Ground truth for whether expert fetch is hidden: the fraction of expert-H2D
# *bytes* overlapped with the draft/router/verify compute NVTX ranges. Bytes on
# a partially overlapped memcpy are apportioned by overlapped-duration fraction
# (never a whole-copy all-or-nothing). Pure interval arithmetic -> off-hardware.
# ===========================================================================

from benchmarks.dflash.parse_overlap import (  # noqa: E402
    Memcpy,
    NvtxRange,
    compute_overlap,
)

COMPUTE_RANGES = ("dflash_draft", "route_ahead_router", "target_verify")


def test_bm4_partial_overlap_apportions_bytes_by_duration():
    # One 100-byte memcpy over [0,10]; a compute range covers [0,8] -> 80% of
    # the duration overlaps -> 80 of 100 bytes are hidden.
    memcpys = [Memcpy(start=0.0, end=10.0, bytes=100)]
    ranges = [NvtxRange(name="dflash_draft", start=0.0, end=8.0)]
    result = compute_overlap(memcpys, ranges, COMPUTE_RANGES)
    assert result.total_h2d_bytes == 100
    assert result.overlapped_h2d_bytes == pytest.approx(80.0)
    assert result.overlap_fraction == pytest.approx(0.8)


def test_bm4_disjoint_ranges_hide_nothing():
    memcpys = [Memcpy(start=0.0, end=10.0, bytes=100)]
    ranges = [NvtxRange(name="target_verify", start=20.0, end=30.0)]
    result = compute_overlap(memcpys, ranges, COMPUTE_RANGES)
    assert result.total_h2d_bytes == 100
    assert result.overlapped_h2d_bytes == pytest.approx(0.0)
    assert result.overlap_fraction == pytest.approx(0.0)


def test_bm4_fully_overlapped_hides_everything():
    memcpys = [Memcpy(start=2.0, end=6.0, bytes=64)]
    ranges = [NvtxRange(name="route_ahead_router", start=0.0, end=10.0)]
    result = compute_overlap(memcpys, ranges, COMPUTE_RANGES)
    assert result.overlapped_h2d_bytes == pytest.approx(64.0)
    assert result.overlap_fraction == pytest.approx(1.0)


def test_bm4_zero_bytes_is_defined_and_not_a_divide_by_zero():
    result = compute_overlap([], [], COMPUTE_RANGES)
    assert result.total_h2d_bytes == 0
    assert result.overlapped_h2d_bytes == pytest.approx(0.0)
    assert result.overlap_fraction == pytest.approx(0.0)


def test_bm4_multiple_compute_ranges_union_not_double_counted():
    # Two overlapping compute ranges cover [0,4] and [2,10] -> union [0,10]
    # fully covers a 50-byte copy over [0,10]; overlap must not exceed 100%.
    memcpys = [Memcpy(start=0.0, end=10.0, bytes=50)]
    ranges = [
        NvtxRange(name="dflash_draft", start=0.0, end=4.0),
        NvtxRange(name="target_verify", start=2.0, end=10.0),
    ]
    result = compute_overlap(memcpys, ranges, COMPUTE_RANGES)
    assert result.overlapped_h2d_bytes == pytest.approx(50.0)
    assert result.overlap_fraction == pytest.approx(1.0)


def test_bm4_only_named_compute_ranges_count():
    # A range whose name is not a compute range (e.g. expert_h2d itself) does
    # not count as compute that hides the fetch.
    memcpys = [Memcpy(start=0.0, end=10.0, bytes=100)]
    ranges = [NvtxRange(name="expert_h2d", start=0.0, end=10.0)]
    result = compute_overlap(memcpys, ranges, COMPUTE_RANGES)
    assert result.overlapped_h2d_bytes == pytest.approx(0.0)


def test_bm4_per_memcpy_fraction_reported():
    memcpys = [
        Memcpy(start=0.0, end=10.0, bytes=100),
        Memcpy(start=0.0, end=10.0, bytes=100),
    ]
    ranges = [NvtxRange(name="dflash_draft", start=0.0, end=5.0)]
    result = compute_overlap(memcpys, ranges, COMPUTE_RANGES)
    assert result.total_h2d_bytes == 200
    assert result.overlapped_h2d_bytes == pytest.approx(100.0)
    assert result.overlap_fraction == pytest.approx(0.5)
