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
