"""CPU unit tests for the MiniMax-M3 perf-summary math (#221 phase 4).

These lock the percentile interpolation and the BF16-baseline summary
fields. They import no torch/model code; the module under test is pure.
"""

from __future__ import annotations

import pytest

from benchmarks.minimax_m3.perf_summary import (
    percentile,
    summarize_performance,
)


def test_percentile_matches_latency_benchmark_interpolation():
    values = [100.0, 200.0, 300.0]
    assert percentile(values, 0.0) == 100.0
    assert percentile(values, 50.0) == 200.0
    assert percentile(values, 99.0) == 298.0
    assert percentile(values, 100.0) == 300.0


def test_percentile_single_sample_returns_that_sample():
    assert percentile([42.0], 50.0) == 42.0
    assert percentile([42.0], 99.0) == 42.0


def test_percentile_unsorted_input_is_sorted_first():
    assert percentile([300.0, 100.0, 200.0], 50.0) == 200.0


def test_summarize_performance_reports_pinned_baseline_fields():
    summary = summarize_performance(
        [100.0, 200.0, 300.0],
        [10.0, 20.0, 30.0],
    )
    assert summary == {
        "sample_count": 3,
        "ttft_p50_ms": 200.0,
        "ttft_p99_ms": 298.0,
        "decode_tok_s_p50": 20.0,
        "decode_tok_s_p99": 29.8,
    }
    assert isinstance(summary["sample_count"], int)


def test_percentile_rejects_empty_samples():
    with pytest.raises(ValueError):
        percentile([], 50.0)


@pytest.mark.parametrize("bad_p", [-0.1, 100.1, float("nan")])
def test_percentile_rejects_out_of_range_p(bad_p):
    with pytest.raises(ValueError):
        percentile([1.0, 2.0, 3.0], bad_p)


@pytest.mark.parametrize(
    "bad",
    [float("nan"), float("inf"), float("-inf"), -1.0],
)
def test_percentile_rejects_non_finite_or_negative_samples(bad):
    with pytest.raises(ValueError):
        percentile([1.0, bad, 3.0], 50.0)


def test_summarize_rejects_empty_lists():
    with pytest.raises(ValueError):
        summarize_performance([], [])


def test_summarize_rejects_length_mismatch():
    with pytest.raises(ValueError):
        summarize_performance([1.0, 2.0], [1.0])


@pytest.mark.parametrize(
    "bad",
    [float("nan"), float("inf"), float("-inf"), -1.0],
)
def test_summarize_rejects_invalid_ttft_samples(bad):
    with pytest.raises(ValueError):
        summarize_performance([1.0, bad], [1.0, 2.0])


@pytest.mark.parametrize(
    "bad",
    [float("nan"), float("inf"), float("-inf"), -1.0],
)
def test_summarize_rejects_invalid_decode_samples(bad):
    with pytest.raises(ValueError):
        summarize_performance([1.0, 2.0], [1.0, bad])
