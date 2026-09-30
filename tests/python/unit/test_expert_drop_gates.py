import pytest

from benchmarks.expert_drop.gates import (
    EVAL_MASS_BUDGETS,
    accuracy_row,
    latency_gate_passes,
)


def test_eval_budgets_are_the_three_curve_points():
    assert EVAL_MASS_BUDGETS == (0.05, 0.10, 0.20)


def test_latency_gate_is_twenty_percent():
    assert latency_gate_passes(100.0, 80.0) is True
    assert latency_gate_passes(100.0, 80.0 + 2**-20) is False


def test_accuracy_row_records_relative_delta_without_a_pass_flag():
    row = accuracy_row(0.10, nll_off=4.0, nll_on=5.0)
    assert row == {
        "budget": 0.10,
        "nll_off": 4.0,
        "nll_on": 5.0,
        "relative_delta": 0.25,
    }
    assert "passed" not in row


def test_accuracy_row_rejects_nonpositive_baseline():
    with pytest.raises(ValueError):
        accuracy_row(0.05, nll_off=0.0, nll_on=1.0)
