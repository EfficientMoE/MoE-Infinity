# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.minimax_m3.gates import kernel_speedup_passes, speedup_row

BENCH_PATH = (
    Path(__file__).parents[3]
    / "benchmarks"
    / "minimax_m3"
    / "bench_indexer_attention.py"
)


def test_speedup_gate_is_exclusive_at_80_percent_boundary():
    assert kernel_speedup_passes(100.0, 80.0) is True
    assert kernel_speedup_passes(100.0, 80.0 + 2**-20) is False


def test_speedup_row_reports_exact_ratio():
    assert speedup_row(8192, 100.0, 75.0) == {
        "seq_len": 8192,
        "t_eager_ms": 100.0,
        "t_kernel_ms": 75.0,
        "speedup": 4.0 / 3.0,
    }


def test_gate_helpers_reject_nonpositive_and_nonfinite():
    bad_values = [0, 0.0, -1.0, float("nan"), float("inf"), float("-inf")]
    for bad in bad_values:
        with pytest.raises(ValueError):
            kernel_speedup_passes(bad, 80.0)
        with pytest.raises(ValueError):
            kernel_speedup_passes(100.0, bad)
        with pytest.raises(ValueError):
            speedup_row(bad, 100.0, 75.0)
        with pytest.raises(ValueError):
            speedup_row(8192, bad, 75.0)
        with pytest.raises(ValueError):
            speedup_row(8192, 100.0, bad)


def test_bench_help_exits_zero():
    result = subprocess.run(
        [sys.executable, str(BENCH_PATH), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
