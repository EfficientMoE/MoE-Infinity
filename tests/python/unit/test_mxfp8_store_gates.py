# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.mxfp8_store.gates import host_ratio_passes, onboarding_row


def test_host_ratio_boundary_and_onboarding_row():
    assert host_ratio_passes(60.0, 100.0)
    assert not host_ratio_passes(60.0 + 1e-9, 100.0)
    row = onboarding_row(4, 6.0, 4.0)
    assert row["seconds_per_expert_mxfp8"] == 1.5
    assert row["seconds_per_expert_bf16"] == 1.0
    assert row["relative_delta"] == 0.5


def test_gate_inputs_reject_nonfinite_or_nonpositive():
    for args in [
        (0.0, 1.0),
        (1.0, 0.0),
        (float("nan"), 1.0),
        (1.0, float("inf")),
    ]:
        with pytest.raises(ValueError):
            host_ratio_passes(*args)
    for args in [(0, 1.0, 1.0), (1, 0.0, 1.0), (1, 1.0, float("nan"))]:
        with pytest.raises(ValueError):
            onboarding_row(*args)


def test_bench_help_exits_zero():
    script = Path("benchmarks/mxfp8_store/bench_mxfp8_store.py")
    assert (
        subprocess.run(
            [sys.executable, str(script), "--help"], check=False
        ).returncode
        == 0
    )
