from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.expert_io_microbench.nsys_parser import (
    PCIE_LANE_GBPS,
    VERDICT_DEFER,
    VERDICT_NO_GO,
    VERDICT_PROCEED,
    _apply_criterion,
    pcie_theoretical_gbps,
)


def test_no_go_via_low_transfer_fraction():
    verdict = _apply_criterion(
        t_transfer_ns=5_000_000,
        t_step_ns=100_000_000,
        util_pcie=0.95,
    )
    assert verdict == VERDICT_NO_GO


def test_no_go_via_low_pcie_utilisation():
    verdict = _apply_criterion(
        t_transfer_ns=30_000_000,
        t_step_ns=100_000_000,
        util_pcie=0.30,
    )
    assert verdict == VERDICT_NO_GO


def test_defer_in_between():
    verdict = _apply_criterion(
        t_transfer_ns=20_000_000,
        t_step_ns=100_000_000,
        util_pcie=0.60,
    )
    assert verdict == VERDICT_DEFER


def test_proceed_when_both_thresholds_cleared():
    verdict = _apply_criterion(
        t_transfer_ns=40_000_000,
        t_step_ns=100_000_000,
        util_pcie=0.85,
    )
    assert verdict == VERDICT_PROCEED


def test_no_go_on_zero_step():
    verdict = _apply_criterion(
        t_transfer_ns=10_000_000,
        t_step_ns=0,
        util_pcie=0.95,
    )
    assert verdict == VERDICT_NO_GO


def test_pcie_bandwidth_gen4_x16_a5000_class():
    bw = pcie_theoretical_gbps(link_width=16, link_gen=4)
    assert bw == pytest.approx(31.504)


def test_pcie_bandwidth_gen5_x16_h100_class():
    bw = pcie_theoretical_gbps(link_width=16, link_gen=5)
    assert bw == pytest.approx(63.008)


def test_pcie_bandwidth_unknown_gen_rejected():
    with pytest.raises(ValueError):
        pcie_theoretical_gbps(link_width=16, link_gen=99)


def test_pcie_lane_table_covers_modern_gens():
    for gen in (3, 4, 5):
        assert gen in PCIE_LANE_GBPS
