#!/usr/bin/env python3
"""Manually benchmark drop-on-miss on a GPU without downloading weights.

For each value in ``EVAL_MASS_BUDGETS``, run the same model and cache budget
twice: once with ``expert_drop_policy=off`` and once with
``expert_drop_policy=on_miss``. Compare decode p99 with
``latency_gate_passes`` (the gate is ``p99_on <= 0.8 * p99_off``), and record
the NLL pair with ``accuracy_row``. This entry point does not assert an
accuracy threshold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.expert_drop.gates import latency_gate_passes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enforce-latency-gate", action="store_true")
    parser.add_argument("--p99-off", type=float)
    parser.add_argument("--p99-on", type=float)
    args = parser.parse_args()

    if not args.enforce_latency_gate:
        return 0
    if args.p99_off is None or args.p99_on is None:
        parser.error("--enforce-latency-gate requires --p99-off and --p99-on")

    passed = latency_gate_passes(args.p99_off, args.p99_on)
    print("pass" if passed else "fail")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
