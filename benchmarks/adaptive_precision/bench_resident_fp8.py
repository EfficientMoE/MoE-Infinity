"""Manually compare one BF16 store and one uniform FP8 store under the same
HBM budget, then apply ``resident_ratio_passes`` and ``nll_gate_passes`` to
the measured resident counts and NLL values. This script does not download
weights; run the GPU comparison separately and provide its measurements.
"""

from __future__ import annotations

import argparse

from resident_fp8_gate import nll_gate_passes, resident_ratio_passes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enforce", action="store_true")
    parser.add_argument("--resident-fp8", type=int)
    parser.add_argument("--resident-bf16", type=int)
    parser.add_argument("--nll-fp8", type=float)
    parser.add_argument("--nll-bf16", type=float)
    args = parser.parse_args()

    if not args.enforce:
        return 0

    required = {
        "--resident-fp8": args.resident_fp8,
        "--resident-bf16": args.resident_bf16,
        "--nll-fp8": args.nll_fp8,
        "--nll-bf16": args.nll_bf16,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"--enforce requires {', '.join(missing)}")

    passes = resident_ratio_passes(
        args.resident_fp8, args.resident_bf16
    ) and nll_gate_passes(args.nll_bf16, args.nll_fp8)
    print("pass" if passes else "fail")
    return 0 if passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
