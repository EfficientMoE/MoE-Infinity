#!/usr/bin/env python3
"""
Test orchestrator for MoE-Infinity I/O integration tests.

1. Detects CUDA availability via torch.cuda.is_available()
2. Always runs Tier 1 tests (no CUDA): pytest -m "not cuda"
3. If CUDA available, runs Tier 2 tests: pytest -m "cuda"
4. Reports summary, exits non-zero on failure.
"""

import subprocess
import sys


def has_cuda():
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def run_pytest(path, marker_expr, label):
    """Run pytest on the given path with an optional marker expression."""
    cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short"]
    if marker_expr:
        cmd += ["-m", marker_expr]
    cmd.append(path)

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    cuda_available = has_cuda()

    # Always run Tier 1: I/O integration tests
    tier1_ok = run_pytest(
        "tests/docker/test_io_integration.py",
        "not cuda",
        "Tier 1: Threading & File I/O (no CUDA)",
    )

    # Always run Tier 1: unit tests (examples smoke, etc.)
    unit_ok = run_pytest(
        "tests/python/unit/",
        None,
        "Tier 1: Unit Tests",
    )

    # Run Tier 2 if CUDA is available
    tier2_ok = True
    if cuda_available:
        tier2_ok = run_pytest(
            "tests/docker/test_io_integration.py",
            "cuda",
            "Tier 2: Pinned Memory & Tensor I/O (CUDA)",
        )
    else:
        print("\nCUDA not available — skipping Tier 2 tests")

    # Summary
    print(f"\n{'=' * 60}")
    print("  Test Summary")
    print(f"{'=' * 60}")
    print(f"  Tier 1 (no CUDA): {'PASSED' if tier1_ok else 'FAILED'}")
    print(f"  Tier 1 (unit):    {'PASSED' if unit_ok else 'FAILED'}")
    if cuda_available:
        print(f"  Tier 2 (CUDA):    {'PASSED' if tier2_ok else 'FAILED'}")
    else:
        print("  Tier 2 (CUDA):    SKIPPED")

    all_ok = tier1_ok and unit_ok and tier2_ok
    print(f"\n  Overall: {'PASSED' if all_ok else 'FAILED'}")
    print(f"{'=' * 60}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
