"""Benchmark the accelerated cache-simulator paths against the reference.

Measures wall-clock and peak memory for building the miss-ratio curve on a
large *synthetic* trace, comparing:

* ``reference``  -- pure-Python :func:`belady_simulate` once per capacity.
* ``vectorized`` -- :func:`opt_mrc` (shared next-use precompute, heap sim).
* ``parallel``   -- :func:`opt_mrc_parallel` (per-capacity across processes).
* ``lru``        -- exact one-pass LRU MRC (whole curve in one pass).
* ``lru_gpu``    -- LRU MRC with the histogram reduced on GPU (if available).
* ``shards``     -- approximate LRU MRC via 1% SHARDS sampling.

All numbers are synthetic. The script prints the exact command context and the
detected hardware so results are reproducible.

Example::

    python -m benchmarks.cache_simulator.research.bench_speed \
        --accesses 10000000 --seed 0
"""

from __future__ import annotations

import argparse
import platform
import resource
import time
from typing import Callable, Dict, List, Tuple

import numpy as np

from benchmarks.cache_simulator.belady_cache_simulator import belady_simulate
from benchmarks.cache_simulator.research.mrc import (
    encode_accesses,
    lru_mrc,
    lru_mrc_gpu,
    opt_mrc,
    opt_mrc_parallel,
    shards_lru_mrc,
)
from benchmarks.cache_simulator.research.synthetic import synthetic_accesses

DEFAULT_CAPS = [128, 256, 512, 1024, 2048, 4096]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accesses", type=int, default=10_000_000)
    parser.add_argument("--num-layers", type=int, default=94)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--capacities",
        type=int,
        nargs="+",
        default=DEFAULT_CAPS,
    )
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="Skip the slow pure-Python reference (for very large traces).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="Processes for the parallel OPT path (0 => executor default).",
    )
    parser.add_argument("--markdown", action="store_true")
    return parser.parse_args()


def _hardware() -> str:
    bits = [
        f"python={platform.python_version()}",
        f"cpu={platform.processor()}",
    ]
    try:
        import os

        bits.append(f"cores={os.cpu_count()}")
    except Exception:  # pragma: no cover
        pass
    try:
        import torch

        bits.append(f"torch={torch.__version__}")
        if torch.cuda.is_available():
            bits.append(f"gpu={torch.cuda.get_device_name(0)}")
        else:
            bits.append("gpu=none")
    except ImportError:  # pragma: no cover
        bits.append("torch=absent")
    return ", ".join(bits)


def _maxrss_mb() -> float:
    # ru_maxrss is kilobytes on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _timed(fn: Callable[[], Dict[int, float]]) -> Tuple[float, float, Dict]:
    """Time ``fn`` and report the process peak-RSS delta (MB).

    Peak RSS is read via ``getrusage`` (near-zero overhead) rather than
    ``tracemalloc``, whose per-allocation hooks would dominate the 10M-element
    pure-Python heap loop and distort the timing.
    """

    rss_before = _maxrss_mb()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    rss_peak = _maxrss_mb()
    return elapsed, max(rss_peak - rss_before, 0.0), result


def _reference_curve(accesses, caps) -> Dict[int, float]:
    total = len(accesses)
    curve = {}
    for cap in caps:
        hits, _, _ = belady_simulate(accesses, cap)
        curve[cap] = (hits / total) if total else 0.0
    return curve


def main() -> None:
    args = parse_args()
    caps = list(args.capacities)

    print(f"[hardware] {_hardware()}")
    print(
        f"[synthetic] accesses={args.accesses} layers={args.num_layers} "
        f"experts={args.num_experts} seed={args.seed} capacities={caps}"
    )

    t0 = time.perf_counter()
    accesses = synthetic_accesses(
        args.accesses,
        num_layers=args.num_layers,
        num_experts=args.num_experts,
        seed=args.seed,
    )
    encoded = encode_accesses(np.asarray(accesses), args.num_experts)
    print(
        f"generated {len(accesses)} synthetic accesses in "
        f"{time.perf_counter() - t0:.2f}s"
    )

    rows: List[Tuple[str, float, float, Dict[int, float]]] = []

    if not args.skip_reference:
        el, pk, cv = _timed(lambda: _reference_curve(accesses, caps))
        rows.append(("reference (pure-Python)", el, pk, cv))

    el, pk, cv = _timed(
        lambda: opt_mrc(encoded, caps, num_experts=args.num_experts)
    )
    rows.append(("vectorized opt_mrc", el, pk, cv))

    el, pk, cv = _timed(
        lambda: opt_mrc_parallel(
            encoded,
            caps,
            num_experts=args.num_experts,
            max_workers=args.max_workers,
        )
    )
    rows.append(("parallel opt_mrc", el, pk, cv))

    el, pk, cv = _timed(
        lambda: lru_mrc(encoded, caps, num_experts=args.num_experts)
    )
    rows.append(("lru one-pass", el, pk, cv))

    el, pk, cv = _timed(
        lambda: lru_mrc_gpu(encoded, caps, num_experts=args.num_experts)
    )
    rows.append(("lru one-pass (gpu)", el, pk, cv))

    shards_err = None

    def _shards():
        nonlocal shards_err
        curve, bound = shards_lru_mrc(
            encoded, caps, num_experts=args.num_experts, sampling_rate=0.01
        )
        shards_err = bound
        return curve

    el, pk, cv = _timed(_shards)
    rows.append(("lru shards (1% sample)", el, pk, cv))

    # Report.
    ref_time = next(
        (t for name, t, _, _ in rows if name.startswith("reference")), None
    )
    header = f"{'method':<26} {'wall_s':>9} {'peak_MB':>9} {'speedup':>9}"
    print("\n" + header)
    print("-" * len(header))
    for name, el, pk, _ in rows:
        speedup = f"{ref_time / el:6.1f}x" if ref_time and el > 0 else "   n/a"
        print(f"{name:<26} {el:9.3f} {pk:9.1f} {speedup:>9}")

    if shards_err is not None:
        print(f"\n[shards] reported abs-error bound ~= {shards_err:.4f}")

    # Verify exact paths agree with the reference (or with vectorized when the
    # reference is skipped) so the benchmark doubles as a parity check.
    baseline_name, baseline_curve = (
        ("reference", rows[0][3])
        if not args.skip_reference
        else ("vectorized", rows[0][3])
    )
    for name, _, _, cv in rows:
        if "shards" in name or "lru" in name:
            continue
        ok = all(abs(cv[c] - baseline_curve[c]) < 1e-12 for c in caps)
        status = "MATCH" if ok else "MISMATCH"
        print(f"[parity:{baseline_name}] {name}: {status}")

    if args.markdown:
        print("\n| method | wall (s) | peak (MB) | speedup |")
        print("| --- | ---: | ---: | ---: |")
        for name, el, pk, _ in rows:
            speedup = f"{ref_time / el:.1f}x" if ref_time and el > 0 else "n/a"
            print(f"| {name} | {el:.3f} | {pk:.1f} | {speedup} |")


if __name__ == "__main__":
    main()
