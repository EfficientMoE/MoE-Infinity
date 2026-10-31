"""Profile the pure-Python reference simulator on a synthetic trace.

Run ``cProfile`` over ``extract_access_sequence`` + ``belady_simulate`` to find
the hotspots before optimizing. Prints the top cumulative-time functions.

Example::

    python -m benchmarks.cache_simulator.research.profile_reference \
        --accesses 2000000 --capacity 2000 --seed 0
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import time
from io import StringIO

from benchmarks.cache_simulator.belady_cache_simulator import belady_simulate
from benchmarks.cache_simulator.research.synthetic import (
    synthetic_accesses,
    synthetic_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accesses", type=int, default=2_000_000)
    parser.add_argument("--capacity", type=int, default=2000)
    parser.add_argument("--num-layers", type=int, default=94)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"[synthetic] accesses={args.accesses} capacity={args.capacity} "
        f"layers={args.num_layers} experts={args.num_experts} seed={args.seed}"
    )

    # Build recorder-shaped steps so parsing is also profiled.
    per_step = (args.num_layers - 3) * 8
    steps = max(1, args.accesses // max(per_step, 1))
    t0 = time.perf_counter()
    records = synthetic_records(
        steps,
        num_layers=args.num_layers,
        num_experts=args.num_experts,
        seed=args.seed,
    )
    t1 = time.perf_counter()
    print(f"generated {steps} synthetic records in {t1 - t0:.2f}s")

    from benchmarks.cache_simulator.belady_cache_simulator import (
        extract_access_sequence,
    )

    profiler = cProfile.Profile()
    profiler.enable()
    accesses = extract_access_sequence(
        records, ignore_layers=3, include_prefill=True
    )
    hits, misses, rate = belady_simulate(accesses, args.capacity)
    profiler.disable()

    print(
        f"[synthetic] hits={hits} misses={misses} "
        f"total={hits + misses} hit_rate={rate:.4f}"
    )

    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(args.top)
    print(stream.getvalue())

    # Also keep a reference wall-clock split so the report can cite it.
    del accesses
    _ = synthetic_accesses(1, seed=args.seed)  # warm import path


if __name__ == "__main__":
    main()
