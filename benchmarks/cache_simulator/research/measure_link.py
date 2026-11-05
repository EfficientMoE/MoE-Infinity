"""Measure the host-to-GPU copy time of one expert on this host.

Feeds ``timing_model.LinkModel(link_bytes_per_s=...)`` with a measured value
instead of the PCIe-spec estimate. For each expert size it copies a pinned
host buffer to the GPU:

* ``single``: one copy at a time, synchronized after each (a demand miss on
  an idle link: launch overhead + transfer);
* ``queued``: ``--repeats`` copies issued back to back on one stream, timed
  as a batch (prefetches queued on a busy link: sustained bandwidth).

Default sizes are the per-expert bytes of the timing-model presets
(Qwen3-30B-A3B BF16: 9,437,184 B; Qwen3-235B-A22B-FP8: 18,874,368 B).

Usage::

    CUDA_VISIBLE_DEVICES=2 python -m \
        benchmarks.cache_simulator.research.measure_link --repeats 200
"""

from __future__ import annotations

import argparse
import statistics
from typing import Dict, List, Optional, Sequence

DEFAULT_SIZES = (9_437_184, 18_874_368)


def measure(size: int, repeats: int, warmup: int = 10) -> Dict[str, float]:
    """Seconds per pinned host-to-device copy of ``size`` bytes."""
    import torch

    host = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    slots = [
        torch.empty(size, dtype=torch.uint8, device="cuda") for _ in range(4)
    ]
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for i in range(warmup):
            slots[i % 4].copy_(host, non_blocking=True)
    stream.synchronize()

    single: List[float] = []
    for i in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record(stream)
            slots[i % 4].copy_(host, non_blocking=True)
            end.record(stream)
        end.synchronize()
        single.append(start.elapsed_time(end) / 1e3)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record(stream)
        for i in range(repeats):
            slots[i % 4].copy_(host, non_blocking=True)
        end.record(stream)
    end.synchronize()
    queued = start.elapsed_time(end) / 1e3 / repeats
    return {
        "bytes": float(size),
        "single_median_s": statistics.median(single),
        "single_p90_s": sorted(single)[int(0.9 * (len(single) - 1))],
        "queued_mean_s": queued,
        "queued_bytes_per_s": size / queued,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args(argv)

    import torch

    print(
        f"device: {torch.cuda.get_device_name()}, torch {torch.__version__}, "
        f"repeats {args.repeats}"
    )
    print(
        "| Bytes | single copy, median (us) | single, p90 (us) "
        "| queued, per copy (us) | queued GB/s |"
    )
    print("| ---: | ---: | ---: | ---: | ---: |")
    for size in args.sizes:
        row = measure(size, args.repeats)
        print(
            f"| {size:,} | {row['single_median_s'] * 1e6:.1f} "
            f"| {row['single_p90_s'] * 1e6:.1f} "
            f"| {row['queued_mean_s'] * 1e6:.1f} "
            f"| {row['queued_bytes_per_s'] / 1e9:.2f} |"
        )


if __name__ == "__main__":
    main()
