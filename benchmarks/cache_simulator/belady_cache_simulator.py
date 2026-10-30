"""Belady optimal cache simulator for expert routing traces.

This script consumes ``.pt`` trace files where each file stores a dict with a
``records`` list. The first element is the prefill step, and subsequent
elements are decode steps. Each record owns a ``topk_ids_of_layer`` tensor of
shape ``[num_layers, seq_len, topk]`` that records the activated experts.

We simulate cache hits during decode using the Belady optimal policy. Each
cache key is a tuple ``(layer_idx, expert_id)``. Layers before
``ignore_layers`` are skipped. To keep the simulator efficient, the future
access positions for every key are precomputed as queues before the main loop.

The pure-Python helpers (``extract_access_sequence``, ``build_future_positions``
and ``belady_simulate``) carry no heavy dependencies so they can be reused and
unit tested without a GPU or trace data on disk.
"""

from __future__ import annotations

import argparse
import glob
import heapq
import math
from collections import defaultdict, deque
from typing import Deque, Dict, List, Sequence, Tuple

try:  # ``rich`` is optional; fall back to plain output when unavailable.
    from rich import print
    from rich.progress import track
except ImportError:  # pragma: no cover - exercised only without rich

    def track(sequence, description="", **_kwargs):
        return sequence


Key = Tuple[int, int]
INF = math.inf


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Simulate decode cache hit rate with Belady optimal policy.",
    )
    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        required=True,
        help="Trace files (.pt) or glob patterns to load.",
    )
    parser.add_argument(
        "-c",
        "--capacity",
        type=int,
        required=True,
        help="Cache capacity measured in (layer, expert_id) entries.",
    )
    parser.add_argument(
        "--ignore-layers",
        type=int,
        default=3,
        help="Number of leading non-MoE layers to skip (default: 3).",
    )
    parser.add_argument(
        "--include-prefill",
        action="store_true",
        help="Include prefill step in the access trace (default: decode only).",
    )
    return parser.parse_args()


def expand_files(raw_paths: Sequence[str]) -> List[str]:
    """Expand glob patterns while preserving order and uniqueness."""

    seen = set()
    expanded: List[str] = []
    for path in raw_paths:
        candidates = glob.glob(path)
        if not candidates:
            candidates = [path]
        for candidate in sorted(candidates):
            if candidate in seen:
                continue
            seen.add(candidate)
            expanded.append(candidate)
    return expanded


def load_records(file_path: str) -> Sequence[dict]:
    """Load and validate records from a trace file."""

    import torch  # Imported lazily so pure helpers stay dependency-free.

    payload = torch.load(file_path, map_location="cpu")
    if "records" not in payload:
        raise ValueError(f"File {file_path} does not contain 'records'.")
    return payload["records"]


def describe_trace(records: Sequence[dict], file_path: str) -> None:
    """Print basic stats for prefill/decode steps."""

    total_steps = len(records)
    prefill_len = records[0]["topk_ids_of_layer"].shape[1] if total_steps else 0
    decode_lengths = (
        [rec["topk_ids_of_layer"].shape[1] for rec in records[1:]]
        if total_steps > 1
        else []
    )

    print(
        f"[bold yellow]{file_path}[/] steps={total_steps} "
        f"(prefill + {max(total_steps - 1, 0)} decode), "
        f"prefill_len={prefill_len}, decode_lens={decode_lengths}"
    )


def extract_access_sequence(
    records: Sequence[dict],
    *,
    ignore_layers: int,
    include_prefill: bool,
) -> List[Key]:
    """Flatten records into a sequential access list of (layer, expert_id)."""

    if not records:
        return []

    start_index = 0 if include_prefill else 1
    accesses: List[Key] = []

    for record in records[start_index:]:
        tensor = record["topk_ids_of_layer"]
        num_layers = tensor.shape[0]
        if ignore_layers >= num_layers:
            continue

        # Decode step seq_len is usually 1, but we keep it generic.
        flattened = tensor.reshape(num_layers, -1)
        for layer_idx in range(ignore_layers, num_layers):
            layer_values = flattened[layer_idx]
            for expert_id in layer_values.tolist():
                if expert_id < 0:
                    continue
                accesses.append((layer_idx, int(expert_id)))

    return accesses


def build_future_positions(accesses: Sequence[Key]) -> Dict[Key, Deque[int]]:
    """Pre-compute future positions for every key as ascending queues."""

    future: Dict[Key, Deque[int]] = defaultdict(deque)
    for idx in range(len(accesses) - 1, -1, -1):
        future[accesses[idx]].appendleft(idx)
    return future


def belady_simulate(
    accesses: Sequence[Key],
    capacity: int,
    *,
    show_progress: bool = False,
    progress_desc: str = "Simulating",
) -> Tuple[int, int, float]:
    """Run Belady optimal simulation.

    Returns a tuple of ``(hits, misses, hit_rate)``.
    """

    total = len(accesses)
    if total == 0:
        return 0, 0, 0.0
    if capacity <= 0:
        return 0, total, 0.0

    future_positions = build_future_positions(accesses)
    cache_next: Dict[Key, float] = {}
    # Max-heap via negative next use.
    heap: List[Tuple[float, Key]] = []

    hits = 0
    misses = 0

    iterable = (
        track(accesses, description=progress_desc)
        if show_progress
        else accesses
    )

    for key in iterable:
        positions = future_positions[key]
        if positions:
            positions.popleft()
        next_use = positions[0] if positions else INF

        if key in cache_next:
            hits += 1
            cache_next[key] = next_use
            heapq.heappush(heap, (-next_use, key))
            continue

        misses += 1
        if len(cache_next) >= capacity:
            # Evict farthest in the future (or never used again).
            evicted = False
            while heap:
                candidate_neg, candidate_key = heapq.heappop(heap)
                candidate_next = -candidate_neg
                if cache_next.get(candidate_key) == candidate_next:
                    cache_next.pop(candidate_key, None)
                    evicted = True
                    break
            if not evicted and cache_next:
                # Fallback if all heap entries were stale: drop any key.
                cache_next.pop(next(iter(cache_next)))
        cache_next[key] = next_use
        heapq.heappush(heap, (-next_use, key))

    hit_rate = hits / total
    return hits, misses, hit_rate


def summarize_result(
    file_path: str, hits: int, misses: int, hit_rate: float
) -> None:
    """Pretty-print a single file's simulation summary."""

    total = hits + misses
    print(
        f"[bold cyan]{file_path}[/] -> hits: {hits}, misses: {misses}, "
        f"total: {total}, hit_rate: {hit_rate:.4f}"
    )


def main() -> None:
    args = parse_args()
    file_paths = expand_files(args.files)
    if not file_paths:
        raise ValueError("No input files found.")

    for file_path in file_paths:
        records = load_records(file_path)
        describe_trace(records, file_path)
        accesses = extract_access_sequence(
            records,
            ignore_layers=args.ignore_layers,
            include_prefill=args.include_prefill,
        )
        hits, misses, hit_rate = belady_simulate(
            accesses,
            args.capacity,
            show_progress=True,
            progress_desc=f"Simulating {file_path}",
        )
        summarize_result(file_path, hits, misses, hit_rate)


if __name__ == "__main__":
    main()
