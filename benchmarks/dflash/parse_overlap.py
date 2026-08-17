# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""BM4 -- expert-H2D / compute overlap from an nsys trace (design §10).

Task 10 of ``docs/superpowers/plans/2026-08-14-pd-dflash-serving-scheduler.md``.
This is the ground truth for whether offloaded-expert fetch is *hidden*: the
fraction of expert host->device (H2D) memcpy **bytes** that overlap the DFlash
draft/router/verify compute NVTX ranges the serving runner emits.

Two layers:

* pure interval arithmetic (``compute_overlap``) apportions each memcpy's bytes
  by its overlapped-duration fraction against the *union* of the compute ranges
  -- a partially hidden copy contributes a proportional slice of its bytes, and
  overlapping compute ranges are unioned so overlap can never exceed 100%; and
* an nsys CSV reader (``parse_nsys_rep``) that runs
  ``nsys stats --report cuda_gpu_trace,nvtx_pushpop_trace --format csv`` on a
  ``.nsys-rep`` and feeds the parsed H2D memcpys and NVTX ranges into
  ``compute_overlap``.

The pure layer is import-safe and unit-tested off-hardware; only
``parse_nsys_rep`` shells out to ``nsys``.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# NVTX ranges that count as compute able to hide an expert fetch (design §10);
# these are three of the five ranges the serving runner emits. ``expert_h2d``
# and ``route_ahead_issue`` are issuance/transfer ranges, not hiding compute.
DEFAULT_COMPUTE_RANGES: Tuple[str, ...] = (
    "dflash_draft",
    "route_ahead_router",
    "target_verify",
)


@dataclass(frozen=True)
class Memcpy:
    """One H2D memcpy interval with its transferred byte count."""

    start: float
    end: float
    bytes: int


@dataclass(frozen=True)
class NvtxRange:
    """One NVTX push/pop range interval."""

    name: str
    start: float
    end: float


@dataclass(frozen=True)
class OverlapResult:
    """Aggregate BM4 overlap over a set of memcpys and compute ranges."""

    total_h2d_bytes: int
    overlapped_h2d_bytes: float
    overlap_fraction: float
    per_memcpy_fraction: Tuple[float, ...]


def _merge_intervals(
    intervals: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Union a set of ``(start, end)`` intervals into disjoint spans."""
    spans = sorted(
        (float(a), float(b)) for a, b in intervals if float(b) > float(a)
    )
    merged: List[Tuple[float, float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _overlap_duration(
    interval: Tuple[float, float], spans: Sequence[Tuple[float, float]]
) -> float:
    """Total length of ``interval`` covered by the disjoint ``spans``."""
    start, end = interval
    covered = 0.0
    for span_start, span_end in spans:
        lo = max(start, span_start)
        hi = min(end, span_end)
        if hi > lo:
            covered += hi - lo
    return covered


def _strip_empty_domain(name: str) -> str:
    """Drop the leading empty-domain ``:`` nsys prefixes default-domain ranges.

    nsys renders push/pop ranges as ``<domain>:<message>``; a range pushed
    without a registered domain (``torch.cuda.nvtx.range_push`` / native
    ``nvtx3::scoped_range`` / ``nvtx.push_range``) is rendered with an EMPTY
    domain, i.e. a leading ``:`` (``:dflash_draft``). Named domains such as
    ``CCCL:cub::...`` are left untouched -- only the empty-domain colon is
    dropped so the bare ``DEFAULT_COMPUTE_RANGES`` names match the trace.
    """
    return name[1:] if name.startswith(":") else name


def compute_overlap(
    memcpys: Sequence[Memcpy],
    ranges: Sequence[NvtxRange],
    compute_ranges: Sequence[str] = DEFAULT_COMPUTE_RANGES,
) -> OverlapResult:
    """Apportion H2D bytes hidden behind the union of compute ranges.

    Each memcpy contributes ``bytes * (overlapped_duration / duration)`` hidden
    bytes; a zero-duration memcpy contributes its full bytes only when its
    start instant lies inside a compute span, and nothing otherwise. The
    aggregate fraction is hidden bytes over total bytes (``0`` when no bytes).

    Range names are matched modulo the empty-domain ``:`` prefix nsys renders
    default-domain push/pop ranges with, so the bare ``DEFAULT_COMPUTE_RANGES``
    match the captured ``:dflash_draft`` / ``:route_ahead_router`` /
    ``:target_verify`` (and ``--compute-range :expert_compute`` still matches).
    """
    compute_names = {_strip_empty_domain(name) for name in compute_ranges}
    spans = _merge_intervals(
        [
            (r.start, r.end)
            for r in ranges
            if _strip_empty_domain(r.name) in compute_names
        ]
    )

    total_bytes = 0
    overlapped_bytes = 0.0
    fractions: List[float] = []
    for memcpy in memcpys:
        nbytes = int(memcpy.bytes)
        total_bytes += nbytes
        duration = float(memcpy.end) - float(memcpy.start)
        if duration > 0.0:
            fraction = (
                _overlap_duration(
                    (float(memcpy.start), float(memcpy.end)), spans
                )
                / duration
            )
        else:
            instant = float(memcpy.start)
            fraction = (
                1.0 if any(lo <= instant <= hi for lo, hi in spans) else 0.0
            )
        fraction = min(max(fraction, 0.0), 1.0)
        fractions.append(fraction)
        overlapped_bytes += nbytes * fraction

    overlap_fraction = (
        overlapped_bytes / total_bytes if total_bytes > 0 else 0.0
    )
    return OverlapResult(
        total_h2d_bytes=total_bytes,
        overlapped_h2d_bytes=overlapped_bytes,
        overlap_fraction=overlap_fraction,
        per_memcpy_fraction=tuple(fractions),
    )


# ---------------------------------------------------------------------------
# nsys CSV reader (shells out to the ``nsys`` CLI)
# ---------------------------------------------------------------------------


def _nsys_stats_csv(rep_path: str, report: str) -> str:
    proc = subprocess.run(
        [
            "nsys",
            "stats",
            "--report",
            report,
            "--format",
            "csv",
            "--force-export=true",
            rep_path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def _read_csv_rows(text: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    reader = csv.reader(io.StringIO(text))
    header: Optional[List[str]] = None
    for record in reader:
        if not record:
            continue
        if header is None:
            # nsys prefixes stats blocks with a title line before the header;
            # the header is the first row that contains a recognised column.
            lowered = [c.strip().lower() for c in record]
            if any("duration" in c or "name" in c for c in lowered):
                header = [c.strip() for c in record]
            continue
        if len(record) < len(header):
            continue
        rows.append(dict(zip(header, record)))
    return rows


def _to_ns(value: str) -> float:
    return float(str(value).replace(",", "").strip())


def _column(row: Dict[str, str], *candidates: str) -> Optional[str]:
    lowered = {k.lower(): k for k in row}
    for candidate in candidates:
        key = lowered.get(candidate.lower())
        if key is not None and row[key] != "":
            return row[key]
    return None


def parse_memcpys(csv_text: str) -> List[Memcpy]:
    """Parse H2D memcpy intervals from an nsys ``cuda_gpu_trace`` CSV."""
    memcpys: List[Memcpy] = []
    for row in _read_csv_rows(csv_text):
        name = (_column(row, "Name") or "").strip()
        if "memcpy" not in name.lower() and "htod" not in name.lower():
            continue
        if "hto" not in name.lower().replace("-", "") and "htod" not in (
            name.lower()
        ):
            if "host-to-device" not in name.lower():
                continue
        start = _column(row, "Start (ns)", "Start")
        duration = _column(row, "Duration (ns)", "Duration")
        nbytes = _column(row, "Bytes (MB)", "Bytes", "Size (MB)", "Size")
        if start is None or duration is None or nbytes is None:
            continue
        start_ns = _to_ns(start)
        dur_ns = _to_ns(duration)
        raw_bytes = _to_ns(nbytes)
        byte_count = (
            int(raw_bytes * 1_000_000)
            if "MB" in " ".join(row.keys())
            else int(raw_bytes)
        )
        memcpys.append(
            Memcpy(start=start_ns, end=start_ns + dur_ns, bytes=byte_count)
        )
    return memcpys


def parse_nvtx_ranges(csv_text: str) -> List[NvtxRange]:
    """Parse NVTX push/pop ranges from an nsys ``nvtx_pushpop_trace`` CSV."""
    ranges: List[NvtxRange] = []
    for row in _read_csv_rows(csv_text):
        name = (_column(row, "Name", "Text") or "").strip()
        if not name:
            continue
        start = _column(row, "Start (ns)", "Start")
        duration = _column(row, "Duration (ns)", "Duration")
        if start is None or duration is None:
            continue
        start_ns = _to_ns(start)
        dur_ns = _to_ns(duration)
        ranges.append(
            NvtxRange(name=name, start=start_ns, end=start_ns + dur_ns)
        )
    return ranges


def parse_nsys_rep(
    rep_path: str,
    compute_ranges: Sequence[str] = DEFAULT_COMPUTE_RANGES,
) -> Dict[str, Any]:
    """Run ``nsys stats`` on ``rep_path`` and return the BM4 overlap report."""
    memcpys = parse_memcpys(_nsys_stats_csv(rep_path, "cuda_gpu_trace"))
    ranges = parse_nvtx_ranges(_nsys_stats_csv(rep_path, "nvtx_pushpop_trace"))
    result = compute_overlap(memcpys, ranges, compute_ranges)
    return {
        "benchmark": "BM4",
        "rep": rep_path,
        "total_h2d_bytes": result.total_h2d_bytes,
        "overlapped_h2d_bytes": result.overlapped_h2d_bytes,
        "overlap_fraction": result.overlap_fraction,
        "exposed_fetch_bytes": result.total_h2d_bytes
        - result.overlapped_h2d_bytes,
        "num_memcpys": len(memcpys),
        "compute_ranges": list(compute_ranges),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.dflash.parse_overlap",
        description="BM4 expert-H2D / compute overlap from an nsys trace.",
    )
    parser.add_argument("--rep", required=True, help="path to a .nsys-rep")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--compute-range",
        nargs="+",
        default=list(DEFAULT_COMPUTE_RANGES),
        help="NVTX range names that count as fetch-hiding compute",
    )
    args = parser.parse_args(argv)

    report = parse_nsys_rep(args.rep, args.compute_range)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


__all__ = [
    "DEFAULT_COMPUTE_RANGES",
    "Memcpy",
    "NvtxRange",
    "OverlapResult",
    "compute_overlap",
    "parse_memcpys",
    "parse_nvtx_ranges",
    "parse_nsys_rep",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
