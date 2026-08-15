"""Immutable result + cost-model contract for the PD-DFlash serving gate.

Task 1 of ``docs/superpowers/plans/2026-08-14-pd-dflash-serving-scheduler.md``.
Pure Python, imported by both the (later) GPU runner and the aggregator so the
schema is defined once. Two public entry points:

* ``evaluate_hide_inequality`` -- the design's §7 route-ahead hiding inequality
  ``(1 - r) * s * M / BW <= t_draft + t_router + overlap`` evaluated from
  *measured* terms only (never a theoretical PCIe bandwidth);
* ``validate_result_matrix`` -- enforces that a §8 result matrix carries every
  baseline (B0-B3) and every metric, permitting B3's explicit
  ``UNAVAILABLE_CAPACITY`` status when a resident upper bound does not fit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REQUIRED_METRICS = (
    "output_tokens_per_second",
    "acceptance_length_a",
    "ttft_seconds",
    "per_round_latency_seconds",
    "goodput_at_slo",
    "expert_cache_hit_rate",
    "route_ahead_prefetch_coverage",
    "wasted_prefetch_bytes",
    "expert_occupancy_bytes",
    "kv_occupancy_bytes",
)

REQUIRED_BASELINES = ("B0", "B1", "B2", "B3")

UNAVAILABLE_CAPACITY = "UNAVAILABLE_CAPACITY"


@dataclass(frozen=True)
class HideInequality:
    """Both sides of the §7 route-ahead hiding inequality and its verdict."""

    resident_fraction: float
    saturation: float
    total_expert_bytes: float
    measured_h2d_bytes_per_second: float
    draft_seconds: float
    router_seconds: float
    overlap_seconds: float
    fetch_seconds: float
    hide_window_seconds: float
    hidden: bool


def _require_finite_non_negative(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and >= 0; got {value!r}")
    return number


def evaluate_hide_inequality(
    *,
    resident_fraction: float,
    saturation: float,
    total_expert_bytes: float,
    measured_h2d_bytes_per_second: float,
    draft_seconds: float,
    router_seconds: float,
    overlap_seconds: float,
) -> HideInequality:
    """Evaluate ``(1 - r) * s * M / BW <= t_draft + t_router + overlap``.

    Raises ``ValueError`` on a resident fraction outside ``[0, 1]``, a
    non-positive measured bandwidth, or any negative time/byte term.
    """
    r = float(resident_fraction)
    if not math.isfinite(r) or not 0.0 <= r <= 1.0:
        raise ValueError(
            f"resident_fraction must be in [0, 1]; got {resident_fraction!r}"
        )
    s = _require_finite_non_negative("saturation", saturation)
    if s > 1.0:
        raise ValueError(f"saturation must be in [0, 1]; got {saturation!r}")
    total = _require_finite_non_negative(
        "total_expert_bytes", total_expert_bytes
    )
    bandwidth = float(measured_h2d_bytes_per_second)
    if not math.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError(
            "measured_h2d_bytes_per_second must be > 0; "
            f"got {measured_h2d_bytes_per_second!r}"
        )
    draft = _require_finite_non_negative("draft_seconds", draft_seconds)
    router = _require_finite_non_negative("router_seconds", router_seconds)
    overlap = _require_finite_non_negative("overlap_seconds", overlap_seconds)

    fetch_seconds = (1.0 - r) * s * total / bandwidth
    hide_window_seconds = draft + router + overlap
    return HideInequality(
        resident_fraction=r,
        saturation=s,
        total_expert_bytes=total,
        measured_h2d_bytes_per_second=bandwidth,
        draft_seconds=draft,
        router_seconds=router,
        overlap_seconds=overlap,
        fetch_seconds=fetch_seconds,
        hide_window_seconds=hide_window_seconds,
        hidden=fetch_seconds <= hide_window_seconds,
    )


def validate_result_matrix(
    rows: Mapping[str, Mapping[str, object]],
) -> None:
    """Assert a §8 matrix has every baseline and every well-formed metric.

    B0-B3 must all be present. Each row must supply every ``REQUIRED_METRICS``
    entry as a finite, non-negative number, with one exception: a B3 row whose
    ``status`` equals ``UNAVAILABLE_CAPACITY`` (resident upper bound did not
    fit) is accepted without metrics. Raises ``ValueError`` otherwise.
    """
    missing = [b for b in REQUIRED_BASELINES if b not in rows]
    if missing:
        raise ValueError(f"missing baselines: {', '.join(missing)}")

    for baseline in REQUIRED_BASELINES:
        row = rows[baseline]
        if baseline == "B3" and row.get("status") == UNAVAILABLE_CAPACITY:
            continue
        _validate_metric_row(baseline, row)


def _validate_metric_row(baseline: str, row: Mapping[str, object]) -> None:
    for metric in REQUIRED_METRICS:
        if metric not in row:
            raise ValueError(f"{baseline} missing metric: {metric}")
        value = row[metric]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{baseline}.{metric} must be a finite, non-negative "
                f"number; got {value!r}"
            )
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(
                f"{baseline}.{metric} must be a finite, non-negative "
                f"number; got {value!r}"
            )


def summarise_row(row: Mapping[str, object]) -> Mapping[str, object]:
    """BM1 router-ahead cost summary for one row (design §10).

    BM1 passes when the route-ahead router projection is strictly cheaper than
    the width-B verify it front-runs (``t_router < t_verify``); the raw seconds
    and their ratio are retained so the aggregator can rank configurations, not
    only gate them. Raises ``ValueError`` on a missing term, a negative time, or
    a non-positive ``t_verify_seconds``.
    """
    for key in ("t_router_seconds", "t_verify_seconds"):
        if key not in row:
            raise ValueError(f"row missing BM1 term: {key}")
    t_router = _require_finite_non_negative(
        "t_router_seconds", row["t_router_seconds"]
    )
    t_verify = _require_finite_non_negative(
        "t_verify_seconds", row["t_verify_seconds"]
    )
    if t_verify <= 0.0:
        raise ValueError(f"t_verify_seconds must be > 0; got {t_verify!r}")
    return {
        "t_router_seconds": t_router,
        "t_verify_seconds": t_verify,
        "bm1_router_to_verify_ratio": t_router / t_verify,
        "bm1_pass": t_router < t_verify,
    }


BM5_INVARIANTS = (
    "acceptance_length_a",
    "route_ahead_prefetch_coverage",
    "wasted_prefetch_bytes",
)


def summarise_bm5_equivalence(
    *,
    python_row: Mapping[str, Any],
    cpp_row: Mapping[str, Any],
    rel_tol: float = 1e-6,
) -> Dict[str, Any]:
    """BM5: shipped C++ issue must not change correctness-visible terms.

    Switching Python per-expert issuance to the shipped C++ batched issuance may
    only move tokens/s: acceptance length, route-ahead coverage, wasted bytes,
    and output tokens must match within ``rel_tol``. Returns the equivalence
    verdict, the mismatched invariants, and both tokens/s for the plot.
    """
    mismatched: List[str] = []
    for key in BM5_INVARIANTS:
        if key not in python_row or key not in cpp_row:
            mismatched.append(key)
            continue
        a = float(python_row[key])
        b = float(cpp_row[key])
        scale = max(abs(a), abs(b), 1.0)
        if abs(a - b) > rel_tol * scale:
            mismatched.append(key)
    return {
        "benchmark": "BM5",
        "equivalent": not mismatched,
        "mismatched": mismatched,
        "python_tokens_per_second": float(
            python_row.get("output_tokens_per_second", float("nan"))
        ),
        "cpp_tokens_per_second": float(
            cpp_row.get("output_tokens_per_second", float("nan"))
        ),
    }


def cpp_hop_verdicts(
    *,
    bm2: Optional[Mapping[str, Any]],
    bm3: Optional[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Keep/remove each benchmark-gated C++ hop from its paired BM verdict.

    A hop is kept only when its paired benchmark is present and passes: the
    batched-issuance hop needs BM2 ``ship_batched``; the route-ahead
    priority-band hop needs BM3 ``ship_priority_band``. A missing BM removes the
    hop (design §10: no C++ change ships without its BM).
    """

    def verdict(report: Optional[Mapping[str, Any]], flag: str, hop: str):
        if report is None:
            return {
                "keep": False,
                "reason": f"{hop} removed: paired benchmark absent",
            }
        keep = bool(report.get(flag, False))
        return {
            "keep": keep,
            "reason": (
                f"{hop} kept: {flag} is true"
                if keep
                else f"{hop} removed: {flag} is false"
            ),
        }

    return {
        "batched_issuance": verdict(bm2, "ship_batched", "batched_issuance"),
        "priority_band": verdict(bm3, "ship_priority_band", "priority_band"),
    }


def aggregate_result_matrices(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[str, int, int], Dict[str, Mapping[str, Any]]]:
    """Group raw observation rows into §8 matrices.

    Keyed by ``(model, block_size, concurrency)``; each value maps a baseline
    label to its single row. A duplicate ``(key, baseline)`` raises, mirroring
    the runner's append-without-overwrite guarantee.
    """
    matrices: Dict[Tuple[str, int, int], Dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["model"]),
            int(row["block_size"]),
            int(row["concurrency"]),
        )
        baseline = str(row["baseline"])
        bucket = matrices.setdefault(key, {})
        if baseline in bucket:
            raise ValueError(f"duplicate baseline {baseline} for {key}")
        bucket[baseline] = row
    return matrices


def evaluate_matrix(
    baseline_rows: Mapping[str, Mapping[str, Any]],
    allow_blocked: Sequence[str] = (),
) -> Tuple[bool, Dict[str, Any]]:
    """Completeness + BM1 verdict for one grouped matrix.

    A baseline is satisfied when it carries the full metric schema; a baseline
    listed in ``allow_blocked`` may instead carry a blocking ``status`` (e.g.
    B2's ``BLOCKED_UNTIL_2D_SCHEDULER`` before the 2-D scheduler lands). Any
    other missing/invalid/blocked baseline fails the group. BM1 summaries are
    attached for every row carrying ``t_router_seconds``/``t_verify_seconds``.
    """
    allow = set(allow_blocked)
    detail: Dict[str, Any] = {
        "present": sorted(baseline_rows),
        "blocked": [],
        "missing": [],
        "invalid": [],
        "bm1": {},
    }
    ok = True
    for baseline in REQUIRED_BASELINES:
        row = baseline_rows.get(baseline)
        if row is None:
            detail["missing"].append(baseline)
            ok = False
            continue
        if row.get("status"):
            if baseline in allow:
                detail["blocked"].append(baseline)
            else:
                detail["blocked"].append(baseline)
                ok = False
            continue
        try:
            _validate_metric_row(baseline, row)
        except ValueError:
            detail["invalid"].append(baseline)
            ok = False
        if "t_router_seconds" in row and "t_verify_seconds" in row:
            detail["bm1"][baseline] = dict(summarise_row(row))
    return ok, detail


def _matrix_key(key: Tuple[str, int, int]) -> str:
    return f"{key[0]}|B{key[1]}|c{key[2]}"


def _write_json(path: str, payload: Mapping[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(
    path: str,
    matrices: Mapping[Tuple[str, int, int], Mapping[str, Mapping[str, Any]]],
) -> None:
    import csv

    columns = ["model", "block_size", "concurrency", "baseline", "status"]
    columns += list(REQUIRED_METRICS)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for (model, block, conc), baseline_rows in sorted(matrices.items()):
            for baseline, row in sorted(baseline_rows.items()):
                record = {
                    "model": model,
                    "block_size": block,
                    "concurrency": conc,
                    "baseline": baseline,
                    "status": row.get("status", ""),
                }
                for metric in REQUIRED_METRICS:
                    record[metric] = row.get(metric, "")
                writer.writerow(record)


def _write_markdown(path: str, report: Mapping[str, Any]) -> None:
    lines = ["# PD-DFlash Phase-A result matrix", ""]
    for group, detail in sorted(report.items()):
        lines.append(f"## {group}")
        lines.append(f"- present: {', '.join(detail['present']) or '(none)'}")
        if detail["blocked"]:
            lines.append(f"- blocked: {', '.join(detail['blocked'])}")
        if detail["missing"]:
            lines.append(f"- missing: {', '.join(detail['missing'])}")
        if detail["invalid"]:
            lines.append(f"- invalid: {', '.join(detail['invalid'])}")
        for baseline, bm1 in sorted(detail["bm1"].items()):
            lines.append(
                f"- BM1 {baseline}: ratio="
                f"{bm1['bm1_router_to_verify_ratio']:.4f} "
                f"pass={bm1['bm1_pass']}"
            )
        lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.dflash.report",
        description="Aggregate PD-DFlash raw observation rows into §8 matrices.",
    )
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--matrix-json")
    parser.add_argument("--csv")
    parser.add_argument("--markdown")
    parser.add_argument("--allow-blocked", nargs="*", default=[])
    args = parser.parse_args(argv)

    rows: List[Mapping[str, Any]] = []
    for path in args.input:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        rows.extend(data if isinstance(data, list) else [data])

    matrices = aggregate_result_matrices(rows)
    report: Dict[str, Any] = {}
    all_ok = True
    for key, baseline_rows in sorted(matrices.items()):
        ok, detail = evaluate_matrix(baseline_rows, args.allow_blocked)
        all_ok = all_ok and ok
        report[_matrix_key(key)] = detail

    if args.matrix_json:
        _write_json(
            args.matrix_json,
            {_matrix_key(k): dict(v) for k, v in matrices.items()},
        )
    if args.csv:
        _write_csv(args.csv, matrices)
    if args.markdown:
        _write_markdown(args.markdown, report)

    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if all_ok else 1


__all__ = [
    "REQUIRED_METRICS",
    "REQUIRED_BASELINES",
    "UNAVAILABLE_CAPACITY",
    "HideInequality",
    "aggregate_result_matrices",
    "cpp_hop_verdicts",
    "evaluate_hide_inequality",
    "evaluate_matrix",
    "main",
    "summarise_bm5_equivalence",
    "summarise_row",
    "validate_result_matrix",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
