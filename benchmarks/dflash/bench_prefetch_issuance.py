# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""BM2 -- saturated route-ahead prefetch issuance micro-benchmark (design §10).

Task 7 of ``docs/superpowers/plans/2026-08-14-pd-dflash-serving-scheduler.md``.
Measures how long it takes to *issue* (enqueue) a route-ahead prefetch for a
saturated block of ``E_l x L`` offloaded expert tensors, three ways:

* ``python-per-expert`` -- the current
  ``ExpertPrefetcher.prefetch_experts_list`` path: one
  ``get_node_default_device`` + ``enqueue_prefetch`` pybind pair per tensor
  (``2 * E_l * L`` boundary crossings);
* ``batched-pybind`` -- a single ``prefetch_handle.prefetch_tensors(tensor_ids)``
  call that constructs and enqueues every ``Task`` inside C++ (one crossing);
  available only once the batched native API (plan Task 8) is built in;
* ``cpp-internal`` -- reserved for a native in-C++ issuance timer; reported as
  ``null`` until such a hook exists (never zero).

The ship gate (design §10, plan Task 7/8): the batched hop is justified iff the
current Python per-expert median exceeds the route-ahead window
``t_draft + t_router`` *and* the batched median is at or below it.

Import-safe by construction: torch and moe_infinity are imported lazily inside
the GPU runner, so ``bm2_decision`` / ``percentiles_us`` / ``build_bm2_report``
(and their tests) are pure-CPU and never initialise CUDA.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

PYTHON_PER_EXPERT = "python-per-expert"
BATCHED_PYBIND = "batched-pybind"
CPP_INTERNAL = "cpp-internal"
ISSUANCE_MODES = (PYTHON_PER_EXPERT, BATCHED_PYBIND, CPP_INTERNAL)


@dataclass(frozen=True)
class Bm2Decision:
    """The BM2 ship gate over per-mode issuance medians (design §10)."""

    per_expert_us: Optional[float]
    batched_us: Optional[float]
    cpp_internal_us: Optional[float]
    window_us: float
    candidate_required: bool
    ship_batched: bool


def _finite_positive_window(window_us: Any) -> float:
    window = float(window_us)
    if not math.isfinite(window) or window <= 0.0:
        raise ValueError(
            "window_us (t_draft + t_router) must be finite and > 0; "
            f"got {window_us!r}"
        )
    return window


def _optional_us(name: str, value: Any) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(
            f"{name} must be a finite, non-negative microsecond median or "
            f"None; got {value!r}"
        )
    return number


def bm2_decision(
    per_expert_us: Optional[float],
    batched_us: Optional[float],
    cpp_internal_us: Optional[float],
    window_us: float,
) -> Bm2Decision:
    """Evaluate the BM2 ship gate from measured medians (microseconds).

    ``candidate_required`` holds when the current per-expert median exceeds the
    route-ahead window. ``ship_batched`` additionally requires a *measured*
    batched median at or below the window -- a missing batched median can never
    ship, so an unavailable candidate mode never masquerades as a win.
    """
    window = _finite_positive_window(window_us)
    per_expert = _optional_us("per_expert_us", per_expert_us)
    batched = _optional_us("batched_us", batched_us)
    cpp_internal = _optional_us("cpp_internal_us", cpp_internal_us)

    candidate_required = per_expert is not None and per_expert > window
    ship_batched = (
        candidate_required and batched is not None and batched <= window
    )
    return Bm2Decision(
        per_expert_us=per_expert,
        batched_us=batched,
        cpp_internal_us=cpp_internal,
        window_us=window,
        candidate_required=candidate_required,
        ship_batched=ship_batched,
    )


def percentiles_us(samples_ns: Sequence[int]) -> Dict[str, float]:
    """Nearest-rank p50/p90/p99 of nanosecond samples, returned in microseconds."""
    if not samples_ns:
        raise ValueError("percentiles_us requires at least one sample")
    ordered = sorted(float(sample) for sample in samples_ns)
    count = len(ordered)

    def nearest_rank(pct: float) -> float:
        rank = min(max(math.ceil(pct * count), 1), count)
        return ordered[rank - 1] / 1000.0

    return {
        "p50": nearest_rank(0.50),
        "p90": nearest_rank(0.90),
        "p99": nearest_rank(0.99),
        "min": ordered[0] / 1000.0,
        "max": ordered[-1] / 1000.0,
        "count": count,
    }


def _mode_stats(
    samples_ns: Optional[Sequence[int]],
) -> Optional[Dict[str, float]]:
    if samples_ns is None:
        return None
    return percentiles_us(samples_ns)


def build_bm2_report(
    *,
    model: str,
    saturated_tensor_count: int,
    window_us: float,
    per_expert_samples_ns: Optional[Sequence[int]],
    batched_samples_ns: Optional[Sequence[int]] = None,
    cpp_internal_samples_ns: Optional[Sequence[int]] = None,
    warmup: int,
    iterations: int,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the machine-readable BM2 report and its ship-gate verdict."""
    modes: Dict[str, Optional[Dict[str, float]]] = {
        PYTHON_PER_EXPERT: _mode_stats(per_expert_samples_ns),
        BATCHED_PYBIND: _mode_stats(batched_samples_ns),
        CPP_INTERNAL: _mode_stats(cpp_internal_samples_ns),
    }

    def median(mode: str) -> Optional[float]:
        stats = modes[mode]
        return None if stats is None else stats["p50"]

    decision = bm2_decision(
        median(PYTHON_PER_EXPERT),
        median(BATCHED_PYBIND),
        median(CPP_INTERNAL),
        window_us,
    )
    report: Dict[str, Any] = {
        "benchmark": "BM2",
        "model": model,
        "saturated_tensor_count": int(saturated_tensor_count),
        "window_us": decision.window_us,
        "warmup": int(warmup),
        "iterations": int(iterations),
        "modes": modes,
        "medians_us": {
            PYTHON_PER_EXPERT: decision.per_expert_us,
            BATCHED_PYBIND: decision.batched_us,
            CPP_INTERNAL: decision.cpp_internal_us,
        },
        "candidate_required": decision.candidate_required,
        "ship_batched": decision.ship_batched,
    }
    if extra:
        report.update(dict(extra))
    return report


def _resolve_window_us(
    window_json: Optional[str], window_us: Optional[float]
) -> float:
    """Resolve ``t_draft + t_router`` (microseconds) for the ship gate.

    ``--window-us`` wins when given; otherwise a Phase-A raw JSON is read and
    its ``t_draft``/``t_router`` seconds (either bare or ``*_seconds``-suffixed)
    are summed. Never substitutes a theoretical or hard-coded default.
    """
    if window_us is not None:
        return _finite_positive_window(window_us)
    if window_json is None:
        raise ValueError(
            "a route-ahead window is required: pass --window-us or a "
            "--window-json carrying t_draft/t_router seconds"
        )
    with open(window_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload if isinstance(payload, list) else [payload]

    def field(row: Mapping[str, Any], *names: str) -> Optional[float]:
        for name in names:
            if name in row and row[name] is not None:
                return float(row[name])
        return None

    for row in rows:
        draft = field(row, "t_draft_seconds", "t_draft")
        router = field(row, "t_router_seconds", "t_router")
        if draft is not None and router is not None:
            return _finite_positive_window((draft + router) * 1e6)
    raise ValueError(
        f"could not find t_draft and t_router seconds in {window_json!r}"
    )


def _load_prefetcher(
    model_repo: str, offload_path: str, device_memory_ratio: float
) -> Any:
    from moe_infinity import MoE  # lazy: heavy, CUDA-initialising

    model = MoE(
        model_repo,
        {
            "offload_path": offload_path,
            "device_memory_ratio": device_memory_ratio,
        },
    )
    prefetcher = model.engine.expert_prefetcher
    if prefetcher is None or prefetcher.archer_engine is None:
        raise RuntimeError(
            "loaded model has no offloaded ExpertPrefetcher/archer_engine; "
            "ensure device_memory_ratio < 1 so experts are actually offloaded"
        )
    return model, prefetcher


def _saturated_tensor_ids(prefetcher: Any) -> List[int]:
    """Every ``(layer, expert)`` tensor id -- the saturated ``E_l x L`` block."""
    tensor_map = prefetcher.expert_tensor_map
    if not tensor_map:
        raise RuntimeError("expert_tensor_map is empty; no experts to issue")
    return [tensor_id for _key, tensor_id in sorted(tensor_map.items())]


def _time_rounds(
    issue: Callable[[], None], warmup: int, iterations: int
) -> List[int]:
    for _ in range(warmup):
        issue()
    samples_ns: List[int] = []
    for _ in range(iterations):
        start = perf_counter_ns()
        issue()
        samples_ns.append(perf_counter_ns() - start)
    return samples_ns


def _python_per_expert_issue(
    engine: Any, tensor_ids: Sequence[int]
) -> Callable[[], None]:
    def issue() -> None:
        for tensor_id in tensor_ids:
            gpu_id = engine.get_node_default_device([tensor_id])
            engine.enqueue_prefetch(tensor_id, gpu_id)

    return issue


def _batched_issue(
    engine: Any, tensor_ids: Sequence[int]
) -> Optional[Callable[[], None]]:
    """Return a one-call batched issuer, or ``None`` if the native API is the
    pre-Task-8 no-op signature (probed once against a single tensor id)."""
    probe = list(tensor_ids[:1])
    try:
        engine.prefetch_tensors(probe)
    except Exception:
        return None

    ids = list(tensor_ids)

    def issue() -> None:
        engine.prefetch_tensors(ids)

    return issue


def run_issuance_benchmark(
    *,
    model_repo: str,
    offload_path: str,
    device_memory_ratio: float,
    modes: Sequence[str],
    warmup: int,
    iterations: int,
    window_us: float,
) -> Dict[str, Any]:
    model, prefetcher = _load_prefetcher(
        model_repo, offload_path, device_memory_ratio
    )
    engine = prefetcher.archer_engine
    tensor_ids = _saturated_tensor_ids(prefetcher)

    per_expert_ns: Optional[List[int]] = None
    batched_ns: Optional[List[int]] = None
    cpp_internal_ns: Optional[List[int]] = None
    unavailable: Dict[str, str] = {}

    if PYTHON_PER_EXPERT in modes:
        per_expert_ns = _time_rounds(
            _python_per_expert_issue(engine, tensor_ids), warmup, iterations
        )
    if BATCHED_PYBIND in modes:
        issuer = _batched_issue(engine, tensor_ids)
        if issuer is None:
            unavailable[BATCHED_PYBIND] = (
                "native prefetch_tensors(tensor_ids) batched API absent "
                "(pre-Task-8 no-op binding); rebuild _store to enable"
            )
        else:
            batched_ns = _time_rounds(issuer, warmup, iterations)
    if CPP_INTERNAL in modes:
        unavailable[CPP_INTERNAL] = (
            "no native in-C++ issuance timer exposed; reported null"
        )

    extra: Dict[str, Any] = {
        "offload_path": offload_path,
        "device_memory_ratio": device_memory_ratio,
        "requested_modes": list(modes),
    }
    if unavailable:
        extra["unavailable_modes"] = unavailable

    return build_bm2_report(
        model=model_repo,
        saturated_tensor_count=len(tensor_ids),
        window_us=window_us,
        per_expert_samples_ns=per_expert_ns,
        batched_samples_ns=batched_ns,
        cpp_internal_samples_ns=cpp_internal_ns,
        warmup=warmup,
        iterations=iterations,
        extra=extra,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.dflash.bench_prefetch_issuance",
        description="BM2 saturated route-ahead prefetch issuance micro-bench.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--offload-dir", required=True)
    parser.add_argument("--device-memory-ratio", type=float, default=0.9)
    parser.add_argument(
        "--mode", nargs="+", default=[PYTHON_PER_EXPERT], choices=ISSUANCE_MODES
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--window-json")
    parser.add_argument("--window-us", type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if not os.environ.get("MOE_DFLASH_SERVING_GPU"):
        parser.error(
            "MOE_DFLASH_SERVING_GPU must be set (opt-in GPU issuance bench)"
        )

    window_us = _resolve_window_us(args.window_json, args.window_us)
    report = run_issuance_benchmark(
        model_repo=args.model,
        offload_path=args.offload_dir,
        device_memory_ratio=args.device_memory_ratio,
        modes=args.mode,
        warmup=args.warmup,
        iterations=args.iterations,
        window_us=window_us,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


__all__ = [
    "PYTHON_PER_EXPERT",
    "BATCHED_PYBIND",
    "CPP_INTERNAL",
    "ISSUANCE_MODES",
    "Bm2Decision",
    "bm2_decision",
    "percentiles_us",
    "build_bm2_report",
    "run_issuance_benchmark",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
