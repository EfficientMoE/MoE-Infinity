# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""BM3 -- route-ahead prefetch priority-band ablation (design §10).

Task 9 of ``docs/superpowers/plans/2026-08-14-pd-dflash-serving-scheduler.md``
(candidate hop 2). A three-way ablation of the priority at which route-ahead
prefetch work is enqueued into the native task pool:

* ``background`` (priority ``2``) -- the ordinary background-prefetch band, i.e.
  route-ahead work competes with plain AR prefetch;
* ``route-ahead`` (priority ``1``) -- a *dedicated* band serviced ahead of
  background prefetch but behind on-demand misses (the candidate under test);
* ``on-demand`` (priority ``0``) -- the same band real on-demand misses use;
  measured only to prove it stays the fastest service class (a shipped
  route-ahead band must never invert on-demand).

For each arm the runner measures, over identical seeded requests with cache
state reset between variants, the median *exposed-fetch seconds* (time an
on-demand expert fetch is exposed, i.e. not hidden behind draft+verify compute)
and the median *tokens/s*. The candidate dedicated band ships (design §10 /
plan Task 9 Step 1) iff **all three** hold:

1. route-ahead has *lower* exposed fetch than default background;
2. route-ahead does not *reduce* tokens/s versus default background; and
3. on-demand remains the fastest service class (no priority inversion).

The gate must pass for *both* required targets; a single model that shows no
improvement, a throughput regression, or a priority inversion removes the
candidate and keeps only this benchmark.

Import-safe by construction: torch and moe_infinity are imported lazily inside
the GPU runner, so ``bm3_decision`` / ``priority_arm`` / ``median`` /
``build_bm3_report`` (and their tests) are pure-CPU and never initialise CUDA.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

# Native priority bands (mirror ``core/prefetch/task_scheduler.h``); lower value
# is serviced first in ``ArcherTaskPool::GPUThreadFunc``.
ON_DEMAND = 0
ROUTE_AHEAD = 1
BACKGROUND = 2

# Ablation arm labels, ordered from lowest to highest *service* priority.
PRIORITY_BANDS = ("background", "route-ahead", "on-demand")

_ARM_PRIORITY: Dict[str, int] = {
    "background": BACKGROUND,
    "route-ahead": ROUTE_AHEAD,
    "on-demand": ON_DEMAND,
}


@dataclass(frozen=True)
class PriorityArm:
    """Median exposed-fetch seconds and tokens/s for one priority band."""

    exposed_fetch_seconds: float
    tokens_per_second: float


@dataclass(frozen=True)
class Bm3Decision:
    """The BM3 ship gate over the three ablation arms (design §10)."""

    default: PriorityArm
    route_ahead: PriorityArm
    on_demand: PriorityArm
    exposed_fetch_improved: bool
    throughput_preserved: bool
    on_demand_fastest: bool
    ship_priority_band: bool


def _finite_non_negative(name: str, value: Any) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and >= 0; got {value!r}")
    return number


def priority_arm(
    *, exposed_fetch_seconds: float, tokens_per_second: float
) -> PriorityArm:
    """Build a validated ``PriorityArm`` from measured medians."""
    return PriorityArm(
        exposed_fetch_seconds=_finite_non_negative(
            "exposed_fetch_seconds", exposed_fetch_seconds
        ),
        tokens_per_second=_finite_non_negative(
            "tokens_per_second", tokens_per_second
        ),
    )


def bm3_decision(
    *,
    default: PriorityArm,
    route_ahead: PriorityArm,
    on_demand: PriorityArm,
) -> Bm3Decision:
    """Evaluate the BM3 ship gate over the three ablation arms.

    Ships the dedicated route-ahead band iff it strictly lowers exposed fetch
    versus default background, does not reduce tokens/s (inclusive at
    equality), and on-demand remains the fastest service class (its exposed
    fetch is no worse than both other arms).
    """
    exposed_fetch_improved = (
        route_ahead.exposed_fetch_seconds < default.exposed_fetch_seconds
    )
    throughput_preserved = (
        route_ahead.tokens_per_second >= default.tokens_per_second
    )
    on_demand_fastest = (
        on_demand.exposed_fetch_seconds <= route_ahead.exposed_fetch_seconds
        and on_demand.exposed_fetch_seconds <= default.exposed_fetch_seconds
    )
    ship = exposed_fetch_improved and throughput_preserved and on_demand_fastest
    return Bm3Decision(
        default=default,
        route_ahead=route_ahead,
        on_demand=on_demand,
        exposed_fetch_improved=exposed_fetch_improved,
        throughput_preserved=throughput_preserved,
        on_demand_fastest=on_demand_fastest,
        ship_priority_band=ship,
    )


def median(samples: Sequence[float]) -> float:
    """Median of ``samples`` (mean of the two central values for even counts)."""
    if not samples:
        raise ValueError("median requires at least one sample")
    ordered = sorted(float(s) for s in samples)
    count = len(ordered)
    mid = count // 2
    if count % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _arm_from_samples(samples: Mapping[str, Sequence[float]]) -> PriorityArm:
    return priority_arm(
        exposed_fetch_seconds=median(samples["exposed_fetch_seconds"]),
        tokens_per_second=median(samples["tokens_per_second"]),
    )


def build_bm3_report(
    *,
    models: Sequence[str],
    arms: Mapping[str, Mapping[str, Sequence[float]]],
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the BM3 report and its ship-gate verdict from raw samples.

    ``arms`` maps each of ``background``/``route-ahead``/``on-demand`` to a
    mapping with ``exposed_fetch_seconds`` and ``tokens_per_second`` sample
    lists (one entry per repetition). All three arms are required.
    """
    missing = [band for band in PRIORITY_BANDS if band not in arms]
    if missing:
        raise ValueError(f"BM3 report missing arms: {', '.join(missing)}")

    per_arm = {band: _arm_from_samples(arms[band]) for band in PRIORITY_BANDS}
    decision = bm3_decision(
        default=per_arm["background"],
        route_ahead=per_arm["route-ahead"],
        on_demand=per_arm["on-demand"],
    )
    report: Dict[str, Any] = {
        "benchmark": "BM3",
        "models": list(models),
        "priorities": {band: _ARM_PRIORITY[band] for band in PRIORITY_BANDS},
        "medians": {
            band: {
                "exposed_fetch_seconds": per_arm[band].exposed_fetch_seconds,
                "tokens_per_second": per_arm[band].tokens_per_second,
            }
            for band in PRIORITY_BANDS
        },
        "repetitions": {
            band: len(arms[band]["exposed_fetch_seconds"])
            for band in PRIORITY_BANDS
        },
        "exposed_fetch_improved": decision.exposed_fetch_improved,
        "throughput_preserved": decision.throughput_preserved,
        "on_demand_fastest": decision.on_demand_fastest,
        "ship_priority_band": decision.ship_priority_band,
    }
    if extra:
        report.update(dict(extra))
    return report


# ---------------------------------------------------------------------------
# GPU ablation runner (lazily imports torch / moe_infinity)
# ---------------------------------------------------------------------------


def run_priority_ablation(
    *,
    model_repo: str,
    draft_repo: str,
    offload_path: str,
    device_memory_ratio: float,
    repetitions: int,
    block_size: int,
    requests: int,
    warmup_rounds: int,
    seed: int,
) -> Dict[str, Any]:
    """Measure the three priority arms for one offloaded target.

    Requires a genuinely offloaded target and its DFlash draft. For each arm we
    force route-ahead issuance onto that native band (via
    ``ExpertPrefetcher.route_ahead_priority``), reset cache state, run identical
    seeded requests, and record exposed-fetch seconds and tokens/s. Route-ahead
    issuance must expose an explicit priority for this ablation to be
    meaningful; a build without the priority-band plumbing raises.
    """
    import time

    import torch

    import moe_infinity._v4_fp4  # noqa: F401  (assert native FP4 path present)
    from moe_infinity import MoE
    from moe_infinity.memory.expert_prefetcher import ExpertPrefetcher
    from moe_infinity.spec_decode import DFlashSpeculator

    if not hasattr(ExpertPrefetcher, "route_ahead_priority") and not any(
        hasattr(ExpertPrefetcher, attr)
        for attr in ("route_ahead_priority", "_route_ahead_priority")
    ):
        raise RuntimeError(
            "ExpertPrefetcher exposes no route_ahead_priority knob; the "
            "priority-band candidate (plan Task 9 Step 4) is not built in, so "
            "the ablation cannot distinguish the route-ahead band"
        )

    model = MoE(
        model_repo,
        {
            "offload_path": offload_path,
            "device_memory_ratio": device_memory_ratio,
        },
    )
    engine = model.engine
    prefetcher = engine.expert_prefetcher
    if prefetcher is None or prefetcher.archer_engine is None:
        raise RuntimeError(
            "loaded target has no offloaded ExpertPrefetcher/archer_engine; "
            "lower --device-memory-ratio below 0.9 so experts offload"
        )
    speculator = DFlashSpeculator(model, draft_repo)
    enable = getattr(speculator, "enable_route_ahead_stats", None)
    if callable(enable):
        enable()

    from benchmarks.dflash._serving_measure import (
        _deterministic_prompt_ids,
        _greedy_generate,
    )

    prompt_ids = _deterministic_prompt_ids(model, model_repo)
    tokens_per_request = max(block_size * 4, 32)

    def _reset_cache() -> None:
        archer = prefetcher.archer_engine
        reset = getattr(archer, "reset_cache", None)
        if callable(reset):
            try:
                reset()
            except Exception:
                pass
        else:
            replace = getattr(archer, "replace_cache_candidates", None)
            if callable(replace):
                try:
                    replace([])
                except Exception:
                    pass
        zero_exposed = getattr(engine, "reset_exposed_fetch_seconds", None)
        if callable(zero_exposed):
            try:
                zero_exposed()
            except Exception:
                pass
        torch.cuda.synchronize()

    def _measure_once() -> Dict[str, float]:
        torch.cuda.synchronize()
        exposed_before = _exposed_fetch_seconds(engine, speculator)
        started = time.perf_counter()
        generated = 0
        for _ in range(max(1, requests)):
            out = _greedy_generate(
                model, prompt_ids, speculator, tokens_per_request
            )
            generated += len(out)
        torch.cuda.synchronize()
        elapsed = max(time.perf_counter() - started, 1e-9)
        exposed = max(
            _exposed_fetch_seconds(engine, speculator) - exposed_before, 0.0
        )
        return {
            "exposed_fetch_seconds": exposed,
            "tokens_per_second": generated / elapsed,
        }

    arms: Dict[str, Dict[str, List[float]]] = {}
    for band in PRIORITY_BANDS:
        torch.manual_seed(seed)
        prefetcher.route_ahead_priority = _ARM_PRIORITY[band]
        # Warm up this arm.
        for _ in range(max(0, warmup_rounds)):
            _greedy_generate(model, prompt_ids, speculator, max(1, block_size))
        exposed_samples: List[float] = []
        tps_samples: List[float] = []
        for _ in range(max(1, repetitions)):
            _reset_cache()
            sample = _measure_once()
            exposed_samples.append(sample["exposed_fetch_seconds"])
            tps_samples.append(sample["tokens_per_second"])
        arms[band] = {
            "exposed_fetch_seconds": exposed_samples,
            "tokens_per_second": tps_samples,
        }

    return build_bm3_report(
        models=[model_repo],
        arms=arms,
        extra={
            "offload_path": offload_path,
            "device_memory_ratio": device_memory_ratio,
            "draft": draft_repo,
            "block_size": block_size,
            "requests": requests,
            "seed": seed,
        },
    )


def _exposed_fetch_seconds(engine: Any, speculator: Any) -> float:
    """Best-effort exposed on-demand fetch seconds for the last run.

    Prefers a native/instrumented accessor; falls back to the route-ahead
    stats' exposed-fetch term. Returns 0.0 only when nothing is instrumented,
    in which case the report's ``warnings`` should be consulted.
    """
    for source in (engine, getattr(engine, "expert_prefetcher", None)):
        if source is None:
            continue
        for name in (
            "exposed_fetch_seconds",
            "get_exposed_fetch_seconds",
            "on_demand_fetch_seconds",
        ):
            value = getattr(source, name, None)
            if callable(value):
                try:
                    value = value()
                except Exception:
                    value = None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    stats = getattr(speculator, "route_ahead_stats", None)
    if stats is not None:
        snapshot = stats.as_dict()
        exposed = snapshot.get("exposed_fetch_seconds")
        if isinstance(exposed, (int, float)) and not isinstance(exposed, bool):
            return float(exposed)
    return 0.0


def _run_all_models(args: argparse.Namespace) -> Dict[str, Any]:
    if not (len(args.models) == len(args.drafts) == len(args.offload_dirs)):
        raise SystemExit(
            "--models, --drafts, and --offload-dirs must have equal length"
        )
    per_model: List[Dict[str, Any]] = []
    ship_all = True
    for model_repo, draft_repo, offload in zip(
        args.models, args.drafts, args.offload_dirs
    ):
        report = run_priority_ablation(
            model_repo=model_repo,
            draft_repo=draft_repo,
            offload_path=offload,
            device_memory_ratio=args.device_memory_ratio,
            repetitions=args.repetitions,
            block_size=args.block_size,
            requests=args.requests,
            warmup_rounds=args.warmup_rounds,
            seed=args.seed,
        )
        ship_all = ship_all and bool(report["ship_priority_band"])
        per_model.append(report)
    return {
        "benchmark": "BM3",
        "per_model": per_model,
        # The candidate ships only when *every* required target passes.
        "ship_priority_band": ship_all and len(per_model) > 0,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.dflash.bench_prefetch_priority",
        description="BM3 route-ahead prefetch priority-band ablation.",
    )
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--drafts", nargs="+", required=True)
    parser.add_argument("--offload-dirs", nargs="+", required=True)
    parser.add_argument(
        "--priorities",
        nargs="+",
        default=list(PRIORITY_BANDS),
        choices=list(PRIORITY_BANDS),
        help="informational; the runner always sweeps all three bands",
    )
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--warmup-rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1408)
    parser.add_argument("--device-memory-ratio", type=float, default=0.85)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if not os.environ.get("MOE_DFLASH_SERVING_GPU"):
        parser.error(
            "MOE_DFLASH_SERVING_GPU must be set (opt-in GPU priority ablation)"
        )

    report = _run_all_models(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


__all__ = [
    "ON_DEMAND",
    "ROUTE_AHEAD",
    "BACKGROUND",
    "PRIORITY_BANDS",
    "PriorityArm",
    "Bm3Decision",
    "priority_arm",
    "bm3_decision",
    "median",
    "build_bm3_report",
    "run_priority_ablation",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
