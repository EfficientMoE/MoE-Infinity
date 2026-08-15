"""Opt-in RTX PRO 6000 B0-B3 serving experiment for PD-DFlash route-ahead.

Task 2 of ``docs/superpowers/plans/2026-08-14-pd-dflash-serving-scheduler.md``
("measure-first gating experiment"). This is the runner a human executes on the
GPU box; it records every design-doc §8 metric for the B0-B3 baselines so the
cost-model hide inequality (``report.py``) can be evaluated before any scheduler
or C++ work is justified.

The module is import-safe by construction: only the pure contract/scheduling
scaffolding lives at module scope, so ``pytest`` collection (and
``--dry-run-contract``) never imports torch, loads a checkpoint, initialises
CUDA, or touches the network. All hardware work is lazily imported inside
``run_experiment``.

Design contract (frozen here, cross-checked by the aggregator):

* baselines are exactly B0-B3 with the design-doc §8 semantics;
* the required generalization targets are ``Qwen/Qwen3-Coder-30B-A3B`` and
  ``openai/gpt-oss-20b`` with their ``z-lab`` DFlash drafts;
* block sizes are 8 and 16, concurrency sweeps 1..32; and
* every emitted observation carries the full ``REQUIRED_METRICS`` schema, plus
  the byte-accurate route-ahead ``wasted_prefetch_bytes`` from the instrumented
  ``RouteAheadStats`` (never an expert-count proxy).

At this Phase-A stage B2 (the 2-D deficit scheduler) does not yet exist, so the
runner emits an explicit ``BLOCKED_UNTIL_2D_SCHEDULER`` status for B2 rather than
silently emulating another baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from benchmarks.dflash.report import REQUIRED_METRICS

BASELINES = {
    "B0": "AR MoE on MoE-Infinity, offloaded, no speculative decoding",
    "B1": "DFlash with unchanged AR prefetcher",
    "B2": "DFlash with token-deficit scheduler and no expert-byte coupling",
    "B3": "target experts resident, no offload upper bound",
}

EXPERIMENTAL_CONFIGURATIONS = {
    "OURS": "DFlash with route-ahead prefetch and the 2-D co-designed scheduler",
}

REQUIRED_MODELS: Tuple[str, ...] = (
    "Qwen/Qwen3-Coder-30B-A3B",
    "openai/gpt-oss-20b",
)

REQUIRED_DRAFTS: Dict[str, str] = {
    "Qwen/Qwen3-Coder-30B-A3B": "z-lab/Qwen3-Coder-30B-A3B-DFlash",
    "openai/gpt-oss-20b": "z-lab/gpt-oss-20b-DFlash",
}

REQUIRED_BLOCK_SIZES: Tuple[int, ...] = (8, 16)
REQUIRED_CONCURRENCY: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)
DEFAULT_BASELINES: Tuple[str, ...] = ("B0", "B1", "B2", "B3")

OFFLOADED_BASELINES: Tuple[str, ...] = ("B0", "B1", "B2")
BLOCKED_STATUS = "BLOCKED_UNTIL_2D_SCHEDULER"

RTX_PRO_6000_NAME_FRAGMENT = "RTX PRO 6000"
RTX_PRO_6000_CAPABILITY: Tuple[int, int] = (12, 0)

# NVTX ranges the BM4 overlap parser (Task 10) associates H2D memcpys with; the
# runner must wrap the corresponding phases in these exact names.
NVTX_RANGES: Tuple[str, ...] = (
    "dflash_draft",
    "route_ahead_router",
    "route_ahead_issue",
    "target_verify",
    "expert_h2d",
)


# ---------------------------------------------------------------------------
# pure contract + validation helpers (CPU-only, unit-tested)
# ---------------------------------------------------------------------------


def build_contract_matrix() -> Dict[str, Any]:
    """Return the canonical §8 experiment contract as plain data.

    ``--dry-run-contract`` prints this and the GPU-gated test asserts it against
    the design doc, so the required models/drafts/baselines/sweeps are pinned in
    one place independent of any single invocation's CLI arguments.
    """
    return {
        "models": list(REQUIRED_MODELS),
        "drafts": dict(REQUIRED_DRAFTS),
        "baselines": dict(BASELINES),
        "experimental_configurations": dict(EXPERIMENTAL_CONFIGURATIONS),
        "block_sizes": list(REQUIRED_BLOCK_SIZES),
        "concurrency": list(REQUIRED_CONCURRENCY),
        "required_metrics": list(REQUIRED_METRICS),
        "nvtx_ranges": list(NVTX_RANGES),
    }


def validate_device_identity(
    device_name: str, capability: Tuple[int, int]
) -> None:
    """Raise unless the visible GPU is an RTX PRO 6000 with capability (12, 0).

    Kept free of torch so it is unit-testable; ``run_experiment`` feeds it the
    live ``torch.cuda`` values.
    """
    if RTX_PRO_6000_NAME_FRAGMENT not in device_name:
        raise RuntimeError(
            f"expected an {RTX_PRO_6000_NAME_FRAGMENT} GPU; got {device_name!r}"
        )
    if tuple(capability) != RTX_PRO_6000_CAPABILITY:
        raise RuntimeError(
            f"expected capability {RTX_PRO_6000_CAPABILITY}; got "
            f"{tuple(capability)!r}"
        )


def require_offloaded(baseline: str, offloaded_expert_count: int) -> None:
    """Raise if an offloaded baseline (B0/B1/B2) has no offloaded experts.

    Guards against mislabelling a resident run as offloaded evidence (plan
    dependency note on #137); B3 is the resident upper bound and is exempt.
    """
    if baseline in OFFLOADED_BASELINES and offloaded_expert_count <= 0:
        raise RuntimeError(
            f"baseline {baseline} requires offloaded target experts; the store "
            "reports none resident on host -- lower --device-memory-ratio below "
            "0.9 so experts actually offload"
        )


def observation_key(
    model: str, baseline: str, block_size: int, concurrency: int, repeat: int
) -> Tuple[str, str, int, int, int]:
    """The immutable identity of one measured row; two rows may never share it."""
    return (model, baseline, int(block_size), int(concurrency), int(repeat))


def make_observation_row(
    *,
    model: str,
    draft: str,
    baseline: str,
    block_size: int,
    concurrency: int,
    repeat: int,
    metrics: Mapping[str, float],
    cost_terms: Optional[Mapping[str, Any]] = None,
    status: Optional[str] = None,
    warnings: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Assemble one JSON observation row with the full §8 metric schema.

    A blocked row (``status`` set, e.g. B2's ``BLOCKED_UNTIL_2D_SCHEDULER``)
    carries no metrics; any other row must supply every ``REQUIRED_METRICS``
    entry, mirroring ``validate_result_matrix`` so a malformed row fails fast at
    write time rather than in the aggregator.
    """
    row: Dict[str, Any] = {
        "model": model,
        "draft": draft,
        "baseline": baseline,
        "block_size": int(block_size),
        "concurrency": int(concurrency),
        "repeat": int(repeat),
    }
    if status is not None:
        row["status"] = status
    else:
        missing = [m for m in REQUIRED_METRICS if m not in metrics]
        if missing:
            raise ValueError(
                f"observation missing metrics: {', '.join(missing)}"
            )
        for name in REQUIRED_METRICS:
            row[name] = metrics[name]
    if cost_terms:
        row["cost_terms"] = dict(cost_terms)
    if warnings:
        row["warnings"] = list(warnings)
    return row


def append_observation(output_path: str, row: Mapping[str, Any]) -> None:
    """Append ``row`` to the output JSON list, never overwriting a prior row.

    Rows are keyed by ``observation_key``; a duplicate key raises rather than
    silently clobbering an existing measurement (plan Task 2 step 6).
    """
    rows: List[Dict[str, Any]] = load_observations(output_path)
    new_key = observation_key(
        row["model"],
        row["baseline"],
        row["block_size"],
        row["concurrency"],
        row["repeat"],
    )
    for existing in rows:
        existing_key = observation_key(
            existing["model"],
            existing["baseline"],
            existing["block_size"],
            existing["concurrency"],
            existing["repeat"],
        )
        if existing_key == new_key:
            raise ValueError(f"refusing to overwrite existing row {new_key}")
    rows.append(dict(row))
    directory = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, output_path)


def load_observations(output_path: str) -> List[Dict[str, Any]]:
    """Read the JSON list of rows at ``output_path`` (``[]`` when absent/empty)."""
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return []
    with open(output_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{output_path} is not a JSON list of rows")
    return data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunnerArgs:
    model: Optional[str]
    draft: Optional[str]
    offload_dir: Optional[str]
    output: Optional[str]
    baselines: Tuple[str, ...]
    block_sizes: Tuple[int, ...]
    concurrency: Tuple[int, ...]
    requests: int
    warmup_rounds: int
    measured_h2d_gbps: Optional[float]
    slo_ms: Optional[float]
    seed: int
    device_memory_ratio: float
    dry_run_contract: bool


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.dflash.pd_dflash_serving",
        description=(
            "Opt-in RTX PRO 6000 B0-B3 route-ahead serving experiment; emits "
            "one JSON observation row per (model, baseline, block, "
            "concurrency, repeat) for benchmarks.dflash.report."
        ),
    )
    parser.add_argument("--model", help="HF target repo (offloaded MoE)")
    parser.add_argument("--draft", help="z-lab DFlash draft repo")
    parser.add_argument("--offload-dir", help="expert offload directory")
    parser.add_argument("--output", help="output JSON path for observations")
    parser.add_argument(
        "--baseline",
        nargs="+",
        default=list(DEFAULT_BASELINES),
        choices=sorted(BASELINES) + list(EXPERIMENTAL_CONFIGURATIONS),
        help="baselines/configurations to run (default: B0 B1 B2 B3)",
    )
    parser.add_argument(
        "--block-size",
        nargs="+",
        type=int,
        default=list(REQUIRED_BLOCK_SIZES),
        help="draft block sizes (default: 8 16)",
    )
    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        default=list(REQUIRED_CONCURRENCY),
        help="concurrent request counts (default: 1 2 4 8 16 32)",
    )
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--warmup-rounds", type=int, default=5)
    parser.add_argument("--measured-h2d-gbps", type=float, default=None)
    parser.add_argument("--slo-ms", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1408)
    parser.add_argument(
        "--device-memory-ratio",
        type=float,
        default=0.85,
        help="fraction of GPU memory for weights; <0.9 forces offload",
    )
    parser.add_argument(
        "--dry-run-contract",
        action="store_true",
        help="print the §8 experiment contract as JSON and exit (no GPU)",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> RunnerArgs:
    parsed = build_arg_parser().parse_args(argv)
    return RunnerArgs(
        model=parsed.model,
        draft=parsed.draft,
        offload_dir=parsed.offload_dir,
        output=parsed.output,
        baselines=tuple(parsed.baseline),
        block_sizes=tuple(parsed.block_size),
        concurrency=tuple(parsed.concurrency),
        requests=parsed.requests,
        warmup_rounds=parsed.warmup_rounds,
        measured_h2d_gbps=parsed.measured_h2d_gbps,
        slo_ms=parsed.slo_ms,
        seed=parsed.seed,
        device_memory_ratio=parsed.device_memory_ratio,
        dry_run_contract=parsed.dry_run_contract,
    )


def _require_run_args(args: RunnerArgs) -> None:
    missing = [
        flag
        for flag, value in (
            ("--model", args.model),
            ("--draft", args.draft),
            ("--offload-dir", args.offload_dir),
            ("--output", args.output),
        )
        if not value
    ]
    if missing:
        raise SystemExit(f"missing required args: {', '.join(missing)}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.dry_run_contract:
        json.dump(build_contract_matrix(), sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    _require_run_args(args)
    return run_experiment(args)


# ---------------------------------------------------------------------------
# hardware path: lazily imports torch / moe_infinity so module import stays cheap
# ---------------------------------------------------------------------------


def _validate_gpu_environment() -> None:
    import torch

    import moe_infinity._v4_fp4  # noqa: F401  (asserts native FP4 path present)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this host")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "expected exactly one visible GPU; set CUDA_VISIBLE_DEVICES=<id>"
        )
    validate_device_identity(
        torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)
    )


def run_experiment(args: RunnerArgs) -> int:
    """Drive the B0-B3 matrix on one RTX PRO 6000 and write observation rows.

    Loads each configuration through the real ``MoE`` + ``DFlashSpeculator``
    serving path (mirroring ``tests/python/dflash/test_gpu_serving_dflash.py``),
    wraps the draft/router/issue/verify/H2D phases in the frozen ``NVTX_RANGES``,
    reads byte-accurate coverage/waste from the instrumented ``RouteAheadStats``,
    and appends one row per ``(model, baseline, block, concurrency, repeat)``.
    B2 is emitted as ``BLOCKED_UNTIL_2D_SCHEDULER`` until Task 6 lands.
    """
    from benchmarks.dflash._serving_measure import measure_configuration

    assert args.model and args.draft and args.offload_dir and args.output
    _validate_gpu_environment()

    draft = args.draft or REQUIRED_DRAFTS.get(args.model, "")
    for baseline in args.baselines:
        for block_size in args.block_sizes:
            for concurrency in args.concurrency:
                if baseline == "B2":
                    append_observation(
                        args.output,
                        make_observation_row(
                            model=args.model,
                            draft=draft,
                            baseline=baseline,
                            block_size=block_size,
                            concurrency=concurrency,
                            repeat=0,
                            metrics={},
                            status=BLOCKED_STATUS,
                        ),
                    )
                    continue
                row = measure_configuration(
                    args=args,
                    baseline=baseline,
                    draft=draft,
                    block_size=block_size,
                    concurrency=concurrency,
                )
                append_observation(args.output, row)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
