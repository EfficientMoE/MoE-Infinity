"""Benchmark every cache policy vs Belady over **real** SGLang expert traces.

This is the real-trace counterpart of ``run_policies.py`` (which uses seeded
synthetic traces). It consumes converted Belady ``.pt`` dumps produced by
``convert_sglang_traces.py`` from an actual SGLang ``per_token``
expert-distribution capture, flattens them into an access sequence with the
same logic the Belady simulator uses, and runs the full policy x capacity
sweep with Belady optimal as the "Ideal" ceiling.

How prefill passes enter the stream is chosen with ``--prefill``:

* ``skip`` (default) - decode passes only (``forward_mode == DECODE``), one
  access per routed (token, layer, expert). Batch-1 decode is the
  latency-critical phase of offloaded MoE inference.
* ``union`` - each prefill pass contributes, per layer, one access per
  *distinct* routed expert (first-use order), i.e. one fetch per expert per
  layer per forward pass, as an offloading engine executes it; decode as in
  ``skip``.
* ``raw`` - prefill records flattened per token like decode
  (``--include-prefill``). Repeated hits on the same expert inside one prefill
  layer pass are counted as cache hits, which no engine would pay for.

Every number produced here is labelled ``real`` (provenance column), in
contrast to the ``synthetic`` rows written by ``run_policies.py``. The
pure-Python simulator (``belady_simulate`` + ``CachePolicy``) remains the
correctness oracle; the vectorized ``miss_ratio_curve`` is used only as an
independent Belady cross-check. Policy x capacity jobs run in parallel worker
processes (``--workers``).

Usage::

    python -m benchmarks.cache_simulator.research.run_policies_real \
        --files "<data>/belady/*.pt" \
        --model qwen3-30b-a3b \
        --dataset mmlu150-greedy64 \
        --ignore-layers 0 \
        --capacities 128 256 512 1024 2048 \
        --out benchmarks/cache_simulator/research
"""

from __future__ import annotations

import argparse
import csv
import glob
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from benchmarks.cache_simulator.belady_cache_simulator import (
    belady_simulate,
    extract_access_sequence,
    is_decode_record,
    load_records,
)
from benchmarks.cache_simulator.policies import (
    GEOMETRY_POLICIES,
    ORACLE_POLICIES,
    build_policy,
    simulate,
)
from benchmarks.cache_simulator.research.mrc import miss_ratio_curve

Key = Tuple[int, int]

#: Every policy the sweep knows, in table order.
POLICY_ORDER = [
    "lru",
    "lfu",
    "lfu_k",
    "model_tracing",
    "score",
    "lhu",
    "arc",
    "2q",
    "lru_k",
    "lirs",
    "s3fifo",
    "sieve",
    "w_tinylfu",
    "layer_lru",
    "lcnu",
    "rothead",
    "hotfit",
    "shadowfit",
    "decayfit",
    "costfit",
    "costfit3",
    "shared_costfit",
    "shared_costfit3",
    "single_layer_ahead",
    "stride_prefetch",
    "multi_layer_ahead",
    "activation_matrix",
]

DEFAULT_POLICIES = list(POLICY_ORDER)

PRETTY = {
    "lru": "LRU",
    "lfu": "LFU",
    "lfu_k": "LFU-K",
    "model_tracing": "Model-tracing",
    "score": "Score-based",
    "lhu": "LHU",
    "arc": "ARC",
    "2q": "2Q",
    "lru_k": "LRU-2",
    "lirs": "LIRS",
    "s3fifo": "S3-FIFO",
    "sieve": "SIEVE",
    "w_tinylfu": "W-TinyLFU",
    "layer_lru": "Per-layer LRU",
    "lcnu": "LCNU (layer-cycle next-use)",
    "rothead": "RotHead (rotation-adaptive)",
    "hotfit": "HotFit (hot-set-fit adaptive)",
    "shadowfit": "ShadowFit (cumulative-rate shadow arbitration)",
    "decayfit": "DecayFit (decayed-rate shadow arbitration)",
    "costfit": "CostFit (self-limiting switch-cost arbitration)",
    "costfit3": "CostFit3 (3-candidate self-limiting arbitration)",
    "shared_costfit": "SharedCostFit (CostFit, shared per-key metadata table)",
    "shared_costfit3": "SharedCostFit3 (CostFit3, shared per-key metadata table)",
    "single_layer_ahead": "1-layer-ahead*",
    "stride_prefetch": "Stride*",
    "multi_layer_ahead": "Multi-layer*",
    "activation_matrix": "Activation-matrix*",
}

PREFILL_MODES = ("skip", "union", "raw")


def expand_files(patterns: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) or [pattern]
        for match in matches:
            if match not in seen:
                seen.add(match)
                out.append(match)
    return out


def _to_list(tensor) -> list:
    return tensor.tolist() if hasattr(tensor, "tolist") else list(tensor)


def union_prefill_sequence(
    records: Sequence[dict], *, ignore_layers: int
) -> List[Key]:
    """Prefill as fetched: per pass and layer, each distinct expert once.

    Decode records are flattened exactly like ``extract_access_sequence``;
    non-decode records contribute the distinct experts of each layer in
    first-use order.
    """
    accesses: List[Key] = []
    for index, record in enumerate(records):
        if is_decode_record(record, index):
            accesses.extend(
                extract_access_sequence(
                    [record], ignore_layers=ignore_layers, include_prefill=True
                )
            )
            continue
        tensor = record["topk_ids_of_layer"]
        num_layers = tensor.shape[0]
        flattened = tensor.reshape(num_layers, -1)
        for layer_idx in range(ignore_layers, num_layers):
            seen = set()
            for expert_id in _to_list(flattened[layer_idx]):
                if expert_id >= 0 and expert_id not in seen:
                    seen.add(expert_id)
                    accesses.append((layer_idx, int(expert_id)))
    return accesses


def _top_k(records: Sequence[dict]) -> int:
    """Routed experts per (token, layer) in the first decode record."""
    for index, record in enumerate(records):
        if is_decode_record(record, index):
            valid = record["topk_ids_of_layer"] >= 0
            return int(valid.sum(-1).max())
    return 0


def load_accesses(
    files: Sequence[str], *, ignore_layers: int, prefill: str = "skip"
) -> Tuple[List[Key], Dict[str, int]]:
    """Concatenate the access sequences of every trace file in order.

    Returns the flattened ``(layer, expert)`` access list plus a small stats
    dict (files, layers, max expert id, top-k, distinct keys, accesses).
    """
    if prefill not in PREFILL_MODES:
        raise ValueError(f"prefill must be one of {PREFILL_MODES}")
    accesses: List[Key] = []
    num_layers = 0
    top_k = 0
    for path in files:
        records = load_records(path)
        if records:
            num_layers = max(
                num_layers, int(records[0]["topk_ids_of_layer"].shape[0])
            )
            top_k = max(top_k, _top_k(records))
        if prefill == "union":
            seq = union_prefill_sequence(records, ignore_layers=ignore_layers)
        else:
            seq = extract_access_sequence(
                records,
                ignore_layers=ignore_layers,
                include_prefill=prefill == "raw",
            )
        accesses.extend(seq)
    max_expert = max((expert for _, expert in accesses), default=-1)
    stats = {
        "files": len(files),
        "num_layers": num_layers,
        "max_expert_id": max_expert,
        "top_k": top_k,
        "distinct_keys": len(set(accesses)),
        "total_accesses": len(accesses),
    }
    return accesses, stats


def geometry_kwargs(name: str, stats: Dict[str, int]) -> Dict[str, float]:
    """Model geometry for the policies that use it (see GEOMETRY_POLICIES)."""
    if name not in GEOMETRY_POLICIES:
        return {}
    kwargs: Dict[str, float] = {"num_layers": stats["num_layers"]}
    num_experts = stats["max_expert_id"] + 1
    if name in ("lcnu", "rothead", "hotfit", "shadowfit", "decayfit", "costfit", "costfit3", "shared_costfit", "shared_costfit3") and stats["top_k"] > 0 and num_experts > 0:
        kwargs["prior"] = min(1.0, stats["top_k"] / num_experts)
    return kwargs


@dataclass
class Row:
    model: str
    dataset: str
    policy: str
    capacity: int
    hit_rate: float
    effective_hit_rate: float
    is_oracle: bool


# Worker state, inherited through fork so the access list is not pickled.
_ACCESSES: List[Key] = []
_STATS: Dict[str, int] = {}


def _run_job(job: Tuple[str, int]) -> Tuple[str, int, float, float]:
    name, cap = job
    if name == "belady":
        _, _, rate = belady_simulate(_ACCESSES, cap)
        return name, cap, rate, rate
    policy = build_policy(name, cap, **geometry_kwargs(name, _STATS))
    _, _, rate = simulate(policy, _ACCESSES)
    return name, cap, rate, policy.effective_hit_rate


def run_sweep(
    accesses: Sequence[Key],
    capacities: Sequence[int],
    model: str,
    dataset: str,
    stats: Dict[str, int],
    policies: Sequence[str] = DEFAULT_POLICIES,
    workers: int = 1,
) -> List[Row]:
    global _ACCESSES, _STATS
    _ACCESSES, _STATS = list(accesses), dict(stats)
    num_experts = max(stats["max_expert_id"] + 1, 1)
    # Independent Belady cross-check via the vectorized MRC engine.
    mrc = miss_ratio_curve(
        _ACCESSES, capacities, policy="opt", num_experts=num_experts
    )
    jobs = [("belady", cap) for cap in capacities] + [
        (name, cap) for name in policies for cap in capacities
    ]
    if workers > 1:
        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=min(workers, len(jobs)), mp_context=ctx
        ) as pool:
            results = list(pool.map(_run_job, jobs))
    else:
        results = [_run_job(job) for job in jobs]
    by_job = {(name, cap): (rate, eff) for name, cap, rate, eff in results}

    rows: List[Row] = []
    for cap in capacities:
        belady_rate = by_job[("belady", cap)][0]
        assert abs(belady_rate - mrc[cap]) < 1e-9, (
            f"Belady reference vs MRC mismatch at cap {cap}: "
            f"{belady_rate} != {mrc[cap]}"
        )
        rows.append(
            Row(model, dataset, "belady", cap, belady_rate, belady_rate, False)
        )
        for name in policies:
            rate, eff = by_job[(name, cap)]
            assert (
                rate <= belady_rate + 1e-9
            ), f"{name} at cap {cap} beat Belady: {rate} > {belady_rate}"
            rows.append(
                Row(
                    model,
                    dataset,
                    name,
                    cap,
                    rate,
                    eff,
                    name in ORACLE_POLICIES,
                )
            )
    return rows


def write_csv(rows: Sequence[Row], path: Path) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "model",
                "dataset",
                "policy",
                "capacity",
                "hit_rate",
                "effective_hit_rate",
                "is_oracle_upper_bound",
                "provenance",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.model,
                    row.dataset,
                    row.policy,
                    row.capacity,
                    f"{row.hit_rate:.6f}",
                    f"{row.effective_hit_rate:.6f}",
                    int(row.is_oracle),
                    "real",
                ]
            )


def render_table(rows: Sequence[Row], commit: str, header: str) -> str:
    caps = sorted({r.capacity for r in rows})
    lookup = {(r.policy, r.capacity): r.hit_rate for r in rows}
    policies = [p for p in POLICY_ORDER if any(r.policy == p for r in rows)]
    lines: List[str] = [header, "", f"Commit: `{commit}`", ""]

    th = "| Policy | " + " | ".join(f"cap={c}" for c in caps) + " |"
    sep = "| --- | " + " | ".join("---" for _ in caps) + " |"
    lines.append(th)
    lines.append(sep)

    def fmt(label: str, policy: str) -> str:
        cells = []
        for cap in caps:
            val = lookup.get((policy, cap))
            cells.append(f"{val:.4f}" if val is not None else "-")
        return f"| {label} | " + " | ".join(cells) + " |"

    lines.append(fmt("Belady (Ideal)", "belady"))
    for policy in policies:
        lines.append(fmt(PRETTY[policy], policy))
    if any(p in ORACLE_POLICIES for p in policies):
        lines.append(
            "\n`*` = trace-oracle prefetch policy (perfect next-layer routing "
            "lookahead); reported demand hit rate is an upper bound on an "
            "achievable predictor. See the CSV for `effective_hit_rate`."
        )
    return "\n".join(lines)


EXPERIMENTS_HEADER = (
    "| policy | model | dataset | capacity | hit_rate | eff_hit_rate "
    "| provenance | commit |\n"
    "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
)


def append_experiments(
    rows: Sequence[Row], path: Path, commit: str, command: str, env: str
) -> None:
    with open(path, "a") as handle:
        handle.write("\n" + EXPERIMENTS_HEADER)
        for row in rows:
            handle.write(
                f"| {row.policy} | {row.model} | {row.dataset} "
                f"| {row.capacity} | {row.hit_rate:.4f} "
                f"| {row.effective_hit_rate:.4f} | real | {commit} |\n"
            )
        handle.write(
            f"\n<!-- run: {command} | env: {env} | commit: {commit} -->\n"
        )


def _git_commit() -> str:
    import subprocess

    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    commit = out.stdout.strip() or "unknown"
    dirty = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", "benchmarks/cache_simulator"],
        capture_output=True,
        check=False,
    )
    return f"{commit}+dirty" if dirty.returncode == 1 else commit


def _env_string() -> str:
    import platform

    import numpy as np

    return (
        f"python {platform.python_version()}, numpy {np.__version__}, "
        f"{platform.system()} {platform.machine()}"
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark cache policies vs Belady on REAL SGLang traces.",
    )
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ignore-layers", type=int, default=0)
    parser.add_argument(
        "--prefill",
        choices=PREFILL_MODES,
        default="skip",
        help="How prefill passes enter the stream (default: decode only).",
    )
    parser.add_argument(
        "--include-prefill",
        action="store_true",
        help="Alias for --prefill raw.",
    )
    parser.add_argument(
        "--capacities",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024, 2048],
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=DEFAULT_POLICIES,
        choices=POLICY_ORDER,
        metavar="POLICY",
        help=f"Policies to run (default: all). Choices: {POLICY_ORDER}",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="Parallel worker processes for the policy x capacity jobs.",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent),
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Suffix for the results file (results_real_<tag>.md).",
    )
    parser.add_argument(
        "--no-experiments-log",
        action="store_true",
        help="Do not append rows to EXPERIMENTS.md.",
    )
    args = parser.parse_args(argv)
    if args.include_prefill:
        args.prefill = "raw"
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    files = expand_files(args.files)
    if not files:
        raise ValueError("No input files found.")

    accesses, stats = load_accesses(
        files, ignore_layers=args.ignore_layers, prefill=args.prefill
    )
    if not accesses:
        raise ValueError(
            "No accesses extracted; check --ignore-layers and that the traces "
            "carry decode records."
        )

    dataset = (
        f"{args.dataset}/"
        + {
            "skip": "decode",
            "union": "decode+prefill-union",
            "raw": "decode+prefill-raw",
        }[args.prefill]
    )
    rows = run_sweep(
        accesses,
        args.capacities,
        args.model,
        dataset,
        stats,
        policies=args.policies,
        workers=args.workers,
    )

    commit = _git_commit()
    env = _env_string()
    command = (
        "python -m benchmarks.cache_simulator.research.run_policies_real "
        f"--model {args.model} --dataset {args.dataset} "
        f"--ignore-layers {args.ignore_layers} --prefill {args.prefill}"
    )

    stream = {
        "skip": "decode passes only",
        "union": "decode + prefill as one fetch per distinct expert per layer",
        "raw": "decode + prefill flattened per token",
    }[args.prefill]
    header = (
        f"# Real cache-policy benchmark — {args.model} / {dataset}\n\n"
        "All numbers are **REAL** — computed over an actual SGLang "
        "`per_token` expert-routing capture (not synthetic). Values are the "
        "demand **hit rate**; `belady` is the Belady-optimal ceiling "
        "('Ideal').\n\n"
        f"Trace stats: {stats['files']} file(s), {stats['num_layers']} "
        f"layers, max expert id {stats['max_expert_id']}, top-k "
        f"{stats['top_k']}, {stats['distinct_keys']} distinct (layer, expert) "
        f"keys, {stats['total_accesses']} accesses ({stream}; "
        f"ignore_layers={args.ignore_layers})."
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.model
    md_path = out_dir / f"results_real_{tag}.md"
    csv_path = out_dir / f"results_real_{tag}.csv"
    md_path.write_text(render_table(rows, commit, header) + "\n")
    write_csv(rows, csv_path)
    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")

    if not args.no_experiments_log:
        append_experiments(
            rows,
            out_dir / "EXPERIMENTS.md",
            commit=commit,
            command=command,
            env=env,
        )
        print(f"Appended {len(rows)} real rows to {out_dir / 'EXPERIMENTS.md'}")
    print(f"Trace stats: {stats}")


if __name__ == "__main__":
    main()
