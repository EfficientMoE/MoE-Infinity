"""Benchmark every cache policy vs Belady over **real** SGLang expert traces.

This is the real-trace counterpart of ``run_policies.py`` (which uses seeded
synthetic traces). It consumes converted Belady ``.pt`` dumps produced by
``convert_sglang_traces.py`` from an actual SGLang ``per_token``
expert-distribution capture, flattens them into a decode access sequence with
the same logic the Belady simulator uses, and runs the full policy x capacity
sweep with Belady optimal as the "Ideal" ceiling.

Every number produced here is labelled ``real`` (provenance column), in
contrast to the ``synthetic`` rows written by ``run_policies.py``. The
pure-Python simulator (``belady_simulate`` + ``CachePolicy``) remains the
correctness oracle; the vectorized ``miss_ratio_curve`` is used only as an
independent Belady cross-check.

Usage::

    python -m benchmarks.cache_simulator.research.run_policies_real \
        --files "<data>/belady/*.pt" \
        --model qwen1.5-moe-a2.7b \
        --dataset mmlu-sharegpt \
        --ignore-layers 0 \
        --capacities 64 128 256 512 1024 2048 \
        --out benchmarks/cache_simulator/research
"""

from __future__ import annotations

import argparse
import csv
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from benchmarks.cache_simulator.belady_cache_simulator import (
    belady_simulate,
    extract_access_sequence,
    load_records,
)
from benchmarks.cache_simulator.policies import (
    ORACLE_POLICIES,
    build_policy,
    simulate,
)
from benchmarks.cache_simulator.research.mrc import miss_ratio_curve

Key = Tuple[int, int]

POLICY_ORDER = [
    "lru",
    "lfu",
    "lfu_k",
    "model_tracing",
    "score",
    "lhu",
    "single_layer_ahead",
    "stride_prefetch",
    "multi_layer_ahead",
    "activation_matrix",
]

PRETTY = {
    "lru": "LRU",
    "lfu": "LFU",
    "lfu_k": "LFU-K",
    "model_tracing": "Model-tracing",
    "score": "Score-based",
    "lhu": "LHU",
    "single_layer_ahead": "1-layer-ahead*",
    "stride_prefetch": "Stride*",
    "multi_layer_ahead": "Multi-layer*",
    "activation_matrix": "Activation-matrix*",
}


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


def load_accesses(
    files: Sequence[str], *, ignore_layers: int, include_prefill: bool
) -> Tuple[List[Key], Dict[str, int]]:
    """Concatenate the decode access sequences of every trace file in order.

    Returns the flattened ``(layer, expert)`` access list plus a small stats
    dict (files, layers, max expert id, distinct keys, total accesses).
    """
    accesses: List[Key] = []
    num_layers = 0
    max_expert = -1
    for path in files:
        records = load_records(path)
        if records:
            num_layers = max(
                num_layers, int(records[0]["topk_ids_of_layer"].shape[0])
            )
        seq = extract_access_sequence(
            records,
            ignore_layers=ignore_layers,
            include_prefill=include_prefill,
        )
        for _, expert in seq:
            if expert > max_expert:
                max_expert = expert
        accesses.extend(seq)
    stats = {
        "files": len(files),
        "num_layers": num_layers,
        "max_expert_id": max_expert,
        "distinct_keys": len(set(accesses)),
        "total_accesses": len(accesses),
    }
    return accesses, stats


@dataclass
class Row:
    model: str
    dataset: str
    policy: str
    capacity: int
    hit_rate: float
    effective_hit_rate: float
    is_oracle: bool


def run_sweep(
    accesses: Sequence[Key],
    capacities: Sequence[int],
    model: str,
    dataset: str,
    num_experts: int,
) -> List[Row]:
    rows: List[Row] = []
    # Independent Belady cross-check via the vectorized MRC engine.
    mrc = miss_ratio_curve(
        accesses, capacities, policy="opt", num_experts=max(num_experts, 1)
    )
    for cap in capacities:
        _, _, belady_rate = belady_simulate(accesses, cap)
        assert abs(belady_rate - mrc[cap]) < 1e-9, (
            f"Belady reference vs MRC mismatch at cap {cap}: "
            f"{belady_rate} != {mrc[cap]}"
        )
        rows.append(
            Row(model, dataset, "belady", cap, belady_rate, belady_rate, False)
        )
        for name in POLICY_ORDER:
            policy = build_policy(name, cap)
            _, _, rate = simulate(policy, accesses)
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
                    policy.effective_hit_rate,
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
    for policy in POLICY_ORDER:
        lines.append(fmt(PRETTY[policy], policy))
    lines.append(
        "\n`*` = trace-oracle prefetch policy (perfect next-layer routing "
        "lookahead); reported demand hit rate is an upper bound on an "
        "achievable predictor. See the CSV for `effective_hit_rate`."
    )
    return "\n".join(lines)


def append_experiments(
    rows: Sequence[Row], path: Path, commit: str, command: str, env: str
) -> None:
    with open(path, "a") as handle:
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
    return out.stdout.strip() or "unknown"


def _env_string() -> str:
    import platform

    import numpy as np

    return (
        f"python {platform.python_version()}, numpy {np.__version__}, "
        f"{platform.system()} {platform.machine()}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark cache policies vs Belady on REAL SGLang traces.",
    )
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ignore-layers", type=int, default=0)
    parser.add_argument("--include-prefill", action="store_true")
    parser.add_argument(
        "--capacities",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024, 2048],
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = expand_files(args.files)
    if not files:
        raise ValueError("No input files found.")

    accesses, stats = load_accesses(
        files,
        ignore_layers=args.ignore_layers,
        include_prefill=args.include_prefill,
    )
    if not accesses:
        raise ValueError(
            "No decode accesses extracted; check --ignore-layers and that the "
            "traces carry decode records."
        )
    num_experts = stats["max_expert_id"] + 1

    rows = run_sweep(
        accesses, args.capacities, args.model, args.dataset, num_experts
    )

    commit = _git_commit()
    env = _env_string()
    command = (
        "python -m benchmarks.cache_simulator.research.run_policies_real "
        f"--model {args.model} --dataset {args.dataset} "
        f"--ignore-layers {args.ignore_layers}"
    )

    header = (
        f"# Real cache-policy benchmark — {args.model} / {args.dataset}\n\n"
        "All numbers are **REAL** — computed over an actual SGLang "
        "`per_token` expert-routing capture (not synthetic). Values are the "
        "demand **hit rate**; `belady` is the Belady-optimal ceiling "
        "('Ideal').\n\n"
        f"Trace stats: {stats['files']} file(s), {stats['num_layers']} "
        f"layers, max expert id {stats['max_expert_id']}, "
        f"{stats['distinct_keys']} distinct (layer, expert) keys, "
        f"{stats['total_accesses']} decode accesses "
        f"(ignore_layers={args.ignore_layers})."
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.model
    md_path = out_dir / f"results_real_{tag}.md"
    csv_path = out_dir / f"results_real_{tag}.csv"
    md_path.write_text(render_table(rows, commit, header))
    write_csv(rows, csv_path)
    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")

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
