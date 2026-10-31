"""Benchmark harness: every cache policy vs Belady over synthetic MoE traces.

Reproduces the MoE-Infinity-style comparison (policy x cache capacity -> hit
rate, with Belady optimal as the "Ideal" upper bound), but on **synthetic**
expert-routing traces rather than real captures. There are no real Qwen3-235B
traces available on this machine, so every number produced here is labelled
SYNTHETIC. The synthetic models mimic MoE routing structure (many layers,
top-k routing per layer, skewed expert popularity, temporal locality) so the
relative ordering of policies is meaningful even though absolute hit rates are
not calibrated to any real workload.

Usage::

    python -m benchmarks.cache_simulator.research.run_policies \
        --out benchmarks/cache_simulator/research

Outputs (under ``--out``):

* ``results_synthetic.md``   - policy x capacity hit-rate table per model.
* ``results_synthetic.csv``  - the same numbers, machine-readable.
* ``figures/mrc_<model>.{pdf,png}`` - hit-rate-vs-capacity (MRC-style) plot per
  model, Belady drawn as "Ideal" (figures are git-ignored; the CSV/table are
  the committed artifacts).

The pure-Python simulator (``belady_simulate`` + ``CachePolicy``) is the
correctness oracle. Trace generation is seeded for reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from benchmarks.cache_simulator.belady_cache_simulator import belady_simulate
from benchmarks.cache_simulator.policies import (
    ORACLE_POLICIES,
    POLICY_FACTORIES,
    build_policy,
    simulate,
)

Key = Tuple[int, int]

# Order the policies for stable reporting; Belady ("Ideal") is handled apart.
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


# --------------------------------------------------------------------------- #
# Synthetic trace generators
# --------------------------------------------------------------------------- #
@dataclass
class SyntheticModel:
    """A synthetic MoE routing workload."""

    name: str
    n_layer: int
    n_expert: int
    top_k: int
    steps: int
    #: Zipf-like skew exponent for expert popularity (0 = uniform).
    skew: float = 1.0
    #: Probability a step reuses the previous step's picks (temporal locality).
    reuse: float = 0.3
    seed: int = 0
    capacities: Sequence[int] = field(default_factory=tuple)

    def generate(self) -> List[Key]:
        rng = random.Random(self.seed)
        # Per-layer expert popularity weights (Zipf-like, shuffled per layer).
        layer_weights: List[List[float]] = []
        for _ in range(self.n_layer):
            ranks = list(range(1, self.n_expert + 1))
            weights = [1.0 / (r**self.skew) for r in ranks]
            rng.shuffle(weights)
            layer_weights.append(weights)

        accesses: List[Key] = []
        prev_picks: Dict[int, List[int]] = {}
        experts = list(range(self.n_expert))
        for _ in range(self.steps):
            for layer in range(self.n_layer):
                if layer in prev_picks and rng.random() < self.reuse:
                    picks = prev_picks[layer]
                else:
                    picks = _weighted_sample(
                        rng, experts, layer_weights[layer], self.top_k
                    )
                    prev_picks[layer] = picks
                for expert in picks:
                    accesses.append((layer, expert))
        return accesses


def _weighted_sample(
    rng: random.Random,
    population: Sequence[int],
    weights: Sequence[float],
    k: int,
) -> List[int]:
    """Sample ``k`` distinct items with the given weights (no replacement)."""
    chosen: List[int] = []
    pool = list(population)
    w = list(weights)
    for _ in range(min(k, len(pool))):
        total = sum(w)
        r = rng.random() * total
        upto = 0.0
        for i, weight in enumerate(w):
            upto += weight
            if upto >= r:
                chosen.append(pool.pop(i))
                w.pop(i)
                break
    return sorted(chosen)


def default_models() -> List[SyntheticModel]:
    """Three synthetic models spanning scale and skew.

    ``qwen3-235b-synthetic`` mirrors the geometry of
    ``Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`` (~94 MoE layers, 128 experts,
    top-8) but with synthetic routing; the other two are smaller stress cases.
    """
    return [
        SyntheticModel(
            name="qwen3-235b-synthetic",
            n_layer=94,
            n_expert=128,
            top_k=8,
            steps=120,
            skew=1.1,
            reuse=0.35,
            seed=2026,
            capacities=(256, 512, 1024, 2048, 4096, 6016),
        ),
        SyntheticModel(
            name="mixtral-8x7b-synthetic",
            n_layer=32,
            n_expert=8,
            top_k=2,
            steps=400,
            skew=0.8,
            reuse=0.4,
            seed=11,
            capacities=(16, 32, 64, 96, 128, 192, 256),
        ),
        SyntheticModel(
            name="uniform-stress-synthetic",
            n_layer=16,
            n_expert=32,
            top_k=4,
            steps=300,
            skew=0.0,
            reuse=0.1,
            seed=7,
            capacities=(16, 32, 64, 128, 256, 384),
        ),
    ]


# --------------------------------------------------------------------------- #
# Running the sweep
# --------------------------------------------------------------------------- #
@dataclass
class Row:
    model: str
    policy: str
    capacity: int
    hit_rate: float
    effective_hit_rate: float
    is_oracle: bool


def run_model(model: SyntheticModel) -> Tuple[List[Key], List[Row]]:
    accesses = model.generate()
    rows: List[Row] = []
    for cap in model.capacities:
        _, _, belady_rate = belady_simulate(accesses, cap)
        rows.append(
            Row(model.name, "belady", cap, belady_rate, belady_rate, False)
        )
        for name in POLICY_ORDER:
            policy = build_policy(name, cap)
            _, _, rate = simulate(policy, accesses)
            # Guard the headline invariant at runtime, too.
            assert (
                rate <= belady_rate + 1e-9
            ), f"{name} at cap {cap} beat Belady: {rate} > {belady_rate}"
            rows.append(
                Row(
                    model.name,
                    name,
                    cap,
                    rate,
                    policy.effective_hit_rate,
                    name in ORACLE_POLICIES,
                )
            )
    return accesses, rows


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def write_csv(rows: Sequence[Row], path: Path) -> None:
    # newline="" + lineterminator="\n" keeps the file LF-only so it survives
    # the repo's mixed-line-ending pre-commit hook on re-generation.
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "model",
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
                    row.policy,
                    row.capacity,
                    f"{row.hit_rate:.6f}",
                    f"{row.effective_hit_rate:.6f}",
                    int(row.is_oracle),
                    "synthetic",
                ]
            )


def render_table(
    models: Sequence[SyntheticModel], rows: Sequence[Row], commit: str
) -> str:
    by_model: Dict[str, List[Row]] = {}
    for row in rows:
        by_model.setdefault(row.model, []).append(row)

    lines: List[str] = []
    lines.append("# Synthetic cache-policy benchmark results\n")
    lines.append(
        "All numbers are **SYNTHETIC** -- generated from seeded synthetic MoE "
        "routing traces (no real Qwen3-235B captures are available on this "
        "machine). Values are the demand **hit rate** (fraction of accesses "
        "served from the resident cache). `belady` is the Belady-optimal "
        "demand-paging ceiling ('Ideal'). Policies marked `*` are "
        "trace-oracle prefetch variants: their demand hit rate stays bounded "
        "by Belady, and their prefetch value shows up in `effective_hit_rate` "
        "(see the CSV), which is an **upper bound** on the underlying "
        "predictor.\n"
    )
    lines.append(f"Commit: `{commit}`\n")

    for model in models:
        model_rows = by_model.get(model.name, [])
        caps = sorted({r.capacity for r in model_rows})
        lookup = {(r.policy, r.capacity): r.hit_rate for r in model_rows}

        lines.append(f"\n## {model.name}\n")
        lines.append(
            f"Geometry: {model.n_layer} layers x {model.n_expert} experts, "
            f"top-{model.top_k}, {model.steps} decode steps, "
            f"skew={model.skew}, reuse={model.reuse}, seed={model.seed}. "
            f"Trace length = {model.n_layer * model.top_k * model.steps} "
            "accesses.\n"
        )

        header = "| Policy | " + " | ".join(f"cap={c}" for c in caps) + " |"
        sep = "| --- | " + " | ".join("---" for _ in caps) + " |"
        lines.append(header)
        lines.append(sep)

        def fmt_row(label: str, policy: str) -> str:
            cells = []
            for cap in caps:
                val = lookup.get((policy, cap))
                cells.append(f"{val:.4f}" if val is not None else "-")
            return f"| {label} | " + " | ".join(cells) + " |"

        lines.append(fmt_row("Belady (Ideal)", "belady"))
        for policy in POLICY_ORDER:
            lines.append(fmt_row(PRETTY[policy], policy))

    lines.append(
        "\n`*` = trace-oracle prefetch policy (perfect next-layer routing "
        "lookahead); reported demand hit rate is an upper bound on the "
        "achievable predictor. See `results_synthetic.csv` for "
        "`effective_hit_rate` (prefetch-coverage upper bound).\n"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Plotting (optional; figures are git-ignored)
# --------------------------------------------------------------------------- #
# Wong colorblind-safe palette (Nature Methods, 2011), extended. Index 0 is
# black -- reserved for the Belady "Ideal" reference line.
WONG_PALETTE = [
    "#000000",
    "#EEBA0C",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#0000FF",
    "#FF0000",
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<"]


def plot_mrc(
    model: SyntheticModel, rows: Sequence[Row], out_dir: Path
) -> List[Path]:
    """Render a hit-rate-vs-capacity (MRC-style) plot, Belady as 'Ideal'."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"  [skip plot for {model.name}: matplotlib unavailable: {exc}]")
        return []

    caps = sorted({r.capacity for r in rows})
    lookup = {(r.policy, r.capacity): r.hit_rate for r in rows}

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.4,
        }
    )
    fig, ax = plt.subplots(figsize=(3.6, 2.7))

    belady_y = [lookup.get(("belady", c)) for c in caps]
    ax.plot(
        caps,
        belady_y,
        color=WONG_PALETTE[0],
        marker="*",
        markersize=5,
        linewidth=1.2,
        linestyle="--",
        label="Belady (Ideal)",
    )
    for i, policy in enumerate(POLICY_ORDER):
        ys = [lookup.get((policy, c)) for c in caps]
        ax.plot(
            caps,
            ys,
            color=WONG_PALETTE[(i % (len(WONG_PALETTE) - 1)) + 1],
            marker=MARKERS[i % len(MARKERS)],
            markersize=3.5,
            linewidth=1.0,
            label=PRETTY[policy],
        )

    ax.set_xlabel("Cache capacity (experts)")
    ax.set_ylabel("Demand hit rate")
    ax.set_title(f"{model.name} (synthetic)", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=5.5, ncol=2, loc="lower right", framealpha=0.9)
    fig.tight_layout()

    pdf_path = out_dir / f"mrc_{model.name}.pdf"
    png_path = out_dir / f"mrc_{model.name}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return [pdf_path, png_path]


# --------------------------------------------------------------------------- #
# EXPERIMENTS.md appender
# --------------------------------------------------------------------------- #
def append_experiments(
    rows: Sequence[Row], path: Path, commit: str, command: str, env: str
) -> None:
    header_needed = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        if header_needed:
            handle.write("# Cache-policy experiments log\n\n")
            handle.write(
                "Rows appended by "
                "`benchmarks/cache_simulator/research/run_policies.py`. "
                "All traces are synthetic (seeded); no real Qwen3-235B "
                "captures are used.\n\n"
            )
            handle.write(
                "| policy | model | dataset | capacity | hit_rate | "
                "eff_hit_rate | provenance | commit |\n"
            )
            handle.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.policy} | {row.model} | synthetic | {row.capacity} "
                f"| {row.hit_rate:.4f} | {row.effective_hit_rate:.4f} "
                f"| synthetic | {commit} |\n"
            )
        handle.write(
            f"\n<!-- run: {command} | env: {env} | commit: {commit} -->\n"
        )


def _git_commit() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # pragma: no cover
        return "unknown"


def _env_string() -> str:
    import platform

    import numpy as np

    return (
        f"python {platform.python_version()}, numpy {np.__version__}, "
        f"{platform.system()} {platform.machine()}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark cache policies vs Belady on synthetic traces.",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent),
        help="Output directory for tables/figures (default: this dir).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip rendering MRC figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = default_models()
    commit = _git_commit()
    env = _env_string()
    command = "python -m benchmarks.cache_simulator.research.run_policies"

    all_rows: List[Row] = []
    for model in models:
        print(f"Running {model.name} ...")
        _, rows = run_model(model)
        all_rows.extend(rows)
        if not args.no_plots:
            figs = plot_mrc(model, rows, out_dir / "figures")
            for fig in figs:
                print(f"  wrote {fig}")

    csv_path = out_dir / "results_synthetic.csv"
    md_path = out_dir / "results_synthetic.md"
    write_csv(all_rows, csv_path)
    md_path.write_text(render_table(models, all_rows, commit))
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")

    append_experiments(
        all_rows,
        out_dir / "EXPERIMENTS.md",
        commit=commit,
        command=command,
        env=env,
    )
    print(f"Appended to {out_dir / 'EXPERIMENTS.md'}")


if __name__ == "__main__":
    main()
