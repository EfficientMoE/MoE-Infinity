"""Regret of ONE default (evictor, prefetcher, admission) on held-out sweeps.

Reads ``run_prefetch_real.py`` CSVs (rounds 2 and 3) and, per model, stream
and capacity, compares the default configuration with:

* every evictor alone (LRU, LFU ~ today's engine demand path, LCNU, ...),
* the best online configuration of every sweep for that cell ("per-cell
  best": what per-regime tuning could at most buy),
* the best oracle-prefetch configuration (an upper bound, not a policy).

The default's evictor is given as a preference list: on decode-only streams
``lcnu_p`` (decode-only routing statistics) is identical to ``lcnu``, so the
first evictor present in a CSV is used.

Usage::

    python -m benchmarks.cache_simulator.research.default_policy_analysis \
        --csv research/results_prefetch_qwen3_30b_held.csv \
              research/results_prefetch_qwen3_30b_held_r3.csv \
        --evictor lcnu_p lcnu --prefetcher "knn:..." \
        --admission "probation:issue=2"
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

KEY = "stall_ms_nominal"


def load(paths: Sequence[str]) -> List[Dict[str, object]]:
    rows = []
    for path in paths:
        with open(path) as handle:
            for row in csv.DictReader(handle):
                out: Dict[str, object] = dict(row)
                for key, val in row.items():
                    try:
                        out[key] = float(val)
                    except (TypeError, ValueError):
                        pass
                out["capacity"] = int(float(row["capacity"]))
                out["stream"] = str(row["dataset"]).split("/")[1]
                out["is_oracle"] = int(float(row.get("is_oracle", 0) or 0))
                rows.append(out)
    return rows


def _pct(new: float, old: float) -> str:
    return f"{100 * (new / old - 1):+.1f}%" if old > 0 else "-"


def _label(r) -> str:
    adm = str(r["admission"])
    pf = str(r["prefetcher"])
    return f"{r['evictor']} + {pf.split(':')[0]} / {adm}"


def cell_table(
    rows, evictors, prefetcher, admission
) -> Tuple[str, List[float]]:
    """Markdown table for one (model, stream) and the default's regrets."""
    lines = [
        "| Slots | LRU | LFU | LCNU | Default | vs LCNU | vs LFU | Per-cell "
        "best online | Stall | Default regret | Oracle bound |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: "
        "| ---: |",
    ]
    regrets: List[float] = []
    for cap in sorted({r["capacity"] for r in rows}):
        sub = [r for r in rows if r["capacity"] == cap]
        alone = {r["evictor"]: r[KEY] for r in sub if r["prefetcher"] == "none"}
        default = None
        for ev in evictors:
            pick = [
                r
                for r in sub
                if r["evictor"] == ev
                and r["prefetcher"] in prefetcher
                and r["admission"] == admission
            ]
            if pick:
                default = pick[0]
                break
        online = [
            r for r in sub if r["prefetcher"] != "none" and not r["is_oracle"]
        ]
        oracle = [r for r in sub if r["is_oracle"]]
        best = min(online, key=lambda r: r[KEY]) if online else None
        bound = min(oracle, key=lambda r: r[KEY]) if oracle else None
        lcnu = next((alone[e] for e in evictors if e in alone), float("nan"))
        cells = [
            str(cap),
            f"{alone.get('lru', float('nan')):.2f}",
            f"{alone.get('lfu', float('nan')):.2f}",
            f"{lcnu:.2f}",
        ]
        if default is not None:
            d = default[KEY]
            cells += [f"{d:.2f}", _pct(d, lcnu), _pct(d, alone.get("lfu", 0))]
        else:
            cells += ["-", "-", "-"]
        if best is not None:
            cells += [_label(best), f"{best[KEY]:.2f}"]
            if default is not None and best[KEY] > 0:
                regret = default[KEY] / best[KEY] - 1
                regrets.append(regret)
                cells.append(f"{100 * regret:+.1f}%")
            else:
                cells.append("-")
        else:
            cells += ["-", "-", "-"]
        cells.append(f"{bound[KEY]:.2f}" if bound is not None else "-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines), regrets


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", nargs="+", required=True)
    parser.add_argument("--evictor", nargs="+", default=["lcnu_p", "lcnu"])
    parser.add_argument(
        "--prefetcher",
        nargs="+",
        required=True,
        help="Default prefetcher spec(s); equivalent spellings may be listed.",
    )
    parser.add_argument("--admission", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    rows = load(args.csv)
    groups: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for r in rows:
        groups[(str(r["model"]), str(r["stream"]))].append(r)
    parts = [
        f"Default: `{args.evictor[0]}` + `{args.prefetcher[0]}` / "
        f"`{args.admission}`. Stall = ms per decode token (nominal timing).",
        "",
    ]
    all_regrets: List[float] = []
    for (model, stream), sub in sorted(groups.items()):
        table, regrets = cell_table(
            sub, args.evictor, args.prefetcher, args.admission
        )
        all_regrets += regrets
        worst = max(regrets) if regrets else float("nan")
        mean = sum(regrets) / len(regrets) if regrets else float("nan")
        parts += [
            f"### {model} / {stream}",
            "",
            table,
            "",
            f"Default regret vs the per-cell best online configuration: "
            f"mean {100 * mean:+.1f}%, worst {100 * worst:+.1f}%.",
            "",
        ]
    if all_regrets:
        parts.append(
            f"Over all {len(all_regrets)} cells: mean regret "
            f"{100 * sum(all_regrets) / len(all_regrets):+.1f}%, worst "
            f"{100 * max(all_regrets):+.1f}%."
        )
    text = "\n".join(parts).rstrip("\n")
    print(text)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
