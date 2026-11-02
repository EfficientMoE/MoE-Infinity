"""Analyse prefetch x eviction sweeps (``run_prefetch_real.py`` CSVs).

Answers the round-2 questions (``auto_research_prefetch_eviction.md``) from
committed sweep CSVs, as Markdown:

1. Per capacity: Belady, the stall floor, the best evictor alone, the best
   online (evictor, prefetcher, admission) and the oracle bound.
2. Separability of evictor x prefetcher: the regret of choosing the evictor
   alone (no prefetch) and the prefetcher with a reference evictor
   (independent choice) against the joint optimum, and a two-way
   decomposition of ``log(stall)`` into evictor, prefetcher and interaction
   shares - for each admission mode and for the best admission per pair.
3. Admission modes: mean stall relative to the best mode per pair, how often
   prefetching makes stall worse than the evictor alone, the demand hit-rate
   cost, and wasted transfers.
4. Whether prefetch changes which evictor is best.
5. Timing sensitivity: the best pair under each timing scenario.

Stall is the timing-model stall per decode token (``stall_ms_<scenario>``).

Usage::

    python -m benchmarks.cache_simulator.research.prefetch_analysis \
        --csv research/results_prefetch_qwen3_30b_held.csv --out report.md
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SCENARIOS = ("nominal", "fifo", "roofline")
ADMISSIONS = ("same_queue", "probation", "pin", "buffer", "do_no_harm")


def load(paths: Sequence[str]) -> List[Dict[str, object]]:
    rows = []
    for path in paths:
        with open(path) as handle:
            for row in csv.DictReader(handle):
                out: Dict[str, object] = {}
                for key, val in row.items():
                    try:
                        out[key] = float(val)
                    except (TypeError, ValueError):
                        out[key] = val
                out["capacity"] = int(out["capacity"])
                out["pf"] = family(str(out["prefetcher"]))
                out["adm"] = str(out["admission"]).split(":")[0]
                rows.append(out)
    return rows


def family(spec: str) -> str:
    """Prefetcher family: base name, oracles split by horizon."""
    name, _, params = spec.partition(":")
    if name == "oracle":
        horizon = dict(
            item.split("=") for item in params.split(",") if "=" in item
        ).get("horizon", "1")
        return f"oracle-h{horizon}"
    return name


def is_oracle(pf: str) -> bool:
    return pf.startswith("oracle")


def _fmt(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}"


def _key(scenario: str) -> str:
    return f"stall_ms_{scenario}"


# --------------------------------------------------------------------------- #
# 1. Per-capacity summary
# --------------------------------------------------------------------------- #
def capacity_summary(rows, scenario="nominal") -> str:
    k = _key(scenario)
    lines = [
        "| Capacity | Belady hit | Floor | Best evictor alone | Stall "
        "| Best online pair | Stall | vs alone | Oracle bound | Stall |",
        "| ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | --- | ---: |",
    ]
    for cap in sorted({r["capacity"] for r in rows}):
        sub = [r for r in rows if r["capacity"] == cap]
        alone = [r for r in sub if r["pf"] == "none"]
        online = [
            r for r in sub if r["pf"] != "none" and not is_oracle(r["pf"])
        ]
        oracle = [r for r in sub if is_oracle(r["pf"])]
        if not alone:
            continue
        a = min(alone, key=lambda r: r[k])
        o = min(online, key=lambda r: r[k]) if online else None
        b = min(oracle, key=lambda r: r[k]) if oracle else None
        floor = a.get(f"stall_floor_ms_{scenario}", float("nan"))
        cells = [
            str(cap),
            _fmt(a["belady_hit_rate"], 4),
            _fmt(floor),
            f"{a['evictor']} ({_fmt(a['hit_rate'], 4)})",
            _fmt(a[k]),
        ]
        if o is not None:
            gain = 100 * (o[k] / a[k] - 1) if a[k] > 0 else 0.0
            cells += [
                f"{o['evictor']} + {o['prefetcher']} / {o['adm']}",
                _fmt(o[k]),
                f"{gain:+.1f}%",
            ]
        else:
            cells += ["-", "-", "-"]
        if b is not None:
            cells += [f"{b['evictor']} + {b['pf']} / {b['adm']}", _fmt(b[k])]
        else:
            cells += ["-", "-"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pair_table(rows, cap: int, scenario="nominal", online_only=True) -> str:
    """Evictor x prefetcher stall (best admission) at one capacity."""
    k = _key(scenario)
    sub = [r for r in rows if r["capacity"] == cap]
    evs = sorted({r["evictor"] for r in sub})
    pfs = ["none"] + sorted(
        {r["pf"] for r in sub if r["pf"] != "none"}
        - (
            {p for p in {r["pf"] for r in sub} if is_oracle(p)}
            if online_only
            else set()
        )
    )
    best: Dict[Tuple[str, str], Dict] = {}
    for r in sub:
        slot = (r["evictor"], r["pf"])
        if slot not in best or r[k] < best[slot][k]:
            best[slot] = r
    lines = [
        "| Evictor | " + " | ".join(pfs) + " |",
        "| --- | " + " | ".join("---:" for _ in pfs) + " |",
    ]
    for ev in evs:
        cells = []
        for pf in pfs:
            r = best.get((ev, pf))
            if r is None:
                cells.append("-")
            elif pf == "none":
                cells.append(_fmt(r[k]))
            else:
                cells.append(f"{_fmt(r[k])} {r['adm'][:4]}")
        lines.append(f"| {ev} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 2. Separability
# --------------------------------------------------------------------------- #
def stall_matrix(rows, cap, admission: Optional[str], scenario="nominal"):
    """``S[evictor][prefetcher]`` (online prefetchers + none) at ``cap``.

    ``admission=None`` takes the best admission per pair.
    """
    k = _key(scenario)
    mat: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r["capacity"] != cap or is_oracle(r["pf"]):
            continue
        if (
            r["pf"] != "none"
            and admission is not None
            and r["adm"] != admission
        ):
            continue
        cur = mat[r["evictor"]].get(r["pf"])
        if cur is None or r[k] < cur:
            mat[r["evictor"]][r["pf"]] = r[k]
    return mat


def decompose(mat) -> Optional[Dict[str, float]]:
    """Two-way decomposition of log(stall): evictor / prefetcher / interaction."""
    evs = sorted(mat)
    pfs = sorted(set.intersection(*(set(mat[e]) for e in evs))) if evs else []
    if len(evs) < 2 or len(pfs) < 2:
        return None
    y = np.array([[math.log(max(mat[e][p], 1e-9)) for p in pfs] for e in evs])
    mu = y.mean()
    a = y.mean(axis=1) - mu
    b = y.mean(axis=0) - mu
    resid = y - mu - a[:, None] - b[None, :]
    ss_e = len(pfs) * float((a**2).sum())
    ss_p = len(evs) * float((b**2).sum())
    ss_i = float((resid**2).sum())
    total = ss_e + ss_p + ss_i
    if total <= 0:
        return {"evictor": 0.0, "prefetcher": 0.0, "interaction": 0.0}
    return {
        "evictor": ss_e / total,
        "prefetcher": ss_p / total,
        "interaction": ss_i / total,
    }


def regrets(mat, reference="lru") -> Optional[Dict[str, object]]:
    """Independent vs joint choice of (evictor, prefetcher)."""
    if not mat or any("none" not in mat[e] for e in mat):
        return None
    joint = min(
        ((s, e, p) for e in mat for p, s in mat[e].items()), key=lambda t: t[0]
    )
    e_alone = min(mat, key=lambda e: mat[e]["none"])
    ref = reference if reference in mat else e_alone
    p_ref = min(mat[ref], key=lambda p: mat[ref][p])
    indep = mat[e_alone].get(p_ref, float("nan"))
    p_seq = min(mat[e_alone], key=lambda p: mat[e_alone][p])
    seq = mat[e_alone][p_seq]
    return {
        "joint": joint,
        "e_alone": e_alone,
        "p_ref": p_ref,
        "indep": indep,
        "seq": (seq, e_alone, p_seq),
    }


def separability(rows, scenario="nominal", reference="lru") -> str:
    lines = [
        "| Capacity | Admission | Joint optimum | Stall | Independent "
        f"(evictor alone + prefetcher on {reference}) | Regret | Sequential "
        "(best prefetcher for that evictor) | Regret | Evictor / prefetcher "
        "/ interaction share of log-stall variance |",
        "| ---: | --- | --- | ---: | --- | ---: | --- | ---: | --- |",
    ]
    for cap in sorted({r["capacity"] for r in rows}):
        for adm in (None,) + ADMISSIONS:
            mat = stall_matrix(rows, cap, adm, scenario)
            if not mat or all(len(v) < 2 for v in mat.values()):
                continue
            reg = regrets(mat, reference)
            dec = decompose(mat)
            if reg is None:
                continue
            j_s, j_e, j_p = reg["joint"]
            seq_s, seq_e, seq_p = reg["seq"]
            ind = reg["indep"]
            cells = [
                str(cap),
                adm or "best per pair",
                f"{j_e} + {j_p}",
                _fmt(j_s),
                f"{reg['e_alone']} + {reg['p_ref']} = {_fmt(ind)}",
                f"{100 * (ind / j_s - 1):+.1f}%" if j_s > 0 else "-",
                f"{seq_e} + {seq_p} = {_fmt(seq_s)}",
                f"{100 * (seq_s / j_s - 1):+.1f}%" if j_s > 0 else "-",
                (
                    " / ".join(
                        f"{100 * dec[x]:.0f}%"
                        for x in ("evictor", "prefetcher", "interaction")
                    )
                    if dec
                    else "-"
                ),
            ]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 3. Admission modes and harm
# --------------------------------------------------------------------------- #
def admission_summary(rows, scenario="nominal") -> str:
    k = _key(scenario)
    alone = {
        (r["evictor"], r["capacity"]): r for r in rows if r["pf"] == "none"
    }
    by_pair: Dict[Tuple, Dict[str, Dict]] = defaultdict(dict)
    for r in rows:
        if r["pf"] == "none" or is_oracle(r["pf"]):
            continue
        by_pair[(r["evictor"], r["pf"], r["capacity"])][r["adm"]] = r
    stats = {
        a: {"rel": [], "wins": 0, "worse": 0, "n": 0, "hit": [], "waste": []}
        for a in ADMISSIONS
    }
    for (ev, pf, cap), modes in by_pair.items():
        best = min(m[k] for m in modes.values())
        base = alone.get((ev, cap))
        for adm, r in modes.items():
            s = stats[adm]
            s["n"] += 1
            s["rel"].append(r[k] / best if best > 0 else 1.0)
            s["wins"] += int(r[k] <= best + 1e-12)
            if base is not None:
                s["worse"] += int(r[k] > base[k] * (1 + 1e-6))
                s["hit"].append(100 * (r["hit_rate"] - base["hit_rate"]))
            s["waste"].append(r.get(f"wasted_xfer_per_tok_{scenario}", 0.0))
    lines = [
        "| Admission | Pairs | Best mode for | Mean stall vs best mode "
        "| Worse than no prefetch | Mean demand hit-rate change (points) "
        "| Mean wasted transfers / token |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for adm in ADMISSIONS:
        s = stats[adm]
        if not s["n"]:
            continue
        lines.append(
            f"| {adm} | {s['n']} | {s['wins']} | "
            f"{100 * (np.mean(s['rel']) - 1):+.1f}% | "
            f"{s['worse']} ({100 * s['worse'] / s['n']:.0f}%) | "
            f"{np.mean(s['hit']):+.2f} | {np.mean(s['waste']):.1f} |"
        )
    return "\n".join(lines)


def harm_by_capacity(rows, scenario="nominal") -> str:
    """Share of online (evictor, prefetcher) combos worse than no prefetch."""
    k = _key(scenario)
    alone = {
        (r["evictor"], r["capacity"]): r[k] for r in rows if r["pf"] == "none"
    }
    caps = sorted({r["capacity"] for r in rows})
    lines = [
        "| Admission | " + " | ".join(f"cap={c}" for c in caps) + " |",
        "| --- | " + " | ".join("---:" for _ in caps) + " |",
    ]
    for adm in ADMISSIONS:
        cells = []
        for cap in caps:
            sub = [
                r
                for r in rows
                if r["capacity"] == cap
                and r["adm"] == adm
                and r["pf"] != "none"
                and not is_oracle(r["pf"])
            ]
            if not sub:
                cells.append("-")
                continue
            worse = sum(
                1 for r in sub if r[k] > alone[(r["evictor"], cap)] * (1 + 1e-6)
            )
            cells.append(f"{worse}/{len(sub)}")
        lines.append(f"| {adm} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 4. Does prefetch change the best evictor?
# --------------------------------------------------------------------------- #
def _kendall(a: Sequence[str], b: Sequence[str]) -> float:
    pos = {x: i for i, x in enumerate(b)}
    items = [x for x in a if x in pos]
    conc = disc = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            sign = pos[items[i]] - pos[items[j]]
            conc += sign < 0
            disc += sign > 0
    total = conc + disc
    return (conc - disc) / total if total else 1.0


def evictor_ranking(rows, scenario="nominal") -> str:
    lines = [
        "| Capacity | Best evictor, no prefetch | Best evictor per prefetcher "
        "(best admission) | Kendall tau of evictor ranking vs no prefetch |",
        "| ---: | --- | --- | --- |",
    ]
    for cap in sorted({r["capacity"] for r in rows}):
        mat = stall_matrix(rows, cap, None, scenario)
        if not mat:
            continue
        evs = sorted(mat)
        if any("none" not in mat[e] for e in evs):
            continue
        base = sorted(evs, key=lambda e: mat[e]["none"])
        pfs = sorted({p for e in evs for p in mat[e]} - {"none"})
        best, taus = [], []
        for pf in pfs:
            have = [e for e in evs if pf in mat[e]]
            if not have:
                continue
            rank = sorted(have, key=lambda e: mat[e][pf])
            best.append(f"{pf}: {rank[0]}")
            taus.append(f"{pf} {_kendall(base, rank):+.2f}")
        lines.append(
            f"| {cap} | {base[0]} | {'; '.join(best)} | {'; '.join(taus)} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 5. Timing sensitivity
# --------------------------------------------------------------------------- #
def sensitivity(rows) -> str:
    lines = [
        "| Capacity | "
        + " | ".join(f"best online pair ({s})" for s in SCENARIOS)
        + " | "
        + " | ".join(f"gain vs best alone ({s})" for s in SCENARIOS)
        + " |",
        "| ---: | "
        + " | ".join("---" for _ in SCENARIOS)
        + " | "
        + " | ".join("---:" for _ in SCENARIOS)
        + " |",
    ]
    for cap in sorted({r["capacity"] for r in rows}):
        sub = [r for r in rows if r["capacity"] == cap]
        names, gains = [], []
        for s in SCENARIOS:
            k = _key(s)
            alone = [r for r in sub if r["pf"] == "none"]
            online = [
                r for r in sub if r["pf"] != "none" and not is_oracle(r["pf"])
            ]
            if not alone or not online:
                names.append("-")
                gains.append("-")
                continue
            a = min(alone, key=lambda r: r[k])
            o = min(online, key=lambda r: r[k])
            names.append(f"{o['evictor']} + {o['pf']} / {o['adm']}")
            gains.append(
                f"{100 * (o[k] / a[k] - 1):+.1f}%" if a[k] > 0 else "-"
            )
        lines.append(f"| {cap} | " + " | ".join(names + gains) + " |")
    return "\n".join(lines)


def best_per_admission(rows, scenario="nominal") -> str:
    """Per capacity, the best online pair for each admission mode."""
    k = _key(scenario)
    caps = sorted({r["capacity"] for r in rows})
    lines = [
        "| Capacity | " + " | ".join(ADMISSIONS) + " |",
        "| ---: | " + " | ".join("---" for _ in ADMISSIONS) + " |",
    ]
    for cap in caps:
        cells = []
        for adm in ADMISSIONS:
            sub = [
                r
                for r in rows
                if r["capacity"] == cap
                and r["adm"] == adm
                and r["pf"] != "none"
                and not is_oracle(r["pf"])
            ]
            if not sub:
                cells.append("-")
                continue
            o = min(sub, key=lambda r: r[k])
            cells.append(f"{_fmt(o[k])} {o['evictor']}+{o['pf']}")
        lines.append(f"| {cap} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def report(rows: List[Dict[str, object]], title: str) -> str:
    caps = sorted({r["capacity"] for r in rows})
    parts = [
        f"## {title}",
        "",
        "### Per capacity (nominal timing, ms of stall per decode token)",
        "",
        capacity_summary(rows),
        "",
        "### Best online pair per admission mode (nominal)",
        "",
        best_per_admission(rows),
        "",
        "### Separability (nominal)",
        "",
        separability(rows),
        "",
        "### Admission modes over all online pairs (nominal)",
        "",
        admission_summary(rows),
        "",
        "### Online combinations worse than the evictor alone (nominal)",
        "",
        harm_by_capacity(rows),
        "",
        "### Online combinations worse than the evictor alone (FIFO link)",
        "",
        harm_by_capacity(rows, "fifo"),
        "",
        "### Does prefetch change the best evictor? (nominal)",
        "",
        evictor_ranking(rows),
        "",
        "### Timing sensitivity",
        "",
        sensitivity(rows),
        "",
    ]
    for cap in caps:
        parts += [
            f"### Evictor x prefetcher at capacity {cap} (nominal stall, "
            "best admission)",
            "",
            pair_table(rows, cap),
            "",
        ]
    return "\n".join(parts)


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", nargs="+", required=True)
    parser.add_argument("--title", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    rows = load(args.csv)
    groups: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for r in rows:
        groups[(str(r["model"]), str(r["dataset"]))].append(r)
    text = "\n".join(
        report(sub, args.title or f"{model} / {dataset}")
        for (model, dataset), sub in sorted(groups.items())
    )
    print(text)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
