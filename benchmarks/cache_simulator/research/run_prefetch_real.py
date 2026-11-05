"""Sweep evictor x prefetcher x admission x capacity on real decode traces.

Round-2 driver (``auto_research_prefetch_eviction.md``). Each job composes
one evictor from ``policies.py`` with one prefetcher from ``prefetchers.py``
under one admission mode, replays the trace, and reports cache metrics plus
the timing model (``timing_model.py``) under several link scenarios:

* demand hit rate (bounded by Belady, asserted) and Belady at that capacity,
* prefetch coverage (misses whose transfer a prefetch issued), issued and
  wasted prefetches, predictor precision / recall at distance 1,
* per timing scenario: stall per token, transfers per token, prefetch
  transfers / wasted / cancelled / late / promoted per token, and the stall
  floor no policy can beat (Belady's transfers minus all compute).

Prefetchers are given as specs ``name[:key=value,...]``, e.g.
``coactivation:horizon=2,budget=8`` or ``hybrid:weights=1/0.5/2``; ``none``
is the evictor alone. Admission modes accept ``buffer:frac=0.125`` (prefetch
buffer as a fraction of the capacity; default ``horizon * budget`` slots,
at most half the capacity) and ``pin:frac=0.5``.

Splits follow ``lcnu_ablation.py``: the request files, in capture order, are
halved; ``--split tune`` replays the first half, ``held`` the second, ``all``
both. ``--mode predict`` scores predictors alone (precision / recall per
distance), with no cache. ``--mode tune`` runs a horizon x budget grid and
writes the per-(prefetcher, admission, evictor, capacity) best spec by
nominal stall to ``--tuned-out``; ``--tuned`` feeds such a file to a sweep.

Usage::

    python -m benchmarks.cache_simulator.research.run_prefetch_real \
        --files "<data>/belady_qwen3_30b/*.pt" --model qwen3-30b-a3b \
        --split held --evictors lru lcnu --prefetchers none prev_token \
        --admissions same_queue do_no_harm --capacities 256 1024 --workers 32
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from benchmarks.cache_simulator.belady_cache_simulator import belady_simulate
from benchmarks.cache_simulator.policies import (
    BYPASS_POLICIES,
    GEOMETRY_POLICIES,
    LCNU_POLICIES,
    build_policy,
    simulate,
)
from benchmarks.cache_simulator.prefetchers import (
    ADMISSION_MODES,
    ORACLE_PREFETCHERS,
    PREFETCHER_FACTORIES,
    PrefetchCachePolicy,
    PrefetchPolicy,
    build_composed,
    build_prefetcher,
)
from benchmarks.cache_simulator.research.run_policies_real import (
    _env_string,
    _git_commit,
    expand_files,
    load_accesses,
)
from benchmarks.cache_simulator.research.timing_model import (
    LINK_PRESETS,
    LinkModel,
    replay_policy,
    stall_floor_s,
)

Key = Tuple[int, int]

EVICTORS = ("lru", "lfu", "lirs", "s3fifo", "w_tinylfu", "layer_lru", "lcnu")
PREFETCHERS = ("prev_token", "topk_ema", "coactivation", "hybrid", "same_id")
#: Round-3 predictors (``--prefetchers`` accepts them too).
MORE_PREFETCHERS = ("logistic", "knn")

#: Per-layer compute measured by ``research/measure_layer.py`` on the capture
#: host (RTX PRO 6000 Blackwell, CUDA graph, batch-1 GEMVs of one layer's
#: attention projections + router + top-k experts; FP8 scaled by bytes).
MEASURED_LAYER_S = {"qwen3-30b-a3b": 189.1e-6, "qwen3-235b-a22b-fp8": 221.4e-6}

#: Timing scenarios: name -> (compute source, demand priority).
SCENARIOS = {
    "nominal": ("measured", True),
    "fifo": ("measured", False),
    "roofline": ("roofline", True),
}
NOMINAL = "nominal"


def links_for(model: str) -> Dict[str, Tuple[LinkModel, bool]]:
    preset = LINK_PRESETS[model]
    out = {}
    for name, (compute, priority) in SCENARIOS.items():
        if compute == "measured":
            link = preset(layer_compute_s=MEASURED_LAYER_S[model])
        else:
            link = preset()
        out[name] = (link, priority)
    return out


# --------------------------------------------------------------------------- #
# Specs
# --------------------------------------------------------------------------- #
def _value(text: str):
    if "/" in text:
        return tuple(float(x) for x in text.split("/"))
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def parse_spec(spec: str) -> Tuple[str, Dict[str, object]]:
    """``name[:k=v,...]`` -> ``(name, params)``."""
    name, _, rest = spec.partition(":")
    params: Dict[str, object] = {}
    for item in filter(None, rest.split(",")):
        key, _, val = item.partition("=")
        params[key] = _value(val)
    return name, params


def spec_label(name: str, params: Dict[str, object]) -> str:
    if not params:
        return name
    parts = []
    for key in sorted(params):
        val = params[key]
        if isinstance(val, tuple):
            val = "/".join(f"{x:g}" for x in val)
        parts.append(f"{key}={val}")
    return f"{name}:" + ",".join(parts)


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
_TRACE: List[Key] = []
_GEOM: Dict[str, float] = {}
_LINKS: Dict[str, Tuple[LinkModel, bool]] = {}
_BELADY: Dict[int, int] = {}


def _make_prefetcher(spec: str, budget_default: int):
    name, params = parse_spec(spec)
    if name == "none":
        return None, {}
    params = dict(params)
    if name in ORACLE_PREFETCHERS:
        params.setdefault("budget", 10**6)
    params.setdefault("budget", budget_default)
    kwargs = dict(params)
    kwargs.update(
        num_layers=int(_GEOM["num_layers"]),
        num_experts=int(_GEOM["num_experts"]),
    )
    return build_prefetcher(name, **kwargs), params


def _buffer_slots(adm_params, pf, cap: int) -> int:
    if "slots" in adm_params:
        slots = int(adm_params["slots"])
    elif "frac" in adm_params:
        slots = int(round(float(adm_params["frac"]) * cap))
    else:
        slots = pf.horizon * min(pf.budget, int(_GEOM["top_k"]) * 2)
    return max(1, min(slots, cap // 2))


def run_job(job: Tuple[str, str, str, int]) -> Dict[str, object]:
    ev_spec, spec, adm_spec, cap = job
    t0 = time.time()
    if ev_spec in ("belady", "belady_bypass"):
        hits = belady_simulate(_TRACE, cap, bypass=ev_spec == "belady_bypass")[
            0
        ]
        return {"evictor": ev_spec, "capacity": cap, "hits": hits}
    pf, _ = _make_prefetcher(spec, int(_GEOM["top_k"]))
    adm, adm_params = parse_spec(adm_spec)
    ev_name, ev_params = parse_spec(ev_spec)
    slots = _buffer_slots(adm_params, pf, cap) if adm == "buffer" and pf else 0
    pol = build_composed(
        ev_name,
        cap,
        pf,
        admission=adm,
        buffer_slots=slots,
        num_layers=int(_GEOM["num_layers"]),
        prior=_GEOM["top_k"] / _GEOM["num_experts"],
        pin_fraction=float(adm_params.get("frac", 0.5))
        if adm == "pin"
        else 0.5,
        top_k=int(_GEOM["top_k"]),
        evictor_kwargs=ev_params,
        issue_budget=int(adm_params["issue"])
        if "issue" in adm_params
        else None,
    )
    hits, misses, rate = simulate(pol, _TRACE)
    return _row(job, pol, hits, misses, slots, time.time() - t0)


def _bypasses(ev_spec: str) -> bool:
    """Whether the evictor may serve a miss without caching it."""
    name, params = parse_spec(ev_spec)
    if "bypass" in params:
        return params["bypass"] != "never"
    return name in BYPASS_POLICIES


def _row(job, pol: PrefetchCachePolicy, hits, misses, slots, secs):
    ev_name, spec, adm_spec, cap = job
    n = len(_TRACE)
    tokens = _GEOM["tokens"]
    belady_hits = _BELADY.get(("belady", cap))
    bypass_hits = _BELADY.get(("belady_bypass", cap))
    bound = bypass_hits if _bypasses(ev_name) else belady_hits
    if bound is not None:
        assert hits <= bound, f"{job} beat its Belady bound: {hits} > {bound}"
    row: Dict[str, object] = {
        "evictor": ev_name,
        "prefetcher": spec,
        "admission": adm_spec,
        "capacity": cap,
        "buffer_slots": slots,
        "hit_rate": hits / n,
        "belady_hit_rate": (belady_hits / n) if belady_hits is not None else "",
        "belady_bypass_hit_rate": (bypass_hits / n)
        if bypass_hits is not None
        else "",
        "covered_rate": pol.prefetch_covered / n,
        "effective_hit_rate": (hits + pol.prefetch_covered) / n,
        "issued_per_token": pol.prefetch_issued / tokens,
        "wasted_per_token": pol.prefetch_wasted / tokens,
        "misses_per_token": misses / tokens,
        "is_oracle": int(pol.uses_oracle),
        "sim_seconds": round(secs, 2),
    }
    stats = pol.prediction_stats()
    if pol.prefetcher is not None and stats:
        row["precision_d1"], row["recall_d1"] = stats[1]
    for name, (link, priority) in _LINKS.items():
        rep = replay_policy(
            pol,
            link,
            demand_priority=priority,
            num_layers=int(_GEOM["num_layers"]),
            tokens=tokens,
        )
        row[f"stall_ms_{name}"] = rep.stall_per_token_s * 1e3
        row[f"xfer_per_tok_{name}"] = rep.transfers_per_token
        row[f"wasted_xfer_per_tok_{name}"] = rep.wasted_transfers / tokens
        if bound is not None:
            floor = stall_floor_s(n - bound, pol.visits, link)
            row[f"stall_floor_ms_{name}"] = floor / tokens * 1e3
        if name != NOMINAL:
            continue
        row[f"time_ms_{name}"] = rep.time_per_token_s * 1e3
        row[f"pf_xfer_per_tok_{name}"] = rep.prefetch_transfers / tokens
        row[f"cancelled_per_tok_{name}"] = rep.prefetch_cancelled / tokens
        row[f"late_per_tok_{name}"] = rep.covered_late / tokens
        row[f"promoted_per_tok_{name}"] = rep.prefetch_promoted / tokens
    return row


def _init(trace, geom, links, belady):
    global _TRACE, _GEOM, _LINKS, _BELADY
    _TRACE, _GEOM, _LINKS, _BELADY = trace, geom, links, belady


def run_jobs(jobs, workers: int):
    if workers <= 1 or len(jobs) <= 1:
        return [run_job(job) for job in jobs]
    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=ctx) as pool:
        return list(pool.map(run_job, jobs, chunksize=1))


def _cost(job) -> float:
    ev_spec, spec, adm, cap = job
    ev_name = parse_spec(ev_spec)[0]
    weight = {"lfu": 2 + cap / 1024}.get(ev_name, 1)
    if ev_name.startswith("belady") or ev_name in LCNU_POLICIES:
        weight = 4
    if spec.startswith(("knn", "logistic")):
        weight *= 2
    return weight * (2 if spec != "none" else 1)


def sweep(trace, geom, links, jobs, workers) -> List[Dict[str, object]]:
    """Belady per capacity first (for the assertion), then every job."""
    caps = sorted({job[3] for job in jobs})
    _init(trace, geom, links, {})
    kinds = ["belady"]
    if any(_bypasses(job[0]) for job in jobs):
        kinds.append("belady_bypass")
    belady = {
        (r["evictor"], r["capacity"]): r["hits"]
        for r in run_jobs(
            [(k, "", "", c) for k in kinds for c in caps], workers
        )
    }
    _init(trace, geom, links, belady)
    order = sorted(jobs, key=_cost, reverse=True)
    rows = run_jobs(order, workers)
    index = {job: i for i, job in enumerate(jobs)}
    rows.sort(
        key=lambda r: index[
            (r["evictor"], r["prefetcher"], r["admission"], r["capacity"])
        ]
    )
    return rows


# --------------------------------------------------------------------------- #
# Composition identity check
# --------------------------------------------------------------------------- #
class _Silent(PrefetchPolicy):
    """A prefetcher that never predicts (exercises the composed demand path)."""

    def _predict(self, layer, d):
        return []


def _identity_job(job: Tuple[str, int, str]) -> Tuple[str, int, str, int]:
    ev_name, cap, variant = job
    name, kwargs = parse_spec(ev_name)
    if name in GEOMETRY_POLICIES:
        kwargs["num_layers"] = int(_GEOM["num_layers"])
    if name in LCNU_POLICIES:
        kwargs["prior"] = _GEOM["top_k"] / _GEOM["num_experts"]
        kwargs.setdefault("decode_width", int(_GEOM["top_k"]))
    policy = build_policy(name, cap, **kwargs)
    if variant != "alone":
        pf = None
        adm = "same_queue"
        if variant != "none":
            pf = _Silent(num_layers=int(_GEOM["num_layers"]))
            adm = variant
        policy = PrefetchCachePolicy(
            policy,
            pf,
            admission=adm,
            num_layers=int(_GEOM["num_layers"]),
            prior=_GEOM["top_k"] / _GEOM["num_experts"],
        )
    return ev_name, cap, variant, simulate(policy, _TRACE)[0]


def identity_check(trace, geom, evictors, capacities, workers):
    """Hits of each evictor alone vs composed with no / a silent prefetcher."""
    variants = ["alone", "none"] + [m for m in ADMISSION_MODES if m != "buffer"]
    jobs = [
        (ev, cap, var)
        for ev in evictors
        for cap in capacities
        for var in variants
    ]
    _init(trace, geom, {}, {})
    if workers <= 1:
        results = [_identity_job(j) for j in jobs]
    else:
        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(
            min(workers, len(jobs)), mp_context=ctx
        ) as pool:
            results = list(pool.map(_identity_job, jobs))
    hits = {(ev, cap, var): h for ev, cap, var, h in results}
    lines = [
        "| Evictor | Capacity | Alone | "
        + " | ".join(f"composed ({v})" for v in variants[1:])
        + " | Identical |",
        "| --- | ---: | ---: | "
        + " | ".join("---:" for _ in variants[1:])
        + " | --- |",
    ]
    ok = True
    for ev in evictors:
        for cap in capacities:
            row = [hits[(ev, cap, v)] for v in variants]
            same = all(h == row[0] for h in row)
            ok &= same
            lines.append(
                f"| {ev} | {cap} | "
                + " | ".join(str(h) for h in row)
                + f" | {'yes' if same else 'NO'} |"
            )
    return ok, "\n".join(lines)


# --------------------------------------------------------------------------- #
# Predictor-only scoring
# --------------------------------------------------------------------------- #
def _predict_job(spec: str) -> Dict[str, object]:
    pf, _ = _make_prefetcher(spec, int(_GEOM["top_k"]))
    pf.bind_trace(_TRACE)
    max_d = pf.horizon
    pending: Dict[int, List[Tuple[int, List[Key]]]] = {}
    n = [0] * (max_d + 1)
    hit = [0] * (max_d + 1)
    routed_total = [0] * (max_d + 1)
    visit, start, last = -1, 0, None
    trace = _TRACE

    def close(v, keys):
        routed = set(keys)
        for d, cands in pending.pop(v, ()):
            n[d] += len(cands)
            hit[d] += sum(1 for k in cands if k in routed)
            routed_total[d] += len(routed)

    for i, key in enumerate(trace):
        if key[0] != last:
            if last is not None:
                close(visit, trace[start:i])
                for d, _, cands in pf.predict():
                    pending.setdefault(visit + d, []).append(
                        (d, [k for k, _ in cands])
                    )
            visit += 1
            start = i
            last = key[0]
        pf.observe(key)
    close(visit, trace[start:])
    row: Dict[str, object] = {"prefetcher": spec}
    for d in range(1, max_d + 1):
        row[f"precision_d{d}"] = hit[d] / n[d] if n[d] else 0.0
        row[f"recall_d{d}"] = (
            hit[d] / routed_total[d] if routed_total[d] else 0.0
        )
    return row


def predict_sweep(trace, geom, specs, workers):
    _init(trace, geom, {}, {})
    if workers <= 1:
        return [_predict_job(s) for s in specs]
    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(min(workers, len(specs)), mp_context=ctx) as pool:
        return list(pool.map(_predict_job, specs))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_split(files, split: str, prefill: str, ignore_layers: int = 0):
    """Access list + geometry for one half (or all) of the request files."""
    half = len(files) // 2
    chosen = {"tune": files[:half], "held": files[half:], "all": files}[split]
    trace, stats = load_accesses(
        chosen, ignore_layers=ignore_layers, prefill=prefill
    )
    if prefill == "skip":
        decode = len(trace)
    else:
        decode = len(load_accesses(chosen, ignore_layers=ignore_layers)[0])
    geom = {
        "num_layers": stats["num_layers"] - ignore_layers,
        "num_experts": stats["max_expert_id"] + 1,
        "top_k": stats["top_k"],
    }
    geom["tokens"] = decode / (geom["num_layers"] * geom["top_k"])
    return trace, geom, len(chosen), stats


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def write_csv(
    rows: Sequence[Dict[str, object]], path: Path, extra: Dict
) -> None:
    keys: List[str] = list(extra)
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            out = dict(extra)
            for key, val in row.items():
                out[key] = f"{val:.6g}" if isinstance(val, float) else val
            writer.writerow(out)


def render_md(rows, header: str) -> str:
    """Per capacity: Belady, each evictor alone, and the best composition."""
    lines = [header, ""]
    caps = sorted({r["capacity"] for r in rows})
    key = f"stall_ms_{NOMINAL}"
    for cap in caps:
        sub = [r for r in rows if r["capacity"] == cap]
        if not sub:
            continue
        bel = sub[0].get("belady_hit_rate", "")
        floor = sub[0].get(f"stall_floor_ms_{NOMINAL}", "")
        lines += [
            f"### capacity {cap}",
            "",
            f"Belady hit rate {bel:.4f}, stall floor {floor:.2f} ms/token "
            "(nominal timing).",
            "",
            "| Evictor | Prefetcher | Admission | Hit rate | Covered "
            "| Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: "
            "| ---: |",
        ]
        best = sorted(sub, key=lambda r: r[key])[:12]
        alone = [r for r in sub if r["prefetcher"] == "none"]
        shown = alone + [r for r in best if r not in alone]
        for r in shown:
            mark = "*" if r["is_oracle"] else ""
            lines.append(
                f"| {r['evictor']} | {r['prefetcher']}{mark} | {r['admission']} "
                f"| {r['hit_rate']:.4f} | {r['covered_rate']:.4f} "
                f"| {r[key]:.2f} | {r['stall_ms_fifo']:.2f} "
                f"| {r['stall_ms_roofline']:.2f} "
                f"| {r[f'xfer_per_tok_{NOMINAL}']:.1f} "
                f"| {r[f'wasted_xfer_per_tok_{NOMINAL}']:.1f} |"
            )
        lines.append("")
    lines.append(
        "`*` = trace-oracle prefetcher (upper bound). Full metrics: CSV."
    )
    return "\n".join(lines)


EXPERIMENTS_HEADER = (
    "| evictor | prefetcher | admission | model | dataset | capacity "
    "| hit_rate | covered | stall_ms_tok | provenance | commit |\n"
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
)


def experiment_rows(rows) -> List[Dict[str, object]]:
    """Compact log: per capacity each evictor alone, the best online pair
    per admission mode and the best oracle pair (full set in the CSV)."""
    key = f"stall_ms_{NOMINAL}"
    out = []
    for cap in sorted({r["capacity"] for r in rows}):
        sub = [r for r in rows if r["capacity"] == cap]
        out += [r for r in sub if r["prefetcher"] == "none"]
        online = [r for r in sub if r["prefetcher"] != "none"]
        for adm in ADMISSION_MODES:
            pick = [
                r
                for r in online
                if parse_spec(r["admission"])[0] == adm and not r["is_oracle"]
            ]
            if pick:
                out.append(min(pick, key=lambda r: r[key]))
        oracle = [r for r in online if r["is_oracle"]]
        if oracle:
            out.append(min(oracle, key=lambda r: r[key]))
    return out


def append_experiments(rows, path: Path, model, dataset, commit, command, env):
    with open(path, "a") as handle:
        handle.write("\n" + EXPERIMENTS_HEADER)
        for r in experiment_rows(rows):
            handle.write(
                f"| {r['evictor']} | {r['prefetcher']} | {r['admission']} "
                f"| {model} | {dataset} | {r['capacity']} "
                f"| {r['hit_rate']:.4f} | {r['covered_rate']:.4f} "
                f"| {r[f'stall_ms_{NOMINAL}']:.2f} | real | {commit} |\n"
            )
        handle.write(
            f"\n<!-- run: {command} | env: {env} | commit: {commit} -->\n"
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
class TunedSpecs:
    """Tuned prefetcher specs, looked up by the nearest key-space fraction.

    Both Qwen3 models have 128 experts and top-8, so a capacity that is the
    same fraction of the key space (``layers x experts``) is also the same
    multiple of one token's working set; a spec tuned on one model at
    ``cap / keyspace`` is reused for the other at the nearest fraction.
    """

    def __init__(self, path: str):
        data = json.loads(Path(path).read_text())
        self.keyspace = float(data["keyspace"])
        self.best: Dict[Tuple[str, str], List[Tuple[float, str]]] = {}
        for e in data["best"]:
            slot = (e["prefetcher"], e["admission"])
            frac = int(e["capacity"]) / self.keyspace
            self.best.setdefault(slot, []).append((frac, e["spec"]))

    def get(self, name: str, adm: str, cap: int, keyspace: float, default):
        options = self.best.get((name, adm))
        if not options:
            return default
        frac = cap / keyspace
        return min(options, key=lambda o: abs(math.log(o[0] / frac)))[1]


def build_jobs(args, geom, tuned: Optional[TunedSpecs] = None):
    keyspace = geom["num_layers"] * geom["num_experts"]
    jobs = []
    for ev, cap in itertools.product(args.evictors, args.capacities):
        for spec in args.prefetchers:
            name, _ = parse_spec(spec)
            if name == "none":
                jobs.append((ev, "none", "same_queue", cap))
                continue
            for adm in args.admissions:
                use = spec
                if tuned is not None and ":" not in spec:
                    adm_name = parse_spec(adm)[0]
                    use = tuned.get(name, adm_name, cap, keyspace, spec)
                jobs.append((ev, use, adm, cap))
    return list(dict.fromkeys(jobs))


#: Prefetchers that never propose more than one visit's routed experts.
_PER_VISIT = {"prev_token", "same_id"}


def grid_jobs(args, geom):
    jobs = []
    for spec in args.prefetchers:
        name, params = parse_spec(spec)
        if name == "none":
            continue
        budgets = list(args.budgets)
        if name in ORACLE_PREFETCHERS:
            budgets = budgets[:1]
        elif name in _PER_VISIT:
            budgets = sorted({min(m, int(geom["top_k"])) for m in budgets})
        for h, m in itertools.product(args.horizons, budgets):
            p = dict(params, horizon=h)
            if name not in ORACLE_PREFETCHERS:
                p["budget"] = m
            label = spec_label(name, p)
            for ev, adm, cap in itertools.product(
                args.evictors, args.admissions, args.capacities
            ):
                jobs.append((ev, label, adm, cap))
    for ev, cap in itertools.product(args.evictors, args.capacities):
        jobs.append((ev, "none", "same_queue", cap))
    return list(dict.fromkeys(jobs))


def select_best(rows) -> List[Dict[str, object]]:
    """Per (prefetcher, admission, capacity): the spec with the lowest mean
    nominal stall relative to each tuning evictor alone."""
    key = f"stall_ms_{NOMINAL}"
    alone = {
        (r["evictor"], r["capacity"]): r[key]
        for r in rows
        if r["prefetcher"] == "none"
    }
    score: Dict[Tuple, List[float]] = {}
    for r in rows:
        if r["prefetcher"] == "none":
            continue
        base = alone[(r["evictor"], r["capacity"])]
        rel = r[key] / base if base > 0 else 1.0 + r[key]
        slot = (
            parse_spec(r["prefetcher"])[0],
            parse_spec(r["admission"])[0],
            r["capacity"],
            r["prefetcher"],
        )
        score.setdefault(slot, []).append(rel)
    best: Dict[Tuple, Tuple[float, str]] = {}
    for (name, adm, cap, spec), rels in score.items():
        mean = sum(rels) / len(rels)
        if (name, adm, cap) not in best or mean < best[(name, adm, cap)][0]:
            best[(name, adm, cap)] = (mean, spec)
    return [
        {
            "prefetcher": name,
            "admission": adm,
            "capacity": cap,
            "spec": spec,
            "mean_relative_stall": round(mean, 6),
        }
        for (name, adm, cap), (mean, spec) in sorted(best.items())
    ]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--model", required=True, choices=sorted(LINK_PRESETS))
    p.add_argument("--dataset", default="mmlu-greedy64")
    p.add_argument("--split", choices=("tune", "held", "all"), default="held")
    p.add_argument("--prefill", choices=("skip", "union"), default="skip")
    p.add_argument(
        "--mode",
        choices=("sweep", "tune", "predict", "identity"),
        default="sweep",
    )
    p.add_argument("--evictors", nargs="+", default=list(EVICTORS))
    p.add_argument(
        "--prefetchers", nargs="+", default=["none", *PREFETCHERS, "oracle"]
    )
    p.add_argument("--admissions", nargs="+", default=list(ADMISSION_MODES))
    p.add_argument("--capacities", type=int, nargs="+", default=[256, 1024])
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--budgets", type=int, nargs="+", default=[4, 8])
    p.add_argument("--tuned", default=None, help="JSON from --mode tune.")
    p.add_argument("--tuned-out", default=None)
    p.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    p.add_argument("--out", default=str(Path(__file__).resolve().parent))
    p.add_argument("--tag", default=None)
    p.add_argument("--no-experiments-log", action="store_true")
    args = p.parse_args(argv)
    for adm in args.admissions:
        if parse_spec(adm)[0] not in ADMISSION_MODES:
            p.error(f"unknown admission {adm!r}")
    for spec in args.prefetchers:
        name = parse_spec(spec)[0]
        if name != "none" and name not in PREFETCHER_FACTORIES:
            p.error(f"unknown prefetcher {spec!r}")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    files = expand_files(args.files)
    trace, geom, n_files, stats = load_split(files, args.split, args.prefill)
    stream = {"skip": "decode", "union": "decode+prefill-union"}[args.prefill]
    dataset = f"{args.dataset}/{stream}/{args.split}"
    tag = args.tag or f"{args.model}_{args.split}_{args.mode}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    commit, env = _git_commit(), _env_string()
    extra = {
        "model": args.model,
        "dataset": dataset,
        "provenance": "real",
        "commit": commit,
    }
    print(f"{n_files} request files, {len(trace)} accesses, geometry {geom}")

    if args.mode == "identity":
        ok, table = identity_check(
            trace, geom, args.evictors, args.capacities, args.workers
        )
        text = (
            f"# Composition identity check - {args.model} / {dataset}\n\n"
            f"Demand hits of each evictor alone vs `PrefetchCachePolicy` with "
            f"no prefetcher and with a never-predicting prefetcher under each "
            f"shared admission mode. Commit `{commit}`.\n\n{table}\n"
        )
        (out / f"results_prefetch_{tag}.md").write_text(text)
        print(text)
        if not ok:
            raise SystemExit("composition changed an evictor's hits")
        return

    if args.mode == "predict":
        specs = [s for s in args.prefetchers if parse_spec(s)[0] != "none"]
        rows = predict_sweep(trace, geom, specs, args.workers)
        write_csv(rows, out / f"results_prefetch_{tag}.csv", extra)
        for r in rows:
            print(r)
        return

    links = links_for(args.model)
    if args.mode == "tune":
        jobs = grid_jobs(args, geom)
    else:
        tuned = TunedSpecs(args.tuned) if args.tuned else None
        jobs = build_jobs(args, geom, tuned)
    print(f"{len(jobs)} jobs on {args.workers} workers")
    start = time.time()
    rows = sweep(trace, geom, links, jobs, args.workers)
    print(f"done in {time.time() - start:.0f} s")
    csv_path = out / f"results_prefetch_{tag}.csv"
    write_csv(rows, csv_path, extra)
    print(f"Wrote {csv_path}")
    if args.mode == "tune":
        best = select_best(rows)
        target = Path(args.tuned_out or out / f"tuned_prefetch_{tag}.json")
        meta = {
            "model": args.model,
            "dataset": dataset,
            "keyspace": geom["num_layers"] * geom["num_experts"],
            "evictors": args.evictors,
            "objective": f"mean stall_ms_{NOMINAL} / evictor-alone stall",
            "commit": commit,
            "best": best,
        }
        target.write_text(json.dumps(meta, indent=1) + "\n")
        print(f"Wrote {target}")
        return
    header = (
        f"# Prefetch x eviction - {args.model} / {dataset}\n\n"
        f"REAL SGLang `per_token` routing, {n_files} requests, {len(trace)} "
        f"accesses ({geom['tokens']:.0f} decode tokens). Commit `{commit}`.\n"
        "Stall = timing-model stall per decode token (nominal: measured "
        "per-layer compute, measured PCIe 5.0 link, demand priority)."
    )
    md_path = out / f"results_prefetch_{tag}.md"
    md_path.write_text(render_md(rows, header) + "\n")
    print(f"Wrote {md_path}")
    if not args.no_experiments_log:
        command = (
            "python -m benchmarks.cache_simulator.research.run_prefetch_real "
            f"--model {args.model} --split {args.split} --prefill {args.prefill}"
            f" --tag {tag}"
        )
        append_experiments(
            rows,
            out / "EXPERIMENTS.md",
            args.model,
            dataset,
            commit,
            command,
            env,
        )


if __name__ == "__main__":
    main()
