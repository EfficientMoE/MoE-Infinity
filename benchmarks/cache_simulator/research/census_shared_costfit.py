"""Memory / structure census + timing for shared-table CostFit (exp675 stage A).

For each (policy, capacity) a FRESH spawned process loads the decode stream,
then measures:

* peak / terminal ``tracemalloc`` bytes over build + full replay,
* a deterministic STRUCTURE CENSUS of the per-key state (record count, slots
  held, ``sys.getsizeof`` of tables + records, walked in sorted-key order),
* per-access wall time (``perf_counter``, median of ``--repeats`` fresh runs,
  tracemalloc off).

Usage::

    python -m benchmarks.cache_simulator.research.census_shared_costfit \
        --files data/belady/stream_qwen30b.pt --capacities 1024 \
        --policies costfit shared_costfit costfit3 shared_costfit3

``--parity`` instead runs the behavior-parity smoke (costfit vs shared_costfit,
costfit3 vs shared_costfit3) on a seeded synthetic stream.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import random
import statistics
import sys
import time
import tracemalloc
from typing import Dict, List, Sequence, Tuple

from benchmarks.cache_simulator.policies import build_policy
from benchmarks.cache_simulator.research.run_policies_real import (
    geometry_kwargs,
    load_accesses,
)

Key = Tuple[int, int]

DEFAULT_POLICIES = ["costfit", "shared_costfit", "costfit3", "shared_costfit3"]
CSV_FIELDS = [
    "policy",
    "capacity",
    "accesses",
    "hit_rate",
    "peak_bytes",
    "terminal_bytes",
    "tables",
    "records",
    "slots",
    "struct_bytes",
    "ns_per_access",
]
_PER_KEY_DICTS = ("_last", "_ema", "_count", "_first", "freq")


# --------------------------------------------------------------------------- #
# structure census
# --------------------------------------------------------------------------- #
def _leaves(policy) -> list:
    """Leaf cache objects (LCNU / S3-FIFO) in deterministic structural order."""
    if hasattr(policy, "_inners"):
        return [x for p in policy._inners.values() for x in _leaves(p)]
    if hasattr(policy, "lcnu") and hasattr(policy, "s3fifo"):
        return _leaves(policy.lcnu) + _leaves(policy.s3fifo)
    return [policy]


def structure_census(policy) -> Dict[str, int]:
    """Per-key state footprint: tables, records, slots held, getsizeof total."""
    tables = records = slots = size = 0
    seen = set()
    for leaf in _leaves(policy):
        shared = getattr(leaf, "shared_table", None)
        if shared is not None:
            if id(shared) in seen:
                continue
            seen.add(id(shared))
            tables += 1
            size += sys.getsizeof(shared.table)
            for key in sorted(shared.table):  # deterministic walk
                rec = shared.table[key]
                records += 1
                slots += rec.live
                size += sys.getsizeof(rec)
        else:
            for attr in _PER_KEY_DICTS:
                d = getattr(leaf, attr, None)
                if isinstance(d, dict):
                    tables += 1
                    records += len(d)
                    slots += len(d)
                    size += sys.getsizeof(d)
    return {
        "tables": tables,
        "records": records,
        "slots": slots,
        "struct_bytes": size,
    }


# --------------------------------------------------------------------------- #
# per-policy worker (spawned: fresh interpreter => clean allocator state)
# --------------------------------------------------------------------------- #
def _replay(policy, accesses: Sequence[Key]) -> None:
    policy.bind_trace(accesses)
    on_access = policy.on_access
    for key in accesses:
        on_access(key)


def _measure(job) -> Dict[str, object]:
    name, cap, files, ignore_layers, prefill, repeats, limit = job
    accesses, stats = load_accesses(
        files, ignore_layers=ignore_layers, prefill=prefill
    )
    if limit:
        accesses = accesses[:limit]
    kw = geometry_kwargs(name, stats)

    tracemalloc.start()
    policy = build_policy(name, cap, **kw)
    _replay(policy, accesses)
    _, peak = tracemalloc.get_traced_memory()
    cur, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    census = structure_census(policy)
    hit_rate = policy.hit_rate

    times: List[float] = []
    for _ in range(repeats):
        p = build_policy(name, cap, **kw)
        t0 = time.perf_counter()
        _replay(p, accesses)
        times.append(time.perf_counter() - t0)
        del p
    ns = statistics.median(times) / max(len(accesses), 1) * 1e9
    return {
        "policy": name,
        "capacity": cap,
        "accesses": len(accesses),
        "hit_rate": hit_rate,
        "peak_bytes": peak,
        "terminal_bytes": cur,
        **census,
        "ns_per_access": round(ns, 1),
    }


def run_census(
    files,
    capacities,
    policies,
    *,
    ignore_layers=0,
    prefill="skip",
    repeats=3,
    limit=0,
) -> List[Dict[str, object]]:
    ctx = multiprocessing.get_context("spawn")
    rows = []
    for cap in capacities:
        for name in policies:
            job = (
                name,
                cap,
                list(files),
                ignore_layers,
                prefill,
                repeats,
                limit,
            )
            with ctx.Pool(1) as pool:  # one fresh process per measurement
                rows.append(pool.apply(_measure, (job,)))
    return rows


def summarize(rows: Sequence[Dict[str, object]]) -> str:
    by = {(r["policy"], r["capacity"]): r for r in rows}
    out = ["", "== summary =="]
    for r in rows:
        out.append(
            f"{r['policy']:>16} cap={r['capacity']:<6} peak={r['peak_bytes'] / 1e6:8.2f}MB "
            f"terminal={r['terminal_bytes'] / 1e6:8.2f}MB struct={r['struct_bytes'] / 1e6:8.2f}MB "
            f"tables={r['tables']} records={r['records']} slots={r['slots']} "
            f"{r['ns_per_access']:8.1f}ns/acc hit={r['hit_rate']:.6f}"
        )
    for (name, cap), r in sorted(by.items()):
        base = (
            by.get((name[len("shared_") :], cap))
            if name.startswith("shared_")
            else None
        )
        if base:
            out.append(
                f"{name} vs {base['policy']} @cap {cap}: "
                f"peak x{r['peak_bytes'] / base['peak_bytes']:.3f}, "
                f"terminal x{r['terminal_bytes'] / base['terminal_bytes']:.3f}, "
                f"struct x{r['struct_bytes'] / base['struct_bytes']:.3f}, "
                f"time x{r['ns_per_access'] / base['ns_per_access']:.2f}, "
                f"hit_rate equal={r['hit_rate'] == base['hit_rate']}"
            )
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# parity smoke
# --------------------------------------------------------------------------- #
def synthetic_stream(
    n: int = 200_000,
    layers: int = 24,
    experts: int = 64,
    top_k: int = 4,
    seed: int = 675,
) -> List[Key]:
    """Decode-like stream: per token, per layer, top_k Zipf-ish experts."""
    rng = random.Random(seed)
    weights = [1.0 / (i + 1) for i in range(experts)]
    out: List[Key] = []
    while len(out) < n:
        for layer in range(layers):
            picks = set()
            while len(picks) < top_k:
                picks.add(rng.choices(range(experts), weights)[0])
            out.extend((layer, e) for e in sorted(picks))
    return out[:n]


def _instrument(policy, log: list) -> None:
    """Log every wrapper ``_select_victim`` call and every inner eviction."""
    orig_sv = policy._select_victim

    def sv(protect, _o=orig_sv):
        log.append(("select_victim", tuple(protect)))
        return _o(protect)

    policy._select_victim = sv
    for label, leaf in enumerate(_leaves(policy)):
        orig = leaf._evict

        def ev(key, _o=orig, _l=label):
            log.append(("evict", _l, key))
            _o(key)

        leaf._evict = ev


def parity_smoke(n: int = 200_000, cap: int = 1024) -> bool:
    stream = synthetic_stream(n)
    kw = {"num_layers": 24, "prior": 4 / 64}
    ok = True
    for base, shared in (
        ("costfit", "shared_costfit"),
        ("costfit3", "shared_costfit3"),
    ):
        res = []
        for name in (base, shared):
            p = build_policy(name, cap, **kw)
            log: list = []
            _instrument(p, log)
            _replay(p, stream)
            res.append((p, log))
        (a, la), (b, lb) = res
        ta = (
            a.hits,
            a.misses,
            a.hit_rate,
            a.switches,
            getattr(a, "choice_switches", None),
            a.choice,
        )
        tb = (
            b.hits,
            b.misses,
            b.hit_rate,
            b.switches,
            getattr(b, "choice_switches", None),
            b.choice,
        )
        mass_eq = a._mass == b._mass and a._sw_mass == b._sw_mass
        res_eq = list(a.resident) == list(b.resident)
        same = ta == tb and la == lb and mass_eq and res_eq
        ok &= same
        print(f"{base:9} {ta}")
        print(f"{shared:9} {tb}")
        print(
            f"  victim/evict events: {len(la)} vs {len(lb)}, sequences equal="
            f"{la == lb}; masses equal={mass_eq}; resident order equal={res_eq}"
            f" -> {'PARITY OK' if same else 'PARITY FAIL'}"
        )
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--files", nargs="+")
    ap.add_argument("--capacities", type=int, nargs="+", default=[1024])
    ap.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES)
    ap.add_argument("--ignore-layers", type=int, default=0)
    ap.add_argument(
        "--prefill", default="skip", choices=("skip", "union", "raw")
    )
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="truncate the stream to N accesses (0 = all)",
    )
    ap.add_argument(
        "--parity",
        action="store_true",
        help="run the synthetic-stream parity smoke and exit",
    )
    args = ap.parse_args(argv)
    if args.parity:
        return 0 if parity_smoke() else 1
    if not args.files:
        ap.error("--files is required unless --parity")
    rows = run_census(
        args.files,
        args.capacities,
        args.policies,
        ignore_layers=args.ignore_layers,
        prefill=args.prefill,
        repeats=args.repeats,
        limit=args.limit,
    )
    writer = csv.DictWriter(
        sys.stdout, fieldnames=CSV_FIELDS, lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    print(summarize(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
