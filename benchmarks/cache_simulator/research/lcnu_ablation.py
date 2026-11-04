"""LCNU hyper-parameter selection, ablations and a routing-oracle bound.

Protocol (no tuning on reported numbers): the per-request trace files are
split in order into a tuning half and a held-out half. LCNU's
``(alpha, beta)`` grid is scored on the tuning half only (mean decode hit rate
over the capacities); the selected setting, its ablations and the baselines
are then reported on the held-out half.

Ablations of the selected setting:

* ``no cycle term`` - drop ``d(L)`` (rank by expected further visits only),
* ``EMA only`` - ``beta = 1`` (recency / burstiness estimate only),
* ``frequency only`` - ``beta = 0`` (smoothed long-run frequency only).

``--oracle-requests N`` additionally runs a bound on what *routing
prediction* could add to eviction: on the first ``N`` requests, LCNU is told
the true routing of the current layer and of the next ``h`` layer visits
(read from the trace), for each ``h`` in ``--horizons``.

Usage::

    python -m benchmarks.cache_simulator.research.lcnu_ablation \
        --files "<data>/belady_qwen3_30b/*.pt" --capacities 128 256 512 1024 2048
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from benchmarks.cache_simulator.belady_cache_simulator import (
    belady_simulate,
    extract_access_sequence,
    load_records,
)
from benchmarks.cache_simulator.policies import (
    LayerCycleNextUsePolicy,
    build_policy,
    layer_segments,
    simulate,
)
from benchmarks.cache_simulator.research.run_policies_real import (
    _top_k,
    expand_files,
)

Key = Tuple[int, int]

ALPHAS = (0.1, 0.2, 0.3, 0.4, 0.5)
BETAS = (0.3, 0.5, 0.7, 0.85)
BASELINES = ("lru", "lfu", "arc", "lirs", "s3fifo", "w_tinylfu")


class OracleHorizonLCNU(LayerCycleNextUsePolicy):
    """LCNU that knows the true routing of the next ``horizon`` layer visits.

    The current layer's routing is always known (its gate has run); a
    resident whose layer runs within ``horizon`` layer-steps is scored by its
    exact next-visit outcome: routed -> ``d``, not routed -> ``d + L/p``.
    Bounds what a perfect routing predictor could add to eviction.
    """

    uses_oracle = True

    def __init__(self, capacity: int, horizon: int = 1, **kwargs):
        super().__init__(capacity, **kwargs)
        self.horizon = horizon

    def bind_trace(self, accesses: Sequence[Key]) -> None:
        super().bind_trace(accesses)
        segments = layer_segments(accesses)
        self._seg_sets = [set(accesses[s:e]) for _, s, e in segments]
        self._seg_of: List[int] = []
        for index, (_, start, end) in enumerate(segments):
            self._seg_of.extend([index] * (end - start))

    def _victim(self) -> Tuple[int, Key]:
        cur, cycle = self._cur, self.cycle
        seg = self._seg_of[self._index]
        best = None
        for layer, keys in self._by_layer.items():
            d0 = (layer - cur) % cycle
            known = None
            if layer == cur:
                known = self._seg_sets[seg]
            elif d0 <= self.horizon and seg + d0 < len(self._seg_sets):
                known = self._seg_sets[seg + d0]
            for key in keys:
                base = cycle * (1.0 / self.routing_probability(key) - 1.0)
                if layer == cur:
                    if key in known and key not in self._used_now:
                        score = -1.0
                    else:
                        score = cycle + base
                elif known is None:
                    score = d0 + base
                elif key in known:
                    score = float(d0)
                else:
                    score = d0 + cycle + base
                if (
                    best is None
                    or score > best[0]
                    or (score == best[0] and key < best[1])
                ):
                    best = (score, key, layer)
        return best[2], best[1]


def routing_stats(requests: Sequence[Sequence[Key]]) -> Dict[str, float]:
    """Skew and token-to-token reuse of the decode routing.

    ``reuse_next_token``: P(an expert routed at layer L for token t is routed
    at layer L for token t+1 of the same request). ``reuse_if_independent``:
    the same probability if every (layer, expert) were routed independently
    per token with its overall frequency.
    """
    counts: Dict[Key, int] = {}
    tokens = reused = routed = 0
    for seq in requests:
        prev: Dict[int, set] = {}
        cur: Dict[int, set] = {}
        last_layer = None
        for key in seq:
            layer = key[0]
            if last_layer is not None and layer < last_layer:  # next token
                tokens += 1
                for lay, experts in cur.items():
                    if lay in prev:
                        routed += len(prev[lay])
                        reused += len(prev[lay] & experts)
                prev, cur = cur, {}
            cur.setdefault(layer, set()).add(key[1])
            counts[key] = counts.get(key, 0) + 1
            last_layer = layer
        tokens += 1
        for lay, experts in cur.items():
            if lay in prev:
                routed += len(prev[lay])
                reused += len(prev[lay] & experts)
    freqs = sorted(counts.values(), reverse=True)
    total = sum(freqs)
    top = freqs[: max(1, len(freqs) // 5)]
    p = [c / tokens for c in freqs]
    return {
        "distinct_keys": len(freqs),
        "top20pct_key_share": sum(top) / total,
        "reuse_next_token": reused / max(routed, 1),
        "reuse_if_independent": sum(x * x for x in p) / sum(p),
    }


def load_requests(
    files: Sequence[str], ignore_layers: int
) -> Tuple[List[List[Key]], Dict[str, int]]:
    """Decode access list per request file, plus the model geometry."""
    requests, num_layers, top_k, max_expert = [], 0, 0, -1
    for path in files:
        records = load_records(path)
        num_layers = max(
            num_layers, int(records[0]["topk_ids_of_layer"].shape[0])
        )
        top_k = max(top_k, _top_k(records))
        seq = extract_access_sequence(
            records, ignore_layers=ignore_layers, include_prefill=False
        )
        max_expert = max([max_expert] + [e for _, e in seq])
        requests.append(seq)
    geometry = {
        "num_layers": num_layers,
        "top_k": top_k,
        "num_experts": max_expert + 1,
    }
    return requests, geometry


# Worker state, inherited through fork (policy makers are closures).
_TRACE: List[Key] = []
_MAKERS: Dict[str, Optional[Callable[[int], object]]] = {}


def _job(args) -> Tuple[str, int, float]:
    label, cap = args
    make = _MAKERS[label]
    if make is None:
        return label, cap, belady_simulate(_TRACE, cap)[2]
    return label, cap, simulate(make(cap), _TRACE)[2]


def evaluate(
    trace: Sequence[Key],
    makers: Dict[str, Optional[Callable[[int], object]]],
    capacities: Sequence[int],
    workers: int,
) -> Dict[str, Dict[int, float]]:
    """Hit rate of every policy maker at every capacity (``None`` = Belady)."""
    global _TRACE, _MAKERS
    _TRACE, _MAKERS = list(trace), dict(makers)
    jobs = [(label, cap) for label in makers for cap in capacities]
    if workers > 1:
        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(
            min(workers, len(jobs)), mp_context=ctx
        ) as pool:
            results = list(pool.map(_job, jobs))
    else:
        results = [_job(job) for job in jobs]
    out: Dict[str, Dict[int, float]] = {label: {} for label in makers}
    for label, cap, rate in results:
        out[label][cap] = rate
    return out


def _lcnu(geometry, **kwargs):
    prior = geometry["top_k"] / max(geometry["num_experts"], 1)

    def make(cap):
        return LayerCycleNextUsePolicy(
            cap, num_layers=geometry["num_layers"], prior=prior, **kwargs
        )

    return make


def _oracle(geometry, horizon, alpha, beta):
    prior = geometry["top_k"] / max(geometry["num_experts"], 1)

    def make(cap):
        return OracleHorizonLCNU(
            cap,
            horizon=horizon,
            alpha=alpha,
            beta=beta,
            num_layers=geometry["num_layers"],
            prior=prior,
        )

    return make


def _baseline(name):
    return lambda cap: build_policy(name, cap)


def table(rows: Dict[str, Dict[int, float]], capacities: Sequence[int]) -> str:
    lines = [
        "| Policy | " + " | ".join(f"cap={c}" for c in capacities) + " |",
        "| --- | " + " | ".join("---" for _ in capacities) + " |",
    ]
    for label, rates in rows.items():
        cells = " | ".join(f"{rates[c]:.4f}" for c in capacities)
        lines.append(f"| {label} | {cells} |")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--ignore-layers", type=int, default=0)
    parser.add_argument(
        "--capacities", type=int, nargs="+", default=[128, 256, 512, 1024, 2048]
    )
    parser.add_argument(
        "--workers", type=int, default=min(16, os.cpu_count() or 1)
    )
    parser.add_argument("--oracle-requests", type=int, default=0)
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16]
    )
    parser.add_argument("--out", default=None, help="Write the report here.")
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print routing statistics and exit.",
    )
    args = parser.parse_args(argv)

    files = expand_files(args.files)
    requests, geometry = load_requests(files, args.ignore_layers)
    if args.stats_only:
        stats = routing_stats(requests)
        print({**geometry, **{k: round(v, 4) for k, v in stats.items()}})
        return
    half = len(requests) // 2
    tune = [k for req in requests[:half] for k in req]
    held = [k for req in requests[half:] for k in req]
    caps = args.capacities
    report = [
        "# LCNU selection and ablations",
        "",
        f"{len(requests)} requests ({len(tune)} tuning / {len(held)} held-out "
        f"decode accesses); geometry {geometry}.",
        "",
    ]

    grid = {
        f"alpha={a} beta={b}": _lcnu(geometry, alpha=a, beta=b)
        for a in ALPHAS
        for b in BETAS
    }
    scores = evaluate(tune, grid, caps, args.workers)
    ranked = sorted(
        scores, key=lambda label: -sum(scores[label].values()) / len(caps)
    )
    best = ranked[0]
    alpha, beta = (float(part.split("=")[1]) for part in best.split())
    report += [
        "## Tuning half: (alpha, beta) grid",
        "",
        table({label: scores[label] for label in ranked}, caps),
        "",
        f"Selected: **{best}** (best mean hit rate over the capacities).",
        "",
    ]

    makers: Dict[str, Optional[Callable[[int], object]]] = {"Belady": None}
    makers.update({name: _baseline(name) for name in BASELINES})
    makers[f"LCNU ({best})"] = _lcnu(geometry, alpha=alpha, beta=beta)
    makers["- no cycle term"] = _lcnu(
        geometry, alpha=alpha, beta=beta, cycle_aware=False
    )
    makers["- EMA only (beta=1)"] = _lcnu(geometry, alpha=alpha, beta=1.0)
    makers["- frequency only (beta=0)"] = _lcnu(geometry, alpha=alpha, beta=0.0)
    report += [
        "## Held-out half",
        "",
        table(evaluate(held, makers, caps, args.workers), caps),
        "",
    ]

    if args.oracle_requests > 0:
        subset = [k for req in requests[: args.oracle_requests] for k in req]
        oracle_makers: Dict[str, Optional[Callable[[int], object]]] = {
            "Belady": None,
            "LCNU (online)": _lcnu(geometry, alpha=alpha, beta=beta),
        }
        for h in args.horizons:
            oracle_makers[f"LCNU + true routing, h={h}"] = _oracle(
                geometry, h, alpha, beta
            )
        report += [
            f"## Routing-oracle bound (first {args.oracle_requests} requests)",
            "",
            table(evaluate(subset, oracle_makers, caps, args.workers), caps),
            "",
        ]

    text = "\n".join(report).rstrip("\n")
    print(text)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
