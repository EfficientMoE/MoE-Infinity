"""Greedy next-token agreement between two trace_capture runs.

Both runs must use the same frozen prompt set and greedy decode. Agreement is
the fraction of decode positions whose token id matches the off baseline; this
is the Phase-3 primary fidelity metric for the real A/B gate.
"""

from __future__ import annotations

import argparse
import json
import statistics


def agreement(off: dict, pol: dict):
    ga, gb = off["gen_tokens"], pol["gen_tokens"]
    total = 0
    match = 0
    first_div = []
    for ta, tb in zip(ga, gb):
        n = min(len(ta), len(tb))
        total += n
        m = sum(1 for i in range(n) if ta[i] == tb[i])
        match += m
        first_div.append(next((i for i in range(n) if ta[i] != tb[i]), n))
    return match / max(total, 1), first_div


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--off", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    off = json.load(open(args.off))
    pol = json.load(open(args.policy))
    rate, fd = agreement(off, pol)
    stats = pol.get("fused_drop_stats") or {}
    res = {
        "label": args.label,
        "off": args.off,
        "policy": args.policy,
        "greedy_agreement": round(rate, 4),
        "mean_first_divergence_pos": round(statistics.mean(fd), 1),
        "p99_off_ms": round(off.get("p99_ms", 0.0), 2),
        "p99_policy_ms": round(pol.get("p99_ms", 0.0), 2),
        "p99_ratio": round(
            pol.get("p99_ms", 0.0) / max(off.get("p99_ms", 1e-9), 1e-9), 3
        ),
        "p50_policy_ms": round(pol.get("p50_ms", 0.0), 2),
        "experts_dropped": stats.get("experts_dropped"),
        "tokens_changed": stats.get("tokens_changed"),
    }
    print(json.dumps(res, indent=2))
    if args.out_json:
        json.dump(res, open(args.out_json, "w"), indent=2)


if __name__ == "__main__":
    main()
