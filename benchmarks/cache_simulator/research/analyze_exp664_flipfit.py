#!/usr/bin/env python3
"""exp664 FlipFit: which online window statistic tracks the LCNU->S3FIFO crossover?

Reads frozen traces and frozen baseline CSVs only. No policy simulation.
Statistics (a-priori, preregistered): sliding window W=8192 accesses,
re-evaluated every 4096, first 3 evaluations skipped (warm-up), median over
remaining evaluations:
  S1 repeat set : |{key: count >= 2}|
  S2 core set   : |{key: count >= 3}|
  S3 cov90 set  : smallest hottest-key set covering >= 90% of window accesses  [PRIMARY]
  S4 cov95 set  : smallest hottest-key set covering >= 95% of window accesses

Arbitration model: `capacity >= S -> S3FIFO else LCNU` (theta = 1.0).
Predicted first-S3FIFO cap = smallest cap in the model's union grid >= median(S).
Observed flip band = (last LCNU-win cap, first S3FIFO-win cap] from frozen CSVs.
Hit: predicted cap lies inside the observed band.

H1: S3 hits on 30b.  H2: S3 hits on granite.
H3: at least one secondary (S1/S2/S4) hits on BOTH models.
Verdict: SUPPORTED iff H1 and H2 and H3; PARTIAL iff H1 and H2; else FAIL.
"""

import csv
import json
import statistics
import sys

sys.path.insert(0, "/Users/mfethe/warp-work/moei-granite3b")
from benchmarks.cache_simulator.belady_cache_simulator import (
    extract_access_sequence,
    load_records,
)

R = "/Users/mfethe/warp-work/moei-granite3b/benchmarks/cache_simulator/research"
W, STEP, WARM = 8192, 4096, 3

MODELS = {
    "qwen30b": {
        "trace": "/Users/mfethe/warp-work/moei-granite3b/data/belady/stream_qwen30b.pt",
        "csvs": [
            f"{R}/results_real_qwen3-30b-a3b_decode.csv",
            f"{R}/results_real_qwen3-30b-a3b-fine.csv",
        ],
    },
    "granite": {
        "trace": "/Users/mfethe/warp-work/moei-granite3b/benchmarks/cache_simulator/data/handcrafted8-greedy512/belady/stream_granite3b.pt",
        "csvs": [
            f"{R}/results_real_granite3_1_moe_3b_decode.csv",
            f"{R}/results_real_granite3.1-moe-3b-fine.csv",
        ],
    },
}


def stats_for(seq):
    per = {s: [] for s in ("S1", "S2", "S3", "S4")}
    n_eval = 0
    for i in range(W, len(seq) + 1, STEP):
        n_eval += 1
        if n_eval <= WARM:
            continue
        win = seq[i - W : i]
        counts = {}
        for k in win:
            counts[k] = counts.get(k, 0) + 1
        vals = sorted(counts.values(), reverse=True)
        per["S1"].append(sum(1 for v in vals if v >= 2))
        per["S2"].append(sum(1 for v in vals if v >= 3))
        total = len(win)
        for name, cov in (("S3", 0.90), ("S4", 0.95)):
            need, cum, n = cov * total, 0, len(vals)
            for j, v in enumerate(vals, start=1):
                cum += v
                if cum >= need:
                    n = j
                    break
            per[name].append(n)
    return {s: statistics.median(v) for s, v in per.items()}, {
        s: len(v) for s, v in per.items()
    }


def winners(csvs):
    rates = {}
    for p in csvs:
        for r in csv.DictReader(open(p)):
            if r["policy"] in ("lcnu", "s3fifo"):
                rates[(r["policy"], int(r["capacity"]))] = float(r["hit_rate"])
    caps = sorted({c for _, c in rates})
    w = {}
    for c in caps:
        l, s = rates.get(("lcnu", c)), rates.get(("s3fifo", c))
        if l is not None and s is not None:
            w[c] = (
                "lcnu"
                if l > s + 5e-7
                else ("s3fifo" if s > l + 5e-7 else "tie")
            )
    return w


out_stats, out_verdict = {}, {"per_model": {}}
for m, cfg in MODELS.items():
    seq = extract_access_sequence(
        load_records(cfg["trace"]), ignore_layers=0, include_prefill=False
    )
    med, n = stats_for(seq)
    w = winners(cfg["csvs"])
    lcnu_caps = [c for c, x in w.items() if x == "lcnu"]
    s3_caps = [c for c, x in w.items() if x == "s3fifo"]
    band = (max(lcnu_caps), min(s3_caps))
    grid = sorted(w)
    hits = {}
    for s in ("S1", "S2", "S3", "S4"):
        pred = next((c for c in grid if c >= med[s]), None)
        hits[s] = (
            {
                "median": med[s],
                "predicted_cap": pred,
                "hit": band[0] < pred <= band[1],
            }
            if pred
            else {"median": med[s], "hit": False}
        )
    out_stats[m] = {
        "medians": med,
        "n_evals_after_warmup": n["S1"],
        "winners": w,
        "band": band,
        "hits": hits,
    }
    h1 = hits["S3"]["hit"] if m == "qwen30b" else None
    out_verdict["per_model"][m] = {
        "band": list(band),
        "hits": {s: hits[s]["hit"] for s in hits},
        "medians": med,
        "predicted": {s: hits[s].get("predicted_cap") for s in hits},
    }
    print(
        m,
        "band",
        band,
        "medians",
        med,
        "hits",
        {s: hits[s]["hit"] for s in hits},
        flush=True,
    )

p30, pg = (
    out_verdict["per_model"]["qwen30b"],
    out_verdict["per_model"]["granite"],
)
sec_both = [s for s in ("S1", "S2", "S4") if p30["hits"][s] and pg["hits"][s]]
out_verdict.update(
    {
        "H1_primary_cov90_30b": p30["hits"]["S3"],
        "H2_primary_cov90_granite": pg["hits"]["S3"],
        "H3_secondary_hits_both": sec_both,
        "H3_pass": bool(sec_both),
        "verdict": (
            "SUPPORTED"
            if (p30["hits"]["S3"] and pg["hits"]["S3"] and sec_both)
            else (
                "PARTIAL"
                if (p30["hits"]["S3"] and pg["hits"]["S3"])
                else "FAIL"
            )
        ),
    }
)
json.dump(
    out_stats,
    open("/Users/mfethe/warp-work/exp664-moei-flipfit/exp664-stats.json", "w"),
    indent=1,
)
json.dump(
    out_verdict,
    open(
        "/Users/mfethe/warp-work/exp664-moei-flipfit/exp664-verdict.json", "w"
    ),
    indent=1,
)
print(
    json.dumps(
        {k: v for k, v in out_verdict.items() if k != "per_model"}, indent=1
    )
)
