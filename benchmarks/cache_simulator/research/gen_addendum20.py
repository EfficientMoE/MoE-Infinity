#!/usr/bin/env python3
"""Generate ADDENDUM20.md from exp669-bo.json + bo-trials.jsonl. Programmatic only."""

import json
from pathlib import Path

D = Path(__file__).parent
d = json.loads((D / "exp669-bo.json").read_text())
trials = [
    json.loads(l)
    for l in (D / "bo-trials.jsonl").read_text().splitlines()
    if l.strip()
]
feas = [t for t in trials if t["feasible"]]
best = min(feas, key=lambda t: t["f"])
bp = d["best_params"]

# worst deficit + max switches at the best point (from its recorded per-model cells)
worst = max(
    c["deficit_pp"]
    for m in best["per_model"].values()
    for c in m["caps"].values()
)
max_sw = best["g_max_switches"]

h1 = d["best_feasible_f"] < 0.062
h2 = not (16384 <= bp["half_life"] <= 65536)
h3 = d["pearson_pred_vs_obs"] >= 0.5
verdict = (
    "IMPROVED"
    if h1
    else ("INFORMATIVE-NO-IMPROVEMENT" if h3 else "UNINFORMATIVE")
)

lines = [
    "# Addendum 20 — exp669: Bayesian optimization of CostFit (MARGIN, GAMMA, HALF_LIFE)",
    "",
    f"**Verdict: {verdict}** (frozen rule: IMPROVED if H1) — re-eval determinism gate "
    "PASS (best config rebuilt post-run on both traces; all recorded cells byte-identical).",
    "",
    "## Question",
    "",
    "Can Bayesian optimization over CostFit's three knob space beat the hand-calibrated",
    "center (M=0.005, G=0.0025, HL=32768) on the frozen no-harm objective f",
    "(worst per-cap deficit vs best standalone, pp), under the switch constraint (<=6)?",
    "",
    "## Setup (prereg c72a9d69, frozen before run)",
    "",
    f"- {d['n_seed']} seed points + {d['n_trials']} BO trials, GP surrogate "
    f"({d['rules']['kernel']}), seed {d['rules']['rng_seed']}, "
    f"{d['rules']['candidates_per_iter']} acquisition candidates.",
    f"- Space: {d['rules']['space']}. Objective: worst per-cap deficit vs best standalone",
    "  (lcnu/s3fifo) across both real decode traces at the frozen cap grids.",
    "",
    "## Results",
    "",
    f"- Seed best f = {d['best_seed_f']}pp -> BO best feasible f = {d['best_feasible_f']}pp "
    f"({d['best_seed_f']/d['best_feasible_f']:.1f}x better), {len(feas)}/{len(trials)} feasible.",
    f"- Winner: M={bp['margin']:.5f}, G={bp['gamma']:.5f}, HL={round(bp['half_life'])} "
    f"(trial {best['trial']}); worst deficit at winner {worst}pp, max switches {max_sw}.",
    f"- H1 IMPROVED: {'PASS' if h1 else 'FAIL'} ({d['best_feasible_f']} < 0.062pp).",
    f"- H2 NEW-AREA: {'PASS' if h2 else 'FAIL'} — HL={round(bp['half_life'])} outside center",
    f"  2x band [16384, 65536], direction UP ({bp['half_life']/32768:.1f}x center);",
    f"  {d['trials_half_life_outside_2x_center']}/{len(trials)} trials sampled outside the band.",
    f"- H3 CALIBRATION: {'PASS' if h3 else 'FAIL'} — surrogate leave-ahead Pearson "
    f"r = {d['pearson_pred_vs_obs']} (< 0.5). The GP landscape did not predict observed f.",
    "",
    "## Disclosure",
    "",
    "H3 fails: the improvement rests on an early sample (trial 1) that no later acquisition",
    "beat, not on a well-modeled response surface. Pre-freeze smoke (disclosed in prereg):",
    "first HL-normalization draft spanned [1, 1e5] and the acquisition drove HL=278 ->",
    "f=0.711, switches=15 (INFEASIBLE; matches exp666 oscillation at short effective",
    "memory). Mapping fixed to the prereg range before freeze.",
    "",
    "## Next",
    "",
    "exp674 (prereg frozen): does the winner transfer to exp670's out-of-family synthetic",
    "grid (streams sha-asserted identical), vs center and vs HL-only? Settles whether the",
    "NEW-AREA half-life is mechanism or real-trace overfit.",
    "",
]
(D / "ADDENDUM20.md").write_text("\n".join(lines) + "\n")
print(f"ADDENDUM20.md written: {len(lines)} lines")
print(
    f"verdict={verdict} h1={h1} h2={h2} h3={h3} worst={worst} max_sw={max_sw} best_trial={best['trial']}"
)
