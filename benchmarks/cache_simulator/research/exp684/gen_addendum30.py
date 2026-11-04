#!/usr/bin/env python3
"""Programmatic ADDENDUM30 generator for exp684 (ring-less probation)."""
import json
from pathlib import Path

D = Path(__file__).parent
R = json.loads((D / "exp684-prob.json").read_text())
assert R["verdict"] == "PROB-HARM" and R["det"] and R["repl"]

rows = R["rows"]
worst_burst = min(rows, key=lambda r: r["pl_vs_l_burst_pp"])
worst_cp = min(rows, key=lambda r: r["cp_vs_cf_rec_pp"])
w = lambda r, k: f"{r[k]:+.2f}pp"

L = [
    "# Addendum 30 — exp684 ring-less probation on crowd streams (PROB-HARM)",
    "",
    "exp683's decomposition (k>=2 door admission burns 1/k of the burst",
    "reuse window; the recovery win was one-timers never displacing the",
    "hot set) implies two levers: admit fast + demote later. exp684 is the",
    "unbounded endpoint of that lever: ProbationLCNU admits at first sight",
    "and evicts the oldest never-rehit resident first (no partition, no",
    "ring). Sealed LayerCycleNextUsePolicy subclass; exp681's exact crowd",
    "streams; arms lcnu | costfit-center | prob_lcnu | center_prob (prob",
    "as the composite's lcnu inner); 8 cells; LEDGER.md pilot freeze",
    "fe1d97f (prereg + gates + branches in one commit).",
    "",
    "## Verdict: PROB-HARM (det True, replication exact vs exp681+exp683)",
    "",
    f"- H1 (no in-burst loss): FALSE — burst deficit vs lcnu in ALL 8",
    f"  cells, range {w(min(rows, key=lambda r: r['pl_vs_l_burst_pp']) if False else worst_burst, 'pl_vs_l_burst_pp')}"
    f" (d{worst_burst['d']}/cap{worst_burst['cap']}) to",
    f"  {w(max(rows, key=lambda r: r['pl_vs_l_burst_pp']), 'pl_vs_l_burst_pp')}.",
    f"- Loop cost: {w(worst_burst, 'pl_vs_l_loop_pp')} (cap2048 cells ~ -2.2pp).",
    "- H3 (recovery): 4/8 — small wins at cap512 (+0.6..+1.2pp),",
    f"  LOSSES at cap2048; the composite pays: center_prob rec",
    f"  {w(worst_cp, 'cp_vs_cf_rec_pp')} vs center at d{worst_cp['d']}/cap{worst_cp['cap']}.",
    "",
    "## Mechanism",
    "",
    "Unbounded probation has no capacity to hold the burst reuse stride:",
    "under a d-key crowd burst every admission evicts the previous",
    "one-timer before its rehit can occur, so crowd keys churn among",
    "themselves and the burst hit rate collapses — the same failure the",
    "door caused, moved inside the cache. The recovery protection the",
    "door bought (one-timers never entering) is only partially retained",
    "(4/8) and the composite's other inners pay for the swap at cap2048.",
    "",
    "## Consequence (pre-registered branch fired)",
    "",
    "PROB-HARM => probation needs bounded, partitioned capacity: a",
    "nursery ring that drains only when full, with the protected region",
    "scoring-managed (Michael's 2026-11-03 design). exp685 = that ring",
    "(ring_frac 0.10/0.25), frozen 63d4e09 with dual-Pareto bars.",
    "",
    f"Data: exp684-prob.json (elapsed {R.get('elapsed_s', '?')}s, 2 passes",
    "byte-identical; center/door replication cells exact).",
]
(D / "ADDENDUM30.md").write_text("\n".join(L) + "\n")
print("ADDENDUM30.md written,", len(L), "lines")
