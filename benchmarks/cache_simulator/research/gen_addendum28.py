#!/usr/bin/env python3
"""Programmatic ADDENDUM28 generator for exp682 (doorkeeper on stationary)."""
import json
from pathlib import Path

D = Path(__file__).parent
R = json.loads((D / "exp682-door.json").read_text())
assert R["verdict"] == "DOOR-HARM" and R["det"] and R["replication"]

rows = [r for r in R["rows"] if isinstance(r.get("arm"), str)]
d4 = [r for r in rows if r["phi"] > 0]
best = max(d4, key=lambda r: r["d_time_pct"])
worst_hit = min(rows, key=lambda r: r["d_hit_pp"])

L = [
    "# Addendum 28 — exp682 doorkeeper admission on stationary streams (DOOR-HARM)",
    "",
    "First IMPROVEMENT-shaped candidate of the series: a doorkeeper",
    "admission filter (first-sight missing keys are not admitted — they",
    "evict nothing; second sight admits via the base policy; K-LRU door)",
    "wrapped around the frozen center, tested on exp678's exact cost-model",
    "testbed (16 cells, replication gate). Mechanism rationale: block the",
    "one-touch cold admissions whose extra misses cluster in time (exp678).",
    "",
    f"Verdict: **{R['verdict']}** — the door HURTS on stationary streams.",
    f"Replication exact ({R['rules']['replication_cells']} cells vs exp678",
    "published — also proving the concurrent lane's additive policies.py",
    "change altered no arm semantics). Determinism byte-identical.",
    "",
    "## Numbers (vs center arm)",
    "",
    f"- H2 violations: {R['h2_bad']}/16 door-cell combos lose > 0.5pp hit.",
    f"- H3 violations: {R['h3_bad']} combos run > 0.1% SLOWER than center.",
    f"- H1: only {R['h1_pass']}/{R['h1_cells']} phi>0 combos improve time.",
    f"- Worst hit loss: {worst_hit['d_hit_pp']:.2f}pp (skew {worst_hit['skew']},",
    f"cap {worst_hit['cap']}, {worst_hit['arm']}). Best time gain: only",
    f"{best['d_time_pct']:+.2f}% (skew {best['skew']}, cap {best['cap']}, {best['arm']}).",
    f"- Generalization probe: lcnu+door beats bare lcnu in {R['lcnu_door_better']}",
    "cells — the harm is policy-independent.",
    "",
    "## Interpretation",
    "",
    "On stationary streams the door's protection value is zero (no crowd to",
    "block) while its double-miss tax on one-touch reuse is constant: net",
    "harm everywhere. This is the correct null-space result for a mechanism",
    "whose benefit is conditional on burst structure — and it sharpens the",
    "prediction for the crowd testbed (exp683): there, one-touch crowd keys",
    "are the PROVEN poison (exp681: 10.86pp in-burst, 6.98pp recovery), so",
    "the door's tax should be outweighed by hot-set preservation. If the",
    "door still harms under crowds, admission filtering is dead as a lever",
    "for this policy family and the fix must live inside inner admission.",
    "",
    "## Pre-registered branches honored",
    "",
    "DOOR-NULL/HARM -> door under exp681 flash-crowd streams (home field).",
    "Launched as exp683 immediately after this publication.",
    "",
    f"Cells 16 | arms center/door512/door4096/lcnu/lcnu_door | det",
    f"{R['det']} | repl {R['replication']} | elapsed {R['elapsed_s']}s",
    "",
    "Generated programmatically from exp682-door.json (gen_addendum28.py).",
    "",
]
out = "\n".join(L)
(D / "ADDENDUM28.md").write_text(out)
print(f"ADDENDUM28.md {len(out.encode())} bytes | h2bad {R['h2_bad']}/16 | h3bad {R['h3_bad']} | worst hit {worst_hit['d_hit_pp']}pp")
