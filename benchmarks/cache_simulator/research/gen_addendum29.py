#!/usr/bin/env python3
"""Programmatic ADDENDUM29 generator for exp683 (door under flash crowds)."""

import json
from pathlib import Path

D = Path(__file__).parent
R = json.loads((D / "exp683-doorcrowd.json").read_text())
assert R["verdict"] == "DOOR-CROWD-HARM" and R["det"]

rows = [r for r in R["rows"]]
worst = max(rows, key=lambda r: r["door_burst_def"] - r["cf_burst_def"])
rec_flip = [r for r in rows if r["door_rec_def"] < 0]
d4c5 = [r for r in rows if r["d"] == 4096 and r["cap"] == 512]

L = [
    "# Addendum 29 — exp683 doorkeeper under flash crowds (DOOR-CROWD-HARM)",
    "",
    "Home-field test for the doorkeeper: exp681's EXACT crowd streams",
    "(anchor cell reproduced to the digit in smoke), arms lcnu/s3fifo/",
    "center/door512/door4096, 8 cells. Pre-registered consequence fires:",
    "**admission filtering is dead at the wrapper level** for this policy",
    "family.",
    "",
    f"Verdict: **{R['verdict']}** (det byte-identical, acc exact).",
    f"- H1 (burst deficit shrinks): only {R['h1_shrink']} combos.",
    f"- H3 (recovery deficit shrinks): {R['h3_shrink']}.",
    f"- H2 safety violated in {R['h2_bad']}/16 (loop hit > 0.5pp below",
    "  center).",
    f"- Worst in-burst harm: {worst['arm']} at d={worst['d']}/cap={worst['cap']}",
    f"  burst deficit {worst['door_burst_def']:.2f}pp vs center's",
    f"  {worst['cf_burst_def']:.2f}pp.",
    "",
    "## The mechanistic split (why this is still a positive result)",
    "",
    f"- Recovery FLIPS TO ADVANTAGE in {len(rec_flip)} combos: at",
    f"  d=4096/cap=512 door4096 turns the center's 6.98pp recovery deficit",
    f"  into a {d4c5[1]['door_rec_def']:.2f}pp advantage — hot-set",
    "  protection works exactly as designed when one-timers are blocked.",
    "- But in-burst harm grows everywhere (door delays the admission of",
    "  repeating crowd keys during the burst window).",
    "- Combined with exp681 (sw=1 aligned control still lost 10.86pp) and",
    "  exp682 (stationary harm): **the lever is admission SPEED with",
    "  post-admission demotion, not admission delay.** Admit immediately;",
    "  evict never-rehit keys first.",
    "",
    "## Pre-registered branch taken",
    "",
    "HARM -> wrapper-level admission filtering closed; exp684 targets the",
    "inner admission/eviction priority: probation subclass (admitted on",
    "first sight at highest eviction priority until first re-hit).",
    "",
    f"Cells 8 | 5 arms | det {R['det']} | elapsed {R['elapsed_s']}s",
    "",
    "Generated programmatically from exp683-doorcrowd.json",
    "(gen_addendum29.py).",
    "",
]
out = "\n".join(L)
(D / "ADDENDUM29.md").write_text(out)
print(
    f"ADDENDUM29.md {len(out.encode())} bytes | rec flips {len(rec_flip)} | worst burst delta {worst['door_burst_def']-worst['cf_burst_def']:+.2f}pp"
)
