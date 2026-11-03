#!/usr/bin/env python3
"""Generate ADDENDUM21.md from exp674-hl-transfer.json. Programmatic only."""
import json
from pathlib import Path

D = Path(__file__).parent
d = json.loads((D / "exp674-hl-transfer.json").read_text())
bp = d["bo_params"]
sv = [v for v in d["violations"] if v["mode"].startswith("SAFETY")]
worst = max(sv, key=lambda v: v["deficit_pp"])
n_skew14 = sum(1 for v in sv if v["cfg"]["s"] == 1.4)
beats = sum(1 for st in d["grid"] for c in st["caps"].values()
            if c["cf_bo"]["rate"] > c["cf_center"]["rate"])
tot = sum(len(st["caps"]) for st in d["grid"])
hl = round(bp["half_life"])

lines = [
    "# Addendum 21 — exp674: does the exp669 BO winner transfer out-of-family?",
    "",
    f"**Verdict: {d['verdict']}** — the BO point (HL={hl}, {bp['half_life']/32768:.1f}x center) "
    "improves the mean deficit but violates the no-harm barrier on the synthetic grid. "
    "The center calibration stands; the BO objective overfit the worst case to two real traces.",
    "",
    "## Question (prereg 71776837, frozen before run)",
    "",
    "exp669 found IMPROVED at NEW-AREA (f=0.027pp on the real traces). Transfer test: evaluate",
    "cf@BO-winner vs cf@center vs cf@HL-only on exp670's 11-stream synthetic grid",
    "(streams sha-asserted byte-identical to exp670), caps {128..4096}, N=638,208/stream.",
    "",
    "## Results",
    "",
    f"- {d['informative_cells']} informative cells (|lcnu - s3fifo| >= 0.1pp); double-run "
    f"determinism {'PASS' if d['determinism_two_runs'] else 'FAIL'}; accounting "
    f"{'PASS' if d['H4_accounting'] else 'FAIL'} (loop==prop every arm every cell).",
    f"- H1 SAFETY: FAIL — {len(sv)}/{d['informative_cells']} cells exceed 0.5pp deficit; worst "
    f"{worst['deficit_pp']}pp (s={worst['cfg']['s']}, lam={worst['cfg']['lam']}, "
    f"E={worst['cfg']['E']}@cap{worst['cap']}); max switches {max(v['switches'] for v in sv)} "
    "(harm is genuine, not oscillation).",
    f"- H2 TRANSFER: FAIL — mean deficit delta vs center {d['mean_delta_bo_pp']}pp "
    "(better on average) but safety gate dominates; BO point wins only {beats}/{tot} cells.",
    f"- HL-only arm: mean {d['mean_delta_hlonly_pp']}pp — roughly half the mean gain carries "
    "in the half-life alone; per-cell safety of the HL-only point was not a frozen gate.",
    f"- Violation structure: {n_skew14}/{len(sv)} safety violations at s=1.4 (highest skew):",
    "  the long-memory BO region over-commits on skewed streams and pays at small-mid caps.",
    "",
    "## Reading",
    "",
    "Third independent confirmation that the barrier catches what mean-based selection misses",
    "(exp666 oscillation, exp673 toxic candidate, exp674 BO point). The NEW-AREA region is real",
    f"and large (HL={hl} wins big on {beats} high-skew cells) but is not certifiable under no-harm",
    "as a point selection. Future BO must optimize the worst case over a grid that includes",
    "out-of-family streams (joint real+synthetic objective) — queued as the follow-up direction.",
    "",
]
(D / "ADDENDUM21.md").write_text("\n".join(lines) + "\n")
print(f"ADDENDUM21.md written: {len(lines)} lines")
print(f"verdict={d['verdict']} safety_cells={len(sv)} worst={worst['deficit_pp']} beats={beats}/{tot} skew14={n_skew14}")
