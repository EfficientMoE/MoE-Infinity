#!/usr/bin/env python3
"""gen_addendum21_fix.py — same-day correction generator for ADDENDUM21.md (exp674).

The original generator left unrendered {beats}/{tot} placeholders and inverted
the sign convention of the mean-delta fields (rates are HIT rates; the runner's
mean_delta_bo_pp = mean(cf_bo - cf_center), so negative = WORSE). This script
recomputes every corrected number directly from the sealed result JSON
(exp674-hl-transfer.json, copied to research/ at publication) and applies
surgical replacements. It aborts without writing if any expected anchor string
or verified number is missing.

Verdict is UNCHANGED (SAFETY-BROKEN rests on H1: 8/59 informative cells exceed
the 0.5pp no-harm barrier, worst 0.779pp; verified independently).
"""

import json
import pathlib
import sys

R = pathlib.Path(__file__).resolve().parent
SRC = R / "exp674-hl-transfer.json"
ADD = R / "ADDENDUM21.md"
REP = R / "capture_report.md"

res = json.loads(SRC.read_text())
assert res["verdict"] == "SAFETY-BROKEN"
assert res["informative_cells"] == 59 and len(res["violations"]) == 8
assert res["worst_bo_deficit"]["deficit_pp"] == 0.779

rows = [
    ((st["cfg"], cap), (c["cf_bo"]["rate"] - c["cf_center"]["rate"]) * 100)
    for st in res["grid"]
    for cap, c in st["caps"].items()
]
beats = [d for _, d in rows if d > 0]
losses = [d for _, d in rows if d < 0]
tot = len(rows)
mean_win = sum(beats) / len(beats)
max_win = max(beats)
mean_loss = sum(losses) / len(losses)
mean_bo = res["mean_delta_bo_pp"]
mean_hlo = res["mean_delta_hlonly_pp"]
assert len(beats) == 11 and len(losses) == 20 and tot == 66

add = ADD.read_text()
fixes = [
    # header line: sign correction
    (
        "the BO point (HL=160975, 4.9x center) improves the mean deficit but violates "
        "the no-harm barrier on the synthetic grid.",
        f"the BO point (HL=160975, 4.9x center) degrades the mean hit rate vs center "
        f"({mean_bo:+.4f}pp) and violates the no-harm barrier on the synthetic grid.",
    ),
    # H2 line: sign + unrendered placeholders
    (
        "H2 TRANSFER: FAIL — mean deficit delta vs center -0.1234pp (better on average) "
        "but safety gate dominates; BO point wins only {beats}/{tot} cells.",
        f"H2 TRANSFER: FAIL — mean hit-rate delta vs center {mean_bo:+.4f}pp (WORSE on "
        f"average; hit rates, so negative = below center, beyond the 0.05pp tie-tolerance "
        "— H2 fails independently of the safety gate); BO point wins only "
        f"{len(beats)}/{tot} cells.",
    ),
    # HL-only line: sign
    (
        "HL-only arm: mean -0.0613pp — roughly half the mean gain carries in the "
        "half-life alone; per-cell safety of the HL-only point was not a frozen gate.",
        f"HL-only arm: mean {mean_hlo:+.4f}pp — roughly half of the mean DEGRADATION "
        "comes from the half-life alone; per-cell safety of the HL-only point was not "
        "a frozen gate.",
    ),
    # reading line: "wins big" corrected with measured magnitudes
    (
        "The NEW-AREA region is real\nand large (HL=160975 wins big on 11 high-skew cells) but is not certifiable under no-harm\nas a point selection.",
        f"The BO point's {len(beats)} wins are marginal\n(mean {mean_win:+.4f}pp, max {max_win:+.3f}pp, mostly at cap 4096) while its "
        f"{len(losses)} losses\naverage {mean_loss:+.4f}pp — the mean degradation is broad, not offset by large wins —\nand it is not certifiable under no-harm as a point selection.",
    ),
]
for old, new in fixes:
    if old not in add:
        sys.exit(f"ABORT: anchor not found in ADDENDUM21.md: {old[:60]!r}")
    add = add.replace(old, new)

note = (
    "\n## Correction (same day, 2026-11-03)\n\n"
    "Fixes an unrendered `{beats}/{tot}` placeholder and three sign/magnitude "
    "misstatements introduced by the original generator: the mean delta fields are "
    "HIT-rate deltas (negative = worse than center), so the BO point is worse on "
    "average (-0.1234pp), the HL-only arm's -0.0613pp is degradation, and the BO "
    "point's 11 wins are marginal (mean +0.0375pp, max +0.134pp) against 20 losses "
    "averaging -0.3848pp. All numbers recomputed from exp674-hl-transfer.json by "
    "gen_addendum21_fix.py. Verdict unchanged: SAFETY-BROKEN (H1: 8/59 cells >0.5pp, "
    "worst 0.779pp, max switches 2).\n"
)
if "## Correction (same day" not in add:
    add = add.rstrip() + "\n" + note
ADD.write_text(add)

rep = REP.read_text()
old_rep = (
    "**Verdict: SAFETY-BROKEN** — the BO point (HL=160975, 4.9x center) improves "
    "the mean deficit but violates the no-harm barrier on the synthetic grid."
)
new_rep = (
    "**Verdict: SAFETY-BROKEN** — the BO point (HL=160975, 4.9x center) degrades "
    f"the mean hit rate vs center ({mean_bo:+.4f}pp) and violates the no-harm "
    "barrier on the synthetic grid (see ADDENDUM21.md same-day correction)."
)
if old_rep in rep:
    REP.write_text(rep.replace(old_rep, new_rep))
    rep_fixed = True
elif new_rep in rep:
    rep_fixed = True  # already applied
else:
    sys.exit("ABORT: capture_report.md anchor not found")

print(
    "ADDENDUM21.md corrected; capture_report.md",
    "corrected" if rep_fixed else "SKIP",
)
print(
    f"beats={len(beats)}/{tot} mean_win={mean_win:+.4f} max_win={max_win:+.3f} "
    f"mean_loss={mean_loss:+.4f} n_loss={len(losses)} mean_bo={mean_bo:+.4f} "
    f"mean_hlo={mean_hlo:+.4f}"
)
