#!/usr/bin/env python3
"""Programmatic ADDENDUM26 generator for exp681 (flash-crowd probe)."""
import json
from pathlib import Path

D = Path(__file__).parent
R = json.loads((D / "exp681-crowd.json").read_text())
assert R["verdict"] == "CROWD-H1" and R["det"]

rows = R["rows"]
worst_b = max(rows, key=lambda r: r["d_burst_pp"])
worst_r = max(rows, key=lambda r: r["d_rec_pp"])
by_cap = {}
for r in rows:
    by_cap.setdefault(r["cap"], []).append(r)

L = [
    "# Addendum 26 — exp681 sub-block flash-crowd probe (CROWD-H1)",
    "",
    "exp677 localized harm to arbitration-interval alignment; exp680 ruled",
    "out cadence. exp681 tested the remaining axis: a cold-crowd burst (layer",
    "uniform, expert uniform over ranks [64,128)) replacing d accesses at",
    "midpoint of a stationary stream (skew 1.1, lam 0.3), d in {256, 1024,",
    "2048, 4096}, caps {512, 2048}, 8 cells; d=4096 is the aligned control",
    "the barrier CAN see.",
    "",
    f"Verdict: **{R['verdict']}** — H1 violated in {R['h1_viol']}/8 cells",
    f"(burst deficit vs best standalone, bar 1.0pp), H2 in {R['h2_viol']},",
    f"H3 clean (sw<=1), accounting exact, determinism byte-identical.",
    "",
    "## Findings",
    "",
    f"- EVERY cell violates the burst bar; range {min(r['d_burst_pp'] for r in rows):.2f}"
    f"-{max(r['d_burst_pp'] for r in rows):.2f}pp. Worst: d={worst_b['d']},",
    f"cap={worst_b['cap']} at **{worst_b['d_burst_pp']:.2f}pp** (costfit burst",
    f"hit {worst_b['burst_cf']:.3f} vs best standalone +{worst_b['d_burst_pp']:.2f}pp).",
    "- Costfit's burst-window hit collapses (0.016-0.188 across cells):",
    "  repeating crowd keys are admitted too slowly to serve the burst,",
    "  while the best standalone adapts within the burst.",
    f"- CAP-DEPENDENT MODE SPLIT: at cap 2048 harm concentrates IN the burst",
    f"(worst {worst_b['d_burst_pp']:.2f}pp); at cap 512 the burst deficit",
    f"shrinks ({by_cap[512][0]['d_burst_pp']:.2f}-{max(r['d_burst_pp'] for r in by_cap[512]):.2f}pp) but RECOVERY breaks:",
    f"worst recovery {worst_r['d_rec_pp']:.2f}pp at d={worst_r['d']}/cap={worst_r['cap']}",
    "(crowd admissions evicted the hot set; the post-burst stream pays).",
    "- Blindness is not even required: the ALIGNED CONTROL (d=4096, boundary-",
    "visible, sw=1) still loses 10.86pp at cap 2048 — the arbitration fires",
    "and cannot help, consistent with exp680's cadence-irrelevance. The",
    "barrier's binary inner-policy switch has no crowd response at any",
    "cadence; the failure is in ADMISSION SPEED of repeating cold keys.",
    "",
    "## Cell table (deficit vs best standalone, pp)",
    "",
    "| d | cap | d_burst | d_rec | sw |",
    "|---:|---:|---:|---:|---:|",
]
for r in sorted(rows, key=lambda x: (x["d"], x["cap"])):
    L.append(f"| {r['d']} | {r['cap']} | {r['d_burst_pp']:.3f} | {r['d_rec_pp']:.3f} | {r['sw']} |")
L += [
    "",
    "## Third failure mode; convergence",
    "",
    "Series to date: alignment (exp677), temporal miss clustering (exp678),",
    "crowd admission lag (exp681, this). All three point at the same lever:",
    "ADMISSION of cold/repeating keys, not arbitration. The doorkeeper",
    "candidate (exp682, in flight on exp678's testbed) blocks first-sight",
    "admissions but its second-sight admit is exactly the crowd-serving",
    "path — pre-registered pivot: door on THESE crowd streams next.",
    "",
    f"Cells {len(rows)} | h1 {R['h1_viol']} | h2 {R['h2_viol']} | h3",
    f"{R['h3_viol']} | acc {R['acc_all']} | det {R['det']} | base",
    f"{R['rules']['base']} | crowd {R['rules']['crowd']}",
    "",
    "Generated programmatically from exp681-crowd.json (gen_addendum26.py).",
    "",
]
out = "\n".join(L)
(D / "ADDENDUM26.md").write_text(out)
print(f"ADDENDUM26.md {len(out.encode())} bytes | worst burst {worst_b['d_burst_pp']}pp (d{worst_b['d']}/cap{worst_b['cap']}) | worst rec {worst_r['d_rec_pp']}pp")
