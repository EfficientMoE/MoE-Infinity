#!/usr/bin/env python3
"""Programmatic ADDENDUM31 generator for exp685 (bounded nursery ring)."""
import json
from pathlib import Path

D = Path(__file__).parent
R = json.loads((D / "exp685-ring.json").read_text())
assert R["verdict"] == "IN_BURST_FIXED_ONLY" and R["det"]
assert R["repl"] == "exp681+exp683 exact"

g = {(c["d"], int(k)): v for c in R["grid"] for k, v in c["caps"].items()}
home = g[(4096, 512)]           # the gating cell
worst_def = max(
    (r for r in R["rows"] if r["burst_def_cr_pp"] > 0),
    key=lambda r: r["burst_def_cr_pp"]) if any(
    r["burst_def_cr_pp"] > 0 for r in R["rows"]) else R["rows"][0]
pp = lambda x: f"{x * 100:+.2f}pp"

L = [
    "# Addendum 31 — exp685 bounded nursery ring on crowd streams (IN_BURST_FIXED_ONLY)",
    "",
    "Michael's 2026-11-03 design, built from exp683's decomposition and",
    "exp684's measured ring-less failure: RingProbationLCNU admits at",
    "first sight into a bounded FIFO nursery ring; the ring drains (oldest",
    "one-timer first) ONLY when full; while it has spare capacity victims",
    "come from LCNU scoring over all residents; a rehit promotes the key",
    "out of the ring (full citizen). ring_frac 0.10/0.25 standalone, 0.25",
    "as the composite's lcnu inner. exp681's exact streams; arms lcnu |",
    "center | door4096 | ring10 | ring25 | center_ring25; 8 cells; LEDGER",
    "safe-settle v2 freeze 63d4e09 (dual anchors byte-exact in smoke:",
    "center == exp681, door4096 == exp683 at d1024/cap512).",
    "",
    "## Verdict: IN_BURST_FIXED_ONLY (det True; repl exact 24 cells)",
    "",
    "- IN-BURST (primary bar, deficit <= exp681 center's): PASS in all 8",
    f"  cells — composite deficit {pp(worst_def['burst_def_cr_pp'] / 100)} worst at",
    f"  d{worst_def['d']}/cap{worst_def['cap']} vs the door's 12.7-14.8pp, and",
    "  exactly 0.0 at ALL cap2048 cells: the door's admission-delay",
    "  catastrophe is erased, not traded.",
    "- RECOVERY: ring25 rec >= lcnu - 0.5pp in all cells (PASS); the one",
    "  FAILED gate is the composite at d4096/cap512:",
    f"  rec_def {pp((home['lcnu']['rec'] - home['center_ring25']['rec']))} == exp681",
    "  center's 6.98pp to 3 decimals — center_ring25 is byte-equal to",
    "  center at cap512, i.e. the ring inner is INVISIBLE to composite",
    "  arbitration there (the s3fifo inner owns victim selection at small",
    "  cap; matches exp684's attribution).",
    "",
    "## The standalone result that matters",
    "",
    "On the home cell (d4096/cap512) ring10 is a new Pareto point BETWEEN",
    "the door and lcnu:",
    f"- lcnu     burst {home['lcnu']['burst']:.4f}  rec {home['lcnu']['rec']:.4f}",
    f"- ring10   burst {home['ring10']['burst']:.4f}  rec {home['ring10']['rec']:.4f}",
    f"- door4096 burst {home['door4096']['burst']:.4f}  rec {home['door4096']['rec']:.4f}",
    f"- center   burst {home['center']['burst']:.4f}  rec {home['center']['rec']:.4f} (recovery collapse)",
    "ring10 keeps ~half the door's recovery edge over lcnu at +0.83pp",
    "more burst than the door — the nursery ring buys protection without",
    "the door's in-burst tax. At cap2048 the ring is pure churn-waste",
    "(ring10 burst -5.6..-9.2pp vs lcnu for +0.2..+1.4pp rec): when",
    "capacity covers the working set the FIFO ring only destroys scoring",
    "value — the ring earns its keep exactly where one-timers displace",
    "citizens.",
    "",
    "## Branch consequence (frozen) + amended diagnosis (disclosed)",
    "",
    "Frozen IN_BURST_FIXED_ONLY branch: exp686a lazy promotion /",
    "ghost-eviction history. Data amendment: the composite's failed gate",
    "is inner INVISIBILITY at cap512 (byte-equal to center), not ring",
    "overflow into protected — so exp686a additionally tests arbitration",
    "visibility (ring inner as a first-class composite candidate at small",
    "cap); secondary: stride-adaptive ring sizing (cap2048 waste).",
    "",
    f"Data: exp685-ring.json (elapsed {R['elapsed_s']}s; 2 passes identical;",
    "exit 0).",
]
(D / "ADDENDUM31.md").write_text("\n".join(L) + "\n")
print("ADDENDUM31.md written,", len(L), "lines")
