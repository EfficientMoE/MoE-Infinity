#!/usr/bin/env python3
"""Programmatic ADDENDUM32 generator for exp686a (ring first-class candidacy)."""
import json
from pathlib import Path

D = Path(__file__).parent
R = json.loads((D / "exp686a-ringfit3.json").read_text())
assert R["verdict"] == "IN_BURST_KEPT_VISIBLE" and R["det"]
assert R["repl"] == "exp681 costfit + exp685 ring10/ring25 exact"

home = [r for r in R["rows"] if r["d"] == 4096 and r["cap"] == 512][0]
g10_ladder = [(r["d"], r["cap"], r["burst_def_ringfit3_10_pp"],
               r["burst_def_cf_pp"]) for r in R["rows"]
              if r["burst_def_ringfit3_10_pp"] > r["burst_def_cf_pp"]]
rf3_equal_cells = [r for r in R["rows"]
                   if r["rec_def_ringfit3_pp"] == r["rec_def_cf_pp"]]
a10_home = home["adopt_ringfit3_10"]
adopt512 = [r for r in R["rows"] if r["cap"] == 512
            and r["adopt_ringfit3_10"]["rec"] == "ring"]
sd = {s["d"]: s for s in R["stride_diag"]}

L = [
    "# Addendum 32 — exp686a ring as a first-class arbitration candidate "
    "(IN_BURST_KEPT_VISIBLE)",
    "",
    "exp685's failed gate was the ring inner being INVISIBLE to 2-way",
    "arbitration at small cap (composite rec_def == center's exactly).",
    "exp686a made the ring a THIRD CANDIDATE of the sealed CostFit3",
    "barrier (HALF_LIFE 32768, MARGIN 0.005, GAMMA 0.0025, RESEGMENT",
    "4096, initial lcnu — all inherited unedited; exp681's exact streams;",
    "8 cells; LEDGER safe-settle v2 freeze 2735a8e; dual anchors",
    "byte-exact in smoke). Pre-registered prediction: the ring CANNOT",
    "displace s3fifo at the primary cell d4096/cap512 -> H1 fails.",
    "",
    "## Verdict: IN_BURST_KEPT_VISIBLE (det True; repl exact 24 cells)",
    "",
    "The prediction failed in the most informative way. On the registered",
    "primary arm (ring 0.25) it held: ringfit3 sat on s3fifo through the",
    f"primary cell (rec_def {home['rec_def_ringfit3_pp']:+.2f}pp == center's "
    f"{home['rec_def_cf_pp']:+.2f}pp; byte-equal behavior",
    f"{len(rf3_equal_cells)}/8 cells) and the ladder passed everywhere",
    "(H2). But the ring 0.10 variant ADOPTED the ring at all four cap512",
    f"cells — {len(adopt512)}/4 — and at the primary cell its recovery is "
    f"BETTER than lcnu's: rec_def {home['rec_def_ringfit3_10_pp']:+.2f}pp "
    "(vs center's +6.98 collapse), loop vs lcnu "
    f"{home['loop_vs_l_ringfit3_10_pp']:+.2f}pp, sw "
    f"{a10_home['sw']}, sw_by {a10_home['sw_by']}. First-class candidacy",
    "alone CAN recover the recovery collapse — when the ring is small",
    "enough to fit the working set.",
    "",
    "## The tradeoff that remains (why not RING-FIRST-CLASS)",
    "",
    "No arm holds BOTH bars at cap512. ringfit3_10 pays for its recovery",
    f"win in-burst: burst_def {home['burst_def_ringfit3_10_pp']:+.2f}pp vs "
    f"center's {home['burst_def_cf_pp']:+.2f}pp at the primary cell "
    f"(ladder fails {len(g10_ladder)}/8 cells: worst "
    f"{max(x[2] - x[3] for x in g10_ladder):+.2f}pp over center). Riding",
    "the ring through the BURST gives up the s3fifo inner's in-burst",
    "edge; riding s3fifo through the RECOVERY collapses. Under a decayed",
    "cumulative-rate scoreboard the two windows need different judges —",
    "the handoff Michael's ladder wants cannot happen at re-segment",
    "boundaries alone.",
    "",
    "## Secondary dose + stride diagnostic (for exp687 / exp686b)",
    "",
    f"- lazy2 standalone (promotion on 2nd rehit): home rec_def "
    f"{home['rec_def_lazy2_pp']:+.2f}pp, burst_def "
    f"{home['burst_def_lazy2_pp']:+.2f}pp — strictly worse than eager "
    "ring25 at the home cell; lazy promotion is a dead dose (kept as a",
    "  negative finding).",
    f"- Stride (pure stream stats, no policy behavior): d4096 crowd "
    f"rehit gap mean {sd[4096]['gap_mean']}/p90 {sd[4096]['gap_p90']} "
    f"accesses, concurrent-probation load mean {sd[4096]['load_mean']} — "
    "yet ring_cap 51 (frac .10 @cap512, far below gap p90) wins recovery",
    "while ring_cap 128 (frac .25) is never adopted: ring sizing tracks",
    "the one-timer share, not the reuse stride; exp686b sweep input.",
    "",
    "## Branch consequence (frozen)",
    "",
    "IN_BURST_KEPT_VISIBLE -> exp687: arbitration-layer redesign —",
    "windowed challenger scoreboard (challenger judged on last-4096 rate",
    "vs incumbent decayed cumulative, same barrier family) so the",
    "composite can sit s3fifo for the burst and hand off to the ring for",
    "recovery; sealed victim-selection policies untouched.",
    "",
    f"Data: exp686a-ringfit3.json (elapsed {R['elapsed_s']}s; 2 passes "
    "identical; exit 0; settle re-executed with quoting fix, disclosed).",
]
(D / "ADDENDUM32.md").write_text("\n".join(L) + "\n")
print("ADDENDUM32.md written,", len(L), "lines")
