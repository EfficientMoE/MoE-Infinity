#!/usr/bin/env python3
"""exp686a settle replacement — quoting-fixed re-execution of the LEDGER
block's python step (disclosed: the fenced block's display-escaped \"
form is a SyntaxError when executed verbatim under `sh -c`; measurement
completed BEFORE the settle step, JSON untouched, so the Result block is
reconstructed from exp686a-ringfit3.json rather than the lost $OUT)."""

import json
import pathlib

D = pathlib.Path("/Users/mfethe/warp-work/exp686a-moei-ringfit3")
p = D / "LEDGER.md"
text = p.read_text()
target = (
    "- STATUS: NEXT  # (marker INSIDE the fenced example; "
    "tail marker below is authoritative)"
)
assert target in text, "no STATUS: NEXT marker"

R = json.loads((D / "exp686a-ringfit3.json").read_text())
assert R["det"] and R["verdict"] == "IN_BURST_KEPT_VISIBLE"
lines = [
    "VERDICT: IN_BURST_KEPT_VISIBLE | det: True | repl: exp681 costfit + "
    "exp685 ring10/ring25 exact | bars: h1 False, h2/h3/h4/acc True | "
    "elapsed 1471.5s | freeze 2735a8e",
    "(reconstructed from exp686a-ringfit3.json; original stdout capture "
    "lost to the settle-script quoting bug AFTER the run completed — "
    "measurement untouched; settle re-executed with the fix, disclosed)",
]
for r in R["rows"]:
    a10 = r["adopt_ringfit3_10"]
    lines.append(
        f"cell d{r['d']}/c{r['cap']}: cf bdef {r['burst_def_cf_pp']:+.2f} "
        f"rdef {r['rec_def_cf_pp']:+.2f} | rf3 bdef "
        f"{r['burst_def_ringfit3_pp']:+.2f} rdef "
        f"{r['rec_def_ringfit3_pp']:+.2f} loop_vs_l "
        f"{r['loop_vs_l_ringfit3_pp']:+.2f} | rf3_10 bdef "
        f"{r['burst_def_ringfit3_10_pp']:+.2f} rdef "
        f"{r['rec_def_ringfit3_10_pp']:+.2f} loop_vs_l "
        f"{r['loop_vs_l_ringfit3_10_pp']:+.2f} adopt burst={a10['burst']} "
        f"rec={a10['rec']} sw={a10['sw']} | lazy2 rdef "
        f"{r['rec_def_lazy2_pp']:+.2f} loop_vs_l "
        f"{r['loop_vs_l_lazy2_pp']:+.2f}"
    )
out = (
    "- STATUS: DONE (exit 0; settle re-executed with quoting fix — "
    "disclosed)\n- **Result:**\n```\n" + "\n".join(lines) + "\n```"
)
p.write_text(text.replace(target, out, 1))
print("LEDGER settled:", p, len(out), "bytes")
