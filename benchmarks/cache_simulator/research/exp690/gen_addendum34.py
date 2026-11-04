#!/usr/bin/env python3
"""gen_addendum34.py — programmatic addendum for exp690 (REC_ONLY)."""
import json
from pathlib import Path

r = json.loads((Path(__file__).parent / "exp690-confirm.json").read_text())
assert r["verdict"] == "REC_ONLY", r["verdict"]
assert r["det"] is True
assert r["repl"] == "exp681+exp685+exp686a exact"
assert r["rec_only"] == ["twoway_lr", "twoway_cf", "fit3_cf"]

home = [x for x in r["rows"] if x["d"] == 4096 and x["cap"] == 512][0]
worst = max(x["twoway_lr_burst_def_pp"] for x in r["rows"])
sws = {a: sorted({x[a]["sw"] for x in r["rows"]}) for a in
       ("twoway_lr", "twoway_cf", "fit3_cf")}

md = f"""# Addendum 34 — exp690: 2-way lcnu+ring and double-window confirmation (REC_ONLY)

Four experiments (exp686a ringfit3, exp687 windowed, exp689 visibility
arms, this run) now triangulate the same frontier at cap512: the burst
window wants the LCNU inner, the recovery window wants the RING, and
every rate-based arbiter so far parks on one of them permanently.

## Design

- T1 twoway_lr: sealed 2-way CostFit with the s3fifo SLOT replaced by
  RingProbationLCNU(0.10) — no s3fifo to hide on.
- T2 twoway_cf: T1 + double-window confirmation (challenger must clear
  the sealed barrier two consecutive windows; exp687's anti-thrash
  prescription).
- T3 fit3_cf: exp686a's RingFit3(0.10) + the same confirmation.
- Triple replication gates: center==exp681, ring10==exp685,
  fit3_10==exp686a, per-cell exact (all passed; det grid identical;
  repair freeze c50a310 fixed only reference arm-name mapping).

## Verdict: REC_ONLY (det True; triple repl exact)

- ALL THREE T-arms converge to the standalone ring10 endpoint at the
  home cell: burst {home['ring10']['burst']:.4f}, rec {home['ring10']['rec']:.4f},
  loop {home['twoway_lr']['loop']:.4f}. Removing s3fifo does NOT keep
  lcnu through burst — the scoreboard switches to ring once (switch
  counts {sws}) and never returns: ring's decayed cumulative rate
  dominates after the loop phase, so the burst ALWAYS inherits the
  ring (burst_def {home['twoway_lr_burst_def_pp']:.2f}pp at home, worst
  {worst:.2f}pp across cells).
- Double-window confirmation is a no-op here: there is nothing to
  confirm (single early switch, zero thrash — exp687's prescription
  solves a problem this menu doesn't have).
- Recovery bar passes everywhere (home rec_def
  {home['twoway_lr_rec_def_pp']:+.2f}pp) and safety holds for all
  T-arms.

## Conclusion (registered REC_ONLY branch)

Arbitration across rate-scored menus cannot time the burst/recovery
handoff: any menu containing ring adopts it before the burst and pays
in-burst; any menu excluding ring collapses recovery. The burst edge
belongs to the LCNU inner itself ({home['lcnu']['burst']:.4f} vs ring
{home['ring10']['burst']:.4f} at home). Next (exp691, registered):
miss-signal adoption freeze — block challenger adoption while the
composite's miss rate over the arbitration window exceeds a threshold
(deployable: no segment labels, reacts to the observable burst
signature). If lcnu is held through the burst and ring is adopted only
after the miss rate settles, BOTH bars can pass for the first time.

Data: exp690-confirm.json (elapsed {r['elapsed_s']}s; exit 0).
"""
out = Path(__file__).parent / "ADDENDUM34.md"
assert not out.exists(), "ADDENDUM34 collision"
out.write_text(md)
print("wrote", out, len(md), "chars")
