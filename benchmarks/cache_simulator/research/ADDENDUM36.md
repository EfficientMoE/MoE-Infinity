# Addendum 36 — exp693: SecondSightRing revisit-only annex (BOTH_PASS)

exp691's registered fallback after the arbitration family closed:
coexistence — protect the recovery set without timing anything.

## Verdict: BOTH_PASS (det True; repl exp681+exp685 exact)

All three annex fractions (5/10/15%) pass every bar: burst, recovery,
stretch, safety — the first BOTH_PASS in the composite program
(exp686a/687/689/690/691 were REC_ONLY or worse).

- Burst is byte-preserved: burst_def = 0.000 exactly at every cap512
  cell (ssr_10 burst 0.0884 == lcnu's 0.0884). First-sight crowd never
  enters the annex, so the LCNU inner serves bursts untouched — the
  property every arbiter lost.
- cap2048 cells: burst_def goes NEGATIVE (ssr beats plain lcnu by up
  to 0.78pp at d256) — fencing revisit keys out of victim scoring
  purges the ranking; annex exclusion is a small win on its own.
- Recovery unchanged (rec_def within +/-0.05pp everywhere): the annex
  neither captures ring's +0.73pp recovery edge nor loses ground.
  Pre-registered null-capture reading: the burst-vs-recovery trade at
  cap512 is a CAPACITY FRONTIER, now confirmed end-to-end — timing,
  judging, hysteresis, and now selective sparing all fail to move
  recovery without paying burst. The frontier itself is the finding.
- Repair disclosed: sequential fence -> batch annex exclusion
  (equivalence proven on 6 workloads; 25x faster).

## Program state after closure

exp694 (registered): move off the arbitration axis — capacity
scaling (1024/4096/8192) to map where the frontier sits, and/or
layer-aware annex placement.

Data: exp693-ssr.json (elapsed 3780.4s; exit 0).
cap2048 ssr_10 burst_def: d256:-0.78pp / d1024:-0.20pp /
d2048:-0.10pp / d4096:-0.05pp.
