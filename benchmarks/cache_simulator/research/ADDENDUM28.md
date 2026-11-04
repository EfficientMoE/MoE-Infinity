# Addendum 28 — exp682 doorkeeper admission on stationary streams (DOOR-HARM)

First IMPROVEMENT-shaped candidate of the series: a doorkeeper
admission filter (first-sight missing keys are not admitted — they
evict nothing; second sight admits via the base policy; K-LRU door)
wrapped around the frozen center, tested on exp678's exact cost-model
testbed (16 cells, replication gate). Mechanism rationale: block the
one-touch cold admissions whose extra misses cluster in time (exp678).

Verdict: **DOOR-HARM** — the door HURTS on stationary streams.
Replication exact (24 cells vs exp678
published — also proving the concurrent lane's additive policies.py
change altered no arm semantics). Determinism byte-identical.

## Numbers (vs center arm)

- H2 violations: 16/16 door-cell combos lose > 0.5pp hit.
- H3 violations: 21 combos run > 0.1% SLOWER than center.
- H1: only 4/16 phi>0 combos improve time.
- Worst hit loss: -2.54pp (skew 0.9,
cap 2048, door512). Best time gain: only
+2.58% (skew 0.9, cap 512, door512).
- Generalization probe: lcnu+door beats bare lcnu in 4/16
cells — the harm is policy-independent.

## Interpretation

On stationary streams the door's protection value is zero (no crowd to
block) while its double-miss tax on one-touch reuse is constant: net
harm everywhere. This is the correct null-space result for a mechanism
whose benefit is conditional on burst structure — and it sharpens the
prediction for the crowd testbed (exp683): there, one-touch crowd keys
are the PROVEN poison (exp681: 10.86pp in-burst, 6.98pp recovery), so
the door's tax should be outweighed by hot-set preservation. If the
door still harms under crowds, admission filtering is dead as a lever
for this policy family and the fix must live inside inner admission.

## Pre-registered branches honored

DOOR-NULL/HARM -> door under exp681 flash-crowd streams (home field).
Launched as exp683 immediately after this publication.

Cells 16 | arms center/door512/door4096/lcnu/lcnu_door | det
True | repl True | elapsed 1637.9s

Generated programmatically from exp682-door.json (gen_addendum28.py).
