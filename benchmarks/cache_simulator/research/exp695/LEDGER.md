# exp695 — Flood decomposition: why lcnu's burst hit grows with capacity

Pre-registered before run. Freeze hash recorded after commit.

## Motivation (registered chain)
exp694 (DUAL_STABLE) mapped the frontier: lcnu burst hit 0.0884@cap512
-> 0.4099@cap4096 -> 0.9985@cap8192 (d4096); ring10's sacrifice
non-monotone (5.96/8.86/9.23/0.88pp). exp695 decomposes WHY by
classifying every burst-window access by key history — classes are a
property of the stream, not the policy.

## Classes (run id = contiguous seg==3 run)
- (a) first-sight: key never seen before this access
- (b) same-burst: key first seen within the SAME burst run
- (c) carry-in: key first seen earlier (earlier burst run or calm)

## Hypotheses (pre-registered)
- H1: lcnu's growing burst hit with capacity is driven by class (b):
  (b_hits@cap4096 - b_hits@cap512) / (tot@4096 - tot@512) >= 0.80
  (d4096).
- H2: ring's mechanism is carry-in protection: ring class-(c) hit >
  lcnu class-(c) hit at caps 1024 AND 2048 (where rec edge >= 1pp);
  AND ring class-(b) hit < lcnu class-(b) (sacrifice located in flood
  re-accesses).
Verdicts: MECH_CONFIRMED (H1+H2) | PARTIAL (one) | REFUTED.

## Protocol
exp694 machinery (streams, mk_arm) unchanged; instrumented pass.
Arms lcnu|ring10; caps 512/1024/2048/4096/8192; depths 4096 (main),
1024 (control); N=638,976.
Gates: CONSIST class counts identical across arms (exit 3);
REPL burst == exp694-capmap.json exactly at d4096 all caps (exit 4);
det rerun lcnu/cap2048/d4096 identical accumulators (exit 2).

## Smoke (mini, pre-freeze)
N=8192 d=1024 cap512: classes policy-independent (a=794 b=135 c=95);
ring b-hits 35 vs lcnu 103 (sacrifice in b); ring c-hits 20 vs 10
(carry-in protection visible at mini). det micro True.

- STATUS: DONE (exit 0)
- **Result:**
```
VERDICT: REFUTED (H1 False: b_share 0.0008; H2 False: ring c-hit < lcnu c-hit @1024/2048 — direction inverted) | det True | repl exp694 exact (burst d4096 all caps) | consistency PASS (class counts policy-independent) | elapsed 141s | freeze ad3b362 | TRUE DECOMPOSITION (d4096): burst window = 4096 accesses: a=6 first-sights, b=4 same-burst, c=4086 CARRY-INS (99.9%) — bursts RE-ACCESS the pre-seeded crowd region, not novel keys | lcnu c-hits by cap: 359/559/766/1675/4086 (= its entire burst hit; growth = more crowd keys still resident when burst lands) | ring c-hits: 118/199/391/1641/4086 (probation holds HOME keys -> crowd carry-ins miss; converges at saturation) | ring b-hits 1-2/4 vs lcnu 4/4 (sacrifice located in carry-ins too) | MECHANISM REWRITTEN: burst-vs-recovery = slot competition between two re-access populations (crowd carry-ins vs home revisits); the actionable decision point is the CALM SEEDING PHASE (crowd admission), not burst-time arbitration — new axis for exp696: admission-time population throttling
```

- FREEZE: ad3b362 (prereg + smoke)
