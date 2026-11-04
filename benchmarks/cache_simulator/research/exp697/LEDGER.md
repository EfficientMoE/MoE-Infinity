# exp697 — Earn-then-immune (ETI): the frontier's interior point

Pre-registered before run. Freeze hash after commit.

## Chain
exp696 pinned: frontier = revisit-miss routing rule; poles ssr
(protect at re-sight) vs ring (earn by surviving churn to a re-hit).
ETI = interior: re-entry revisits join the ring and must earn by
re-hitting once; the re-hit grants scorer-immune annex membership.
ETILCNU: miss -> admit + ring (every miss, ring semantics); hit ->
ring.pop + annex grant (annex overflow drains oldest; annex members
are ring-drain-proof: victim drains oldest NON-annex ring member);
fallback batch-exclude annex from scorer.

## Pre-registered churn-arithmetic expectation
Burst floods ~2600 distinct keys into a cap/10 ring; drain spacing
evicts the median carry-in before its in-window re-access (ring burst
0.0488 == this arithmetic). ETI's earn event (a HIT) can rarely fire
for carry-ins -> expect COLLAPSES_TO_RING on burst, rec keeps ring's
gain. If burst recovers >= 1pp, the churn model is wrong (that would
be the finding).

## Bars (d4096, caps 512-8192; PP)
- HARM gate (all caps, else HARM): eti rec >= lcnu - 0.10 AND eti
  burst >= lcnu - 0.50.
- DOMINATES_RING: cap in {1024,2048}: rec >= ring +0.10 AND burst >=
  ring +0.10.
- MIDDLE_POINT: cap in {1024,2048}: rec >= ring -0.10 AND burst >=
  ring +1.00.
- COLLAPSES_TO_RING: |burst - ring| < 1.00 at BOTH 1024 AND 2048 AND
  rec >= ring -0.10 at both.
- else MIXED. MECH (report-only): class-c hits vs ring @1024/2048.

## Gates
REPL lcnu/ring10/ssr_10 == exp694 exactly (both depths all caps;
exit 4). CONSIST class counts equal (exit 3). det eti cap2048 d4096
double-pass (exit 2).

## Smoke (mini pre-freeze) + liveness probe
mini N=8192 d1024 cap512: eti == ring to 4dp (burst .0537 rec .6536;
c=20 b=35 identical) — the predicted collapse visible at mini; ssr
distinct. det micro True. Synthetic divergence probe: annex populates
(3 members) and state diverges from ring-only — mechanism live, not
dead code.

- STATUS: DONE (exit 0; verdict HARM by gate artifact, disclosed)
- **Result:**
```
VERDICT: HARM (REGISTRATION ARTIFACT, disclosed) | det True | repl lcnu/ring10/ssr_10 == exp694 exact both depths | consistency PASS | elapsed 759s+det | freeze c8bb520 | HARM gate anchored burst to lcnu (+/-0.50pp) — ring10 itself fails that clause by construction; gate mis-calibrated for rec-trading family (registration error, self-caught at adjudication; substantive result below) | COLLAPSES_TO_RING PREDICTION REFUTED: interior point exists | d4096: cap1024 eti 0.0623/0.7830 vs ring 0.0488/0.7898 (burst +1.35pp, rec -0.68pp, rec +1.66pp vs lcnu); cap2048 0.1216/0.8765 vs ring 0.0957/0.8816 (+2.59pp/-0.51pp, +0.93 vs lcnu); cap512 +0.73/-0.71; cap4096 +0.37/+0.02 (tiny rec WIN too) | EXCHANGE RATE: ~0.27-0.30pp rec per pp burst — 5-10x FLATTER than the lcnu->ring trade (~3.8pp burst per pp rec) | MECHANISM EXACT: burst recovery carried entirely by class-c carry-in hits (eti 254 vs ring 199 @1024 = +55/4086 = +1.35pp exact; 497 vs 391 @2048 = +106/4086 = +2.59pp exact; b-class identical 1/4) — earn event fires: carry-ins surviving ring drain to a re-hit get annex protection, converting their LATER in-window re-accesses to hits
```

- FREEZE: c8bb520 (prereg + smoke + liveness)