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

- STATUS: NEXT
