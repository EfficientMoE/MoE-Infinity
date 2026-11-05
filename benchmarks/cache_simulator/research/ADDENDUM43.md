# Addendum 43 — exp699: beyond-pole existence test (NEITHER; regime-specific family)

## Verdict: NEITHER (run 2; run 1 aborted on an impossible gate)

Run 1 exit 3: the registered consistency gate demanded identical
class-count signatures ACROSS burst depths — impossible, since class
totals scale with d. The run-1 signatures (6+4+4086=4096,
1+0+255=256, 2+0+1022=1024) prove the underlying data was correct.
The gate aborted before adjudication; no results were observed;
data discarded without selection. Gate v2 (freeze c0c7dfe) checks
the valid invariant: within-depth totals equal across caps and
summing to d. Bars unchanged.

## Beyond-pole: real but narrow

- Grid A (cap4096, d4096): SIX eti variants beat the ssr pole on
  burst at rec >= ssr - 0.03pp — eti_2010 0.4187 (+0.59pp) through
  eti_2020/2030/2040 0.4285 (+1.57pp); optimum eti_4020 0.4292.
  Annex plateaus from 20%; ring fraction beyond 20% is inert.
- Grid B: FAILS — no eti variant beats ssr at cap2048 (any depth),
  and at d256/d1024 all policies converge on burst
  (0.3750 uniform at d256).

## Flat exchange does not generalize

eti_0510 == ring10 EXACTLY at cap4096 at every depth
(0.3750/0.9531 d256; 0.3945/0.9487 d1024; 0.4011/0.9539 d4096).
The flat interior exchange of exp697/698 exists only in the d4096
trade-band (caps 1024-2048).

## Model

Burst depth and capacity JOINTLY determine which re-access
population dominates arbitration. The class-c carry-in mechanism
(99.9% of burst traffic, exp695) needs deep bursts and sufficient
capacity headroom to express; below that, LCNU-family policies are
indistinguishable. The earn-then-immune family's advantages are
regime-specific:
- interior flat exchange: d4096, caps 1024-2048
- beyond-pole: d4096, cap4096 (annex >= 20%)
- elsewhere: convergence.

Cross-experiment determinism: eti_2020@cap4096 == exp698 exactly.

## Next

exp700 registered: phase-boundary sweep — burst depth d across
{256, 512, 1024, 2048, 4096} at caps {1024, 2048, 4096} for
{lcnu, ring10, ssr_10, eti_2020}: locate the d x cap surface where
policies diverge (spread > 0.5pp) vs converge, testing the
carry-in-scaling model quantitatively.

Data: exp699-beyondpole.json (run 2; exit 0; det True; anchors ==
exp694; freeze run1 839c7e2 / gate-v2 c0c7dfe; 2230s).
