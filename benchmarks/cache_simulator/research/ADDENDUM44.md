# Addendum 44 — exp700: phase-boundary sweep (STRUCTURE_UNEXPLAINED; ridge found)

## Verdict: STRUCTURE_UNEXPLAINED — registered plane-fit bar failed honestly

R^2 = 0.5804 < 0.80 on the pre-registered log-log plane
spread ~ a + b1 log d + b2 log cap. MONOTONE_BOUNDARY passed (the
0.5pp contour is monotone in d at every fixed cap).

## The spread matrix (max-min burst, pp)

```
d\cap    1024    2048    4096
  256    3.12    1.56    0.00
  512    5.86    2.54    0.20
 1024    8.11    5.76    0.49
 2048    8.30    7.76    0.88
 4096    8.86   11.91    2.73
```

- cap1024 column saturates (~8pp) once d >= cap.
- cap2048 column rises to the RIDGE MAXIMUM 11.91pp at d4096
  (burst = 2x capacity) — the ssr-vs-ring trade band of exp694.
- cap4096 column is low; its rising tail (2.73pp at d4096) is the
  exp698/699 beyond-pole effect (eti_2020 vs ssr_10).
- beta(log d)=+1.42, beta(log cap)=-2.88: proportional scaling
  (d and cap together) shrinks spread.

## Reading

The divergence surface is not a monotone hill: it is a RIDGE along
d ~ 2 x cap, collapsing to convergence at small d (carry-in crowd
too small to matter — exp695/699) and washed out toward saturation
at large cap. The plane-fit bar could not capture this by
construction; the honest verdict records that the registered model
class was wrong, not that there is no structure.

## Next

exp701: pairwise decomposition of the exp700 matrix (per-arm cells
are in the JSON — pure analysis, no rerun) to identify which arm
pair carries the ridge; plus off-grid ridge probes (d in {3072,
5120} x cap {1536, 2048, 2560}) to pin its location.

Data: exp700-phaseboundary.json (exit 0; det True; anchors ==
exp694 and exp699 exactly; freeze d9f52af; 2234s).
