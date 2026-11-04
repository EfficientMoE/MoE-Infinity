# exp700 — phase-boundary sweep

Pre-registered before run. Freeze hash after commit.

## Chain
exp699: convergence at d256/d1024 vs divergence at d4096; exp695:
burst = 99.9% carry-in re-accesses. Model: policy spread scales with
burst depth d and capacity headroom. Sweep the surface.

## Grid
d in {256, 512, 1024, 2048, 4096} x caps {1024, 2048, 4096} x arms
{lcnu, ring10, ssr_10, eti_2020} — 60 passes, per-d streams.

## Bars (PP)
- PRIMARY: spread matrix (max-min burst, pp) + log-log LSQ fit
  spread ~ a + b1 log d + b2 log cap; R^2 >= 0.8 => MODEL_HOLDS.
- 0.5pp contour monotone in d at fixed cap => MONOTONE_BOUNDARY.
- Verdicts: MODEL_HOLDS+MONOTONE_BOUNDARY > MODEL_HOLDS >
  STRUCTURE_UNEXPLAINED.

## Gates
REPL == exp694 (d4096, caps 1024/2048/4096, all four arms) and ==
exp699 grid B (d1024/d256, cap4096) (exit 4). DET double-pass
eti_2020@d2048/cap2048 (exit 2). CONSISTENCY within-depth class
totals equal across caps + sum == d (gate v2 form) (exit 3).
Exit 0 settled.

## Smoke (mini pre-freeze, N=8192)
d512 arms coincide (0.1719), d2048 slight divergence (0.3203 vs
0.3193) — direction matches model.

- STATUS: NEXT
