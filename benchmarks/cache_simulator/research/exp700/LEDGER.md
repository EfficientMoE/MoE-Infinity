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

- STATUS: DONE (exit 0)
- **Result:**
```
VERDICT: STRUCTURE_UNEXPLAINED (R2=0.5804 < 0.8 bar) — with MONOTONE_BOUNDARY PASS and a clean RIDGE structure | det True | REPL == exp694 (d4096 all caps all arms) and == exp699 (d1024/d256 cap4096) exact | consistency v2 PASS all legs | elapsed 2234s | freeze d9f52af | MATRIX (max-min burst pp): cap1024 col SATURATES ~8pp by d1024 (3.12/5.86/8.11/8.30/8.86); cap2048 col RISES to RIDGE MAX 11.91 @d4096 (1.56/2.54/5.76/7.76/11.91); cap4096 col LOW (0.00/0.20/0.49/0.88/2.73) | beta logd=+1.42 logcap=-2.88 (net -1.46 under proportional scaling) | READING: divergence surface is NOT a log-log plane — it has a RIDGE along d~2xcap (max 11.9pp = the ssr-vs-ring trade band) collapsing to convergence at small d AND washed out by saturation at large cap; cap4096's rising tail (2.73) = the beyond-pole effect (eti_2020 vs ssr) | 0.5pp contour monotone in d all caps | registered bar honest-failed: plane fit cannot capture the ridge | exp701: pairwise decomposition of THIS matrix (JSON has per-arm cells — pure analysis) + off-grid ridge probes (d3072/d5120 x cap1536/2560/2048) to pin ridge location
```

- FREEZE: d9f52af (prereg + smoke)

- SEAL NOTE: ARTIFACT.sha256 written by follow-up controller 2026-11-04 post-launch (run started 14:32, runner mtime 14:31, unchanged; sha verified == frozen commit d9f52af research/exp700/run_phaseboundary.py)
