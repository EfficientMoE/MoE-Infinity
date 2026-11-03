# Addendum 20 — exp669: Bayesian optimization of CostFit (MARGIN, GAMMA, HALF_LIFE)

**Verdict: IMPROVED** (frozen rule: IMPROVED if H1) — re-eval determinism gate PASS (best config rebuilt post-run on both traces; all recorded cells byte-identical).

## Question

Can Bayesian optimization over CostFit's three knob space beat the hand-calibrated
center (M=0.005, G=0.0025, HL=32768) on the frozen no-harm objective f
(worst per-cap deficit vs best standalone, pp), under the switch constraint (<=6)?

## Setup (prereg c72a9d69, frozen before run)

- 9 seed points + 45 BO trials, GP surrogate (RBF ls=0.25 sig=1.0 noise=0.0001), seed 20261104, 2048 acquisition candidates.
- Space: M[0.001,0.02] G[0.0005,0.01] HL[4096,262144]log. Objective: worst per-cap deficit vs best standalone
  (lcnu/s3fifo) across both real decode traces at the frozen cap grids.

## Results

- Seed best f = 0.072pp -> BO best feasible f = 0.027pp (2.7x better), 41/45 feasible.
- Winner: M=0.00162, G=0.00999, HL=160975 (trial 1); worst deficit at winner 0.027pp, max switches 1.
- H1 IMPROVED: PASS (0.027 < 0.062pp).
- H2 NEW-AREA: PASS — HL=160975 outside center
  2x band [16384, 65536], direction UP (4.9x center);
  22/45 trials sampled outside the band.
- H3 CALIBRATION: FAIL — surrogate leave-ahead Pearson r = 0.122 (< 0.5). The GP landscape did not predict observed f.

## Disclosure

H3 fails: the improvement rests on an early sample (trial 1) that no later acquisition
beat, not on a well-modeled response surface. Pre-freeze smoke (disclosed in prereg):
first HL-normalization draft spanned [1, 1e5] and the acquisition drove HL=278 ->
f=0.711, switches=15 (INFEASIBLE; matches exp666 oscillation at short effective
memory). Mapping fixed to the prereg range before freeze.

## Next

exp674 (prereg frozen): does the winner transfer to exp670's out-of-family synthetic
grid (streams sha-asserted identical), vs center and vs HL-only? Settles whether the
NEW-AREA half-life is mechanism or real-trace overfit.

