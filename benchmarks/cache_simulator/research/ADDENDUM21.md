# Addendum 21 — exp674: does the exp669 BO winner transfer out-of-family?

**Verdict: SAFETY-BROKEN** — the BO point (HL=160975, 4.9x center) degrades the mean hit rate vs center (-0.1234pp) and violates the no-harm barrier on the synthetic grid. The center calibration stands; the BO objective overfit the worst case to two real traces.

## Question (prereg 71776837, frozen before run)

exp669 found IMPROVED at NEW-AREA (f=0.027pp on the real traces). Transfer test: evaluate
cf@BO-winner vs cf@center vs cf@HL-only on exp670's 11-stream synthetic grid
(streams sha-asserted byte-identical to exp670), caps {128..4096}, N=638,208/stream.

## Results

- 59 informative cells (|lcnu - s3fifo| >= 0.1pp); double-run determinism PASS; accounting PASS (loop==prop every arm every cell).
- H1 SAFETY: FAIL — 8/59 cells exceed 0.5pp deficit; worst 0.779pp (s=1.4, lam=0.6, E=32@cap256); max switches 2 (harm is genuine, not oscillation).
- H2 TRANSFER: FAIL — mean hit-rate delta vs center -0.1234pp (WORSE on average; hit rates, so negative = below center, beyond the 0.05pp tie-tolerance — H2 fails independently of the safety gate); BO point wins only 11/66 cells.
- HL-only arm: mean -0.0613pp — roughly half of the mean DEGRADATION comes from the half-life alone; per-cell safety of the HL-only point was not a frozen gate.
- Violation structure: 5/8 safety violations at s=1.4 (highest skew):
  the long-memory BO region over-commits on skewed streams and pays at small-mid caps.

## Reading

Third independent confirmation that the barrier catches what mean-based selection misses
(exp666 oscillation, exp673 toxic candidate, exp674 BO point). The BO point's 11 wins are marginal
(mean +0.0375pp, max +0.134pp, mostly at cap 4096) while its 20 losses
average -0.3848pp — the mean degradation is broad, not offset by large wins —
and it is not certifiable under no-harm as a point selection. Future BO must optimize the worst case over a grid that includes
out-of-family streams (joint real+synthetic objective) — queued as the follow-up direction.

## Correction (same day, 2026-11-03)

Fixes an unrendered `{beats}/{tot}` placeholder and three sign/magnitude misstatements introduced by the original generator: the mean delta fields are HIT-rate deltas (negative = worse than center), so the BO point is worse on average (-0.1234pp), the HL-only arm's -0.0613pp is degradation, and the BO point's 11 wins are marginal (mean +0.0375pp, max +0.134pp) against 20 losses averaging -0.3848pp. All numbers recomputed from exp674-hl-transfer.json by gen_addendum21_fix.py. Verdict unchanged: SAFETY-BROKEN (H1: 8/59 cells >0.5pp, worst 0.779pp, max switches 2).
