# Addendum 21 — exp674: does the exp669 BO winner transfer out-of-family?

**Verdict: SAFETY-BROKEN** — the BO point (HL=160975, 4.9x center) improves the mean deficit but violates the no-harm barrier on the synthetic grid. The center calibration stands; the BO objective overfit the worst case to two real traces.

## Question (prereg 71776837, frozen before run)

exp669 found IMPROVED at NEW-AREA (f=0.027pp on the real traces). Transfer test: evaluate
cf@BO-winner vs cf@center vs cf@HL-only on exp670's 11-stream synthetic grid
(streams sha-asserted byte-identical to exp670), caps {128..4096}, N=638,208/stream.

## Results

- 59 informative cells (|lcnu - s3fifo| >= 0.1pp); double-run determinism PASS; accounting PASS (loop==prop every arm every cell).
- H1 SAFETY: FAIL — 8/59 cells exceed 0.5pp deficit; worst 0.779pp (s=1.4, lam=0.6, E=32@cap256); max switches 2 (harm is genuine, not oscillation).
- H2 TRANSFER: FAIL — mean deficit delta vs center -0.1234pp (better on average) but safety gate dominates; BO point wins only {beats}/{tot} cells.
- HL-only arm: mean -0.0613pp — roughly half the mean gain carries in the half-life alone; per-cell safety of the HL-only point was not a frozen gate.
- Violation structure: 5/8 safety violations at s=1.4 (highest skew):
  the long-memory BO region over-commits on skewed streams and pays at small-mid caps.

## Reading

Third independent confirmation that the barrier catches what mean-based selection misses
(exp666 oscillation, exp673 toxic candidate, exp674 BO point). The NEW-AREA region is real
and large (HL=160975 wins big on 11 high-skew cells) but is not certifiable under no-harm
as a point selection. Future BO must optimize the worst case over a grid that includes
out-of-family streams (joint real+synthetic objective) — queued as the follow-up direction.

