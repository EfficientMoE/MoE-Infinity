# Addendum 23 — exp677 W-sweep at the drift edge (W-SWEEP-SAFE)

exp676's worst transition deficit (0.781pp) sat at the slowest tested
transition. exp677 swept transition width the other way — faster:
W in {1024, 2048, 4096, 8192} (down to one arbitration window),
locality-drop regime (1.1, 0.6) -> (1.1, 0.0), B=1024, N=638,976,
caps {512, 2048, 4096}, same frozen bars (trans <=1.0pp, post <=0.5pp,
<=3 switches/4096-window). 12 cells; determinism byte-identical.

Verdict: **W-SWEEP-SAFE** — 12/12 cells inside bars; 0 violations.

## Findings

- Transition deficits are zero in 11/12 cells. The only
  nonzero cell is W=4096, cap=4096: 0.439pp (0% of the 1.0pp bar).
- Pre-registered monotonicity REFUTED (mono=False): shrinking W does not
  increase harm. The exposure peak is at W=4096 = the policy's RESEGMENT
  interval — a transition lasting exactly one arbitration window forces
  the switch decision mid-transition on stale decayed mass. Faster
  blockwise transitions (W < RESEGMENT) are fully absorbed.
- At cap 4096 the barrier never fires (0 switches at any W): the
  0.439pp cell is lcnu's own transient vs s3fifo, and the barrier
  correctly judged switching not worth it — still inside bars.
- Caps 512/2048: exactly 1 switch per 4096-window at every W.

## Cell table (d_trans = best-standalone minus costfit, pp)

| W | cap | d_trans | d_post | max_sw/4096win |
|---:|---:|---:|---:|---:|
| 1024 | 2048 | 0.000 | 0.000 | 1 |
| 1024 | 4096 | 0.000 | 0.000 | 0 |
| 1024 | 512 | 0.000 | 0.000 | 1 |
| 2048 | 2048 | 0.000 | 0.000 | 1 |
| 2048 | 4096 | 0.000 | 0.000 | 0 |
| 2048 | 512 | 0.000 | 0.000 | 1 |
| 4096 | 2048 | 0.000 | 0.000 | 1 |
| 4096 | 4096 | 0.439 | 0.000 | 0 |
| 4096 | 512 | 0.000 | 0.000 | 1 |
| 8192 | 2048 | 0.000 | 0.000 | 1 |
| 8192 | 4096 | 0.000 | 0.000 | 0 |
| 8192 | 512 | 0.000 | 0.000 | 1 |

## Interpretation

Blockwise drift is now bounded on both sides of transition speed:
slow (exp676: 0.781pp worst) and fast (exp677: 0.439pp worst). The
remaining untested axis is sub-block (intra-window) regime change,
where the arbitration cannot react between resegments — queued as the
next drift probe if the cost-model axis (exp678, in flight) returns
COST-ROBUST.

Cells: 12 | H1 True | H2 True | H3 True | determinism True | mono False

Generated programmatically from exp677-wsweep.json (gen_addendum23.py).
