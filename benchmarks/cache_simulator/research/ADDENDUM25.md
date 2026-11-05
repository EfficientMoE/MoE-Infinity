# Addendum 25 — exp680 staggered arbitration at the alignment peak (STAG-SAFE-NO-GAIN)

exp677 localized the only harmful transition to W=4096 = the RESEGMENT
arbitration interval (0.439pp, cap 4096). exp680 tested the direct
intervention: double the arbitration cadence (RESEGMENT 4096 -> 2048,
instance-attr override; sealed center file untouched, all center params
identical) on EXACTLY exp677's streams (same seed scheme, sliced to W
{2048, 4096, 8192}; caps {2048, 4096}; 6 cells).

Verdict: **STAG-SAFE-NO-GAIN** — safety holds (H1-H3, accounting 6/6),
determinism byte-identical, replication exact (center reproduced
exp677's published 0.439pp peak deficit within
0.005) — but H4 fails: staggering changed NOTHING.

## Findings

- Staggered and center deficits are IDENTICAL in 6/6 cells,
  including the peak cell — switch counts and hit trajectories match.
- WHY (mechanistic): at the peak cell (W=4096, cap=4096) the barrier
  NEVER fires (0 switches) — the 0.439pp deficit is lcnu's own transient
  vs s3fifo with the barrier correctly declining to switch. Arbitration
  cadence cannot matter in a cell where no arbitration outcome changes
  the trajectory. At cap 2048 the doubled cadence never flipped a
  decision either (1 switch per window at both cadences).
- Conclusion: ARBITRATION CADENCE IS NOT THE LEVER. The alignment
  exposure of exp677 is not removable by arbitrating more often; the
  harm flows through the inner policy's own transition response, not
  through switch timing.

## Determinism incident (disclosed, gate upheld)

First run returned det=False. Deep-diff showed ALL arms (incl. pure
standalones) differing between passes from the first cell: the runner
hoisted spawned seed generators to module level, so pass 1 advanced
state and pass 2 drew different seeds. All reported numbers came from
pass 1 (= exp677-exact seeds, hence replication passed). Fixed (fresh
spawn inside run_all), fully re-run: det=True with pass-1 numbers
reproduced. No publication occurred on det=False.

## Cell table (d_trans = best-standalone minus arm, pp)

| W | cap | d_center | d_stag | sw_c | sw_s |
|---:|---:|---:|---:|---:|---:|
| 2048 | 2048 | 0.000 | 0.000 | 1 | 1 |
| 2048 | 4096 | 0.000 | 0.000 | 0 | 0 |
| 4096 | 2048 | 0.000 | 0.000 | 1 | 1 |
| 4096 | 4096 | 0.439 | 0.439 | 0 | 0 |
| 8192 | 2048 | 0.000 | 0.000 | 1 | 1 |
| 8192 | 4096 | 0.000 | 0.000 | 0 | 0 |

## Next (pre-registered pivot)

Cadence ruled out -> sub-block flash-crowd probe (intra-window regime
change the arbitration cannot see), then the anti-clustering candidate
answering exp678's COST-H1 finding.

Cells 6 | h4 False | replication
True | det True | center params
[0.005, 0.0025, 32768] | staggered RESEGMENT 2048

Generated programmatically from exp680-stagger.json (gen_addendum25.py).
