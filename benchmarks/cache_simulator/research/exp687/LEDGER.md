# exp687 LEDGER — windowed challenger scoreboard (Tier1 lane 7687a6bf1295)

## PREREG (frozen before full-grid measurement)

- Q: Can a WINDOWED challenger scoreboard (incumbent judged on sealed
  decayed cumulative rate; each challenger on its last-4096 window rate;
  sealed barrier family otherwise: HALF_LIFE 32768, MARGIN 0.005,
  GAMMA 0.0025, RESEGMENT 4096, initial lcnu) repair the burst/recovery
  handoff that exp686a's decayed-cumulative scoreboard could not
  ("cannot hand off between burst and recovery windows")?
- Registered branch: exp686a LEDGER conclusion, registered next step
  (windowed challenger scoreboard). Sealed params inherited UNEDITED
  from policies_costfit3.CostFit3Policy; victim selection + inners
  identical to exp686a RingFit3 ({lcnu, s3fifo, ring}).
- Streams: exp681 EXACT (spawn 68120261103), N=638976, d∈{256,1024,
  2048,4096}, caps {512,2048}, 8 cells, double grid for determinism.
- Arms (8): lcnu, s3fifo, center (REPL1 vs exp681 costfit), ring10,
  ring25 (REPL2 vs exp685), ringfit3_10 (REPL3 vs exp686a, proves
  sealed base untouched), windowfit3_10 (PRIMARY), windowfit3_25 (dose).

## BARS (pre-committed; no post-hoc changes)

- REPL1 center == exp681 costfit per-cell EXACT (loop/burst/rec) → exit 4
- REPL2 ring10+ring25 == exp685 per-cell EXACT → exit 5
- REPL3 ringfit3_10 == exp686a per-cell EXACT incl. switches + choice
  timeline (JSON-normalized) → exit 6
- UNIT window-vs-decayed decision bites ×3 on hand-set state → exit 7
- H1 windowfit3_10 rec_def ≤ 2.5pp @ d=4096/cap=512 (PRIMARY)
- H2 windowfit3_10 burst_def ≤ center burst_def, ALL 8 cells
- H3 windowfit3_10/25 loop ≥ lcnu loop − 0.5pp, ALL cells
- H4 windowfit3_10/25 switches ≤ 6 ALL cells; credited==loop every
  composite; center double-sim identical → exit 3 on acc fail
- H0 structural sub-adjudication (measured, not gated): at home cell,
  incumbent-at-mid, switch indices in [mid−4096, mid+8192], first ring
  adoption ≥ mid, recovery-end relation.
- det: byte-identical double grid → exit 2 if not.

## VERDICTS (pre-committed order)

WINDOWED-HANDOFF (H1&H2&H3&H4) > RECOVERY-ONLY (H1 only) >
STRUCTURAL-SPLIT (H1 fails at home cell AND H0 boundary mechanism
confirmed: incumbent lcnu/s3fifo held through burst AND no ring
adoption inside [mid, mid+d+4096)) > NO-GAIN (H1 fails, mechanism not
alignment) > HARM/THRASH (H3 or H4 fails).

## PRE-REGISTERED PREDICTION (frozen)

STRUCTURAL analysis: mid=319488=78·4096 is window-ALIGNED; at d=4096
the burst fills exactly one resegment window and the recovery the next;
the first PURE-recovery window completes only AT the recovery's final
boundary, so adoption can benefit only post-recovery accesses;
mid-burst escape is impossible (challenger burst-window rates
0.029–0.088 vs incumbent decayed ~0.6+barrier). Therefore at the home
cell NO windowed arbitration with these alignments passes H1 AND H2
simultaneously — the run adjudicates WHICH side fails.
SMOKE-UPDATED expectation (see smoke disclosure): single-window
challenger noise (σ ≈ 0.0073 at p≈0.68, n=4096) exceeds the sealed
MARGIN 0.005 → ring↔s3fifo oscillation on stationary stream; H4 at
risk (smoke already shows 8–9 switches on d256 cells); pre-burst ring
adoption makes H2 at risk as well. Most likely verdict: HARM/THRASH or
STRUCTURAL-SPLIT. Genuine uncertainty: whether late-stream long ring
tenures (smoke: ring held 77823→413695 at d256/c512) dominate loop
enough to keep H3, and whether any cell realizes mid-recovery adoption.

## SMOKE (pre-freeze instrument validation, disclosed)

d=256 cell only (non-primary for H1; H2/H3/H4 surface partially
visible — disclosed in full):
- unit bites 3/3 (window drives challenge; decayed challenger lead
  ignored; sub-barrier gap no-switch).
- anchors: REPL1/REPL2/REPL3 all EXACT on both caps (center==exp681
  costfit, ring10/25==exp685, ringfit3_10==exp686a incl. normalized tl).
- windowfit3_10 d256/cap512: loop=0.683051 burst=0.015625
  rec=0.674072 sw=8 tl=[(0,lcnu),(4095,ring),(8191,s3fifo),
  (12287,ring),(16383,s3fifo),(32767,ring),(57343,s3fifo),
  (77823,ring),(413695,s3fifo)]
- windowfit3_10 d256/cap2048: loop=0.871394 burst=0.054688
  rec=0.870850 sw=9 tl=[(0,lcnu),(8191,ring),(12287,s3fifo),
  (16383,ring),(20479,s3fifo),(24575,ring),(28671,s3fifo),
  (32767,ring),(57343,s3fifo),(126975,ring)]
- Smoke exit 0. One instrument fix before smoke re-pass (tl tuple vs
  JSON-list normalization in repl3 compare) — disclosed.

## FREEZE

- Runner: run_windowed.py (this directory; sha256 below).
- Freeze commit: <FILLED AT COMMIT> (path-scoped under
  benchmarks/cache_simulator/research/exp687/ on exp651/qwen30b-results).

## RUN

- STATUS: NEXT (full grid + double-run determinism; ~30 min).

## SETTLE

- <FILLED AFTER RUN>
