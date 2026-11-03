# PREREG — exp680: staggered arbitration at the alignment peak

Date: 2026-11-03. exp680 free. Builds on exp677 (#202): the only nonzero
transition deficit (0.439pp) sits at W=4096 = RESEGMENT — mechanism found,
now the intervention.

## Question

Does doubling arbitration cadence (RESEGMENT 4096 -> 2048, all else identical
to the certified center) reduce the alignment-peak transition deficit without
breaking post-stability or the switch budget?

## Design (frozen)

- Streams: EXACTLY exp677's (same generator, same seed scheme
  default_rng(67720261103).spawn(4), sliced to W {2048, 4096, 8192}); B=1024,
  N=638,976; locality-drop (1.1,0.6)->(1.1,0.0). Caps {2048, 4096}. 6 cells.
- Arms: lcnu, s3fifo, costfit@center (M=0.005,G=0.0025,HL=32768),
  costfit-staggered = center params + instance-attr RESEGMENT=2048 (sealed
  file untouched; override disclosed).
- Center arm MUST reproduce exp677's published numbers (peak cell deficit
  0.439pp within 0.005) — cross-experiment replication gate.

## Hypotheses (frozen bars)

- H1 staggered trans deficit vs best standalone <= 1.0pp (all cells).
- H2 staggered post deficit <= 0.5pp.
- H3 staggered switches <= 3 per 4096-window.
- H4 (headline): at peak cell (W=4096, cap=4096) staggered trans deficit
  < center trans deficit.
- Accounting: loop == effective_hit_rate for BOTH costfit arms, every cell.
- Determinism: double grid run identical.
- Verdict: IMPROVED iff safety(H1-H3,acc) + H4 + replication;
  STAG-SAFE-NO-GAIN iff safety + replication, no H4;
  REPLICATION-FAIL iff center deviates from exp677; STAG-BROKEN otherwise.

## Pre-freeze smoke (disclosed)

- Reduced stream (159,744 acc): segment mix exact (trans == 4096 accesses at
  W=4096); staggered arm carries RESEGMENT=2048 instance attr; accounting
  loop==prop both arms (0.9305); switch windows computed via rd.window_switches.
- One assert fixed pre-smoke (2*W -> W transition-access count; matches
  exp677's smoke note "trans == 2xB for W=2048" i.e. count == W).
- Seed derivation verified identical to exp677 run_all (int(spawn[i]
  .integers(0,2**63))). Disk 10 GiB; writes <5 MB; exception scoped as
  exp676/677/678.

## Post-run amendment (2026-11-03, pre-publication): det=False root cause

First full run completed with verdict STAG-SAFE-NO-GAIN, replication True,
accounting 6/6, but determinism False. Deep-diff diagnosis (diag_det.py,
background worker): ALL arms including pure standalones differed between the
two grid passes from the FIRST cell — the streams themselves differed. Root
cause: this runner hoisted `SEEDS = default_rng(...).spawn(4)[1:4]` to module
level, so pass 1 advanced the seed generators and pass 2 drew different seeds
(exp677 spawns fresh inside run_all; that difference was the bug). All
reported numbers come from pass 1, whose seeds were the first draws ==
exp677's exact streams (hence replication passed). Fix: spawn fresh inside
run_all (SEED_BASE constant); bars, arms, streams, hypotheses unchanged.
Full rerun under the fixed runner; publication only if det=True and pass-1
numbers reproduce. Determinism gate upheld: no publication on det=False.

## Follow-up (registered)

IMPROVED -> stagger becomes candidate for center re-optimization lane
(follow-up: cadence sweep {1024, 3072, 6144} + joint (M,G,HL) refit under
drift). NO-GAIN -> cadence is not the lever; pivot to sub-block flash-crowd
probe. BROKEN -> stagger rejected, alignment risk stands; design cooldown
barrier instead.
