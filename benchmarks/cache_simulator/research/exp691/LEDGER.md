# exp691 — miss-signal arbitration: storm-revert + adoption freeze

- REGISTERED: 2026-11-04 (before run; freeze below)
- Predecessors: exp686a (IN_BURST_KEPT_VISIBLE), exp687 (HARM/THRASH),
  exp689 (VIS_ONLY; barrier refuted), exp690 (REC_ONLY; all menus park
  on ring; burst edge belongs to the LCNU inner itself).
- Question: can the arbiter react to the OBSERVABLE burst signature —
  a miss-rate storm on the composite's own credited stream (no segment
  labels) — to hold lcnu through bursts and re-adopt ring for recovery?

## Design
- Streams: exp681 exact (seed 68120261103, spawn(2), N=638,976,
  BURSTS=[256,1024,2048,4096] x CAPS=[512,2048]); run_all identical to
  exp690's.
- Arms: lcnu | center (REPL1 exp681 exit 4) | ring10 (REPL3 exp685
  exit 6) | twoway_lr (CONTROL == exp690 exit 8) | freeze_50 freeze_65
  (adoption-block-only: storms freeze arbitration, incumbent keeps) |
  storm_50 storm_65 storm_80 (storm-revert: windowed miss > TH forces
  choice to lcnu; calm windows resume sealed barrier rules; forced
  reverts increment switches but NOT _sw_mass — disclosed semantic).
- Storm signal: 1 - windowed credited hit rate at the sealed RESEGMENT
  boundary (4096 accesses), computed from cumulative credited deltas.
- Smoke (pre-freeze): 4 mechanism units PASS; mini N=61,440 shows
  storm_50 catching the burst (burst 0.0796 vs twoway 0.0291; storm_w=2)
  while storm_80/freeze_65 fire nothing at mini scale (full-N burst
  miss ~0.97 will trigger); twoway_lr mini matches exp690 mini.

## Bars (pre-committed; exit codes)
- REPL1 center == exp681 per-cell EXACT            (exit 4)
- REPL3 ring10 == exp685 per-cell EXACT            (exit 6)
- REPLC twoway_lr == exp690 per-cell EXACT         (exit 8)
- RECOV  T-arm rec_def <= 3.5pp @ d4096/cap512
- BURST  T-arm burst_def <= center's deficit ALL 8 cells
- STRETCH T-arm burst_def <= 1.5pp @ home
- SAFETY T-arm loop >= lcnu - 0.5pp ALL cells
- acc center loop==loop2 every cell (exit 3); det double grid (exit 2)
- Verdicts: BOTH_PASS > REC_ONLY > BURST_ONLY > NO-SIGNAL > HARM.
- Registered risk: d=256 bursts (256 accesses < one 4096 window) may
  defeat detection -> BURST bar can fail at small d even when large-d
  cells pass; read per-cell before declaring the family dead.

## Next-step ladder (registered)
- BOTH_PASS -> exp692: TH dose-response fine sweep + window-size
  sweep (1024/2048/8192) on the winning threshold; then formalize
  storm-revert as the composite's default.
- REC_ONLY -> storms fired but adoption still mistimed: exp692 =
  per-window CHOICE memory (ring re-adoption only after 2 calm
  windows) or abandon arbitration for split-capacity coexistence.
- BURST_ONLY -> detection latency too coarse: exp692 = shorter
  detection window (512/1024) at the winning TH.
- NO-SIGNAL/HARM -> miss-signal family closed; exp692 = split-
  capacity coexistence policy (lcnu main + ring annex, no arbiter).

## Run block (filled by adjudicator; sealed after settle)
- STATUS: DONE (exit 0)
- **Result:**
```
VERDICT: REC_ONLY | det: True | repl: exp681+exp685+exp690 exact (incl twoway_lr control) | storm detector fires exactly ONCE per stream and only AT/AFTER the burst (storm_w=1 at d>=2048; 0 at d=256: bursts shorter than the 4096-access window are invisible — registered risk confirmed) | revert-to-lcnu too late for burst (0.0288 == ring) AND sealed barrier (~0.75pp after switch inflation) blocks ring re-adoption in recovery (+0.73pp advantage < barrier) -> rec collapses to lcnu 0.6780 | freeze arms == twoway_lr byte-identical (adoption happens pre-burst; freezing during storms is a no-op) | all 5 T-arms recov+safety PASS, burst_def FAIL (5.96 home / 9.23 worst) | elapsed 2052s | freeze a1df70b
```

- FREEZE: a1df70b (pushed)