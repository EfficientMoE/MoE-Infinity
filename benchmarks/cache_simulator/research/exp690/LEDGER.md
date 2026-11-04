# exp690 — handoff timing: 2-way lcnu+ring & double-window confirmation

- REGISTERED: 2026-11-04 (before run; freeze commit recorded below)
- Predecessors: exp686a (ringfit3 IN_BURST_KEPT_VISIBLE), exp687
  (windowed HARM/THRASH 8-14 switches), exp689 (VIS_ONLY; center_zb
  refuted the barrier hypothesis; ring wins rec at ALL cap512 cells,
  pays burst; convergent frontier).
- Question: with s3fifo removed from the menu (T1 twoway_lr) or
  single-window adoption blocked (T2/T3 double-window confirmation),
  does ANY arm hold center's in-burst fix AND close the recovery gap?

## Design
- Streams: exp681 exact (seed 68120261103, spawn(2), N=638,976,
  BURSTS=[256,1024,2048,4096] × CAPS=[512,2048]); run_all body
  byte-identical to exp689's (verified).
- Arms: lcnu | center (costfit) | ring10 | fit3_10 (exp686a RingFit3
  ring_frac=0.10) | twoway_lr (sealed 2-way CostFit with s3fifo slot
  -> RingProbationLCNU(0.10); machinery unedited) | twoway_cf (T1 +
  confirmation: challenger must clear the sealed barrier in TWO
  consecutive re-arbitration windows; non-clearing window resets) |
  fit3_cf (RingFit3(0.10) + same confirmation on the sealed 3-way
  challenger scan).
- Sealed machinery unedited: CostFitPolicy (HALF_LIFE 32768, MARGIN
  0.005, GAMMA 0.0025, RESEGMENT 4096, initial lcnu) and CostFit3Policy.

## Bars (pre-committed; exit codes)
- REPL1 center == exp681 per-cell EXACT            (exit 4)
- REPL3 ring10 == exp685 per-cell EXACT            (exit 6)
- REPL4 fit3_10 == exp686a per-cell EXACT          (exit 7)
- RECOV  T-arm rec_def <= 3.5pp @ d4096/cap512
- BURST  T-arm burst_def <= center's deficit ALL 8 cells
- STRETCH T-arm burst_def <= 1.5pp @ home
- SAFETY T-arm loop >= lcnu - 0.5pp ALL cells
- acc credited==loop for center every cell; center double-sim identical (exit 3)
- det byte-identical double grid                   (exit 2)
- Verdicts: BOTH_PASS > REC_ONLY > BURST_ONLY > NO-SIGNAL > HARM.
- Smoke (pre-freeze): unit confirmation semantics (first clear no
  switch, second consecutive switches; non-clearing resets; 3-way
  picks best challenger after 2 clears) PASS; full-N home-cell probe
  center+ring10 == exp689 settled values EXACT.

## Next-step ladder (registered)
- BOTH_PASS -> exp691 dose-response on the winning confirm window
  count (2 vs 3 vs 4) + ring_frac sweep on the 2-way.
- REC_ONLY -> exp691 burst-annex: keep lcnu incumbent through burst
  windows by making the challenger scan burst-blind (segment-aware
  arbitration), else acceptance that burst/rec want split capacity.
- BURST_ONLY/NO-SIGNAL -> exp691 split-capacity single policy (lcnu
  main + ring annex carved from the same cap) — abandon arbitration
  for coexistence.
- HARM -> revert to exp689 verdict set; arbitration abandoned at
  cap512.

## Run block (filled by adjudicator; sealed after settle)
- STATUS: NEXT

- FREEZE: fd8b92d (pushed)
- REPAIR (pre-run, disclosed): replication gate reference arm names fixed — exp681 JSON keys the arm "costfit" and exp686a keys it "ringfit3_10"; grid/adjudicator/bars untouched. First launch crashed at the gate (KeyError, exit 1, no verdict); relaunched on fixed code. Freeze for executed code: this commit.
