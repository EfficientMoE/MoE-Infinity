# exp693 — SecondSightRing: revisit-only annex coexistence (no arbiter)

- REGISTERED: 2026-11-04 (before run; freeze below)
- Predecessors: exp686a/687/689/690/691 — arbitration family CLOSED:
  burst is shorter than any natural arbitration window (no rate signal
  can detect AND act inside it); burst-vs-recovery at cap512 is a
  capacity frontier. exp691's registered fallback: coexistence.
- NOTE: originally drafted as exp692; renamed — warp-work/exp692 is
  owned by the prompt-heterogeneity lane (checked before freezing).

## Design
- Mechanism: annex membership granted ONLY on revisit admission
  (ever-admitted set). First-sight keys (burst crowd) are never
  protected — they compete by LCNU score and evict each other, like
  plain lcnu. Revisits (recovery working set) are annex-protected:
  _victim fences annex members out of base scoring (deterministic;
  _by_layer + stale-mark invalidation) and only drains the oldest
  annex member when every resident is protected. Annex FIFO-bounded
  (annex_frac x capacity); annex hit promotes out (frees protection).
- Arms: lcnu | center (REPL1 exp681 exit 4) | ring10 (REPL3 exp685
  exit 6) | ssr_05 | ssr_10 | ssr_15 (annex_frac 5/10/15%).
- Streams: exp681 exact (seed 68120261103, spawn(2), N=638,976,
  BURSTS x CAPS); run_all identical to exp690/691.
- Smoke (pre-freeze): 6 units PASS — first-sight-unprotected,
  revisit-annexed, annex-FIFO-cap, hit-promotes, annex-spared-under-
  stress (30 forced victims, zero annex hits), structural integrity
  (by_layer == resident, 5000 random accesses). Mini N=61,440:
  ssr burst 0.1240 == lcnu 0.1240 (vs ring 0.0291) with annex 26-77
  members; determinism double-run PASS.
- Smoke fixes disclosed: (1) _victim fence loop initially returned an
  annex member when len(fenced) >= len(annex) — guard removed (fencing
  IS respected by _layer_candidate since it reads _by_layer); (2) unit
  3's precondition corrected (LCNU scoring protects residents through
  moderate floods — confirmed evicted keys needed for FIFO test).

## REPAIR (disclosed, pre-publication; run killed before first d-cell completed)
- The frozen sequential fence re-asked base _victim() once per annex
  member; annex members systematically hold top scores, so every miss
  cost O(annex) full candidate recomputes (~25x slowdown; 15 min
  without finishing d=256). Replaced by batch exclusion: remove all
  annex members from _by_layer, one base pick, restore. Proven
  equivalent on 6 workloads x 20k accesses (identical hit sequences,
  resident sets, annex states) + smoke units + mini reproduce
  (burst 0.1243 / rec 0.6187 identical pre/post). Bars untouched.

## Bars (pre-committed; exit codes)
- REPL1 center == exp681 per-cell EXACT            (exit 4)
- REPL3 ring10 == exp685 per-cell EXACT            (exit 6)
- RECOV  rec_def <= 3.5pp @ d4096/cap512 (registered composite bar:
  do not trail lcnu's recovery by more than center-class damage)
- BURST  burst_def <= center's deficit ALL 8 cells
- STRETCH burst_def <= 1.5pp @ home
- SAFETY loop >= lcnu - 0.5pp ALL cells
- acc center loop==loop2 every cell (exit 3); det double grid (exit 2)
- Verdicts: BOTH_PASS > REC_ONLY > BURST_ONLY > NO-SIGNAL > HARM.
- Scientific read beyond the verdict: the annex's value shows in
  rec_def approaching ring10's -0.73pp; rec_def ~= 0 with BOTH_PASS
  means "no harm, no recovery gain" — a null for the capture
  hypothesis, reported as such.

## Next-step ladder (registered)
- BOTH_PASS + rec_def < -0.3pp (annex captures ring's edge) -> exp694:
  frac fine sweep (7.5/12.5/20) + annex-priority variant (recent
  revisits preferred over FIFO).
- BOTH_PASS with rec_def ~= 0 (null capture) -> capacity frontier
  CONFIRMED end-to-end: publish closure; exp694 moves to a different
  axis (layer-aware annex placement or capacity scaling beyond cap512).
- BURST_ONLY (annex hurts burst) -> exp694: shrink frac / expose
  instead of protect on burst admission pressure.
- HARM -> annex family dead; exp694: different axis entirely.

## Run block (filled by adjudicator; sealed after settle)
- STATUS: DONE (exit 0)
- **Result:**
```
VERDICT: BOTH_PASS | winners: ssr_05+ssr_10+ssr_15 (ALL bars green) | det: True | repl: exp681+exp685 exact | elapsed 3780s | freeze 9c4d4dc + repair 9fbf133 (batch annex exclusion, equivalence-proven) | FIRST BOTH_PASS in the composite program (686a/687/689/690/691 were REC_ONLY or worse) | burst_def = 0.000 EXACTLY at every cap512 cell (annex never touches burst: ssr burst 0.0884 == lcnu) | cap2048: burst_def NEGATIVE up to -0.78pp (ssr BEATS lcnu — fencing revisit keys purges victim competition) | rec_def ~0 (+/-0.05pp): null capture — no recovery gain | REGISTERED INTERPRETATION: capacity frontier CONFIRMED end-to-end; closure published; exp694 -> different axis (capacity scaling / layer-aware annex)
```

- FREEZE: 9c4d4dc (pushed)
- REPAIR FREEZE: 9fbf133 (pushed; rebuilt clone after parent-repo deletion)
