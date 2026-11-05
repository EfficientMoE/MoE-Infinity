# Addendum 33 — exp689 ring visibility: lazy/ghost/zero-barrier arms (VIS_ONLY)

exp685's failed gate (composite recovery collapse at cap512) was tested
with three simple levers on exp681's exact streams (8 cells, dual
replication anchors exit 4/6, LEDGER safe-settle v2, freeze c05c4d5 —
renumbered from a draft exp686a after the follow-up chain's exp686a/
exp687 raced it; disclosure in LEDGER).

## Verdict: VIS_ONLY (det True; repl exp681+exp685 exact)

- center_ring10 (ring10 inner, SEALED barrier intact): recovery beats
  lcnu at ALL FOUR cap512 cells (d256:-0.71 d1024:-1.10 d2048:-1.37 d4096:-0.73pp) — the ring IS adopted
  where exp685's ring25 was invisible — but pays in-burst (worst
  burst_def +9.23pp vs center), failing the ladder bar.
  Same frontier endpoint as exp686a ringfit3_10 and exp687
  windowfit3_10: three judges, convergent tradeoff.
- center_zb (MARGIN=GAMMA=0): the barrier hypothesis is REFUTED —
  with no barrier the composite still collapses recovery at the home
  cell (rec_def +6.98pp == center's) because the decayed
  scoreboard itself mistimes the handoff (exp687's conclusion,
  now confirmed without confounds).
- center_lazy (k=2): dead dose at composite level — never adopted
  (byte-equal to center at cap512), safety violated. Matches exp686a's
  standalone lazy2 negative.
- center_ghost: visible + recov bars pass but recovery worse than
  plain ring10 (+0.90 vs -0.73pp at home) — ghost admission is dominated.

## Mechanistic summary (3 experiments, convergent)

At d4096/cap512 the burst window wants the LCNU inner (burst 0.0884)
and the recovery window wants the RING (rec 0.6853); s3fifo is never
best in either critical window — it is what every cumulative-rate
scoreboard settles on between them. Un-tested next lever (exp690,
registered): a 2-way lcnu+ring composite (no s3fifo to hide on) plus
double-window confirmation hysteresis on the 3-way to block single-
window adoption (exp687's thrash showed switches fire on one window
of evidence; two consecutive wins would block mid-burst ring adoption
while allowing the post-burst handoff).

Data: exp689-ringvis.json (elapsed 943.0s; 2 passes
identical; exit 0).
