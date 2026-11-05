# Addendum 35 — exp691: miss-signal arbitration (REC_ONLY; family closed)

exp690's registered REC_ONLY branch: react to the OBSERVABLE burst
signature (credited miss-rate storm, no labels) instead of rate menus.

## Verdict: REC_ONLY (det True; repl exp681+exp685+exp690 exact)

- The storm detector fires exactly once per stream, and only at/after
  the burst (0.0288 == ring's burst at home).
  At d=256 it never fires: a 256-access burst dissolves inside the
  4096-access arbitration window before any boundary can see it
  (registered risk, confirmed).
- Revert-to-lcnu therefore cannot save the burst, and it destroys
  recovery: after the forced revert the sealed barrier (MARGIN 0.5pp
  + GAMMA x inflated sw_mass ~ 0.75pp total) exceeds ring's calm-window
  advantage (+0.73pp at home), so ring is never re-adopted and rec
  collapses to lcnu's 0.6780.
- Adoption-freeze arms are byte-identical to the twoway control:
  adoption happens BEFORE the burst, so freezing during storms is a
  no-op.
- All five T-arms: recov + safety PASS, burst_def FAIL
  (5.96pp home, worst 9.23pp).

## Family closure (exp686a, exp687, exp689, exp690, exp691)

Five experiments exhaust the arbitration family at this workload:
menus (3-way, 2-way), judges (decayed cumulative, windowed,
zero-barrier), confirmation hysteresis, and now the deployable
miss-signal. Every variant lands REC_ONLY or worse because the burst
is SHORTER than the smallest natural arbitration window: no rate
signal can both detect and act inside it. Burst-vs-recovery at cap512
is a capacity frontier, not a timing problem. Next (exp692,
registered): SecondSightRing coexistence — an annex fed only by
REVISIT keys (first-sight crowd never enters it, so bursts cannot
pollute it; the pre-burst revisit stream is exactly the recovery set)
at frac sweep 5/10/15 — the first design that needs no timing at all.

Data: exp691-storm.json (elapsed 2051.9s; exit 0).
