# Addendum 30 — exp684 ring-less probation on crowd streams (PROB-HARM)

exp683's decomposition (k>=2 door admission burns 1/k of the burst
reuse window; the recovery win was one-timers never displacing the
hot set) implies two levers: admit fast + demote later. exp684 is the
unbounded endpoint of that lever: ProbationLCNU admits at first sight
and evicts the oldest never-rehit resident first (no partition, no
ring). Sealed LayerCycleNextUsePolicy subclass; exp681's exact crowd
streams; arms lcnu | costfit-center | prob_lcnu | center_prob (prob
as the composite's lcnu inner); 8 cells; LEDGER.md pilot freeze
fe1d97f (prereg + gates + branches in one commit).

## Verdict: PROB-HARM (det True, replication exact vs exp681+exp683)

- H1 (no in-burst loss): FALSE — burst deficit vs lcnu in ALL 8
  cells, range -8.72pp (d4096/cap2048) to
  +4.30pp.
- Loop cost: -2.25pp (cap2048 cells ~ -2.2pp).
- H3 (recovery): 4/8 — small wins at cap512 (+0.6..+1.2pp),
  LOSSES at cap2048; the composite pays: center_prob rec
  -5.37pp vs center at d4096/cap2048.

## Mechanism

Unbounded probation has no capacity to hold the burst reuse stride:
under a d-key crowd burst every admission evicts the previous
one-timer before its rehit can occur, so crowd keys churn among
themselves and the burst hit rate collapses — the same failure the
door caused, moved inside the cache. The recovery protection the
door bought (one-timers never entering) is only partially retained
(4/8) and the composite's other inners pay for the swap at cap2048.

## Consequence (pre-registered branch fired)

PROB-HARM => probation needs bounded, partitioned capacity: a
nursery ring that drains only when full, with the protected region
scoring-managed (Michael's 2026-11-03 design). exp685 = that ring
(ring_frac 0.10/0.25), frozen 63d4e09 with dual-Pareto bars.

Data: exp684-prob.json (elapsed 683.9s, 2 passes
byte-identical; center/door replication cells exact).
