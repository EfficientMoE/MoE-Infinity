# Addendum 24 — exp678 cost-model fragility (COST-H1: proxy breaks under bursty reload)

All prior admitted results rank policies by hit rate — every miss priced
equal. exp678 scored the same arms under a time metric with variable
miss cost: 1 + kappa*u_e (per-expert) + phi*(misses in trailing 64)/64
(queueing burst). 24 cells: skew {0.9,1.3}, caps {512,2048}, kappa
{0,5,20}, phi {0,4}; stationary streams, seeds frozen in PREREG.

Verdict: **COST-H1** — H1 (time) violated in 1/24 cells;
hit-rate control H2 clean everywhere; accounting and determinism hold;
0 metric inversions.

## The violating cell

skew 1.3, cap 2048, **kappa=0, phi=4**: time deficit 1.057% vs
the best standalone (lcnu) while the hit-rate deficit is only 0.079pp.
costfit takes 502 MORE misses than lcnu (55251 vs
54749) at near-equal hit rate — but those extra misses CLUSTER, and
under the burst penalty each miss inside a cluster costs more (it scales
with trailing-64 miss density). Simulated time 0.1229
vs 0.1216.

## Finding

- The hit-rate proxy is blind to the TEMPORAL CLUSTERING of misses. Two
  policies within 0.1pp of hit rate can differ >1% in simulated reload
  time purely from burst structure. The proxy's blind spot is real and
  reproducible (deterministic double-run; accounting 24/24).
- Per-expert cost spread alone (kappa>0, phi=0) did NOT break the bar:
  clustering, not expert heterogeneity, is the load-bearing failure mode.
- Deficits grow with phi and with skew; cap 512 stays inside bars
  everywhere — the failure needs the tighter cap.

## Consequence

Any deployment where expert reloads contend (queueing) can mis-rank
policies by hit rate alone. Follow-up queued: anti-clustering candidate
(spread evictions / burst-aware margin) and the (phi, cap) boundary map.

Cells 24 | inversions 0 | det
True | worst 1.057% (sk 1.3, cap
2048, k 0.0, phi 4.0)

Generated programmatically from exp678-costmodel.json (gen_addendum24.py).
