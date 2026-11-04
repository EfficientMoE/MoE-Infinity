# exp682 PREREG — anti-clustering candidate: doorkeeper admission (frozen pre-launch)

## Question

exp678 (COST-H1) established: the center policy's extra misses vs lcnu
CLUSTER in time under bursty reload cost; heterogeneity (kappa) is not the
failure mode. exp680 ruled out arbitration cadence. Does a DOORKEEPER
admission filter — first-sight missing keys are NOT admitted (evict
nothing), second sight admits — remove the clustered extra misses while
holding hit rate? This is the first IMPROVEMENT-shaped candidate of the
series (prior addenda were characterization).

## Design (frozen)

- Testbed: exp678's frozen stationary streams + time-cost metric
  (1 + kappa*u_e + phi*trailing-64 miss rate), seeds identical
  (default_rng(67820261103).spawn per skew; u from SEED+1000+i).
- Grid: skew {0.9, 1.3} x cap {512, 2048} x kappa {0, 20} x phi {0, 4}
  = 16 cells (subset of exp678's 24; kappa endpoints only).
- Arms per cell: lcnu, costfit@frozen-center (replication gate vs exp678
  published per-cell time/hit, exact), door512, door4096 (center wrapped
  with doorkeeper K in {512, 4096}, LRU door), lcnu+door4096
  (generalization probe, lcnu exposes .resident).
- Wrapper only; sealed policy files untouched. Resident-probe fast path;
  door membership second-sight admission; LRU-bounded door.

## Bars (frozen; verdict from pass-1 grid)

- Replication: center time/hit equal exp678's published values in all
  overlapping cells (streams identical by construction).
- H1 efficacy: door time beats center by >= 0.1% absolute in >= half of
  phi>0 cells (per door arm; door4096 is primary).
- H2 safety: door hit >= center hit - 0.5pp in every cell.
- H3: no cell where door time exceeds center by > 0.1%.
- acc: center double-sim identical (misses == misses2). det: pass-2 grid
  byte-identical.
- Verdict ladder: DOOR-WINS > DOOR-NULL > DOOR-HARM.

## Pre-freeze smoke (disclosed)

- Mock-policy hand stream: door semantics exact (first sight misses without
  admission, second sight admits, resident hits bypass); LRU bound verified
  behaviorally (D,E,F with K=2: F evicts D, D re-enters as first sight).
  One smoke-check bug (closure introspection on the mock) fixed in the
  CHECK, not the runner.
- Reduced stream (122,880 acc): center double-sim identical; door arm
  deterministic; lcnu has .resident (lcnu_door arm valid).
- Honest smoke signal: at tiny N, door512 hit trails center by ~0.5pp —
  the known doorkeeper tradeoff (one-touch reuse pays a double miss
  before working-set protection accrues). Full-length streams decide;
  DOOR-HARM is an explicit possible outcome.

## Next (pre-registered branches)

- DOOR-WINS -> dose-response K sweep + mechanism check (misses-per-window
  variance drop), then propose as upstream candidate in the capture report.
- DOOR-NULL/HARM -> protection does not pay at stationary skew; test door
  under exp681's flash-crowd streams (where one-touch crowd admissions are
  the explicit poison) — the crowd testbed is the door's home field.
