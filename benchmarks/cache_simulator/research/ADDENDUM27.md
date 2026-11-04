# Addendum 27 — exp681 sub-block flash-crowd probe (CROWD-H1)

exp677 localized harm to arbitration-interval alignment; exp680 ruled
out cadence. exp681 tested the remaining axis: a cold-crowd burst (layer
uniform, expert uniform over ranks [64,128)) replacing d accesses at
midpoint of a stationary stream (skew 1.1, lam 0.3), d in {256, 1024,
2048, 4096}, caps {512, 2048}, 8 cells; d=4096 is the aligned control
the barrier CAN see.

Verdict: **CROWD-H1** — H1 violated in 8/8 cells
(burst deficit vs best standalone, bar 1.0pp), H2 in 2,
H3 clean (sw<=1), accounting exact, determinism byte-identical.

## Findings

- EVERY cell violates the burst bar; range 1.07-10.86pp. Worst: d=4096,
cap=2048 at **10.86pp** (costfit burst
hit 0.188 vs best standalone +10.86pp).
- Costfit's burst-window hit collapses (0.016-0.188 across cells):
  repeating crowd keys are admitted too slowly to serve the burst,
  while the best standalone adapts within the burst.
- CAP-DEPENDENT MODE SPLIT: at cap 2048 harm concentrates IN the burst
(worst 10.86pp); at cap 512 the burst deficit
shrinks (2.73-5.37pp) but RECOVERY breaks:
worst recovery 6.98pp at d=4096/cap=512
(crowd admissions evicted the hot set; the post-burst stream pays).
- Blindness is not even required: the ALIGNED CONTROL (d=4096, boundary-
visible, sw=1) still loses 10.86pp at cap 2048 — the arbitration fires
and cannot help, consistent with exp680's cadence-irrelevance. The
barrier's binary inner-policy switch has no crowd response at any
cadence; the failure is in ADMISSION SPEED of repeating cold keys.

## Cell table (deficit vs best standalone, pp)

| d | cap | d_burst | d_rec | sw |
|---:|---:|---:|---:|---:|
| 256 | 512 | 2.734 | 0.000 | 1 |
| 256 | 2048 | 6.641 | 0.000 | 1 |
| 1024 | 512 | 5.371 | 0.000 | 1 |
| 1024 | 2048 | 4.102 | 0.000 | 1 |
| 2048 | 512 | 4.053 | 1.294 | 1 |
| 2048 | 2048 | 4.883 | 0.000 | 1 |
| 4096 | 512 | 1.074 | 6.982 | 1 |
| 4096 | 2048 | 10.864 | 0.000 | 1 |

## Third failure mode; convergence

Series to date: alignment (exp677), temporal miss clustering (exp678),
crowd admission lag (exp681, this). All three point at the same lever:
ADMISSION of cold/repeating keys, not arbitration. The doorkeeper
candidate (exp682, in flight on exp678's testbed) blocks first-sight
admissions but its second-sight admit is exactly the crowd-serving
path — pre-registered pivot: door on THESE crowd streams next.

Cells 8 | h1 8 | h2 2 | h3
0 | acc True | det True | base
skew 1.1 lam 0.3 stationary, B=1024 | crowd uniform ranks [64,128) at midpoint

Generated programmatically from exp681-crowd.json (gen_addendum26.py).
