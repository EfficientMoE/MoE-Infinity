# ADDENDUM 50 — exp706: qwen30b-lo CROSS1 location (capacity-grid extension past 2048)

Verdict: **LOCROSS_MAP** — L1 PASS (CROSS1 = 2816 slots), L2 PASS
(CROSS1/WS20 = 0.604 inside frozen [0.50, 0.66]), L3 REPLICATION-ANCHOR
PASS exact (spread and lcnu whole byte-equal to sealed exp704 at both overlap
cells), L4 det PASS. Continuity report-only: ring strictly below lcnu burst
8/9 caps with 1 saturation tie
(3328, 0.0pp — ring never wins, consistent
with exp704 B3 semantics). Spread decays 4.211pp -> 0.055pp
across the grid. Exit 0. 1036s.

## The law is now 3-for-3 (mechanism consolidated)
- CROSS1/WS20: granite 0.625 / hi 0.528 / lo 0.604
  — all three traces inside the frozen transfer band. The exp705 honest
  negative resolves as the rule, not the exception: best-vs-worst policy
  convergence needs only ~53-63% of the
  final-20% working set resident, across both model families and the most
  diverse-footprint trace (lo: 5842 distinct keys).
- Deployable sizing rule: cap ~ 0.6 x WS20 experts (~5.27 GiB
  per 1000 WS20 keys for the 30B family, 2.64 GiB for 3B-class).
- lo CROSS1 = 2816 slots x 9 MiB = **24.75 GiB** — completes exp705's C1
  hardware mapping 3/3 (was a >= 18.0 GiB lower bound; now located at
  24.75 GiB, >= 15.0 as pre-registered for the 30B class).

## Measured grid (lo trace, frozen arms lcnu/ring10/ssr_10/eti_2020, W=4096)
| cap slots | cache GiB | lcnu whole | ring10 whole | spread pp |
|---|---|---|---|---|
| 1792 | 15.75 | 0.785 | 0.723 | 4.2 |
| 2048 | 18.00 | 0.823 | 0.796 | 2.1 |
| 2176 | 19.12 | 0.839 | 0.821 | 1.7 |
| 2304 | 20.25 | 0.853 | 0.840 | 1.6 |
| 2432 | 21.38 | 0.867 | 0.856 | 1.3 |
| 2560 | 22.50 | 0.879 | 0.870 | 1.1 |
| 2816 | 24.75 | 0.901 | 0.896 | 0.7 |
| 3072 | 27.00 | 0.919 | 0.917 | 0.2 |
| 3328 | 29.25 | 0.934 | 0.933 | 0.1 |

Policy premium at the law point: spread 0.717pp vs 4.211pp at grid start —
below CROSS1 the policy choice still moves burst hit rates by up to
4.2pp on this extension; at 3072-3328 the arms are
statistically indistinguishable (0.156/0.055pp).

## Anchors & determinism
- L3: recomputed spread@1792 == 4.211 == sealed
  exp704 exactly; same at 2048; lcnu whole@{1792,2048} byte-equal
  (0.7852596645607701 / 0.8228962971319695).
  Cross-process determinism of this simulator stack holds a third time.
- L4: policy_cell(lcnu, 2304) recomputed inside the run == grid cell exact.
- WS20(lo) = 4662 parsed at runtime from sealed exp705 (byte constants C4-verified there).
- Integrity: runner sha256 f5d0cf97dc9a02d0...
  matches ARTIFACT.sha256; freeze commit a75c7de preceded the run artifacts.

## Tier & limitations
- Measured tier (real-trace simulation on the frozen exp656 lo trace), but a
  single-trace extension of sealed exp704 methodology: no new policy classes,
  no tuning; grid and bars frozen pre-run. CROSS1 grid resolution is 128-256
  slots at the crossing (true crossing in (2560, 2816]); ratio interval
  inherits that resolution (~0.005 wide).
- 3328 = 0.713 x WS20 stays below WS20: the grid brackets the law band but
  does not reach the working set itself; extrapolation beyond 0.713 is not
  measured here.
- lo law value 0.604 is the family's SECOND trace — per-family variance
  (hi 0.528 vs lo 0.604) is real and the rule is a band, not a constant.

Data: exp706/exp706-locross.json; PREREG.md/ARTIFACT.sha256 in exp706/;
sealed refs exp704-capacitymap.json, exp705-costcurves.json.
Follow-up controller re-adjudicated every gate programmatically from primary
sources: exact match, verdict LOCROSS_MAP stands.
