# exp706 PREREG — qwen30b-lo CROSS1 location (capacity-grid extension past 2048)

Program STACK-1001, lane 79990. Measured tier (real-trace simulation, frozen
trace; extends sealed exp704's lo grid). No new capture; arms and simulator
identical to exp704's sealed runner.

## Motivation
exp704 (ledger #227) found CROSS1 (first capacity where best-vs-worst burst
spread < 1pp) = granite 768 / qwen-hi 1792 / qwen-lo NOT_REACHED by 2048
(spread 2.116pp @2048). exp705 (ledger #228) priced the curves and found C2
FAIL as an honest negative: convergence arrives at only ~53-63% of the
final-20% working set (CROSS1/WS20 = 0.625 granite / 0.528 hi; WS20 lo = 4662).
next_hint: "extend lo cap grid beyond 2048 to locate qwen30b-lo CROSS1".
Law prediction: CROSS1_lo in [0.53, 0.63] x 4662 = [2471, 2937] slots.

## Frozen inputs (read-only; anchors parsed at runtime AND pinned)
- Trace: exp656 stream_qwen30b_lo.pt (N=638208 decode accesses, distinct=5842,
  prompts=[512,512,126,512], n_win=155, burst_n=16) — identical to exp704 lo.
- Sealed exp704 lo artifact (~/warp-work/exp704-moei-capacitymap/
  exp704-capacitymap.json): spread_pp@1792 = 4.211, spread_pp@2048 = 2.116;
  lcnu whole@1792 = 0.7852596645607701, @2048 = 0.8228962971319695.
  Runner parses these at runtime from the sealed JSON and asserts the pins.
- Sealed exp705 WS20(lo) = 4662 (research/exp705/exp705-costcurves.json,
  parsed at runtime, pinned).
- Arms identical to exp704: lcnu, ring10 (RingProbationLCNU 0.10),
  ssr_10 (SecondSightLCNU 0.10), eti_2020 (ETILCNU 0.20/0.20); W=4096;
  burst = top-16 novelty windows (frozen novelty computation as exp704).
- Grid (frozen): caps [1792, 2048, 2176, 2304, 2432, 2560, 2816, 3072, 3328];
  1792/2048 are overlap-replication cells; 3328 = 0.713 x WS20 brackets the
  law's upper prediction with margin.

## Pre-registered gates (frozen before any real run)
- L3 REPLICATION-ANCHOR (hard): recomputed spread@1792 == 4.211 and
  spread@2048 == 2.116 EXACTLY (3-dp rounding, same as sealed), and lcnu
  whole@{1792,2048} == sealed values EXACTLY (cross-process determinism is
  established in this stack: exp703/exp705 anchors were byte-equal). Any
  mismatch -> exit 5, numbers not publishable as comparable.
- L1 LOCATE: CROSS1_lo = first cap in ascending grid order with
  spread_pp < 1.0 (exp704 semantics). If none -> NOT_REACHED(<=3328).
  Consistency sub-check: recomputed spread@2048 >= 1.0 must hold (sealed
  2.116), else the extension contradicts the sealed artifact.
- L2 LAW-TRANSFER: if CROSS1_lo found: 0.50 <= CROSS1_lo / 4662 <= 0.66
  (53-63% working-set law with ~2.5pp margin each side). If NOT_REACHED:
  L2 = REFUTED-FOR-LO (3328/4662 = 0.713 > 0.66 already falsifies the
  interval) unless a cap in [2471,2937] was measured and stayed >= 1.0.
- L4 DET: policy_cell(lcnu, 2304) recomputed -> exact equality with grid cell.
- Continuity (report-only): ring10 burst < lcnu burst at every cap
  (exp704 lo B3 held 9/9); spread(3328) < spread(1792) decay.
- Structure gate (exit 3): N, distinct, prompts, n_win as frozen above.
- Verdict: LOCROSS_MAP (per-gate PASS/FAIL recorded honestly).
  Exit codes: 0 settled | 2 det | 3 structure | 5 anchor.

## Scope honesty
Single-trace extension of a sealed methodology; no new policy classes, no
tuning, grid and bars frozen before the run. The lo trace is the most
diverse-footprint of the three (5842 distinct keys; hi 5097, granite 1262),
so NOT_REACHED is a live possibility and is pre-declared informative
(lo would remain policy-sensitive above 71% of its working set — a third
regime distinct from granite/hi).

## Disclosures
- Smoke: synthetic mini-trace through the SAME loader/novelty/policy_cell
  machinery (validates record-format handling, the class of exp705's D1);
  disclosed in LEDGER.md before freeze.
- If the real run crashes pre-measurement, one bounded relaunch is allowed
  only if runner sha still matches ARTIFACT.sha256 (standing rule).
