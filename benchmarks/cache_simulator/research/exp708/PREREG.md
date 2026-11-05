# exp708 PREREG — beyond-law flat check: does policy convergence persist to the full working set?

Program STACK-1001, lane 79990. MEASURED tier (real-trace simulation; countable
toward 1001-native if settled).

## Motivation
exp707 (ledger #230) completed the cost layer: deployable rule cap ~ 0.6 x WS20,
band 0.528-0.625 over 3 traces. But the measured lo grid stops at 3328 slots =
0.713 x WS20(lo); the region between the law point and the working set itself
(0.77-1.0 x WS20) is unmeasured. Two live hypotheses: (i) FLAT — above CROSS1
capacity alone decides and spread stays negligible all the way to WS20;
(ii) RISE — arms re-diverge as cap approaches the full final-20% working set
(e.g. annex/ring behavior when nearly everything fits). exp706's 3072/3328 cells
(0.156/0.055pp) hint FLAT but do not reach WS20. This experiment maps
caps 3328-4662 on the lo trace, ending exactly at WS20.

## Frozen inputs (read-only; parsed at runtime AND pinned)
- Sealed exp706 result (~/warp-work/exp706-moei-locross/exp706-locross.json):
  anchor cell @3328 (spread_pp 0.055; lcnu whole 0.9339024267950261; ring10
  whole 0.9332333659245888) and the 16 burst windows.
- Sealed exp705 result (research/exp705/exp705-costcurves.json): WS20(lo)=4662,
  b_e 9,437,184 B.
- Frozen lo trace stream (exp656 stream_qwen30b_lo.pt): N=638208, distinct=5842,
  prompts [512,512,126,512], n_win=155, burst_n=16 — identical CFG to exp706.

## Design
Mirrors exp706's sealed run_locross.py exactly (loader, novelty, burst windows,
arms lcnu/ring10/ssr_10/eti_2020, W=4096, spread = best-vs-worst burst hit-rate
in pp). Grid: caps [3328, 3584, 3840, 4096, 4352, 4662] (3328 = anchor overlap
with sealed exp706; 4662 == WS20(lo) exactly = the cap ceiling).

## Pre-registered gates (frozen before any real run)
- B1 ANCHOR (hard, exit 5): @3328 spread AND lcnu whole AND ring10 whole
  byte-equal to sealed exp706; recomputed burst windows equal sealed exp706's
  16-window list (cross-run determinism, 4th witness).
- B2 BEYOND-FLAT (verdict-determining, settled either way): let s_max = max
  spread_pp over {3584, 3840, 4096, 4352, 4662}. Verdict BEYOND_FLAT if
  s_max <= 1.0pp; verdict BEYOND_RISE if s_max > 1.0pp, recording the first
  cap >= 3584 with spread > 1.0pp. Both outcomes are settled (exit 0) — a
  question experiment: the verdict word carries the answer, not pass/fail.
- B3 WS20-CELL (report-only): spread @4662 (cap == WS20 exactly) expected
  <= 0.1pp under FLAT; value + flag reported regardless of outcome.
- B4 DETERMINISM (hard, exit 2): det cell (lcnu, 3584) recomputed through a
  fresh stream load + fresh arm == grid cell exactly.
- Structure (exit 3): stream constants N/distinct/prompts/n_win must equal the
  frozen CFG (else exit 3, mirroring exp706).
- Exit 0 settled (BEYOND_FLAT or BEYOND_RISE) | 2 det | 3 structure | 5 anchor.

## Scope honesty
Single trace (lo), single grid extension, frozen arms — no new policies, no
tuning, nothing measured twice except the pinned anchor cell. A BEYOND_RISE
outcome bounds the deployable rule's validity to [CROSS1, first-rise cap) and
motivates a ceiling-aware addendum; BEYOND_FLAT extends "capacity alone decides
above CROSS1" to the working set itself. Either way the 0.6xWS20 rule band
(exp707) is unchanged — this probes the region ABOVE it. Grid resolution
256-310 slots; a rise narrower than one grid step could be missed (disclosed).

## Disclosures
- Smoke: synthetic mini-trace through the SAME machinery (records, loader,
  novelty, burst, policy_cell), mirroring exp706's disclosed smoke; disclosed
  in LEDGER.md before freeze. No result JSON from smoke.
- If the real run crashes pre-measurement, one bounded relaunch allowed only if
  runner sha still matches ARTIFACT.sha256 (standing rule).
