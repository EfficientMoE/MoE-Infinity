# exp709 PREREG — beyond-law ceiling probe on hi + granite (completes exp708 to 3/3 traces)

Program STACK-1001, lane 79990. MEASURED tier (real-trace simulation; countable
toward 1001-native if settled).

## Motivation
exp708 (ledger #231) settled BEYOND_FLAT on the lo trace: best-vs-worst policy
spread stays <= 0.021pp from CROSS1 all the way to cap == WS20(lo). The other
two traces are measured only up to their exp704 grid tops: hi stops at 2048
slots = 0.603 x WS20(hi)=3396; granite stops at 1024 = 0.834 x WS20(granite)=1228.
The claim "capacity alone decides above CROSS1, through full residency" is
therefore 1/3 traces at the ceiling. This experiment maps both remaining
beyond-law regions up to exactly WS20 with exp708's frozen machinery.

Pivot disclosure: next_hint listed a family-3 trace probe first. The only
non-qwen/granite traces on disk (exp670 kimi/olmoe) are 300-token decode-only
slices (974/5,083 keys, ~9-15 windows of W=4096, no prefill records) —
structurally incompatible with the 155-192-window burst methodology that
defines the law. A true family-3 probe needs a fresh full-session trace
capture (GPU/Winnie job); deferred and recorded as such, not attempted here.

## Frozen inputs (read-only; parsed at runtime AND pinned)
- Sealed exp704 result (~/warp-work/exp704-moei-capacitymap/exp704-capacitymap.json):
  hi anchor @2048 (spread 0.029; lcnu whole 0.9110196431477865; ring10 whole
  0.9100151062011719; 20 burst windows); granite anchor @1024 (spread 0.16;
  lcnu whole 0.9708386607501003; ring10 whole 0.9708355269755314; 16 burst
  windows). Pins injected into the runner via repr() from the sealed JSON.
- Sealed exp705 (research/exp705/exp705-costcurves.json): WS20(hi)=3396,
  WS20(granite)=1228.
- Frozen streams: exp656 stream_qwen30b_hi.pt (N=786432, distinct=5097,
  prompts [512x4], n_win=192, burst_n=20); exp650 stream_granite3b.pt
  (N=638208, distinct=1262, prompts [313,512,336,245,169,197,512,209],
  n_win=155, burst_n=16) — identical CFG to sealed exp704.

## Design
Mirrors exp708's sealed run_beyondlaw.py exactly (loader, novelty, burst
windows, arms lcnu/ring10/ssr_10/eti_2020, W=4096, spread = best-vs-worst burst
hit-rate in pp, rounded 3dp). Grids:
- hi: caps [2048, 2304, 2560, 2816, 3072, 3396] (2048 = anchor overlap with
  sealed exp704; 3396 == WS20(hi) exactly).
- granite: caps [1024, 1088, 1152, 1228] (1024 = anchor overlap; 1228 ==
  WS20(granite) exactly).

## Pre-registered gates (frozen before any real run)
- PIN (hard, exit 5, pre-measurement): runtime-parsed sealed anchors and WS20
  equal the frozen pins; each grid's last cap == WS20.
- Structure (hard, exit 3, pre-measurement): both streams' N/distinct/
  prompts/n_win equal frozen CFG.
- C1 ANCHOR (hard, exit 5): per trace, anchor-cell spread AND lcnu whole AND
  ring10 whole byte-equal to sealed exp704; recomputed burst windows equal
  sealed exp704's list (cross-run determinism, 5th/6th witness).
- C2 BEYOND-FLAT (verdict-determining, settled either way): per trace,
  s_max = max spread_pp over caps above the anchor. Trace FLAT iff
  s_max <= 1.0pp (first cap > 1.0pp recorded otherwise). Verdict
  CEILING_FLAT_BOTH if both FLAT (=> with exp708, flat-to-WS20 holds 3/3
  traces); else CEILING_RISE_<TRACES>. Both outcomes exit 0.
- C3 WS20-CELL (report-only): spread at cap == WS20 per trace; flag <= 0.1pp.
- C4 DETERMINISM (hard, exit 2): det cell (hi: lcnu@2304; granite: lcnu@1088)
  recomputed through a fresh stream load + fresh arm == grid cell exactly.
- Exit 0 settled | 2 det | 3 structure | 5 pin/anchor.

## Scope honesty
Two traces, frozen arms, no tuning, nothing measured twice except the pinned
anchor cells. Grid resolution: hi 256-324 slots, granite 64-76 slots; a rise
narrower than one grid step could be missed. granite's anchor (0.16pp @1024)
already sits above its CROSS1 (768); hi's anchor (0.029pp @2048) is above its
CROSS1 (1792). A RISE outcome bounds the exp707 deployable rule's upper
validity on that trace; FLAT closes the ceiling question 3/3 for the two
measured model families (qwen30b-A3B, granite-3B). Family generality beyond
these two families remains untested (see pivot disclosure).

## Disclosures
- Smoke: synthetic mini-trace through the SAME machinery (records, loader,
  structure check, novelty, burst, policy_cell, spread, fresh-load det),
  mirroring exp708's disclosed smoke; disclosed in LEDGER.md before freeze.
  No result JSON from smoke.
- If the real run crashes pre-measurement, one bounded relaunch allowed only if
  runner sha still matches ARTIFACT.sha256 (standing rule).
