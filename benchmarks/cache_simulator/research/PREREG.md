# exp681 PREREG — sub-block flash-crowd probe (frozen before launch)

## Question

The arbitration barrier acts only at 4096-access boundaries. exp677 localized
harm to W=4096 alignment; exp680 ruled out cadence. The untested axis: a
regime change STRICTLY INSIDE the window — a cold-crowd burst shorter than
the arbitration interval. Does the center policy (CostFit @ exp676 frozen
center) violate burst or recovery bars under a flash crowd the barrier
cannot see mid-window? The d=4096 aligned cell is the control where the
barrier CAN react.

## Design (frozen)

- Stationary base: skew 1.1, lam 0.3, B=1024, N=638,976, E=128, L=48.
  Single regime; no exp676-style transition. Base seed + crowd seed from
  default_rng(68120261103).spawn(2), drawn FRESH inside run_all (exp680
  disclosed lesson).
- Flash crowd: at block-aligned midpoint, replace d accesses with layer
  uniform + expert uniform over ranks [64, 128) (bottom half by base weight).
  d ∈ {256, 1024, 2048, 4096}. Crowd draws share one prefix across cells;
  base stream identical across cells (verified in smoke).
- Segments: 3 = burst, 4 = recovery (4096 accesses post-burst), 0 = base.
- Arms: lcnu, s3fifo (standalones), costfit @ frozen center (MARGIN 0.005,
  GAMMA 0.0025, HALF_LIFE 32768, RESEGMENT 4096 — sealed file untouched).
- Caps {512, 2048}. 8 cells.

## Bars (frozen; verdict from pass-1 grid)

- H1: burst-window hit deficit vs best standalone ≤ 1.0pp.
- H2: recovery-window hit deficit vs best standalone ≤ 0.5pp.
- H3: barrier switches ≤ 3 per 4096-window.
- acc: costfit loop == prop (exact).
- det: pass-2 grid byte-identical to pass-1.
- Verdict ladder: CROWD-H1 (any H1 violation) > CROWD-H2 > CROWD-SAFE.

## Pre-freeze smoke (disclosed)

- Reduced N=157,696: base deterministic; prefix/tail/revert identity across
  d verified (short burst region reverts to base); crowd prefix identical
  across d; seg counts exact; burst hits < loop for all arms (crowd cold);
  accounting exact; double-eval identical; adjudicate path exercised.
- One smoke ASSERTION was misdesigned (compared base windows at different
  absolute offsets); generator was correct. Fixed the check, not the code.
- Smoke mini-adjudicate (d=4096, cap=512, small N): CROWD-H1 signal — lcnu
  burst 0.1096 vs costfit 0.0781 (~3.2pp). Not evidence; the full run is.
- Disk 6 GiB free (> 5 GiB kill threshold); writes < 5 MB total; scoped
  exception same as exp676-680 series. No model weights, no network.

## Next (pre-registered branches)

- CROWD-H1/H2 → quantify how deficit scales with d; anti-clustering
  candidate gets burst-aware margin (shared mechanism with exp678 COST-H1).
- CROWD-SAFE → blindness confirmed harmless at this scale; close drift axis,
  pivot to anti-clustering candidate directly.

## Post-freeze amendment (2026-11-03 19:28 EST): serialization crash, rerun

First full run completed BOTH grids (~8 min compute) then crashed at the
final json.dumps: adjudicate rows carried numpy scalars (np.bool_ acc,
np.float64 metrics) which Python 3.14 json rejects (exp678/680 runners
cast; exp681 rows did not). Zero effect on design, streams, or bars —
fixed by float()/bool()/int() casts in the row build, disclosed here.
Runner sha superseded e2c3e88b -> (see rerun registration); full rerun
launched after the fix. Result values are unaffected by the cast (same
floats, same comparisons).
