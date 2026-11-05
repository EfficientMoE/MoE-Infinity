# ADDENDUM 52 — exp708: beyond-law flat check (lo grid 3328–4662, cap ceiling = WS20)

Verdict: **BEYOND_FLAT** — B2 settled: s_max over the beyond-law caps
{3584..4662} = 0.021pp (frozen threshold 1.0pp), no first-rise cap
anywhere on the grid; B3 report-only WS20 cell (cap == WS20(lo) exactly) spread
0.015pp <= 0.1pp as expected under FLAT. B1 anchors exact vs sealed
exp706 at the overlap cell @3328 (spread 0.055pp; lcnu/ring10 whole
byte-equal; 16-window burst list equal — cross-run determinism holds a 4th
time). B4 det exact. Structure gate passed pre-measurement. Exit 0.
318s.

## What this closes
- exp707's deployable sizing rule (cap ~ 0.6 x WS20, band 0.528-0.625)
  was measured only up to 0.713 x WS20(lo). This experiment maps the remaining
  region 0.77-1.00 x WS20 and ends exactly at the working set: **"capacity
  alone decides above CROSS1" now holds through cap == WS20** — policy choice
  is immaterial (<= 0.02pp best-vs-worst burst spread) everywhere above
  the law point, all the way to fully resident.
- No re-divergence near the ceiling: arms do NOT separate again as the cache
  approaches the full final-20% working set (annex/ring near-fit behavior
  absent). The exp707 rule's validity region extends upward without bound
  found; the flat region is [CROSS1, WS20] on this trace.

## Measured grid (lo trace, frozen arms lcnu/ring10/ssr_10/eti_2020, W=4096)
| cap slots | cap/WS20 | cache GiB | lcnu whole | ring10 whole | spread pp |
|---|---|---|---|---|---|
| 3328 | 0.714 | 29.2500 | 0.9339 | 0.9332 | 0.055 |
| 3584 | 0.769 | 31.5000 | 0.9466 | 0.9466 | 0.008 |
| 3840 | 0.824 | 33.7500 | 0.9578 | 0.9578 | 0.003 |
| 4096 | 0.879 | 36.0000 | 0.9668 | 0.9668 | 0.021 |
| 4352 | 0.934 | 38.2500 | 0.9740 | 0.9740 | 0.008 |
| 4662 | 1.000 | 40.9746 | 0.9808 | 0.9808 | 0.015 |

Spread stays 0.003-0.021pp across the entire
beyond-law region (50x below the 1.0pp bar); lcnu whole-stream hit rate rises
0.9339 -> 0.9808 as capacity reaches the working set.

## Anchors & determinism
- B1: overlap cell @3328 byte-equal sealed exp706 (0.9339024267950261 lcnu
  whole) — 4th independent cross-run determinism witness for this simulator
  stack; 16 burst windows identical.
- B4: det cell (lcnu, 3584) recomputed through a fresh stream load + fresh arm
  == grid cell exactly.
- WS20(lo) = 4662 parsed at runtime from sealed exp705 (C4-verified there).
- Integrity: runner sha256 4a161af69af38364...
  matches ARTIFACT.sha256; freeze commit 7145f31 preceded the run artifacts.

## Tier & limitations
- Measured tier (real-trace simulation on the frozen exp656 lo trace), single
  purpose: extend the sealed exp706 grid to the working set. No new policies,
  no tuning; grid and bars frozen pre-run.
- Grid resolution 256-310 slots: a rise narrower than one grid step could be
  missed (disclosed in PREREG). Single trace (lo); the beyond-law region on
  granite/hi is bounded by their exp704 grids (their WS20 cells
  1228/3396 were not re-measured here).
- Law band {granite 0.625, hi 0.528, lo 0.604} unchanged — this probes the
  region ABOVE the rule, not the rule itself.

Data: exp708/exp708-beyondlaw.json; PREREG.md/ARTIFACT.sha256 in exp708/;
sealed refs exp706-locross.json, exp705-costcurves.json.
Follow-up controller re-adjudicated every gate programmatically from primary
sources (18/18 checks): exact match, verdict BEYOND_FLAT stands.
