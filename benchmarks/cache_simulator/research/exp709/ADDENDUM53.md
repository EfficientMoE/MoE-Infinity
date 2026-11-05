# ADDENDUM 53 — exp709: beyond-law ceiling probe on hi + granite (cap ceiling = WS20; completes exp708 to 3/3 traces)

Verdict: **CEILING_FLAT_BOTH** — C2 settled FLAT on both remaining traces: hi s_max
0.198pp @ 2304 over caps {2304, 2560, 2816, 3072, 3396}; granite s_max
0.125pp @ 1088 over {1088, 1152, 1228} (frozen bar 1.0pp; no first-rise cap on
either grid). C3 report-only WS20 cells: hi 0.037pp, granite 0.000pp (both <= 0.1pp).
C1 anchors byte-equal sealed exp704 (all 4 arms' whole+burst at hi@2048 and
granite@1024; burst-window lists identical — cross-run determinism 5th/6th
witness). C4 det exact both traces. PIN + structure gates passed
pre-measurement. Exit 0, 344s (run 2; see disclosure).

## What this closes (with exp708: 3/3 traces)
- "Capacity alone decides above CROSS1" now holds **through cap == WS20 on all
  three measured traces**: lo s_max 0.021pp to WS20 4662 (exp708), hi
  0.198pp to 3396, granite 0.125pp to 1228. Best-vs-worst burst spread never
  exceeds 0.198pp anywhere between the law point and full residency.
- The exp707 deployable sizing rule (cap ~ 0.6 x WS20) has no upper validity
  bound found on either model family (qwen30b-A3B, granite-3B): above the rule,
  policy choice is immaterial; spend memory, not policy complexity.
- Non-monotone detail kept, not smoothed: hi spread rises from 0.029pp
  (anchor) to 0.198pp @2304 then decays to 0.037pp at WS20 — a sub-0.2pp
  ripple, 5x below the bar; granite decays monotonically 0.160 -> 0.000pp.

## Measured grid — hi (qwen30b-A3B, WS20 3396, CROSS1 1792)
| cap slots | cap/WS20 | cache GiB | lcnu whole | ring10 whole | spread pp |
|---|---|---|---|---|---|
| 2048 | 0.603 | 18.000 | 0.9110 | 0.9100 | 0.029 |
| 2304 | 0.678 | 20.250 | 0.9371 | 0.9372 | 0.198 |
| 2560 | 0.754 | 22.500 | 0.9555 | 0.9555 | 0.117 |
| 2816 | 0.829 | 24.750 | 0.9681 | 0.9681 | 0.149 |
| 3072 | 0.905 | 27.000 | 0.9764 | 0.9764 | 0.056 |
| 3396 | 1.000 | 29.848 | 0.9834 | 0.9834 | 0.037 |

## Measured grid — granite (granite-3B, WS20 1228, CROSS1 768)
| cap slots | cap/WS20 | cache GiB | lcnu whole | ring10 whole | spread pp |
|---|---|---|---|---|---|
| 1024 | 0.834 | 4.500 | 0.9708 | 0.9708 | 0.160 |
| 1088 | 0.886 | 4.781 | 0.9827 | 0.9827 | 0.125 |
| 1152 | 0.938 | 5.062 | 0.9917 | 0.9917 | 0.006 |
| 1228 | 1.000 | 5.396 | 0.9976 | 0.9976 | 0.000 |

## Integrity, determinism, disclosures
- Runner sha256 6e1605392dbccf50... == ARTIFACT.sha256 == freeze-commit 0ac3207 blob;
  PREREG byte-equal to freeze blob; freeze commit pushed before the run and
  precedes the result mtime.
- **Crash + bounded relaunch (disclosed deviation):** run 1 died silently
  mid-grid after hi caps 2048-2816 (no traceback, no jetsam record; most likely
  launcher process-group teardown). The runner prints timings only until
  settlement, so run 1 emitted no measured values (verified programmatically:
  run1-crashed.log contains no spread/verdict/value tokens) — the relaunch
  cannot be outcome-informed. PREREG's relaunch clause names pre-measurement
  crashes; this relaunch used the lane standing rule (one bounded relaunch iff
  runner sha == ARTIFACT.sha256 and prereg frozen), enforced by an O_EXCL
  RELAUNCH.lock. C1 byte-exact anchors + C4 fresh-load det make run 2
  outcome-equivalent to an uninterrupted run 1.
- Grid resolution: hi 256-324 slots, granite 64-76 slots; a rise narrower than
  one grid step could be missed (PREREG). Two model families only; a true
  family-3 probe needs a fresh full-session trace capture (GPU job; deferred).

Data: exp709/exp709-ceiling2.json (run.log + run1-crashed.log stay local, gitignored); PREREG.md /
ARTIFACT.sha256 / LEDGER.md in exp709/; sealed refs exp704-capacitymap.json,
exp705-costcurves.json, exp708-beyondlaw.json. Follow-up controller
re-adjudicated every gate programmatically from primary sources with an
independently written adjudicator (28/28 checks): verdict CEILING_FLAT_BOTH stands.
