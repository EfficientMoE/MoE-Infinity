# exp709 LEDGER (lane 79990, STACK-1001)

## Pre-freeze (disclosed)
- Smoke (synthetic mini-trace, same machinery incl. structure check + fresh-load det):
  SMOKE_OK — det=True sane=True N=12800 distinct=96. No result JSON. Log: smoke.log.
- Pre-measurement dry check (pins + real-stream structure only, NO grid):
  hi pins True structure 4/4; granite pins True structure 4/4.
- Pins injected via repr() from sealed exp704 JSON (no hand transcription).
- Pivot from family-3 probe disclosed in PREREG (exp670 kimi/olmoe traces are
  300-token decode-only slices, incompatible with burst methodology).

## Freeze
- PREREG.md sha256 prefix b6b656f8e68d47b0; runner sha in ARTIFACT.sha256.
- Freeze commit precedes the real run (see git log).

## Real run
- (appended by follow-up controller after settlement)

## Crash + bounded relaunch (follow-up controller, 2026-11-04 20:50 ET)
- Run 1 (pid 27973, launched 20:33 ET) died silently ~20:35 after hi caps
  2048/2304/2560/2816 (progress lines only; log preserved as run1-crashed.log).
  No traceback, no result JSON, no jetsam/memorystatus record for the pid ->
  external signal; most likely the launching session's process-group teardown
  (launcher session ended ~20:33).
- No measured values were ever emitted (the runner prints timings only until
  settlement), so the relaunch cannot be outcome-informed. C1 (byte-exact
  anchors vs sealed exp704) and C4 (fresh-load determinism) make the rerun
  outcome-equivalent to run 1.
- Deviation disclosed: PREREG's relaunch clause names *pre-measurement*
  crashes; this crash was mid-grid. Relaunch made under the lane standing rule
  (one bounded relaunch iff runner sha == ARTIFACT.sha256 and prereg frozen),
  same precedent as exp681's disclosed rerun. Checked at runtime: runner sha
  6e1605392dbccf50... == ARTIFACT.sha256 == freeze commit 0ac3207 blob;
  PREREG sha b6b656f8e68d47b0... unchanged. RELAUNCH.lock (O_EXCL) enforces one.
- Run 2: pid 44654, launched 20:50 ET, detached (setsid/nohup), same exp650
  venv (CPython 3.14.8, torch 2.15.0). No further relaunch permitted.
