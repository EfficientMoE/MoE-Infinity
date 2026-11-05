# PREREG — exp711: full 2^3 factorial attribution of exp674's SAFETY-BROKEN BO point (STACK-1001, lane 79990)

Registered 2026-11-04 23:20 ET by the follow-up controller, immediately after exp710
settled (HL-BOUNDARY@262144, ADDENDUM54, ledger #233). Namespace exp711 free at
registration (ls ~/warp-work: no exp711*; experiment_queue.json tail = exp710; full-queue
grep "exp711": 0 hits; research/ has no exp711 — lane-0026 shared namespace respected).

## Question
exp674: the exp669 BO winner (M,G,HL) = (margin, gamma, half_life) read from exp674
bo_params is SAFETY-BROKEN on the 11 synthetic streams (worst 0.779pp, 8/59 informative
cells > 0.5pp). exp674 + exp710: HL alone at the BO value (center M/G) is SAFE
(worst 0.367pp). So the break needs the (M,G) move. Which factor (M, G, both, or an
interaction with HL) carries it? A full 2^3 factorial over {center, BO} per factor
answers this exactly, with 3 of 8 corners already sealed (replication bookends).

## Design (frozen at ARTIFACT.sha256)
- Runner: run_mgfactorial.py (this dir). Machinery imported from ~/warp-work/moei-granite3b
  (policies.build_policy; same import path as exp674/exp710). gen_stream / stream_sha /
  stream_cfgs AST-identical to exp710 run_hlregion.py (checked programmatically at
  registration; gen_stream/stream_sha also AST-identical to exp674 run_hlgen.py).
  make_costfit is exp674's 3-param form with one statement reordered (params unpacked
  before build_policy) — semantically identical; S2 full-N H0 replication validates it.
- Streams: exactly exp670/exp674/exp710's 11 (default_rng(67020261104).spawn(11)); each
  stream's sha256[:8] asserted equal to research/exp670-synth.json, else abort.
  N=638,208 accesses/stream. Caps {128,256,512,1024,2048,4096} -> 66 cells.
- Factors: M: MARGIN 0.005 -> BO margin; G: GAMMA 0.0025 -> BO gamma; H: HALF_LIFE
  32768 -> BO half_life. BO values parsed at runtime from sealed exp674 bo_params (exact
  floats). _decay = 0.5**(1/HL) per arm, asserted with MARGIN/GAMMA/HALF_LIFE.
- Arms: lcnu, s3fifo + costfit at all 8 corners cf_000, cf_M00, cf_0G0, cf_MG0, cf_00H,
  cf_M0H, cf_0GH, cf_MGH (letter present = factor at BO value).
- Full grid run twice (determinism), serial, single process (host swap is high; no pool).
  Est. 2.5-3.5 h; write footprint < 5 MB.

## Hypotheses / verdict rules (bars copied from exp674/exp710)
- Informative cell: |lcnu - s3fifo| >= 0.1pp (expected 59).
- H0 REPLICATION: cf_000 == exp674 cf_center, cf_00H == exp674 cf_hlonly, cf_MGH == exp674
  cf_bo, AND lcnu/s3fifo == exp674, per-cell dicts EXACTLY on all 66 cells (330 checks).
  Any inequality -> ACCOUNTING-FAIL.
- H4 ACCOUNTING: loop rate == prop rate every cell, every corner; two full runs
  bit-identical (else abort, no verdict).
- H1 SAFETY (per corner): worst informative-cell deficit vs best standalone <= 0.5pp AND
  switches <= 6 on every informative cell.
- VERDICT (ACCOUNTING-FAIL if H0 or H4 fails; else this ladder, in order):
  1. cf_MG0 fails H1 (break exists at center HL):
     M00 fails & 0G0 safe -> MG-BREAK:M-DRIVEN; M00 safe & 0G0 fails -> MG-BREAK:G-DRIVEN;
     both fail -> MG-BREAK:BOTH-KNOBS; both safe -> MG-BREAK:SYNERGY.
  2. else if cf_MGH safe -> NO-BREAK (would contradict sealed exp674; H0 makes this
     unreachable unless machinery drift, reported as-is).
  3. else (MG0 safe, MGH fails; break needs HL): M0H fails & 0GH safe -> HL-INTERACTION:MxH;
     M0H safe & 0GH fails -> HL-INTERACTION:GxH; both fail -> HL-INTERACTION:BOTH;
     both safe -> HL-INTERACTION:THREE-WAY.
- Descriptive (no gate): standard 2^3 Yates effects (main + interactions, /4) on per-corner
  worst deficit and on mean paired delta vs cf_000; per-corner over-bar cell lists and
  overlap with exp674 cf_bo violation cells.

## Disclosed smoke (run BEFORE freeze; outputs in smoke.log)
- S1 sha assertion bites on a perturbed seed (full-N stream 0).
- S2 full-N H0 machinery replication on stream 4 x caps {512, 2048} with ONLY the three
  exp674-sealed bookend corners (000, 00H, MGH) + standalones -> exact equality expected
  (10 checks). Observes only values already sealed in exp674; no new-corner data on
  measured streams.
- S3 H0 negative control: a 1e-6 perturbation on cf_MGH must be detected.
- S4 reduced grid (2 streams, caps {128,256}, N=60,000, unpinned): all 10 arms; per-arm
  param/_decay asserts; accounting; adjudicate well-formed; double-gen identical; verdict
  ladder unit checks (pure logic).
- Any smoke-caused runner edit is disclosed here BEFORE freeze.
- RESULT (2026-11-04 23:22 ET): SMOKE_OK — S1 bites; S2 10/10 exact; S3 detects; S4 det,
  well-formed, accounting, ladder all True. S2 stream-4 2-cap 5-arm pass = 76 s ->
  full run est. ~2.8 h. No smoke-caused runner edits.

## Crash policy
One bounded relaunch only if runner sha == ARTIFACT.sha256 and this PREREG is unchanged
(lane standing rule; O_EXCL RELAUNCH.lock). Runner prints timings only until settlement,
so a crashed run cannot leak outcome values.

## Follow-up (registered)
- MG-BREAK:M-DRIVEN / G-DRIVEN: the named knob carries the harm; next = single-knob dose
  line on that knob at center (find its certified band, as exp710 did for HL).
- MG-BREAK:BOTH-KNOBS / SYNERGY: report; next = joint (M,G) band probe along the
  center->BO segment.
- HL-INTERACTION:*: harm is conditional on long memory; certified HL band from exp710
  binds only at center M/G — report as a deployment caveat.
- ACCOUNTING-FAIL: machinery investigation; no science claim.
