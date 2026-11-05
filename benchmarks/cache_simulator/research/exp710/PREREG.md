# PREREG — exp710: HL-only region probe to the BO search ceiling (STACK-1001, lane 79990)

Registered 2026-11-04 21:10 ET by the follow-up controller. Namespace exp710 free
at registration (ls ~/warp-work: no exp710*; experiment_queue.json tail = exp709;
full-queue grep: no exp710 — lane-0026 shared namespace respected).

Provenance: adopts the never-frozen exp679 design v1 (~/warp-work/exp679-moei-hlregion/
PREREG.md, 2026-11-03, launch was disk-gate-blocked and never run; that directory has
no runner, no seals, no data — so this is a fresh registration, not a relaunch of a
sealed experiment). Frontier item "HL-region probe at cap ceiling" (ledger #200 list).
One tightening vs exp679 v1, made before any data: H0 gets a second replication
bookend (HL 32768 arm == exp674 cf_center), since exp674 already sealed that arm.

## Question
exp669's BO pointed at HL=160,975 (4.9x center). exp674 proved the JOINT point
(M,G,HL) SAFETY-BROKEN out-of-family (worst 0.779pp, 8/59 informative cells >0.5pp),
but its cf_hlonly arm (center M/G, HL=160,975) stayed INSIDE the no-harm bar (worst
0.367pp, 0 cells >0.5pp, max 2 switches). Worst deficit rises with HL (center
0.176pp -> 160,975: 0.367pp, 73% of the bar). Does HL-only stay inside the bar up to
the BO search ceiling 262,144, or does it break first? Measures the certified
center's half-life margin and attributes the exp674 break (joint M/G vs HL).

## Design (frozen at ARTIFACT.sha256)
- Runner: run_hlregion.py (this dir). Machinery imported from ~/warp-work/moei-granite3b
  (policies.build_policy; same import path as exp674). gen_stream / stream_sha are
  AST-identical to exp674 run_hlgen.py (checked programmatically at registration).
- Streams: exactly exp670/exp674's 11 (default_rng(67020261104).spawn(11), same cfg
  order: 9 E=128 L=48 skew {0.8,1.1,1.4} x locality {0,0.3,0.6} + 2 E=32 L=32
  variants); each stream's sha256[:8] asserted equal to research/exp670-synth.json —
  else abort. N=638,208 accesses/stream. Caps {128,256,512,1024,2048,4096} -> 66 cells.
- Arms: lcnu, s3fifo (standalones) + costfit at center M=0.005, G=0.0025 with
  HL in {32768 (center / in-run control), 65536, 131072, HL_REPL, 262144 (ceiling)};
  HL_REPL = exp674 bo_params.half_life parsed at runtime (160974.8276...); _decay =
  0.5**(1/HL) per arm, asserted.
- Full grid run twice (determinism); est. 60-120 min; write footprint < 5 MB.

## Hypotheses / verdict rules (bars copied from exp674)
- Informative cell: |lcnu - s3fifo| >= 0.1pp (expected 59 as exp674).
- H0 REPLICATION: HL_REPL arm == exp674 cf_hlonly AND HL 32768 arm == exp674
  cf_center, per-cell {rate, prop, switches} EXACTLY on all 66 cells (any
  inequality is a machinery fault -> ACCOUNTING-FAIL).
- H1 SAFETY (per HL arm): worst informative-cell deficit vs best standalone
  <= 0.5pp AND switches <= 6 on every informative cell.
- H4 ACCOUNTING: loop rate == prop rate every cell, every HL arm; two full runs
  bit-identical (else abort, no verdict).
- VERDICT: ACCOUNTING-FAIL if H0 or H4 fails; else HL-CEILING-SAFE iff every HL
  arm passes H1; else HL-BOUNDARY@<smallest failing HL>.
- Descriptive (no gate): worst-deficit-vs-HL curve; mean paired delta vs HL 32768;
  certified HL upper edge = largest HL with every smaller HL also safe.

## Disclosed smoke (run BEFORE freeze; outputs in smoke.log)
- S1 sha assertion bites on a perturbed seed (full-N stream 0).
- S2 full-N H0 machinery replication on stream 4 x caps {512, 2048} with ONLY the
  two exp674-sealed replica arms -> exact equality expected (4 cells). Observes
  only values already sealed in exp674; no new-arm (65536/131072/262144) data on
  measured streams.
- S3 H0 negative control: a 1e-6 perturbation must be detected.
- S4 reduced grid (2 streams, caps {128,256}, N=60,000, unpinned): all 7 arms;
  per-arm HL/_decay asserts; accounting; adjudicate well-formed; double-gen identical.
- Any smoke-caused runner edit is disclosed here BEFORE freeze.

## Crash policy
One bounded relaunch only if runner sha == ARTIFACT.sha256 and this PREREG is
unchanged (lane standing rule; O_EXCL RELAUNCH.lock). Runner prints timings only
until settlement, so a crashed run cannot leak outcome values.

## Follow-up (registered)
- HL-CEILING-SAFE: HL axis closed as harmless out-of-family; exp674's break
  attributed to (M,G) -> next: M/G-joint probe at the exp674 point.
- HL-BOUNDARY below ceiling: certified HL band frozen at last passing value; HL
  exploration stops there; report in capture_report.
- ACCOUNTING-FAIL: machinery investigation; no science claim.
