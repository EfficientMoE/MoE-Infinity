# exp712 — PREREG: single-knob GAMMA dose line at center (M, HL) — certified gamma band

Program STACK-1001, lane 79990. Written 2026-11-05 ~02:10 EST, before any real run. Frozen
by sha256 in ARTIFACT.sha256 and pushed before launch.

## Why this experiment (frozen consequence of exp711)
exp711 (ADDENDUM55, ledger #234) returned **MG-BREAK:G-DRIVEN**: with HL and margin at center,
moving gamma alone from center 0.0025 to the exp669/exp674 BO value breaks the no-harm bar
(0G0: 6/59 informative cells, worst 1.145pp), while every center-gamma corner passes. exp711's
pre-registered G-DRIVEN branch named the next step: a single-knob gamma dose line at center M/HL
to find gamma's certified band (the exp710 design, applied to gamma). This is that experiment.

## Design
- Policy: costfit (benchmarks/cache_simulator/policies.py), margin = 0.005 and half_life = 32768
  (center) on every arm; ONLY gamma moves.
- Gamma line (6 arms, geometric, endpoints inclusive): g_k = 0.0025 * (BO/0.0025) ** (k/5),
  k = 0..4, and g5 = BO exactly, where BO = float(exp674 bo_params["gamma"]) parsed at runtime
  from the sealed exp674 result JSON (no hand transcription). Runtime values are printed in
  smoke.log (g0 = 0.0025 ... g5 = BO, strictly increasing; ratio per step = (BO/0.0025)^(1/5)).
- Streams: exactly the 11 synthetic streams of exp670/674/710/711 (sha8 asserted vs
  research/exp670-synth.json); caps {128,256,512,1024,2048,4096}; N = 638,208 accesses.
- Standalone references per cell: lcnu, s3fifo (best-of-two is the no-harm reference).
- Generator/stream/costfit-construction code is AST-identical to the sealed exp711 runner
  (gen_stream, stream_sha, stream_cfgs, make_costfit); evaluate/run_all differ only by renames.

## Gates (verified programmatically by the adjudicator; same bars as exp674/710/711)
- **H0 replication (validity)**: g0 must equal exp711 cf_000 and g5 must equal exp711 cf_0G0
  EXACTLY in every one of the 66 cells (rate, prop, switches), and lcnu/s3fifo must equal
  exp711's standalones in every cell: 66x2 + 66x2 = 264 cell-comparisons, all exact. Any
  mismatch -> ACCOUNTING-FAIL (no verdict on safety).
- **H4 accounting**: every costfit arm's loop hit-rate == policy-reported effective_hit_rate in
  every cell (396 arm-cells); and two full runs bit-identical (determinism). Fail -> ACCOUNTING-FAIL.
- **H1 per arm (safety)**: over informative cells (|lcnu - s3fifo| >= 0.1pp), the arm is SAFE iff
  no cell has deficit vs max(lcnu, s3fifo) > 0.5pp AND no cell has switches > 6.

## Verdict ladder (frozen; evaluated in gamma order g0..g5)
- certified edge e = largest index such that g0..g_e are all SAFE (e = -1 if g0 fails).
- e = 5 -> **G-LINE-SAFE** (contradicts exp711's g5 failure; would only arise via H0 failure, so
  under H0-pass this is impossible by construction — reported for completeness).
- 1 <= e <= 4 -> **G-BOUNDARY@g{e+1}**: the certified gamma band at center M/HL is
  [0.0025, g_e]; first failing dose g_{e+1}.
- e <= 0 -> **G-BAND-COLLAPSED**: no dose above center is certified (g1 already fails, or g0 fails).
- nonmonotone flag = any SAFE arm above the first failing arm (reported, does not change verdict).
- Expected under H0-pass: g0 SAFE (exp711 000 passed), g5 FAIL (exp711 0G0 failed) -> one of
  G-BOUNDARY@g1..g5 or G-BAND-COLLAPSED. No outcome is favoured in advance.

## Descriptive (no gate)
Per-arm worst deficit, cells over bar (with cell list), max switches, mean paired delta vs g0;
monotonicity of worst-deficit and over-bar count along the line.

## Consequences (pre-registered)
- G-BOUNDARY@g_j with j >= 2: re-run the exp674 joint transfer with gamma clamped to the certified
  edge (BO margin + BO HL kept) — tests whether clamping gamma alone repairs the BO point.
- G-BOUNDARY@g1 / G-BAND-COLLAPSED: gamma is not tunable upward at center; next = gamma dose line
  BELOW center (does lower gamma buy safety margin?) or retire gamma from the BO search space.
- ACCOUNTING-FAIL: diagnose replication drift before any safety claim.

## Disclosed smoke (run BEFORE freeze; machinery only, no outcome information)
S1 sha assertion bites on a perturbed seed; S2 full-N H0 replication on stream 4 x caps
{512, 2048} with ONLY the two sealed bookend arms g0/g5 (8 comparisons, exact vs exp711 —
these values are already public in exp711); S3 H0 negative control (1e-6 perturbation detected);
S4 two-run determinism at reduced N=60k on 2 streams x 2 unpinned caps, adjudicator
well-formedness, H4 accounting, ladder unit tests (6 synthetic patterns), gamma-line
construction checks. Interior arms g1..g4 are evaluated only at reduced N=60k in S4 and their
safety outcome is not inspected.

## Scope limits
Synthetic streams only; no real-trace or serving claim. Single-knob line: interactions with M/HL
are exp711's domain and are not re-tested here.
