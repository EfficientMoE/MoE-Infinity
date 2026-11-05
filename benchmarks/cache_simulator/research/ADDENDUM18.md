## Addendum 18 — exp670: CostFit generalizes out-of-family (VERDICT: GENERALIZES)

All prior CostFit evidence (Addenda 14–17) rests on two real traces from one
capture family. exp670 tests the sealed center config (MARGIN 0.005, GAMMA
0.0025, HALF_LIFE 32768) on 11 synthetic streams it has never seen: Zipf
skew s {0.8, 1.1, 1.4} × temporal locality λ {0.0, 0.3, 0.6} (E=128, L=48,
638,208 accesses; recency-weighted replay H=2048, R=4096) plus two E=32
topology variants. Caps {128…4096}; lcnu/s3fifo/costfit evaluated in the
same harness path; informative-cell rule (inner gap ≥ 0.1pp) and verdict
bars frozen in prereg 606e9507 before any stream was generated.

**Result: GENERALIZES** — H1 True, H2 True, H3 True;
59 informative cells across 9/9
primary streams (all informative); zero violations; full double-run
determinism (stream shas and all rates identical across both runs).

Worst deficit at any informative cell: **0.176pp** (bar 0.5pp).
Max switches anywhere: **2** (bar 6). No near-boundary cells
(no cell with deficit > 0.3pp or switches ≥ 5). The design is more stable
on the synthetic family than on the real traces (real center: 4 switches
@30b/1024; here max 2).

Cells where arbitration BEATS both standalones (4): s0.8/λ0.0/E128@1024 (+0.002pp); s1.1/λ0.0/E128@1024 (+0.015pp); s1.1/λ0.3/E128@512 (+0.002pp); s1.4/λ0.3/E128@1024 (+0.005pp).

Uninformative cells (7): s1.4/λ0.3/E128@4096; s1.1/λ0.3/E32@1024; s1.1/λ0.3/E32@2048; s1.1/λ0.3/E32@4096; s1.4/λ0.6/E32@1024; s1.4/λ0.6/E32@2048; s1.4/λ0.6/E32@4096 — all in the
cache-covers-working-set regime (inner gap < 0.1pp; arbitration inert by
construction), concentrated in E=32 topology at high caps.

Interpretation: the self-limiting barrier mechanism is not an artifact of
the capture family — it tracks the better inner within 0.18pp and never
oscillates across a 3×3 structure grid and a 4× topology shift. Combined
with exp668's parameter-region robustness, the design now has both
parameter and structure generalization evidence.

Artifacts: exp670-synth.json (per-stream per-cap rates/switches, stream
shas, violation map); generator + evaluation frozen in run_synth.py
(sha in ARTIFACT.sha256); prereg 606e9507 with pre-freeze smoke disclosed.
