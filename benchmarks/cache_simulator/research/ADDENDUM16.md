## Addendum 16 — exp667 CostFit: self-limiting switch-cost arbitration — VERDICT: SUPPORTED (first fully-passing arbitration design of the series)

Completes the fixed-margin refutation pair (exp665: margin too large → late
switch @4096; exp666: margin too small → oscillation @1024) with the remaining
untested variant: replace the fixed margin with a SELF-LIMITING barrier —
required_lead = MARGIN + GAMMA × (decayed recent-switch count). A first switch
from cold start faces only the base margin (regime shifts stay cheap); each
flip raises its own barrier (1 recent switch → 0.75pp, 2 → ~1.0pp, 4 →
~1.5pp), damping thrash. Prereg sha 462b9cdb frozen before sweep, 3-cap smoke
disclosed in prereg.

**Rules (frozen):** H1 near-dominance (>= best standalone − 0.5pp, both
traces); H2 stability (<= 6 switches per cap); H3 credit-accounting (credited
at access time; divergence from either inner's standalone cumulative at
switched caps; resim determinism).

**Verdict: SUPPORTED** — H1 PASS (30b worst +0.113pp, granite worst +0.112pp);
H2 PASS (30b max 4 switches — exp666 DecayFit had 8 at cap 1024; granite max 1);
H3 PASS (determinism + accounting clean at all 15 caps; analyzer rerun
byte-identical).

| Trace | Cap | CostFit | Best standalone | Deficit (pp) | Switches | Note |
|---|---|---|---|---|---|---|
| qwen30b | 128 | 0.208345 | lcnu 0.208345 | +0.000 | 0 | — exact LCNU anchor
| qwen30b | 256 | 0.327289 | lcnu 0.327289 | +0.000 | 0 | — exact LCNU anchor
| qwen30b | 384 | 0.414760 | lcnu 0.414760 | +0.000 | 0 | — exact LCNU anchor
| qwen30b | 512 | 0.485328 | lcnu 0.485328 | +0.000 | 0 | — exact LCNU anchor
| qwen30b | 768 | 0.590480 | lcnu 0.590480 | +0.000 | 0 | — exact LCNU anchor
| qwen30b | 1024 | 0.667390 | lcnu 0.668520 | +0.113 | 4 |
| qwen30b | 1536 | 0.809308 | s3fifo 0.808347 | -0.096 | 1 | — BEATS both standalones
| qwen30b | 2048 | 0.894752 | s3fifo 0.895616 | +0.086 | 3 |
| qwen30b | 3072 | 0.968435 | s3fifo 0.968484 | +0.005 | 1 |
| qwen30b | 4096 | 0.988625 | s3fifo 0.988872 | +0.025 | 1 |
| granite | 128 | 0.283762 | lcnu 0.283762 | +0.000 | 0 | — exact LCNU anchor
| granite | 256 | 0.468164 | lcnu 0.468164 | +0.000 | 0 | — exact LCNU anchor
| granite | 512 | 0.728697 | lcnu 0.728697 | +0.000 | 0 | — exact LCNU anchor
| granite | 768 | 0.886415 | lcnu 0.886415 | +0.000 | 0 | — exact LCNU anchor
| granite | 1024 | 0.972067 | s3fifo 0.973187 | +0.112 | 1 |

**Mechanism findings**

- The self-limiting barrier damps the exp666 oscillation exactly as designed:
  cap 1024 goes 8 → 4 switches while IMPROVING its rate
  (0.667390 vs DecayFit 0.666973; deficit +0.113pp).
- Cold-start switching is preserved: 4096 still takes exactly 1 switch at
  +0.025pp (barrier at first switch = base margin only).
- The cap-level WIN at 1536 is preserved (−0.096pp vs both standalones) —
  running the LCNU prefix before switching to S3FIFO beats either fixed policy.
- 9 of 15 caps reproduce the best standalone EXACTLY (zero-switch caps are
  bit-identical to the anchor policy): CostFit never pays for its arbitration
  when arbitration is not needed.
- Series conclusion: prediction-free shadow arbitration becomes PRACTICAL with
  decayed scoreboards + self-limiting switch costs: near-dominant everywhere
  (worst +0.113pp), ≤4 switches, 2× metadata overhead unchanged.

**Artifacts:** results_real_{qwen3-30b-a3b,granite3.1-moe-3b}-costfit.{csv,md},
policies_costfit.py, analyze_exp667_costfit.py, exp667-verdict.json
(programmatic from verdict JSON).
