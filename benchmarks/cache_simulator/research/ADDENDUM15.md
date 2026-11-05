## Addendum 15 — exp666 DecayFit: decayed-rate shadow arbitration — VERDICT: FAIL (H2 oscillation; H1 passes with first cap-level WIN)

Follow-up to Addendum 14 (exp665): the cumulative scoreboard's history-drag was
replaced by an exponentially decayed scoreboard (half-life 32768 accesses,
margin 0.5pp, re-arbitration every 4096, initial LCNU; prereg sha 40f2f8dd
frozen before sweep, 3-cap smoke disclosed in prereg).

**Rules (frozen):** H1 near-dominance (>= best standalone − 0.5pp, both
traces); H2 stability (<= 6 switches per cap); H3 credit-accounting (credited
at access time; CSV must diverge from either inner's standalone cumulative at
switched caps).

**Verdict: FAIL.** H1 PASS (30b worst +0.155pp, granite worst +0.112pp);
H2 FAIL at 30b cap 1024 (8 switches); H3 PASS (determinism + accounting clean
at all 15 caps).

| Trace | Cap | DecayFit | Best standalone | Deficit | Switches | Note |
|---|---|---|---|---|---|---|
| qwen30b | 128 | 0.208345 | lcnu 0.208345 | +0.000 | 0 | — exact LCNU anchor
| qwen30b | 256 | 0.327289 | lcnu 0.327289 | +0.000 | 0 | — exact LCNU anchor
| qwen30b | 384 | 0.414760 | lcnu 0.414760 | +0.000 | 0 | — exact LCNU anchor
| qwen30b | 512 | 0.485328 | lcnu 0.485328 | +0.000 | 0 | — exact LCNU anchor
| qwen30b | 768 | 0.590480 | lcnu 0.590480 | +0.000 | 0 | — exact LCNU anchor
| qwen30b | 1024 | 0.666973 | lcnu 0.668520 | +0.155 | 8 | — OSCILLATION (H2 breach)
| qwen30b | 1536 | 0.809308 | s3fifo 0.808347 | -0.096 | 1 | — BEATS both standalones
| qwen30b | 2048 | 0.895268 | s3fifo 0.895616 | +0.035 | 3 |
| qwen30b | 3072 | 0.968435 | s3fifo 0.968484 | +0.005 | 1 |
| qwen30b | 4096 | 0.988625 | s3fifo 0.988872 | +0.025 | 1 |
| granite | 128 | 0.283762 | lcnu 0.283762 | +0.000 | 0 | — exact LCNU anchor
| granite | 256 | 0.468164 | lcnu 0.468164 | +0.000 | 0 | — exact LCNU anchor
| granite | 512 | 0.728697 | lcnu 0.728697 | +0.000 | 0 | — exact LCNU anchor
| granite | 768 | 0.886415 | lcnu 0.886415 | +0.000 | 0 | — exact LCNU anchor
| granite | 1024 | 0.972067 | s3fifo 0.973187 | +0.112 | 1 |

**Mechanism findings**

- The decayed scoreboard FIXES exp665's 4096 failure: deficit +0.025pp
  (was +0.656pp); the switch tracks the current gap instead of dragging
  all-history averages.
- FIRST CAP-LEVEL WIN of the series: at 1536 DecayFit (0.809308) beats both
  standalones (s3fifo 0.808347, lcnu lower) by 0.096pp — running the LCNU
  prefix before switching to S3FIFO is worth more than either fixed policy.
- The preregistered oscillation risk realized at exactly the flagged cap:
  1024 (LCNU overall leader) exhibits local rate inversions near the 0.5pp
  margin → 8 switches, +0.155pp. The margin that fixes 4096 (needs to be
  < 1.09pp sustained gap) is small enough to be crossed repeatedly at 1024.
- Fixed-margin arbitration is now refuted in BOTH directions: too large →
  late switch (exp665 @4096); too small → oscillation (exp666 @1024). The
  margin band that satisfies both caps simultaneously is narrow or empty;
  a margin schedule (or hysteresis with per-switch cost accounting) is the
  remaining untested variant.

**Artifacts:** results_real_{qwen3-30b-a3b,granite3.1-moe-3b}-decayfit.{csv,md},
policies_decayfit.py, analyze_exp666_decayfit.py, exp666-verdict.json
(programmatic from verdict JSON; worst deficits 0.155/0.112pp).
