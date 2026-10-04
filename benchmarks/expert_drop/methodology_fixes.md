# Methodology fixes (next-steps) — teacher-forced fidelity + p99 repeats (#234/#257)

Follow-up to `ab_gate_results.md`, which flagged two measurement problems. Both
are now addressed, and the **fidelity conclusion changes materially**.

## Fix 1 — drift-free (teacher-forced) fidelity

`teacher_forced_fidelity.py` feeds the off baseline's greedy tokens back as
forced decode context, scoring every position on the *same* context. This
removes the autoregressive drift that made free-running greedy agreement
collapse to ~0 for any policy. Metrics over the frozen 24×128 set:

| model | arm | teacher-forced agreement | nll/tok | free-running agreement (old) |
|---|---|---|---|---|
| OLMoE-1B-7B | off (control) | 0.980 | 0.30 | — |
| OLMoE-1B-7B | mass-0.20 | **0.091** | 7.30 | 0.011 |
| Qwen3-30B-A3B | off (control) | 0.993 | 0.20 | — |
| Qwen3-30B-A3B | mass-0.20 | **0.934** | 0.25 | 0.187 |

**Headline correction.** The off controls (0.98 / 0.99) confirm the harness and
expose a ~1–2% offload nondeterminism floor. With drift removed:

- **Qwen3-30B tolerates mass-0.20 well — 0.934 per-step agreement.** Its earlier
  near-zero free-running agreement was *pure drift* (0.934¹²⁸ ≈ 0), not real
  degradation. Drop-on-miss is **viable on the large model** — the realistic
  offload target.
- **OLMoE does not — 0.091 per-step.** On the small model (64 experts) dropping
  ~20% of routing mass genuinely flips ~91% of next-token argmaxes.

So tail-drop fidelity is strongly **model-dependent**, and the viable regime is
the large-MoE regime. Free-running greedy agreement is unusable here; use
teacher-forced agreement + nll as the fidelity metric going forward.

## Fix 2 — p99 is run-to-run noise-limited (repeat runs)

OLMoE off decode p99 across independent runs: **36.7, 40.0, 41.6 (Phase-0),
55.7 (gate)** → ~40–50% spread with no config change; mass-0.20: 32.1, 35.2,
33.2. The policy effect (~0.6–0.8× for mass) is comparable to or below this
noise on a single 24×128 run. Qwen off similarly spans 218.5↔237.8.

Consequence: **single-run p99 ratios cannot rank policies.** Median over repeat
runs (`results/nextsteps_summary.json`, OLMoE ×3 / Qwen ×2):

| model | off | mass-0.20 | D (base0.05, slope0.05) |
|---|---|---|---|
| OLMoE median p99 ms | 40.0 | 35.2 (0.88×) | 35.2 (0.88×) |
| Qwen3-30B median p99 ms | 181.4 | 171.2 (0.94×) | 205.2 (1.13×) |

With stable medians, on Qwen **mass-0.20 modestly helps p99 (0.94×) while D does
not (1.13×, tight 2.2% spread — a real non-improvement, not noise)**: D's
base-0.05 budget barely drops on most tokens and the extra cache churn offsets
the rest.

## Overall verdict (both methodology fixes applied)

No tail-drop policy tested meets the #257 target `p99 ≤ 0.8×off` at acceptable
fidelity under stable measurement:

- **OLMoE**: mass/D reach ~0.88× p99 but teacher-forced fidelity is 0.09 —
  unacceptable (small model, too little redundancy).
- **Qwen3-30B**: fidelity is fine (mass 0.934) but p99 only reaches 0.94× (mass)
  or 1.13× (D) — misses the 0.8 target.

Drop-on-miss, measured properly, trims p50/p90 and gives a *modest* p99 help on
large models at good fidelity, but does **not** hit the aggressive p99 tail
target. This matches #257's own gate result and the menu doc's "moving p99 is
hard"; the remaining lever is a larger-expert-model re-test (checkpoint option
(c)), not more drop-policy tuning.

## Policy-C crash + D teacher-forced stall — root-caused and FIXED

Two guesses were wrong before the real cause was traced. (1) "Zero-fetch edge
case" — **falsified by data**: the shipped mass-0.20 empties a token's
non-resident set for **60.9% of OLMoE tokens (28901/47466) and never crashes**
(C 61.6%, D 23%), so emptying it is the normal, safe case. (2) "Async race" —
also wrong.

**Actual root cause (traced).** The fault surfaces at `wait_expert()` as a
worker-thread exception (`pointer resides on host memory and is not registered
with any CUDA device`) rethrown from the combine. When an expert is taken by the
**host-staging / overflow fallback** it executes on CPU, so `OutputFunc` sets
`out_gpu_id < 0` and puts the output tensor on CPU (line ~1594) — but the combine
`final_hidden_states_.add_(output * router_weight_[...])` never co-located it
with the CUDA accumulator. Aggressive drops (C targets the sole non-resident;
D's adaptive budget) change cache/overflow dynamics enough to push experts onto
the host-staging path and trip this latent bug; the shipped mass drop hits it
rarely. It is a **missing device co-location, not a race**.

**Fix (shipped, rebuilt `dev43`).** In `OutputFunc`, move the expert output to
`final_hidden_states_.device()` before the combine when they differ (strict
no-op on the common same-device path, so the working off/mass paths are
unaffected by construction). Verified:

- **C free-running: no longer crashes** (OLMoE C ran clean, p99 33.2, dropped 2781).
- **D teacher-forced: no longer stalls** (exit 0; OLMoE tf_agreement 0.075, nll
  7.8 — low, as expected for an aggressive drop on the small model, in line with
  mass 0.091). This confirms the C crash and the D-TF stall were the *same* bug.

**Residual:** C *teacher-forced* still stalls on the off-token stream (a
separate, slower hang). C is a non-lever (≈zero offline p99 benefit over the base
mass drop), so it stays deprioritized; the combine fix is the substantive
correctness improvement and also hardens any `out_gpu_id < 0` / cross-device
output path for the shipped runtime.

## Revised recommendation

1. **Target large MoE models.** On Qwen3-30B, mass-0.20 keeps 0.934 teacher-forced
   fidelity — tail-drop is viable there; OLMoE is the wrong model to judge it on.
2. **Fidelity metric**: teacher-forced agreement + nll (done); drop free-running
   greedy agreement.
3. **Latency**: compare policies on **median-of-≥3** p99 only.
4. **Fix the zero-fetch edge case** (above) — unblocks C and aggressive D, and is
   a latent correctness issue in the shipped fused path.
5. **D**: re-evaluate at lower slope on Qwen with teacher-forced fidelity +
   median p99 once the edge case is fixed (D free-running already runs clean;
   only D-TF hit the stall).

## Artifacts

`teacher_forced_fidelity.py`; `results/tf_{olmoe,qwen}_{off,mass}.json`;
`results/rep_{olmoe,qwen}_{off,mass,D}_*.json`; `results/nextsteps_summary.json`
(median/spread bands, written by the repeat-run job).
