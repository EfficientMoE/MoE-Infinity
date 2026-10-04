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

Consequence: **single-run p99 ratios cannot rank policies.** Report median over
≥3 runs (the full bands land in `results/nextsteps_summary.json` from the
repeat-run job; OLMoE ×3, Qwen ×2). D's earlier "no benefit / 1.2× worse" single
points are within this noise and are not evidence against D.

## Unified root cause for the Policy-C crash and the D teacher-forced stall

Both aggressive levers hit the **same zero-fetch edge case**: when a drop leaves
a token with *only resident* experts (zero non-resident to fetch), the fused
dispatch path misbehaves — **C crashes** (`pointer resides on host memory ...`,
C targets exactly the sole non-resident) and **D stalls** under teacher forcing
(route-wait never satisfied because zero fetches were enqueued). The shipped
mass drop rarely empties the non-resident set (it trims the low-mass tail), so
it does not trip this. This is the concrete bug to fix before C or aggressive-D:
handle the zero-non-resident-survivor case in `ApplyFusedExpertDrop` / the
route-wait + combine path (skip the fetch-wait and the host→device weight
write-back when no survivor needs a fetch).

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
