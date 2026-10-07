# Phase 2 policy simulation + coarse Pareto (#234/#257)

`policy_sim.py` runs the six candidates from the decision menu on the captured
off-policy traces. **Parity guard (enforced each run):** the vectorized
`base_drop_np` matches the scalar `select_expert_drops` on 4000 sampled rows per
budget, and `C(head_budget=0)` / `D(slope=0)` are bit-identical to the shipped
mass drop.

## Metric caveats

- **`pred_p99` is REFERENCE ONLY.** It comes from the off-calibrated cost model,
  which passed the ±10% gate on the *off* arm but **failed it on tail-acting
  arms** (see `cost_model_calibration.md`). Use the p99 columns to rank/triage,
  not as absolute predictions. Real p99 must come from implementation.
- **`tv_mean` (routing total-variation vs the pre-drop distribution) is
  cost-model-independent** and is the trustworthy offline fidelity proxy for
  ranking. True greedy next-token agreement still requires the real A/B gate.
- **E is approximate**: the Phase-0 trace has no per-fetch arrival timing, so E's
  "which experts arrived by the deadline" is modeled from a per-fetch-time floor,
  not measured stragglers. Treat E rows as illustrative only.

## Results (p99 ratio vs off = reference; tv_mean = fidelity cost, lower better)

### OLMoE-1B-7B (off p99 ref 38.6 ms)

| policy | knob | p99 ratio | tv_mean | frac dropped |
|---|---|---|---|---|
| mass | 0.05 / 0.10 / 0.20 | 0.907 / 0.816 / 0.699 | 0.071 / 0.156 / 0.285 | 0.12 / 0.23 / 0.34 |
| C | head_budget 0.05…0.20 @ base 0.20 | 0.699 (flat) | ~0.285 | ~0.35 |
| D | slope 0.01 / 0.02 / 0.05 @ base 0.05 | 0.824 / 0.767 / 0.663 | 0.103 / 0.135 / 0.227 | 0.17 / 0.21 / 0.30 |
| A | L_us 2000 / 1000 / 500 @ base 0.20 | 0.699 / 0.699 / 0.668 | 0.285 / 0.285 / 0.294 | ~0.35 |
| B | redirect_min 0.05 / 0.10 / 0.20 | 0.642 / 0.653 / 0.658 | 0.389 / 0.379 / 0.370 | 0.41 / 0.40 / 0.39 |
| E | L_us 300 / 150 | 0.896 / 0.653 | 0.012 / 0.140 | 0.02 / 0.20 |

### Qwen3-30B-A3B (off p99 ref 199.4 ms)

| policy | knob | p99 ratio | tv_mean | frac dropped |
|---|---|---|---|---|
| mass | 0.05 / 0.10 / 0.20 | 0.979 / 0.866 / 0.730 | 0.005 / 0.065 / 0.147 | 0.01 / 0.13 / 0.26 |
| C | head_budget 0.05…0.20 @ base 0.20 | 0.730 (flat) | ~0.148 | ~0.26 |
| D | slope 0.01 / 0.02 / 0.05 @ base 0.05 | 0.860 / 0.781 / 0.605 | 0.059 / 0.091 / 0.203 | 0.11 / 0.17 / 0.33 |
| A | L_us 2000 / 1000 / 500 @ base 0.20 | 0.529 / 0.276 / 0.149 | 0.239 / 0.472 / 0.615 | 0.34 / 0.54 / 0.66 |
| B | redirect_min 0.05 / 0.10 / 0.20 | 0.603 / 0.608 / 0.683 | 0.603 / 0.580 / 0.324 | 0.65 / 0.62 / 0.36 |
| E | L_us 600 / 300 / 150 | 0.655 / 0.276 / 0.149 | 0.104 / 0.397 / 0.549 | 0.18 / 0.54 / 0.66 |

## Findings

1. **C (head-budget hybrid) is a non-lever** at base budget 0.20 on both models:
   p99 and fidelity are indistinguishable from the plain mass drop (the "sole
   non-resident, mass ≤ head_budget" condition almost never fires once the base
   budget has already trimmed the tail). C may help only at a *low* base budget;
   as specified it does not move p99.
2. **D (residency-adaptive budget) is the most fidelity-efficient lever** among
   the low-complexity options: it reaches a lower p99 ratio than the mass drop at
   a given tv (Qwen 0.605 @ tv 0.203 vs mass-0.20 0.730 @ tv 0.147; D spends
   fidelity only on high-miss tokens). Stays inside the fused worker, no new
   state — the plan's "D" complexity profile holds.
3. **A (deadline high-mass) and E (partial-wait) are the strongest p99 levers**
   but at steep fidelity cost (tv 0.4–0.6 on Qwen) and higher complexity (A needs
   a fetch-cost model; E reopens the wait/combine path). A is far stronger on
   Qwen than OLMoE because Qwen's larger fetch term makes the projected critical
   path cross the deadline more often.
4. **B (resident redirect)** reaches low p99 but perturbs routing the most
   (highest tv) except at a high redirect threshold; mass conservation does not
   translate into low routing distortion because mass is concentrated onto a
   single resident expert.

## Shortlist decision (feeds the Phase-1 fork → real implementation)

Per the fork (measure p99 for the cheapest shortlist by real implementation):

- **Implement D (adaptive-budget) first** — best low-complexity p99/fidelity
  tradeoff, zero new state, stays zero-sync in the fused worker.
- **Implement C alongside** (plan-mandated cheap pair; trivial one-branch add-on)
  but expect little movement unless paired with a lower base budget; keep it as a
  bounded safety lever.
- **Hold A/B/E** as escalations only if D's *measured* fidelity is unacceptable
  at its p99 target. A is the natural escalation (strongest p99) if a fetch-cost
  model is accepted; E only if a hard per-layer latency cap becomes a product
  requirement.

Next: TDD the C and D selectors in the fused C++ route worker against this
Python simulator as the reference, then run the real two-model A/B gate to get
measured p99 + greedy next-token agreement (the metrics the offline model cannot
certify).
