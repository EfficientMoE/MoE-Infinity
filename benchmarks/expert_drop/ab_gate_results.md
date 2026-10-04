# Phase 3/5 real A/B gate: measured p99 + greedy fidelity for C/D (#234/#257)

C (head-budget) and D (adaptive-budget) are implemented in the fused C++ route
worker (commit adding them to `SelectExpertDrops`). This is the real two-model
gate: off vs shipped mass-0.20 vs C vs D, measuring decode p99 and greedy
next-token agreement vs the off baseline on the frozen 24×128 set, ratio 0.05,
`CUDA_VISIBLE_DEVICES=2`.

## Headline: the gate does not certify a winner, for two methodological reasons

1. **p99 is run-to-run noise-limited at ratio 0.05.** The OLMoE off p99 measured
   41.59 ms in the Phase-0 run and 55.68 ms in this gate run — same config,
   ~34% apart. D's p99 ratio came out 1.011 here while the shipped mass ratio
   came out 0.577 in the *same* gate run and 0.80 in Phase 0. A single 24×128
   run cannot resolve a policy's p99 effect below this noise floor.
2. **Free-running greedy agreement is a brittle fidelity metric here.** Any
   single early token flip makes the autoregressive context diverge, so
   agreement over 128 free-running tokens collapses to ~0 for *every* drop arm,
   regardless of budget — it cannot rank a mild policy against an aggressive one.

## Measured (OLMoE-1B-7B, this gate run)

| arm | p99 ms | p99 ratio | greedy agreement | experts_dropped | tokens_changed |
|---|---|---|---|---|---|
| off | 55.68 | 1.000 | — | 0 | 0 |
| mass-0.20 (shipped) | 32.15 | 0.577 | 0.011 | 92542 | 33912 |
| C (base 0.20, head 0.15) | — (crash) | — | — | — | — |
| D (base 0.05, slope 0.05, miss0 2) | 56.28 | 1.011 | 0.013 | 66678 | 27477 |

## The fused drop path is correct; fidelity loss is budget-driven, not a bug

Decoded prompt-0 completions (greedy):

- **off**: `"The sky appears blue because of a phenomenon called Rayleigh
  scattering. When sunlight enters the Earth's atmosphere, it"` — coherent, correct.
- **mass-0.05** (low budget probe, GPU 3): `"The video player at the end of the.
  ? - Go ahead and fix-up on the."` — grammatical English, but already off-topic
  by token ~2.
- **mass-0.20**: `"permissible/ is all of history's Post-market is theajs ToTo
  classer he |"` — degenerate.

So the path is not corrupting state (low budget → valid English); dropping
non-resident experts at ratio 0.05 genuinely changes the greedy trajectory, and
the change grows with budget. Greedy free-running agreement therefore measures
"did the trajectory ever diverge" (almost always yes by token ~0–2), not the
per-step quality of the policy.

## Policy C currently crashes in the fused forward

The C arm aborts mid-decode with
`RuntimeError: The specified pointer resides on host memory and is not
registered with any CUDA device`, raised from the native MoE forward
(`olmoe ... decoder_layer -> expert dispatch`). mass and D at the same budget do
not crash, so it is specific to C's extra single high-mass drop — most likely
when C empties the non-resident survivor set for a token and the exec/combine
path hits an edge case. C is also a **non-lever** in the offline sim (no p99
movement over the base mass drop at budget 0.20), so it is deprioritized pending
a dispatcher-side fix.

## C++ ↔ offline-sim drop-count divergence (expected)

The live C++ D run dropped 66678 experts / 27477 tokens-changed; the offline
`policy_sim` D (slope 0.05) predicted ~42029 tokens-changed. These differ
because the offline sim applies D to the **off run's** residency snapshot, while
the live policy perturbs the cache as it drops, so live residency diverges from
the off trace. The selector *logic* is parity-checked separately (the Phase-2
`verify_parity` asserts the base drop and C(0)/D(0) match `select_expert_drops`
bit-for-bit); only the residency context differs.

## Conclusions and required methodology fixes

- **Do not certify any tail-drop policy on this setup as-is.** The shipped
  mass-0.20 itself fails greedy fidelity and shows unstable p99; C/D inherit the
  same measurement problems.
- **Fidelity:** replace free-running greedy agreement with a drift-free metric —
  teacher-forced per-position argmax agreement (feed the off tokens as context,
  measure the policy's next-token match) and/or mean/p99 logit-softmax KL. This
  isolates per-step policy quality from autoregressive divergence.
- **Latency:** stabilize p99 with repeated runs (≥3) and more ITL samples, or a
  controlled closed-loop, before comparing policies; the current ±34% run-to-run
  spread exceeds the policy effect.
- **Policy C:** fix the fused-forward edge case (empty non-resident survivor set)
  before any C evaluation; its offline benefit is ~zero regardless.
- **Policy D:** implemented and stable (no crash); its real value is unresolved
  until the two methodology fixes above are in place.

## Artifacts

`results/gate_olmoe_{off,mass,D}.json`, `results/fid_olmoe_{mass,D}.json`,
and the Qwen3-30B arms (`gate_qwen_{off,mass,D}.json`,
`fid_qwen_{mass,D}.json`) from the same gate; probe outputs in
`/tmp/opencode/probe_*.json`.
