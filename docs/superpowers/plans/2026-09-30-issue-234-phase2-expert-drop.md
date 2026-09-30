# Issue 234 Phase 2 — drop-on-miss expert skipping

Draft for review. This does not implement the policy. Merge is blocked on accepting the decision checkpoint below.

**Goal:** Under an explicit opt-in, skip non-resident routed experts instead of stalling on a fetch, then renormalize the surviving routing weights.

**Constraint:** `expert_drop_policy` defaults to `off`. With the default, masks, weights, fetches, and accuracy stay identical to current main.

## Decision checkpoint (proposed)

The issue's preliminary recommendation was an NAEE ratio rule: drop expert *i* when `w_i < β · w_1`, only if it is not GPU-resident. The measurement comment on #234 (OLMoE-1B-7B, 24 prompts × 128 tokens) shows that rule does not have the assumed headroom:

- At β ≤ 0.3 the rule almost never fires on misses (eligible fraction 0.6–27%).
- At β = 0.5, dropped routing mass is highest where the miss rate is highest (~23% of mass at cache capacity 8) and lowest where misses are rare.

This draft therefore does **not** adopt a bare ratio threshold.

Proposed policy, to accept or replace before any runtime code:

1. `expert_drop_policy`: `off` (default) or `on_miss`. No unconditional `threshold` mode in this phase.
2. `on_miss` may remove a routed expert only when that expert is not GPU-resident.
3. Candidates are the lowest-weight selected experts. Drop them until the next drop would push the token's cumulative dropped weight above `expert_drop_mass_budget`, or until `min_k` experts remain. `min_k` is at least 1.
4. Shared experts are not in the routed top-k mask. This phase must not change the shared-expert module path (`shared_expert` parameters stay resident and always execute).
5. Renormalize by rescaling surviving post-softmax weights (`w /= w.sum()`). Do not re-softmax logits.
6. If the layer's selected weights are flat (`w_min / w_max` above `expert_drop_flatness_floor`), skip the rule for that layer and count a flatness bypass. This is the OLMoE guard.
7. The expert predictor and route-ahead prefetch still observe the pre-drop route, so a dropped expert is not starved out of future fetches.
8. `expert_drop_mass_budget` and `expert_drop_flatness_floor` ship unset-as-disabled until the three-point accuracy curve in the QA gate picks them. This draft does not choose numeric defaults.

## Where it runs

`DistributedExpertExecutor.dispatch_local` (`moe_infinity/distributed/expert_executor.py`) is the last Python seam before `set_inputs` and `enqueue_expert`. Apply a pure function to `router_mask` and `router_weights` there, after the route is known and before `set_inputs`. `enqueue_expert` then sees only survivors, so a dropped expert is not fetched.

There is no Python predicate today for "is this expert GPU-resident?" Residency is owned by `ExpertResidencyManager`. The first code task is a read-only query used only when the policy is on. The off path must not call it.

## File map

| File | Change |
|---|---|
| `moe_infinity/runtime/expert_drop.py` | Pure selection, mass budget, flatness bypass, renormalization. No dispatcher imports. |
| `moe_infinity/utils/config.py` | `expert_drop_policy` (`off`/`on_miss`), `expert_drop_min_k`, `expert_drop_mass_budget`, `expert_drop_flatness_floor`. Validate on parse. |
| `moe_infinity/distributed/expert_executor.py` | When policy is `on_miss`, rewrite mask and weights before `set_inputs`. Record drop counters. Prefetch still uses the original route. |
| Native residency query | Read-only "resident expert ids for this layer" binding. No admission or eviction change. |
| `tests/python/unit/test_expert_drop.py` | CPU tests listed below. |

## QA gate (from #234)

- Unit, CPU: surviving weights sum to 1; a token never drops below `min_k`; mass budget is a hard cap; flat input is a no-op; policy `off` returns the inputs unchanged.
- Default-config regression: unset policy does not alter dispatch.
- E2E, undersized cache, `on_miss` on: decode p99 latency improves at least 20% versus the same budget with the policy off. Export drop rate and a per-layer drop histogram.
- Accuracy: delta versus no-drop stays within the threshold agreed at this checkpoint. Record a threshold-versus-accuracy curve of at least three budgets in the implementation PR before any budget becomes the default.

## Non-goals

Activation quantization, permanent pruning, dropping resident experts, ACE or ASET score tables, and any #184 allowlist entry.
