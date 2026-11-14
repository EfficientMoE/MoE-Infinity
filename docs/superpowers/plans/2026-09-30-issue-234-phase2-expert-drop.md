# Issue 234 Phase 2 — drop-on-miss expert skipping

Draft for review. This is the spec. The task list is `docs/superpowers/plans/2026-09-30-expert-drop-on-miss.md`. Merge is blocked on accepting the decision checkpoint below.

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
6. Flatness is per token, over that token's selected weights. If `w_min / w_max > expert_drop_flatness_floor`, skip the token and count a flatness bypass. This is the OLMoE guard. Equality with the floor does not bypass.
7. This step's `set_inputs`, `enqueue_expert`, `dispatch_experts`, and the current-layer route-ahead pin use the post-drop mask, so a dropped expert is not fetched for this execution. `correct_prefetch(layer + 1)` receives the pre-drop expert union so the id still counts toward the next layer. `correct_to_native_route` receives the post-drop ids so the cache is not refilled with an expert this step refused to run.
8. Pinned config, copied by the implementation plan:
   - `expert_drop_policy`: `"off"` (default) or `"on_miss"`.
   - `expert_drop_min_k`: `1`. Values below 1 are rejected.
   - `expert_drop_mass_budget`: `0.0`, range `[0, 1]`. `on_miss` with budget `0` does not rewrite tensors. No non-zero budget is the default. The eval curve records budgets `0.05`, `0.10`, and `0.20` and does not fail on perplexity.
   - `expert_drop_flatness_floor`: `0.5`, range `[0, 1]`.
   - Tie break: lower weight first, then higher expert id.
   - A missing residency probe, an exception, or a vector whose length is not the expert count means every expert is treated as resident.
   - A selected weight that is non-finite or negative leaves that token unchanged.
   - The output mask is a subset of the input mask. False bits stay false.
   - Budget `0` returns the input tensors as the same objects.

## Where it runs

`DistributedExpertExecutor.dispatch_local` (`moe_infinity/distributed/expert_executor.py`) is the last Python seam before `set_inputs` and `enqueue_expert`. Apply a pure function to a copy of `router_mask` and `router_weights` there, and pass the rewritten tensors to `set_inputs` and to the enqueue path. The multi-rank `dispatch()` RPC path is unchanged: residency is local to the worker that owns the cache.

GPU residency is `expert_node->node->device.is_cuda()`, the same predicate as the cache-hit check in `ExpertDispatcher::Enqueue` (`core/parallel/expert_dispatcher.cpp` around the `cache_hit` assignment). Expose it as a read-only `resident_on_gpu(layer_idx)` binding. The off path must not call it.

## File map

| File | Change |
|---|---|
| `moe_infinity/runtime/expert_drop.py` | Pure selection, mass budget, flatness bypass, renormalization. No dispatcher imports. |
| `moe_infinity/utils/config.py` | `expert_drop_policy` (`off`/`on_miss`), `expert_drop_min_k`, `expert_drop_mass_budget`, `expert_drop_flatness_floor`. Validate on parse. |
| `moe_infinity/distributed/expert_executor.py` | When policy is `on_miss` and the budget is positive, rewrite mask and weights before `set_inputs`. Current-layer fetch uses the post-drop mask. Next-layer `correct_prefetch` uses the pre-drop union. |
| `core/parallel/expert_dispatcher.h` / `.cpp`, `core/python/py_archer_prefetch.cpp` | Read-only `resident_on_gpu(layer_idx)`. No admission or eviction change. |
| `tests/python/unit/test_expert_drop.py`, `test_expert_drop_executor.py`, `test_resident_on_gpu_binding.py`, `test_expert_drop_gates.py` | CPU tests for selection, dispatch wiring, the binding name, and the latency gate. |

## QA gate (from #234)

- Unit, CPU: surviving weights sum to 1; a token never drops below `min_k`; mass budget is a hard cap; flat input is a no-op; policy `off` returns the inputs unchanged.
- Default-config regression: unset policy does not alter dispatch.
- E2E, undersized cache, `on_miss` on: decode p99 latency improves at least 20% versus the same budget with the policy off. Export drop rate and a per-layer drop histogram.
- Accuracy: delta versus no-drop stays within the threshold agreed at this checkpoint. Record a threshold-versus-accuracy curve of at least three budgets in the implementation PR before any budget becomes the default.

## Non-goals

Activation quantization, permanent pruning, dropping resident experts, ACE or ASET score tables, and any #184 allowlist entry.
