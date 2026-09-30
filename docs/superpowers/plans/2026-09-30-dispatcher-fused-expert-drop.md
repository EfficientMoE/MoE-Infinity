# Dispatcher-fused expert drop (#234 Phase 2, option (b))

## Problem

`expert_drop_policy=on_miss` (PR #257) fails its latency gate (`p99_on <= 0.8 * p99_off`) because the drop decision runs in Python on the decode critical path:

| Implementation | rows=1 per-layer cost | OLMoE E2E p99 on/off |
|---|---|---|
| Host loop (PR #257) | ~48–150µs blocking (D2H sync + Python loop) | 1.10 (FAIL) |
| Device-vectorized + torch.compile (`proto/expert-drop-device-select`) | ~51µs enqueue + ~100µs integration overhead | 1.23 (FAIL) |

Measured on RTX PRO 6000, `device_memory_ratio=0.05`, budget 0.05, 24 prompts × 128 greedy tokens. Evidence: PR #257 thread comments r4144789416 and r4145094640.

**Root cause:** at rows=1 (single-request decode, the gate's scenario) any Python-launched selection adds more overhead than dropping saves. The decision must cost ~0 on the critical path.

## Key architectural insight (verified in source)

`ExpertDispatcher::DispatchExperts` (core/parallel/expert_dispatcher.cpp:1809) already runs an **asynchronous route pipeline**:

1. Device: `router_mask_.any(0)` → `active_flags` (GPU)
2. Async D2H copy into pinned `active_flags_host`
3. `cudaLaunchHostFunc` → `RouteReadyCallback` (line 1945) fires on a host callback thread when the copy lands, walks `active_flags_host`, and calls `Enqueue` per active expert

The callback is **off the decode critical path** (host thread, gated only by the stream reaching the copy). Residency is available lock-free via `node->resident_on_gpu` (atomic, added in #257 review fixes, line 944). Expert outputs are combined at exec time by reading `router_weight_` per expert (line 1608): `final_hidden_states_.add_(output * router_weight_[:, expert_idx])`.

**Fusion design:** extend the same pipeline. Copy the (tiny) weights row block down with the flags, make the drop decision in the callback in plain C++ (~1µs for 64 experts), enqueue only survivors, and push renormalized weights back before exec reads them. Zero extra syncs, zero Python.

## Non-goals

- No change to the eager fallback path (`_dispatch_eager_local`): the Python host-loop selection remains for non-native routing. Document that the gate targets the native path.
- No change to default behavior: `expert_drop_policy=off` stays byte-identical (object-identity tests must stay green).
- No accuracy work: NLL/quality at various budgets remains the separate #234 accuracy checkpoint.
- The `proto/expert-drop-device-select` branch is evidence only; it is not merged by this plan.

## Tasks (strict order)

### Task 1 — Investigation spike: weight write-back ordering (timeboxed)

The single technical risk: corrected weights must be visible to exec workers before they read `router_weight_.index(...)`.

- Read `RouteReadyCallback` → `Enqueue` → `GPUExecFunc`/exec-queue consumption to establish the exact ordering primitives (transfer events, stream waits) between route callback and exec streams.
- Decide between:
  - (a) async H2D of corrected weights from the callback + `cudaEventRecord` on the route stream, exec streams `cudaStreamWaitEvent` before first `router_weight_` read (reuses the existing transfer-event mechanism), or
  - (b) per-expert scalar correction passed through `ExecArgs` (no tensor write-back; combine multiplies by an extra factor).
- Deliverable: a one-page decision note in the PR description with the chosen mechanism and the ordering proof (which event, which stream, why no race). **Checkpoint: do not start Task 3 until this is written.**

### Task 2 — Pure C++ drop selector with exact reference parity

- New `core/parallel/expert_drop_select.h/.cc`: `SelectExpertDrops(rows, num_experts, const float* weights, const uint8_t* mask, const uint8_t* resident, params) -> DropDecision` (dropped ids per row, corrected weights, counts).
- Semantics copied exactly from `moe_infinity/runtime/expert_drop.py::select_expert_drops`: float64 cumulative budget (`cum <= budget`), `min_k` floor, flatness bypass strict `w_min/w_max > floor`, tie-break lower-weight-then-higher-id, renormalize survivors by their sum (skip if sum == 0), fail-open on non-finite/negative/zero-max rows.
- Tests: C++ gtest in the existing `build-cpp-tests` harness AND a Python parity test that feeds identical random cases (≥200, seeded, geometry sweep) to the C++ selector via a test-only binding and to the Python reference, asserting exact mask/count equality and weight allclose. The randomized corpus from `tests/python/unit/test_expert_drop_device.py` is the template.

### Task 3 — Fuse into the route pipeline

- `DispatchExperts`: when policy is on, extend the D2H staging to include the routed weight rows (pinned, same stream, same copy batch — [tokens × num_experts] float32; 256 B at OLMoE decode).
- `RouteReadyCallback`: after flags land, snapshot residency via `resident_on_gpu` atomic loads, run `SelectExpertDrops`, then:
  - enqueue only surviving experts (dropped experts never reach `Enqueue`);
  - apply the Task-1 weight-correction mechanism;
  - record counts into new atomics (`drop_tokens_seen_`, `drop_tokens_changed_`, `drop_experts_dropped_`, `drop_flatness_bypasses_`, per-layer array).
- Config plumbing: `SetExpertDropPolicy(min_k, mass_budget, flatness_floor, enabled)` binding called from `model_offload.py` at init from the existing `ArcherConfig` fields. Policy off ⇒ the staging copy of weights is not even issued (byte-identical path).
- Stats binding `get_expert_drop_stats()` returning the atomics; Python `DistributedExpertExecutor.get_expert_drop_stats` merges native stats when native routing is active.

### Task 4 — Retire the Python interception on the native path

- `_apply_expert_drop` becomes a no-op when native routing will handle the layer (native path + policy on): Python must not double-drop. The eager fallback path keeps the existing host-loop selection unchanged.
- Tracing contract: `correct_prefetch(layer+1)` must keep receiving the PRE-drop union. In the fused path the pre-drop union is exactly `active_flags_host` before filtering — expose it through `take_last_active_experts`-style accessor (`take_last_routed_experts`) and use it for the correction ids; post-drop survivors remain what `take_last_active_experts` returns. Add a unit test pinning: correction ids == pre-drop union, dispatched ids == post-drop survivors.
- Keep all existing #257 unit tests green, including policy-off object identity.

### Task 5 — Gate re-run and decision

- Same harness (`benchmarks/expert_drop/ab_latency_olmoe.py` from the proto branch, promoted into this PR), same settings: OLMoE, ratio 0.05, budgets {0.05, 0.10, 0.20}, 24 prompts × 128 greedy tokens, ≥2.9k ITL samples/arm.
- Report off vs fused-on p50/p90/p99 per budget on the PR.
- Success: `p99_on <= 0.8 * p99_off` at ≥1 budget. If the gate still fails, record the result on #234 and stop — the remaining lever is checkpoint option (c) (large-expert model re-test), not more engineering here.

## Verification gates (every task)

- `pytest tests/python/unit/test_expert_drop*.py tests/python/unit/test_utils_config.py tests/python/unit/test_resident_on_gpu_binding.py tests/python/unit/test_gpu_only_expert_routing.py` green.
- C++ built with the sm_120 in-place build; gtest suite green; `-fsyntax-only` is not sufficient for Tasks 3–4 (real build + runtime smoke required because cudaLaunchHostFunc paths cannot be syntax-checked meaningfully).
- pre-commit (ruff 0.6.9, clang-format) clean.
- Policy-off byte-identity: existing object-identity tests plus one E2E smoke diff (same seed, off vs unbuilt-policy binary → identical token ids).

## Branching

- New branch `feat/expert-drop-dispatcher-fused` from `feat/expert-drop-on-miss` (after #256 merges and #257 rebases, from #257's successor). Stacked PR referencing #234 and superseding the enabled-path mechanics of #257; #257's config surface, probe, stats API, and tests carry over.

## Risks

1. **Weight write-back race** (Task 1). Mitigation: decision-note checkpoint before implementation; prefer mechanism (b) (ExecArgs scalar) if event ordering is not provably safe.
2. **Callback-thread budget**: `cudaLaunchHostFunc` callbacks serialize on a runtime thread; the selector must stay allocation-free and O(rows × experts) (~µs). Mitigation: gtest includes a wall-time bound assertion at OLMoE geometry.
3. **TOCTOU residency drift**: residency may change between snapshot and enqueue. Same semantics as #257 (decision is a hint; `Enqueue` handles both hit and miss states). No new mechanism required — document it.
4. **Batched serving path**: rows>32 decode via serving engine also flows through `DispatchExperts`; the C++ selector is O(rows×experts) and stays off-critical-path, so no special-casing — but the gtest sweep must include rows=64.

## Estimated shape

- Task 1: half day (investigation + note)
- Task 2: 1 day (selector + parity corpus)
- Task 3: 1–2 days (pipeline fusion + stats)
- Task 4: half day (interception removal + tracing tests)
- Task 5: half day (gate runs + reporting)
