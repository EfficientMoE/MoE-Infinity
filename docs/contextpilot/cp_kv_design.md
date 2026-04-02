# Phase C CP-Aware KV Management Interface Design

## Why Phase C (what Phase B did not cover)

Phase B integrated ContextPilot before tokenization and improved TTFT by roughly 20% in shared-prefix workloads. It primarily optimized prompt layout and token dedup but did not let scheduler and KV allocator query CP index signals directly.

This leaves two gaps:

1. Scheduling gap: requests are still admitted mostly by queue order and memory pressure handling, not CP-predicted reuse potential.
2. Allocation gap: KV allocation in both serving paths (`PagedKVCache` in v2 and `KVCacheManager` in native) does not consume a unified CP API for reuse-oriented decisions.

Phase C addresses both by introducing a stable CP-aware abstraction for:

- predicted prefix reuse scoring,
- CP block metadata retrieval,
- CP notifications on allocation/free,
- and scheduling priority ranking.

Given `phase_b_comparison.json`, shared-prefix TTFT already improved by ~21.18% vs baseline. Phase C is designed to push that further toward the 30-35% target by making scheduler/allocator behavior CP-driven rather than only pre-tokenization optimized.

## Interface design rationale

`CPAwareKVManager` in `moe_infinity/serving/cp_kv_interface.py` defines five methods:

1. `predict_prefix_reuse(request_id, token_ids) -> float`
   - Standardized score in [0.0, 1.0].
   - Enables path-agnostic admission/scheduling decisions.

2. `get_cp_cached_blocks(request_id) -> list[int]`
   - Exposes CP-known block hashes for reuse hints.
   - Allows path-specific allocators to map hash hints to concrete block IDs.

3. `notify_blocks_allocated(request_id, block_hashes) -> None`
   - Keeps CP index synchronized after allocation/registration events.

4. `notify_blocks_freed(request_id, block_hashes) -> None`
   - Keeps CP index synchronized on true deallocation.
   - Must follow Task 6 lifecycle semantics: terminal frees only trigger removal behavior; swap events are no-op.

5. `get_allocation_priority(request_ids) -> list[str]`
   - Returns ordering from highest to lowest predicted overlap.
   - Supports batch-level scheduling decisions without forcing scheduler to know CP internals.

Two concrete implementations:

- `NullCPAwareKVManager`
  - No-op, deterministic fallback when CP is disabled/unavailable.
  - Guarantees zero behavior change to current schedulers when feature is off.

- `ContextPilotKVManager`
  - Delegates to middleware methods when available.
  - Provides safe defaults and clamps scores into [0.0, 1.0] to protect scheduler invariants.

## How CPAwareKVManager connects to both scheduler paths

### v2 serving path (`moe_infinity/serving/scheduler.py` + `serving/kv_cache.py`)

- Scheduler phase:
  - Use `get_allocation_priority` on waiting request IDs before admission loop.
  - Optionally use `predict_prefix_reuse` to break ties under token/block budget pressure.

- KV phase:
  - On sequence allocation and block growth, compute/track block hashes and call `notify_blocks_allocated`.
  - On terminal free (`free_sequence` via completed/abort paths), call `notify_blocks_freed`.
  - On swap-out/in, do not call terminal-free semantics (no-op for CP removal path).

### native path (`moe_infinity/engine/scheduler.py` + `memory/kv_cache_manager.py`)

- Scheduler phase:
  - Apply `get_allocation_priority` to waiting queue before `_allocate_with_prefix_cache` attempts.
  - Use `predict_prefix_reuse` as additional signal when deciding victims or ordering under preemption pressure.

- KV/prefix-cache phase:
  - Existing block-hash path already computes `hash_block_tokens`; integrate those hashes with `notify_blocks_allocated`.
  - On terminal `finish_request/abort_request` free path, call `notify_blocks_freed`.
  - Preserve Task 6 semantics for non-terminal deallocation fallbacks (`defer`, not remove).

## Data-flow diagram (ASCII)

```text
            +-----------------------------+
Incoming -> | Scheduler (v2/native path)  |
requests    +-----------------------------+
                       |       ^
                       |       |
        get_allocation_priority | predict_prefix_reuse
                       v       |
             +-------------------------+
             | CPAwareKVManager        |
             |  - NullCPAwareKVManager |
             |  - ContextPilotKVManager|
             +-------------------------+
                       |
                       | delegates (if CP enabled)
                       v
             +-------------------------+
             | ContextPilotMiddleware  |
             +-------------------------+
                       |
                       v
             +-------------------------+
             | ContextPilot index      |
             +-------------------------+

KV events:
  allocate/register ----> notify_blocks_allocated(...)
  terminal free    -----> notify_blocks_freed(...)
  swap in/out      -----> no-op for terminal eviction
```

## Risks and mitigations

1. Risk: score instability or out-of-range values from middleware
   - Mitigation: clamp score to [0.0, 1.0] in `ContextPilotKVManager`.

2. Risk: missing middleware capabilities in mixed deployments
   - Mitigation: capability probing (`getattr` + callable checks) and deterministic fallbacks.

3. Risk: semantic mismatch between swap and terminal free
   - Mitigation: enforce Task 6 lifecycle policy; only terminal free paths map to CP removal semantics.

4. Risk: path divergence (v2 vs native) if each integrates ad hoc
   - Mitigation: both paths consume the same `CPAwareKVManager` API, isolating CP specifics in one adapter layer.

5. Risk: regression when CP disabled
   - Mitigation: `NullCPAwareKVManager` identity ordering and zero-score behavior keep baseline scheduler behavior intact.
