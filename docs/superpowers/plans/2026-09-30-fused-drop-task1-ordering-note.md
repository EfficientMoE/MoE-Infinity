# Task 1 decision note: corrected-weight write-back ordering

Plan: `2026-09-30-dispatcher-fused-expert-drop.md`. All references are to
`core/parallel/expert_dispatcher.cpp` at `proto/expert-drop-device-select`
head (8822220).

## Where the decision actually runs

`DispatchExperts` (1809) stages `active_flags` into pinned memory and
schedules `RouteReadyCallback` via `cudaLaunchHostFunc` (1855). The callback
(1945) does **no routing work**: it only pushes `RouteArgs` onto
`route_queue_`. The walk + `EnqueueExpert` loop runs in **`RouteFunc`
(1974)** — a dedicated, ordinary worker thread. This matters: CUDA host-func
callbacks may not issue CUDA API calls, but `RouteFunc` may. The fused drop
selection therefore lives in `RouteFunc`, after the flags/weights have
landed and before the `EnqueueExpert` loop (2004).

## How exec consumes weights

Exec workers combine per expert by reading the live GPU tensor:
`final_hidden_states_.add_(output * router_weight_.index({Slice(),
expert_idx}))` (1606-1615). Ordering today: miss-path `ExecArgs` carry
`transfer_event` recorded after the fetch H2D (1372) and `GPUExecFunc` does
`cudaStreamWaitEvent` on it (1459-1461); cache-hit path sets
`transfer_event = nullptr` (651, 685) and waits nothing.

## Mechanism (a) — event-ordered async H2D: rejected

Recording one `weights_done` event and having every exec wait it fails on
the cache-hit path: `transfer_event` is released back to the pool after the
first wait (1461), so a single shared event double-releases, and per-expert
events multiply pool traffic. Threading the wait transitively through the
fetch stream covers only the miss path. Workable, but it adds event
plumbing to three code paths for no gain — see (a').

## Mechanism (b) — per-row scale via ExecArgs: rejected

The renorm factor is per **row** (1 / surviving float32 sum), not per
expert. For rows > 1 the combine would need a per-row scale vector on the
GPU at accumulate time — which is itself an H2D upload, i.e. mechanism (a)
with extra steps. Zeroing dropped weights still has to happen anyway
(dropped experts are simply not enqueued, but their weights must not leak
into any other consumer of `router_weight_`).

## RECOMMENDED: (a') synchronous write-back from RouteFunc

In `RouteFunc`, after `SelectExpertDrops` and **before** the
`EnqueueExpert` loop, issue one blocking `cudaMemcpy` (or
`copy_(..., /*non_blocking=*/false)`) of the corrected `[tokens ×
num_experts]` float32 block into `router_weight_`'s storage.

Ordering-safety argument:

1. The copy is host-blocking: when it returns, the corrected values are in
   `router_weight_`'s global memory.
2. Every exec kernel that reads `router_weight_` is launched by a
   `GPUExecFunc` consuming an `exec_queue_` entry pushed by
   `EnqueueExpert`/`Enqueue` — all of which happen **after** the copy
   returns, on the same `RouteFunc` thread (program order) with the queue
   push providing the cross-thread happens-before. Kernels launched after
   the copy completes observe the copied data. No events, no stream
   reasoning, no double-release hazard.
3. Cost: at OLMoE decode geometry the block is 1×64×4 B = 256 B (≤ 64 KB
   even at rows=64/experts=256); a blocking pinned H2D is ~10 µs — paid on
   the route worker thread, which is already off the decode critical path
   (the host returned from `DispatchExperts` long before).

Failure paths: if `SelectExpertDrops` reports any per-row guard trip it
fails open per row (reference semantics); if the write-back itself errors,
skip it and enqueue the PRE-drop expert set (fail-open to exactly today's
behavior). `FailDispatch` (1871) needs no changes: no new events or leases
exist, and a failed generation drops `RouteArgs` before routing
(1977-1980), i.e. before any selection or write-back.

One side effect to document in Task 3: `router_weight_` aliases the tensor
Python passed to `SetInputs` (1792), so the write-back mutates the
Python-visible weights. In fused mode the Python interception is retired
(plan Task 4) and nothing reads them post-dispatch; the Task 4 test must
pin this.
