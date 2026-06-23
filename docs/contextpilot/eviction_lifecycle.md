# KV Block Eviction Lifecycle

## State Machine

```text
                (allocate_sequence / set_block_table)
                           |
                           v
                     +-----------+
                     | ALLOCATED |
                     +-----------+
                           |
                           | prefill/decode append
                           v
                     +-----------+
      swap_in -------|  ACTIVE   |------- swap_out (recoverable) ------+
        (no-op)      +-----------+                                      |
                           |                                            |
                           | finish/abort (terminal free)               |
                           v                                            |
                      +---------+                                       |
                      |  FREED  |<------------------+                   |
                      +---------+                   |                   |
                                                    |                   v
                                               +-----------+      +-----------+
                                               |  SWAPPED  |------| (CPU KV)  |
                                               +-----------+      +-----------+
```

Interpretation for ContextPilot (CP):
- `SWAPPED` is **recoverable** KV (CPU copy exists): CP action = **no-op**.
- `FREED` is **true deallocation**: CP action = **remove_requests()** only on terminal completion/abort.
- Non-terminal frees (allocation rollback / swap failure fallback) are real frees but request continues: CP action = **defer**.

## v2 Serving Path Events

| Event | File:Line | Function | Transition | CP Action |
|---|---|---|---|---|
| Request enters scheduler queues | `moe_infinity/serving/scheduler.py:45-62` | `Scheduler.add_request` | status `WAITING` setup; no KV yet | no-op |
| Prompt KV allocation for prefill | `moe_infinity/serving/scheduler.py:116-121`; `moe_infinity/serving/kv_cache.py:215-225` | `Scheduler.schedule` → `PagedKVCache.allocate_sequence` | `ALLOCATED` + status `WAITING -> PREFILL` | no-op |
| Prefill/decode KV growth | `moe_infinity/serving/scheduler.py:155-161`; `moe_infinity/serving/kv_cache.py:226-234` | `Scheduler.update_after_step` → `PagedKVCache.append_tokens` | `ACTIVE` (adds blocks/tokens) + status `PREFILL -> DECODE` | no-op |
| Preemption swap-out (recoverable) | `moe_infinity/serving/scheduler.py:239-244`; `moe_infinity/serving/kv_cache.py:261-273`; `moe_infinity/serving/kv_cache.py:245-253` | `_preempt_oldest_running_group` + `swap_out` + `free_gpu_blocks` | `ACTIVE -> SWAPPED` (CPU buffer kept, GPU ids freed only) | no-op |
| Swap-in resume | `moe_infinity/serving/scheduler.py:274-291`; `moe_infinity/serving/kv_cache.py:274-300` | `_recover_swapped_groups` + `swap_in` | `SWAPPED -> ACTIVE` | no-op |
| Completion detected in engine callback path | `moe_infinity/serving/engine.py:187-191`; `moe_infinity/serving/engine.py:212-217` | `ContinuousBatchingEngine.step` | marks finished output, invokes token callback with `finished=True`, removes callback | no-op (observation only) |
| Terminal free on completion | `moe_infinity/serving/scheduler.py:169-177`; `moe_infinity/serving/kv_cache.py:235-244`; `moe_infinity/serving/kv_cache.py:121-125` | `Scheduler.update_after_step` → `PagedKVCache.free_sequence` → `BlockTable.release` | status `(...)->FINISHED`, then `ACTIVE/SWAPPED -> FREED` | **remove_requests()** |
| Terminal free on abort | `moe_infinity/serving/engine.py:236-251`; `moe_infinity/serving/scheduler.py:180-192`; `moe_infinity/serving/kv_cache.py:235-244` | `ContinuousBatchingEngine.abort_request` → `Scheduler.abort_request` → `free_sequence` | status `(...)->CANCELLED`, then `ACTIVE/SWAPPED -> FREED` | **remove_requests()** |

### v2 callback answers (Task prompt questions)
- Completion callback: `ContinuousBatchingEngine.step` emits `RequestOutput(finished=True)` and runs user callback at `moe_infinity/serving/engine.py:212-217`.
- Abort callback: there is **no symmetric token callback** on abort; abort path directly calls `scheduler.abort_request` at `moe_infinity/serving/engine.py:250` and drops callbacks at `:251`.
- Preemption behavior: `Scheduler._preempt_oldest_running_group` uses `swap_out()` + `free_gpu_blocks()` (`serving/scheduler.py:239-243`), **not** `free_sequence()`.
- Free primitive: true release is `PagedKVCache.free_sequence()` (`serving/kv_cache.py:235-244`) which calls `BlockTable.release()` (`:121-125`).

## Native Engine Path Events

| Event | File:Line | Function | Transition | CP Action |
|---|---|---|---|---|
| Prefix-cache lookup and fresh block allocation | `moe_infinity/engine/scheduler.py:152-200`; `moe_infinity/engine/scheduler.py:166-170`; `moe_infinity/engine/scheduler.py:185-198` | `_allocate_with_prefix_cache` (uses `hash_block_tokens`) | `ALLOCATED` (mix of cached+new GPU blocks) | no-op |
| Request enters running | `moe_infinity/engine/scheduler.py:91-95` | `Scheduler.schedule` | status `WAITING -> RUNNING` (`ACTIVE`) | no-op |
| Preemption swap-out (normal path) | `moe_infinity/engine/scheduler.py:233-271`; `moe_infinity/memory/kv_cache_manager.py:144-179` | `_preempt_with_transfer` → `prepare_swap_out`/`commit_swap_out` | `ACTIVE -> SWAPPED` (GPU blocks released, CPU mapping retained) | no-op |
| Swap-in (normal resume) | `moe_infinity/engine/scheduler.py:272-317`; `moe_infinity/memory/kv_cache_manager.py:180-224` | `_swap_in_request` → `prepare_swap_in`/`commit_swap_in` | `SWAPPED -> ACTIVE` | no-op |
| Prefix-cache registration on completed request | `moe_infinity/engine/scheduler.py:202-222`; `moe_infinity/memory/kv_cache_manager.py:122-124`; `moe_infinity/memory/block_pool.py:127-139` | `_register_completed_blocks_in_cache` → `cache_gpu_block` | block marked reusable (hash index update), not freed | no-op |
| Terminal free on finish/abort | `moe_infinity/engine/scheduler.py:128-150`; `moe_infinity/engine/scheduler.py:223-224`; `moe_infinity/memory/kv_cache_manager.py:125-140`; `moe_infinity/memory/block_pool.py:106-115` | `finish_request` / `abort_request` → `kv_mgr.free_sequence` | terminal `ACTIVE/SWAPPED -> FREED` | **remove_requests()** |
| Non-terminal deallocation: schedule rollback (token budget exceeded) | `moe_infinity/engine/scheduler.py:83-89` | `Scheduler.schedule` | freshly allocated blocks freed; request put back to waiting | defer |
| Non-terminal deallocation: preempt fallback when swap-out buffers unavailable | `moe_infinity/engine/scheduler.py:234-240` | `_preempt_with_transfer` (no pairs path) | request marked swapped but KV dropped; must reprefill | defer |
| Non-terminal deallocation: transfer timeout fallback during swap-out | `moe_infinity/engine/scheduler.py:259-266` | `_preempt_with_transfer` | after fallback `free_sequence`, request remains logically alive | defer |
| Non-terminal deallocation: swap-in timeout fallback | `moe_infinity/engine/scheduler.py:298-309` | `_swap_in_request` | fallback free + requeue to `WAITING` for reprefill | defer |

### Native path prompt-question answers
- `_preempt_oldest_running_group()` exists in `serving/scheduler.py` (v2), not native engine. Native preemption is `_preempt_with_transfer()` and it normally swaps (`commit_swap_out`) rather than terminally freeing.
- `cache_full_block()` / `get_cached_block()` behavior (`memory/block_pool.py:116-139`): cached blocks are not separately evicted; cache membership is dropped when block is reallocated (`allocate_block`, `:97-102`) or hash-collision replacement occurs (`cache_full_block`, `:133-136`).

## Recommended EvictionSyncAdapter Hook Points

Design question: **Where exactly should `EvictionSyncAdapter.on_kv_blocks_freed()` be called?**

### v2 serving path (preferred exact sites)
1. `moe_infinity/serving/scheduler.py:176` inside `Scheduler.update_after_step` immediately after `self.kv_cache.free_sequence(seq_id)` for completed sequences.
   - Then call CP `remove_requests()` for owning request id.
2. `moe_infinity/serving/scheduler.py:191` inside `Scheduler.abort_request` immediately after `self.kv_cache.free_sequence(sequence.seq_id)` for cancelled sequences.
   - Then call CP `remove_requests()`.

**Do not hook** swap-only sites:
- `moe_infinity/serving/scheduler.py:239-243` (`swap_out` + `free_gpu_blocks`)
- `moe_infinity/serving/scheduler.py:274-288` (`swap_in`)

### native engine path (preferred exact sites)
1. `moe_infinity/engine/scheduler.py:150` inside `Scheduler.finish_request` right after `self.kv_mgr.free_sequence(request_id)`.
   - This covers normal completion and abort (`abort_request` delegates at `:224`).

2. Non-terminal free sites should **not** immediately call CP `remove_requests()`; mark as deferred only:
   - `moe_infinity/engine/scheduler.py:87`
   - `moe_infinity/engine/scheduler.py:236`
   - `moe_infinity/engine/scheduler.py:263`
   - `moe_infinity/engine/scheduler.py:303`

Rationale: these frees occur while request lifecycle continues (requeue/reprefill). Immediate `remove_requests()` would prematurely invalidate CP state.

## CP action summary

- `remove_requests()`: only terminal deallocation (`FINISHED` / abort terminal).
- `no-op`: swap-out / swap-in / prefix-cache bookkeeping.
- `defer`: non-terminal frees caused by scheduling or transfer fallback; wait until terminal finish/abort.
