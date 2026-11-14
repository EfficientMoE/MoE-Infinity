# Wave 1 — Deep-Dive Verdicts (2026-08-29, native debugging pass)

Follow-up to `2026-08-29-wave1-hardening-results.md`. Used `PYTHONFAULTHANDLER=1` (Python stack at fatal signal) and `compute-sanitizer` (found at `/usr/local/cuda-13.0/bin/`) to root-cause the native crashes and the not-exercisable features.

## Root-caused verdicts

### #177 adaptive memory → ⛔ ROLLBACK (definitive root cause)
`PYTHONFAULTHANDLER` located the crash precisely:
```
__pthread_mutex_lock
ArcherTopologyHandle::GetSparseCacheLimit(...)          # at ExpertDispatcher construction
ArcherTopologyHandle::GetNodeFromTensorID(...)          # at ArcherPrefetchHandle::FetchTensors (runtime)
```
Root cause: `kTopologyHandle` is a **shared global** `std::unique_ptr` (`model_topology.cpp:33`), set in `ArcherPrefetchHandle`'s ctor (`archer_prefetch_handle.cpp:32`) and **`.reset()` to null in its dtor (`archer_prefetch_handle.cpp:85`)**. The adaptive resize's "transactional storage recreation" destroys/recreates a prefetch handle, nulling the global **while other threads still dereference it** → null-`mutex_` segfaults. A ctor null-guard (applied, `expert_dispatcher.cpp:133`) just moves the crash to `GetNodeFromTensorID` during fetch. This is a **fundamental global-lifecycle defect**, not a harness issue — a real ship-blocker. Proper fix: ref-count or scope `kTopologyHandle` per-model instead of a dtor-nulled global.

### #185 async KV swap → 🟡 NEEDS-DATA (swap trigger gap)
Swaps fire only via scheduler preemption (`_preempt_oldest_running_group`) when a **running** sequence needs a block and the pool is full. On DeepSeek-V2-Lite (**MLA → tiny KV**), the exercisable band is essentially empty:
- Pool that fits the workload → **0 swaps** (`swap_*_observed_ms=0.0`).
- Pool too small (`--kv-cache-ratio 0.004`) → `RuntimeError: BlockAllocator exhausted: requested 1 blocks but only 0 available` — **the allocator raises instead of swapping out**.
So the async-swap path is not demonstrably exercised, and allocator-exhaustion doesn't route into preemption/swap. Real gap; no perf verdict possible with this benchmark/model.

### #181 prefix KV reuse → 🟡 NEEDS-DATA (dtype fixed; native segfault remains)
The FlashInfer **dtype-plan bug is genuinely fixed** (prefill/decode now pass `q_data_type`/`kv_data_type`); JIT needs `CUDA_HOME`; a `prefix_cache_hits_total` stat KeyError is patched. The run then **segfaults in a native pybind call** in the prefix-reuse path (COW/KV attach). Pinpointing was blocked because a fresh-offload-store Qwen3-30B-A3B load **hung at 0% for 14+ min** (I/O contention). Needs `compute-sanitizer` on a pre-warmed store to localize the invalid access.

### #179 phase policy → 🟡 NEEDS-DATA (wiring fixed; 0-token streaming remains)
The **real wiring bug is fixed** — `phase_specific_expert_policy` now propagates to the serving `engine_config` and `_expert_policy_stats().enabled` reflects the master gate, so the client's policy-state verification **passes**. The run then fails at `summarize()` because the server streams **0 non-empty `choices[0].text` chunks** over `/v1/completions` (prompt = `[1]*N` token-id list, greedy). Separate serving/streaming or detokenization gap; needs a server-side SSE trace.

### #178 chunked prefill → 🟡 INCONCLUSIVE
Dual Qwen3 servers start, client connects, then **hangs 30+ min with zero progress**.

### #184 adaptive mixed-precision → ⚪ NEEDS-DATA
No runnable arm CLI; depends on #179's ExpertResidencyManager substrate.

## Fixes applied this pass (in worktrees)
- **#177** `core/parallel/expert_dispatcher.cpp`: null-guard `kTopologyHandle->GetSparseCacheLimit` (mitigation only — root defect is the dtor-nulled global).

## Final integration standing (all three reports combined)

| Verdict | PRs |
|---|---|
| ✅ **Merge now** | **#182** decode CUDA graphs |
| 🟢 **Merge after hardening (feature works)** | **#183** INT8 KV quant (engages, `native_int8`, ~37% faster decode) |
| ⛔ **Do not merge** | **#176** (regresses), **#180** (variance; feature inert, 0 bytes admitted), **#177** (adaptive nulls shared `kTopologyHandle` → segfault) |
| 🟡 **Blocked on deeper defects (advanced, not shippable)** | **#181** (native segfault after dtype fix), **#179** (0-token streaming after wiring fix), **#185** (swap trigger gap), **#178** (client hang), **#184** (no driver) |

The native-debugging pass converted vague "crashes" into **specific, actionable root causes** — most importantly #177's shared-global lifecycle defect, which is a concrete ship-blocker with a clear fix direction.
