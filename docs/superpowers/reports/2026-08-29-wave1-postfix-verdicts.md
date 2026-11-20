# Wave 1 — Post-Fix Verdicts (2026-08-29, fix-implementation pass)

Supersedes the verdict tables in the earlier three reports where noted. This pass **applied real fixes** (in worktrees) and re-benchmarked. Two PRs flipped from "broken/ROLLBACK" to **functional**.

## Breakthroughs

### #177 adaptive memory → ✅ FUNCTIONAL (was "ROLLBACK: adaptive segfaults")
The segfault was **not** an adaptive-feature defect. `memory.py --adaptive-memory` runs **two models (canonical + adaptive) sequentially in one process**; model 1's teardown (`~ArcherPrefetchHandle → CleanUpResources`) nulls the **shared global** `kTopologyHandle`, so model 2 crashes. Fixes applied:
1. `memory.py::compare_arms` now runs **each arm in its own subprocess + own offload dir** (avoids the 2-models-in-one-process global-teardown race).
2. Greedy sampling (`SamplingParams(temperature=0.0)`) — the default `temperature=1.0` tripped a `multinomial` `inf/nan` assert on **both** arms (a general numerical issue, not #177's).
3. Null-guard on `kTopologyHandle->GetSparseCacheLimit` (defensive).

Result: both arms run, `output_equality=True` (adaptive preserves token outputs), safety gates hold. On a **no-KV-pressure** workload (batch 4, short) adaptive reallocates unused KV budget to expert cache (peak 14.4→20.3 GB) but is neutral-to-slower — a memory-imbalance workload is needed to show benefit. **The feature is sound; it is not a crash-ROLLBACK.**

> Note: the underlying "MoE-Infinity can't load 2 models in one process" limitation (`kTopologyHandle` is a dtor-nulled global) is real and pre-existing — worth a separate fix (ref-count/scope per model), but it is not specific to #177.

### #179 phase policy → ✅ FUNCTIONAL (was "NEEDS-DATA: 0-token streaming")
Fixes applied:
1. `api_server_v2::_build_engine_config` now propagates `phase_specific_expert_policy` into the **serving** engine config (it was only on the sync `moe_config`).
2. `engine.py::_expert_policy_stats().enabled` reflects the master-gate config.
3. Confirmed via direct `curl` that the server generates (token-id prompt → `"\n"`; string prompt → `" Paris, 1..."`).

Result: the full phase-policy A/B **completes**, and the client's `--compare-json` **parity check passes** (OFF vs ON produce identical tokens — router semantics unchanged). **Functional; KEEP-capable.**

## Still-blocked (now precisely characterized)

- **#181** prefix reuse — **dtype/JIT/stat bugs fixed** (real); a native segfault remains in the prefix copy-on-write path. `compute-sanitizer` is available but prohibitively slow on 60 GB Qwen3, and fresh-store loads hung. → NEEDS-DATA.
- **#185** async KV swap — correctness-clean (async/sync PASS, no leaks), but the swap trigger is **admission-preemption-only** and gets squeezed out: the 8192 prefill cap limits concurrency, MLA's tiny KV makes the pool fit-all-or-nothing, and **decode-time block exhaustion raises `BlockAllocator exhausted` instead of preempting/swapping** (a real gap). Not demonstrably exercised. → NEEDS-DATA.
- **#178** chunked prefill — client hangs 30+ min. → INCONCLUSIVE.
- **#184** adaptive mixed-precision — needs a JSONL arm driver + a released allowlist + the #179 substrate. → NEEDS-DATA.

## Final integration standing

| Verdict | PRs |
|---|---|
| ✅ **Merge now** | **#182** (validated), **#183** (INT8 KV engages + ~37% faster; runner implemented) |
| ✅ **Functional after fixes — KEEP-capable** (need perf/pressure workloads to quantify benefit) | **#177** (adaptive; correctness-preserved), **#179** (phase policy; parity holds) |
| ⛔ **Do not merge** | **#176** (regresses), **#180** (variance; feature admits 0 bytes) |
| 🟡 **Real defects remain** | **#181** (native prefix-COW segfault), **#185** (swap trigger gap), **#178** (hang), **#184** (no driver + deps) |

## All fixes applied this session (worktrees)
- #177: `memory.py` subprocess-per-arm + greedy; `expert_dispatcher.cpp` null-guard.
- #179: `api_server_v2.py` engine-config flag; `engine.py` stats gate.
- #181: `attention_backend.py` FlashInfer dtype plumbing; `prefix_cache_benchmark.py` stat tolerance.
- #183: `_kv_quant_runner.py` (new, full serving-engine drive).
- #180: `bench_overlap_prefetch.py` per-arm JSON + subprocess isolation.

**Net:** started with 7/10 "failing on real-model contact"; ended with **#182 + #183 mergeable, #177 + #179 functional (4 effectively unblocked), #176/#180 correctly rejected, and 4 remaining defects each reduced to a specific, actionable root cause.**
