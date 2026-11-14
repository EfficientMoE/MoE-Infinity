# Wave 1 Productization Plan — remaining defects (#185, kTopologyHandle, #181, #178, #184)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax. Each sub-plan is independent and produces working, testable software on its own; do them in the priority order below.

**Goal:** Turn the four remaining Wave-1 defects (and the shared-global limitation underneath #177) from "known root cause" into shipped fixes with real benchmark verdicts.

**Architecture:** Each item is a self-contained sub-plan against one branch worktree under `/mnt/raid0nvme0/leyang/moe-wt/<branch>` (`$WT`). Environment is warm (10 sm_120 builds, models in `HF_HOME=/mnt/raid0nvme0/leyang/hf`, isolation launcher `$WT/_runwt.py`, `compute-sanitizer` at `/usr/local/cuda-13.0/bin/`, `kMaxTokens` raised to 8192). Run everything via the launcher: `CUDA_VISIBLE_DEVICES=<n> MOE_WT_ROOT=$WT/<branch> env -u PYTHONPATH /home/leyang/anaconda3/bin/python $WT/_runwt.py <script|-m module> [args]`.

**Tech Stack:** C++/CUDA native ext (CMake/Ninja, `MOE_ENABLE_SM120=1 MOE_ENABLE_SM90=0 CUTLASS_DIR=~/cutlass`), PyTorch 2.12+cu130, FlashInfer 0.6.18, pytest (`tests/cpp/unit/prefetch`, `tests/python/serving`), DeepSeek-V2-Lite-Chat / Qwen3.

**Disk hygiene (learned the hard way):** use **one offload dir per model** (`$OFF/deepseek`, `$OFF/qwen3`), never one-per-run; each Qwen3-30B store is ~57 GB and per-run stores filled the volume. Clean `moe-offload` between waves.

---

## Priority & dependency order

1. **P0 — `kTopologyHandle` single-owner lifecycle** (foundational; makes multi-model safe, retires the #177 subprocess workaround, prevents the class of null-global segfaults).
2. **P1 — #185 async KV swap: decode-time preemption** (scheduler redesign; unblocks the swap verdict).
3. **P1 — #181 prefix-reuse native COW segfault** (localize with sanitizer on a small model, then fix).
4. **P2 — #178 chunked-prefill client hang** (diagnose the httpx/stats-poll stall, then fix).
5. **P2 — #184 adaptive mixed-precision** (rebase onto #179 substrate + allowlist + JSONL arm driver).

**Assumption to confirm:** full multi-model-per-process support is **out of scope** (it requires de-globalizing 6 singletons — a large refactor). P0 instead makes the single-model contract *safe and explicit*. If you want true multi-model, that becomes its own plan.

---

## Sub-plan P0 — `kTopologyHandle` (and sibling globals) single-owner lifecycle

**Background:** `core/model/model_topology.cpp:33` declares `std::unique_ptr<ArcherTopologyHandle> kTopologyHandle = nullptr;` (plus `kTensorIndex`, `kArcherTensorHandle`, `kTaskPool`, `kDeviceMemoryPool`, `kHostMemoryPool`). `ArcherPrefetchHandle`'s ctor (`core/prefetch/archer_prefetch_handle.cpp:29-35`) creates all six; `CleanUpResources` (`:81-88`, called from `~ArcherPrefetchHandle`) `.reset()`s all six. `has_cleaned_up_resources_` guards double-cleanup for **one** handle but not across handles. Constructing a 2nd handle silently clobbers the 1st's globals; destroying the 1st nulls the 2nd's → the null-`mutex_` segfaults seen in `GetSparseCacheLimit`/`GetNodeFromTensorID`.

**Fix strategy:** enforce the single-owner contract explicitly — the 2nd concurrent construction raises a clear Python exception instead of corrupting globals; teardown stays single-owner. Land on the branch that touches `archer_prefetch_handle.cpp` (e.g. `adaptive-expert-kv-memory`), then propagate to `dev`/main.

**Files:**
- Modify: `core/prefetch/archer_prefetch_handle.cpp` (ctor guard)
- Test: `tests/cpp/unit/prefetch/test_topology_lifecycle.cpp` (new) + register in `tests/cpp/unit/prefetch/CMakeLists.txt`
- Test: `tests/python/unit/test_multi_model_guard.py` (new)

- [ ] **Step 1: Write the failing native test.** Construct two `ArcherPrefetchHandle` in one process; assert the 2nd throws `std::runtime_error` (rather than leaving `kTopologyHandle` dangling). Add a second case: construct, destroy, construct again succeeds.
- [ ] **Step 2: Run it, verify it fails.** `cmake --build build --target test_topology_lifecycle && ctest --test-dir build -R TopologyLifecycle --output-on-failure` → Expected: FAIL (segfault/no-throw).
- [ ] **Step 3: Implement the guard.** In `ArcherPrefetchHandle::ArcherPrefetchHandle` (before line 29):
  ```cpp
  if (kTopologyHandle != nullptr) {
    throw std::runtime_error(
        "MoE-Infinity supports one model per process; a model is already "
        "loaded. Construct additional models in separate processes.");
  }
  ```
  Keep `CleanUpResources` unchanged (single owner ⇒ safe to reset on destruction; it nulls `kTopologyHandle` at `:85`, re-enabling a fresh construction).
- [ ] **Step 4: Run the test, verify PASS.** Same ctest → Expected: PASS.
- [ ] **Step 5: Python-surface test.** `test_multi_model_guard.py`: a 2nd `MoE(...)` while one is alive raises `RuntimeError` (not a segfault). `pytest tests/python/unit/test_multi_model_guard.py -q`.
- [ ] **Step 6: Rebuild + regression.** `MOE_ENABLE_SM120=1 MOE_ENABLE_SM90=0 CUTLASS_DIR=~/cutlass python setup.py build_ext --inplace`; run the branch CPU suite → Expected: prior pass counts unchanged.
- [ ] **Step 7: Keep the #177 subprocess-per-arm workaround** (benchmarks legitimately need multiple models ⇒ separate processes). Update `benchmarks/serving/memory.py` docstring to state each arm runs in its own process because of the single-model-per-process contract.
- [ ] **Step 8: Commit** `fix(core): enforce single-model-per-process; raise instead of corrupting global topology`.

---

## Sub-plan P1 — #185 async KV swap: decode-time preemption

**Background (proven):** swaps fire only via admission preemption (`serving/scheduler.py:303-304 → _preempt_oldest_running_group:529`). Decode growth allocates in `update_after_step` (`:429-430 append_tokens`), and `BlockAllocator.allocate` (`serving/kv_cache.py:147-156`) **raises `RuntimeError("BlockAllocator exhausted …")`** when full — so decode-time pressure crashes instead of swapping. Swap-out is async (`submit_swap_out_group`), so a synchronous decode retry cannot free blocks in-step.

**Fix strategy:** make `schedule()` preempt **proactively** using a free-block watermark so blocks are freed (async) before decode needs them, and make decode-append **defer** a sequence when momentarily short instead of raising.

**Files:**
- Modify: `moe_infinity/serving/scheduler.py` (`schedule`, `update_after_step`, add `_decode_block_demand`)
- Modify: `moe_infinity/serving/kv_cache.py` (add `can_append` probe)
- Test: `tests/python/serving/test_decode_preemption.py` (new)
- Bench: `benchmarks/serving/kv_offload_benchmark.py` (already has `--max-batch-size`/`--max-tokens-per-step` knobs added this session)

- [ ] **Step 1: Failing test.** Fake KV store with N blocks; admit G > capacity groups; drive decode steps; assert (a) no `RuntimeError`, (b) `swap_out` invoked, (c) all sequences complete. Run → Expected: FAIL.
- [ ] **Step 2: Proactive preemption in `schedule()`** (after admission, before finalizing decode at `:334-352`):
  ```python
  free = self.kv_cache.block_allocator.num_free_blocks
  demand = self._decode_block_demand(next_decode_sequences)
  while free < demand and self._running:
      if not self._preempt_oldest_running_group():
          break
      free = self.kv_cache.block_allocator.num_free_blocks
  ```
  Add `_decode_block_demand(self, seqs)` = count of running DECODE seqs whose next token crosses a block boundary (`committed_kv_tokens % block_size == 0`).
- [ ] **Step 3: Defer-on-shortage in `update_after_step`** (`:429-432`):
  ```python
  if not self.kv_cache.can_append(seq_id, num_new):
      continue   # leave running; retried next step after async swaps free blocks
  self.kv_cache.append_tokens(seq_id, num_new_tokens=num_new)
  ```
  Add `KVCache.can_append(seq_id, n) -> bool` (checks free blocks vs blocks-needed for `n` tokens, no allocation).
- [ ] **Step 4: Run test, verify PASS.**
- [ ] **Step 5: Real A/B** (GPU4, single per-model offload dir):
  ```bash
  for M in async sync; do CUDA_VISIBLE_DEVICES=4 MOE_WT_ROOT=$WT/async-hierarchical-kv env -u PYTHONPATH \
    python $WT/_runwt.py $WT/async-hierarchical-kv/benchmarks/serving/kv_offload_benchmark.py \
    --model deepseek-ai/DeepSeek-V2-Lite-Chat --offload-dir /mnt/raid0nvme0/leyang/moe-offload/deepseek \
    --enable-kv-offload --num-requests 32 --prompt-length 384 --max-new-tokens 384 \
    --warmup-requests 2 --trials 3 --host-memory-mib 8192 --max-inflight-mib 4096 \
    --kv-cache-ratio 0.004 --max-batch-size 8 --max-tokens-per-step 4096 \
    --checksum --kv-swap-mode $M --output-json /mnt/raid0nvme0/leyang/moe-bench-results/185_$M.json; done
  ```
  Expected: `swap_out_observed_ms > 0`, `failures == 0`.
- [ ] **Step 6: Verdict** — KEEP if async hides swap latency (E2E/decode throughput ≥ sync at KV pressure) with 0 failures/leaks.
- [ ] **Step 7: Regression** `pytest tests/python/serving -q -m "not gpu"`.
- [ ] **Step 8: Commit** `feat(serving): decode-time proactive preemption so KV pressure swaps instead of raising`.

---

## Sub-plan P1 — #181 prefix-reuse native COW segfault

**Background:** dtype/JIT/stat bugs already fixed this session (`runtime/attention_backend.py` FlashInfer `q_data_type`/`kv_data_type`; `prefix_cache_benchmark.py` stat tolerance; needs `CUDA_HOME=/usr/local/cuda-13.0`). A native pybind segfault remains in the prefix copy-on-write / KV-attach path.

**Fix strategy:** reproduce on a **small** resident model so `compute-sanitizer` is tractable, localize the invalid access, fix it.

**Files:** `moe_infinity/serving/kv_cache.py` (COW refcounts/block copy), `moe_infinity/serving/prefix_cache.py`, `moe_infinity/runtime/attention_backend.py`; test `tests/python/serving/test_prefix_cache_cuda.py` (exists, env-gated).

- [ ] **Step 1: Small-model repro.** Pre-warm ONE offload store for the smallest paged-supported Qwen3; confirm a 1-request warm/cold run reproduces the segfault (avoids the 60 GB hang).
- [ ] **Step 2: Localize with memcheck** (minimal shape):
  ```bash
  CUDA_HOME=/usr/local/cuda-13.0 CUDA_VISIBLE_DEVICES=1 MOE_WT_ROOT=$WT/prefix-kv-reuse env -u PYTHONPATH \
    /usr/local/cuda-13.0/bin/compute-sanitizer --tool memcheck \
    python $WT/_runwt.py $WT/prefix-kv-reuse/benchmarks/serving/prefix_cache_benchmark.py \
    --model <small-qwen3> --offload-dir <prewarmed> --shared-prefix-tokens 64 --suffix-tokens 8 --output-json /tmp/181.json
  ```
  Expected: first invalid read/write + CUDA API stack ⇒ the offending copy/kernel (hypothesis: layer-complete COW block copy in `kv_cache.py`, or a page-index off-by-one in `attention_backend.py`). If host-side, use `PYTHONFAULTHANDLER=1`.
- [ ] **Step 3: Failing unit test** at the localized boundary (COW copy of a shared partial-tail block across layers) → FAIL.
- [ ] **Step 4: Fix** minimally (bounds/refcount/dtype).
- [ ] **Step 5: Verify** — unit test PASS; `MOE_PREFIX_CACHE_CUDA=1 pytest tests/python/serving/test_prefix_cache_cuda.py -q` PASS; warm/cold benchmark: equal cold==warm token+logit digests, warm TTFT p50 ≪ cold.
- [ ] **Step 6: Commit** `fix(serving): correct prefix-cache COW <bug>; prefix reuse no longer segfaults on bf16 paged path`.

---

## Sub-plan P2 — #178 chunked-prefill client hang

**Background:** dual Qwen3 servers start, client connects, then hangs 30+ min. Client is `httpx.AsyncClient(timeout=httpx.Timeout(600.0))` (`benchmarks/serving/chunked_prefill_latency.py:137`) with a background ~100 Hz `/admin/stats` poll (`:108-113`) and streamed `/v1/completions` (`:65-69`). Hang candidates: 600 s timeout masking a stuck stream; the stats poll starving/deadlocking the server; server-side chunked-prefill generation stall.

- [ ] **Step 1: Instrument.** Reduce client timeout to `httpx.Timeout(60.0)`, add a print before/after each `await` (config GET, stats-poll start, each stream). Identify the await that never returns.
- [ ] **Step 2: Bisect.** (a) Disable the stats-poll task; if it completes, raise the poll interval (`:113 timeout=0.5`). (b) If a `stream_request` hangs, `curl -N` that server's `/v1/completions` with the exact payload to see if it stops emitting `data:`/`[DONE]` mid-stream (server chunked-prefill stall).
- [ ] **Step 3: Fix** the root cause (poll interval / per-stream inactivity timeout / server terminal-token emission).
- [ ] **Step 4: Verify** the paired A/B completes: candidate reduces decode-row TPOT/TTFT tails with identical terminal tokens vs baseline.
- [ ] **Step 5: Commit** `fix(bench/serving): <root cause> so chunked-prefill A/B completes`.

---

## Sub-plan P2 — #184 adaptive mixed-precision (rebase + driver)

**Background:** `benchmarks/adaptive_precision/bench_{e2e,policy,transfer}.py` expose no CLI (only `report.py <input.jsonl> --output`); `ARMS=("canonical","static_low","adaptive")`. Depends on #179's single `ExpertResidencyManager` substrate + a fingerprint-qualified released allowlist.

- [ ] **Step 1: Combined worktree** = `plan/adaptive-mixed-precision-experts` rebased onto `plan/phase-specific-expert-policy`; resolve conflicts in `core/prefetch/expert_residency.*` and `core/parallel/expert_dispatcher.*`; rebuild sm_120; smoke-import.
- [ ] **Step 2: Allowlist manifest.** Derive the schema from `moe_infinity/runtime/expert_variant_manifest.py` + `adaptive_precision_allowlist.py`; produce a fingerprint-qualified manifest for `openai/gpt-oss-20b`. If none can be produced, #184 is genuinely NEEDS-DATA (record the exact missing artifact).
- [ ] **Step 3: Arm driver** `benchmarks/adaptive_precision/run_arms.py`: for each arm, drive `bench_e2e` to emit one JSON run-record per line (schema per `report.py::validate_run`) → `runs.jsonl`.
- [ ] **Step 4: Reduce + gate** `python report.py runs.jsonl --output 184_report.json`; read `release_gate`.
- [ ] **Step 5: Verdict** — KEEP iff `release_gate==PASS` (quality attestation per admitted variant + peak/transfer bytes down vs canonical + TPOT not regressed).
- [ ] **Step 6: Commit** `feat(bench): adaptive-precision arm driver; run on #179 substrate`.

---

## Self-Review checklist
- [ ] Each sub-plan independently shippable, failing-test-first where code changes (P0, #185, #181, #184 driver). ✅
- [ ] Diagnostic-first sub-plans (#181 localize, #178 bisect) have exact reproduction commands before any fix; remaining unknowns (#181 exact line, #178 exact await, #184 manifest) are the *output of a diagnostic step*, not hidden placeholders. ✅
- [ ] Priority respects dependencies (#184 → #179; P0 foundational). ✅
- [ ] sm_120 rebuild flags on every native change. ✅
- [ ] One offload dir per model (disk hygiene). ✅

## Execution handoff
**(1) Subagent-driven** (fresh subagent per sub-plan, review between) or **(2) Inline** (execute here with checkpoints). Recommended order: **P0 → #185 → #181 → #178 → #184.**
