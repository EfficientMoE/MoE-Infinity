# Learnings — expert-io-microbench

## Project Setup
- Worktree: /home/xly/MoE-Infinity-expert-io-microbench
- Branch: feat/expert-io-microbench
- Main repo: /home/xly/MoE-Infinity
- Plan: /home/xly/MoE-Infinity/.sisyphus/plans/expert-io-microbench.md

## Architecture Insights
- C++ extensions built via `setup.py` using `torch.utils.cpp_extension.CUDAExtension` — NOT CMake
- Three extensions: `moe_infinity._store`, `moe_infinity._engine`, `moe_infinity._kv_cache`, `moe_infinity._paged_attn`
- `_store` extension includes all critical C++ files: expert_dispatcher.cpp, model_topology.cpp, task_scheduler.cpp, archer_prio_aio_handle.cpp
- NVTX v3 is header-only — nvtx3.hpp from CUDA toolkit or vendored into core/include/nvtx3/

## ExpertTracer API (VERIFIED)
- Constructor: `ExpertTracer(capacity: int, config: PretrainedConfig)` — singleton pattern
- Existing methods: `create_entry()`, `update_entry(seq_id, expert_list, layer_idx)`, `finish_entry(seq_id)`, `get_entry(seq_id)`, `find_most_similar(matrix, layer_idx)`
- NO `record_activation()`, `get_top_experts()`, `to_dict()` methods — those don't exist
- New I/O methods must lazy-initialize since constructor needs model config

## Top 4 Bottlenecks (Hephaestus Analysis)
1. dispatch_local `.cpu().numpy()` sync — every layer, every step
2. Global `wait_expert()` barrier — step bubble
3. `SetDevice` strong sync after cudaMemcpyAsync
4. Cache lock contention (try_lock + cv.wait)

## Conventions
- Python NVTX: `@nvtx.annotate("EventName", color="...")` — follow deepseek_v2_wrapper.py:23
- C++ NVTX: `nvtx3::scoped_range("name")` RAII — never push/pop
- nsys report name: `nvtx_sum` (not `nvtxsum`) — with fallback to nvtxsum
- Offload path for testing: /tmp/moe_bench_offload
- Model for testing: deepseek-ai/DeepSeek-V2-Lite-Chat


## NVTX Baseline (Task 1)
- `nvtx` is installed and exposes `annotate`
- Both DeepSeek wrappers already carry `@nvtx.annotate("DeepSeekPrepare")` and `@nvtx.annotate(message="DeepseekMoEBlock")`
- Existing `.nsys-rep` baselines (report1/report2) currently surface only `:DeepseekMoEBlock` in `nvtx_sum`
- No `DeepSeekPrepare` range appears in the inspected profile summaries, so the current NVTX baseline is coarse-grained around the MoE block only

## ExpertTracer I/O Tracking (Task 4)
- Added lazy I/O tracking initializer (`_init_io_tracking`) that can be called on objects created with `object.__new__`, independent of the main constructor.
- Used `collections.deque(maxlen=10000)` as an in-memory ring buffer and cached profiling gate with module-level `_io_profile_env_cache` plus per-instance `_io_profiling_enabled`.
- Added strict stage validation against the required I/O stage set and event schema `{ts_ns, layer_idx, expert_id, stage, duration_ns, bytes_transferred}`.
- Implemented per-stage aggregation in `get_io_stats()` with `numpy.percentile` for p50/p95/p99 and byte totals per stage.
- Added `to_jsonl(filepath)` exporter that writes current buffered events as one JSON object per line.


## IOProfiler Toggle Infrastructure (Task 5)
- Added profiling singleton with pid-aware reinit to avoid forked-process stale state.
- enabled is cached from MOE_INFINITY_PROFILE_IO at singleton construction; disabled path returns a no-op context manager for minimal overhead.
- Sampling uses MOE_INFINITY_PROFILE_IO_SAMPLE clamped to [0,1], skipping before timing/event allocation.
- JSONL schema emitted as {ts_ns, stage, layer, expert, dur_ns, bytes} with bytes default 0 and append-mode output controlled by MOE_INFINITY_PROFILE_IO_OUT.
- flush writes-and-clears buffered events only when output path is set; reset clears buffer for tests; atexit flush is registered only when profiling is enabled.
- Added profile_and_annotate helper usable as decorator/context manager to compose IO timing with optional NVTX annotation and graceful fallback when nvtx import is unavailable.

## 2026-04-04
- `setup.py` can safely keep NVTX header-only by searching common CUDA toolkit roots for `nvtx3/nvtx3.hpp` and appending the discovered include dir.
- Editable installs in this environment needed a CUDA 12.x toolkit to avoid Torch's CUDA major-version check; picking a compatible `CUDA_HOME` in `setup.py` let both `pip install -e . --no-deps` and `NVTX_DISABLE=1 pip install -e . --no-deps` pass.
- `distributed/expert_executor.py` now has Python-side stage wrappers for local execution path only: `moe_routing` + `profiler.time("routing")`, `expert_dispatch` + `profiler.time("expert_dispatch")`, and `expert_wait_barrier` + `profiler.time("sync_wait")`.
- Graceful imports are implemented with module-level `try/except` and no-op context fallbacks (`nullcontext`) so behavior is unchanged when NVTX/IOProfiler is unavailable.
- Task 7 added Python-stage NVTX+IOProfiler coverage at coarse-grained entry points: `prefetch_predict` (`ExpertPredictor.predict`), `prefetch_trigger` (`ExpertPrefetcher.prefetch_experts`), `transfer_schedule` (`UnifiedTransferScheduler.enqueue` and `_run_request`), and `cache_lookup` (`ExpertOffloadCoordinator.prefetch_experts` and `_handle_expert_fetch`).
