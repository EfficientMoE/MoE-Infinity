# Wave 1 — FIX-FIRST Hardening Results (2026-08-29, fresh session)

Follow-up to `2026-08-29-wave1-integration-synthesis.md`. Each flagged PR was given real code/harness fixes and re-benchmarked. Fixes were applied **in the worktrees** (documented here as required PR changes), not on `main`.

## Updated verdicts after hardening

| PR | Before | After hardening | What changed |
|----|--------|-----------------|--------------|
| **#183** INT8 KV quant | broken (missing file) | 🟢 **KEEP-leaning** | **Implemented the missing `benchmarks/serving/_kv_quant_runner.py`** (drives `ContinuousBatchingEngine` with `kv_cache_format`). Result: `int8_sym` **engages** (effective=int8_sym, backend=`native_int8`, no silent fallback) and **decodes ~37% faster** (7.0 vs 5.1 tok/s, ITL 119 vs 179 ms) — consistent with halved KV bandwidth. |
| **#180** overlap prefetch | KEEP (tentative) | ⛔ **ROLLBACK / NEEDS-DATA** | **Replication ×3 flipped it.** enforce vs off p50 = **+23%, +21%, +0.6% (slower)**, `covered_route_bytes=0` every rep — the −29% was variance; the prefetch never admits bytes on this workload. |
| **#177** adaptive memory | FIX-FIRST | ⛔ **ROLLBACK** | Config-key harness bug fixed (`kv_cache_ratio`→`kv_cache_memory_ratio`); baseline runs, but the **adaptive path segfaults** (native crash) even at batch 4 with `CUDA_LAUNCH_BLOCKING=1`. |
| **#181** prefix KV reuse | FIX-FIRST | 🟡 **NEEDS-DATA (advanced)** | Fixed 3 real bugs: FlashInfer **prefill/decode dtype-plan** (bf16 vs fp16, `attention_backend.py`), FlashInfer JIT `CUDA_HOME`, and a `prefix_cache_hits_total` KeyError. Now gets past all of those but hits a **native segfault** (pybind call) in the prefix path — needs `compute-sanitizer` (absent). |
| **#179** phase policy | FIX-FIRST | 🟡 **NEEDS-DATA (advanced)** | Fixed the **real wiring bug**: `phase_specific_expert_policy` was set on `moe_config` but not the serving `engine_config` (added in `_build_engine_config`), and `_expert_policy_stats().enabled` now reflects the master-gate config. Policy-state verification now **passes**; the run then fails later because the server **streams 0 tokens** (separate serving/streaming issue). |
| **#185** async KV swap | NEEDS-DATA | 🟡 **NEEDS-DATA (confirmed)** | Re-ran with tiny GPU KV (`--kv-cache-ratio 0.03`) + long decode to force swaps — still **0 swaps** (`swap_*_observed_ms=0.0`). The benchmark's knobs don't drive the swap coordinator; async also runs slightly slower idle. |
| **#178** chunked prefill | INCONCLUSIVE | 🟡 **INCONCLUSIVE (confirmed)** | Dual Qwen3 servers start and the client connects, but the client **hangs 30+ min with zero progress**. Harness/serving throughput issue; no verdict. |
| **#182** decode CUDA graphs | KEEP | ✅ **KEEP** (unchanged) | — |
| **#176** GPU-only routing | ROLLBACK | ⛔ **ROLLBACK** (unchanged) | — |
| **#184** adaptive mixed-precision | NEEDS-DATA | ⚪ **NEEDS-DATA** (unchanged) | No runnable arm CLI; depends on #179. |

## Fixes applied in worktrees (required PR changes)

- **#181** `moe_infinity/runtime/attention_backend.py`: pass `q_data_type`/`kv_data_type` to the FlashInfer prefill plan (was defaulting fp16 while the model runs bf16); make the dtype-carrying call primary for decode. Also requires `CUDA_HOME` for JIT.
- **#181** `benchmarks/serving/prefix_cache_benchmark.py`: tolerate missing `prefix_cache_hits_total`/`matched_tokens_total` stats (`.get(...,0)`).
- **#179** `entrypoints/openai/api_server_v2.py::_build_engine_config`: propagate `phase_specific_expert_policy` into the serving engine config. `serving/engine.py::_expert_policy_stats`: make the master-gate config authoritative for `enabled`.
- **#177** `benchmarks/serving/memory.py`: `kv_cache_ratio`→`kv_cache_memory_ratio` + `enable_kv_cache_offload`.
- **#180** `benchmarks/expert_io_microbench/bench_overlap_prefetch.py`: per-arm JSON via file (not stdout); route arm subprocess through `_runwt.py`.
- **#183** `benchmarks/serving/_kv_quant_runner.py`: **new file** implementing `run_single` (serving-engine drive + format-resolution/KV/timing stats).

## Net integration picture (unchanged conclusion, now firmer)

- **Merge now:** #182.
- **Merge after hardening (feature works):** **#183** (int8 KV engages + faster decode; still needs a storage-ratio metric + WikiText-2 PPL gate).
- **Do not merge:** #176 (regresses), #180 (variance, feature inert), #177 (adaptive segfaults).
- **Advanced but still blocked:** #181 (native segfault after dtype fix), #179 (0-token streaming after wiring fix), #185 (feature not exercisable), #178 (client hang), #184 (no driver).

The hardening reinforces the core finding: these plan-PRs carry **layered** real-model bugs — fixing one repeatedly surfaces the next — so most are materially unready despite green CI on skipped/degenerate gates.
