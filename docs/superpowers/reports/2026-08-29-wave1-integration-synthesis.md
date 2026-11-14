# Wave 1 — Integration Synthesis: PRs #176–#185 (2026-08-29)

Microbenchmark campaign on 6× RTX PRO 6000 Blackwell (sm_120). Raw JSON: `/mnt/raid0nvme0/leyang/moe-bench-results/`. Per-round detail: `2026-08-29-wave1-roundA-verdicts.md`. Plan: `../plans/2026-08-29-microbench-integration-10-prs.md`.

## Complete verdict table

| PR | Feature | Verdict | Basis |
|----|---------|---------|-------|
| **#182** | decode CUDA graphs | ✅ **KEEP** | eager==replay, 0 capture failures, coverage 1.0, KV checksums equal, 1.05× replay. Cleanly validated. |
| **#180** | overlap-aware prefetch | 🟢 **KEEP (replicate ×3)** | enforce p50 −29% (1657→1181 ms), coverage 1.0, waste 0. `covered_bytes=0` caveat. |
| **#176** | GPU-only expert routing | ⛔ **ROLLBACK** | tpot_p99>+5% & p50>+2% @ concurrency 4 (PR's KEEP was concurrency-1 degenerate). |
| **#177** | adaptive expert/KV memory | ⛔ **FIX-FIRST** | baseline OK; adaptive arm → CUDA device-side assert (`engine.py:334`). |
| **#179** | phase-specific expert policy | 🟡 **FIX-FIRST** | off arm runs; ON arm → "server policy state does not match" (flag not wired to verified state). |
| **#185** | async hierarchical KV swap | 🟡 **NEEDS-DATA** | async/sync PASS but 0 swaps triggered; async idle overhead (284 vs 366 tok/s). |
| **#178** | chunked prefill | 🟡 **INCONCLUSIVE** | dual Qwen3 servers start; client run did not complete in-session. |
| **#181** | prefix KV reuse | ⛔ **FIX-FIRST** | FlashInfer **dtype bug**: q `bfloat16` vs planned `float16` on Qwen3 (`flashinfer/utils.py:360`). |
| **#183** | INT8 KV-cache quant | ⛔ **FIX-FIRST** | benchmark imports `benchmarks/serving/_kv_quant_runner.py` — **file never committed** to the branch. |
| **#184** | adaptive mixed-precision | ⚪ **NEEDS-DATA** | `bench_*.py` have no CLI; depends on #179 substrate (itself FIX-FIRST). |

## The systemic finding

**7 of 10 PRs fail on first real-model contact.** Every plan-PR recorded its "KEEP" on **skipped or degenerate** evidence (model absent, downloads disabled, or concurrency-1). Running them on real hardware exposed a distinct real-model bug in each — none of which their CI could catch because the real-model gates were all skipped:

| Bug class | PRs |
|---|---|
| Config-key mismatch (`kv_cache_ratio` vs `kv_cache_memory_ratio`) | #177 |
| Subprocess bypasses isolation / stdout-JSON corruption | #180 |
| Missing committed file (`_kv_quant_runner.py`) | #183 |
| FlashInfer dtype-plan mismatch (bf16 vs fp16) | #181 |
| FlashInfer JIT missing `CUDA_HOME` | #181 (env) |
| Client `--server-url` `/v1` doubling + server readiness race | #178, #179 |
| Feature crash on real model (device-side assert) | #177 |
| Workload never exercises the feature (0 swaps) | #185 |
| Native `kMaxTokens=256` prefill cap | #185, #177, #178, #181, #183 (raised → 8192) |
| No runnable CLI | #184 |

## Integration recommendation

1. **Merge now:** **#182** (decode CUDA graphs) — the only cleanly-validated, correctness-clean, net-positive PR.
2. **Replicate ×3, then merge:** **#180** (overlap prefetch enforce) — strong p50 win; confirm it is not variance (`covered_bytes=0`).
3. **Do not merge (regresses):** **#176** — expert-path routing regresses at real concurrency. Since **#176 ⊥ #180**, #180 is the expert-path pick anyway.
4. **Return for real-model hardening before re-benchmark:** **#177** (adaptive device-assert), **#179** (ON-state wiring), **#181** (FlashInfer bf16 dtype), **#183** (commit `_kv_quant_runner.py`), **#185** (workload that forces swaps), **#178** (finish client run).
5. **Blocked on prerequisites:** **#184** needs a runnable arm driver + a real #179 substrate.

Exclusions/deps confirmed relevant: **#176⊥#180** (pick #180), **#178⊥#181** (both need fixes first), **#184→#179**.

## Environment + fixes delivered (reproducible)
- Wave-0: 10 sm_120 worktree builds, models (DeepSeek-V2-Lite, Qwen3-30B-A3B, Qwen3-8B, gpt-oss-20b), FlashInfer 0.6.18, `_runwt.py` isolation launcher (`$WT`), writable `HF_HOME=/mnt/raid0nvme0/leyang/hf`.
- `kMaxTokens 256→8192` (`expert_module.cpp:18`) + all 10 rebuilt.
- Harness fixes applied in worktrees (documented as required PR fixes): #177 config key, #180 stdout-JSON + subprocess routing, #179 server URL + readiness.
- FlashInfer JIT needs `CUDA_HOME=/usr/local/cuda-13.0`.
