# Wave 1 Round A — Microbenchmark Verdicts (2026-08-29)

Host: 6× RTX PRO 6000 Blackwell (sm_120, 96 GB), 1.5 TB RAM. Model: `deepseek-ai/DeepSeek-V2-Lite-Chat` (host offload). Raw JSON: `/mnt/raid0nvme0/leyang/moe-bench-results/`. Isolation: `$WT/_runwt.py` + per-branch worktree/offload dir. Native prefill cap raised `kMaxTokens 256 → 8192` (`core/parallel/expert_module.cpp:18`) and all 10 worktrees rebuilt.

## Verdict table

| PR | Feature | Verdict | Decisive evidence | Notes |
|----|---------|---------|-------------------|-------|
| **#182** | decode CUDA graphs | ✅ **KEEP** | fixture: `eager_replay_equal=True`, capture_failures=0, replay_coverage=1.0, per-layer KV checksums equal, observed_ratio **1.05×** | Only cleanly-validated PR. No harness issues. |
| **#176** | GPU-only expert routing | ⛔ **ROLLBACK** | `gpu_routing_verdict=ROLLBACK`: `tpot_p99` >+5% **and** `tpot_p50` >+2% @ concurrency 4 | Contradicts the PR's concurrency-1 KEEP (that condition is degenerate — decode collapses into TTFT). |
| **#180** | overlap-aware prefetch | 🟡 **KEEP (tentative)** | enforce p50 **1657 → 1181 ms (−29%)**, coverage 1.0, waste_ratio 0.0 | `covered_route_bytes=0` for enforce → improvement may be variance; **replicate ×3** before trusting. |
| **#177** | adaptive expert/KV memory | ⛔ **ROLLBACK / NEEDS-DATA** | baseline (adaptive OFF) PASS @ batch 4 (peak 14.1 GB, expert_hit 0.86); **adaptive ON → CUDA device-side assert** (`serving/engine.py:334`) | The adaptive feature crashes on the real model. |
| **#185** | async hierarchical KV swap | 🟡 **NEEDS-DATA** | async & sync both `status=PASS` (0 failures, tickets retired) but **0 swaps triggered** (`swap_*_observed_ms=0.0`); async idle throughput 284 vs sync 366 tok/s | Workload never exhausted GPU KV → feature not exercised. Needs memory-limit tuning to force swaps. |
| **#179** | phase-specific expert policy | 🟡 **NEEDS-DATA** | off arm runs; **ON arm: `RuntimeError: server policy state does not match --policy 'on'`** | `--phase-specific-expert-policy` server flag not reflected in the state the client verifies. |

## Harness / real-model bugs surfaced (these PRs' real-model paths were never exercised in CI)

- **#177**: benchmark passed `kv_cache_ratio` but `ArcherConfig` field is `kv_cache_memory_ratio` (+ needs `enable_kv_cache_offload`). Fixed in-worktree to reach the model; the adaptive path then device-asserts.
- **#180**: per-arm subprocess wrote model-load logs to stdout, corrupting its own `json.loads(proc.stdout)`. Fixed to write per-arm JSON to a file; also its child subprocess bypassed isolation (patched to route through `_runwt.py`).
- **#179**: client `--server-url` must omit `/v1` (client appends it → `/v1/v1/config` 404); server readiness needed a grace after "Uvicorn running". Both fixed; ON-state wiring gap remains.
- **#185**: hit the `kMaxTokens=256` cap (raised to 8192); flooding still exceeded it at 32×2048 (got 26847) — re-run at 8×768 succeeded but produced no swaps.

## Overarching finding

Of the 6 expert/memory-cluster PRs, **only #182 is cleanly validated on hardware**; **#176 regresses** at real concurrency; **#180 is promising but unreplicated**; and **#177 / #179 / #185 are blocked by real-model bugs or unexercised feature paths** — because every one of these plan-PRs recorded its "KEEP" on **skipped or degenerate** evidence (model absent / concurrency-1). Running them on real hardware is what exposed this.

## Remaining work

- **Round B** (FlashInfer + long context, now unblocked by the cap raise): **#178** chunked prefill (client/server, 2 servers), **#181** prefix KV reuse (Qwen3), **#183** INT8 KV quant (Qwen3 + WikiText-2 — note the `datasets`/`hub` loader bug), **#184** adaptive mixed-precision (depends on #179; no standalone CLI → likely NEEDS-DATA).
- **Wave 2** composition (exclusions #176⊥#180, #178⊥#181; dependency #184→#179; KV/serving/expert stacks) and **Wave 3** synthesis (merge order + composition matrix).
- **Follow-ups to get real verdicts**: replicate #180 ×3; force swaps for #185 (lower KV capacity); debug #177 adaptive device-assert; fix #179 ON-state verification.
