# BM4 — expert-H2D / compute overlap (§7 hide inequality), offloaded gpt-oss-20b

Design §7 hide inequality / §10 **BM4** for the PD-DFlash serving plan
(`docs/design/pd-dflash-moe-serving.md`). This run **canonically closes** the
BM4 measurement that PR #170 left blocked: the three DFlash compute NVTX ranges
`parse_overlap.py` unions are now emitted on the real verify path and captured,
so the parser returns a real overlap fraction instead of the false `0.0`.

## TL;DR — canonical BM4 result

| field | value |
|---|---|
| **overlap_fraction (canonical)** | **0.0539 (5.39 %)** |
| overlapped_h2d_bytes | 0.980 GB |
| exposed_fetch_bytes | 17.20 GB |
| total_h2d_bytes | 18.18 GB (833 HtoD memcpys) |
| measured H2D BW | 57.55 GB/s |
| expert cache hit rate | 0.764 |
| compute_ranges | `dflash_draft`, `route_ahead_router`, `target_verify` |

**§7 hide inequality — `(1−r)·s·M/BW ≤ t_draft + t_router + overlap_with_prev_verify`: EXPOSED.**

| term | value | source |
|---|---|---|
| LHS fetch time (18.18 GB @ 57.55 GB/s) | 315.89 ms | cuda_gpu_trace HtoD |
| t_draft (`dflash_draft`) | 127.18 ms | NVTX range, 11 draft passes |
| t_router (`route_ahead_router`) | 141.47 ms | NVTX range, 336 layer gates |
| overlap_with_prev_verify (`target_verify` ∩ H2D) | 17.03 ms (0.980 GB) | parser |
| RHS hide window | 285.68 ms | t_draft + t_router + overlap |
| **verdict** | **EXPOSED by 30.21 ms** | LHS > RHS (margin −30.21 ms) |

Both framings agree: the direct measured overlap (5.39 %) and the formal
inequality (exposed by 30.21 ms). See `hide-inequality.json`.

## What was fixed (what closed the 0.0)

1. **The three DFlash compute ranges are now emitted on the real verify path**
   as `torch.cuda.nvtx.range_push`/`range_pop` (captured by nsys
   `nvtx_pushpop_trace`; default-domain `nvtx.start_range` is dropped by nsys
   2025.1.3, the PR #170 blocker):
   - `dflash_draft` — the drafter pass (`spec_decode/dflash.py::_run_drafter`);
   - `target_verify` — the target verify forward
     (`_verify_target_block` and `verify_round`);
   - `route_ahead_router` — the target router/gate
     (`models/gpt_oss.py::SyncGptOssMLP.forward`, the `self.router` + top-k
     block; fires 336×, exactly matching the native `:moe_routing` count).
2. **Empty-domain matching in `parse_overlap.py`.** nsys renders default-domain
   push/pop ranges with a leading `:` (`:dflash_draft`), so the bare
   `DEFAULT_COMPUTE_RANGES` were a latent second miss. `compute_overlap` now
   matches modulo the empty-domain `:` prefix, so the canonical default
   invocation reproduces the number (and `--compute-range :expert_compute`
   still works). The markers are additive/read-only: routing, compute and
   emitted tokens are unchanged (no-regression: `test_gpt_oss_offload_topology`
   passes; the run itself generates with hit rate 0.764).

## Physical interpretation

The DFlash draft/router windows overlap **0 %** of the fetch on their own (the
drafter is dense and the gate is µs-scale); `target_verify` overlaps 5.39 %. The
bulk of the 18 GB H2D is (a) the cold-cache prefill/first-verify fetch (the
prefill forward is not a §7 hide phase) and (b) hidden instead by the native
**archer async pipeline** — `:expert_compute` overlaps 24.8 % and
`:expert_wait_barrier` 40.6 % of the H2D bytes (diagnostic, non-canonical).
For this single cold-start request the §7 route-ahead window does not cover the
fetch, so BM4 is EXPOSED.

## Capture command (exactly what was run)

```bash
CUDA_VISIBLE_DEVICES=0 HF_HOME=/mnt/raid0nvme0/public/huggingface \
MOE_ENABLE_SM120=1 MOE_DFLASH_SERVING_GPU=1 \
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --force-overwrite=true \
  --output=benchmarks/dflash/results/bm4-gptoss-20b/nsys/gptoss-bm4 \
python -m benchmarks.dflash.pd_dflash_serving \
  --model openai/gpt-oss-20b --draft z-lab/gpt-oss-20b-DFlash \
  --offload-dir /mnt/raid0nvme0/leyang/moe-offload/gpt-oss-20b-fp4 \
  --baseline OURS --block-size 16 --concurrency 1 \
  --requests 1 --warmup-rounds 1 --seed 1408 \
  --device-memory-ratio 0.85 --probe-h2d \
  --output benchmarks/dflash/results/bm4-gptoss-20b/raw/gptoss-bm4.json

python -m benchmarks.dflash.parse_overlap \
  --rep benchmarks/dflash/results/bm4-gptoss-20b/nsys/gptoss-bm4.nsys-rep \
  --output benchmarks/dflash/results/bm4-gptoss-20b/bm4-gptoss.json
```

Env: single RTX PRO 6000 Blackwell (sm_120, cap 12.0), nsys 2025.1.3, Python
3.12, CUDA 13. `device_memory_ratio=0.85 < 0.9` → gpt-oss expert offload enabled;
run confirmed offloaded (hit rate 0.764, 833 HtoD fetches). A clean SM120
`_store` rebuild (sm_80+sm_90+sm_120) was required so the extension carries the
`GptOssMoeDenseActDense` expert symbol (no source under `core/` changed).

## Artifacts

Committed (small): `bm4-gptoss.json` (canonical parser output),
`hide-inequality.json` (§7 terms + verdict), `raw/gptoss-bm4.json` (runner row),
`nvtx_pushpop_range_summary.csv`, `cuda_memcpy_summary.csv`.
Not committed (git-ignored, large): `nsys/gptoss-bm4.nsys-rep` (16 MB) and its
`.sqlite`/CSV exports (regenerate via the command above).
