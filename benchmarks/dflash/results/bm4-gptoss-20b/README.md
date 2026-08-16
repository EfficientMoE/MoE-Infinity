# BM4 — expert-H2D / compute overlap (hide inequality), offloaded gpt-oss-20b

Design §7 hide inequality / §10 **BM4** for the PD-DFlash serving plan
(`docs/superpowers/plans/2026-08-14-pd-dflash-serving-scheduler.md`, Task 10).
Measurement only — no source logic was changed. The BM4 parser
(`benchmarks/dflash/parse_overlap.py`) was used exactly as merged on `origin/dev`.

## TL;DR

**BM4 canonical overlap is BLOCKED by missing NVTX instrumentation, not by an
exposed fetch.** The nsys trace does contain the raw GPU signal (822 host→device
expert memcpys, **17.61 GB**, measured H2D BW **57.56 GB/s**), but **none** of the
three compute NVTX ranges `parse_overlap.py` keys on
(`dflash_draft`, `route_ahead_router`, `target_verify`) are present in the trace,
so the parser returns a **false** `overlap_fraction = 0.0`. The §7 hide-inequality
verdict cannot be closed canonically because its `overlap_with_prev_verify` term
(BM4) and `t_router` are unavailable.

Diagnostic (non-canonical) overlap against the **native** archer compute ranges
that *are* captured shows the physical signal is real: **26.8 %** of H2D bytes
overlap `:expert_compute`, **47.7 %** overlap the broader native compute set.

## Exact missing NVTX instrumentation (the blocker)

`parse_overlap.py` reads `nsys stats --report cuda_gpu_trace,nvtx_pushpop_trace`
and unions the compute ranges `DEFAULT_COMPUTE_RANGES =
("dflash_draft", "route_ahead_router", "target_verify")`. In this trace:

1. **`dflash_draft` and `target_verify` are emitted by the runner but never
   captured.** `benchmarks/dflash/_serving_measure.py::nvtx_range()` emits them
   via `nvtx.start_range(message=..., color="green")` — an NVTX **start/end**
   range on the **default (dummy) domain**. Verified by an isolated probe
   (`nsys profile --trace=nvtx` on a script calling both APIs):
   - `nvtx.push_range("PUSHPOP_PROBE")` → appears in `nvtx_pushpop_trace` ✓
   - `nvtx.start_range("target_verify")` → appears in **neither**
     `nvtx_pushpop_trace` **nor** `nvtx_startend_trace` ✗
   So the runner's phase markers are not recorded at all by this
   nsys (2025.1.3) + `nvtx` (python) combination, and even if they were, they
   would land in the start/end report, which the parser does not read.

2. **`route_ahead_router` is never emitted at all.** No code path in
   `_serving_measure.py` wraps a `route_ahead_router` range (only `dflash_draft`
   and `target_verify` are wrapped). One of the three required compute ranges has
   no producer. (`route_ahead_issue` and `expert_h2d`, declared in
   `pd_dflash_serving.NVTX_RANGES`, are likewise never wrapped; the parser does
   not need them.)

Net effect: `parse_overlap.py` finds zero matching compute ranges → all 822 H2D
memcpys are reported "exposed" → `overlap_fraction = 0.0`. **This 0.0 is an
instrumentation artifact, not evidence that expert fetch is exposed.**

### What the trace *does* contain (NVTX push/pop range names)

Only native C++ archer ranges are captured, e.g. `:expert_compute`,
`:moe_routing`, `:expert_dispatch`, `:expert_enqueue`, `:task_queue_pop`,
`:cpu_to_gpu`, `:gpu_fetch`, `:disk_to_cpu`, `:cuda_stream_sync`,
`:expert_wait_barrier`, `:aio_read_submit`, `:aio_wait`. Full list in
`nvtx_pushpop_range_summary.csv`.

## Canonical BM4 result (blocked)

`bm4-gptoss.json` (raw `parse_overlap.py` output, unmodified parser):

| field | value | note |
|---|---|---|
| total_h2d_bytes | 18,035,167,000 (17.61 GB) | real — 822 HtoD memcpys |
| overlapped_h2d_bytes | 0.0 | **false zero** — no compute range matched |
| overlap_fraction | **0.0** | **false zero** — missing instrumentation |
| exposed_fetch_bytes | 18,035,167,000 | artifact of the false zero |
| compute_ranges | dflash_draft, route_ahead_router, target_verify | none present in trace |

## §7 hide inequality — status: INDETERMINATE (blocked)

`(1 − r)·s·M / BW  ≤  t_draft + t_router + overlap_with_prev_verify`
(`report.evaluate_hide_inequality`). Measured inputs that ARE available from the
same run:

| term | value | source |
|---|---|---|
| measured H2D BW | 57.56 GB/s | runner `--probe-h2d` (real device probe) |
| H2D bytes moved (run) | 17.61 GB | cuda_gpu_trace HtoD memcpys |
| raw H2D transfer time | 450.5 ms (≈306 ms at peak BW) | Σ HtoD memcpy durations |
| s = min(1, B·k/E_l) | 1.0 | B=16, top-k=4, E_l=32 (gpt-oss-20b) |
| t_draft ≈ ttft | 0.213 s | runner metric (dflash_draft range not captured) |
| expert cache hit rate | 0.743 | runner metric |
| expert_occupancy_bytes | 17.9 GB | runner metric |
| **t_router** | **unavailable** | route_ahead_router not instrumented |
| **overlap_with_prev_verify** | **unavailable** | BM4 blocked (above) |

The inequality **cannot be closed** because the two right-hand-side terms BM4 is
meant to supply (`overlap_with_prev_verify`, and `t_router`) are exactly the
missing instrumentation. No hidden/exposed verdict is asserted — doing so would
require fabricating those terms.

## Diagnostic overlap vs NATIVE archer compute ranges (non-canonical)

Using the parser's own `--compute-range` override against the native ranges that
*are* captured (real numbers; **not** the plan's draft/router/verify phases):

| native compute range set | overlap_fraction | overlapped H2D |
|---|---|---|
| `:expert_compute` | 0.268 | 4.83 GB / 17.61 GB |
| `:expert_compute :moe_routing` | 0.268 | 4.83 GB |
| `:expert_compute :moe_routing :expert_dispatch :task_execute :expert_enqueue` | **0.477** | 8.59 GB |

Interpretation: ~27–48 % of expert-fetch bytes overlap native archer compute on
the GPU timeline. This is a lower-bound proxy (native ranges are finer-grained
per-expert compute, not the DFlash draft/router/verify phases) and confirms the
blocker is instrumentation naming/emission-API, not absence of GPU data.

## Precise repo fix required (NOT applied — measurement only)

To make BM4 measurable as designed, one of:

- **Runner side** (`benchmarks/dflash/_serving_measure.py`): emit the phase
  markers with `nvtx.push_range(...)/nvtx.pop_range()` (push/pop, which nsys
  captures) instead of `nvtx.start_range()/end_range()`, **and** add the missing
  `route_ahead_router` (and, if the parser is extended, `route_ahead_issue`,
  `expert_h2d`) ranges; or
- **Parser side** (`benchmarks/dflash/parse_overlap.py`): additionally read
  `nvtx_startend_trace` — but note the isolated probe shows default-domain
  `start_range` is not captured by this nsys build at all, so the runner-side fix
  (push/pop, or a registered NVTX domain) is required regardless.

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
3.13. `device_memory_ratio=0.85 < 0.9` → gpt-oss expert offload enabled
(`_gpt_oss_offload_enabled`, `_GPT_OSS_RESIDENT_RATIO=0.9`); run confirmed
offloaded (hit rate 0.74, 822 HtoD fetches).

## Native `_store` rebuild note (prerequisite; measurement-enabling only)

The in-tree `_store.cpython-313*.so` (built 2026-08-11) predated gpt-oss commit
`ac143a6` ("feat(gpt-oss): execute fetched MXFP4 experts in Archer"). An
incremental rebuild reused stale objects and produced a `.so` lacking the
`GptOssMoeDenseActDense` expert type, so the first run aborted with
`ExpertDispatcher::ExpertDispatcher: unknown expert type 6`
(`core/parallel/expert_dispatcher.cpp:167`). A clean rebuild
(`rm -rf build/temp.*-cpython-313 build/lib.*-cpython-313; setup.py build_ext
--inplace`, MOE_ENABLE_SM120=1, sm_80+sm_90+sm_120) restored the
`GptOssMoeDenseActDense` symbol and the run then succeeded. `core/` is identical
between HEAD and `origin/dev`; no source was modified.

## Artifacts

Committed (small): `bm4-gptoss.json` (canonical parser output),
`diag-overlap-*.json` (native-range diagnostics), `raw/gptoss-bm4.json` &
`raw/gptoss-validate.json` (runner rows), `nvtx_pushpop_range_summary.csv`,
`cuda_memcpy_summary.csv`.
Not committed (large, git-ignored): `nsys/gptoss-bm4.nsys-rep` (15 MB),
`nsys/gptoss-bm4.sqlite` (43 MB), full per-event trace CSVs (~83 MB).
