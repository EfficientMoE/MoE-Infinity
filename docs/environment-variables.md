# Environment variables

This page lists the env keys read by production code and build files in this
repo.
It excludes test harness variables and other repo-local fixtures.

Exact string comparisons matter. Several toggles only react to `"1"` or
`"0"`.

## Runtime and serving

| Variable | Default | Where read | Effect | Notes |
| --- | --- | --- | --- | --- |
| `MOE_API_KEYS` | unset | `moe_infinity/entrypoints/openai/api_server_v2.py` startup | Comma-separated bearer tokens for the OpenAI server. If non-empty, auth middleware enforces them. | CLI `--api-key` wins when present. Empty string means no auth. |
| `CONTEXTPILOT_ENABLED` | `"1"` | `moe_infinity/entrypoints/openai/api_server_v2.py` startup and status | If `"0"`, force ContextPilot off. | `--enable-contextpilot` (or the corresponding programmatic enable) is still required; any nonzero value only removes the disable override and does not enable ContextPilot by itself. |

## Acceleration fallbacks

| Variable | Default | Where read | Effect | Notes |
| --- | --- | --- | --- | --- |
| `MOE_DISABLE_FUSED_KERNELS` | `"0"` | `moe_infinity/kernel/__init__.py` import time | If `"1"`, QKV and FFN use eager/PyTorch fallback, while decode attention bypasses the fused decode wrapper and routes through `paged_attention_fwd`, which uses the CUDA paged-attention kernel when available and otherwise falls back to torch SDPA. | Read at import time, so set it before importing `moe_infinity.kernel`. |
| `MOE_DISABLE_CUDA_GRAPHS` | unset, treated as off | `moe_infinity/serving/cuda_graph.py` init | If `"1"`, disable CUDA graph warmup, capture, and replay. | Read when `CudaGraphRunner` is constructed. |
| `MOE_INFINITY_UNVALIDATED_HW` | unset | `moe_infinity/utils/hw_profile.py` (each `supports` call), `extensions/kernel/hw_profile.h` (first use) | If `"1"`, kernels tuned for the RTX PRO 6000 Blackwell also run on other GPUs. | For re-tuning on a new card. Never enables sm_120-only kernels on other architectures. See [hardware-gated kernels](./hardware-gated-kernels.md). |
| `MOE_LINEAR_ATTN_PRO6000` | `"1"` | `moe_infinity/kernel/linear_attention_fused.py` per call | If `"0"`, Qwen3.5 GDN / GLM-5.3-Flash KDA prefill uses flash-linear-attention's chunked kernels instead of MoE-Infinity's fused prep + recurrence/output kernels. | Only matters with `flash-linear-attention` installed; GDN uses the fused path from 4096 tokens per call. |
| `MOE_LINEAR_ATTN_CUDA_PRO6000` | `"0"` | `moe_infinity/kernel/linear_attention_fused.py` per call | If `"1"`, GDN prefill on the fused path uses the L2-blocked pipeline with the CUDA `mma.sync` recurrence (`moe_infinity/kernel/csrc/gdn_recur.cu`). | Opt-in: built with nvcc on first use (~1 min, cached in `~/.cache/moe_infinity/` or `MOE_LINEAR_ATTN_BUILD_DIR`); falls back to the default Triton path if the build fails. |

## Profiling and tracing

| Variable | Default | Where read | Effect | Notes |
| --- | --- | --- | --- | --- |
| `MOE_INFINITY_PROFILE_IO` | `"0"` in `IOProfiler`, truthy check in `ExpertTracer` | `moe_infinity/profiling/io_profiler.py`, `moe_infinity/memory/expert_tracer.py` | Enable I/O event capture, atexit flush, and trace recording. | The two consumers do not check it the same way. `IOProfiler` uses `== "1"`; `ExpertTracer` currently uses truthiness of the env string. |
| `MOE_INFINITY_PROFILE_IO_OUT` | unset | `moe_infinity/profiling/io_profiler.py` | Append JSONL events to this path on flush. | No output if unset. |
| `MOE_INFINITY_PROFILE_IO_SAMPLE` | `"1.0"` | `moe_infinity/profiling/io_profiler.py` | Sampling probability, clamped to `[0.0, 1.0]`. | `0` disables, `1` records all. |

With `gpu_only_expert_routing=true`, full sampling emits `gpu_route_submit`,
`gpu_route_fallback`, and the existing `sync_wait` stages. Native NVTX adds
`gpu_route_handoff` and `expert_completion_handoff`. A nonzero
`route_failures` count or any unexpected `gpu_route_fallback` event is a
rollback signal; set `gpu_only_expert_routing=false` and retain the eager path.

## C++ runtime configuration

These keys are read by the compiled C++/CUDA offload engine (the `_store`
extension built from `core/`, see `setup.py`). They tune the native
prefetch/AIO and caching-allocator paths and are parsed with C `getenv`.

| Variable | Default | Where read | Effect | Notes |
| --- | --- | --- | --- | --- |
| `MOE_IO_THREADS` | unset, treated as `0` (auto) | `core/prefetch/archer_prefetch_handle.cpp` handle construction | Number of background I/O threads the native prefetch/AIO engine uses to read expert weights. | Parsed with `atoi`; `0` or unset defers to the AIO layer's hardware-derived `ArcherPrioAioContext::GetDefaultNumIoThreads()`. |
| `MOE_IO_BLOCK_SIZE_KB` | `1024` (1 MiB) | `core/aio/archer_prio_aio_handle.cpp` static init | Read/transfer block size for the priority AIO path (in KB), which also sizes the pinned-memory pool blocks. | Only applied when the value is in `[256, 65536]` KB; any other value falls back to the 1 MiB default. |
| `MOE_INLINE_WAIT` | unset, treated as on | `core/parallel/expert_dispatcher.cpp` `WaitHiddenStates()`, read once at first use | If exactly `"0"`, the caller thread waits passively for the dispatch instead of running grouped-decode exec items inline while it waits (work-stealing wait). | Any other value, or unset, keeps the inline wait. Set it before the first decode step; later changes are ignored. Use `0` to rule the inline wait out when debugging a hang or comparing timing. |
| `MOE_GROUPED_DECODE` | unset, treated as on | `core/parallel/expert_dispatcher.cpp` dispatcher constructor; `moe_infinity/runtime/grouped_moe_pro6000.py` `grouped_decode_pro6000_enabled()` | If exactly `"0"`, never use the grouped decode path (resident BF16 gated-SiLU experts of a decode-sized layer in one kernel pair) and run every expert through the per-expert path. | Any other value or unset keeps it on, and only where the hardware gate passes (RTX PRO 6000, otherwise per-expert regardless). Read at dispatcher construction. Grouped and per-expert kernels round differently, so on some prompts the two settings produce different token sequences; with grouped decode on, the fp32 accumulation order varies from run to run, so default-environment decode is not bit-reproducible. `MOE_GROUPED_DECODE=0` was token-identical in 34 of 34 runs of a 35B model. Use `0` when you need reproducible tokens or to compare against the per-expert path. |
| `MOE_DETERMINISTIC_ACCUM` | unset, treated as on | `core/parallel/expert_dispatcher.cpp` dispatcher constructor (`det_accum_enabled_`); setter `set_deterministic_accum(enabled, budget_bytes)` | If exactly `"0"`, every dispatch adds each expert's weighted output into the shared fp32 accumulator with atomics, as before. Otherwise, a dispatch whose slot buffer fits the budget stores each expert's output in its own (token, rank) slot and `wait_expert` reduces the slots in a fixed order, so the accumulator does not depend on which exec stream finishes first. | Larger dispatches keep the atomic path. The grouped decode kernel writes the same slots, so grouped and per-expert experts of one layer mix freely. Costs one zero-filled `[tokens, R(+1), hidden]` fp32 buffer, a rank table and one reduction kernel per dispatch. `R` is `min(top_k hint, num_experts)` (the hint comes from `set_grouped_decode`; `num_experts` when none was set); with a hint there is one extra dump row, and a token with more active experts than `R` adds the surplus atomically (counted in `accum_slot_overflow`; correct, not deterministic). Output bits change once when the setting changes. `get_routing_stats()` reports `accum_slot_dispatches`, `accum_atomic_dispatches`, `accum_slot_last_bytes`, `accum_slot_peak_bytes`, `accum_slot_overflow`. |
| `MOE_DETERMINISTIC_ACCUM_BYTES` | `268435456` (256 MiB) | `core/parallel/expert_dispatcher.cpp` dispatcher constructor (`det_accum_budget_bytes_`) | Slot-buffer budget in bytes: slot mode is used when `tokens * R * hidden * 4` is at most this value. | Parsed with `strtoll`; a non-numeric or negative value is ignored. For `hidden = 2048`, `R = 8` the default covers 4096 tokens. |
| `MOE_GROUPED_MOE_CFG` | unset (built-in launch configuration) | `extensions/kernel/grouped_moe_decode_pro6000.cu` `env_cfg()`, read once at the first grouped launch | Overrides the launch configuration of the grouped decode kernels: `"mt,wr,wk,u;mt,wr,wk,u"` (four integers for the up/gate kernel, a `;`, four for the down kernel). | Tuning only. A value that does not parse as exactly eight integers is ignored. |
| `MOE_EXPERT_DROP_TRACE` | unset (tracing off) | `core/parallel/expert_dispatcher.cpp` dispatcher constructor | A non-empty value enables the expert-drop trace tap and is the output path of the binary dump of pre-drop routing weights, masks and GPU residency per dispatch. | The dump is written by `dump_expert_drop_trace(path)` (an empty path uses this variable) and, best effort, at dispatcher teardown. Format and parser: `benchmarks/expert_drop/trace_parser.py`. |
| `MOEINF_GPU_SIZE` | unset, full device memory | `core/memory/caching_allocator.h` allocator init (CUDA path) | Overrides the GPU caching-allocator capacity, in bytes. | Read via `std::stoull`. If unset, the allocator uses the device's total global memory (`cudaGetDeviceProperties`). |
| `MOEINF_PIN_SIZE` | required (no default) | `core/memory/caching_allocator.h` allocator init (pinned path) | Sets the pinned-host caching-allocator capacity, in bytes. | Read via `std::stoull`. Unset trips a fatal log when the pinned allocator initializes, so set it when the pinned path is used. |

## Model-specific toggles

| Variable | Default | Where read | Effect | Notes |
| --- | --- | --- | --- | --- |
| `MOE_INFINITY_MXFP4_DEQUANT` | unset, treated as off | `moe_infinity/runtime/model_offload.py` checkpoint load path | If `"1"`, dequantize MXFP4 weights to BF16 while loading the offload cache. | Only matters for MXFP4 checkpoints. |
| `MOE_DSV4_FORCE_NATIVE` | unset, auto | `moe_infinity/models/deepseek_v4/official_offload_adapter.py` | If `"0"`, force the non-native DeepSeek-V4 path. Otherwise auto-select native on Blackwell when the extension is available. | An explicit `use_native` argument still wins. |

For `CUDA_VISIBLE_DEVICES` ordering, expert ownership, and one-host multi-GPU behavior, see [Single-server multi-GPU](multi-gpu.md).

## Build and packaging

| Variable | Default | Where read | Effect | Notes |
| --- | --- | --- | --- | --- |
| `CUDA_HOME` | resolved by search, final fallback `/usr/local/cuda` | `setup.py` build | Locate the CUDA toolkit and feed it to the extension build. | The script also writes the resolved value back into `os.environ["CUDA_HOME"]` and `cpp_extension.CUDA_HOME`. |
| `CUTLASS_DIR` | `~/cutlass` in `setup.py`, `$HOME/cutlass` in root `CMakeLists.txt` | `setup.py`, `CMakeLists.txt` | Locate CUTLASS headers. | Same env key, same purpose, two build systems. |
| `NVTX_DISABLE` | `"0"` | `setup.py` build | If `"1"`, compile out NVTX instrumentation macros. | Build-time only. |
| `MOE_ENABLE_SM90` | `"1"` | `setup.py` build | Include sm_90 kernels in the compiled extensions. | Build-time only. |
| `MOE_ENABLE_SM120` | `"0"` | `setup.py` build | Include sm_120 kernels and the native FP4 extension arch flags. | Build-time only. |

The package version is not set via an environment variable; it is derived from
git tags at build time by setuptools-scm (see `pyproject.toml`).

## Standard third-party envs

| Variable | Default | Where read | Effect | Notes |
| --- | --- | --- | --- | --- |
| `TRANSFORMERS_CACHE` | unset | `moe_infinity/entrypoints/big_modeling.py` checkpoint download | HuggingFace snapshot cache directory. | Standard HuggingFace env, not project-defined. |
| `HOME` | shell home directory | `CMakeLists.txt` fallback | Used only to construct the default CUTLASS path when `CUTLASS_DIR` is unset in CMake. | Standard shell env, not project-defined. |

## Test-only and harness variables

These show up in repo tests and benchmarks, but they are not part of the
production env surface documented above.

- `MOE_GLM_TINY`, `MOE_GLM_SMOKE`, `MOE_GLM_MEDIUM`
- `GPT_OSS_E2E`, `GPT_OSS_CHECKPOINT`, `GPT_OSS_DEVICE_MEMORY_RATIO`
- `MOE_DFLASH_OFFLOAD` — DFlash GPU tests use it to override the offload root without editing the test; defaults to a model-specific cache path.
- `DSV4_FLASH_CKPT`
- `MOE_DFLASH_GPU`, `MOE_DFLASH_MEM_RATIO`
- `MOE_DFLASH_SERVING_GPU` — opt-in gate for the native prefetch/compute-sample smoke tests (`tests/python/dflash/test_prefetch_native_gpu.py`). Unset skips them cleanly.
- `MOE_PREFETCH_NATIVE_MODEL`, `MOE_PREFETCH_NATIVE_OFFLOAD` — override the model/offload root used by the native prefetch smoke tests.
- `WORLD_SIZE`, `RANK`, `LOCAL_RANK`

Overlap-aware expert prefetch is configured through `ArcherConfig`
(`overlap_prefetch_policy` and its knobs, see
[Configuration](configuration.md)), not environment variables. Only the
benchmark/smoke opt-ins above gate the overlap workflow; runtime tuning stays in
config.

Repo evidence:
- `moe_infinity/kernel/__init__.py`
- `moe_infinity/serving/cuda_graph.py`
- `moe_infinity/profiling/io_profiler.py`
- `moe_infinity/memory/expert_tracer.py`
- `moe_infinity/runtime/model_offload.py`
- `moe_infinity/models/deepseek_v4/official_offload_adapter.py`
- `moe_infinity/entrypoints/openai/api_server_v2.py`
- `core/prefetch/archer_prefetch_handle.cpp`
- `core/aio/archer_prio_aio_handle.cpp`
- `core/memory/caching_allocator.h`
- `setup.py`
- `CMakeLists.txt`
