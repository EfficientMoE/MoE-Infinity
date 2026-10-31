# Changelog

All notable changes to MoE-Infinity will be documented in this file.

## [Unreleased]

### Changed
- The non-MLA paged KV cache now allocates one plane per decoder layer
  (previously all layers aliased one plane). Memory use for the paged KV
  cache grows by the layer count; deployments tuned to the undersized cache
  may need a lower `device_memory_ratio`.

### Added

- DeepSeek-V4.1-Flash (`deepseek_v41`) Path B support: the model type is
  protected as `deepseek_v4_fp4` (previously the generic `quant_method:
  fp8` fallback silently mis-mapped it to the GLM 128x128 FP8 identity);
  the moe-store `wrappers/deepseek_v41` family provides the official-code
  mp1 offload loader with FP4 expert streaming, host-offloaded engram
  tables, blockwise `wo_a` dequant, and an eager sparse-attn fallback for
  smem-limited GPUs. Validated on the real checkpoint for the mp1
  text-only greedy smoke (`tests/python/v41/e2e_offload.py`: deterministic
  coherent output, 14.8 GiB peak resident on a 96 GB SM120 GPU); dspark
  MTP and vision excluded; no continuous-serving, sampled, batch>1, or
  multi-GPU claim. Path A (HF-native) remains guarded on transformers
  shipping `DeepseekV41ForCausalLM`. (#281)
- Expert-cache policy baselines + benchmark harness under `benchmarks/cache_simulator/`: a `CachePolicy` ABC (`policies.py`) with one shared `simulate()` harness and ten baselines over a flat `(layer, expert)` access sequence — LRU, LFU, windowed LFU-K, model-tracing (BrainStorm global counts), score-based eviction (time-decayed activation), LHU (HOBBIT aged frequency), and four trace-oracle prefetch variants (single-layer-ahead, stride, multi-layer-ahead, activation-matrix). Prefetch predictors use the recorded next-layer routing as a perfect oracle and are labelled upper bounds; their demand hit rate stays bounded by the Belady-optimal ceiling (prefetch value is reported separately as `effective_hit_rate`). `research/run_policies.py` sweeps cache capacity over three synthetic MoE workloads (no real Qwen3-235B captures are available, so all numbers are synthetic), emits a policy×capacity hit-rate table (`results_synthetic.{md,csv}`) + per-model hit-rate-vs-capacity (MRC-style) plots with Belady as "Ideal", and appends runs to `research/EXPERIMENTS.md`. Behavioral unit tests (`tests/python/unit/test_cache_simulator_policies.py`) cover deterministic per-policy cases plus the invariants `hits+misses==len`, capacity never exceeded, and `hit_rate(policy) <= hit_rate(Belady)` on every fuzzed trace. Large figures are git-ignored; tables and sources are committed.
- Offline MoE expert-cache simulator under `benchmarks/cache_simulator/`: a Belady-optimal hit-rate bound over `.pt` expert-distribution recorder traces (`belady_cache_simulator.py`) and over pickled prefill/decode activation traces (`cache_strategy.py`), plus a prefetch/replacement policy library (`fetch_engines.py`: streaming FIFO, LRU, LFU, windowed LFU-K). Runs on CPU with no model or GPU; large trace data stays machine-local and git-ignored with fetch instructions in the tool README. Behavioral unit tests (`tests/python/unit/test_cache_simulator_*.py`) cover the simulation core, checking Belady optimality against an independent brute-force offline optimum and cross-checking all three cache implementations over fuzzed traces. A `convert_sglang_traces.py` helper turns SGLang `per_token` expert-distribution dumps into both simulator formats (physical→logical remap plus the `.pkl` export), with a SGLang capture-and-conversion runbook (`collect_traces_sglang.md`) and agent runbooks for researching cache/prefetch baselines and accelerating the simulator (`auto_research_policies.md` plus `prompts/`). Documented in `docs/benchmarking.md`.
- Qwen3.5-MoE (`Qwen3_5MoeForConditionalGeneration`, `Qwen3_5MoeForCausalLM`) paged-attention serving (#270): a `Qwen3_5MoePagedAttention` shim routes the gated full-attention layers through the paged KV cache. The cache is sized and keyed by full-attention layer only. The serving runner carries the Gated DeltaNet conv and recurrent state per sequence (host-resident while swapped out, released on finish, abort, or recompute reset, and reserved in the memory budget), and applies M-RoPE positions and rope deltas for image and video prompts. Before this, multi-token decode output for this family was degenerate. Prefix reuse, chunked prefill, and speculative serving execution stay disabled for hybrid linear-attention models with stable reasons.
- Opt-in phase-specific expert admission, prefetch, eviction, and telemetry over one shared residency manager, with the legacy policy retained by default and one-flag rollback requiring no offload-store migration.
- Opt-in `int8_sym` KV-cache storage format (`kv_cache_format`, default `native`): symmetric INT8 payload plus one FP16 scale per `(layer, page, KV head, token)`, with a native CUDA decode kernel, a validated FP32 dequantized SDPA fallback, single-owner `LayeredPagedKVStore` lifecycle, capability-gated selection with visible native/MLA fallback, numerical/logit/storage quality gates, a long-context A/B benchmark matrix, and a one-setting (`--kv-cache-format native`) rollback. No universal low-bit KV support is claimed.
- Documentation hub at `docs/README.md` for users, operators, contributors, and project-history readers.
- DFlash documentation that distinguishes direct batch-1 greedy and sampled draft/verify from the greedy-gated `MoE.generate` and serving integrations, explains the current batch>1 greedy-only constraint, and limits continuous-batching and route-ahead claims to validated paths.
- Opt-in asynchronous hierarchical KV swap between GPU and bounded pinned host memory, with event-driven scheduling, cancellation tombstones, telemetry, and a synchronous fallback. The external tier remains an interface only; no external or distributed store is implemented.
- Opt-in correctness-preserving prefix KV reuse in the OpenAI continuous-batching path (`--enable-prefix-caching`, default off). Reuse is gated on a supported Qwen3 paged-attention layer registry plus real FlashInfer prefill/decode; unsupported runtimes execute the unchanged cold path with a stable disabled reason. Reuse requires exact namespace/parent-path/token identity, admits every sequence in a request group atomically with pinned leases before eviction, copies shared partial tails on write, publishes only successfully committed full prompt blocks, invalidates on `/v1/reload`, and excludes reused-prefix and non-cold requests from DFlash delegation. `--prefix-cache-max-entries` (default 1000) bounds the index; `moe_prefix_cache_*` metrics and `/admin/stats` expose lifecycle counters. Motivated by SGLang RadixAttention (<https://lmsys.org/blog/2024-01-17-sglang/>) and vLLM automatic prefix caching (<https://docs.vllm.ai/en/stable/examples/features/automatic_prefix_caching>); no universal speedup is claimed. Rollback is removing the flag and restarting.
- Unified DFlash `SessionDriver`/`SpecSession` protocol, capability-selected bare-HF and rich backends, shared trace evidence, and per-request sampling streams.
- Direct bare-HF batch-1/batch>1 greedy, sampled, and mixed-row execution with dense reconstruction, right-padded output, and `last_generated_lengths`.
- Stage 4a persistent serving sessions and default-off Stage 4b engine-owned DeepSeek V2/V3 MLA pages for eligible greedy batch-1 requests.
- No-download unified-execution benchmark/validator and compatibility assertions that fail closed on sampling, ordering, cache invariant, or ownership failures.
- Documented GLM-5.3 (`zai-org/GLM-5.3`) as running through the existing GlmMoeDsa path (same base as GLM-5.2), with a config-resolution regression test (`tests/python/unit/test_glm53_registry.py`) and a chat-template contract test for `reasoning_effort`/`clear_thinking` (`tests/python/unit/test_glm53_chat_template.py`). Validated by a real-checkpoint generate smoke (`MOE_GLM_SMOKE_MODEL=zai-org/GLM-5.3`).
- GLM-5.3-Flash (`zai-org/GLM-5.3-Flash`) glm5_next family support: guarded registry entry, nested text_config parsing, `SyncGlm5NextMoEBlock`, and offload runtime wiring (routed FP8 experts offloaded; KDA/DSA/mHC/vision resident, text-only). Validated by a tiny GPU harness (generate + determinism, serving completions, FP8 store/reload parity) and a real-checkpoint generate smoke (`scripts/smoke_glm5_next.py`).
- Hy3-FP8 (`hy_v3`) support: per-tensor static FP8 expert offloading via scalar→block-grid scale expansion over the GLM FP8-in-store path; whole non-expert namespace resident; MTP layer excluded (#283).

### Fixed

- `_load_resident_shared_experts` now buffers resident weight fusions across safetensors shards. GLM-5.3-Flash splits a KDA layer's `q/k/v_conv1d.weight` across shards, which the previous per-shard fusion match never assembled, raising "Missing shared_experts weights" on the real checkpoint.
- glm5_next is excluded from the native engine path (like `glm_moe_dsa`/`minimax_m3_vl`); its hybrid KDA + DSA attention is not representable in the native paged backend, which previously crashed `MoE` init deriving an MLA kv-cache head_dim from the composite config.

### Changed

- Qwen3.5-MoE keeps its text backbone resident and offloads routed experts only on the text-only path.
- GLM-5.2-FP8 keeps routed FP8 experts in the host store while non-routed FP8 weights dequantize to BF16 on load.
- Root README is now a concise discovery surface and points readers to the docs hub, model compatibility, DFlash, serving, troubleshooting, architecture, and changelog.
- Release notes are split out of README and tracked here instead of being presented as shipped releases.
- Package version is now derived from git tags by setuptools-scm and written to `moe_infinity/_version.py` at build time, replacing the manual `MOEINF_VERSION`, `NIGHTLY_BASE_VERSION`, and hardcoded `setup.py`/`__init__.py` version strings.
- GitHub Releases are now the authoritative distribution channel: tagged
  releases attach Python 3.10–3.12 CUDA wheels and an sdist, nightlies retain
  build artifacts in Actions, and PyPI mirroring is an optional manual step.

### Deprecated

- `MoE.generate()` emits `DeprecationWarning` in the current code and is scheduled for removal. It remains documented as the current in-process synchronous transition path; `MoE.serve()` is recommended for continuous-batching HTTP serving and is not a drop-in in-process return API. No removal version is announced.

### Fixed

- Serving-path DFlash now truncates KV cache to the committed prefix after each verify step, so emitted and cached tokens stay aligned.
- GPT-OSS resident-load path now materializes MXFP4 blocks, scales, router, biases, and attention sinks instead of leaving placeholder tensors in place.
- GLM FP8 store and reload parity now stays stable across fresh stores and reloads.
- `MOE_DISABLE_FUSED_KERNELS=1` no longer raises `TypeError` on decode: `fused_decode_attention` now has its own eager fallback instead of delegating to `paged_attention_fwd`, which takes a different KV cache layout and a required `num_kv_heads` argument.
- The ordinary non-speculative continuous-batching step now runs the model once per scheduler step instead of three times. `step()` reuses the logits and `SamplerOutput` it already computed for KV/prefix metadata commit and sequence update, removing ~3x redundant forward compute and the repeated expert-tracing, KV, and prefix-cache side effects that inflated serving-path expert-access traces (#276). Greedy output is unchanged; seeded `temperature > 0` output may differ from prior releases because the redundant per-step sampling draw is gone.

### Known Limitations

- `MoE.generate()` retains narrower compatibility behavior than the direct API and does not generally expose sampled batch > 1.
- Stage 4b is default-off, greedy batch-1 DeepSeek V2/V3 only, resident-only, and non-preemptible; all ineligible and hybrid/Qwen paths use Stage 4a.
- No real DeepSeek or Qwen DFlash pair is claimed. GPT-OSS has named valid pairs but no executor route-ahead path.
- Required GPU fixture skips/unavailability are not rollout success; these Unreleased notes do not claim a release or unrun GPU validation.
- Multi-node distributed inference is still unsupported.
