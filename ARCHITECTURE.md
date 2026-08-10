# MoE-Infinity Architecture

This document is a map for new contributors. Read it first, then dive into the
code. Everything in this doc should remain true as the codebase evolves. If you
change the architectural layout, update this file in the same commit.

## 1. What MoE-Infinity Does

MoE-Infinity is a Python + C++ library for running Mixture-of-Experts (MoE)
inference on memory-constrained GPUs. Its core trick is **expert offloading**:
- Expert weights live in host (CPU) memory or SSD.
- At runtime, the router picks which experts a token needs; the engine fetches
  those experts to GPU just in time.
- An activation-aware cache keeps hot experts resident so most tokens never
  touch slow storage.

On top of the runtime there are two serving paths:
1. The current synchronous HuggingFace-compatible path through the deprecated
   `MoE.generate(..., speculative_draft=...)` method. It emits
   `DeprecationWarning` and is scheduled for removal.
2. An async OpenAI-compatible HTTP server (`api_server_v2.py`) with continuous
   batching, paged KV cache, and streaming.

## 2. Module Map

All Python source lives under `moe_infinity/`. Two native source trees support
it:
- `core/`, the C++/CUDA offload engine (C++ in `core/**/*.cpp`, CUDA in
  `core/**/*.cu`, pybind bindings under `core/python/`).
- `extensions/kernel/`, standalone CUDA kernels (fused MoE MLP, activation,
  top-k softmax, paged attention, and the `v4_fp4/` FP4 dequant path).

These compile into the extension modules you see as `_engine.so`,
`_kv_cache.so`, `_paged_attn.so`, `_store.so`, `_v4_fp4.so`, and `_marlin.so`
(the exact source-to-module mapping is defined in `setup.py`).

```
moe_infinity/
├── entrypoints/         Public entry points
│   ├── big_modeling.py  MoE class: HuggingFace-style API
│   └── openai/
│       ├── api_server_v2.py   OpenAI-compatible HTTP server
│       └── protocol.py        Request / response models
│
├── runtime/             Model loading, hooks, attention backends
│   ├── model_offload.py       OffloadEngine: loads model, monkey-patches MoE
│   │                          block classes with Sync* wrappers, sets up
│   │                          expert tracing
│   ├── attention_backend.py   Attention backend dispatch (SDPA, FlashAttention,
│   │                          FlashInfer, fallback backend)
│   ├── hooks.py               Forward-pass hooks for tracing and prefetching
│   └── compile.py             Optional torch.jit compilation of expert MLPs
│
├── models/              Model wrappers
│   ├── mixtral.py, deepseek_v2_wrapper.py, ...
│   │                          Each wrapper defines a Sync<Model>MoeBlock that
│   │                          replaces the upstream HF MoE block at runtime.
│   │                          Wrappers import expert / gate classes from
│   │                          upstream `transformers`.
│   └── model_utils.py         Rotary embedding helpers
│
├── engine/              Deprecated synchronous generation path (powers MoE.generate)
│   ├── generation_loop.py     GenerationEngine, spec_strategy seam, standard fallback
│   ├── scheduler.py           Request scheduler with block-level KV allocation
│   ├── request_manager.py     Thread-safe request lifecycle
│   ├── types.py               Request, Sequence, SamplingParams, status enums
│   ├── transfer_types.py      TransferRequest, TransferPriority, TransferType
│   ├── unified_transfer_scheduler.py  Coordinates expert + KV transfers
│   └── kv_cache_offload_coordinator.py  Orchestrates KV offload to CPU/SSD
│
├── serving/             Async continuous-batching path (powers api_server_v2)
│   ├── engine.py              ContinuousBatchingEngine, async request loop, DFlash gate
│   ├── scheduler.py           SequenceGroup-level scheduler with paged KV
│   ├── model_runner.py        Runs a prefill/decode step for a batch
│   ├── batch.py               BatchBuilder + SchedulerOutput
│   ├── kv_cache.py            PagedKVCache, BlockAllocator, BlockTable
│   ├── spec_state.py          SpecDecodeState and committed-count bookkeeping
│   ├── spec_verify.py         apply_verify_step rollback helper
│   ├── memory_manager.py      GPU memory budget coordination
│   ├── sequence.py            SequenceData, SequenceStatus, SamplingParams
│   ├── sampler.py             Token sampling (temperature, top_p, top_k, stop)
│   ├── stream.py              StreamManager for SSE streaming responses
│   ├── prefix_cache.py        Prefix-cache hit detection
│   ├── validation.py          Request validation + error shaping
│   ├── health.py              /health endpoint state
│   ├── watchdog.py            Startup / decode timeout enforcement
│   ├── expert_batch.py        BatchedExpertDispatch helper
│   ├── expert_prefetch_coordinator.py   Cross-request prefetch hints
│   ├── eviction_sync.py       Request-termination → ContextPilot eviction
│   └── contextpilot_*.py      Optional prompt-optimization middleware
│
├── spec_decode/         DFlash speculative decoding and route-ahead support
│   ├── dflash.py              DFlashSpeculator, config readers, validators,
│   │                          route-ahead hook
│   ├── _route_ahead_ctx.py    Verify-time route-ahead contextvars and
│   │                          prefetch handle
│   ├── _route_ahead_stats.py  Opt-in route-ahead coverage/waste metrics
│   └── _prefetch_route.py     Exact expert-set math for route-ahead
│
├── memory/              Expert cache + KV cache memory management
│   ├── expert_tracer.py       Records expert activation history
│   ├── expert_predictor.py    Predicts next expert set
│   ├── expert_prefetcher.py   Issues prefetch requests to native engine
│   ├── expert_priority_score.py  Scoring heuristic for cache eviction
│   ├── offloading_policy.py   LRU / ARC cache policies
│   ├── kv_cache_manager.py    Python-side KV block bookkeeping
│   ├── block_pool.py          Block allocator abstraction
│   ├── cpu_block_cache.py     CPU-resident KV block staging area
│   └── memory_coordinator.py  Shared GPU memory budget between experts + KV
│
├── distributed/         Multi-GPU expert dispatch
│   ├── expert_executor.py     DistributedExpertExecutor: routes tokens to
│   │                          experts across local GPUs (and across RPC
│   │                          workers when enabled)
│   └── expert_prefetcher.py   DistributedExpertPrefetcher: cross-rank prefetch
│
├── kernel/              Custom kernels (Triton / CUDA adapters)
│   ├── router.py              Fused softmax+topk router
│   ├── sglang_adapter.py      sglang topk_softmax adapter
│   └── paged_attention_ops.py Paged attention forward ops
│
├── profiling/
│   └── io_profiler.py         Per-layer I/O timing, NVTX ranges
│
├── utils/               Configs, checkpoint paths, device helpers, HF glue
│   ├── config.py              ArcherConfig (offload settings)
│   ├── hf_config.py           parse_moe_param / parse_expert_id / etc.
│   ├── checkpoints.py         Locate safetensors / pytorch_bin files
│   ├── device.py              Device selection helpers
│   ├── async_transfer.py      Host-device transfer helpers
│   └── gptq.py                GPTQ-packed tensor detection
│
└── common/
    └── constants.py           MODEL_MAPPING_NAMES, MODEL_MAPPING_TYPES
```

## 3. Two Execution Paths

MoE-Infinity has two runtime paths. The deprecated sync path currently owns
`MoE.generate(..., speculative_draft=...)`. The recommended continuous-batching
path is the async HTTP service started by `MoE.serve()` or `api_server_v2.py`;
it is not a drop-in in-process return API.

### Path A - Deprecated synchronous path (`MoE.generate()`)

```mermaid
flowchart TD
  U[User code] --> M[MoE.generate(..., speculative_draft=...)]
  M --> R[_resolve_spec_strategy()]
  R -->|None / False / non-greedy / batch>1| S[GenerationEngine._generate_standard()]
  R -->|attach| E[GenerationEngine.spec_strategy]
  E --> D[DFlashSpeculator.generate()]
  D --> F[MoE._native_model_forward_rich()]
  F --> G[Sync* MoE blocks]
  G --> X[DistributedExpertExecutor.dispatch_local()]
  D --> K[accept, rollback, cache rewind]
  D --> C[route_ahead_context()]
  C --> P[DistributedExpertExecutor._maybe_route_ahead_prefetch()]
  C --> O[SyncGptOssMLP._observe_resident_route_ahead()]
```

Notes:
- `MoE.generate()` emits `DeprecationWarning` and is scheduled for removal. This
  diagram records the current transition path rather than a stable method contract.
- `_resolve_spec_strategy()` attaches a speculator per call. Passing `None` or `False` detaches it and the standard path runs.
- The greedy gate lives in `GenerationEngine.generate()` and `_spec_strategy_applies()`. If the request is not a singleton greedy decode, the engine uses `_generate_standard()` and the output stays on the pre-DFlash baseline.
- `DFlashSpeculator._forward_target()` uses `moe._native_model_forward_rich()` when available, so expert dispatch and any configured prefetch hook stay intact.
- The route-ahead context is verify-only. `DistributedExpertExecutor._maybe_route_ahead_prefetch()` reads the active context, computes the exact expert union from the router mask, and pins that exact set. If the context is inactive, no prefetcher is bound, or the union is empty, the legacy path is unchanged.
- `SyncGptOssMLP` stays resident. When no executor seam exists, it only records read-only route-ahead stats.

### Path B - Async continuous batching (`MoE.serve()`)

```mermaid
flowchart TD
  R[HTTP request] --> H[api_server_v2 route]
  H --> A[ContinuousBatchingEngine.add_request()]
  A --> S[Scheduler.schedule()]
  S --> B[BatchBuilder.from_scheduler_output()]
  B --> C{ContinuousBatchingEngine.step()\n_can_delegate_speculative(batch)?}
  C -->|yes| D[ContinuousBatchingEngine._step_speculative()]
  C -->|no| E[ContinuousBatchingEngine._execute_batch()]
  E --> N[Sampler.sample()]
  D --> G[DFlashSpeculator.generate()]
  D --> U[Scheduler.update_after_step(... committed_counts=...)]
  U --> Q[SequenceData + PagedKVCache]
  R --> X[StreamManager.push_token + SSE]
  X --> Y[abort_request() on disconnect or cleanup]
```

Notes:
- `Scheduler.schedule()` and `BatchBuilder.from_scheduler_output()` run before the speculative gate. `ContinuousBatchingEngine.step()` then checks `_can_delegate_speculative(batch)`; only eligible fresh singleton greedy prefill requests enter `_step_speculative()`, otherwise the normal `_execute_batch()` + sampler path runs.
- `_step_speculative()` emits one `RequestOutput` per committed token, then calls `scheduler.update_after_step(..., committed_counts={...})`.
- `Scheduler.update_after_step()` advances `SequenceData`, appends committed tokens through `PagedKVCache.append_tokens()`, and frees completed sequences with `PagedKVCache.free_sequence()`.
- `serving/spec_state.py` and `serving/spec_verify.py` define low-level tested helper contracts. `SpecDecodeState.record_verify()` and `apply_verify_step()` model committed-count bookkeeping for the rollback tests; the live `ContinuousBatchingEngine` path uses `Scheduler.update_after_step(..., committed_counts=...)` together with `PagedKVCache` directly.
- Streaming and cleanup are ordinary server behavior. `StreamManager.push_token()` emits SSE chunks, and `_completion_event_generator()` / `_chat_event_generator()` call `abort_request()` when the client disconnects.

### Shared Components

Both paths share:
- `runtime/model_offload.py` for model loading and MoE block monkey-patching
- `runtime/attention_backend.py` for attention kernel dispatch
- `memory/` for expert cache / KV cache bookkeeping
- `distributed/expert_executor.py` for expert dispatch on the GPU side
- `kernel/` for routing and attention kernels
- The native `_engine.so` / `_kv_cache.so` / `_paged_attn.so` / `_store.so`
  extensions (built from `core/`)

## 4. Request Lifecycle (Continuous Batching)

1. **Intake.** `api_server_v2` validates the request, tokenizes the prompt, and calls `engine.add_request(...)`, which creates a `SequenceData`.
2. **Scheduling.** On each async tick, `Scheduler.schedule()` decides which sequences to prefill, which to decode, and which to preempt. It allocates paged KV blocks through `PagedKVCache`.
3. **Batching.** `BatchBuilder.from_scheduler_output()` assembles packed input tensors, attention metadata, and expert routing metadata for the step.
4. **Forward pass.** `ContinuousBatchingEngine.step()` builds the batch, checks `_can_delegate_speculative(batch)`, and either enters `_step_speculative()` or falls back to `_execute_batch()` + sampling. The normal path still reaches `DistributedExpertExecutor.dispatch_local()` through the model blocks, so the same expert dispatch and prefetch hooks stay in play.
5. **Speculative delegation.** If the batch is a fresh singleton greedy prefill request, `_step_speculative()` delegates to `DFlashSpeculator.generate()` and then records committed counts through `Scheduler.update_after_step(..., committed_counts=...)`.
6. **Rollback bookkeeping.** `SpecDecodeState` and `apply_verify_step()` are low-level tested helpers for committed-count and truncation math. The live serving engine uses `Scheduler.update_after_step(..., committed_counts=...)` with `PagedKVCache` directly; these helpers describe and verify the contract but are not the live integration point.
7. **Streaming.** `StreamManager` pushes partial deltas to open SSE clients. The FastAPI response emits OpenAI-shaped chunks.
8. **Termination.** When a sequence finishes or the client disconnects, the scheduler releases its KV blocks, `abort_request()` clears callbacks and request state, and the cancel path updates request stats.

## 5. API Stability Boundaries

`__all__` only describes import convenience. It does not turn an internal module into a stable contract. The table below is the compatibility boundary.

| Surface | Stability | Intended audience | Compatibility promise |
|---|---|---|---|
| `moe_infinity.MoE`, `moe_infinity.OffloadEngine`, `moe_infinity.__version__` | Documented package surface | Users | The top-level class and package names are documented surfaces; individual methods have the lifecycle stated in their own rows or guides. |
| `MoE.generate()` | Deprecated, pending removal | Existing synchronous callers | Emits `DeprecationWarning`. It remains documented for transition and current validation only, with no compatibility promise beyond transition documentation. |
| `MoE.serve()`, `moe_infinity.entrypoints.openai.api_server_v2`, `moe_infinity.entrypoints.openai.protocol`, and the documented routes in `docs/serving.md` | Documented server surface | Operators and integrators | Request/response shapes and route behavior follow the documented server contract. |
| `moe_infinity.spec_decode.DFlashConfig`, `DFlashSpeculator`, `read_dflash_config`, `validate_pairing`, `glm_dflash_available`, `glm_dflash_drafter_for`, `validate_glm_pairing` | Experimental exported `spec_decode` surface | Power users and contributors | Exported for experimentation and integration work, but not a stable compatibility contract. Changes should still be reflected in docs/release notes. |
| `moe_infinity.spec_decode.dflash.validate_drafter`, `validate_drafter_module`, `bind_shared_weights` | Internal DFlash helpers | Contributors working in `dflash.py` | Module-level implementation details; exported only from `dflash.py`, not from the package root. |
| `moe_infinity.engine.*`, `moe_infinity.serving.*`, `moe_infinity.memory.*`, `moe_infinity.models.*`, `moe_infinity.kernel.*`, `moe_infinity.distributed.*`, `moe_infinity.runtime.*` | Internal | Contributors | No compatibility promise. Import paths, class names, and helper signatures may change. |

If a symbol is exported but not listed in the documented package surface row, treat it as convenience-only unless the docs explicitly elevate it. Internal helpers may change without notice; documented behavior is tracked through the docs and changelog/release notes.

## 6. Where to Look When …

| Symptom | Start here |
|---|---|
| Adding a new MoE model | `models/` - add a `Sync<Model>MoeBlock` wrapper, register it in `runtime/model_offload.py`, add the model type to `common/constants.py` |
| Changing how experts are fetched | `distributed/expert_executor.py` + `core/parallel/expert_dispatcher*` |
| Changing cache eviction | `memory/offloading_policy.py` (LRU/ARC) + `memory/expert_priority_score.py` |
| Changing request scheduling (async path) | `serving/scheduler.py` |
| Changing request scheduling (sync path) | `engine/scheduler.py` |
| Changing the OpenAI API surface | `entrypoints/openai/api_server_v2.py` + `entrypoints/openai/protocol.py` |
| Changing attention kernels | `runtime/attention_backend.py` + `kernel/paged_attention_ops.py` |
| Profiling a slow path | `profiling/io_profiler.py` + NVTX ranges in `distributed/expert_executor.py` |
