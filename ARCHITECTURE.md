# MoE-Infinity Architecture

This document is a map for new contributors. Read it first, then dive into the
code. Everything in this doc should remain true as the codebase evolves — if
you change architectural layout, update this file in the same commit.

## 1. What MoE-Infinity Does

MoE-Infinity is a Python + C++ library for running Mixture-of-Experts (MoE)
inference on memory-constrained GPUs. Its core trick is **expert offloading**:
- Expert weights live in host (CPU) memory or SSD.
- At runtime, the router picks which experts a token needs; the engine fetches
  those experts to GPU just in time.
- An activation-aware cache keeps hot experts resident so most tokens never
  touch slow storage.

On top of the runtime there are two serving paths:
1. A synchronous HuggingFace-compatible `MoE` class (`MoE.generate(...)`).
2. An async OpenAI-compatible HTTP server (`api_server_v2.py`) with continuous
   batching, paged KV cache, and streaming.

## 2. Module Map

All Python source lives under `moe_infinity/`. Two C++/CUDA source trees
support it: `core/` (the native offload engine) and `csrc/` kernels that get
compiled into the extension modules you see as `_engine.so`, `_kv_cache.so`,
etc.

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
│   │                          FlashInfer, placeholder)
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
├── engine/              Synchronous generation path (powers MoE.generate)
│   ├── generation_loop.py     GenerationEngine: token-by-token loop
│   ├── scheduler.py           Request scheduler with block-level KV allocation
│   ├── request_manager.py     Thread-safe request lifecycle
│   ├── types.py               Request, Sequence, SamplingParams, status enums
│   ├── transfer_types.py      TransferRequest, TransferPriority, TransferType
│   ├── unified_transfer_scheduler.py  Coordinates expert + KV transfers
│   └── kv_cache_offload_coordinator.py  Orchestrates KV offload to CPU/SSD
│
├── serving/             Async continuous-batching path (powers api_server_v2)
│   ├── engine.py              ContinuousBatchingEngine: async request loop
│   ├── scheduler.py           SequenceGroup-level scheduler with paged KV
│   ├── model_runner.py        Runs a prefill/decode step for a batch
│   ├── batch.py               BatchBuilder + SchedulerOutput
│   ├── kv_cache.py            PagedKVCache, BlockAllocator, BlockTable
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

MoE-Infinity currently has **two separate scheduling paths**. This is
intentional today but should be unified in a future refactor (see
[Future Work](#6-future-work)).

### Path A — Synchronous (`engine/`)

Used by `MoE.generate()`. Token-by-token generation, one request at a time.

```
User code
   │
   ▼
MoE.generate()                            (entrypoints/big_modeling.py)
   │
   ▼
GenerationEngine.generate()               (engine/generation_loop.py)
   │
   ├─► Scheduler.schedule_step()          (engine/scheduler.py)
   │      allocates KV blocks
   │      produces SchedulerOutput
   │
   ├─► model.forward(...)                 HuggingFace model with Sync* MoE blocks
   │      Sync* block calls
   │      DistributedExpertExecutor       (distributed/expert_executor.py)
   │            ↓
   │      native ExpertDispatcher         (_engine.so, core/)
   │            ↓
   │      fetches experts, runs MLPs, merges
   │
   └─► Sampler picks next token           (serving/sampler.py — shared)
```

### Path B — Async Continuous Batching (`serving/`)

Used by `api_server_v2` and `MoE.serve()`. Many requests in flight, paged KV
cache, preemption, streaming.

```
HTTP request
   │
   ▼
FastAPI handler                           (entrypoints/openai/api_server_v2.py)
   │
   ▼
ContinuousBatchingEngine.add_request()    (serving/engine.py)
   │
   ▼  (async loop)
Scheduler.schedule()                      (serving/scheduler.py)
   │   sorts WAITING / PREFILL / DECODE queues
   │   allocates paged KV blocks
   │   emits SchedulerOutput with separate prefill & decode batches
   ▼
BatchBuilder.build()                      (serving/batch.py)
   │   packs sequences into contiguous tensors
   ▼
ModelRunner.run_step()                    (serving/model_runner.py)
   │   forward pass, calls same Sync* MoE blocks + native engine
   ▼
Sampler.sample()                          (serving/sampler.py)
   │
   ▼
StreamManager → SSE response              (serving/stream.py)
```

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

1. **Intake.** `api_server_v2` validates the request, tokenizes the prompt,
   and calls `engine.add_request(...)`, which creates a `SequenceData`.
2. **Scheduling.** On each async tick, `Scheduler.schedule()` decides which
   sequences to prefill, which to decode, and which to preempt. It allocates
   paged KV blocks through `PagedKVCache`.
3. **Batching.** `BatchBuilder` assembles input tensors, attention metadata,
   and expert routing metadata for the step.
4. **Forward pass.** `ModelRunner` runs one prefill or decode step via the
   HuggingFace model; `Sync*MoeBlock` layers call
   `DistributedExpertExecutor.dispatch_local()` which hands work to the native
   dispatcher. Experts are fetched on-demand with prefetch hints from
   `ExpertPrefetcher` and the tracer.
5. **Sampling.** `Sampler` applies temperature, top-p, top-k, and stop rules.
6. **Streaming.** `StreamManager` pushes partial deltas to any open SSE
   clients. The FastAPI response emits OpenAI-shaped chunks.
7. **Termination.** When a sequence finishes, the scheduler releases its KV
   blocks, the expert tracer records the activation pattern, and
   `EvictionSync` notifies optional prompt-optimization layers.

## 5. Public API Surface

The public API is intentionally small.

From the top-level package:
- `moe_infinity.MoE` — the HuggingFace-style wrapper class.
- `moe_infinity.OffloadEngine` — lower-level engine (rarely needed directly).
- `moe_infinity.__version__`.

From the OpenAI server:
- `python -m moe_infinity.entrypoints.openai.api_server_v2 --help`
- Endpoints: `/v1/completions`, `/v1/chat/completions`, `/health`, and
  optional `/contextpilot/*` admin endpoints.

Everything under `engine/`, `serving/`, `runtime/`, `memory/`, `distributed/`,
`kernel/`, and `models/` is **internal**. Import paths may change without
notice. If you need something from these modules from outside the repo, open
an issue first.

## 6. Future Work

- **Unify `engine/` and `serving/`.** Today the synchronous and async paths
  are two independent schedulers with duplicate data structures (`Sequence`
  vs `SequenceGroup`, `SchedulerOutput` vs `SchedulerOutput`, etc.). A future
  refactor should make `MoE.generate()` a synchronous facade over the
  continuous batching engine so there is only one scheduling code path.
- **Expand `distributed/` tests.** The distributed module has smoke tests
  only (see `tests/python/unit/test_distributed_smoke.py`). Deeper coverage
  requires a multi-process CUDA harness.
- **Multi-node distributed inference.** The current distributed module only
  supports single-host multi-GPU via NCCL; cross-host RPC scaffolding exists
  but is not production-tested.

## 7. Where to Look When …

| Symptom | Start here |
|---|---|
| Adding a new MoE model | `models/` — add a `Sync<Model>MoeBlock` wrapper, register it in `runtime/model_offload.py`, add the model type to `common/constants.py` |
| Changing how experts are fetched | `distributed/expert_executor.py` + `core/parallel/expert_dispatcher*` |
| Changing cache eviction | `memory/offloading_policy.py` (LRU/ARC) + `memory/expert_priority_score.py` |
| Changing request scheduling (async path) | `serving/scheduler.py` |
| Changing request scheduling (sync path) | `engine/scheduler.py` |
| Changing the OpenAI API surface | `entrypoints/openai/api_server_v2.py` + `entrypoints/openai/protocol.py` |
| Changing attention kernels | `runtime/attention_backend.py` + `kernel/paged_attention_ops.py` |
| Profiling a slow path | `profiling/io_profiler.py` + NVTX ranges in `distributed/expert_executor.py` |
