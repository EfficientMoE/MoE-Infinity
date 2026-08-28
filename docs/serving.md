# OpenAI-compatible serving guide

Authoritative reference for the async OpenAI-compatible server.

`MoE.generate()` remains the current in-process synchronous path, but it emits
`DeprecationWarning` and is scheduled for removal. `MoE.serve()` is recommended
for continuous batching; it starts an async HTTP service and is not a drop-in
replacement for an in-process return value.

## Quick Start

> **Security:** The parser defaults to `--host 0.0.0.0`, and authentication is
> disabled when neither `--api-key` nor `MOE_API_KEYS` is configured. On an
> untrusted, shared, or cloud host, bind to `127.0.0.1` or configure an API key
> before exposure. Completion routes and privileged `/admin/stats`, `/v1/config`,
> and `/v1/reload` endpoints share this authentication posture.

```bash
python -m moe_infinity.entrypoints.openai.api_server_v2 \
    --host 127.0.0.1 \
    --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --offload-dir ./offload_dir \
    --device-memory-ratio 0.5 \
    --kv-cache-ratio 0.25 \
    --max-batch-size 8
```

```bash
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-V2-Lite-Chat","prompt":"Hello","max_tokens":32}'
```

For DFlash speculative decoding, use a validated pair such as `openai/gpt-oss-20b` with `z-lab/gpt-oss-20b-DFlash`, or leave `--speculative-draft` unset for non-DFlash targets.

For one-host multi-GPU ownership and `CUDA_VISIBLE_DEVICES` ordering, see [Single-server multi-GPU](multi-gpu.md).

## CLI Reference

Stable options from `api_server_v2.py`:

| Option | Default | Meaning |
|---|---:|---|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | Listen port |
| `--model` | required | Model checkpoint or local path |
| `--offload-dir` | required | Expert offload directory |
| `--speculative-draft` | none | DFlash drafter checkpoint |
| `--device-memory-ratio` | `0.75` | GPU memory reserved for expert cache |
| `--kv-cache-ratio` | `0.25` | Remaining memory reserved for KV cache |
| `--max-batch-size` | `32` | Max concurrent sequences per batch |
| `--api-key` | none | Comma-separated Bearer keys |
| `--rate-limit` | `0` | Requests/minute/key; `0` disables |
| `--max-waiting-requests` | `0` | Queue depth backpressure threshold; `0` disables |
| `--max-n` | `16` | Cap for parallel sampling `n` / `best_of` |
| `--enable-prefix-caching` | off | Enable correctness-preserving prefix KV reuse (Qwen3 + FlashInfer) |
| `--prefix-cache-max-entries` | 1000 | Max prefix-index entries (startup-only, >= 1) |
| `--startup-timeout` | none | Startup watchdog timeout, seconds |
| `--decode-step-timeout` | none | Decode watchdog timeout, seconds |
| `--enable-pyspy-dump` | off | Scaffolded flag; currently accepted and stored, but no py-spy dump is triggered |
| `--enable-contextpilot` | off | Enable ContextPilot middleware |
| `--contextpilot-debug` | off | Enable ContextPilot fault-injection/admin hooks |

Internal / deprecated:

- `MoE.generate(...)` emits `DeprecationWarning` and is scheduled for removal;
  `MoE.serve(...)` is the continuous-batching HTTP transition path, not a
  drop-in in-process call replacement.
- `--max-waiting-requests` and `--max-n` feed internal module state used by middleware.

## Python Startup

```python
from moe_infinity import MoE

model = MoE("deepseek-ai/DeepSeek-V2-Lite-Chat", {
    "offload_path": "./offload_dir/deepseek-v2-lite",
    "device_memory_ratio": 0.5,
})
model.serve(host="127.0.0.1", port=8000, offload_dir="./offload_dir")
```

Binding the Python startup path to `0.0.0.0` has the same exposure and auth
implications as the CLI. Configure `api_key` or `MOE_API_KEYS` before using a
non-loopback bind on an untrusted network.

`MoE.serve()` accepts the same serving knobs as the CLI for host, port, memory ratios, batch sizing, the prefix-cache flag, offload path, and DFlash drafter setup. With `--enable-prefix-caching` on a supported Qwen3 + FlashInfer runtime the flag activates request-path prefix KV reuse; otherwise the cold path runs unchanged.

## Request Fields and Streaming

### `/v1/completions`

Implemented request fields:

- `model`, `prompt`
- `max_tokens`, `temperature`, `top_p`
- `n`, `best_of`
- `stream`, `stop`, `logprobs`, `response_format`

Accepted but currently no-op / unsupported:

- `echo`, `suffix`, `user`

Accepted but currently no-op:

- `presence_penalty`, `frequency_penalty`, `logit_bias`

### `/v1/chat/completions`

Implemented request fields:

- `model`, `messages`
- `max_tokens`, `temperature`, `top_p`
- `n`, `stream`, `stop`
- `logprobs`, `top_logprobs`
- `response_format`

Accepted but currently no-op / unsupported:

- `user`

Accepted but currently no-op:

- `presence_penalty`, `frequency_penalty`, `logit_bias`

### Streaming

- Streaming uses SSE.
- The terminal event is `data: [DONE]`.
- `finish_reason` is `stop` on EOS or stop-sequence termination, `length` on `max_tokens`, and `error` for JSON-object validation failures.
- Parallel sampling is constrained to `n <= --max-n` and `best_of <= --max-n`.
- Streaming with parallel sampling requires `n=1` and `best_of=1`.

## Authentication and Rate Limiting

- API key source: `--api-key` first, then `MOE_API_KEYS` if no CLI keys are provided.
- Format: `Authorization: Bearer <key>`.
- If no keys are configured, auth is disabled.
- Exempt routes: `GET /health`, `GET /metrics`, `GET /v1/models`.
- Protected routes: completions, chat completions, `/admin/stats`, `/v1/config`, `/v1/reload`.
- `--rate-limit` is per-key requests/minute. `0` disables limiting.
- Rate-limit failures return `429`.

## Backpressure and Parallel Sampling

- `--max-waiting-requests` applies queue backpressure on the scheduler.
- When the queue reaches the threshold, the server returns `503` with a queue-full error.
- `--max-n` caps `n` and `best_of`.
- `n` must be positive; `best_of` must be positive and at least `n`.

## Prefix Caching

Prefix KV reuse is opt-in and default off. It reuses the physical KV of an exact
shared prompt prefix so a warm request only recomputes its divergent suffix,
with cold/warm equivalence.

- `--enable-prefix-caching` enables reuse; `--prefix-cache-max-entries` (default
  `1000`, minimum `1`, startup-only) bounds the prefix index.
- **Supported scope.** Reuse activates only on the Qwen3 paged-attention path
  with a complete per-layer registry, matching KV geometry, and real FlashInfer
  prefill/decode. Any other runtime keeps the existing cold path and reports a
  stable `prefix_cache_disabled_reason` (for example
  `prefix-aware-prefill-unavailable`, `incomplete-paged-layer-registry`, or
  `kv-store-binding-mismatch`).
- **Exact identity.** A hit requires exact compatibility namespace (model,
  tokenizer, adapter, dtype, KV geometry, attention backend/layout, position
  config, runtime epoch), exact parent-entry path, and exact token blocks.
  SHA-256 is only a bucket accelerator and never substitutes for token equality.
- **Layer completeness.** One physical block names K/V for every layer in a
  single validated layered store; export/import, checkpoint/restore, copy-on-
  write, publication, and eviction operate across all layers or fail closed.
- **Atomic pinned admission.** All sequences in a request group (including
  `n>1`) are pinned with leases before any eviction, then admitted together or
  not at all; failed admission restores tables, free count, refcounts, statuses,
  and open-lease count.
- **Committed-range publication.** Only fully committed, block-complete prompt
  ranges are published after a successful forward; failed or partial chunks and
  DFlash verify tails are never published.
- **Copy-on-write.** Indexed blocks are immutable; any write to a shared partial
  tail first copies every layer to a private block.
- **Reload and rollback.** `/v1/reload` invalidates the prefix cache once after a
  successful module reload. Rollback is removing `--enable-prefix-caching`,
  restarting, and confirming `moe_prefix_cache_active 0`; there is no persisted
  state to migrate.
- **DFlash exclusion.** Reused-prefix and non-cold requests use ordinary paged
  execution and are excluded from DFlash delegation.
- **Observability.** `/admin/stats` and zero-safe `moe_prefix_cache_*`
  Prometheus metrics expose enabled/active/disabled-reason, entries, open leases,
  hits, matched tokens, and invalidations.

Motivated by SGLang RadixAttention
(<https://lmsys.org/blog/2024-01-17-sglang/>) and vLLM automatic prefix caching
(<https://docs.vllm.ai/en/stable/examples/features/automatic_prefix_caching>);
no universal speedup is claimed. Prefix reuse does not change expert ownership;
the multi-GPU guide covers that layout separately.

## DFlash Serving

Use DFlash with a validated pair such as:

```bash
python -m moe_infinity.entrypoints.openai.api_server_v2 \
    --model openai/gpt-oss-20b \
    --offload-dir ./offload_dir \
    --speculative-draft z-lab/gpt-oss-20b-DFlash
```

Startup validates the drafter/target pair: hidden size, vocab size, mask-token bounds, target layer IDs, and drafter `fc` shape.

Delegation is server-wide and only applies when a request is:

- a fresh singleton prefill request
- greedy (`temperature=0`, no sampling)
- `top_k <= 0`
- `top_p >= 1.0`
- `repetition_penalty == 1.0`
- `logprobs <= 0`
- no stop strings
- within the current step token budget

The exact gate is implemented in [`moe_infinity/serving/engine.py`](../moe_infinity/serving/engine.py) and also requires batch==1, no prior output tokens, and `max_tokens <= scheduler.max_tokens_per_step`.

The delegated path runs the speculative loop and emits tokens normally through SSE.

Route-ahead is internal to the DFlash verify path and is used only when speculative delegation is active.

## Operational Endpoints

| Endpoint | Method | Auth | Body | Response / scope | Production note |
|---|---|---|---|---|---|
| `/v1/models` | GET | exempt | none | loaded model list | safe for readiness checks |
| `/health` | GET | exempt | none | `STARTING` / `HEALTHY` / `UNHEALTHY` | safe to expose |
| `/metrics` | GET | exempt | none | Prometheus text metrics | safe to scrape internally |
| `/admin/stats` | GET | protected | none | queue, batch, cache, engine stats | keep private |
| `/v1/config` | GET | protected | none | current runtime config | admin-only |
| `/v1/config` | POST | protected | JSON updates | only `max_batch_size` and `max_tokens_per_step` are mutable | admin-only |
| `/v1/reload` | POST | protected | reload request | hot-reload Python modules | admin-only |

The optional `/contextpilot/toggle`, `/contextpilot/inject-fault`, and
`/contextpilot/status` routes are intentionally cataloged in
[`docs/contextpilot/README.md`](contextpilot/README.md) rather than this core
serving table. The fault-injection route is debug-only and requires
`--contextpilot-debug`.

## Watchdogs and Diagnostics

- `--startup-timeout` is disabled by default.
- When enabled, startup timeout causes a hard exit.
- `--decode-step-timeout` is disabled by default.
- When enabled, decode timeout marks the server unhealthy instead of exiting.
- `--enable-pyspy-dump` is currently a scaffolded no-op flag: it is accepted and stored in watchdog config, but the watchdog flow does not trigger py-spy dumps yet.

If you need timeout debugging, use the watchdog logs and the troubleshooting guide rather than expecting a py-spy dump.

## Failure Modes

- `401`: invalid or missing API key.
- `429`: rate limit exceeded.
- `503`: request queue full.
- `400`: invalid `n` / `best_of` / other request validation.
- `finish_reason="error"`: JSON-object validation failure.
- Penalties and `logit_bias` are accepted by the request models but are not applied in sampling.
