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
| `--enable-prefix-caching` | off | Enable prefix-cache bookkeeping flag |
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

`MoE.serve()` accepts the same serving knobs as the CLI for host, port, memory ratios, batch sizing, the prefix-cache feature flag, offload path, and DFlash drafter setup. The flag currently enables scaffolding only; it does not activate request-path reuse.

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

- `--enable-prefix-caching` toggles the feature flag.
- The cache implementation is hash-based LRU with `block_size=16` and `max_entries=1000` by default.
- Current serving code does not wire the cache into request execution, so there is no active reuse path yet.
- Observability is internal (`hit_rate`, lookup/insert bookkeeping), not a public endpoint.

The prefix-cache scaffold does not change expert ownership; the multi-GPU guide covers that layout separately.

## DFlash Serving

Use DFlash with a validated pair such as:

```bash
python -m moe_infinity.entrypoints.openai.api_server_v2 \
    --model openai/gpt-oss-20b \
    --offload-dir ./offload_dir \
    --speculative-draft z-lab/gpt-oss-20b-DFlash
```

Startup validates structural pairing (hidden size, vocabulary, mask-token
bounds, target layers, block constraints, and drafter shape) separately from
executor/route-ahead reachability.

The persistent path creates one canonical session per eligible sequence. It
preserves the request's temperature, top-k, top-p, budget, EOS set, and
request-scoped generator. It does not silently turn sampled requests into
greedy requests. Unsupported grammar/guided/logit-bias metadata, penalties,
logprobs, or stop strings use the standard serving fallback before drafting.
That fallback is not evidence of sampled serving.

Two cache execution contexts are observable in `/admin/stats`:

- `temporary_dynamic` is the Stage 4a compatibility mode. It keeps a temporary
  private DynamicCache while the engine owns scheduling, callbacks,
  cancellation, and request accounting. Sampled sessions and ineligible model
  layouts remain here; this is not sampled paged-MLA serving.
- `paged_mla` is the Stage 4b default-off target enabled by
  `enable_deepseek_mla_paging=True`. It is restricted to eligible greedy
  batch-1 DeepSeek V2/V3 MLA sessions. The engine owns packed latent/rope target
  pages; the draft cache is separate. Admission is bounded by
  `max_resident_paged_speculative_sessions` (default `1`) and must leave at
  least `min_free_mla_blocks_after_admission` free blocks (default `1`) after
  reserving the block-rounded peak for the full declared
  `prompt + max_tokens` budget plus up to `DFlash block_size - 1` transient
  verify tokens. Active sessions' committed and transient headroom that is not
  yet allocated is included. A rejected eligible request immediately uses
  `temporary_dynamic`; it does not wait for a paged seat, and its sampling
  parameters are unchanged. That dense Stage 4a fallback owns a private target
  cache and can therefore increase total GPU memory use even though it consumes
  no MLA pages.

Paged MLA is currently resident-only and has no preemption/swap implementation.
The scheduler does not preempt DRAFT/VERIFY sessions. Qwen and hybrid layouts
fall back to Stage 4a; hybrid paged rollback is not claimed. Cancellation after
an in-flight backend call releases session resources, and per-sequence page
ownership prevents one cancellation or rollback from truncating another row.
Completion/cancellation frees ownership, so a later request can be admitted.
`/admin/stats` reports active paged sessions, current free blocks, configured
limits, and counters for `admitted`, `session_cap`, `free_block_reserve`, and
`ineligible` decisions. `begin_failed` is recorded only when adapter/session
construction fails; `admitted` increments only after construction succeeds.

The guard is block-based admission control, not a general fairness proof. It
does not preempt or swap admitted sessions. External cache consumers can still
invalidate reserved headroom; such allocator failures clean up the affected
request and are currently re-raised by the engine step.

There is no real DeepSeek DFlash target/drafter pair validation in the repo.
Stage 4b's tiny/local DeepSeek adapter tests establish ownership and attention
metadata only. GPT-OSS has named valid pairs, but its resident expert path has
no executor route-ahead. Qwen evidence is tiny-fixture only.

Route-ahead is observer-only. Pairing evidence, executor reachability,
prefetch-fired evidence, and cache ownership are reported as separate facts.
See [DFlash unified execution](dflash.md) for direct batching, RNG caveats,
benchmarks, and exact CPU/GPU gates.

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
