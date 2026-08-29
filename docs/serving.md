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

## KV swap lifecycle and recovery

KV transfer state is independent from request `SequenceStatus`:

```text
GPU_RESIDENT -> SWAP_OUT_IN_FLIGHT -> HOST_RESIDENT
HOST_RESIDENT -> SWAP_IN_IN_FLIGHT -> GPU_RESIDENT
SWAP_OUT_IN_FLIGHT / SWAP_IN_IN_FLIGHT -> CANCEL_PENDING -> CANCELLED
HOST_RESIDENT / SWAP_IN_IN_FLIGHT -> FAILED -> reprefill recovery
```

The scheduler calls `poll_transfers()` once at the start of a scheduling pass
and aggregates all members of a request group before changing queues. Group
phases are `OUT_IN_FLIGHT`, `HOST_RESIDENT`, `IN_IN_FLIGHT`,
`ROLLBACK_IN_FLIGHT`, and `REPREFILL_PENDING`. Partial completion never makes a
member runnable. Successful H2D restores each member's prior status; terminal
metadata, checksum, or retry failure first removes cache transfer records and
then separately moves the group from `SequenceStatus.SWAPPED` to
`SequenceStatus.WAITING` for reprefill.

The producer stream happens-before the transfer stream, and an early consumer
waits on the completion event. A ticket retains its stream, event, and staging
tensors until `retire` succeeds. Cancellation removes scheduler visibility but
keeps a generation-keyed tombstone, EVICTING/RESTORING blocks, host lease, and
ticket until DMA retires. Shutdown stops admission, synchronizes every event,
retires tickets, finalizes blocks and leases, closes transfer streams, and only
then releases the pool. Exceeding the shutdown warning deadline does not permit
unsafe reclamation.

`/v1/reload` reloads Python modules; it does not migrate live KV state. Drain
and replace/restart the old engine for model changes.

### KV swap telemetry

`/admin/stats` exposes the `kv_swap` object and `/metrics` exports:

- `moe_kv_swap_inflight`, `moe_kv_swap_inflight_bytes`
- `moe_kv_swap_retiring_records`, `moe_kv_swap_host_resident`
- `moe_kv_swap_host_bytes`, `moe_kv_swap_host_capacity_bytes`
- `moe_kv_swap_backpressure_total`
- `moe_kv_swap_out_completed_total`, `moe_kv_swap_in_completed_total`
- `moe_kv_swap_failures_total{direction="out|in"}`
- `moe_kv_swap_bytes_total{direction="d2h|h2d"}`
- `moe_kv_swap_duration_seconds_sum{direction="d2h|h2d"}`

Durations are observed completion latency from monotonic submission to event
observation, not pure PCIe time. Metrics contain no sequence IDs, request IDs,
tokens, or checksums.

### External tier and non-goals

`ExternalKVStore` is a future pinned-host-to-external seam. This release has no
external-store implementation, distributed storage, SSD/RDMA/object-store
backend, multi-node protocol, or KV quantization. The
[Mooncake paper](https://arxiv.org/abs/2407.00079) motivates decoupling transfer
control from storage tiers; it is not a dependency or performance claim.
In short, there is no external store implementation in this change.
In short, there is no external storage implementation in this change.

### Rollout and rollback

1. **Stage 0:** keep default sync; land CPU/CUDA correctness tests and telemetry.
2. **Stage 1:** enable async on one canary; alert on failures, checksum failures,
   backpressure, pinned utilization, and p99 swap latency.
3. **Stage 2:** expand async opt-in only after workload-specific A/B review;
   retain immediate restart/drain rollback to sync.
4. **Stage 3:** consider changing defaults only in a separate change backed by
   production data.
Rollback requires draining/restarting with `kv_swap_mode="sync"`; never change
the backend while transfers are in flight.
