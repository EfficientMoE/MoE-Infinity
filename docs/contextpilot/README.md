# ContextPilot Integration Guide

## Overview

ContextPilot is an optional optimization layer for prompt-heavy workloads with overlap, such as shared-prefix RAG, repeated system prompts, and multi-turn chat. In MoE-Infinity it can run inside the OpenAI-compatible server as middleware, or deeper in the scheduling and KV stack.

## What is ContextPilot

ContextPilot reorders prompt context and removes repeated content so the model does less redundant prefill work. In practice, that means lower TTFT, better KV cache reuse, and fewer prompt tokens sent through the serving path.

## Architecture

MoE-Infinity supports two integration phases:

1. **Phase B, In-Process Middleware**
   - Runs ContextPilot logic inside `api_server_v2.py` before tokenization.
   - Best when you want lower proxy overhead and direct runtime toggles.
   - Includes runtime enable/disable control and fault injection for testing.

2. **Phase C, Deep Scheduler Integration**
   - Extends ContextPilot signals into KV allocation and request ordering.
   - Uses the CP-aware KV interface plus eviction sync hooks so terminal frees stay in sync with ContextPilot state.
   - Best gains, highest integration depth.

Core modules:

- `moe_infinity/serving/contextpilot_middleware.py`: prompt reorder, dedup, request metrics, cleanup hooks
- `moe_infinity/serving/eviction_sync.py`: maps terminal request completion and abort events to ContextPilot eviction
- `moe_infinity/serving/contextpilot_circuit_breaker.py`: protects the serving path when ContextPilot misbehaves or slows down
- `moe_infinity/serving/cp_kv_interface.py`: CP-aware KV adapter for Phase C scheduling and allocation decisions

## Installation

```bash
pip install moe-infinity[contextpilot]
```

Notes:

- Real ContextPilot requires Python 3.10+.
- The middleware gracefully disables itself when the package is unavailable.

## Phase B: In-Process Middleware

### When to use it

Use Phase B when you want ContextPilot inside the OpenAI-compatible server process, with fewer moving parts and direct runtime controls. This is the right fit once your runtime is on Python 3.10+.

### Start the server

```bash
python -m moe_infinity.entrypoints.openai.api_server_v2 \
    --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --offload-dir ./offload_dir \
    --enable-contextpilot
```

### Phase B CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--enable-contextpilot` | off | Enables ContextPilot middleware before tokenization |
| `--contextpilot-debug` | off | Enables debug-only fault injection endpoint |

### Phase B env vars

| Env var | Default | Purpose |
|---|---|---|
| `CONTEXTPILOT_ENABLED` | `1` | Set to `0` to force-disable ContextPilot even if `--enable-contextpilot` is passed |

### Runtime toggle and fault injection

Enable or disable ContextPilot at runtime:

```bash
curl -X POST http://localhost:8000/contextpilot/toggle \
    -H "Content-Type: application/json" \
    -d '{"enabled": true}'
```

Inject a test fault, debug mode required:

```bash
curl -X POST http://localhost:8000/contextpilot/inject-fault \
    -H "Content-Type: application/json" \
    -d '{"fault": "reorder_exception", "duration_s": 5}'
```

Accepted fault values are `none` and `reorder_exception`.

## Phase C: Deep Scheduler Integration

### What is enabled

Phase C keeps the Phase B middleware path and adds CP-aware KV and scheduling hooks:

- `ContextPilotKVManager` exposes predicted prefix reuse, cached block hints, allocation notifications, free notifications, and request ordering.
- `EvictionSyncAdapter` removes ContextPilot state on terminal completion and abort, while leaving swap events alone.
- Scheduler and KV code can prioritize requests with higher predicted overlap instead of relying only on queue order.

### Trade-offs

| Benefit | Cost |
|---|---|
| Best TTFT and token-savings gains | Highest integration depth |
| Better overlap-aware scheduling | More coupling with KV and scheduler internals |
| Better CP state sync on terminal frees | More pieces to validate during upgrades |

Representative dry-run gains versus baseline:

| Phase | TTFT p50 | Token savings |
|---|---:|---:|
| Phase B | 21% faster | 27% |
| Phase C | 26% faster | 28% |

## Observability

### Admin and status endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/contextpilot/toggle` | `POST` | Enable or disable ContextPilot at runtime |
| `/contextpilot/inject-fault` | `POST` | Inject a test fault, only when debug mode is enabled |
| `/contextpilot/status` | `GET` | Return current status, counters, and recent metrics |

Check current state:

```bash
curl http://localhost:8000/contextpilot/status
```

### `/contextpilot/status` fields

| Field | Meaning |
|---|---|
| `enabled` | Current runtime toggle state |
| `env_enabled` | Whether `CONTEXTPILOT_ENABLED` still allows ContextPilot |
| `circuit_breaker_state` | `closed`, `open`, or `half_open` summary state |
| `requests_processed` | Total requests seen by the middleware |
| `reorder_count` | Requests that went through reorder logic |
| `dedup_count` | Requests that went through dedup logic |
| `avg_reorder_latency_ms` | Average reorder latency over the in-memory sample window |
| `p99_reorder_latency_ms` | P99 reorder latency over the in-memory sample window |
| `token_savings_total` | Total prompt tokens removed so far |
| `token_savings_avg_pct` | Average percent token savings across processed requests |
| `eviction_sync` | Nested counters for terminal eviction events: `incoming`, `removed`, `not_found` |
| `cp_index_size` | Current number of live ContextPilot index entries, when available |
| `fallback_count` | Total times MoE-Infinity fell back after ContextPilot errors |
| `last_fallback_count` | Latest recorded fallback counter snapshot |
| `debug` | Whether debug endpoints are enabled |
| `fault` | Active injected fault mode |

## Troubleshooting

### Python 3.8/3.9 auto-disable ContextPilot

Real ContextPilot needs Python 3.10+. On Python 3.8/3.9, the middleware auto-disables with a warning. Install the package in a Python 3.10+ environment to enable ContextPilot features.

### `pip install contextpilot` may pull the wrong package

There is a package name conflict on PyPI. One package named `contextpilot` is a different project and may pull in `elasticsearch`. If you hit that dependency path, use the known ContextPilot checkout under `/tmp/ContextPilot`, or install the intended package in a clean Python 3.10+ environment before enabling Phase B or Phase C.

### ContextPilot seems disabled even with the CLI flag

Check `CONTEXTPILOT_ENABLED`. If it is set to `0`, the server will ignore `--enable-contextpilot`. The `/contextpilot/status` endpoint exposes both `enabled` and `env_enabled` so you can tell which gate is active.

### Fault injection endpoint returns 403

Start the server with `--contextpilot-debug`. Without that flag, `/contextpilot/inject-fault` is intentionally blocked.

## API Reference

| Module | Role |
|---|---|
| `moe_infinity/serving/contextpilot_middleware.py` | In-process prompt reorder, dedup, request metrics, and request cleanup |
| `moe_infinity/serving/eviction_sync.py` | Sync layer that converts terminal request lifecycle events into ContextPilot eviction signals |
| `moe_infinity/serving/contextpilot_circuit_breaker.py` | Small circuit breaker used to contain repeated failures or latency spikes |
| `moe_infinity/serving/cp_kv_interface.py` | CP-aware abstraction used by KV and scheduler code in Phase C |

For lower-level design details, see:

- [`cp_kv_design.md`](cp_kv_design.md)
- [`eviction_lifecycle.md`](eviction_lifecycle.md)
