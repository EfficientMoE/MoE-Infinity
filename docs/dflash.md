# DFlash speculative decoding

MoE-Infinity ships a native DFlash draft, verify, rollback path for GPT-OSS and other supported MoE models. The deprecated synchronous wrapper `MoE.generate(..., speculative_draft=...)` remains the current in-process integration and emits `DeprecationWarning`; `MoE.serve(..., speculative_draft=...)` is the recommended continuous-batching HTTP path and is not a drop-in return API. Direct `DFlashSpeculator.generate(...)` is an experimental alternative for custom harnesses, with no stable API promise, and supports batch-1 greedy and sampled decoding. The `MoE` integrations are greedy-gated and delegate only singleton requests in the server. Batch > 1 DFlash stays on the bare HuggingFace target path.

For server-side gating, sampled-request fallback, and exact troubleshooting language, see [Serving](serving.md), [Architecture](../ARCHITECTURE.md), and [Troubleshooting](troubleshooting.md).

## Quick start

This example requires a CUDA-capable source installation, access to both
checkpoints, and enough GPU cache, host memory, and SSD capacity for GPT-OSS-120B.
Checkpoint loading may download weights when they are not already cached. For a
configurable runnable script using the same defaults, see
[`examples/dflash_gpt_oss_example.py`](../examples/dflash_gpt_oss_example.py).
The drafter loads with `trust_remote_code=True`, which executes code from the
checkpoint repository. Use only a trusted drafter and pin the repository
revision for reproducible or security-sensitive deployments.

```python
from transformers import AutoTokenizer

from moe_infinity import MoE
from moe_infinity.spec_decode import DFlashSpeculator

target = "openai/gpt-oss-120b"
drafter = "z-lab/gpt-oss-120b-DFlash"
prompt = "Question: What is the capital of France?\nAnswer:"

tokenizer = AutoTokenizer.from_pretrained(target)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

model = MoE(target, {
    "offload_path": "/ssd/moe-infinity/gpt-oss-120b",
    "device_memory_ratio": 0.75,
})
spec = DFlashSpeculator(model, drafter)

output_ids = model.generate(
    input_ids,
    max_new_tokens=64,
    do_sample=False,
    speculative_draft=spec,
)
```

If you want sampled batch-1 DFlash, call `spec.generate(...)` directly. The
`MoE.generate` and serving integrations stay on the standard path for sampled
requests today. The Qwen3.5-MoE wrapper is the explicit exception: sampled
`MoE.generate(..., speculative_draft=...)` raises `ValueError` there, because
that model path requires greedy speculative decode.

## Capability matrix

| Capability | Status | Evidence or constraints |
| --- | --- | --- |
| Batch-1 direct `spec.generate`, greedy | Implemented, validated | `tests/python/dflash/test_native_step.py::test_native_multistep_greedy_matches_plain_greedy`, `tests/python/dflash/test_edge_cases.py` |
| Batch-1 direct `spec.generate`, sampled | Implemented, validated on CPU tiny fixtures | `tests/python/dflash/test_sampled_spec.py::test_sampled_generate_is_seed_deterministic`; this path is direct, not through `MoE.generate` |
| Batch > 1 direct `spec.generate`, greedy | Implemented, validated | `tests/python/dflash/test_batched_spec.py::test_batched_matches_looped_singles_token_identical` |
| `MoE.generate(..., speculative_draft=...)`, greedy batch-1 | Implemented, validated | `tests/python/dflash/test_engine_wire.py`, `tests/python/dflash/test_spec_seam.py` |
| `MoE.generate(..., speculative_draft=...)`, sampled batch-1 | Standard path for most models; Qwen3.5-MoE raises `ValueError` | The native speculator is greedy-gated in the engine, and the Qwen3.5 wrapper rejects sampled speculative decode on this path |
| Continuous-batching serving | Implemented, validated on GPT-OSS-20B | `tests/python/dflash/test_gpu_serving_dflash.py`; only a fresh singleton greedy request can delegate |
| Route-ahead expert prefetch | Implemented, scheduling-only, validated on synthetic offloaded shells | `tests/python/dflash/test_route_ahead_metrics.py`, `tests/python/dflash/test_route_ahead_wire.py`, `tests/python/dflash/test_qwen35_hybrid_rollback.py`; gpt-oss does not wire the executor |

## Configuration

### Direct speculator API

- `DFlashSpeculator(moe, draft_model_path, device=None, dtype=torch.bfloat16)`
  loads the drafter with `trust_remote_code=True` and defaults to bfloat16.
- `DFlashSpeculator.from_models(moe, draft_model, config=None, device=None)`
  skips checkpoint loading and reuses an already-built drafter module.
- `DFlashSpeculator.generate(input_ids, max_new_tokens=256, temperature=0.0,
  stop_token_ids=None, top_k=0, top_p=1.0, attention_mask=None)` is the public
  experimental decode entry point; it has no stable API guarantee.
- `DFlashSpeculator.enable_route_ahead_stats()` creates or resets a read-only
  recorder. `route_ahead_stats` starts as `None`.
- `read_dflash_config(draft_hf_config)` requires `block_size`, `mask_token_id`,
  `target_layer_ids`, `hidden_size`, and `vocab_size`. `num_target_layers` is
  optional and defaults to `-1`.
- `validate_pairing(...)` checks `hidden_size`, `vocab_size`, `mask_token_id <
  vocab_size`, `block_size >= 2`, and target layer range.
- `validate_drafter(...)`, `validate_drafter_module(...)`, and
  `bind_shared_weights(...)` are internal helpers inside `dflash.py`; they are
  used by the implementation and tests, but they are not exported from
  `moe_infinity.spec_decode.__init__`.

### MoE and server knobs

- `MoE.generate(..., speculative_draft=...)` attaches the speculator only for
  the current call. Omit the kwarg, or pass `None` or `False`, to detach it.
  For Qwen3.5-MoE, sampled requests on this path raise `ValueError` instead of
  falling back silently.
- Greedy DFlash delegation is gated by `temperature=0`, `top_k=0`, and
  `top_p=1.0`; `do_sample=False` is the usual caller intent for that path, but
  it is not a separate runtime gate.
- `MoE.serve(..., speculative_draft=None)` defaults to `device_memory_ratio=
  0.75`, `kv_cache_ratio=0.25`, `max_batch_size=32`, and
  `enable_prefix_caching=False`.
- `python -m moe_infinity.entrypoints.openai.api_server_v2 --speculative-draft
  <drafter>` exposes the same path from the CLI. The parser default is off.

If a sampled serving request falls back to the standard path, that is expected behavior, not a pairing failure; the pairing check is a separate validation step.

## Draft, Verify, Commit, and Rollback

1. **Draft**
   - Build a block of the form `[anchor, MASK, MASK, ...]`.
   - The target hidden states at the configured `target_layer_ids` are
     concatenated into the drafter context feature.
   - The drafter and target share the target `embed_tokens` and `lm_head`.

2. **Verify**
   - The target runs one full-logits forward over the whole block.
   - Greedy mode uses the target argmax agreement rule.
   - Sampled mode uses warped draft and target probabilities with lossless
     rejection sampling.

3. **Commit**
   - The emitted step is `accepted drafts + bonus token`.
   - The cached prefix is `anchor + accepted drafts`.
   - The bonus token is emitted but not cached, because it becomes the next
     anchor.

4. **Rollback**
    - Serving keeps `cached_len == prompt_len + emitted_len` after each committed
      step. That is the state contract pinned by `SpecDecodeState`.
    - `SpecDecodeState.record_verify(block_len, committed)` advances the cached
      and emitted counts, and `apply_verify_step()` turns that into the
      `truncate_target` passed to `PagedKVCache.truncate_tokens(seq_id, new_len)`.
    - `PagedKVCache.truncate_tokens(seq_id, new_len)` frees tail blocks and
      truncates swapped-out buffers.
    - Sliding-window targets snapshot and replay the committed prefix when plain
      `DynamicCache.crop()` is not enough.

## Sampled decoding

- Sampled DFlash is batch-1 only.
- The direct path uses `warped_probs`, `acceptance_sampled`, and
  `committed_tokens_sampled`.
- Warp order matches the engine sampler, temperature first, then top-k, then
  top-p, then softmax.
- If you use `MoE.generate` or `MoE.serve`, sampled requests stay on the
  standard path today, except that Qwen3.5-MoE intentionally rejects sampled
  speculative decode with `ValueError` on the `MoE.generate` path.
- Batch > 1 sampled DFlash is not supported.

## Batched decoding

- Batch > 1 direct DFlash is greedy only and requires a bare HuggingFace target.
- Prompts must be left padded, or all the same length. `attention_mask` must be
  0/1 valued and end with a real token.
- `max_new_tokens` can be a single int or a per-sequence list.
- The returned tensor is right padded, and `last_generated_lengths` stores the
  true per-row new-token counts.
- `MoE.generate(..., speculative_draft=...)` with batch > 1 raises
  `NotImplementedError`.

## Continuous-batching serving

- `MoE.serve(..., speculative_draft=...)` and the OpenAI server CLI accept a
  DFlash speculator.
- The serving engine delegates only for a fresh singleton prefill request with
  no prior output tokens that is greedy, stop-free, penalty-free, has
  `top_k <= 0`, `top_p >= 1.0`, `repetition_penalty == 1.0`, `logprobs <= 0`,
  and is within the per-step token cap.
- A delegated serving step may emit several accepted tokens, and the engine
  streams them one by one after recording committed counts.
- The validated real-checkpoint serving harness is GPT-OSS-20B. GPT-OSS-120B
  has no recorded real-checkpoint serving validation. Qwen3.5 serving wiring
  has tiny-fixture coverage, but no real-checkpoint serving validation is
  recorded.

## Route-ahead expert prefetch

Route-ahead is a scheduling-only add-on. It warms the exact routed expert union
for the layer being dispatched, but it does not change routing or output
tokens.

- Activation happens only during a DFlash verify forward.
- If route-ahead is active and a prefetcher exists, the executor pins the exact
  routed union with `fetch_experts_lock_cache(...)` and enqueues the same set
  with `speculative_prefetch(..., expert_ids=..., prefetch_layer_id=...)`.
- If the context is inactive, there is no prefetcher, or the union is empty,
  the executor falls back to the legacy pooled prefetch or a no-op.
- The union is pinned one layer at a time because `ReplaceCacheCandidates` is
  global and clears background queues.
- Offloaded executor-backed models such as DeepSeek, Qwen, and Mixtral reach
  this seam. GPT-OSS does not, because its expert path never wires an
  `expert_executor`.

Metric names:

- `RouteAheadStepSummary(layers, predicted, actual, covered, kept, wasted)`
- `RouteAheadStats.as_dict()` returns `steps`, `layers_observed`,
  `predicted_experts`, `actual_experts`, `covered_experts`, `kept_experts`,
  `wasted_experts`, `coverage`, and `waste_ratio`

Coverage is `covered / actual` and defaults to `1.0` when `actual == 0`.
Waste ratio is `wasted / predicted` and defaults to `0.0` when
`predicted == 0`.

## Observability

- `route_ahead_stats` defaults to `None`. `enable_route_ahead_stats()` returns
  the recorder and resets it on reuse.
- `step_trace`, `last_target_cache`, `last_draft_cache`, and
  `last_generated_lengths` are advanced internal diagnostics only; they have no
  stability or compatibility guarantee.
- `step_trace` records `prev_start`, `accept`, `start`, `emitted_len`,
  `target_cache_len`, and `draft_cache_len`.
- `last_target_cache` and `last_draft_cache` expose the final caches after a run.
- `last_generated_lengths` is only set by the batched path.

## Compatibility

| Model | Target / drafter | Residency | Sync | Serving | Sampling | Route-ahead | Hardware / validation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-OSS-20B | `openai/gpt-oss-20b` / `z-lab/gpt-oss-20b-DFlash` | Tunable via `offload_path` and `device_memory_ratio`; the GPU-gated harness uses a high resident setting | Greedy batch-1 validated | Continuous-batching greedy validated | Direct batch-1 sampled implemented, not yet validated on the real pair | Not wired: the GPT-OSS expert path does not attach an `expert_executor` | GPU-gated, no board asserted in repo tests |
| GPT-OSS-120B | `openai/gpt-oss-120b` / `z-lab/gpt-oss-120b-DFlash` | Tunable via `offload_path` and `device_memory_ratio`; the GPU-gated harness uses a high resident setting | Greedy batch-1 validated | Implemented, not yet validated on a real serving harness | Direct batch-1 sampled implemented, not yet validated on the real pair | Not wired: the GPT-OSS expert path does not attach an `expert_executor` | GPU-gated, no board or TP count asserted in repo tests |
| Qwen3.5-MoE | No published real-model drafter checkpoint is recorded in repo tests | Text backbone, shared expert, and `lm_head` stay resident, routed experts offload | Greedy batch-1 hybrid rollback validated on tiny CPU fixtures | Serving wiring has tiny-fixture coverage; no real-checkpoint serving validation is recorded | `MoE.generate` sampled DFlash is rejected, direct sampled speculator is not yet validated on a real checkpoint | Wired and exercised on synthetic/tiny CPU fixtures; not validated with a real target/drafter checkpoint | No real-model hardware validation recorded |

Route-ahead is an execution-path capability of executor-backed offloaded models,
not evidence that a validated DFlash target/drafter checkpoint pair exists.
DeepSeek, Qwen, and Mixtral paths can reach the executor seam described above,
but they are absent from this matrix unless the repo records a corresponding
DFlash pairing and validation scope. No additional drafter pair is implied.

The real-model harnesses skip cleanly unless `MOE_DFLASH_GPU=1` is set and the
checkpoints are present in the HuggingFace cache.

## Validation

- CPU gate, no GPU or checkpoint download required:
  `pytest -q tests/python/dflash -m "not gpu"`
- GPT-OSS-20B GPU-gated validation:
  `MOE_DFLASH_GPU=1 pytest -q tests/python/dflash/test_gpu_20b_dflash.py`
- GPT-OSS-120B GPU-gated validation:
  `MOE_DFLASH_GPU=1 pytest -q tests/python/dflash/test_gpu_120b.py`
- GPT-OSS-20B serving versus sync validation:
  `MOE_DFLASH_GPU=1 pytest -q tests/python/dflash/test_gpu_serving_dflash.py`

## Known limitations and troubleshooting

- Batch > 1 with `speculative_draft` through `MoE.generate` is not supported.
- Batch > 1 sampled DFlash is not supported.
- GPT-OSS route-ahead is not wired because the model path never attaches an
  `expert_executor`.
- Qwen3.5 greedy DFlash is supported through the hybrid cache replay path, but
  sampled DFlash on a real Qwen3.5 checkpoint is not yet validated.
- Missing checkpoints or `MOE_DFLASH_GPU` simply skip the GPU-gated harnesses.
- `block_size`, `target_layer_ids`, hidden size, or vocab mismatches fail fast
  during drafter validation.

## Related files

- `examples/dflash_gpt_oss_example.py`
- `tests/python/dflash/`
