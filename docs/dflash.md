# DFlash Speculative Decoding (gpt-oss-120b)

MoE-Infinity ships a **native** DFlash (block-diffusion) speculative-decoding
path for gpt-oss. A greedy, batch-1 `MoE.generate(..., speculative_draft=...)`
routes through the engine's `spec_strategy` seam and runs the native
draft → verify → rollback loop in `moe_infinity/spec_decode/dflash.py`.

## Usage

```python
from moe_infinity import MoE
from moe_infinity.spec_decode import DFlashSpeculator

model = MoE("openai/gpt-oss-120b", {"offload_path": "/ssd/moe-infinity/gpt-oss-120b"})
speculator = DFlashSpeculator(model, "z-lab/gpt-oss-120b-DFlash")  # trust_remote_code

output_ids = model.generate(
    input_ids,
    max_new_tokens=64,
    do_sample=False,          # greedy → routes through the native DFlash strategy
    speculative_draft=speculator,
)
```

Omit `speculative_draft` (or pass a non-greedy sampling config / batch > 1) and
`generate` uses the standard autoregressive path, **byte-identical** to before.

## v1 scope

- **Greedy only** (`do_sample=False`); sampled speculative decoding is deferred.
- **Resident by default.** Expert offload is a tunable knob (`device_memory_ratio`);
  v1 does not couple expert prefetch to the speculative loop.
- **Sync path only** — the async serving path is not spec-enabled.
- **batch == 1.**

## How it works

Per step: prefill the target for an anchor; build a `block_size=10` block
`[anchor, MASK×9]`; the non-causal drafter (reusing the target `embed_tokens` /
`lm_head`, with a 5-layer target feature at layers `[1, 9, 17, 25, 33]` injected
into every drafter layer) fills the 9 masks; the target verifies the whole block
in **one full-logits forward**; the leading drafts the target's argmax agrees with
are accepted, plus a bonus token (emitted but not cached); both KV caches roll
back to the committed prefix.

gpt-oss uses **sliding-window attention**, so the target rollback snapshots each
sliding layer *before* the verify forward and rebuilds it from the committed
prefix — a plain `DynamicCache.crop()` is invalid once the window is saturated
(the layer has already evicted the tokens a partial accept must restore).

## Testing

- **Autonomous (CPU, tiny model)** — no GPU / no checkpoints:
  `pytest tests/python/dflash -q`. The losslessness gates are
  `test_native_e2e.py` (native DFlash == plain greedy, token-identical) and
  `test_spec_off_regression.py` (spec-off byte-identical to the pre-integration
  baseline).
- **GPU-gated (120B)**:
  `MOE_DFLASH_GPU=1 pytest tests/python/dflash/test_gpu_120b.py -q`
  with `openai/gpt-oss-120b` + `z-lab/gpt-oss-120b-DFlash` cached; resident, TP=2
  on SM120. Reports token agreement-rate vs plain-decode self-consistency, the
  acceptance-length histogram, and decode tok/s (DFlash vs no-spec). Skips
  cleanly when the flag is unset or the checkpoints/GPU are unavailable.
