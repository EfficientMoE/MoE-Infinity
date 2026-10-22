# GLM-5.3-Flash guide

GLM-5.3-Flash is the `glm5_next` route in MoE-Infinity. The registered class
is `Glm5NextForConditionalGeneration`, and the checkpoint used in this repo is
`zai-org/GLM-5.3-Flash`.

For GLM-5.3 (`zai-org/GLM-5.3`), see the [GLM-5.2 guide](glm-5.2.md): it is
the same `GlmMoeDsaForCausalLM` base and runs through the identical registry
entry and FP8 expert-offload path. Its chat template adds `reasoning_effort`
(`low`/`high`, anything else falls back to `max`) and `clear_thinking`
(default false replays prior-turn reasoning; `true` strips it), pinned by
`tests/python/unit/test_glm53_chat_template.py`.

## Quick start

GLM-5.3-Flash runs through the drop-in `MoE` class:

```python
from moe_infinity import MoE

model = MoE("zai-org/GLM-5.3-Flash", {
    "offload_path": "/ssd/moe-infinity/glm-5.3-flash",
    "device_memory_ratio": 0.5,
})
```

## Maturity and compatibility

- Transformers floor: `>= 5.16` (the repo-wide floor stays `>= 5.3`; the
  family registers only when `Glm5NextForConditionalGeneration` is importable
  and is skipped automatically otherwise)
- Supported path: the HuggingFace-compatible `MoE` wrapper, text-only
- Attention: `attn_implementation="eager"` is forced for this family
- Vision inputs are not supported; the vision tower loads resident but unused

## Architecture notes

GLM-5.3-Flash nests its text model under `text_config`
(`model_type="glm5_next_text"`): 45 decoder layers (first 3 dense) plus 1 MTP
layer, 288 routed FP8 experts per MoE layer with 8 active per token, one
shared expert, sigmoid `noaux_tc` routing with `routed_scaling_factor=2.5`
(same router family as GlmMoeDsa), hybrid KDA linear attention (34 layers)
interleaved with DSA sparse attention (11 layers), mHC hyper-connections, and
a 24-layer vision tower.

On disk, routed experts are per-expert
`model.language_model.layers.<L>.mlp.experts.<E>.{gate,up,down}_proj.weight`
FP8 tensors with blockwise-128x128 `_scale_inv` scales (pinned by the trimmed
`tests/fixtures/glm_5_3_flash/model.safetensors.index.trimmed.json` fixture).

## FP8 expert offloading

The routed experts stay in the host store as FP8 and stream to GPU on demand
through `SyncGlm5NextMoEBlock`.

What stays resident on GPU:
- 3 dense layers
- shared expert
- KDA linear-attention layers
- DSA indexer and sparse-attention layers
- mHC hyper-connections
- vision tower (unused, text-only serving)
- MTP weights (unused, no glm5_next drafter)

## Serving and speculative decoding

Serving goes through the normal MoE-Infinity server entrypoints
(`api_server_v2`), covered by the tiny harness. No glm5_next drafter is
registered: MTP weights are resident but unused, and speculative draft/verify
is not a supported mode for this family.

## Storage and memory requirements

- disk: ~330 GB for the checkpoint plus the offload store
- host memory: routed-expert store (FP8)
- GPU: ~34 GB BF16 resident backbone plus the expert cache
  (`device_memory_ratio` tunes the cache)

## Validation

Repo coverage includes:
- `tests/python/unit/test_glm5_next_registry.py`
- `tests/python/unit/test_glm5_next_moe_block.py`
- `tests/python/integration/test_glm5_next_tiny_generate.py`
- `tests/python/integration/test_glm5_next_serving.py`
- `tests/python/integration/test_glm5_next_fp8_parity.py`
- `scripts/smoke_glm5_next.py` (real-checkpoint smoke)

The tiny harness (`MOE_GLM5_NEXT_TINY=1`, GPU) validates registration,
routing, deterministic greedy generation, FP8 store/reload parity, and the
serving smoke path on a synthetic 4-layer model; it does not claim full
production topology coverage.

## Known limitations

- text-only: no vision input path
- no registered glm5_next drafter for DFlash; MTP unused
- eager attention only (no FlashAttention/FlashInfer claim)
- no claim for multi-node serving or untested topologies

See also: [Model compatibility matrix](model-compatibility.md).
