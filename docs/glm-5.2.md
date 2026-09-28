# GLM-5.2 guide

GLM-5.2 is the `glm_moe_dsa` route in MoE-Infinity. The registered class is
`GlmMoeDsaForCausalLM`, and the checkpoint used in this repo is
`zai-org/GLM-5.2-FP8`.

## Maturity and compatibility

- Transformers floor: `>= 5.12`
- Registry status: conditional import, so the family is skipped when the class
  is unavailable
- Supported path: the HuggingFace-compatible `MoE` wrapper
- Native serving engine: disabled for this family (`use_native_engine = False`)

## Dependencies

GLM-5.2 uses the shared MoE offload runtime plus the GLM-specific MTP and DSA
helpers in `moe_infinity/spec_decode/`.

The repo also exposes `glm_dflash_available()`, but it currently warns that no
GLM drafter is registered, so route-ahead DFlash is not part of the supported
path.

## FP8 expert offloading

The routed experts stay in the host store as FP8. On load, the router expands
the packed expert tensors to per-expert state, then the dispatcher streams the
needed experts to GPU.

What stays resident on GPU:
- 3 dense layers
- shared expert
- MLA attention
- DSA indexer
- MTP layer

What is dequantized to BF16 on load:
- non-routed FP8 weights that execute outside the expert dispatcher

## Synchronous generation

Synchronous generation goes through `MoE.generate`, which is deprecated,
emits `DeprecationWarning`, and is scheduled for removal. It remains the current
validated synchronous GLM path during the transition. The GLM family does not
switch to a native engine, and the repo does not advertise a separate
GLM-specific serving backend.

For the current codebase, that means the validated path is the HF-style model
wrapper plus the offload runtime, not a custom native executor.

## Serving and speculative decoding

Serving support exists through the normal MoE-Infinity server entrypoints, but
GLM-specific speculative decoding is limited:

- built-in MTP is present
- no GLM drafter is registered for route-ahead prefetch
- speculative draft/verify is not a validated GLM serving mode in this repo

## Storage and memory requirements

The guide assumes the FP8 routed experts are too large to keep fully resident,
so host memory is required for the expert store.

Memory layout summary:
- host store: routed experts
- GPU resident: dense blocks, shared expert, MLA attention, DSA indexer, MTP
- execution precision: routed experts remain FP8 in storage, with BF16 compute
  where needed outside the dispatcher

## Validation

Repo coverage includes:
- `tests/python/unit/test_model_registry.py`
- `tests/python/unit/test_glm_*`
- `tests/python/integration/test_glm_smoke.py`
- `tests/python/integration/test_glm_serving.py`
- `tests/python/integration/test_glm_mtp.py`
- `tests/python/integration/test_glm_fp8_store_parity.py`
- `tests/python/integration/test_glm_fp8_reload_parity.py`

The tiny harness validates the registration, routing, store parity, reload
parity, and serving smoke path, but it does not claim full production topology
coverage.

## Known limitations

- `use_native_engine = False` for GLM-5.2
- no registered GLM drafter for route-ahead DFlash
- no claim here for multi-node serving or untested topologies
- speculative decoding is not treated as production-validated for this family

See also: [Model compatibility matrix](model-compatibility.md).
