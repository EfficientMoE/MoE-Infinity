# GLM-5.3 / GLM-5.3-Flash Validation Plan (end-to-end support + tests)

> **For agentic workers:** steps use checkbox (`- [ ]`) syntax for tracking. GPU/disk-gated
> steps that cannot run in CI must paste their local command output into the PR description
> under a "Local validation evidence" heading (repo convention, see
> `2026-09-05-new-moe-model-support.md`).

**Goal:** Promote `zai-org/GLM-5.3` and `zai-org/GLM-5.3-Flash` from `implemented/experimental`
to `validated` in `docs/model-compatibility.md` by (a) closing the test gaps left by the
2026-09-05 plan (unexecuted Tasks 1.3, 3.0, 3.6) and (b) bringing GLM-5.3-Flash's test tier to
parity with GLM-5.2's validated tier (tiny generate + tiny serving + FP8 parity + real-checkpoint
smoke).

**Current state (verified 2026-10-19):**
- GLM-5.3: unit config-resolution test only (`tests/python/unit/test_glm53_registry.py`,
  fixture `tests/fixtures/glm_5_3/config.json`). `test_glm_smoke.py` hardcodes
  `zai-org/GLM-5.2-FP8`; the planned env override was never added.
- GLM-5.3-Flash: registry/parsing/block wiring implemented; unit tests
  (`test_glm5_next_registry.py`, `test_glm5_next_moe_block.py`) and tiny generate test
  (`test_glm5_next_tiny_generate.py`, `MOE_GLM5_NEXT_TINY=1`) exist but **have never executed**:
  installed transformers is 5.12.1 and glm5_next requires >= 5.16; `.venv-tf516` was never
  created. No serving test, no FP8 parity tests (tiny builder strips `quantization_config`),
  no determinism assertion, no real-checkpoint harness, no `docs/glm-5.3-flash.md`.
- Host: 6x RTX PRO 6000 Blackwell 96 GB (sm_120 -> build with `MOE_ENABLE_SM120=1`),
  5.3 TB free on `/mnt/raid0nvme0`, HF cache at `/mnt/raid0nvme0/public/huggingface/hub`
  (has GLM-5.2-FP8 only; GLM-5.3 ~760 GB and GLM-5.3-Flash ~330 GB must be downloaded).

**Validated-tier template (GLM-5.2):** unit suite (CI) + `MOE_GLM_TINY=1` tiny harness
(generate, serving via `api_server_v2` `/v1/completions`, FP8 store parity, FP8 reload parity)
+ `MOE_GLM_SMOKE=1` real-checkpoint generate + 32k-prefill smoke.

**Explicit non-goals:** vision inputs (text-only serving for glm5_next), MTP/DFlash for
glm5_next (weights resident-unused), multi-GPU validation, FlashInfer (eager attention is
forced for both families), adaptive candidate evaluation (FP8 models remain protected).

**Parallelization:** Track A and Track B are independent. Long-running background jobs
(checkpoint downloads, `.venv-tf516` build) start first; code/test work proceeds while they run.

---

## Track A - GLM-5.3 (base): promote to validated (no runtime code)

### Task A1: Parametrize the GLM smoke harness

**Files:** `tests/python/integration/test_glm_smoke.py`

- [ ] Add `MOE_GLM_SMOKE_MODEL` env override (default `zai-org/GLM-5.2-FP8`) used by both
  `test_glm_generate_smoke` and `test_glm_long_context_prefill`. Keep `MOE_GLM_SMOKE=1` gating
  unchanged so the GLM-5.2 path is byte-identical by default.
- [ ] Sanity: `.venv/bin/python -m pytest tests/python/integration/test_glm_smoke.py -q`
  (collects, skips without the env var).

### Task A2: GLM-5.3 chat-template unit test

**Files:** `tests/fixtures/glm_5_3/` (template fixture), new
`tests/python/unit/test_glm53_chat_template.py`

The only behavioral delta vs GLM-5.2 is post-training + chat template: `reasoning_effort`
(`low`/`high`/`max`, default `max`) and `clear_thinking`.

- [ ] Fetch `chat_template.jinja` (or `tokenizer_config.json` template field) from
  `zai-org/GLM-5.3` into the fixture dir (metadata only, no weights).
- [ ] CPU-only unit test: render the template with/without `reasoning_effort` and
  `clear_thinking` kwargs; pin default `max`; assert the rendered prompt reflects each setting.
- [ ] Run: `.venv/bin/python -m pytest tests/python/unit/test_glm53_chat_template.py -q`.

### Task A3 (GPU + ~760 GB disk gated): GLM-5.3 real-checkpoint smoke

- [ ] Download `zai-org/GLM-5.3` into the shared HF cache (background job).
- [ ] Run on one 96 GB GPU:
  `MOE_GLM_SMOKE=1 MOE_GLM_SMOKE_MODEL=zai-org/GLM-5.3 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest tests/python/integration/test_glm_smoke.py -q`
  Expected: both tests pass (generate + 32k prefill). Capture output as PR evidence.
- [ ] Optional stretch: tiny serving harness is arch-shared; a real-checkpoint serving smoke
  (api_server_v2 + one `/v1/completions` call against GLM-5.3) upgrades the validated scope
  from "sync generation" to "sync generation + serving". Decide based on A3 runtime cost.

### Task A4: Docs promotion

**Files:** `docs/model-compatibility.md`, `README.md`, `CHANGELOG.md`

- [ ] Matrix row GLM-5.3 -> `validated`, scope worded to exactly what passed in A3.
- [ ] CHANGELOG Unreleased note; README GLM note stays accurate.
- [ ] `.venv/bin/python -m pytest tests/python/unit/test_docs.py -q` (docs-consistency test).

---

## Track B - GLM-5.3-Flash (glm5_next): finish support + full test tier

### Task B0 (hard prereq): `.venv-tf516` environment gate

- [ ] Create `.venv-tf516` (python 3.12): torch matching `.venv`'s CUDA build,
  `moe-store` (same source/tag as `.venv`), `transformers>=5.16,<6`,
  `MOE_ENABLE_SM120=1 MOE_ENABLE_SM90=0 pip install --no-build-isolation -e .`.
- [ ] Gate check: `.venv-tf516/bin/python -c "from transformers.models.glm5_next.modeling_glm5_next import Glm5NextForConditionalGeneration; print('ok')"`.
- [ ] Guard check on the primary venv stays green:
  `.venv/bin/python -m pytest tests/python/unit/test_glm5_next_registry.py tests/python/unit/test_glm5_next_moe_block.py -q` (skips cleanly on 5.12.1).

### Task B1: First real execution of the existing glm5_next tests

This is a bug-discovery milestone, not a formality: the wiring (WeightConverter key renames at
`model_offload.py:~1663`, concat-fuse at `~1706`, KDA/DSA/mHC resident policy, eager forcing)
has never run.

- [ ] `.venv-tf516/bin/python -m pytest tests/python/unit/test_glm5_next_registry.py tests/python/unit/test_glm5_next_moe_block.py -q` - expected pass (no skips).
- [ ] `MOE_GLM5_NEXT_TINY=1 CUDA_VISIBLE_DEVICES=0 .venv-tf516/bin/python -m pytest tests/python/integration/test_glm5_next_tiny_generate.py -q`.
- [ ] Fix any wiring bugs minimally (no refactors); add a regression unit test for each fix.

### Task B2: Close the tiny-tier gaps (parity with GLM-5.2's validated tier)

**Files:** `tests/python/integration/_glm5_next_tiny.py` (FP8 variant),
new `tests/python/integration/test_glm5_next_serving.py`,
new `tests/python/integration/test_glm5_next_fp8_store_parity.py`,
`tests/python/integration/test_glm5_next_tiny_generate.py` (determinism),
new unit test + `tests/fixtures/glm_5_3_flash/model.safetensors.index.json` (trimmed).

- [ ] **Determinism:** extend tiny generate test - two greedy runs, identical token ids
  (2026-09-05 plan Task 3.5 specified this; current test omits it).
- [ ] **Tiny serving:** model on `test_glm_serving.py` - build tiny glm5_next, spawn
  `moe_infinity.entrypoints.openai.api_server_v2`, assert `/v1/completions` returns text.
  Gate: `MOE_GLM5_NEXT_TINY=1` + gpu marker.
- [ ] **FP8 parity:** add `build_tiny_glm5_next_fp8` (keep blockwise-128x128
  `quantization_config`, quantize routed experts to FP8 E4M3 + `_scale_inv`); add store/reload
  parity tests modeled on `test_glm_fp8_store_parity.py` / `test_glm_fp8_reload_parity.py`.
  This is the core offload claim (routed FP8 experts in host store) and is currently untested
  for this family.
- [ ] **Key-rename pinning:** trimmed real `model.safetensors.index.json` fixture (first MoE
  layer's keys suffice) + CPU unit test asserting the v5 rename/concat-fuse mapping resolves
  every routed-expert key to the expected store ids.
- [ ] All new tests: `pytest.importorskip("transformers.models.glm5_next")` + env-var gates;
  CI on 5.12.1 must skip them cleanly.

### Task B3 (GPU + ~330 GB disk gated): real-checkpoint smoke

**Files:** new `scripts/smoke_glm5_next.py`

- [ ] Download `zai-org/GLM-5.3-Flash` into the shared HF cache (background job).
- [ ] `scripts/smoke_glm5_next.py` (argparse: `--model`, `--offload-dir`,
  `--device-memory-ratio`, default 0.5): load via `MoE`, text-only prompt through the tokenizer
  chat template, greedy `max_new_tokens>=32`, assert non-empty coherent completion; print
  resident GPU footprint (expected ~34 GB BF16 backbone + expert cache).
- [ ] Run: `CUDA_VISIBLE_DEVICES=0 .venv-tf516/bin/python scripts/smoke_glm5_next.py --model zai-org/GLM-5.3-Flash --offload-dir /mnt/raid0nvme0/leyang/offload/glm-5.3-flash`.
  Capture output as PR evidence.

### Task B4: Docs

**Files:** new `docs/glm-5.3-flash.md`, `docs/model-compatibility.md`, `README.md`,
`CHANGELOG.md`

- [ ] `docs/glm-5.3-flash.md` modeled on `docs/glm-5.2.md`: quick start, FP8 expert offload,
  resident set (KDA/DSA/mHC/vision/MTP), transformers >= 5.16 requirement, text-only scope,
  limitations (no DFlash, no vision, eager attention).
- [ ] Matrix row updated to status matching exactly what passed (B1+B2 -> keep
  experimental-with-tiny-harness; B3 pass -> `validated`, text-only sync generation
  [+ serving if B2 serving test passed on real checkpoint scope wording]).
- [ ] CHANGELOG Unreleased; docs-consistency test green.

---

## Sequencing

```
start:  [bg] GLM-5.3 download   [bg] GLM-5.3-Flash download   [bg] B0 venv build
        A1 -> A2 (code, primary venv, immediate)
B0 done:        B1 -> B2 (tf516)
GLM-5.3 dl done:         A3 -> A4
B2 + dl done:            B3 -> B4
```

Delivery: two branches / draft PRs - `test/glm-5.3-validation` (Track A),
`test/glm5-next-validation` (Track B). CI stays green on transformers 5.12.1 throughout.

## Risks

- **B1 unknowns:** first-ever execution of glm5_next wiring; WeightConverter renames and the
  mHC/KDA resident policy are the likeliest failure points. Fix minimally + regression-test.
- **transformers 5.16 availability:** if no stable `>=5.16,<6` wheel ships glm5_next, B0 pins
  the earliest release that does; record the exact version in the PR.
- **Download time/gating:** ~1.1 TB total; checkpoints assumed ungated (GLM family norm). If
  gated, needs an HF token before A3/B3.
- **sm_120:** both venv builds must use `MOE_ENABLE_SM120=1`; forgetting it fails at kernel
  load, not install.
