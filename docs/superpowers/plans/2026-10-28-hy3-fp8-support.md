# Hy3-FP8 (hy_v3) Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve `tencent/Hy3-FP8` (295B MoE, 21B active) on memory-constrained GPUs by offloading its ~290 GB of per-tensor-static-FP8 routed experts (new scale format, GLM-5.2 FP8-in-store design) while the ~9 GB non-expert stack stays resident in BF16.

**Architecture:** Path A (HF-native) from day one — transformers ≥ 5.6.0 ships `HYV3ForCausalLM` (pinned 5.12.1 verified). Reuse the GLM-5.2 FP8 offload shape: routed experts stay FP8 in the host store with scales delivered to the dispatcher, dequant on use; non-expert weights dequant to BF16 at load. The deltas vs GLM are (a) **per-tensor scalar scales** (`weight_scale`/`input_scale`) instead of 128×128 block `weight_scale_inv`, requiring a new `ExpertFormat`, and (b) a `HYV3MoE` block override (HF's batched `gate_up_proj [E, 2*inter, H]` params never materialize; the checkpoint's per-expert `gate/up/down_proj` keys feed the host store directly). MTP layer 80 is excluded from generation support (glmmoedsa-style `layer_id >= num_layers` guard).

**Tech Stack:** Python 3.12, torch ≥ 2.8, safetensors, moe-store (sibling repo `/mnt/raid0nvme0/leyang/moe-store`, editable install), transformers ≥ 5.6 (`hy_v3`), pytest.

**Tracking issue:** EfficientMoE/MoE-Infinity#283

---

## Verified facts (2026-10-28, via HF `config.json` + `model.safetensors.index.json`)

Do not re-derive these; they were confirmed against the live checkpoint.

- `config.json`: `model_type: hy_v3`, `architectures: ["HYV3ForCausalLM"]`, 80 hidden layers + 1 MTP layer (`num_nextn_predict_layers: 1`), hidden 4096, `moe_intermediate_size: 1536`, dense `intermediate_size: 13312`, `first_k_dense_replace: 1`, 192 experts / top-8 / 1 shared (`num_shared_experts: 1`), GQA 64 heads / 8 KV, head_dim 128, `qk_norm: true`, 256K context, vocab 120832, rope theta 11158840.
- Routing: `moe_router_use_sigmoid: true`, `moe_router_enable_expert_bias: true`, `router_scaling_factor: 2.826`, `route_norm: true`. Keys: `model.layers.{L}.mlp.router.gate.weight [192, 4096]` + `model.layers.{L}.mlp.expert_bias` for L in 1..80.
- `quantization_config: {quant_method: fp8, activation_scheme: static, kv_cache_scheme: static, ignored_layers: [lm_head, model.embed_tokens]}` — **no `weight_block_size`**; every quantized linear carries scalar `weight_scale` + `input_scale`.
- 140,432 tensors. Routed experts: `model.layers.{1..80}.mlp.experts.{0..191}.{gate_proj,up_proj,down_proj}.{weight,weight_scale,input_scale}` — 138,240 expert tensors; `gate/up.weight` F8_E4M3 `[1536, 4096]`, `down.weight` `[4096, 1536]` ⇒ ~18.9 MB FP8 per expert, ~3.6 GB per layer, ~290 GB total.
- Layer 0 is dense (`mlp.{gate,up,down}_proj`, 13312). Layers 1–79 are standard MoE decoder layers. Layer 80 is the MTP layer: own attention, own 192-expert MoE, plus `eh_proj`/`enorm`/`hnorm`/`final_layernorm` — **excluded from generation support** (resident-skip, same policy as glmmoedsa MTP).
- Shared expert per MoE layer: `mlp.shared_mlp.{gate,up,down}_proj` (FP8 per-tensor, resident).
- Non-layer keys: `model.embed_tokens.weight`, `model.norm.weight`, `lm_head.weight` (BF16, fp8-ignored).
- The expert key pattern matches MoE-Infinity's `_ROUTED_EXPERT_RE` (`\.mlp\.experts\.\d+\.`) and the `model.layers.{L}.mlp.experts` topology strings in `runtime/model_offload.py`; the glmmoedsa regex family in moe-store (`model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.`) matches with a new MTP guard at `num_hidden_layers = 80`.
- Mis-map hazard: `quant_method: fp8` currently routes `resolve_model_precision_capabilities` to `protected:fp8` → `ExpertFormat.GLM_FP8_BLOCK128`, and `_identify_fp8_blockwise_pairs` looks for `*_scale_inv` keys that do not exist here — the generic path must never fire for `hy_v3`.

## File structure

**moe-store** (`/mnt/raid0nvme0/leyang/moe-store`):

| File | Responsibility |
| --- | --- |
| `moe_store/registry/constants.py` (modify) | Guarded `hyv3` → `HYV3ForCausalLM` entry (transformers ≥ 5.6) + expert-type mapping. |
| `moe_store/parsing/hf_config.py` (modify) | `parse_expert_id` branch `model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.` with `layer_id >= num_hidden_layers` MTP guard; `parse_moe_param` (192 / top-8). |
| `tests/fixtures/hy_v3/config.json` (create) | Tiny-config fixture mirroring the real config fields above. |
| `tests/test_hyv3_registry.py` (create) | Registry resolution, expert-id parsing incl. MTP/shared/dense negative cases. |

**MoE-Infinity** (this repo):

| File | Responsibility |
| --- | --- |
| `moe_infinity/runtime/expert_precision.py` (modify) | New `ExpertFormat.FP8_PERTENSOR` (+ `ExecutionKind`), `PROTECTED_MODEL_PATHS["hy_v3"] = "protected:hy3_fp8_pertensor"`. |
| `moe_infinity/runtime/model_offload.py` (modify) | Per-tensor FP8 load-time split: routed-expert keys stay FP8 with scalar scales stashed; non-expert FP8 weights dequant to BF16 (`weight.to(bf16) * weight_scale`); `input_scale` tensors dropped for BF16 execution. Mirror `_has_fp8_blockwise` / `deliver_fp8_scales_to_dispatcher` with a per-tensor variant. |
| `moe_infinity/boundary/transformers_model_overrides.py` (modify) | `HYV3MoE` override: sigmoid + `expert_bias` top-8 routing (`router_scaling_factor`, `route_norm`), archer per-expert dispatch, shared expert resident. |
| `tests/python/unit/test_expert_precision_registry.py` (modify) | Protection + anti-mis-map regression (hy_v3 must NOT resolve to GLM_FP8_BLOCK128). |
| `tests/python/hy3/conftest.py` (create) | Real-checkpoint skip gate (pattern of `tests/python/v4/conftest.py`). |
| `tests/python/hy3/e2e_offload.py` (create) | GPU-gated end-to-end smoke: offloaded load, greedy decode. |
| `docs/model-compatibility.md` (modify) | Hy3 / Hy3-FP8 rows with honest evidence levels. |
| `README.md`, `CHANGELOG.md` (modify) | Support-matrix row + Unreleased entry — only in the phase whose evidence exists. |

Engine/native code: **no changes required to start** — BF16 execution after host-side (or copy-time Python) dequant works with the existing dispatcher; an optional later phase adds a scalar-scale fast path beside `extensions/kernel/v4_fp4/fp8_dequant.cu` for dequant-on-copy parity with GLM.

---

## Phase 0 — moe-store: registry + parsing

### Task 1: Guarded registry entry + tiny fixture

- [ ] Add `tests/fixtures/hy_v3/config.json` (tiny dims: 4 layers, 8 experts, top-2, `first_k_dense_replace: 1`, `num_nextn_predict_layers: 1`, real field names).
- [ ] Failing test `tests/test_hyv3_registry.py::test_registry_resolves_hyv3` — arch substring `hyv3` resolves from `MODEL_MAPPING_NAMES` when `transformers.HYV3ForCausalLM` imports; guarded otherwise (follow the `glm5next` guard pattern in `registry/constants.py`).
- [ ] Implement the guarded entry + expert-type mapping; run `pytest tests/test_hyv3_registry.py -q` → PASS.

### Task 2: `parse_expert_id` / `parse_moe_param`

- [ ] Failing parametrized tests:
  - `("model.layers.2.mlp.experts.171.down_proj.weight", (2, 171))`
  - `("model.layers.2.mlp.experts.171.down_proj.weight_scale", (2, 171))`
  - `("model.layers.2.mlp.experts.171.down_proj.input_scale", (2, 171))`
  - MTP guard: `("model.layers.80.mlp.experts.0.gate_proj.weight", (None, None))` with `num_hidden_layers = 80`
  - negatives: `shared_mlp`, `mlp.router.gate.weight`, `mlp.expert_bias`, layer-0 dense `mlp.gate_proj.weight`
- [ ] Implement branch (reuse glmmoedsa regex + MTP guard); `parse_moe_param` returns (192, 8) from the real fixture fields.
- [ ] `pytest tests/test_hyv3_registry.py -q` → PASS; commit.

## Phase 1 — MoE-Infinity: per-tensor FP8 precision protection

### Task 3: Format + protection registration

- [ ] Failing tests in `tests/python/unit/test_expert_precision_registry.py`: `hy_v3` resolves to the new protected reason; **regression pin**: a config with `model_type: hy_v3` + `quant_method: fp8` must NOT resolve to `GLM_FP8_BLOCK128`.
- [ ] Add `ExpertFormat.FP8_PERTENSOR`, execution kind (BF16 GEMM after dequant), `PROTECTED_MODEL_PATHS["hy_v3"]`, `_PROTECTED_FORMAT_BY_REASON` entry; keep ordering (protected map checked before generic quant_method map — verify with the test).
- [ ] Mirror in `_resolve_adaptive_precision` (`runtime/model_offload.py`, duplicated resolution logic).

### Task 4: Load-time split for per-tensor scales

- [ ] Unit tests with synthetic state dicts: `_has_fp8_pertensor` detection (`quant_method: fp8`, no `weight_block_size`, `*.weight_scale` scalar keys present); routed-expert keys retained FP8 + scale stash; non-expert `q_proj/shared_mlp/dense mlp` dequantized to BF16; `input_scale` keys consumed (not leaked into the module state dict).
- [ ] Implement beside the blockwise path in `model_offload.py`; scale delivery mirrors `deliver_fp8_scales_to_dispatcher` (scalar per expert-tensor).
- [ ] Note: embeddings/lm_head are `ignored_layers` (BF16 already) — assert no scale lookup is attempted for them.

## Phase 2 — HF-native MoE wrapper

### Task 5: `HYV3MoE` override

- [ ] CPU unit test: tiny-config `HYV3ForCausalLM`, swap in the override, assert routing parity vs stock HF block (sigmoid scores + `expert_bias` for selection only, bias-free weights for combine, `router_scaling_factor` 2.826, `route_norm` renormalization) on fixed logits.
- [ ] Implement in `boundary/transformers_model_overrides.py` following `SyncMixtralSparseMoeBlock` / glmmoedsa patterns: per-expert archer dispatch, shared expert + router resident, batched HF params never allocated (meta-device init or param stripping, matching the existing override approach).
- [ ] Wire arch → override in the boundary registration table.

## Phase 3 — MTP exclusion + e2e smoke

### Task 6: MTP resident-skip + smoke

- [ ] Layer-80 handling: MTP tensors (incl. its MoE) skipped from host-store registration and from resident load for basic generation; assert topology registers exactly layers 1–79 expert groups when `num_nextn_predict_layers: 1`.
- [ ] `tests/python/hy3/e2e_offload.py` (GPU + real-checkpoint gated): load `tencent/Hy3-FP8` offloaded, greedy-decode a short prompt, assert non-degenerate output; record evidence level in docs only after this passes.

## Phase 4 — Docs & evidence

- [ ] `docs/model-compatibility.md`: Hy3-FP8 row (`implemented/experimental` until the smoke passes on hardware, then `validated` with the exact command), plus BF16 `tencent/Hy3` row as `not recorded` follow-up.
- [ ] `CHANGELOG.md` Unreleased entry; README support-matrix row.
- [ ] Optional follow-up (separate issue): scalar-scale dequant-on-copy CUDA fast path beside `fp8_dequant.cu`; A/B harness modeled on `benchmarks/contextpilot/` once a serving baseline exists.
