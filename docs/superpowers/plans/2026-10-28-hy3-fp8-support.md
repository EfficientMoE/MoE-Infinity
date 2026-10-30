# Hy3-FP8 (hy_v3) Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve `tencent/Hy3-FP8` (295B MoE, 21B active) on memory-constrained GPUs by offloading its ~290 GB of per-tensor-static-FP8 routed experts while the ~9 GB non-expert stack stays resident in BF16.

**Architecture:** Path A (HF-native) from day one — transformers ≥ 5.6 ships `HYV3ForCausalLM` (pinned 5.12.1 verified). Follow the **glm5_next integration shape**: whole non-expert namespace resident, only routed experts offloaded, Sync MoE block with per-expert `ModuleList`. Follow the **GLM-5.2 FP8-in-store design** for expert bytes, with one twist: Hy3 scales are per-tensor scalars, so at load time each scalar `weight_scale` is **expanded to a uniform 128×128 block-scale grid** and delivered through the existing `glm_fp8_block128` dispatcher path — zero native/CUDA changes. Resident FP8 weights dequant to BF16 with a trivial scalar multiply. MTP layer 80 is excluded (same mechanics as GLM's tail MTP layer).

**Tech Stack:** Python 3.12, torch ≥ 2.8, safetensors, transformers ≥ 5.6 (`hy_v3`), moe-store (sibling repo `/mnt/raid0nvme0/leyang/moe-store`, editable install), pytest.

**Tracking issue:** EfficientMoE/MoE-Infinity#283 — **Draft PR:** #284

**Test command conventions** (run from the repo named in each task):
- MoE-Infinity venv python: `/mnt/raid0nvme0/leyang/MoE-Infinity/.venv/bin/python`
- moe-store tests: `cd /mnt/raid0nvme0/leyang/moe-store && /mnt/raid0nvme0/leyang/MoE-Infinity/.venv/bin/python -m pytest <file> -q`
- MoE-Infinity tests: `cd <worktree> && .venv/bin/python -m pytest <file> -q` (worktree uses the main checkout's venv via the same interpreter path)

---

## Verified facts (2026-10-28, via HF `config.json` + `model.safetensors.index.json`)

Do not re-derive these; they were confirmed against the live checkpoint and installed code.

- `config.json`: `model_type: hy_v3`, `architectures: ["HYV3ForCausalLM"]`, 80 hidden layers + 1 MTP layer (`num_nextn_predict_layers: 1`), hidden 4096, `moe_intermediate_size: 1536`, dense `intermediate_size: 13312`, `first_k_dense_replace: 1`, 192 experts / top-8 / `num_shared_experts: 1`, GQA 64 heads / 8 KV / head_dim 128, `qk_norm: true`, 256K context, vocab 120832.
- Routing: `moe_router_use_sigmoid: true`, `moe_router_enable_expert_bias: true`, `router_scaling_factor: 2.826`, `route_norm: true`.
- `quantization_config: {quant_method: fp8, activation_scheme: static, kv_cache_scheme: static, ignored_layers: [lm_head, model.embed_tokens]}` — **no `weight_block_size`**; every quantized linear carries scalar `weight_scale` + `input_scale`.
- 140,432 tensors. Routed experts: `model.layers.{1..80}.mlp.experts.{0..191}.{gate_proj,up_proj,down_proj}.{weight,weight_scale,input_scale}`; `gate/up.weight` F8_E4M3 `[1536, 4096]`, `down.weight` `[4096, 1536]` — both divide evenly by 128 in both dims.
- Layer 0 dense (`mlp.{gate,up,down}_proj`). Layer 80 = MTP (own attention + 192-expert MoE + `eh_proj`/`enorm`/`hnorm`/`final_layernorm`); the HF model builds layers 0–79 only, so layer-80 keys have no module targets.
- Router keys: `model.layers.{L}.mlp.router.gate.weight [192, 4096]` + `model.layers.{L}.mlp.expert_bias` (L = 1..80); shared expert `mlp.shared_mlp.{gate,up,down}_proj` (FP8 per-tensor).
- **Transformers key renames** (`transformers/conversion_mapping.py`, `mapping["hy_v3"]` = qwen2_moe copy + 3 `WeightRenaming` entries): `mlp.router.gate.weight → mlp.gate.weight`, `mlp.expert_bias → mlp.e_score_correction_bias`, `mlp.shared_mlp. → mlp.shared_experts.`. The qwen2_moe expert-fusion entries are `WeightConverter`s, which MoE-Infinity's `_resident_ckpt_key_renamer` deliberately skips — per-expert keys survive untouched.
- **HF modules** (`transformers/models/hy_v3/modeling_hy_v3.py`): `HYV3TopKRouter(config)` holds `self.weight [E, H]` and `forward(hidden_states, e_score_correction_bias) -> (router_logits, top_k_weights, top_k_index)` (sigmoid → `+bias` for selection only → gather raw sigmoid weights → normalize → `* router_scaling_factor`). `HYV3MLP(config, intermediate_size=None)` is a standard gate/up/down SiLU MLP. `HYV3MoE` holds batched `HYV3Experts` (`gate_up_proj [E, 2*inter, H]`) which the Sync block replaces entirely. `e_score_correction_bias` is a **buffer on `HYV3MoE`**, zeros-initialized.
- **Existing MoE-Infinity seams** (file `moe_infinity/runtime/model_offload.py`, line numbers at `origin/main` a0b4894):
  - `_identify_fp8_blockwise_pairs` (L533) matches only `*_scale_inv` → finds nothing for Hy3.
  - `_ROUTED_EXPERT_RE = r"\.mlp\.experts\.\d+\."` (L544) matches Hy3 expert keys as-is (incl. layer-80 MTP experts — harmless superset, same as GLM, see L547-555 comment).
  - `_has_fp8_blockwise` (L558) keys off `quant_method` containing "fp8" → already True for Hy3, so the L1210 branch fires.
  - FP8 split loop L1210-1243: routed → stash scale in `self._blockwise_fp8_scales[base]`; non-routed → `dequant_fp8_blockwise`.
  - `deliver_fp8_scales_to_dispatcher` (L874) pushes the stash to the native dispatcher (`set_scales`), which implements `glm_fp8_block128` dequant-on-copy with scale grid `[⌈N/128⌉, ⌈K/128⌉]` float32.
  - `_cast_state_dict_tensors` (L2298) keeps raw only `*_scale_inv` / fp8-dtype tensors; Hy3's float scalar scales would be cast to BF16 (precision loss) without a fix.
  - `_rebuild_blockwise_fp8_scales_from_ckpt` (L1982) re-stashes scales on cached-store runs, `*_scale_inv` only.
  - `_load_resident_shared_experts` (L1753) loads resident params from ckpt applying `WeightRenaming`s (L1662 `_resident_ckpt_key_renamer`); its `_resolve` (L1772) hard-requires `_scale_inv` for FP8 weights.
  - Residency policy gates: `_is_shared_expert_param` (L1622, glm5_next whole-namespace branch L1649), resident-loader call gate L1490-1495, resident-GPU-placement gate L1507-1509.
  - `first_k_dense_replace` gates keyed on **model name substring** `"deepseek"/"glm"`: L1370 and L2232 — `"tencent/Hy3-FP8"` matches neither; Hy3 has `first_k_dense_replace=1`.
  - `moe_infinity/entrypoints/big_modeling.py` L243-247 forces `use_native_engine=False` for `glm_moe_dsa`/`minimax_m3_vl`.
- **Precision registry** (`moe_infinity/runtime/expert_precision.py`): `PROTECTED_MODEL_PATHS` (L96) has no `hy_v3`; `quant_method: fp8` falls to generic `protected:fp8` → `GLM_FP8_BLOCK128` whose loader semantics (`_scale_inv` pairs) silently no-op on Hy3. Mirrored protected-first ordering in `_resolve_adaptive_precision` (model_offload.py L217-229) uses `PROTECTED_MODEL_PATHS` too, so one dict entry fixes both.
- **moe-store**: no `hyv3` anywhere. Registry guard pattern = `glm5next` (constants.py L118-120); `parse_expert_id` template = `glmmoedsa` branch (hf_config.py L334-348, incl. the `layer_id >= num_layers` MTP guard); Sync-block template = `SyncGlmMoeDsaMoEBlock` (wrappers/glm_moe_dsa.py) with the glm5_next subclassing trick (wrappers/glm5_next.py).

## File structure

**moe-store** (`/mnt/raid0nvme0/leyang/moe-store`):

| File | Responsibility |
| --- | --- |
| `moe_store/registry/constants.py` (modify) | Guarded `hyv3` → `HYV3ForCausalLM` entry, expert-type 5. |
| `moe_store/parsing/hf_config.py` (modify) | `parse_moe_param` + `parse_expert_id` branches with MTP guard. |
| `moe_store/wrappers/hy_v3.py` (create) | `SyncHYV3MoEBlock` (subclass of `SyncGlmMoeDsaMoEBlock`). |
| `moe_store/wrappers/__init__.py` (modify) | Export `SyncHYV3MoEBlock`. |
| `tests/fixtures/hy_v3/config.json` (create) | Tiny-dims fixture with real field names. |
| `tests/test_hyv3_registry.py` (create) | Registry guard, type resolution, parsing incl. negatives. |
| `tests/test_hyv3_wrappers.py` (create) | Block layout names, routing parity, rename-mapping pin. |

**MoE-Infinity** (worktree `/mnt/raid0nvme0/leyang/MoE-Infinity-hy3-fp8`, branch `feat/hy3-fp8-support`):

| File | Responsibility |
| --- | --- |
| `moe_infinity/runtime/expert_precision.py` (modify) | `PROTECTED_MODEL_PATHS["hy_v3"]` + reason entry. |
| `moe_infinity/runtime/model_offload.py` (modify) | Per-tensor pair identification, scalar→block-grid expansion, state-dict split, cast keep-raw, rebuild path, resident-loader scalar dequant, residency gates, `first_k_dense_replace` gates. |
| `moe_infinity/boundary/transformers_model_overrides.py` (modify) | Install/restore `SyncHYV3MoEBlock` for `hy_v3`. |
| `moe_infinity/entrypoints/big_modeling.py` (modify) | Force `use_native_engine=False` for `hy_v3`. |
| `tests/python/unit/test_expert_precision_registry.py` (modify) | Protection + anti-mis-map pin. |
| `tests/python/unit/test_fp8_pertensor.py` (create) | Pair id, grid expansion, dequant, state-dict split. |
| `tests/python/hy3/conftest.py` (create) | Real-checkpoint skip gate. |
| `tests/python/hy3/e2e_offload.py` (create) | GPU-gated greedy-decode smoke. |
| `docs/model-compatibility.md`, `CHANGELOG.md`, `README.md` (modify) | Evidence rows (last task only). |

Engine/native code: **no changes** — after scalar→block-grid expansion the store format is byte-and-semantics-identical to `glm_fp8_block128`, which `core/` and `extensions/kernel/v4_fp4/fp8_dequant.cu` already implement.

---

## Phase 0 — moe-store: registry + parsing

### Task 1: Tiny fixture + guarded registry entry

**Repo:** moe-store
**Files:**
- Create: `tests/fixtures/hy_v3/config.json`
- Create: `tests/test_hyv3_registry.py`
- Modify: `moe_store/registry/constants.py`

- [ ] **Step 1: Write the fixture** — `tests/fixtures/hy_v3/config.json`:

```json
{
  "architectures": ["HYV3ForCausalLM"],
  "model_type": "hy_v3",
  "hidden_size": 64,
  "intermediate_size": 128,
  "moe_intermediate_size": 32,
  "num_hidden_layers": 4,
  "num_nextn_predict_layers": 1,
  "first_k_dense_replace": 1,
  "num_experts": 8,
  "num_experts_per_tok": 2,
  "num_shared_experts": 1,
  "num_attention_heads": 4,
  "num_key_value_heads": 2,
  "head_dim": 16,
  "qk_norm": true,
  "hidden_act": "silu",
  "moe_router_use_sigmoid": true,
  "moe_router_enable_expert_bias": true,
  "router_scaling_factor": 2.826,
  "route_norm": true,
  "rms_norm_eps": 1e-05,
  "vocab_size": 128,
  "max_position_embeddings": 512,
  "quantization_config": {
    "quant_method": "fp8",
    "activation_scheme": "static",
    "kv_cache_scheme": "static",
    "ignored_layers": ["lm_head", "model.embed_tokens"]
  }
}
```

- [ ] **Step 2: Write the failing registry tests** — `tests/test_hyv3_registry.py` (mirror `tests/test_deepseekv41_registry.py`):

```python
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import json
from pathlib import Path

import pytest
from transformers import PretrainedConfig

import moe_store.registry.constants as constants
from moe_store.parsing.hf_config import parse_expert_id, parse_moe_param

FIXTURE = Path(__file__).parent / "fixtures" / "hy_v3" / "config.json"


@pytest.fixture()
def hy3_config() -> PretrainedConfig:
    return PretrainedConfig.from_dict(json.loads(FIXTURE.read_text()))


def test_hyv3_registration_is_guarded() -> None:
    model_class = constants.HYV3ForCausalLM
    assert ("hyv3" in constants.MODEL_MAPPING_NAMES) is (
        model_class is not None
    )
    if model_class is not None:
        assert constants.MODEL_MAPPING_NAMES["hyv3"] is model_class
        assert constants.MODEL_MAPPING_TYPES["hyv3"] == 5


def test_parse_expert_type_resolves_hyv3(hy3_config) -> None:
    if constants.HYV3ForCausalLM is None:
        pytest.skip("transformers build lacks HYV3ForCausalLM")
    assert constants.parse_expert_type(hy3_config) == 5
```

- [ ] **Step 3: Run to verify failure**

Run: `cd /mnt/raid0nvme0/leyang/moe-store && /mnt/raid0nvme0/leyang/MoE-Infinity/.venv/bin/python -m pytest tests/test_hyv3_registry.py -q`
Expected: FAIL — `AttributeError: module 'moe_store.registry.constants' has no attribute 'HYV3ForCausalLM'`

- [ ] **Step 4: Implement the guarded entry** — `moe_store/registry/constants.py`. After the `MiniMaxM3SparseForConditionalGeneration` try/except block (L50-53), add:

```python
try:
    from transformers import HYV3ForCausalLM
except ImportError:
    HYV3ForCausalLM = None
```

After the `minimaxm3` registration block (L148-150), add:

```python
# Hy3 (arch "HYV3ForCausalLM", model_type "hy_v3") routes 192 per-expert
# gate_proj/up_proj/down_proj experts (expert-type 5) with a sigmoid +
# expert_bias router. The batched HF HYV3Experts module is replaced by the
# per-expert SyncHYV3MoEBlock on offload. Requires transformers >= 5.6 which
# ships the class; registered only when importable (mirrors the guards above).
if HYV3ForCausalLM is not None:
    MODEL_MAPPING_NAMES["hyv3"] = HYV3ForCausalLM
    MODEL_MAPPING_TYPES["hyv3"] = 5
```

- [ ] **Step 5: Re-run**

Run: same pytest command. Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
cd /mnt/raid0nvme0/leyang/moe-store
git add moe_store/registry/constants.py tests/fixtures/hy_v3/config.json tests/test_hyv3_registry.py
git commit -m "feat(hy3): guarded hyv3 registry entry (EfficientMoE/MoE-Infinity#283)"
```

### Task 2: `parse_moe_param` / `parse_expert_id` branches

**Repo:** moe-store
**Files:**
- Modify: `moe_store/parsing/hf_config.py`
- Modify: `tests/test_hyv3_registry.py`

- [ ] **Step 1: Append failing parse tests** to `tests/test_hyv3_registry.py`:

```python
def test_parse_moe_param_hyv3(hy3_config) -> None:
    assert parse_moe_param(hy3_config) == (4, 8, 0)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("model.layers.2.mlp.experts.171.down_proj.weight", (2, 171)),
        # scale members of a real expert parse identically to the weight.
        ("model.layers.2.mlp.experts.0.gate_proj.weight_scale", (2, 0)),
        ("model.layers.2.mlp.experts.0.up_proj.input_scale", (2, 0)),
        ("model.layers.1.mlp.experts.7.up_proj.weight", (1, 7)),
        # MTP layer (index == num_hidden_layers) must NOT be offloaded.
        ("model.layers.4.mlp.experts.0.gate_proj.weight", (None, None)),
        # shared expert, router, bias, and dense layer-0 MLP are not experts.
        ("model.layers.2.mlp.shared_mlp.gate_proj.weight", (None, None)),
        ("model.layers.2.mlp.router.gate.weight", (None, None)),
        ("model.layers.2.mlp.expert_bias", (None, None)),
        ("model.layers.0.mlp.gate_proj.weight", (None, None)),
    ],
)
def test_parse_expert_id_hyv3(hy3_config, name, expected) -> None:
    assert parse_expert_id(name, hy3_config) == expected
```

(The fixture has `num_hidden_layers: 4`, so layer 4 plays the role the real checkpoint's layer 80 plays.)

- [ ] **Step 2: Run to verify failure**

Run: `cd /mnt/raid0nvme0/leyang/moe-store && /mnt/raid0nvme0/leyang/MoE-Infinity/.venv/bin/python -m pytest tests/test_hyv3_registry.py -q`
Expected: FAIL — `RuntimeError: Unsupported architecture hyv3forcausallm` from `parse_moe_param`.

- [ ] **Step 3: Implement the branches** — `moe_store/parsing/hf_config.py`.

In `parse_moe_param`, after the `glm5next` branch (L154-159), add:

```python
    elif "hyv3" in arch:
        num_encoder_layers = 0
        num_decoder_layers = config.num_hidden_layers
        num_layers = config.num_hidden_layers
        num_experts = config.num_experts
```

In `parse_expert_id`, after the `glm5next` branch (L349-364), add:

```python
    elif "hyv3" in arch:
        decoder_sparse_step = 1
        layer_type = "decoder"

        # Confirmed against the live tencent/Hy3-FP8 index (2026-10-28):
        # "model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.
        # {weight,weight_scale,input_scale}". Shared experts use
        # ".mlp.shared_mlp." and the router ".mlp.router.gate." /
        # ".mlp.expert_bias", none of which contain "experts.<id>.".
        result = re.findall(
            r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.", param_name
        )
        if result:
            layer_id, expert_id = result[0]
            layer_id = int(layer_id)
            expert_id = int(expert_id)
            # MTP guard: Hy3 stores its MTP layer (own 192-expert MoE) at
            # index num_hidden_layers (80), same convention as GlmMoeDsa.
            if layer_id >= num_layers:
                return None, None
```

- [ ] **Step 4: Re-run**

Run: same pytest command. Expected: PASS (12 tests)

- [ ] **Step 5: Commit**

```bash
git add moe_store/parsing/hf_config.py tests/test_hyv3_registry.py
git commit -m "feat(hy3): parse_moe_param/parse_expert_id with MTP guard (EfficientMoE/MoE-Infinity#283)"
```

## Phase 1 — MoE-Infinity: precision protection

### Task 3: `PROTECTED_MODEL_PATHS["hy_v3"]` + anti-mis-map pin

**Repo:** MoE-Infinity worktree
**Files:**
- Modify: `moe_infinity/runtime/expert_precision.py`
- Modify: `tests/python/unit/test_expert_precision_registry.py`

- [ ] **Step 1: Write the failing tests.** In `tests/python/unit/test_expert_precision_registry.py`, add `("hy_v3", "protected:hy3_fp8_pertensor")` to the parametrize list of `test_model_specific_low_bit_paths_are_protected` (L56-ish), and append (mirroring `test_deepseek_v41_never_resolves_to_glm_fp8_identity`):

```python
def test_hy3_resolves_pertensor_reason_not_generic_fp8():
    # Hy3-FP8 carries quant_method "fp8" but per-tensor scalar scales, not
    # 128x128 block scale_inv tensors. The generic protected:fp8 reason must
    # never catch hy_v3: the loader keys per-tensor handling off the
    # hy3-specific reason staying distinct.
    config = SimpleNamespace(
        model_type="hy_v3",
        quantization_config={"quant_method": "fp8"},
    )
    caps = resolve_model_precision_capabilities(config, set())
    assert caps.protected_reason == "protected:hy3_fp8_pertensor"
    assert caps.protected_reason != "protected:fp8"
    # After load-time scalar->block-grid expansion the store/dispatcher
    # format identity is the existing glm_fp8_block128 path.
    assert set(caps.formats) == {ExpertFormat.GLM_FP8_BLOCK128}
    assert all(cap.protected for cap in caps.formats.values())
```

(Match the file's existing import style — it already imports `resolve_model_precision_capabilities` and `ExpertFormat`; reuse its config-stub helper if one exists instead of `SimpleNamespace`.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/python/unit/test_expert_precision_registry.py -q`
Expected: FAIL — resolved reason is `protected:fp8`.

- [ ] **Step 3: Implement.** In `moe_infinity/runtime/expert_precision.py`:

Append to `PROTECTED_MODEL_PATHS` (L96-106):

```python
    # Hy3-FP8 routed experts are FP8 E4M3 with *per-tensor* scalar
    # weight_scale/input_scale (activation_scheme static, no
    # weight_block_size). The loader expands each scalar to a uniform
    # 128x128 block grid, after which the store format is exactly
    # glm_fp8_block128 — but the generic protected:fp8 reason must never
    # catch this model_type: its loader path looks for *_scale_inv block
    # tensors that do not exist here and would silently skip dequant.
    "hy_v3": "protected:hy3_fp8_pertensor",
```

Append to `_PROTECTED_FORMAT_BY_REASON` (L174-201):

```python
    "protected:hy3_fp8_pertensor": (
        ExpertFormat.GLM_FP8_BLOCK128,
        ExecutionKind.FP8_DEQUANT_BF16_GEMM,
    ),
```

- [ ] **Step 4: Re-run**

Run: `.venv/bin/python -m pytest tests/python/unit/test_expert_precision_registry.py -q`
Expected: PASS. (`_resolve_adaptive_precision` in model_offload.py reads the same dict at L217, so no second edit.)

- [ ] **Step 5: Commit**

```bash
git add moe_infinity/runtime/expert_precision.py tests/python/unit/test_expert_precision_registry.py
git commit -m "feat(hy3): protect hy_v3 experts with per-tensor FP8 reason (#283)"
```

### Task 4: Per-tensor FP8 helpers (pure functions)

**Repo:** MoE-Infinity worktree
**Files:**
- Modify: `moe_infinity/runtime/model_offload.py` (module level, next to `_identify_fp8_blockwise_pairs` L533)
- Create: `tests/python/unit/test_fp8_pertensor.py`

- [ ] **Step 1: Write the failing tests** — `tests/python/unit/test_fp8_pertensor.py`:

```python
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import torch

from moe_infinity.runtime.model_offload import (
    _dequant_fp8_pertensor,
    _expand_pertensor_scale_to_blocks,
    _identify_fp8_pertensor_pairs,
    _split_fp8_pertensor_state_dict,
)


def test_identify_pertensor_pairs_matches_weight_scale_suffix():
    keys = [
        "model.layers.2.mlp.experts.0.gate_proj.weight",
        "model.layers.2.mlp.experts.0.gate_proj.weight_scale",
        "model.layers.2.mlp.experts.0.gate_proj.input_scale",
        "model.layers.2.self_attn.q_proj.weight",
        "model.layers.2.self_attn.q_proj.weight_scale",
        # blockwise key must NOT be picked up by the per-tensor identifier
        "model.layers.2.mlp.experts.1.up_proj.weight",
        "model.layers.2.mlp.experts.1.up_proj.weight_scale_inv",
        # orphan scale without a weight
        "model.layers.3.mlp.experts.0.up_proj.weight_scale",
    ]
    pairs = _identify_fp8_pertensor_pairs(keys)
    assert pairs == [
        (
            "model.layers.2.mlp.experts.0.gate_proj.weight",
            "model.layers.2.mlp.experts.0.gate_proj.weight_scale",
        ),
        (
            "model.layers.2.self_attn.q_proj.weight",
            "model.layers.2.self_attn.q_proj.weight_scale",
        ),
    ]


def test_expand_scalar_scale_to_block_grid():
    scale = torch.tensor(0.5)
    grid = _expand_pertensor_scale_to_blocks((1536, 4096), scale)
    assert grid.shape == (12, 32)
    assert grid.dtype == torch.float32
    assert torch.all(grid == 0.5)
    # non-multiple dims round up
    assert _expand_pertensor_scale_to_blocks((130, 100), scale).shape == (2, 1)


def test_dequant_fp8_pertensor_roundtrip():
    w_ref = torch.tensor([[0.5, -1.0], [2.0, 0.25]], dtype=torch.float32)
    scale = torch.tensor(2.0)
    w_fp8 = (w_ref / scale).to(torch.float8_e4m3fn)
    out = _dequant_fp8_pertensor(w_fp8, scale, dtype=torch.bfloat16)
    assert out.dtype == torch.bfloat16
    assert torch.allclose(out.float(), w_ref, atol=1e-2)


def test_split_routes_experts_to_scales_and_dequants_resident():
    expert_w = torch.ones(256, 128).to(torch.float8_e4m3fn)
    attn_w = (torch.ones(128, 128) * 0.5).to(torch.float8_e4m3fn)
    state_dict = {
        "model.layers.2.mlp.experts.0.gate_proj.weight": expert_w,
        "model.layers.2.mlp.experts.0.gate_proj.weight_scale": torch.tensor(2.0),
        "model.layers.2.mlp.experts.0.gate_proj.input_scale": torch.tensor(1.0),
        "model.layers.2.self_attn.q_proj.weight": attn_w,
        "model.layers.2.self_attn.q_proj.weight_scale": torch.tensor(4.0),
        "model.layers.2.self_attn.q_proj.input_scale": torch.tensor(1.0),
        "model.norm.weight": torch.ones(128, dtype=torch.bfloat16),
    }
    scales = {}
    _split_fp8_pertensor_state_dict(state_dict, scales)

    # routed expert: weight stays FP8, scale expanded to the block grid
    k = "model.layers.2.mlp.experts.0.gate_proj.weight"
    assert state_dict[k].dtype == torch.float8_e4m3fn
    assert scales[k].shape == (2, 1)
    assert torch.all(scales[k] == 2.0)

    # resident weight: dequantized to BF16 in place
    q = state_dict["model.layers.2.self_attn.q_proj.weight"]
    assert q.dtype == torch.bfloat16
    assert torch.allclose(q.float(), torch.full((128, 128), 2.0), atol=1e-2)

    # every *_scale / input_scale key consumed; untouched keys survive
    assert not [key for key in state_dict if key.endswith("_scale")]
    assert not [key for key in state_dict if key.endswith(".input_scale")]
    assert "model.norm.weight" in state_dict
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/python/unit/test_fp8_pertensor.py -q`
Expected: FAIL — ImportError (names don't exist).

- [ ] **Step 3: Implement** in `moe_infinity/runtime/model_offload.py`, directly below `_identify_fp8_blockwise_pairs` (L541):

```python
def _identify_fp8_pertensor_pairs(keys):
    # Hy3-FP8 style: scalar scales named "<weight key>_scale" (i.e.
    # "...weight" + "_scale"), distinct from the GLM "..._scale_inv" block
    # tensors handled by _identify_fp8_blockwise_pairs above.
    key_set = set(keys)
    pairs = []
    for k in keys:
        if k.endswith(".weight_scale"):
            base = k[: -len("_scale")]
            if base in key_set:
                pairs.append((base, k))
    return pairs


def _expand_pertensor_scale_to_blocks(
    weight_shape, scale: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    # Expand a per-tensor scalar scale to the uniform [ceil(N/b), ceil(K/b)]
    # float32 grid the native glm_fp8_block128 dequant-on-copy path consumes.
    # A uniform grid makes blockwise dequant numerically identical to
    # per-tensor dequant, so no new native format is needed.
    n, k = int(weight_shape[0]), int(weight_shape[1])
    rows = (n + block_size - 1) // block_size
    cols = (k + block_size - 1) // block_size
    return torch.full((rows, cols), float(scale), dtype=torch.float32)


def _dequant_fp8_pertensor(
    weight: torch.Tensor, scale: torch.Tensor, *, dtype: torch.dtype
) -> torch.Tensor:
    return (weight.to(torch.float32) * float(scale)).to(dtype)


def _split_fp8_pertensor_state_dict(state_dict, scales_out) -> None:
    # Per-tensor sibling of the blockwise split at the _has_fp8_blockwise
    # call site: routed experts stay FP8 with their scalar scale expanded
    # into scales_out; everything else dequantizes to BF16 in place. All
    # consumed scale keys and the activation input_scale keys (unused by
    # BF16 execution) are removed from the state dict.
    for base_key, scale_key in _identify_fp8_pertensor_pairs(
        list(state_dict.keys())
    ):
        w = state_dict.get(base_key)
        s = state_dict.get(scale_key)
        if w is None or s is None:
            continue
        if w.dtype != torch.float8_e4m3fn:
            del state_dict[scale_key]
            continue
        if _is_routed_expert_key(base_key):
            scales_out[base_key] = _expand_pertensor_scale_to_blocks(
                w.shape, s
            )
        else:
            state_dict[base_key] = _dequant_fp8_pertensor(
                w, s, dtype=torch.bfloat16
            )
        del state_dict[scale_key]
    for k in [k for k in state_dict if k.endswith(".input_scale")]:
        del state_dict[k]
```

- [ ] **Step 4: Re-run**

Run: `.venv/bin/python -m pytest tests/python/unit/test_fp8_pertensor.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add moe_infinity/runtime/model_offload.py tests/python/unit/test_fp8_pertensor.py
git commit -m "feat(hy3): per-tensor FP8 helpers with block-grid scale expansion (#283)"
```

### Task 5: Wire the split into the load path

**Repo:** MoE-Infinity worktree
**Files:**
- Modify: `moe_infinity/runtime/model_offload.py` (three call sites)

No isolated unit test exists for the monolithic load loop; the helpers are covered by Task 4 and the end-to-end behavior by Task 10. Keep each edit minimal.

- [ ] **Step 1: First-run split.** At the end of the `if _has_fp8_blockwise(self.config):` block (after the blockwise `for base_key, scale_key in fp8_pairs:` loop ending at L1243), add:

```python
                            _split_fp8_pertensor_state_dict(
                                state_dict, self._blockwise_fp8_scales
                            )
```

(The `self._blockwise_fp8_scales = {}` guard at L1215-1216 already ran.)

- [ ] **Step 2: Keep scalar scales out of the BF16 cast.** In `_cast_state_dict_tensors` (L2298+), change the keep-raw condition:

```python
                if is_blockwise_fp8_ckpt and (
                    k.endswith("_scale_inv")
                    or k.endswith("_scale")
                    or v.dtype == torch.float8_e4m3fn
                ):
```

(`"_scale"` catches both `weight_scale` and `input_scale`; `_scale_inv` keeps its own arm because it does not end with `"_scale"`.)

- [ ] **Step 3: Cached-store (second-run) rebuild.** In `_rebuild_blockwise_fp8_scales_from_ckpt` (L1982), inside the per-ckpt `safe_open` loop after the `_scale_inv` loop (L2015-2020), add:

```python
                for k in keys:
                    if not k.endswith(".weight_scale"):
                        continue
                    base = k[: -len("_scale")]
                    if base not in keys:
                        continue
                    # Safe superset, same policy as the _scale_inv loop
                    # above: the dispatcher only applies scales to routed
                    # weights actually stored as FP8.
                    shape = f.get_slice(base).get_shape()
                    self._blockwise_fp8_scales[base] = (
                        _expand_pertensor_scale_to_blocks(
                            shape, f.get_tensor(k)
                        )
                    )
```

- [ ] **Step 4: Resident loader scalar dequant.** In `_load_resident_shared_experts`:

Replace the `_resolve` body (L1772-1782) with:

```python
        def _resolve(param, name, weight, scale):
            if weight.dtype == torch.float8_e4m3fn:
                if scale is None:
                    raise RuntimeError(
                        f"FP8 resident weight {name!r} is missing its scale "
                        "(_scale_inv block tensor or per-tensor weight_scale); "
                        "cannot dequantize."
                    )
                if scale.numel() == 1:
                    weight = _dequant_fp8_pertensor(
                        weight, scale, dtype=param.dtype
                    )
                else:
                    weight = dequant_fp8_blockwise(
                        weight, scale, dtype=param.dtype, block_size=128
                    )
            return weight.to(dtype=param.dtype, device="cpu").contiguous()
```

And extend both scale lookups to fall back to the per-tensor suffix — safetensors branch (L1795):

```python
                        scale_key = keys.get(name + "_scale_inv") or keys.get(
                            name + "_scale"
                        )
```

torch.load branch (L1832):

```python
                    param.data = _resolve(
                        param,
                        name,
                        state[name],
                        state.get(name + "_scale_inv", state.get(name + "_scale")),
                    )
```

- [ ] **Step 5: Run the existing suites to catch regressions**

Run: `.venv/bin/python -m pytest tests/python/unit -q`
Expected: PASS (no existing test exercises `_scale` suffixes; blockwise behavior unchanged).

- [ ] **Step 6: Commit**

```bash
git add moe_infinity/runtime/model_offload.py
git commit -m "feat(hy3): wire per-tensor FP8 split into load, rebuild, resident paths (#283)"
```

### Task 6: Residency policy + topology gates

**Repo:** MoE-Infinity worktree
**Files:**
- Modify: `moe_infinity/runtime/model_offload.py` (five call sites)
- Modify: `moe_infinity/entrypoints/big_modeling.py` (L243-247)

- [ ] **Step 1: Whole-namespace resident policy.** In `_is_shared_expert_param` (L1622), after the `minimax_m3_vl` branch (L1657-1659), add:

```python
        # hy_v3 (Hy3-FP8): same whole-namespace policy as glm5_next. Only the
        # routed experts are offloaded; attention, dense layer-0 MLP, shared
        # experts (module name "shared_experts" after the WeightRenaming from
        # on-disk "shared_mlp"), router gate, expert bias, and lm_head stay
        # resident (parse_expert_id returns None for every non-routed key).
        if getattr(self.config, "model_type", "") == "hy_v3":
            _, expert_id = parse_expert_id(name, self.config)
            return expert_id is None
```

- [ ] **Step 2: Resident-loader + GPU-placement gates.** Add `"hy_v3"` to both model_type tuples:
  - L1490-1495: `("glm_moe_dsa", "qwen3_5_moe", "glm5_next", "minimax_m3_vl", "hy_v3")`
  - L1507-1509: `("glm5_next", "minimax_m3_vl", "hy_v3")`

- [ ] **Step 3: Load the router bias parameter.** `_load_resident_shared_experts` builds `wanted` from `model.named_parameters()` (L1759-1763). `SyncHYV3MoEBlock` (Task 7) deliberately declares `e_score_correction_bias` as an `nn.Parameter` (not a buffer like HF's `HYV3MoE`) precisely so this loop finds it; the renamer maps the on-disk `mlp.expert_bias` to `mlp.e_score_correction_bias`. **No code change here** — this step is a design note; verify in Task 7's layout test.

- [ ] **Step 4: `first_k_dense_replace` gates.** The two model-*name* gates miss `"tencent/Hy3-FP8"`. Change both to also honor the model type:

L1370:

```python
                if (
                    "deepseek" in model_name
                    or "glm" in model_name.lower()
                    or getattr(self.config, "model_type", "") == "hy_v3"
                ):
```

L2232:

```python
        if (
            "deepseek" in self.model_name
            or "glm" in self.model_name.lower()
            or getattr(self.config, "model_type", "") == "hy_v3"
        ):
```

(`moe_text_config(self.config).first_k_dense_replace` returns 1 for Hy3 — the attr is top-level and `moe_text_config` falls back to the config itself.)

- [ ] **Step 5: Force the HF generation path.** In `moe_infinity/entrypoints/big_modeling.py` L243-247, extend the tuple and comment:

```python
        # GLM-DSA remains unsupported by the native engine. ... (keep existing
        # comment) ... hy_v3 (Hy3-FP8) starts on the HF eager path like
        # GLM-5.2; a PagedAttention shim is a follow-up.
        if getattr(model_config, "model_type", "") in (
            "glm_moe_dsa",
            "minimax_m3_vl",
            "hy_v3",
        ):
            self.use_native_engine = False
```

- [ ] **Step 6: Run unit suite, commit**

Run: `.venv/bin/python -m pytest tests/python/unit -q` — Expected: PASS

```bash
git add moe_infinity/runtime/model_offload.py moe_infinity/entrypoints/big_modeling.py
git commit -m "feat(hy3): whole-namespace residency + first_k_dense/topology gates (#283)"
```

## Phase 2 — Sync MoE block + boundary wiring

### Task 7: `SyncHYV3MoEBlock` (moe-store)

**Repo:** moe-store
**Files:**
- Create: `moe_store/wrappers/hy_v3.py`
- Modify: `moe_store/wrappers/__init__.py`
- Create: `tests/test_hyv3_wrappers.py`

- [ ] **Step 1: Write the failing tests** — `tests/test_hyv3_wrappers.py`:

```python
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import json
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
try:
    from transformers.models.hy_v3.configuration_hy_v3 import HYV3Config
    from transformers.models.hy_v3.modeling_hy_v3 import HYV3MoE
except ImportError:
    pytest.skip("transformers build lacks hy_v3", allow_module_level=True)

from moe_store.wrappers.hy_v3 import SyncHYV3MoEBlock

FIXTURE = Path(__file__).parent / "fixtures" / "hy_v3" / "config.json"


@pytest.fixture()
def tiny_config() -> HYV3Config:
    raw = json.loads(FIXTURE.read_text())
    raw.pop("quantization_config", None)
    return HYV3Config(**raw)


def test_block_param_names_match_renamed_checkpoint_layout(tiny_config):
    block = SyncHYV3MoEBlock(tiny_config)
    names = {n for n, _ in block.named_parameters()}
    # After transformers' hy_v3 WeightRenaming entries the on-disk keys
    # "mlp.router.gate.weight" / "mlp.expert_bias" / "mlp.shared_mlp.*"
    # become "mlp.gate.weight" / "mlp.e_score_correction_bias" /
    # "mlp.shared_experts.*"; routed expert keys are untouched.
    assert "gate.weight" in names
    assert "e_score_correction_bias" in names  # Parameter, NOT a buffer
    assert "experts.0.gate_proj.weight" in names
    assert "experts.7.down_proj.weight" in names
    assert "shared_experts.up_proj.weight" in names


def test_hy_v3_weight_renamings_cover_block_layout(tiny_config):
    from transformers.conversion_mapping import (
        get_checkpoint_conversion_mapping,
    )

    entries = [
        e
        for e in (get_checkpoint_conversion_mapping("hy_v3") or [])
        if type(e).__name__ == "WeightRenaming"
    ]

    def rename(key):
        for e in entries:
            renamed, matched = e.rename_source_key(key)
            if matched is not None:
                key = renamed
        return key

    assert (
        rename("model.layers.2.mlp.router.gate.weight")
        == "model.layers.2.mlp.gate.weight"
    )
    assert (
        rename("model.layers.2.mlp.expert_bias")
        == "model.layers.2.mlp.e_score_correction_bias"
    )
    assert (
        rename("model.layers.2.mlp.shared_mlp.gate_proj.weight")
        == "model.layers.2.mlp.shared_experts.gate_proj.weight"
    )
    # routed expert keys must pass through unchanged
    key = "model.layers.2.mlp.experts.0.gate_proj.weight"
    assert rename(key) == key


def test_routing_parity_with_stock_hyv3_moe(tiny_config):
    torch.manual_seed(0)
    stock = HYV3MoE(tiny_config)
    block = SyncHYV3MoEBlock(tiny_config)
    # copy router weight and give both the same non-zero bias
    bias = torch.randn(tiny_config.num_experts) * 0.1
    with torch.no_grad():
        block.gate.weight.copy_(stock.gate.weight)
        block.e_score_correction_bias.copy_(bias)
        stock.e_score_correction_bias.copy_(bias)

    hidden = torch.randn(5, tiny_config.hidden_size)
    _, ref_w, ref_idx = stock.gate(hidden, stock.e_score_correction_bias)
    idx, w = block._route(hidden)
    assert torch.equal(idx, ref_idx)
    assert torch.allclose(w, ref_w)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /mnt/raid0nvme0/leyang/moe-store && /mnt/raid0nvme0/leyang/MoE-Infinity/.venv/bin/python -m pytest tests/test_hyv3_wrappers.py -q`
Expected: FAIL — `ModuleNotFoundError: moe_store.wrappers.hy_v3`

- [ ] **Step 3: Implement** — `moe_store/wrappers/hy_v3.py`:

```python
"""Sync MoE block for Hy3 (hy_v3).

Router semantics come straight from the HF HYV3TopKRouter (sigmoid scoring,
expert-bias used for selection only, route_norm renormalization,
router_scaling_factor), reused as-is so parity is structural. The batched HF
HYV3Experts module (gate_up_proj [E, 2*inter, H]) is replaced with a
per-expert ModuleList whose parameter names (experts.<i>.gate_proj/up_proj/
down_proj) match the per-expert checkpoint layout the offload engine indexes.

Naming: parameters follow the post-WeightRenaming module namespace
(gate.weight, e_score_correction_bias, shared_experts.*) so the resident
loader's checkpoint-key renamer maps on-disk keys (mlp.router.gate.weight,
mlp.expert_bias, mlp.shared_mlp.*) onto them. e_score_correction_bias is an
nn.Parameter here — HF keeps it as a buffer, but the resident loader only
materializes parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from transformers.models.hy_v3.modeling_hy_v3 import (
        HYV3MLP,
        HYV3TopKRouter,
    )

    _HYV3_AVAILABLE = True
except ImportError:
    _HYV3_AVAILABLE = False
    HYV3MLP = HYV3TopKRouter = None

from moe_store.wrappers.glm_moe_dsa import SyncGlmMoeDsaMoEBlock


class SyncHYV3MoEBlock(SyncGlmMoeDsaMoEBlock):
    def __init__(self, config):
        if not _HYV3_AVAILABLE:
            raise ImportError(
                "transformers >= 5.6 is required for hy_v3 support"
            )
        nn.Module.__init__(self)
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        self.gate = HYV3TopKRouter(config)
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(config.num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        self.experts = nn.ModuleList(
            [
                HYV3MLP(
                    config,
                    intermediate_size=config.moe_intermediate_size,
                )
                for _ in range(config.num_experts)
            ]
        )
        self.shared_experts = HYV3MLP(
            config,
            intermediate_size=config.moe_intermediate_size
            * config.num_shared_experts,
        )

    def _route(self, hidden_flat: torch.Tensor):
        if self.e_score_correction_bias.device != hidden_flat.device:
            self.e_score_correction_bias.data = (
                self.e_score_correction_bias.data.to(hidden_flat.device)
            )
        # HYV3TopKRouter applies sigmoid, bias-for-selection, route_norm,
        # and router_scaling_factor internally and returns
        # (router_logits, topk_weights, topk_indices).
        _, topk_weights, topk_indices = self.gate(
            hidden_flat, self.e_score_correction_bias
        )
        return topk_indices, topk_weights
```

(`forward` and `_local_experts` are inherited from `SyncGlmMoeDsaMoEBlock`, which already does the router-mask scatter + `expert_executor.dispatch_local` + `shared_experts` combine. Hy3's `enable_moe_fp32_combine` is `false` in the real config, matching the parent's plain-sum combine.)

In `moe_store/wrappers/__init__.py`, add alongside the glm imports (L10-11) and `__all__` (L42-43):

```python
from .hy_v3 import SyncHYV3MoEBlock
```
```python
    "SyncHYV3MoEBlock",
```

- [ ] **Step 4: Re-run**

Run: same pytest command, plus `tests/test_wrappers_import.py`.
Expected: PASS (3 + import tests)

- [ ] **Step 5: Commit**

```bash
git add moe_store/wrappers/hy_v3.py moe_store/wrappers/__init__.py tests/test_hyv3_wrappers.py
git commit -m "feat(hy3): SyncHYV3MoEBlock per-expert wrapper (EfficientMoE/MoE-Infinity#283)"
```

### Task 8: Boundary install/restore (MoE-Infinity)

**Repo:** MoE-Infinity worktree
**Files:**
- Modify: `moe_infinity/boundary/transformers_model_overrides.py`

- [ ] **Step 1: Import.** Add `SyncHYV3MoEBlock` to the `from moe_store.wrappers import (...)` list (L15-32, alphabetical position after `SyncGptOssMLP`).

- [ ] **Step 2: Install.** In `install_transformers_model_overrides` after the `glm5_next` block (L115-120), add:

```python
    try:
        hy_v3 = _model_module("hy_v3")
        hy_v3._old_hy_v3_moe = hy_v3.HYV3MoE
        hy_v3.HYV3MoE = SyncHYV3MoEBlock
    except (ImportError, AttributeError):
        pass
```

- [ ] **Step 3: Restore.** Add the inverse to `restore_context_transformers_model_overrides` (after L162) **and** `restore_runtime_transformers_model_overrides` (after L219):

```python
    try:
        hy_v3 = _model_module("hy_v3")
        if hasattr(hy_v3, "_old_hy_v3_moe"):
            hy_v3.HYV3MoE = hy_v3._old_hy_v3_moe
    except (ImportError, AttributeError):
        pass
```

- [ ] **Step 4: Sanity-run install/restore round trip**

Run: `.venv/bin/python -c "
from moe_infinity.boundary.transformers_model_overrides import (
    install_transformers_model_overrides,
    restore_runtime_transformers_model_overrides,
)
import transformers.models.hy_v3.modeling_hy_v3 as m
orig = m.HYV3MoE
install_transformers_model_overrides()
assert m.HYV3MoE is not orig, 'override not installed'
restore_runtime_transformers_model_overrides()
assert m.HYV3MoE is orig, 'override not restored'
print('OK')
"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add moe_infinity/boundary/transformers_model_overrides.py
git commit -m "feat(hy3): install SyncHYV3MoEBlock override for hy_v3 (#283)"
```

## Phase 3 — end-to-end smoke (GPU + real checkpoint gated)

### Task 9: conftest + greedy-decode smoke

**Repo:** MoE-Infinity worktree
**Files:**
- Create: `tests/python/hy3/__init__.py` (empty)
- Create: `tests/python/hy3/conftest.py`
- Create: `tests/python/hy3/e2e_offload.py`

- [ ] **Step 1: conftest** (pattern of `tests/python/v4/conftest.py`):

```python
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import os

import pytest

DEFAULT_CKPT = (
    "/mnt/raid0nvme0/public/huggingface/hub/"
    "models--tencent--Hy3-FP8/snapshots"
)


def _resolve_ckpt() -> str:
    ckpt = os.environ.get("HY3_FP8_CKPT", "")
    if ckpt:
        return ckpt
    if os.path.isdir(DEFAULT_CKPT):
        snaps = sorted(os.listdir(DEFAULT_CKPT))
        if snaps:
            return os.path.join(DEFAULT_CKPT, snaps[-1])
    return DEFAULT_CKPT


@pytest.fixture(scope="session")
def hy3_ckpt_dir():
    ckpt = _resolve_ckpt()
    if not os.path.exists(
        os.path.join(ckpt, "model.safetensors.index.json")
    ):
        pytest.skip(f"Hy3-FP8 checkpoint not found at {ckpt}")
    return ckpt
```

- [ ] **Step 2: smoke test** — `tests/python/hy3/e2e_offload.py`:

```python
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""GPU-gated end-to-end smoke for Hy3-FP8 offloaded serving.

Run manually (needs a GPU with >= 40 GB and the real checkpoint):
    HY3_FP8_CKPT=/path/to/Hy3-FP8 .venv/bin/python -m pytest \
        tests/python/hy3/e2e_offload.py -q -s
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def test_offloaded_greedy_decode(hy3_ckpt_dir, tmp_path):
    from transformers import AutoTokenizer

    from moe_infinity import MoE

    config = {
        "offload_path": os.path.join(str(tmp_path), "hy3-offload"),
        "device_memory_ratio": 0.75,
    }
    model = MoE(hy3_ckpt_dir, config)
    tokenizer = AutoTokenizer.from_pretrained(hy3_ckpt_dir)

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    output_ids = model.generate(
        inputs.input_ids,
        max_new_tokens=16,
        do_sample=False,
    )
    text = tokenizer.decode(
        output_ids[0][inputs.input_ids.shape[-1] :],
        skip_special_tokens=True,
    )
    print(f"decoded: {text!r}")
    # non-degenerate: some tokens, not a single repeated token
    new_tokens = output_ids[0][inputs.input_ids.shape[-1] :]
    assert new_tokens.numel() >= 8
    assert len(set(new_tokens.tolist())) > 1
    assert "Paris" in text
```

- [ ] **Step 3: Verify the suite is collected and skips cleanly without hardware**

Run: `.venv/bin/python -m pytest tests/python/hy3 -q`
Expected: `1 skipped` (no checkpoint / no CUDA on this machine) — zero errors.

- [ ] **Step 4: Run on a GPU box with the real checkpoint** (first-run store build streams ~299 GB; allow hours). Record the exact command, GPU model, and decoded output — this is the evidence for the docs row. If the decode is degenerate, debug before touching docs: the most likely culprits are (a) scales never delivered (`set_scales` warning in logs — check `deliver_fp8_scales_to_dispatcher`), (b) router bias all-zeros (check `_load_resident_shared_experts` found `mlp.e_score_correction_bias`), (c) `first_k_dense_replace` gate missed (layer-0 treated as MoE).

- [ ] **Step 5: Commit**

```bash
git add tests/python/hy3/
git commit -m "test(hy3): real-checkpoint greedy-decode smoke, GPU-gated (#283)"
```

## Phase 4 — docs & evidence

### Task 10: Compatibility rows, CHANGELOG, README

**Repo:** MoE-Infinity worktree
**Files:**
- Modify: `docs/model-compatibility.md`, `CHANGELOG.md`, `README.md`

- [ ] **Step 1:** Add to `docs/model-compatibility.md` general support table (match surrounding row format):
  - `Hy3-FP8 (hy_v3)` — protected path `protected:hy3_fp8_pertensor` (per-tensor FP8 → uniform block-grid → `glm_fp8_block128` dispatcher path); evidence level `implemented/experimental` until Task 9 Step 4 has run on hardware, then `validated` with the exact command and GPU.
  - `Hy3 (BF16)` — `not recorded` (expected to work via the same wrapper minus FP8 handling; untested).
  - Note the MTP layer-80 exclusion and the HF-eager serving path (`use_native_engine=False`).
- [ ] **Step 2:** `CHANGELOG.md` Unreleased → Added: `Hy3-FP8 (hy_v3) support: per-tensor static FP8 expert offloading via scalar→block-grid scale expansion over the GLM FP8-in-store path; whole non-expert namespace resident (#283).`
- [ ] **Step 3:** `README.md` support-matrix row for Hy3/Hy3-FP8 — only with the evidence level actually earned.
- [ ] **Step 4:** Commit:

```bash
git add docs/model-compatibility.md CHANGELOG.md README.md
git commit -m "docs(hy3): compatibility rows + changelog for Hy3-FP8 (#283)"
```

- [ ] **Step 5:** Push the branch, flip PR #284 checklist items, and (after review) mark the PR ready. Follow-ups to file as separate issues if wanted: PagedAttention shim for hy_v3 (native engine), BF16 `tencent/Hy3` validation row, scalar-scale fast path benchmarks.

---

## Self-review notes

- Spec coverage: issue #283 phases 0–4 all map to Tasks 1–10; the issue's "optional CUDA fast path" is explicitly deferred (scale-grid expansion makes it unnecessary for correctness).
- Type consistency: `_split_fp8_pertensor_state_dict(state_dict, scales_out)` signature matches both the Task 4 tests and the Task 5 call site; `SyncHYV3MoEBlock._route` return order `(indices, weights)` matches the inherited `SyncGlmMoeDsaMoEBlock.forward` consumption.
- Known uncertainty, checked in Task 9 Step 4: whether any non-expert buffer beyond `e_score_correction_bias` needs explicit materialization (the L1516-1518 resident-GPU loop moves zeros-initialized buffers as-is; rotary `inv_freq` is computed, not loaded, so this should be complete).
