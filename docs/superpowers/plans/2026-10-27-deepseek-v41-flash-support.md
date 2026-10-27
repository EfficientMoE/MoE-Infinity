# DeepSeek-V4.1-Flash (deepseek_v41) Full Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve `deepseek-ai/DeepSeek-V4.1-Flash` on memory-constrained GPUs by offloading its ~283 GB of FP4 routed experts (reusing the validated DeepSeek-V4-Flash FP4 path) and its ~197 GB engram n-gram tables (new capability), leaving a ~30 GB resident footprint.

**Architecture:** Mirror the two-path DeepSeek-V4-Flash strategy. Path B (official reference `inference/` code + offload adapter) is the primary route because no released `transformers` (≤ 5.21.0) ships `DeepseekV41ForCausalLM`; Path A (HF-native wrapper) stays guard-registered and activates automatically when transformers ships the class. Expert tensors are format-identical to V4-Flash (FP4 E2M1 packed 2/byte + UE8M0 group-32 scales), so `extensions/kernel/v4_fp4` and the V4 `ExpertBundle`/host-store machinery are reused; the deltas are the nested `text_config`, the `bias`/`bias_vl` noaux-tc router, the engram tables, and the new resident attention stack (KV compressor, indexer, o-LoRA, mHC, attn_sink) which stays untouched reference code on GPU.

**Tech Stack:** Python 3.12, torch ≥ 2.8, safetensors, moe-store (sibling repo `/mnt/raid0nvme0/leyang/moe-store`, editable install), MoE-Infinity native extensions (`_v4_fp4`), pytest.

**Tracking issue:** EfficientMoE/MoE-Infinity#281

---

## Verified facts (2026-10-27, via HF index + shard-header range requests)

Do not re-derive these; they were confirmed against the live checkpoint.

- `config.json`: `model_type: deepseek_v41`, `architectures: ["DeepseekV41ForCausalLM"]`, nested `text_config` (hidden 5120, `moe_intermediate_size` 2304, 40 layers, 384 routed experts, 6 per token, 1 shared, `scoring_func: sqrtsoftplus`, `topk_method: noaux_tc`, `routed_scaling_factor: 1.5`, `norm_topk_prob: true`, `swiglu_limit: 10.0`) and `vision_config`. Top-level `quantization_config: {quant_method: fp8, weight_block_size: [32, 32], scale_fmt: ue8m0, expert_dtype: fp4}`.
- Expert keys (96,085 tensors total): `layers.{0..39}.ffn.experts.{0..383}.{w1,w2,w3}.{weight,scale}`.
  - `w1/w3.weight` `I8 [2304, 2560]` (FP4 E2M1, 2 per byte along K=5120), `w1/w3.scale` `F8_E8M0 [2304, 160]` (group 32).
  - `w2.weight` `I8 [5120, 1152]` (K=2304), `w2.scale` `F8_E8M0 [5120, 72]`.
  - Identical in kind to V4-Flash → `v4_fp4_dequant.cu` and `moe_store/wrappers/deepseek_v4/fp4_expert.py` apply unchanged (shapes are parameterized).
- Router keys per layer: `layers.{L}.ffn.gate.weight [384, 5120] (BF16)`, `gate.bias`, `gate.bias_vl` (modality-dependent noaux-tc bias).
- Engram: `layers.{1,14}.engram.embed.weight F8_E4M3 [384006168|384016682, 256]` + `.scale F8_E8M0 [N, 8]` (group 32 along dim 256) ≈ 197 GB combined; plus small `engram.{q_weight,k_weight,wkv.*}`.
- dspark MTP: `mtp.{0..2}.*` including its own `mtp.{K}.ffn.experts.{0..127}.*` MoE — **excluded from generation support** (resident-skip, same policy as Qwen3.5 MTP).
- Vision tower: `vision.blocks.{0..31}.*` ≈ small; resident; text-only mode first.
- Shared expert: `layers.{L}.ffn.shared_experts.{w1,w2,w3}.{weight,scale}` (FP8, resident).
- The moe-store regex `layers\.(\d+)\.ffn\.experts\.(\d+)\.` matches the real layout (the "UNCONFIRMED" comment in `parsing/hf_config.py` is now confirmed and must be updated).
- Existing moe-store groundwork that already passes: `tests/test_deepseekv41_registry.py` (guarded registry entry, `parse_moe_param` nested text_config, `parse_expert_id` V4 layout) and fixture `tests/fixtures/deepseek_v41_flash/config.json`.

## File structure

**moe-store** (`/mnt/raid0nvme0/leyang/moe-store`):

| File | Responsibility |
| --- | --- |
| `moe_store/parsing/hf_config.py` (modify) | Replace UNCONFIRMED comment with verified layout note. |
| `moe_store/wrappers/deepseek_v41/__init__.py` (create) | Public surface of the V4.1 wrapper family. |
| `moe_store/wrappers/deepseek_v41/expert_bundle.py` (create) | `DeepSeekV41ExpertTensorIndexer` — nested `text_config` dims, reuses V4 `ExpertBundle`/`TensorRef`. |
| `moe_store/wrappers/deepseek_v41/routing.py` (create) | `topk_route_v41` — sqrtsoftplus + per-token `bias`/`bias_vl` selection. |
| `moe_store/wrappers/deepseek_v41/engram_offload.py` (create) | `EngramHostStore` — pinned-host FP8 table, row gather + UE8M0 dequant on GPU. |
| `moe_store/wrappers/deepseek_v41/official_offload_adapter.py` (create) | Path B: null-expert model build, expert store patch, engram patch, non-expert resident load. |
| `tests/test_deepseekv41_registry.py` (modify) | Add mtp/engram negative parse cases. |
| `tests/test_deepseekv41_wrappers.py` (create) | Indexer, routing, engram-store unit tests (synthetic tensors, CPU). |

**MoE-Infinity** (this repo):

| File | Responsibility |
| --- | --- |
| `moe_infinity/runtime/expert_precision.py` (modify) | `PROTECTED_MODEL_PATHS["deepseek_v41"]` → reuse `protected:deepseek_v4_fp4`. |
| `tests/python/unit/test_expert_precision_registry.py` (modify) | Protection + anti-mis-map regression tests. |
| `tests/python/v41/conftest.py` (create) | Real-checkpoint skip gate (pattern of `tests/python/v4/conftest.py`). |
| `tests/python/v41/e2e_offload.py` (create) | GPU-gated end-to-end smoke: load Path B offloaded model, greedy decode. |
| `docs/model-compatibility.md` (modify) | V4.1 rows with honest evidence levels. |
| `docs/deepseek-v41-flash.md` (create) | Family guide (container/mount/ckpt prep once validated). |
| `README.md`, `CHANGELOG.md` (modify) | Support matrix row + Unreleased entry — only in the phase whose evidence exists. |
| `benchmarks/contextpilot/v41flash_ab.py` (create, Phase 6) | A/B harness adapted from `v4flash_ab.py`. |

Engine/native code: **no changes required** — `core/python/py_archer_prefetch.cpp` already registers the `deepseek_v4_fp4` expert dtype and `extensions/kernel/v4_fp4/v4_fp4_dequant.cu` already implements the exact E2M1/UE8M0-32 format.

---

## Phase 0 — moe-store: confirm checkpoint layout in code

### Task 1: Replace the UNCONFIRMED caveat; add negative parse cases

**Repo:** moe-store
**Files:**
- Modify: `moe_store/parsing/hf_config.py` (the `deepseekv41` branch of `parse_expert_id`)
- Test: `tests/test_deepseekv41_registry.py`

- [ ] **Step 1: Add failing negative-parse test cases**

Append these parametrize cases to `test_parse_expert_id_uses_v4_layout` in `tests/test_deepseekv41_registry.py`:

```python
        # dspark MTP keeps its own 128-expert MoE; it must NOT be offloaded.
        ("mtp.0.ffn.experts.5.w1.weight", (None, None)),
        ("mtp.2.ffn.experts.127.w2.scale", (None, None)),
        # engram tables are not routed experts.
        ("model.layers.1.engram.embed.weight", (None, None)),
        # scale member of a real expert parses identically to its weight.
        ("model.layers.5.ffn.experts.0.w1.scale", (5, 0)),
```

- [ ] **Step 2: Run the test**

Run: `cd /mnt/raid0nvme0/leyang/moe-store && /mnt/raid0nvme0/leyang/MoE-Infinity/.venv/bin/python -m pytest tests/test_deepseekv41_registry.py -q`
Expected: PASS already (the regex requires a `layers.` prefix, so `mtp.*` cannot match) — these are pin-down regression cases. If any case FAILS, the regex drifted; stop and fix the regex, not the test.

- [ ] **Step 3: Update the comment in `parse_expert_id`**

In `moe_store/parsing/hf_config.py`, replace:

```python
        # The V4.1 checkpoint key layout is UNCONFIRMED. Reuse the V4 pattern
        # until it can be checked against real safetensors shards.
```

with:

```python
        # Confirmed against the live deepseek-ai/DeepSeek-V4.1-Flash index
        # (2026-10-27): routed experts use the V4-native layout
        # "layers.{L}.ffn.experts.{E}.{w1,w2,w3}.{weight,scale}";
        # "mtp.{K}.ffn.experts.*" (dspark) and "layers.{L}.engram.*" never
        # match because the pattern anchors on "layers.N.ffn.experts.".
```

- [ ] **Step 4: Re-run and commit**

Run: `/mnt/raid0nvme0/leyang/MoE-Infinity/.venv/bin/python -m pytest tests/test_deepseekv41_registry.py -q`
Expected: PASS (12 tests)

```bash
git add moe_store/parsing/hf_config.py tests/test_deepseekv41_registry.py
git commit -m "test(deepseek-v41): confirm expert key layout, pin mtp/engram exclusion"
```

---

## Phase 1 — moe-store: V4.1 expert indexing and routing

### Task 2: `DeepSeekV41ExpertTensorIndexer`

**Repo:** moe-store
**Files:**
- Create: `moe_store/wrappers/deepseek_v41/__init__.py`
- Create: `moe_store/wrappers/deepseek_v41/expert_bundle.py`
- Test: `tests/test_deepseekv41_wrappers.py`

The V4 indexer (`wrappers/deepseek_v4/expert_bundle.py`) reads `num_hidden_layers` etc. from the **top level** of `config.json`; V4.1 nests them under `text_config`. Everything else (key schema, `ExpertBundle`, `TensorRef`, shard-header scanning, FP4 metadata) is reused.

- [ ] **Step 1: Write the failing test with a synthetic 2-layer/4-expert checkpoint**

Create `tests/test_deepseekv41_wrappers.py`:

```python
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

HIDDEN = 64
INTER = 32
LAYERS = 2
EXPERTS = 4


@pytest.fixture()
def tiny_v41_ckpt(tmp_path: Path) -> Path:
    tensors = {}
    for layer in range(LAYERS):
        for expert in range(EXPERTS):
            prefix = f"layers.{layer}.ffn.experts.{expert}"
            # w1/w3: [inter, hidden/2] packed, scale [inter, hidden/32]
            tensors[f"{prefix}.w1.weight"] = torch.zeros(
                INTER, HIDDEN // 2, dtype=torch.int8
            )
            tensors[f"{prefix}.w1.scale"] = torch.full(
                (INTER, HIDDEN // 32), 127, dtype=torch.uint8
            ).view(torch.float8_e8m0fnu)
            tensors[f"{prefix}.w2.weight"] = torch.zeros(
                HIDDEN, INTER // 2, dtype=torch.int8
            )
            tensors[f"{prefix}.w2.scale"] = torch.full(
                (HIDDEN, INTER // 32), 127, dtype=torch.uint8
            ).view(torch.float8_e8m0fnu)
            tensors[f"{prefix}.w3.weight"] = torch.zeros(
                INTER, HIDDEN // 2, dtype=torch.int8
            )
            tensors[f"{prefix}.w3.scale"] = torch.full(
                (INTER, HIDDEN // 32), 127, dtype=torch.uint8
            ).view(torch.float8_e8m0fnu)
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, str(tmp_path / shard))
    index = {"weight_map": {k: shard for k in tensors}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    config = {
        "model_type": "deepseek_v41",
        "architectures": ["DeepseekV41ForCausalLM"],
        "text_config": {
            "model_type": "deepseek_v41_text",
            "num_hidden_layers": LAYERS,
            "n_routed_experts": EXPERTS,
            "hidden_size": HIDDEN,
            "moe_intermediate_size": INTER,
            "swiglu_limit": 10.0,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


def test_v41_indexer_reads_nested_text_config(tiny_v41_ckpt: Path) -> None:
    from moe_store.wrappers.deepseek_v41 import (
        DeepSeekV41ExpertTensorIndexer,
    )

    indexer = DeepSeekV41ExpertTensorIndexer(str(tiny_v41_ckpt))
    assert indexer.num_hidden_layers == LAYERS
    assert indexer.n_routed_experts == EXPERTS
    assert indexer.hidden_size == HIDDEN
    assert indexer.intermediate_size == INTER
    assert indexer.swiglu_limit == 10.0


def test_v41_bundle_has_six_tensors_with_fp4_metadata(
    tiny_v41_ckpt: Path,
) -> None:
    from moe_store.wrappers.deepseek_v41 import (
        DeepSeekV41ExpertTensorIndexer,
    )

    indexer = DeepSeekV41ExpertTensorIndexer(str(tiny_v41_ckpt))
    bundle = indexer.bundle(1, 3)
    assert bundle.num_tensors == 6
    assert bundle.quant_format == "fp4_e2m1"
    assert bundle.scale_format == "ue8m0"
    assert bundle.block_size == 32
    w1 = bundle.part("w1", "weight")
    assert w1.key == "layers.1.ffn.experts.3.w1.weight"
    assert w1.shape == (INTER, HIDDEN // 2)
    loaded = indexer.load_bundle_tensors(bundle)
    assert len(loaded) == 6
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /mnt/raid0nvme0/leyang/moe-store && /mnt/raid0nvme0/leyang/MoE-Infinity/.venv/bin/python -m pytest tests/test_deepseekv41_wrappers.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'moe_store.wrappers.deepseek_v41'`

- [ ] **Step 3: Implement the indexer**

Create `moe_store/wrappers/deepseek_v41/expert_bundle.py`:

```python
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""DeepSeek-V4.1-Flash expert tensor indexing.

Routed experts use the exact V4-Flash on-disk format (FP4 E2M1 packed two
values per byte along K, UE8M0 scales with group size 32):

    layers.{L}.ffn.experts.{E}.w1.weight  int8     [inter, hidden/2]
    layers.{L}.ffn.experts.{E}.w1.scale   f8_e8m0  [inter, hidden/32]
    layers.{L}.ffn.experts.{E}.w2.weight  int8     [hidden, inter/2]
    layers.{L}.ffn.experts.{E}.w2.scale   f8_e8m0  [hidden, inter/32]
    layers.{L}.ffn.experts.{E}.w3.weight  int8     [inter, hidden/2]
    layers.{L}.ffn.experts.{E}.w3.scale   f8_e8m0  [inter, hidden/32]

Verified shapes for deepseek-ai/DeepSeek-V4.1-Flash (hidden=5120, inter=2304):
    w1/w3.weight int8 [2304, 2560]   scale f8_e8m0 [2304, 160]
    w2.weight    int8 [5120, 1152]   scale f8_e8m0 [5120, 72]

The only difference from V4-Flash is the config: model dims live under the
nested ``text_config`` object. ``mtp.{K}.ffn.experts.*`` (dspark MTP) and
``layers.{L}.engram.*`` are NOT routed experts and are never enumerated here.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

from moe_store.wrappers.deepseek_v4.expert_bundle import (
    DeepSeekV4ExpertTensorIndexer,
)

__all__ = ["DeepSeekV41ExpertTensorIndexer"]


class DeepSeekV41ExpertTensorIndexer(DeepSeekV4ExpertTensorIndexer):
    """V4 indexer with V4.1's nested ``text_config`` dimension lookup."""

    @staticmethod
    def _load_config(ckpt_dir: str, config: Optional[object]) -> Dict:
        if config is not None:
            get = (
                config.get
                if isinstance(config, dict)
                else lambda k, d=None: getattr(config, k, d)
            )
            text = get("text_config") or config
            tget = (
                text.get
                if isinstance(text, dict)
                else lambda k, d=None: getattr(text, k, d)
            )
            cfg = {
                "num_hidden_layers": tget("num_hidden_layers"),
                "n_routed_experts": tget("n_routed_experts"),
                "hidden_size": tget("hidden_size"),
                "moe_intermediate_size": tget("moe_intermediate_size"),
                "swiglu_limit": tget("swiglu_limit", 0.0),
            }
            if all(
                cfg[k] is not None
                for k in (
                    "num_hidden_layers",
                    "n_routed_experts",
                    "hidden_size",
                    "moe_intermediate_size",
                )
            ):
                return cfg
        cfg_path = os.path.join(ckpt_dir, "config.json")
        with open(cfg_path) as f:
            raw = json.load(f)
        text = raw.get("text_config", raw)
        return {
            "num_hidden_layers": text["num_hidden_layers"],
            "n_routed_experts": text["n_routed_experts"],
            "hidden_size": text["hidden_size"],
            "moe_intermediate_size": text["moe_intermediate_size"],
            "swiglu_limit": text.get("swiglu_limit", 0.0),
        }
```

Create `moe_store/wrappers/deepseek_v41/__init__.py`:

```python
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from .expert_bundle import DeepSeekV41ExpertTensorIndexer

__all__ = ["DeepSeekV41ExpertTensorIndexer"]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `/mnt/raid0nvme0/leyang/MoE-Infinity/.venv/bin/python -m pytest tests/test_deepseekv41_wrappers.py -q`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the wrapper-import sweep and commit**

Run: `/mnt/raid0nvme0/leyang/MoE-Infinity/.venv/bin/python -m pytest tests/test_wrappers_import.py tests/test_deepseekv41_wrappers.py -q`
Expected: PASS

```bash
git add moe_store/wrappers/deepseek_v41 tests/test_deepseekv41_wrappers.py
git commit -m "feat(deepseek-v41): expert tensor indexer with nested text_config"
```

### Task 3: `topk_route_v41` with `bias`/`bias_vl`

**Repo:** moe-store
**Files:**
- Create: `moe_store/wrappers/deepseek_v41/routing.py`
- Modify: `moe_store/wrappers/deepseek_v41/__init__.py`
- Test: `tests/test_deepseekv41_wrappers.py`

V4.1 keeps the V4 scoring (`sqrtsoftplus`, noaux-tc selection bias, `norm_topk_prob`, `routed_scaling_factor`) but ships two selection biases: `gate.bias` (text tokens) and `gate.bias_vl` (vision tokens). The per-token bias is chosen by a boolean vision mask. **Note:** the exact reference semantics must be reconciled against the vendored `inference/model.py` in Task 6 Step 2; this implementation encodes the config-documented behavior and the parity test in Task 7 is the arbiter.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_deepseekv41_wrappers.py`:

```python
def test_topk_route_v41_matches_v4_for_all_text_tokens() -> None:
    from moe_store.wrappers.deepseek_v4 import topk_route
    from moe_store.wrappers.deepseek_v41 import topk_route_v41

    torch.manual_seed(0)
    hidden = torch.randn(5, HIDDEN)
    gate_w = torch.randn(EXPERTS * 4, HIDDEN)
    bias = torch.randn(EXPERTS * 4)
    bias_vl = torch.randn(EXPERTS * 4)
    mask = torch.zeros(5, dtype=torch.bool)  # all text

    w_ref, i_ref = topk_route(
        hidden, gate_w, bias, top_k=3,
        routed_scaling_factor=1.5, norm_topk_prob=True,
    )
    w, i = topk_route_v41(
        hidden, gate_w, bias, bias_vl, mask, top_k=3,
        routed_scaling_factor=1.5, norm_topk_prob=True,
    )
    assert torch.equal(i.sort(dim=-1).values, i_ref.sort(dim=-1).values)
    torch.testing.assert_close(
        w.gather(1, i.argsort(dim=-1)), w_ref.gather(1, i_ref.argsort(dim=-1))
    )


def test_topk_route_v41_vision_tokens_use_bias_vl() -> None:
    from moe_store.wrappers.deepseek_v41 import topk_route_v41

    torch.manual_seed(1)
    n_experts = 8
    hidden = torch.randn(4, HIDDEN)
    gate_w = torch.randn(n_experts, HIDDEN)
    # bias strongly prefers experts {0,1}; bias_vl strongly prefers {6,7}
    bias = torch.full((n_experts,), -100.0)
    bias[:2] = 100.0
    bias_vl = torch.full((n_experts,), -100.0)
    bias_vl[6:] = 100.0
    mask = torch.tensor([False, True, False, True])

    _, indices = topk_route_v41(
        hidden, gate_w, bias, bias_vl, mask, top_k=2,
        routed_scaling_factor=1.0, norm_topk_prob=True,
    )
    assert set(indices[0].tolist()) == {0, 1}
    assert set(indices[1].tolist()) == {6, 7}
    assert set(indices[2].tolist()) == {0, 1}
    assert set(indices[3].tolist()) == {6, 7}


def test_topk_route_v41_weights_exclude_selection_bias() -> None:
    from moe_store.wrappers.deepseek_v41 import topk_route_v41

    torch.manual_seed(2)
    hidden = torch.randn(3, HIDDEN)
    gate_w = torch.randn(8, HIDDEN)
    bias = torch.randn(8) * 100  # bias shifts selection only
    mask = torch.zeros(3, dtype=torch.bool)
    weights, _ = topk_route_v41(
        hidden, gate_w, bias, bias, mask, top_k=2,
        routed_scaling_factor=1.0, norm_topk_prob=True,
    )
    # normalized sqrtsoftplus scores are positive and sum to 1 per token
    torch.testing.assert_close(
        weights.sum(dim=-1), torch.ones(3), atol=1e-5, rtol=1e-5
    )
    assert (weights > 0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/mnt/raid0nvme0/leyang/MoE-Infinity/.venv/bin/python -m pytest tests/test_deepseekv41_wrappers.py -q`
Expected: FAIL with `ImportError: cannot import name 'topk_route_v41'`

- [ ] **Step 3: Implement routing**

Create `moe_store/wrappers/deepseek_v41/routing.py`:

```python
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from moe_store.wrappers.deepseek_v4.routing import sqrtsoftplus

__all__ = ["topk_route_v41"]


def topk_route_v41(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    bias_vl: torch.Tensor,
    vision_token_mask: torch.Tensor,
    top_k: int,
    routed_scaling_factor: float = 1.0,
    norm_topk_prob: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """V4 noaux-tc routing with V4.1's modality-dependent selection bias.

    ``vision_token_mask`` is a bool tensor of shape [num_tokens]; True rows
    select with ``bias_vl``, False rows with ``bias``. As in V4, the bias
    affects expert *selection* only; returned weights come from the
    sqrtsoftplus scores of the selected experts.
    """
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    logits = F.linear(flat.float(), gate_weight.float())
    scores = sqrtsoftplus(logits)
    per_token_bias = torch.where(
        vision_token_mask.reshape(-1, 1),
        bias_vl.float().unsqueeze(0),
        bias.float().unsqueeze(0),
    )
    select = scores + per_token_bias
    indices = torch.topk(select, top_k, dim=-1, sorted=False).indices
    weights = scores.gather(1, indices)
    if norm_topk_prob:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return weights * routed_scaling_factor, indices
```

Update `moe_store/wrappers/deepseek_v41/__init__.py`:

```python
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from .expert_bundle import DeepSeekV41ExpertTensorIndexer
from .routing import topk_route_v41

__all__ = ["DeepSeekV41ExpertTensorIndexer", "topk_route_v41"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/mnt/raid0nvme0/leyang/MoE-Infinity/.venv/bin/python -m pytest tests/test_deepseekv41_wrappers.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add moe_store/wrappers/deepseek_v41 tests/test_deepseekv41_wrappers.py
git commit -m "feat(deepseek-v41): sqrtsoftplus routing with bias/bias_vl selection"
```

---

## Phase 2 — MoE-Infinity: precision protection

### Task 4: Protect `deepseek_v41` experts; fix the FP8 mis-map

**Repo:** MoE-Infinity
**Files:**
- Modify: `moe_infinity/runtime/expert_precision.py:96-101` (`PROTECTED_MODEL_PATHS`)
- Test: `tests/python/unit/test_expert_precision_registry.py`

Today `model_type: deepseek_v41` is not in `PROTECTED_MODEL_PATHS`, so `resolve_model_precision_capabilities` falls through to the `quant_method == "fp8"` branch and returns the GLM 128×128-block FP8 identity — wrong for V4.1 (32-group UE8M0 + FP4 experts). The expert format is byte-identical to V4-Flash, so V4.1 reuses the existing `protected:deepseek_v4_fp4` reason (and therefore the existing `_v4_fp4` extension and `deepseek_v4_fp4` engine dtype; no native changes).

- [ ] **Step 1: Write the failing tests**

In `tests/python/unit/test_expert_precision_registry.py`, add `("deepseek_v41", "protected:deepseek_v4_fp4")` to the existing protected-paths parametrize list (next to the `("deepseek_v4", "protected:deepseek_v4_fp4")` entry at line 57), and add this regression test at module level:

```python
def test_deepseek_v41_never_resolves_to_glm_fp8_identity():
    """model_type must win before the generic quant_method fp8 fallback.

    The V4.1 checkpoint carries quantization_config.quant_method == "fp8"
    (32x32 blocks, ue8m0 scales, fp4 experts); resolving it through the
    GLM_FP8_BLOCK128 path would be a silent format mismatch.
    """
    config = SimpleNamespace(
        model_type="deepseek_v41",
        quantization_config={
            "quant_method": "fp8",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        },
    )
    caps = resolve_model_precision_capabilities(
        config, extension_names={"_v4_fp4"}
    )
    assert caps.protected_reason == "protected:deepseek_v4_fp4"
    assert ExpertFormat.DEEPSEEK_V4_FP4 in caps.formats
    assert ExpertFormat.GLM_FP8_BLOCK128 not in caps.formats
    assert all(cap.protected for cap in caps.formats.values())
```

(Reuse the module's existing imports; add `from types import SimpleNamespace` if not present.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/python/unit/test_expert_precision_registry.py -q`
Expected: FAIL — the new parametrize case and the regression test report `protected_reason == "protected:fp8"` instead of `protected:deepseek_v4_fp4`.

- [ ] **Step 3: Add the protection entry**

In `moe_infinity/runtime/expert_precision.py`, change `PROTECTED_MODEL_PATHS` to:

```python
PROTECTED_MODEL_PATHS = {
    "gpt_oss": "protected:gpt_oss_mxfp4",
    "glm_moe_dsa": "protected:glm_fp8",
    "deepseek_v3": "protected:existing_fp8",
    "deepseek_v4": "protected:deepseek_v4_fp4",
    # V4.1-Flash routed experts are byte-identical in format to V4-Flash
    # (FP4 E2M1 packed + ue8m0 group-32 scales); non-expert FP8 uses 32x32
    # ue8m0 blocks, so the generic protected:fp8 (GLM 128x128 e4m3) path
    # must never catch this model_type.
    "deepseek_v41": "protected:deepseek_v4_fp4",
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/python/unit/test_expert_precision_registry.py tests/python/unit/test_adaptive_precision_wiring.py tests/python/unit/test_adaptive_precision_docs.py -q`
Expected: PASS (adaptive wiring/docs tests guard the protected-set documentation; if `test_adaptive_precision_docs.py` asserts an exhaustive protected list, update `docs/model-compatibility.md` line 3 in the same commit: append `DeepSeek-V4.1-Flash FP4` to the protected enumeration).

- [ ] **Step 5: Commit**

```bash
git add moe_infinity/runtime/expert_precision.py tests/python/unit/test_expert_precision_registry.py docs/model-compatibility.md
git commit -m "feat(deepseek-v41): protect fp4 experts from adaptive conversion and fp8 mis-map"
```

---

## Phase 3 — moe-store: Path B official offload loader (GPU + real-checkpoint gated)

### Task 5: Vendor and pin the reference inference code

**Repo:** moe-store
**Files:**
- Create: `moe_store/wrappers/deepseek_v41/OFFICIAL_PIN.md`

The HF repo ships `inference/{model.py,kernel.py,engram.py,vision.py,generate.py,convert.py,image_processor.py}`. Like V4-Flash (which runs inside the validated official container with `/workspace/official` mounted), the reference code is **not** vendored into moe-store; it is pinned by revision and mounted. This is remote code: review the pinned revision before first execution.

- [ ] **Step 1: Pin the revision**

Run: `curl -s "https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4.1-Flash" | python3 -c "import json,sys; print(json.load(sys.stdin)['sha'])"`
Record the printed sha. Then create `moe_store/wrappers/deepseek_v41/OFFICIAL_PIN.md`:

```markdown
# DeepSeek-V4.1-Flash official reference pin

- Repo: `deepseek-ai/DeepSeek-V4.1-Flash`
- Pinned revision: `<sha from the command above>`
- Reviewed files: `inference/model.py`, `inference/kernel.py`,
  `inference/engram.py`, `inference/vision.py`, `inference/generate.py`
- Download:
  `huggingface-cli download deepseek-ai/DeepSeek-V4.1-Flash --revision <sha> --include "inference/*" "encoding/*" "config.json" "chat_template.jinja" --local-dir /workspace/official-v41`
- Security: reference code executes locally; only this reviewed, pinned
  revision may be used. Bumping the pin requires re-review.
```

- [ ] **Step 2: Download and review the pinned inference code**

Run the `huggingface-cli download` command from the pin file. Read `inference/model.py` end to end and record in `OFFICIAL_PIN.md`, under a `## Verified reference API` heading, the following facts (needed by Task 6):
- the model class name and constructor signature (V4 used `M.Transformer(margs)` with `M.ModelArgs(**config)`),
- the MoE module attribute names (V4: `block.ffn`, `moe.gate`, `moe.shared_experts`, `moe.experts`, `moe.experts_start_idx/end_idx`, `moe.dim`, `moe.n_routed_experts`),
- the gate call signature (V4: `gate(x, input_ids)`; V4.1 likely `gate(x, vision_mask)` or similar for `bias_vl`),
- the expert class name and its `swiglu_limit` handling,
- how `engram` modules consume `embed.weight`/`embed.scale` (row lookup entry point for Task 8),
- the linear/dequant helper used for FP4+FP8 weights (`inference/kernel.py`).

- [ ] **Step 3: Commit**

```bash
git add moe_store/wrappers/deepseek_v41/OFFICIAL_PIN.md
git commit -m "docs(deepseek-v41): pin official reference inference revision"
```

### Task 6: V4.1 offload adapter

**Repo:** moe-store
**Files:**
- Create: `moe_store/wrappers/deepseek_v41/official_offload_adapter.py`
- Modify: `moe_store/wrappers/deepseek_v41/__init__.py`
- Test: `tests/test_deepseekv41_wrappers.py` (store reuse unit test, CPU)

`OfficialExpertHostStore` from `wrappers/deepseek_v4/official_offload_adapter.py` already encodes the exact V4.1 key layout (`layers.{L}.ffn.experts.{E}.{proj}.{part}`), LRU GPU residency, copy-stream prefetch, and pinned-host caching — it is reused as-is. The new module contributes the V4.1 model construction (null routed experts) and forward patch.

- [ ] **Step 1: Write the failing CPU unit test for store reuse**

Append to `tests/test_deepseekv41_wrappers.py`:

```python
def test_v41_adapter_reuses_v4_host_store(tiny_v41_ckpt: Path) -> None:
    from moe_store.wrappers.deepseek_v4.official_offload_adapter import (
        OfficialExpertHostStore,
    )
    from moe_store.wrappers.deepseek_v41.official_offload_adapter import (
        make_expert_store,
    )

    store = make_expert_store(
        str(tiny_v41_ckpt),
        device=torch.device("cpu"),
        max_resident_experts=2,
    )
    assert isinstance(store, OfficialExpertHostStore)
    bundle = store.get(1, 3)
    assert set(bundle) == {
        "w1.weight", "w1.scale",
        "w2.weight", "w2.scale",
        "w3.weight", "w3.scale",
    }
    assert store.is_resident(1, 3)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/mnt/raid0nvme0/leyang/MoE-Infinity/.venv/bin/python -m pytest tests/test_deepseekv41_wrappers.py::test_v41_adapter_reuses_v4_host_store -q`
Expected: FAIL with `ModuleNotFoundError` (no `official_offload_adapter` in `deepseek_v41`)

- [ ] **Step 3: Implement the adapter**

Create `moe_store/wrappers/deepseek_v41/official_offload_adapter.py`. The `make_expert_store` and `patch_moe_with_offload_v41` bodies below are complete; the `load_offloaded_v41_flash` body follows the V4 template and **must be reconciled with the `## Verified reference API` section of `OFFICIAL_PIN.md` from Task 5 Step 2** (attribute names marked `# PIN:` are the reconciliation points):

```python
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Path B: offloaded DeepSeek-V4.1-Flash on the official reference code.

Follows wrappers/deepseek_v4/official_offload_adapter.py. The expert host
store is reused unchanged because V4.1 keeps the V4 expert key layout and
FP4/UE8M0 tensor format. dspark MTP modules (``mtp.*``) and engram tables
are handled separately: MTP stays unloaded (generation does not use it),
engram offload is wired by EngramHostStore (engram_offload.py).
"""

from __future__ import annotations

import json

import torch

from moe_store.wrappers.deepseek_v4.official_offload_adapter import (
    OfficialExpertHostStore,
    resolve_use_native,
    _expert_forward,
    _expert_forward_native,
    _load_native_fp4,
)

__all__ = [
    "make_expert_store",
    "patch_moe_with_offload_v41",
    "load_offloaded_v41_flash",
]


def make_expert_store(
    ckpt_path: str,
    device: torch.device,
    max_resident_experts: int = 16,
    shard_file: str = None,
) -> OfficialExpertHostStore:
    """V4.1 uses the V4 store unchanged: same key prefix, same 6 members."""
    return OfficialExpertHostStore(
        ckpt_path,
        device,
        max_resident_experts=max_resident_experts,
        shard_file=shard_file,
    )


def patch_moe_with_offload_v41(
    model, store, official_module, use_native=None
):
    use_native = resolve_use_native(use_native, store.device)
    if use_native:
        _load_native_fp4()

    def make_forward(moe, layer_id):
        gate = moe.gate                      # PIN: gate module name
        shared = moe.shared_experts          # PIN: shared expert attr
        dim = moe.dim                        # PIN: hidden dim attr
        n_routed = moe.n_routed_experts      # PIN: routed count attr
        swiglu_limit = getattr(shared, "swiglu_limit", 0.0)
        start_idx = moe.experts_start_idx    # PIN: EP slice attrs
        end_idx = moe.experts_end_idx

        def forward(x, *gate_args):
            # PIN: V4 passed (x, input_ids); V4.1's gate consumes the
            # vision-token mask for bias_vl. Forward whatever the official
            # MoE.forward signature provides after x, unchanged.
            shape = x.size()
            x = x.view(-1, dim)
            weights, indices = gate(x, *gate_args)
            y = torch.zeros_like(x, dtype=torch.float32)
            counts = torch.bincount(
                indices.flatten(), minlength=n_routed
            ).tolist()
            active = [e for e in range(start_idx, end_idx) if counts[e] > 0]
            store.prefetch(layer_id, active)
            for e in active:
                idx, top = torch.where(indices == e)
                bundle = store.get(layer_id, e)
                if use_native:
                    out = _expert_forward_native(
                        x[idx], bundle, swiglu_limit
                    )
                else:
                    out = _expert_forward(
                        x[idx], bundle, swiglu_limit, official_module
                    )
                y[idx] += out.float() * weights[idx, top, None].float()
            if official_module.world_size > 1:
                official_module.dist.all_reduce(y)
            y += shared(x)
            return y.type_as(x).view(shape)

        return forward

    expert_layer = 0
    for block in model.layers:               # PIN: decoder blocks attr
        if hasattr(block, "ffn") and hasattr(block.ffn, "gate"):
            moe = block.ffn                   # PIN: ffn attr name
            moe.forward = make_forward(moe, expert_layer)
            for e in range(len(moe.experts)):
                moe.experts[e] = None
            expert_layer += 1
    return model


def load_offloaded_v41_flash(
    official_module,
    ckpt_path: str,
    config_path: str,
    device: torch.device,
    max_resident_experts: int = 16,
    use_native: bool = None,
    text_only: bool = True,
):
    """Build the official model with null routed experts, then patch.

    Mirrors deepseek_v4.load_offloaded_v4_flash; every PIN-marked line must
    match the Verified reference API section of OFFICIAL_PIN.md.
    """
    M = official_module
    with open(config_path) as f:
        margs = M.ModelArgs(**json.load(f))   # PIN: args class
    margs.max_batch_size = 1

    orig_moe_init = M.MoE.__init__           # PIN: MoE class name

    def _null_routed_moe_init(self, layer_id, args):
        import torch.nn as nn

        orig_expert = M.Expert               # PIN: Expert class name

        class _NullExpert(nn.Module):
            def __init__(self, *a, **k):
                super().__init__()

            def forward(self, x, weights=None):
                return x

        M.Expert = _NullExpert
        try:
            orig_moe_init(self, layer_id, args)
        finally:
            M.Expert = orig_expert
        for i in range(len(self.experts)):
            self.experts[i] = None
        self.shared_experts = orig_expert(    # PIN: shared ctor signature
            args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit
        )

    M.MoE.__init__ = _null_routed_moe_init
    try:
        with torch.device(device):
            model = M.Transformer(margs)     # PIN: model class name
    finally:
        M.MoE.__init__ = orig_moe_init
    torch.set_grad_enabled(False)

    store = make_expert_store(
        ckpt_path, device, max_resident_experts=max_resident_experts
    )
    patch_moe_with_offload_v41(model, store, M, use_native=use_native)
    torch.cuda.empty_cache()
    _load_non_expert_weights_v41(
        model, ckpt_path, device, text_only=text_only
    )
    return model, store


def _load_non_expert_weights_v41(model, ckpt_path, device, text_only=True):
    """Resident load of everything except routed experts, engram tables,
    mtp.* (unused for generation), and vision.* when text_only."""
    import os
    from glob import glob

    from safetensors import safe_open

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    shards = sorted(glob(os.path.join(ckpt_path, "*.safetensors")))
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as f:
            for k in f.keys():
                if ".ffn.experts." in k and k.startswith("layers."):
                    continue  # offloaded routed experts
                if ".engram.embed." in k:
                    continue  # offloaded by EngramHostStore
                if k.startswith("mtp."):
                    continue  # dspark MTP unused for generation
                if text_only and k.startswith(
                    ("vision.", "aligner.", "image_")
                ):
                    continue
                if k.endswith(".scale"):
                    wp = params.get(k[:-6] + ".weight")
                    if wp is not None and getattr(wp, "scale", None) is not None:
                        t = f.get_tensor(k).to(device)
                        if wp.scale.shape == t.shape:
                            wp.scale.copy_(t)
                    continue
                target = params.get(k) or buffers.get(k)
                if target is None:
                    continue
                t = f.get_tensor(k).to(device)
                if target.shape == t.shape:
                    target.copy_(t)
    return model
```

Update `moe_store/wrappers/deepseek_v41/__init__.py` to also export `make_expert_store`, `patch_moe_with_offload_v41`, `load_offloaded_v41_flash`.

- [ ] **Step 4: Run the CPU test to verify it passes**

Run: `/mnt/raid0nvme0/leyang/MoE-Infinity/.venv/bin/python -m pytest tests/test_deepseekv41_wrappers.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Reconcile every `# PIN:` line against `OFFICIAL_PIN.md` Verified reference API**

Open the vendored `inference/model.py` next to this adapter. For each `# PIN:` comment, confirm the attribute/class/signature or fix it. Record any deviation as an update to `OFFICIAL_PIN.md`. Do not proceed to Task 7 until all PIN lines are reconciled.

- [ ] **Step 6: Commit**

```bash
git add moe_store/wrappers/deepseek_v41 tests/test_deepseekv41_wrappers.py
git commit -m "feat(deepseek-v41): official-code offload adapter reusing V4 expert store"
```

### Task 7: GPU-gated end-to-end smoke test

**Repo:** MoE-Infinity
**Files:**
- Create: `tests/python/v41/__init__.py` (empty)
- Create: `tests/python/v41/conftest.py`
- Create: `tests/python/v41/e2e_offload.py`

- [ ] **Step 1: Write the checkpoint gate**

Create `tests/python/v41/conftest.py` (pattern of `tests/python/v4/conftest.py`):

```python
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import os
import sys

import pytest

CKPT_ENV = "MOE_V41_FLASH_CKPT"
OFFICIAL_ENV = "MOE_V41_OFFICIAL_DIR"


@pytest.fixture(scope="session")
def v41_ckpt_dir() -> str:
    ckpt = os.environ.get(CKPT_ENV, "")
    if not ckpt or not os.path.isdir(ckpt):
        pytest.skip(f"set {CKPT_ENV} to a local DeepSeek-V4.1-Flash checkpoint")
    return ckpt


@pytest.fixture(scope="session")
def v41_official_module(v41_ckpt_dir: str):
    official = os.environ.get(OFFICIAL_ENV, "")
    if not official or not os.path.isdir(official):
        pytest.skip(f"set {OFFICIAL_ENV} to the pinned official inference dir")
    sys.path.insert(0, official)
    import model as official_model  # the reference inference/model.py

    return official_model
```

- [ ] **Step 2: Write the smoke test**

Create `tests/python/v41/e2e_offload.py`:

```python
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""GPU-gated smoke: Path B offloaded V4.1-Flash greedy decode.

Run (host with >= 40 GB VRAM and the full checkpoint on local SSD):

    MOE_V41_FLASH_CKPT=/ckpt/DeepSeek-V4.1-Flash \
    MOE_V41_OFFICIAL_DIR=/workspace/official-v41/inference \
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
        tests/python/v41/e2e_offload.py -q -s
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def test_greedy_decode_produces_stable_tokens(
    v41_ckpt_dir, v41_official_module
):
    from moe_store.wrappers.deepseek_v41 import load_offloaded_v41_flash

    device = torch.device("cuda:0")
    model, store = load_offloaded_v41_flash(
        v41_official_module,
        ckpt_path=v41_ckpt_dir,
        config_path=os.path.join(v41_ckpt_dir, "config.json"),
        device=device,
        max_resident_experts=16,
        text_only=True,
    )
    # Tokenize a fixed prompt with the official encoding; greedy 16 tokens.
    # Determinism check: two runs produce identical token ids.
    prompt_ids = torch.tensor([[0, 100, 200, 300]], device=device)
    outs = []
    for _ in range(2):
        ids = prompt_ids.clone()
        for _ in range(16):
            logits = model(ids)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=-1)
        outs.append(ids)
    assert torch.equal(outs[0], outs[1])
    assert len(store.resident_experts()) <= 16
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    print(f"peak resident GiB: {peak:.1f}")
```

Note: the `model(ids)` call signature and the prompt encoding must follow the official `generate.py`; adjust this test during Task 6 Step 5 reconciliation if the reference forward needs position/vision-mask arguments. Record the final working invocation in `docs/deepseek-v41-flash.md` (Task 11).

- [ ] **Step 3: Verify collection-time behavior without a checkpoint**

Run: `.venv/bin/python -m pytest tests/python/v41 -q`
Expected: `2 skipped` (no checkpoint env vars) — the suite must never fail on machines without the 510 GB checkpoint.

- [ ] **Step 4: Run for real on the GPU host, record evidence**

Run the command from the module docstring. Expected: PASS, with printed peak resident GiB. Record checkpoint revision, GPU, peak memory, and tokens/s in `docs/deepseek-v41-flash.md` (created in Task 11; keep notes until then).

- [ ] **Step 5: Commit**

```bash
git add tests/python/v41
git commit -m "test(deepseek-v41): GPU-gated Path B e2e smoke with checkpoint skip gate"
```

---

## Phase 4 — Engram table host offload (new capability)

### Task 8: `EngramHostStore` with row-gather dequant

**Repo:** moe-store
**Files:**
- Create: `moe_store/wrappers/deepseek_v41/engram_offload.py`
- Modify: `moe_store/wrappers/deepseek_v41/__init__.py`
- Test: `tests/test_deepseekv41_wrappers.py`

The two engram tables (`layers.{1,14}.engram.embed`) are FP8 `[N≈384M, 256]` with UE8M0 scales `[N, 8]` (group 32 along the 256-dim) — ~98 GB each. Per decoded token the engram module looks up a handful of n-gram rows, so the access pattern is sparse row gather: keep the table pinned on host, gather requested rows, dequant to BF16 on device.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_deepseekv41_wrappers.py`:

```python
def test_engram_host_store_gathers_and_dequants_rows() -> None:
    from moe_store.wrappers.deepseek_v41 import EngramHostStore

    torch.manual_seed(3)
    rows, dim, group = 64, 256, 32
    reference = torch.randn(rows, dim)
    # Build an fp8/ue8m0 quantized table: per 32-wide group, scale = 2^ceil
    scales_f = reference.view(rows, dim // group, group).abs().amax(-1)
    exponents = torch.clamp(scales_f.log2().ceil(), -126, 127)
    scale_e8m0 = (exponents + 127).to(torch.uint8).view(
        torch.float8_e8m0fnu
    )
    scaled = reference.view(rows, dim // group, group) / exponents.exp2()[
        ..., None
    ]
    table_fp8 = scaled.view(rows, dim).to(torch.float8_e4m3fn)

    store = EngramHostStore(
        weight=table_fp8, scale=scale_e8m0, device=torch.device("cpu")
    )
    idx = torch.tensor([0, 63, 7, 7])
    out = store.gather(idx, dtype=torch.float32)
    assert out.shape == (4, dim)
    expected = (
        table_fp8[idx].to(torch.float32).view(4, dim // group, group)
        * scale_e8m0[idx].to(torch.float32)[..., None]
    ).view(4, dim)
    torch.testing.assert_close(out, expected)
    # quantization round-trip stays within fp8 e4m3 relative tolerance
    torch.testing.assert_close(
        out, reference[idx], atol=0.15, rtol=0.15
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/mnt/raid0nvme0/leyang/MoE-Infinity/.venv/bin/python -m pytest tests/test_deepseekv41_wrappers.py::test_engram_host_store_gathers_and_dequants_rows -q`
Expected: FAIL with `ImportError: cannot import name 'EngramHostStore'`

- [ ] **Step 3: Implement the store**

Create `moe_store/wrappers/deepseek_v41/engram_offload.py`:

```python
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Host-offloaded engram embedding tables for DeepSeek-V4.1-Flash.

Each table is fp8-e4m3 [N, 256] with ue8m0 scales [N, 8] (group 32 along
the embedding dim) and ~98 GB on disk — far too large to stay resident.
Engram lookups touch only a handful of rows per token, so the table lives
in (ideally pinned) host memory and ``gather`` moves just the requested
rows to the compute device and dequantizes there.
"""

from __future__ import annotations

import torch

__all__ = ["EngramHostStore"]


class EngramHostStore:
    def __init__(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        device: torch.device,
        pin: bool = True,
    ):
        if weight.dtype != torch.float8_e4m3fn:
            raise ValueError(f"expected fp8_e4m3 table, got {weight.dtype}")
        if scale.dtype != torch.float8_e8m0fnu:
            raise ValueError(f"expected ue8m0 scales, got {scale.dtype}")
        if weight.shape[0] != scale.shape[0]:
            raise ValueError("weight/scale row count mismatch")
        if weight.shape[1] % scale.shape[1] != 0:
            raise ValueError("embedding dim not divisible by scale groups")
        self.device = torch.device(device)
        self.group = weight.shape[1] // scale.shape[1]
        if pin and self.device.type == "cuda":
            weight = weight.pin_memory()
            scale = scale.pin_memory()
        self._weight = weight
        self._scale = scale

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        layer_id: int,
        device: torch.device,
        pin: bool = True,
    ) -> "EngramHostStore":
        import json
        import os
        import struct

        from safetensors import safe_open

        index_path = os.path.join(ckpt_path, "model.safetensors.index.json")
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        wkey = f"layers.{layer_id}.engram.embed.weight"
        skey = f"layers.{layer_id}.engram.embed.scale"
        tensors = {}
        for key in (wkey, skey):
            shard = os.path.join(ckpt_path, weight_map[key])
            with safe_open(shard, framework="pt", device="cpu") as f:
                tensors[key] = f.get_tensor(key)
        return cls(tensors[wkey], tensors[skey], device, pin=pin)

    def gather(
        self, indices: torch.Tensor, dtype: torch.dtype = torch.bfloat16
    ) -> torch.Tensor:
        rows = self._weight.index_select(0, indices.cpu())
        srows = self._scale.index_select(0, indices.cpu())
        rows = rows.to(self.device, non_blocking=True)
        srows = srows.to(self.device, non_blocking=True)
        n, dim = rows.shape
        out = rows.to(torch.float32).view(
            n, dim // self.group, self.group
        ) * srows.to(torch.float32)[..., None]
        return out.view(n, dim).to(dtype)
```

Export `EngramHostStore` from `moe_store/wrappers/deepseek_v41/__init__.py`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `/mnt/raid0nvme0/leyang/MoE-Infinity/.venv/bin/python -m pytest tests/test_deepseekv41_wrappers.py -q`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add moe_store/wrappers/deepseek_v41 tests/test_deepseekv41_wrappers.py
git commit -m "feat(deepseek-v41): host-offloaded engram tables with row-gather dequant"
```

### Task 9: Wire engram offload into the Path B loader

**Repo:** moe-store + MoE-Infinity (e2e rerun)
**Files:**
- Modify: `moe_store/wrappers/deepseek_v41/official_offload_adapter.py`

- [ ] **Step 1: Patch the engram modules after `_load_non_expert_weights_v41`**

Using the engram lookup entry point recorded in `OFFICIAL_PIN.md` (Task 5 Step 2), add to `load_offloaded_v41_flash`, after the resident weight load:

```python
    from .engram_offload import EngramHostStore

    engram_layer_ids = getattr(margs, "engram_layer_ids", (1, 14))
    for layer_id in engram_layer_ids:
        store_ng = EngramHostStore.from_checkpoint(
            ckpt_path, layer_id, device
        )
        engram = model.layers[layer_id].engram  # PIN: engram attr name
        # PIN: replace the reference embedding lookup with the host store;
        # the reference does `self.embed(row_ids)` (verify exact call) —
        # patch it to `store_ng.gather(row_ids, dtype=x.dtype)`.
        engram._moe_engram_store = store_ng
        engram.embed_lookup = store_ng.gather
```

Reconcile the two PIN lines against the vendored `inference/engram.py` before running.

- [ ] **Step 2: Rerun the GPU e2e smoke (Task 7 Step 4 command)**

Expected: PASS with peak resident GiB unchanged (±1 GiB) vs the text-only Task 7 run — engram rows must stream, not load resident. If peak jumps by ~tens of GiB, the reference module still holds a resident table; fix the patch.

- [ ] **Step 3: Commit**

```bash
git add moe_store/wrappers/deepseek_v41
git commit -m "feat(deepseek-v41): stream engram rows from host store in Path B loader"
```

---

## Phase 5 — Path A HF-native wrapper (gated on transformers)

### Task 10: Guarded sync MoE wrapper

**Repo:** moe-store
**Files:**
- Create: `moe_store/wrappers/deepseek_v41/sync_moe_block.py`
- Test: `tests/test_deepseekv41_wrappers.py`

Blocked until a transformers release ships `DeepseekV41ForCausalLM` (none ≤ 5.21.0 does; the registry guard in `moe_store/registry/constants.py:89-93` activates automatically). Keep this task last; do not fake the class.

- [ ] **Step 1: Write the guarded test**

Append to `tests/test_deepseekv41_wrappers.py`:

```python
def test_sync_v41_moe_block_requires_transformers_class() -> None:
    import moe_store.registry.constants as constants

    if constants.DeepseekV41ForCausalLM is None:
        pytest.skip(
            "transformers does not provide DeepseekV41ForCausalLM yet"
        )
    from moe_store.wrappers.deepseek_v41 import SyncDeepSeekV41MoEBlock

    assert SyncDeepSeekV41MoEBlock is not None
```

- [ ] **Step 2: Implement `SyncDeepSeekV41MoEBlock` following `wrappers/deepseek_v4/sync_moe_block.py`**

Model the implementation on `SyncDeepSeekV4MoEBlock` (same expert bundle provider and `fp4_expert_forward` execution, routing swapped to `topk_route_v41`). The HF module attribute names cannot be written down until the transformers implementation exists; when the class ships, copy `sync_moe_block.py` from `deepseek_v4`, replace the routing call with:

```python
        weights, indices = topk_route_v41(
            hidden_states,
            self.gate_weight,
            self.gate_bias,
            self.gate_bias_vl,
            self.vision_token_mask_for(hidden_states),
            top_k=self.top_k,
            routed_scaling_factor=self.routed_scaling_factor,
            norm_topk_prob=self.norm_topk_prob,
        )
```

and align the parameter names against `transformers/models/deepseek_v41/modeling_deepseek_v41.py`. Guard the import in `__init__.py` the same way `registry/constants.py` guards the class.

- [ ] **Step 3: Run guarded test (skips today), commit**

Run: `/mnt/raid0nvme0/leyang/MoE-Infinity/.venv/bin/python -m pytest tests/test_deepseekv41_wrappers.py -q`
Expected: PASS with 1 skip.

```bash
git add moe_store/wrappers/deepseek_v41 tests/test_deepseekv41_wrappers.py
git commit -m "feat(deepseek-v41): guarded HF-native sync MoE block (Path A)"
```

---

## Phase 6 — MoE-Infinity docs and evidence

### Task 11: Compatibility matrix, family guide, changelog

**Repo:** MoE-Infinity
**Files:**
- Modify: `docs/model-compatibility.md`
- Create: `docs/deepseek-v41-flash.md`
- Modify: `README.md` (support matrix + FP4 bullet)
- Modify: `CHANGELOG.md` (`Unreleased`)

Docs land phase-by-phase with honest evidence levels — never claim `validated` before the Task 7/9 GPU runs are recorded.

- [ ] **Step 1: Add the compatibility row (after Phase 2 lands)**

In `docs/model-compatibility.md` General model support table, after the DeepSeek-V4 row:

```markdown
| DeepSeek-V4.1 (`DeepseekV41ForCausalLM`) | implemented/experimental | not recorded | Path B official mp1 loader; FP4 experts reuse the V4 path; engram tables host-offloaded; MTP (dspark) unused; vision resident only in non-text-only mode; no transformers release ships the class yet (Path A guarded). |
```

Update line 3 (protected enumeration) to include `DeepSeek-V4.1-Flash FP4` (done in Task 4 Step 4 if the docs test required it; verify).

- [ ] **Step 2: Write the family guide**

Create `docs/deepseek-v41-flash.md` with: pinned official revision (from `OFFICIAL_PIN.md`), download commands, env vars (`MOE_V41_FLASH_CKPT`, `MOE_V41_OFFICIAL_DIR`), the e2e smoke invocation from `tests/python/v41/e2e_offload.py`, measured peak VRAM / tokens/s from Task 7 Step 4 and Task 9 Step 2, and the explicit non-claims (no MTP, no DFlash pairing, no multi-GPU TP claim, vision not validated).

- [ ] **Step 3: README + CHANGELOG (after Task 7 Step 4 evidence exists)**

README support matrix, after the DeepSeek-V4-Flash row:

```markdown
| DeepSeek-V4.1-Flash (FP4 expert + engram offloading) | `deepseek-ai/DeepSeek-V4.1-Flash` |
```

CHANGELOG `Unreleased`:

```markdown
- DeepSeek-V4.1-Flash (`deepseek_v41`) Path B support: FP4 routed-expert
  offloading reusing the V4-Flash store/kernels, host-offloaded engram
  tables, protected expert precision (no adaptive conversion), dspark MTP
  and vision excluded. Path A (HF-native) remains guarded on transformers
  shipping `DeepseekV41ForCausalLM`. (#281)
```

- [ ] **Step 4: Run docs-consistency tests and commit**

Run: `.venv/bin/python -m pytest tests/python/unit/test_adaptive_precision_docs.py -q`
Expected: PASS

```bash
git add docs/model-compatibility.md docs/deepseek-v41-flash.md README.md CHANGELOG.md
git commit -m "docs(deepseek-v41): compatibility rows, family guide, changelog (#281)"
```

### Task 12: A/B benchmark harness

**Repo:** MoE-Infinity
**Files:**
- Create: `benchmarks/contextpilot/v41flash_ab.py`

- [ ] **Step 1: Adapt the harness**

Copy `benchmarks/contextpilot/v4flash_ab.py` to `benchmarks/contextpilot/v41flash_ab.py` and apply exactly these substitutions:
- model loading: `load_offloaded_v4_flash(...)` → `load_offloaded_v41_flash(official_module, ckpt_path=..., config_path=..., device=..., max_resident_experts=args.max_resident_experts, text_only=True)` (single-process mp1; drop the `torchrun`/mp4 rank plumbing and the rank-0 broadcast block — V4.1 Path B is written single-GPU first),
- mounts/docstring: `/ckpt` → the `MOE_V41_FLASH_CKPT` checkpoint dir, `/workspace/official` → `MOE_V41_OFFICIAL_DIR`,
- output path: `results/v41flash_ab_<ts>.json`.
Keep the TTFT proxy method (max_new_tokens=1 prefill timing) and the 4 ContextPilot workload fixtures unchanged.

- [ ] **Step 2: Smoke-run on the GPU host, record results**

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/contextpilot/v41flash_ab.py --workloads all --repeats 3 --max-new-tokens 64 --max-resident-experts 16 --temperature 0 --out benchmarks/contextpilot/results/v41flash_ab_$(date +%Y%m%d_%H%M%S).json`
Expected: JSON written with per-workload ttft/e2e/decode_tok_s for both variants. Commit the results file and update `docs/deepseek-v41-flash.md` evidence table.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/contextpilot/v41flash_ab.py benchmarks/contextpilot/results docs/deepseek-v41-flash.md
git commit -m "bench(deepseek-v41): mp1 A/B harness + first offload evidence (#281)"
```

---

## Non-goals / explicit exclusions

- **dspark MTP** (`mtp.*`, incl. its 128-expert MoE): weights skipped at load; no speculative-decode claim.
- **DFlash pairing**: no published V4.1 drafter exists; no pairing claim (consistent with `docs/model-compatibility.md` DFlash matrix discipline).
- **Multi-GPU TP**: V4-Flash validated mp4; V4.1 Path B here is mp1-first. TP sharding is follow-up work.
- **Vision inputs**: `text_only=True` throughout this plan; vision residency is wired (`text_only=False` path) but not validated.
- **Serving engine (Stage 4a/4b) integration**: out of scope; V4.1 enters through the Path B adapter like V4, not through `api_server_v2` continuous serving.
- **SSD-tier engram store**: Phase 4 is pinned-host only; mmap/SSD streaming for hosts with < 256 GB RAM is follow-up.

## Risks

- **RISK-1 (reference API drift):** Task 6/9 PIN lines are templated from V4; the vendored V4.1 `inference/model.py` may rename classes/attributes. Mitigation: Task 5 Step 2 recording + mandatory reconciliation steps before any GPU run.
- **RISK-2 (host RAM):** full pinned experts (~283 GB) + engram (~197 GB) exceed most hosts. `OfficialExpertHostStore` caches host-side lazily per touched expert, so steady-state host usage tracks the working set, but worst case needs ~512 GB RAM or the SSD follow-up.
- **RISK-3 (bias_vl semantics):** the exact noaux-tc bias_vl rule (per-token vs per-sequence modality) is encoded from config only; Task 6 Step 5 reconciliation against reference routing is the arbiter, and `topk_route_v41` must be fixed there if semantics differ.
- **RISK-4 (torch fp8 dtypes):** `float8_e8m0fnu` requires torch ≥ 2.7; the repo floor already satisfies this (`_v4_fp4` tests exercise it), but the tiny-fixture tests fail fast if a dev env is older.

## Self-review notes

- Spec coverage: status gap (Task 4 fixes the silent fp8 mis-map), expert offload (Tasks 2/6/7), engram offload (Tasks 8/9), Path A gating (Task 10 + existing registry guard), docs/evidence discipline (Tasks 11/12), moe-store parsing confirmation (Task 1). MTP/vision/TP exclusions declared.
- Type consistency: `topk_route_v41` signature identical across Task 3 (definition), Task 10 (call); `make_expert_store`/`load_offloaded_v41_flash` names consistent across Tasks 6/7/9/12; env var names consistent across Tasks 7/11/12.
- Placeholders: Task 6/9/10 contain PIN-marked reconciliation points by design (remote reference code cannot be transcribed sight-unseen); each has an explicit verification step that resolves it before execution continues.
