# Uniform FP8 Residency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a load-time `uniform_fp8` resident mode that keeps every routed expert in one released FP8 generation, with no online format changes.

**Architecture:** `uniform_fp8_targets` rejects any manifest that is not entirely `fp8_e4m3_block128`. `load_time_fp8_plan` turns that map into a `PrecisionPlan` with empty transitions and empty evictions. `maybe_apply_uniform_fp8` pushes those targets through the existing `set_precision_targets` binding and does not construct `AdaptivePrecisionPolicy`. The #184 allowlist stays empty, so serving remains bf16 until a later reviewed row exists.

**Tech Stack:** Python 3.10+, PyTorch, pytest. CPU tests do not require a GPU.

**Spec:** `docs/superpowers/plans/2026-09-30-issue-234-phase3-quantized-resident.md`

## Global Constraints

- `adaptive_expert_precision` default stays `False`.
- `adaptive_resident_mode` default is `"legacy"`. The only other legal value is `"uniform_fp8"`.
- Format is `ExpertFormat.FP8_E4M3_BLOCK128` (`"fp8_e4m3_block128"`). Execution is `ExecutionKind.FP8_DEQUANT_BF16_GEMM` (`"fp8_dequant_bf16_gemm"`). Block size is 128. Output dtype is bfloat16. No new architecture floor.
- A uniform plan has `transitions == ()` and `evictions == ()`.
- `uniform_fp8` does not approve a manifest. Unreleased manifests still return `fallback_reason == "manifest_unapproved"`.
- `RELEASED_ADAPTIVE_ENTRIES` stays `frozenset()`. This plan does not add a fingerprint, converter version, or attestation digest.
- Resident-count gate: `resident_fp8 >= 1.8 * resident_bf16`. NLL gate: `(nll_fp8 - nll_bf16) / nll_bf16 <= 0.01`.
- Do not add a kernel, a split dispatch, INT4, MXFP8, or an expert-drop policy. Do not close issue #234.

## Review Focus

- A manifest that mixes FP8 with BF16 or INT4 must raise before any `set_precision_targets` call. Task 1 and Task 3 pin this.
- An empty variant list must raise. Task 1 pins this.
- `uniform_fp8` on an unreleased manifest must stay `manifest_unapproved`. Task 3 pins this.
- `"legacy"` must not call `set_precision_targets`. Task 3 pins this.
- The production allowlist must still be empty after this patch. Task 3 pins this.

---

### Task 1: Uniform FP8 plan

**Files:**
- Create: `moe_infinity/runtime/resident_precision.py`
- Test: `tests/python/unit/test_resident_precision.py`

**Interfaces:**
- Consumes: `CandidateVariantSpec`, `ExpertFormat`, `ExecutionKind`, `ExpertKey`, `PrecisionPlan`
- Produces:
  - `uniform_fp8_targets(variants: Sequence[CandidateVariantSpec]) -> dict[ExpertKey, ExpertFormat]`
  - `load_time_fp8_plan(variants: Sequence[CandidateVariantSpec], *, generation: int) -> PrecisionPlan`

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from moe_infinity.memory.adaptive_precision_policy import ExpertKey
from moe_infinity.runtime.expert_precision import (
    CandidateVariantSpec,
    ExecutionKind,
    ExpertFormat,
)
from moe_infinity.runtime.resident_precision import (
    load_time_fp8_plan,
    uniform_fp8_targets,
)


def _variant(layer, expert, *, fmt=ExpertFormat.FP8_E4M3_BLOCK128,
             execution=ExecutionKind.FP8_DEQUANT_BF16_GEMM,
             version="adaptive-expert-v1", aligned=100):
    return CandidateVariantSpec(
        layer_id=layer,
        expert_id=expert,
        format=fmt,
        execution=execution,
        tensor_ids=(1,),
        tensor_roles=("gate.weight",),
        payload_bytes=aligned,
        aligned_bytes=aligned,
        workspace_bytes=0,
        source_format=ExpertFormat.BF16,
        converter_version=version,
    )


def test_targets_are_fp8_for_every_expert():
    variants = (_variant(0, 1), _variant(2, 4, aligned=50))
    targets = uniform_fp8_targets(variants)
    assert targets == {
        ExpertKey(0, 1): ExpertFormat.FP8_E4M3_BLOCK128,
        ExpertKey(2, 4): ExpertFormat.FP8_E4M3_BLOCK128,
    }


def test_plan_has_no_transitions_and_sums_aligned_bytes():
    plan = load_time_fp8_plan((_variant(0, 1), _variant(0, 2)), generation=3)
    assert plan.epoch == 3
    assert plan.transitions == ()
    assert plan.evictions == ()
    assert plan.accounted_bytes == 200
    assert set(plan.targets.values()) == {ExpertFormat.FP8_E4M3_BLOCK128}


def test_mixed_format_raises():
    mixed = (
        _variant(0, 1),
        _variant(0, 2, fmt=ExpertFormat.BF16, execution=ExecutionKind.BF16_GEMM),
    )
    with pytest.raises(ValueError, match="uniform"):
        uniform_fp8_targets(mixed)


def test_int4_raises():
    mixed = (
        _variant(0, 1),
        _variant(
            0,
            2,
            fmt=ExpertFormat.MARLIN_INT4_GROUP128,
            execution=ExecutionKind.MARLIN_W4A16,
        ),
    )
    with pytest.raises(ValueError, match="uniform"):
        uniform_fp8_targets(mixed)


def test_converter_mismatch_raises():
    with pytest.raises(ValueError, match="uniform"):
        uniform_fp8_targets(
            (_variant(0, 1, version="adaptive-expert-v1"),
             _variant(0, 2, version="adaptive-expert-v2"))
        )


def test_execution_mismatch_raises():
    with pytest.raises(ValueError, match="uniform"):
        uniform_fp8_targets(
            (_variant(0, 1),
             _variant(0, 2, execution=ExecutionKind.BF16_GEMM))
        )


def test_duplicate_expert_raises():
    with pytest.raises(ValueError, match="uniform"):
        uniform_fp8_targets((_variant(0, 1), _variant(0, 1, aligned=40)))


def test_empty_variants_raise():
    with pytest.raises(ValueError, match="uniform"):
        uniform_fp8_targets(())


def test_negative_generation_raises():
    with pytest.raises(ValueError):
        load_time_fp8_plan((_variant(0, 1),), generation=-1)
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `pytest tests/python/unit/test_resident_precision.py -v`

Expected: FAIL with an import error for `moe_infinity.runtime.resident_precision`.

- [ ] **Step 3: Implement the two functions**

`uniform_fp8_targets` raises `ValueError` whose message contains `"uniform"` when `variants` is empty, when two rows share `(layer_id, expert_id)`, when any `format` is not `ExpertFormat.FP8_E4M3_BLOCK128`, when any `execution` is not `ExecutionKind.FP8_DEQUANT_BF16_GEMM`, or when `converter_version` is not identical across rows. Otherwise return one `ExpertKey` per row mapped to `ExpertFormat.FP8_E4M3_BLOCK128`.

`load_time_fp8_plan` raises `ValueError` when `generation < 0`. Otherwise return a `PrecisionPlan` whose `epoch` is `generation`, `targets` is `uniform_fp8_targets(variants)`, `transitions` is `()`, `evictions` is `()`, and `accounted_bytes` is the sum of `aligned_bytes`.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `pytest tests/python/unit/test_resident_precision.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add moe_infinity/runtime/resident_precision.py tests/python/unit/test_resident_precision.py
git commit -m "feat(precision): plan a load-time uniform FP8 residency"
```

### Task 2: Resident mode config

**Files:**
- Modify: `moe_infinity/utils/config.py` (field next to `adaptive_expert_precision`, check at the end of `__post_init__`)
- Test: `tests/python/unit/test_utils_config.py`

**Interfaces:**
- Consumes: nothing
- Produces: `ArcherConfig.adaptive_resident_mode: str` with default `"legacy"`

- [ ] **Step 1: Write the failing tests**

Follow the file's `monkeypatch.setattr("torch.cuda.device_count", lambda: 1)` pattern and the existing `UserWarning` from `ArcherConfig`.

```python
def test_adaptive_resident_mode_defaults_to_legacy(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    with pytest.warns(UserWarning):
        config = ArcherConfig(offload_path="/tmp")
    assert config.adaptive_expert_precision is False
    assert config.adaptive_resident_mode == "legacy"


def test_uniform_fp8_mode_is_accepted(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    with pytest.warns(UserWarning):
        config = ArcherConfig(offload_path="/tmp", adaptive_resident_mode="uniform_fp8")
    assert config.adaptive_resident_mode == "uniform_fp8"
    assert config.adaptive_expert_precision is False


def test_unknown_resident_mode_is_rejected(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    with pytest.raises(ValueError, match="adaptive_resident_mode"):
        with pytest.warns(UserWarning):
            ArcherConfig(offload_path="/tmp", adaptive_resident_mode="int4")
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `pytest tests/python/unit/test_utils_config.py::test_adaptive_resident_mode_defaults_to_legacy tests/python/unit/test_utils_config.py::test_uniform_fp8_mode_is_accepted tests/python/unit/test_utils_config.py::test_unknown_resident_mode_is_rejected -v`

Expected: FAIL because `adaptive_resident_mode` does not exist, or because `"int4"` is accepted.

- [ ] **Step 3: Add the field and validate it**

Help text: `"legacy"` keeps the online adaptive policy; `"uniform_fp8"` pins every released expert to load-time FP8 and does not change format online. Default `"legacy"`.

At the end of `__post_init__`, reject anything other than `"legacy"` or `"uniform_fp8"` with `adaptive_resident_mode must be 'legacy' or 'uniform_fp8', got {value!r}`. Do not require `adaptive_expert_precision` to be true when the mode is `uniform_fp8`.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `pytest tests/python/unit/test_utils_config.py::test_adaptive_resident_mode_defaults_to_legacy tests/python/unit/test_utils_config.py::test_uniform_fp8_mode_is_accepted tests/python/unit/test_utils_config.py::test_unknown_resident_mode_is_rejected -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add moe_infinity/utils/config.py tests/python/unit/test_utils_config.py
git commit -m "feat(config): add uniform FP8 resident mode"
```

### Task 3: Apply the plan without bypassing the allowlist

**Files:**
- Modify: `moe_infinity/runtime/resident_precision.py`
- Modify: `moe_infinity/runtime/model_offload.py` (the `if resolution.enabled:` block that registers variants and constructs `AdaptivePrecisionPolicy`)
- Test: `tests/python/unit/test_resident_precision.py`
- Test: `tests/python/unit/test_adaptive_precision_wiring.py`

**Interfaces:**
- Consumes: `load_time_fp8_plan` from Task 1; `adaptive_resident_mode` from Task 2; `ExpertDispatcher.set_precision_targets(targets, epoch)`
- Produces: `maybe_apply_uniform_fp8(dispatcher, variants, *, mode: str, generation: int) -> bool`

- [ ] **Step 1: Write the failing tests**

Append to `tests/python/unit/test_resident_precision.py`:

```python
from moe_infinity.runtime.resident_precision import maybe_apply_uniform_fp8


class _Dispatcher:
    def __init__(self):
        self.targets = None
        self.epoch = None

    def set_precision_targets(self, targets, epoch):
        self.targets = list(targets)
        self.epoch = epoch
        return True


def test_legacy_mode_does_not_set_targets():
    dispatcher = _Dispatcher()
    applied = maybe_apply_uniform_fp8(
        dispatcher, (_variant(0, 1),), mode="legacy", generation=3
    )
    assert applied is False
    assert dispatcher.targets is None


def test_uniform_mode_sets_fp8_targets_once():
    dispatcher = _Dispatcher()
    applied = maybe_apply_uniform_fp8(
        dispatcher, (_variant(0, 1), _variant(0, 2)), mode="uniform_fp8", generation=3
    )
    assert applied is True
    assert dispatcher.epoch == 3
    assert {(row[0], row[1], row[2], row[3]) for row in dispatcher.targets} == {
        (0, 1, "fp8_e4m3_block128", 3),
        (0, 2, "fp8_e4m3_block128", 3),
    }


def test_mixed_manifest_does_not_call_the_dispatcher():
    dispatcher = _Dispatcher()
    mixed = (
        _variant(0, 1),
        _variant(0, 2, fmt=ExpertFormat.BF16, execution=ExecutionKind.BF16_GEMM),
    )
    with pytest.raises(ValueError, match="uniform"):
        maybe_apply_uniform_fp8(
            dispatcher, mixed, mode="uniform_fp8", generation=1
        )
    assert dispatcher.targets is None
```

Append to `tests/python/unit/test_adaptive_precision_wiring.py`:

```python
def test_uniform_fp8_mode_does_not_approve_an_unreleased_manifest(
    tmp_path, monkeypatch
):
    entry = ReleasedAdaptiveEntry(
        "a" * 64, ExpertFormat.FP8_E4M3_BLOCK128, "adaptive-expert-v1", "b" * 64
    )
    manifest = SimpleNamespace(release_entries=frozenset({entry}))
    monkeypatch.setattr(
        expert_variant_manifest.ExpertVariantManifest,
        "load_current",
        lambda *args, **kwargs: manifest,
    )
    result = _resolve_adaptive_precision(
        SimpleNamespace(model_type="qwen3_moe", quantization_config=None),
        _archer(
            adaptive_expert_precision=True,
            adaptive_variant_build=False,
            adaptive_hbm_budget_bytes=1024,
            adaptive_resident_mode="uniform_fp8",
        ),
        str(tmp_path),
        extension_names=set(),
        purpose="serve",
        checkpoint_fingerprint="a" * 64,
        released_entries=frozenset(),
    )
    assert result.enabled is False
    assert result.fallback_reason == "manifest_unapproved"


def test_production_allowlist_stays_empty():
    from moe_infinity.runtime.adaptive_precision_allowlist import (
        RELEASED_ADAPTIVE_ENTRIES,
    )

    assert RELEASED_ADAPTIVE_ENTRIES == frozenset()
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `pytest tests/python/unit/test_resident_precision.py::test_legacy_mode_does_not_set_targets tests/python/unit/test_resident_precision.py::test_uniform_mode_sets_fp8_targets_once tests/python/unit/test_resident_precision.py::test_mixed_manifest_does_not_call_the_dispatcher tests/python/unit/test_adaptive_precision_wiring.py::test_uniform_fp8_mode_does_not_approve_an_unreleased_manifest tests/python/unit/test_adaptive_precision_wiring.py::test_production_allowlist_stays_empty -v`

Expected: FAIL on the missing helper. The allowlist test passes if the set is already empty; keep it as the regression pin.

- [ ] **Step 3: Implement the helper and branch `model_offload`**

`maybe_apply_uniform_fp8` returns `False` without touching `dispatcher` when `mode != "uniform_fp8"`. When `mode == "uniform_fp8"`, build `load_time_fp8_plan` first. If that raises, do not call `set_precision_targets` and do not catch the exception, so the caller cannot fall through to the online policy. On success, call `dispatcher.set_precision_targets(plan.as_native_targets(generations), plan.epoch)` where `generations` maps `(ExpertKey(layer_id, expert_id), variant.format)` to `generation`, and return `True`.

In `model_offload`, inside the existing `if resolution.enabled:` block, keep `set_adaptive_hbm_budget_bytes` and the `register_expert_variant` loop as they are. After `native_generation = int(resolution.manifest.generation[1:])`, call `maybe_apply_uniform_fp8` with `getattr(self.archer_config, "adaptive_resident_mode", "legacy")`. If it returns `True`, do not construct `AdaptivePrecisionPolicy` and do not call `set_precision_policy`. If it returns `False`, keep the existing catalog and policy construction.

Do not edit `RELEASED_ADAPTIVE_ENTRIES`. Do not read `adaptive_resident_mode` inside `_resolve_adaptive_precision`.

- [ ] **Step 4: Run the tests and the existing adaptive wiring file**

Run: `pytest tests/python/unit/test_resident_precision.py tests/python/unit/test_adaptive_precision_wiring.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add moe_infinity/runtime/resident_precision.py moe_infinity/runtime/model_offload.py tests/python/unit/test_resident_precision.py tests/python/unit/test_adaptive_precision_wiring.py
git commit -m "feat(precision): apply uniform FP8 targets at load time"
```

### Task 4: Resident-count and NLL gates

**Files:**
- Create: `benchmarks/adaptive_precision/resident_fp8_gate.py`
- Create: `benchmarks/adaptive_precision/bench_resident_fp8.py`
- Test: `tests/python/unit/test_resident_fp8_gate.py`

**Interfaces:**
- Consumes: nothing from the runtime tasks
- Produces: `KERNEL_MATRIX`, `resident_ratio_passes(resident_fp8: int, resident_bf16: int) -> bool`, `nll_gate_passes(nll_bf16: float, nll_fp8: float) -> bool`

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from benchmarks.adaptive_precision.resident_fp8_gate import (
    KERNEL_MATRIX,
    nll_gate_passes,
    resident_ratio_passes,
)


def test_kernel_matrix_is_the_existing_blockwise_path():
    assert KERNEL_MATRIX == {
        "format": "fp8_e4m3_block128",
        "execution": "fp8_dequant_bf16_gemm",
        "block_size": 128,
        "output_dtype": "bfloat16",
        "architecture_floor": None,
        "online_tier_changes": False,
        "uniform_per_dispatch": True,
    }


def test_resident_ratio_gate_is_1_8x():
    assert resident_ratio_passes(18, 10) is True
    assert resident_ratio_passes(17, 10) is False


def test_nll_gate_allows_one_percent_worse():
    assert nll_gate_passes(100.0, 101.0) is True
    assert nll_gate_passes(100.0, 101.0 + 0.1) is False


def test_gates_reject_nonpositive_baselines():
    with pytest.raises(ValueError):
        resident_ratio_passes(18, 0)
    with pytest.raises(ValueError):
        nll_gate_passes(0.0, 1.0)
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `pytest tests/python/unit/test_resident_fp8_gate.py -v`

Expected: FAIL with an import error for `benchmarks.adaptive_precision.resident_fp8_gate`.

- [ ] **Step 3: Implement the gate module and the bench entry point**

`resident_ratio_passes` raises `ValueError` unless both counts are integers `>= 0` and `resident_bf16 > 0`. It returns `resident_fp8 >= 1.8 * resident_bf16`.

`nll_gate_passes` raises `ValueError` unless both values are finite and `nll_bf16 > 0`. It returns `(nll_fp8 - nll_bf16) / nll_bf16 <= 0.01`.

`KERNEL_MATRIX` is the dict in the test.

`bench_resident_fp8.py` exits 0 on `--help`. Its docstring states the manual GPU comparison: one bf16 store and one uniform FP8 store, same HBM budget, then `resident_ratio_passes` and `nll_gate_passes`. It does not download weights. `--enforce` with `--resident-fp8`, `--resident-bf16`, `--nll-fp8`, and `--nll-bf16` prints `pass` or `fail` and exits 0 only when both gates pass.

- [ ] **Step 4: Run the unit tests and the help command**

Run: `pytest tests/python/unit/test_resident_fp8_gate.py -v && python benchmarks/adaptive_precision/bench_resident_fp8.py --help`

Expected: PASS, and `--help` exits 0.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/adaptive_precision/resident_fp8_gate.py benchmarks/adaptive_precision/bench_resident_fp8.py tests/python/unit/test_resident_fp8_gate.py
git commit -m "feat(bench): add uniform FP8 residency gates"
```
