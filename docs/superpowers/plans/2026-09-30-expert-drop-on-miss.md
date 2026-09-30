# Drop-on-Miss Expert Skipping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `on_miss` policy that skips non-resident routed experts under a per-token dropped-mass budget and renormalizes the surviving weights.

**Architecture:** A pure function in `moe_infinity/runtime/expert_drop.py` decides the drop. `DistributedExpertExecutor.dispatch_local` calls it only when `expert_drop_policy="on_miss"` and `expert_drop_mass_budget > 0`, then feeds the rewritten mask to `set_inputs`, the current-layer route-ahead pin, and `enqueue_expert` / `dispatch_experts`. Next-layer `correct_prefetch` receives the pre-drop expert union when any expert was dropped. Residency is a read-only `resident_on_gpu(layer_idx)` binding over `node->device.is_cuda()`.

**Tech Stack:** Python 3.10+, PyTorch, pybind11, pytest. CPU tests do not require a GPU.

**Spec:** `docs/superpowers/plans/2026-09-30-issue-234-phase2-expert-drop.md`

## Global Constraints

- `expert_drop_policy` default is `"off"`. Legal values are `"off"` and `"on_miss"` only.
- `expert_drop_min_k` default is `1` and must be `>= 1`.
- `expert_drop_mass_budget` default is `0.0` and must be in `[0, 1]`. Budget `0` does not rewrite tensors. No non-zero budget is the default.
- `expert_drop_flatness_floor` default is `0.5` and must be in `[0, 1]`. Bypass when `w_min / w_max > floor`. Equality does not bypass.
- Renormalize surviving post-softmax weights with `w /= w.sum()`. Do not re-softmax.
- Drop only experts whose residency bit is false. Tie break is lower weight, then higher expert id.
- Output mask is a subset of the input mask.
- A non-finite or negative selected weight leaves that token unchanged.
- Missing probe, exception, or a residency vector of the wrong length treats every expert as resident.
- Current-layer `set_inputs`, enqueue, `dispatch_experts`, and route-ahead use the post-drop mask. `correct_to_native_route` uses post-drop ids. `correct_prefetch(layer + 1)` uses the pre-drop union only when `experts_dropped > 0`; otherwise the existing expert list is unchanged.
- `dispatch()` (RPC) is not modified.
- The off path does not call `resident_on_gpu` and passes the original tensor objects to `set_inputs`.
- Eval mass budgets are `0.05`, `0.10`, and `0.20`. The harness records relative NLL and does not fail on it. Decode p99 passes when `p99_on <= 0.8 * p99_off`.
- Do not add an #184 allowlist entry, and do not close issue #234.

## Review Focus

- A selected weight that is NaN, +Inf, or negative must leave that token's mask and weights unchanged. Task 1 pins this.
- Rank other than 2, or a weight tensor whose shape differs from the mask, must enqueue every originally selected expert. Task 4 pins this.
- A dispatcher with no `resident_on_gpu`, a probe that raises, or a vector of the wrong length must drop nothing. Task 4 pins this.
- After a drop on the native-routing wait path, `correct_to_native_route` must see the executed ids and `correct_prefetch(layer + 1)` must still see the dropped id. Task 4 pins this.
- `expert_drop_policy="off"` must pass the caller's mask and weight objects through to `set_inputs` without copying. Task 4 pins this.

---

### Task 1: Pure drop selection

**Files:**
- Create: `moe_infinity/runtime/expert_drop.py`
- Test: `tests/python/unit/test_expert_drop.py`

**Interfaces:**
- Consumes: nothing
- Produces: `ExpertDropCounts` and `select_expert_drops(router_mask: torch.Tensor, router_weights: torch.Tensor, resident: torch.Tensor, *, min_k: int, mass_budget: float, flatness_floor: float) -> tuple[torch.Tensor, torch.Tensor, ExpertDropCounts]`

`ExpertDropCounts` is a frozen dataclass with `tokens_seen: int`, `tokens_changed: int`, `experts_dropped: int`, `flatness_bypasses: int`.

- [ ] **Step 1: Write the failing tests**

```python
import torch
import pytest

from moe_infinity.runtime.expert_drop import select_expert_drops


def test_drops_lightest_nonresident_inside_budget():
    mask = torch.tensor([[True, True, True, False]])
    weights = torch.tensor([[0.5, 0.25, 0.125, 0.0]])
    resident = torch.tensor([True, False, False, True])
    out_mask, out_w, counts = select_expert_drops(
        mask, weights, resident, min_k=1, mass_budget=0.25, flatness_floor=0.5
    )
    assert out_mask.tolist() == [[True, True, False, False]]
    assert torch.allclose(out_w, torch.tensor([[2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0]]))
    assert counts.tokens_seen == 1
    assert counts.tokens_changed == 1
    assert counts.experts_dropped == 1
    assert counts.flatness_bypasses == 0


def test_tie_drops_higher_expert_id_first():
    mask = torch.tensor([[True, True, True, False]])
    weights = torch.tensor([[0.5, 0.25, 0.25, 0.0]])
    resident = torch.tensor([True, False, False, True])
    out_mask, out_w, counts = select_expert_drops(
        mask, weights, resident, min_k=1, mass_budget=0.25, flatness_floor=0.5
    )
    assert out_mask.tolist() == [[True, True, False, False]]
    assert counts.experts_dropped == 1
    assert torch.allclose(out_w, torch.tensor([[2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0]]))


def test_min_k_blocks_a_drop_the_budget_would_allow():
    mask = torch.tensor([[True, True, True]])
    weights = torch.tensor([[0.5, 0.25, 0.25]])
    resident = torch.tensor([False, False, False])
    out_mask, out_w, counts = select_expert_drops(
        mask, weights, resident, min_k=3, mass_budget=1.0, flatness_floor=1.0
    )
    assert torch.equal(out_mask, mask)
    assert torch.equal(out_w, weights)
    assert counts.experts_dropped == 0


def test_flatness_equal_to_floor_still_drops():
    mask = torch.tensor([[True, True]])
    weights = torch.tensor([[0.5, 0.25]])
    resident = torch.tensor([True, False])
    _, _, counts = select_expert_drops(
        mask, weights, resident, min_k=1, mass_budget=0.25, flatness_floor=0.5
    )
    assert counts.experts_dropped == 1
    assert counts.flatness_bypasses == 0


def test_flatness_above_floor_bypasses():
    mask = torch.tensor([[True, True]])
    weights = torch.tensor([[0.5, 0.25 + 2**-10]])
    resident = torch.tensor([True, False])
    out_mask, out_w, counts = select_expert_drops(
        mask, weights, resident, min_k=1, mass_budget=1.0, flatness_floor=0.5
    )
    assert torch.equal(out_mask, mask)
    assert torch.equal(out_w, weights)
    assert counts.flatness_bypasses == 1
    assert counts.experts_dropped == 0


def test_budget_zero_returns_the_same_objects():
    mask = torch.tensor([[True, False]])
    weights = torch.tensor([[0.5, 0.5]])
    resident = torch.tensor([False, False])
    out_mask, out_w, counts = select_expert_drops(
        mask, weights, resident, min_k=1, mass_budget=0.0, flatness_floor=0.5
    )
    assert out_mask is mask
    assert out_w is weights
    assert counts.tokens_seen == 0
    assert counts.experts_dropped == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.25])
def test_nonfinite_or_negative_selected_weight_is_unchanged(bad):
    mask = torch.tensor([[True, True]])
    weights = torch.tensor([[0.5, bad]])
    resident = torch.tensor([False, False])
    out_mask, out_w, counts = select_expert_drops(
        mask, weights, resident, min_k=1, mass_budget=1.0, flatness_floor=1.0
    )
    assert torch.equal(out_mask, mask)
    assert out_w[0, 0].item() == 0.5
    if bad != bad:  # NaN
        assert out_w[0, 1].item() != out_w[0, 1].item()
    else:
        assert out_w[0, 1].item() == bad
    assert counts.experts_dropped == 0
    assert counts.tokens_changed == 0


def test_preserves_weight_dtype_and_device():
    mask = torch.tensor([[True, True]], dtype=torch.bool)
    weights = torch.tensor([[0.75, 0.25]], dtype=torch.float64)
    resident = torch.tensor([True, False])
    out_mask, out_w, _ = select_expert_drops(
        mask, weights, resident, min_k=1, mass_budget=0.5, flatness_floor=1.0
    )
    assert out_mask.dtype == torch.bool
    assert out_mask.device == mask.device
    assert out_w.dtype == torch.float64
    assert out_w.device == weights.device
    assert torch.allclose(out_w, torch.tensor([[1.0, 0.0]], dtype=torch.float64))


def test_never_sets_a_false_mask_bit():
    mask = torch.tensor([[False, True, False]])
    weights = torch.tensor([[0.4, 0.6, 0.2]])
    resident = torch.tensor([False, False, False])
    out_mask, _, _ = select_expert_drops(
        mask, weights, resident, min_k=1, mass_budget=1.0, flatness_floor=1.0
    )
    assert out_mask.tolist() == [[False, True, False]]
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `pytest tests/python/unit/test_expert_drop.py -v`

Expected: FAIL with `ModuleNotFoundError` or import error for `moe_infinity.runtime.expert_drop`.

- [ ] **Step 3: Implement `select_expert_drops`**

Per token row, in order:

1. If `mass_budget == 0`, return the input tensors and a zero `ExpertDropCounts` without reading rows.
2. Selected indexes are the true mask bits. If there are `<= min_k` of them, leave the row.
3. If any selected weight is non-finite or `< 0`, leave the row.
4. If the max selected weight is `0`, leave the row.
5. If `min(selected) / max(selected) > flatness_floor`, increment `flatness_bypasses` and leave the row.
6. Candidates are selected indexes whose `resident` bit is false, sorted by weight ascending and expert id descending.
7. Walk that list. Stop when remaining selected count is `<= min_k` or when `dropped_mass + weight > mass_budget`. Otherwise clear that mask bit, set its weight to `0`, and add the weight to `dropped_mass`.
8. If nothing was cleared, leave the original weights. Otherwise divide the surviving selected weights by their sum. If that sum is `0`, leave the zeros unscaled.

Copy mask and weights only when a row changes or when a row is visited with `mass_budget > 0` and a new tensor is required for the result. The returned mask is bool on the input mask's device. The returned weights use the input weight dtype and device. `tokens_seen` counts rows actually visited. `tokens_changed` counts rows whose mask changed.

Raise `ValueError` when `router_mask.dim() != 2`, `router_weights.shape != router_mask.shape`, `resident.shape != (router_mask.shape[-1],)`, `min_k < 1`, or either float is outside `[0, 1]` or non-finite. Do not import the dispatcher.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `pytest tests/python/unit/test_expert_drop.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add moe_infinity/runtime/expert_drop.py tests/python/unit/test_expert_drop.py
git commit -m "feat(drop): select non-resident experts under a mass budget"
```

### Task 2: Config fields

**Files:**
- Modify: `moe_infinity/utils/config.py` (the `ArcherConfig` fields next to `adaptive_expert_precision`, and the end of `__post_init__`)
- Test: `tests/python/unit/test_utils_config.py`

**Interfaces:**
- Consumes: nothing
- Produces: `ArcherConfig.expert_drop_policy: str`, `expert_drop_min_k: int`, `expert_drop_mass_budget: float`, `expert_drop_flatness_floor: float` with the Global Constraints defaults

- [ ] **Step 1: Write the failing tests**

Add these to `tests/python/unit/test_utils_config.py`. Follow the file's `monkeypatch.setattr("torch.cuda.device_count", lambda: 1)` pattern, and expect the existing `UserWarning` from `ArcherConfig` construction when `use_native_engine` stays at its default.

```python
def test_expert_drop_defaults(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    with pytest.warns(UserWarning):
        config = ArcherConfig(offload_path="/tmp")
    assert config.expert_drop_policy == "off"
    assert config.expert_drop_min_k == 1
    assert config.expert_drop_mass_budget == 0.0
    assert config.expert_drop_flatness_floor == 0.5


def test_expert_drop_on_miss_accepts_budget(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    with pytest.warns(UserWarning):
        config = ArcherConfig(
            offload_path="/tmp",
            expert_drop_policy="on_miss",
            expert_drop_mass_budget=0.10,
        )
    assert config.expert_drop_policy == "on_miss"
    assert config.expert_drop_mass_budget == 0.10


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expert_drop_policy": "threshold"},
        {"expert_drop_min_k": 0},
        {"expert_drop_mass_budget": -0.1},
        {"expert_drop_mass_budget": 1.1},
        {"expert_drop_mass_budget": float("nan")},
        {"expert_drop_flatness_floor": -0.01},
        {"expert_drop_flatness_floor": 1.01},
    ],
)
def test_expert_drop_rejects_illegal_values(monkeypatch, kwargs):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    with pytest.raises(ValueError):
        with pytest.warns(UserWarning):
            ArcherConfig(offload_path="/tmp", **kwargs)
```

- [ ] **Step 2: Run the new tests and confirm they fail**

Run: `pytest tests/python/unit/test_utils_config.py::test_expert_drop_defaults tests/python/unit/test_utils_config.py::test_expert_drop_on_miss_accepts_budget tests/python/unit/test_utils_config.py::test_expert_drop_rejects_illegal_values -v`

Expected: FAIL because the attributes do not exist, or because illegal values are accepted.

- [ ] **Step 3: Add the four fields and validate them at the end of `ArcherConfig.__post_init__`**

Field help strings state that the default is off, that `on_miss` drops only non-resident routed experts, and that budget `0` rewrites nothing. Validation messages:

- `expert_drop_policy must be 'off' or 'on_miss', got {value!r}`
- `expert_drop_min_k must be >= 1, got {value}`
- `expert_drop_mass_budget must be in [0, 1], got {value}`
- `expert_drop_flatness_floor must be in [0, 1], got {value}`

Use `math.isfinite` for the two floats before the range check so NaN fails the range message.

- [ ] **Step 4: Run the new tests and confirm they pass**

Run: `pytest tests/python/unit/test_utils_config.py::test_expert_drop_defaults tests/python/unit/test_utils_config.py::test_expert_drop_on_miss_accepts_budget tests/python/unit/test_utils_config.py::test_expert_drop_rejects_illegal_values -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add moe_infinity/utils/config.py tests/python/unit/test_utils_config.py
git commit -m "feat(config): add opt-in expert drop policy"
```

### Task 3: Read-only residency probe

**Files:**
- Modify: `core/parallel/expert_dispatcher.h` (public section, next to `GetCacheHitRate`)
- Modify: `core/parallel/expert_dispatcher.cpp`
- Modify: `core/python/py_archer_prefetch.cpp` (the `expert_dispatcher` bindings, next to `enqueue_expert`)
- Test: `tests/python/unit/test_resident_on_gpu_binding.py`

**Interfaces:**
- Consumes: `experts_[expert_idx][layer_idx]->node->device.is_cuda()`
- Produces: `ExpertDispatcher::ResidentOnGpu(int layer_idx) const -> std::vector<std::uint8_t>`, bound as `resident_on_gpu(layer_idx)`

- [ ] **Step 1: Write the failing binding test**

```python
import pytest


def test_resident_on_gpu_is_bound():
    store = pytest.importorskip("moe_infinity._store")
    if "mock" in type(store).__module__ or not getattr(store, "__file__", None):
        pytest.skip("_store is mocked in this environment")
    dispatcher = getattr(store, "expert_dispatcher", None)
    if dispatcher is None:
        pytest.skip("expert_dispatcher extension class unavailable")
    assert hasattr(dispatcher, "resident_on_gpu")
```

- [ ] **Step 2: Run the test and confirm it fails when the extension imports**

Run: `pytest tests/python/unit/test_resident_on_gpu_binding.py -v`

Expected: FAIL with `hasattr` false when `moe_infinity._store` imports. SKIP is acceptable only when the extension is absent.

- [ ] **Step 3: Implement `ResidentOnGpu`**

Return an empty vector when `layer_idx < 0`, `experts_` is empty, or any expert in `[0, num_experts_)` lacks a node at that layer. An empty vector means "unknown" to Python. Otherwise return one byte per expert, `1` when `node->device.is_cuda()` and `0` otherwise. Do not take the dispatcher mutex in a way that calls into admission, and do not change `Enqueue`. Bind the method as `resident_on_gpu` with argument name `layer_idx`.

- [ ] **Step 4: Run the binding test**

Run: `pytest tests/python/unit/test_resident_on_gpu_binding.py -v`

Expected: PASS when the extension imports, otherwise SKIP.

- [ ] **Step 5: Commit**

```bash
git add core/parallel/expert_dispatcher.h core/parallel/expert_dispatcher.cpp core/python/py_archer_prefetch.cpp tests/python/unit/test_resident_on_gpu_binding.py
git commit -m "feat(dispatcher): expose per-layer GPU residency"
```

### Task 4: Wire the drop into local dispatch

**Files:**
- Modify: `moe_infinity/distributed/expert_executor.py` (`DistributedExpertExecutor.__init__`, `dispatch_local`, `wait_dispatch_local`)
- Test: `tests/python/unit/test_expert_drop_executor.py`

**Interfaces:**
- Consumes: `select_expert_drops` and `ExpertDropCounts` from Task 1; config fields from Task 2; `resident_on_gpu(layer_idx)` from Task 3; `union_experts_from_mask` from `moe_infinity.spec_decode._prefetch_route`
- Produces:
  - `_apply_expert_drop(self, layer_id: int, router_mask, router_weights) -> tuple[torch.Tensor, torch.Tensor, list[int] | None]`
  - `get_expert_drop_stats(self) -> dict`
  - `reset_expert_drop_stats(self) -> None`
  - `_pending_prefetch` gains an optional 8th element, `trace_ids: list[int] | None`

`get_expert_drop_stats` returns a new dict with integer fields `tokens_seen`, `tokens_changed`, `experts_dropped`, `flatness_bypasses`, `budget_disabled`, `residency_unknown`, `shape_bypasses`, and `drops_by_layer: dict[int, int]` copied from the executor's counters.

- [ ] **Step 1: Write the failing executor tests**

Use `SimpleNamespace` configs and a fake dispatcher, as `tests/python/unit/test_gpu_only_expert_routing.py` does. Patch `torch.cuda.device_count` to `1`.

```python
def test_policy_off_passes_original_tensors():
    mask = torch.tensor([[True, False, True]])
    weights = mask.float()
    executor, dispatcher = make_executor(policy="off")
    hidden = torch.ones(1, 2)
    with patch("torch.cuda.device_count", return_value=1):
        executor.dispatch_local(3, hidden, mask, weights)
    set_inputs = next(call for call in dispatcher.calls if call[0] == "set_inputs")
    assert set_inputs[2] is mask
    assert set_inputs[3] is weights
    assert dispatcher.resident_calls == []
    enqueued = [call[2] for call in dispatcher.calls if call[0] == "enqueue_expert"]
    assert enqueued == [0, 2]


def test_on_miss_enqueues_survivors_and_prefetches_original_union():
    executor, dispatcher = make_executor(policy="on_miss", budget=0.25)
    dispatcher.resident = [1, 0, 0, 1]
    prefetcher = FakePrefetcher()
    executor.set_prefetcher(prefetcher)
    hidden = torch.ones(1, 4)
    mask = torch.tensor([[True, True, True, False]])
    weights = torch.tensor([[0.5, 0.25, 0.125, 0.0]])
    with patch("torch.cuda.device_count", return_value=1):
        executor.dispatch_local(4, hidden, mask, weights)
        executor.wait_dispatch_local()
    enqueued = [call[2] for call in dispatcher.calls if call[0] == "enqueue_expert"]
    assert enqueued == [0, 1]
    assert prefetcher.corrected == [(5, [0, 1, 2])]
    stats = executor.get_expert_drop_stats()
    assert stats["experts_dropped"] == 1
    assert stats["drops_by_layer"] == {4: 1}


def test_budget_zero_does_not_probe():
    executor, dispatcher = make_executor(policy="on_miss", budget=0.0)
    hidden = torch.ones(1, 2)
    mask = torch.tensor([[True, True]])
    with patch("torch.cuda.device_count", return_value=1):
        executor.dispatch_local(0, hidden, mask, mask.float())
    assert dispatcher.resident_calls == []
    assert executor.get_expert_drop_stats()["budget_disabled"] == 1


def test_bad_shape_enqueues_everyone():
    executor, dispatcher = make_executor(policy="on_miss", budget=0.25)
    dispatcher.resident = [0, 0, 0]
    hidden = torch.ones(1, 2, 3)
    mask = torch.ones(1, 2, 3, dtype=torch.bool)
    with patch("torch.cuda.device_count", return_value=1):
        executor.dispatch_local(1, hidden, mask, mask.float())
    assert any(call[0] == "enqueue_expert" for call in dispatcher.calls)
    assert executor.get_expert_drop_stats()["experts_dropped"] == 0
    assert executor.get_expert_drop_stats()["shape_bypasses"] == 1


@pytest.mark.parametrize("resident", [None, "raise", [1]])
def test_unknown_residency_drops_nothing(resident):
    executor, dispatcher = make_executor(policy="on_miss", budget=1.0)
    dispatcher.resident = resident
    hidden = torch.ones(1, 2)
    mask = torch.tensor([[True, True]])
    weights = torch.tensor([[0.75, 0.25]])
    with patch("torch.cuda.device_count", return_value=1):
        executor.dispatch_local(0, hidden, mask, weights)
    enqueued = [call[2] for call in dispatcher.calls if call[0] == "enqueue_expert"]
    assert enqueued == [0, 1]
    assert executor.get_expert_drop_stats()["residency_unknown"] == 1
    assert executor.get_expert_drop_stats()["experts_dropped"] == 0


def test_native_wait_splits_executed_ids_from_trace_ids():
    executor, dispatcher = make_executor(policy="off")
    dispatcher.active = [0, 1]
    prefetcher = FakePrefetcher()

    def overlap():
        return True

    prefetcher._overlap_active = overlap
    executor.set_prefetcher(prefetcher)
    executor._last_dispatch_used_native_routing = True
    executor._pending_prefetch = (
        prefetcher,
        7,
        None,
        None,
        None,
        [],
        None,
        [0, 1, 2],
    )
    executor.wait_dispatch_local()
    assert prefetcher.native_route == [(7, [0, 1])]
    assert prefetcher.corrected == [(8, [0, 1, 2])]


def test_reset_clears_drop_stats():
    executor, _dispatcher = make_executor(policy="on_miss", budget=0.0)
    hidden = torch.ones(1, 1)
    mask = torch.tensor([[True]])
    with patch("torch.cuda.device_count", return_value=1):
        executor.dispatch_local(0, hidden, mask, mask.float())
    executor.reset_expert_drop_stats()
    stats = executor.get_expert_drop_stats()
    assert stats["budget_disabled"] == 0
    assert stats["drops_by_layer"] == {}
```

`make_executor` puts `gpu_only_expert_routing=False` and `speculative_prefetch_overlap=False` on the namespace. `resident is None` means the fake has no `resident_on_gpu` attribute. `"raise"` means the method raises `RuntimeError`. The fake records `resident_calls`.

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `pytest tests/python/unit/test_expert_drop_executor.py -v`

Expected: FAIL because `_apply_expert_drop` or `get_expert_drop_stats` does not exist.

- [ ] **Step 3: Wire `dispatch_local` and `wait_dispatch_local`**

Read policy and budget with `getattr` so a `SimpleNamespace` that omits them stays off and budget `0`.

`_apply_expert_drop` returns the original tensors and `None` when the policy is not `on_miss`. When the policy is `on_miss` and the budget is `0`, increment `budget_disabled` and return the original tensors and `None` without calling the probe. On a bad rank or a weight shape mismatch, increment `shape_bypasses` and return the originals. Otherwise build the resident bool vector: missing method, exception, or `len != num_experts` increments `residency_unknown` and uses an all-true vector of length `num_experts`. Call `select_expert_drops` with `min_k` and `flatness_floor` from `getattr` defaults `1` and `0.5`. On `ValueError` or `RuntimeError`, increment `shape_bypasses` and return the originals. Add `counts` into the stats. Set `drops_by_layer[layer_id]` only when `experts_dropped > 0`. Return `union_experts_from_mask(router_mask)` as `trace_ids` only when `experts_dropped > 0`; otherwise return `None`.

In `dispatch_local`, call `_apply_expert_drop` after the existing precision-policy block and before `set_inputs`. Pass the returned mask and weights to `set_inputs`, `_maybe_route_ahead_prefetch`, and `_dispatch_eager_local`. Do not change the precision-policy block. Store `trace_ids` as the 8th element of `_pending_prefetch`.

In `wait_dispatch_local`, a 7-element pending tuple means `trace_ids is None`. Keep filling `expert_list` from `take_last_active_experts` when it is `None` and native routing ran, and keep passing that list to `correct_to_native_route`. Pass `trace_ids` to `correct_prefetch` when it is not `None`; otherwise pass `expert_list` as today.

`get_expert_drop_stats` returns a shallow copy with a copied `drops_by_layer`. `reset_expert_drop_stats` zeros every counter and replaces the layer map with an empty dict. Initialize those counters in `__init__`.

Do not edit `dispatch()`.

- [ ] **Step 4: Run the new tests and the existing routing regression**

Run: `pytest tests/python/unit/test_expert_drop_executor.py tests/python/unit/test_gpu_only_expert_routing.py -v`

Expected: PASS. `test_native_active_list_drives_existing_prefetch_correction` still passes with a 7-element pending tuple.

- [ ] **Step 5: Commit**

```bash
git add moe_infinity/distributed/expert_executor.py tests/python/unit/test_expert_drop_executor.py
git commit -m "feat(executor): drop non-resident experts on local dispatch"
```

### Task 5: Latency gate and eval curve constants

**Files:**
- Create: `benchmarks/expert_drop/gates.py`
- Create: `benchmarks/expert_drop/bench_drop_on_miss.py`
- Test: `tests/python/unit/test_expert_drop_gates.py`

**Interfaces:**
- Consumes: `get_expert_drop_stats` from Task 4, by name only. This task does not call the executor.
- Produces: `EVAL_MASS_BUDGETS = (0.05, 0.10, 0.20)`, `latency_gate_passes(p99_off: float, p99_on: float) -> bool`, `accuracy_row(budget: float, nll_off: float, nll_on: float) -> dict[str, float]`

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from benchmarks.expert_drop.gates import (
    EVAL_MASS_BUDGETS,
    accuracy_row,
    latency_gate_passes,
)


def test_eval_budgets_are_the_three_curve_points():
    assert EVAL_MASS_BUDGETS == (0.05, 0.10, 0.20)


def test_latency_gate_is_twenty_percent():
    assert latency_gate_passes(100.0, 80.0) is True
    assert latency_gate_passes(100.0, 80.0 + 2**-20) is False


def test_accuracy_row_records_relative_delta_without_a_pass_flag():
    row = accuracy_row(0.10, nll_off=4.0, nll_on=5.0)
    assert row == {"budget": 0.10, "nll_off": 4.0, "nll_on": 5.0, "relative_delta": 0.25}
    assert "passed" not in row


def test_accuracy_row_rejects_nonpositive_baseline():
    with pytest.raises(ValueError):
        accuracy_row(0.05, nll_off=0.0, nll_on=1.0)
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `pytest tests/python/unit/test_expert_drop_gates.py -v`

Expected: FAIL with import error for `benchmarks.expert_drop.gates`.

- [ ] **Step 3: Implement the gate module and the bench entry point**

`latency_gate_passes` returns `p99_on <= 0.8 * p99_off`. Both arguments must be finite and `> 0`; otherwise raise `ValueError`.

`accuracy_row` returns those four keys. `relative_delta` is `(nll_on - nll_off) / nll_off`. Raise `ValueError` when `nll_off <= 0` or any input is non-finite. Do not include a pass/fail key.

`bench_drop_on_miss.py` parses `--help` and exits 0. Its docstring states the manual GPU run: same model and cache budget twice, once with `expert_drop_policy=off` and once with `on_miss` at each `EVAL_MASS_BUDGETS` value, then `latency_gate_passes` on decode p99 and `accuracy_row` for the NLL pair. It does not download weights and it does not assert an accuracy threshold. `--enforce-latency-gate` with `--p99-off` and `--p99-on` prints `pass` or `fail` and exits 0 only on pass.

- [ ] **Step 4: Run the unit tests and the help command**

Run: `pytest tests/python/unit/test_expert_drop_gates.py -v && python benchmarks/expert_drop/bench_drop_on_miss.py --help`

Expected: PASS, and `--help` exits 0.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/expert_drop/gates.py benchmarks/expert_drop/bench_drop_on_miss.py tests/python/unit/test_expert_drop_gates.py
git commit -m "feat(bench): add drop-on-miss latency gate"
```
