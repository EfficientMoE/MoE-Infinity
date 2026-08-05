# Plan: Move non-performance-critical model-topology *logic* out of C++ into Python

## 1. Goal

Reduce the amount of **model-structure semantics** computed inside the C++ hot-path module `core/model/model_topology.cpp`, moving it into Python where the topology is already constructed, **without** moving any performance-critical runtime lookup or any memory/IO work. Keep the `uint32_t TensorID -> Node*` runtime path 100% in C++.

Success = C++ `InitializeTopology` stops re-deriving `is_sparse` and `corr_id` (layer/expert identity) from loop indices; instead it consumes those values computed in Python. All existing behavior (device placement, caching, prefetch, dispatch, generation output) is byte-for-byte unchanged.

## 2. Findings that constrain scope (from code analysis)

Verified against the current tree:

- Python already builds the topology grouping in `moe_infinity/runtime/model_offload.py::get_topology()` (~lines 1121-1245), returning `List[Tuple[str, List[List[TensorID]]]]`, and calls `self.archer_engine.set_topology(topo)` (line ~1271).
- pybind contract: `core/python/py_archer_prefetch.cpp` (~lines 62-66) binds `set_topology` to `std::vector<std::tuple<std::string, std::vector<std::vector<TensorID>>>>`; `TensorID = uint32_t` (`core/common/types.h:13`).
- `ArcherPrefetchHandle::SetTopology` (`core/prefetch/archer_prefetch_handle.cpp:348-353`) forwards to `ArcherTopologyHandle::InitializeTopology` (`core/model/model_topology.cpp:507-737`).
- `InitializeTopology` interleaves **movable** logic and **non-movable** work:
  - MOVABLE (pure integer/label logic): `is_sparse = (num_groups_in_stage > 1)`; `corr_id = (layer_id & 0xFFFFFFFF) | ((expert_id & 0xFFFFFFFF) << 32)`; last-stage nodes get high bits `0xFFFFFFFF`; node ordering; `children_visit_cnt` parent-child sizing for sparse stages.
  - NON-MOVABLE (must stay C++): `byte_size` from `kTensorIndex`; GPU round-robin `default_device` placement; `kHostMemoryPool` host allocation + partition-file reads via `kArcherTensorHandle`; `SetDevice(...)` data movement.
- Runtime lookups that MUST stay C++ and MUST remain unchanged: `GetNodeFromTensorID` (`model_topology.cpp:739-760`, ~12 hot-path callers in `archer_prefetch_handle.cpp` + `expert_dispatcher.cpp:268`), `GetNodeBodyFromCorrID` (762-780), `GetSparseNodes/GetDenseNodes/GetLFUNodes/GetSparseCacheLimit/GetNumLayersAndExperts`.

**Honest scope caveat (for reviewer):** the movable surface is small (~30-60 lines of index math) and is coupled to the memory-allocation loop it lives in. The value is *separation of concerns / single source of truth for model semantics*, NOT performance. If the reviewer judges the risk/reward unfavorable, the correct outcome may be "do not refactor" — see Section 8.

## 3. Non-goals (explicitly out of scope)

- Do NOT move `GetNodeFromTensorID`, `GetNodeBodyFromCorrID`, or any `Get*Nodes`/cache function to Python.
- Do NOT move `byte_size`/`kTensorIndex`, device placement, `kHostMemoryPool`, partition reads, or `SetDevice` to Python.
- Do NOT change the on-disk format, `name_id_map.json`, or checkpoint layout.
- Do NOT change the numeric values of `corr_id`/`is_sparse` — only *where* they are computed.

## 4. Design: enrich the Python->C++ topology contract

Change the topology element from
`(stage_name, id_groups)` to
`(stage_name, is_sparse, id_groups, corr_ids)`
where `is_sparse: bool`, `id_groups: List[List[TensorID]]` (unchanged), `corr_ids: List[uint64]` (one packed `corr_id` per group, in group order, already including the `0xFFFFFFFF` high-bits marker for last-stage nodes).

C++ `InitializeTopology` then:
- reads `is_sparse` and `corr_id` from the tuple instead of computing them from loop indices;
- keeps the byte_size / device-placement / host-alloc / SetDevice code paths exactly as-is.

Backward-compat option (decision for reviewer): add a **new** binding `set_topology_v2` and keep `set_topology` intact, so a stale `name_id_map`/older caller path still works. Default recommendation: add v2, route Python through v2, leave v1 in place unused (lower blast radius, easy rollback).

## 5. Step-by-step tasks (each independently verifiable)

Ordering: C++ side first (compiles standalone), then Python switch, then verify.

- [ ] **T1 (read/confirm):** Read `core/model/model_topology.cpp:507-737` and `model_topology.h:172-175` in full; confirm exact `corr_id` packing and the last-stage `0xFFFFFFFF` rule. Record the exact lines that compute `is_sparse` and `corr_id`. Evidence: quoted current lines in the PR description.
- [ ] **T2 (C++ signature):** Add `InitializeTopologyV2(const std::vector<std::tuple<std::string, bool, std::vector<std::vector<TensorID>>, std::vector<uint64_t>>>&)` in `model_topology.{h,cpp}` that consumes `is_sparse`+`corr_ids` and otherwise reuses the existing body (extract shared code into a helper to avoid duplication). Evidence: file compiles.
- [ ] **T3 (handle passthrough):** Add `ArcherPrefetchHandle::SetTopologyV2(...)` in `archer_prefetch_handle.{h,cpp}` forwarding to `InitializeTopologyV2`. Evidence: compiles.
- [ ] **T4 (pybind):** Bind `set_topology_v2` in `core/python/py_archer_prefetch.cpp`. Evidence: compiles; `import moe_infinity` exposes the method.
- [ ] **T5 (Python producer):** In `model_offload.py::get_topology()`, compute `is_sparse` and per-group `corr_id` (reusing existing name parsing / `parse_expert_id`) and return the enriched tuples; switch the call site (~line 1271) to `set_topology_v2`. Evidence: `python -c "import moe_infinity"` OK; unit assertion that produced `corr_id`/`is_sparse` equal the values C++ v1 would have produced (see T7).
- [ ] **T6 (build):** Rebuild the extension (`pip install --no-build-isolation -e .`). Evidence: exit code 0. **BLOCKED by disk — see Section 7.**
- [ ] **T7 (equivalence test):** Add a temporary parity check (or a small pytest under `tests/python/`) that runs a supported small model (e.g. `deepseek-ai/DeepSeek-V2-Lite-Chat` if weights are available, else a mocked topology) and asserts the resulting node `corr_id`/`is_sparse`/device placement are identical between v1 and v2 paths. Evidence: test passes.
- [ ] **T8 (smoke generation):** Run `examples/deepseek_v2_chat_example.py` (or the smallest available example) and confirm generation output is unchanged vs a pre-refactor run. Evidence: identical decoded output on a fixed prompt+seed.
- [ ] **T9 (diagnostics/cleanup):** `lsp_diagnostics` clean on changed files; remove temporary parity scaffolding; leave `set_topology` (v1) present but unused unless reviewer approves deletion.

## 6. Verification matrix

| Step | Command / check | Pass criteria |
|---|---|---|
| C++ edits (T2-T4) | build compiles | exit 0 |
| Python edits (T5) | `python -c "import moe_infinity"` | no error |
| Parity (T7) | pytest parity test | v1 == v2 corr_id/is_sparse/device |
| Smoke (T8) | example generation, fixed seed | output identical to baseline |
| Final (T9) | `lsp_diagnostics` on changed files | no new errors |

## 7. Risks & BLOCKERS

- **BLOCKER — disk full.** `/mnt/raid0nvme0` is at 100% (≈2.0 GB free of 14 TB). A from-source rebuild of the CUDA/C++ extension (T6) produces large object files and will very likely fail with `ENOSPC`, and generation (T8) writes an offload dir needing many GB. **T6/T7/T8 cannot be completed until disk space is freed.** This plan should not be marked done on assertion alone; build+run evidence is required.
- **ABI/rebuild risk:** any C++ signature change requires a successful rebuild before Python can use it. Mitigated by adding v2 alongside v1 (no removal) so a partial state still imports.
- **Semantic drift risk:** Python-computed `corr_id` must exactly match the C++ formula, including the last-stage `0xFFFFFFFF` marker and layer/expert index derivation for every supported model family (Mixtral/DeepSeek/Qwen3/GLM/GPT-OSS/NLLB naming differs). Mitigated by the T7 parity test across at least one dense + one sparse stage; ideally gated per model family.
- **Low reward:** net C++ reduction is small; primary benefit is clarity/single-source-of-truth, not speed.

## 8. Decision gate for reviewer

Given Section 2's caveat and Section 7's blocker, choose one:
- **(A) Proceed** with the scoped v2 extraction as specified.
- **(B) Narrow further** — only pass `is_sparse` (drop `corr_id` move) to minimize semantic-drift risk.
- **(C) Do not refactor** — conclude the C++ residue is inherently memory-coupled and the movable logic is too thin to justify a new ABI + parity burden.

## 9. Rollback

All changes are additive (v2 binding alongside v1). Rollback = revert the Python call site to `set_topology` (v1) and, if desired, remove the v2 symbols. No data/format migration involved, so rollback is a pure code revert with a rebuild.
