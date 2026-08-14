# Repository Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a layered, source-aligned documentation system for MoE-Infinity that serves users, operators, contributors, and maintainers without overstating support.

**Architecture:** Keep `README.md` as the concise discovery and quick-start layer, route deeper reading through `docs/README.md`, and place detailed truth in focused guides. Use `ARCHITECTURE.md` for contributor flow and API boundaries, `CHANGELOG.md` for history, and contribution/release documents to enforce updates. Derive claims from source, tests, and recorded validation rather than registry presence or implementation seams alone.

**Tech Stack:** Markdown, Python 3 AST and compilation checks, pytest, FastAPI route inspection, argparse inspection, Git read-only verification, repository-local shell commands.

---

## Execution Notes

- This recovered plan describes the ten implemented task scopes.
- Every command runs from the repository root unless a different directory is stated.
- GPU, model-download, and destructive cache tests are excluded from the default verification path.
- Commits are optional checkpoints. No commit, staging, push, or Git configuration change was performed during the execution that recovered this plan.

## Exact File Map

### Discovery and navigation

- Modify `README.md` — overview, feature discovery, supported models, install, examples, serving and DFlash summaries, benchmark links.
- Create `docs/README.md` — audience-oriented documentation hub.

### Focused user and operator guides

- Create `docs/model-compatibility.md` — registry-complete evidence matrix.
- Create `docs/configuration.md` — active `ArcherConfig` fields and memory rules.
- Create `docs/environment-variables.md` — runtime, serving, profiling, model, build, and packaging keys.
- Modify `docs/dflash.md` — direct and integrated DFlash, rollback, sampling, route-ahead, compatibility, and validation.
- Create `docs/serving.md` — CLI, request fields, auth, streaming, limits, endpoints, inactive prefix-cache scaffolding, DFlash, and watchdogs.
- Create `docs/multi-gpu.md` — supported single-host topology and ownership.
- Create `docs/troubleshooting.md` — symptom-first operational guidance.
- Create `docs/glm-5.2.md` — GLM-specific FP8, serving, MTP, and validation scope.
- Modify `moe_infinity/models/deepseek_v4/README.md` — DeepSeek-V4 official loader, FP4 offload, validation, and historical data caveats.

### Contributor, benchmark, and process surfaces

- Modify `ARCHITECTURE.md` — module map, two execution paths, lifecycle, and API boundaries.
- Modify `docs/benchmarking.md` — terminology, catalog, exclusions, commands, fairness, and reproducibility.
- Modify `benchmarks/expert_io_microbench/README.md` — executable profiler runbook.
- Modify `examples/dflash_gpt_oss_example.py` — accurate documentation comments for the current DFlash path.
- Create `CHANGELOG.md` — `Unreleased` additions, changes, fixes, and known limitations.
- Modify `CONTRIBUTING.md` — documentation-impact matrix and test guidance.
- Modify `.github/PULL_REQUEST_TEMPLATE.md` — documentation and evidence checklist.
- Modify `RELEASE.md` — release documentation and artifact verification.

### Recovered planning artifacts

- Create `docs/superpowers/specs/2026-08-09-repository-documentation-design.md` — approved documentation design.
- Create `docs/superpowers/plans/2026-08-09-repository-documentation.md` — this ten-task implementation plan.

## Task 1: Inventory Source Truth and Documentation Gaps

**Files:**
- Inspect `README.md`.
- Inspect `ARCHITECTURE.md`.
- Inspect every Markdown file under `docs/`, `benchmarks/`, and `moe_infinity/models/`.
- Inspect `moe_infinity/utils/config.py`, `moe_infinity/common/constants.py`, `moe_infinity/entrypoints/openai/api_server_v2.py`, `setup.py`, and benchmark entry points.

- [x] Record the cumulative documentation surfaces and audience gaps.
- [x] Extract active `ArcherConfig` fields and defaults from `moe_infinity/utils/config.py`.
- [x] Extract production environment keys and build controls from Python, `setup.py`, and `CMakeLists.txt`.
- [x] Extract API parser options and FastAPI route decorators.
- [x] Extract model registry families, conditional imports, adapter paths, and tests.
- [x] Classify files under `benchmarks/` as workflows, contributor tools, helpers, or generated results.
- [x] Scan existing claims for stale DFlash, prefix-cache, multi-GPU, model-support, and performance wording.

**Validation:**

```bash
python -m compileall -q examples benchmarks
python -m moe_infinity.entrypoints.openai.api_server_v2 --help
```

Checklist:

- [x] Every later guide has an identified source of truth.
- [x] No capability level is inferred solely from registration.
- [x] GPU and model-download evidence is separated from CPU fixture evidence.

## Task 2: Establish the README and Documentation Hub Layers

**Files:**
- Modify `README.md`.
- Create `docs/README.md`.

- [x] Reduce the README to overview, discovery, prerequisites, minimal examples, and authoritative links.
- [x] Add a supported-model discovery table without duplicating the detailed compatibility matrix.
- [x] Add source-install and optional acceleration guidance.
- [x] Add concise Qwen3.5 and GLM examples with model-specific caveats.
- [x] State that prefix-cache scaffolding exists but the OpenAI request path has no active reuse integration.
- [x] Organize the docs hub under Users, Operators, Contributors, and Project History.

**Validation:**

```bash
python - <<'PY'
from pathlib import Path
text = Path('README.md').read_text()
assert 'docs/README.md' in text
assert 'docs/model-compatibility.md' in text
assert 'does not actively reuse cached prefixes' in text
assert 'Qwen/Qwen3.5-35B-A3B' in text
print('README discovery assertions passed')
PY
```

Checklist:

- [x] README summaries link to focused guides.
- [x] The hub links every new authoritative guide.
- [x] No inactive feature is advertised as active production behavior.

## Task 3: Build the Model Compatibility and Family Guides

**Files:**
- Create `docs/model-compatibility.md`.
- Create `docs/glm-5.2.md`.
- Modify `moe_infinity/models/deepseek_v4/README.md`.
- Modify model discovery and notes in `README.md`.

- [x] Include every family from `MODEL_MAPPING_NAMES`, including conditional and registry-only entries.
- [x] Separate sync generation, continuous serving, expert offload, speculative decoding, topology, and limitations.
- [x] Mark tiny-fixture, real-checkpoint, and unrecorded validation distinctly.
- [x] Document Qwen3.5 as text-only and state that no real-checkpoint serving validation is recorded.
- [x] Document GLM FP8 residency, native-engine status, MTP scope, and missing GLM DFlash drafter.
- [x] Preserve DeepSeek-V4 official-loader requirements and label incomplete-provenance measurements as historical.

**Validation:**

```bash
pytest -q tests/python/unit/test_model_registry.py
python - <<'PY'
from pathlib import Path
matrix = Path('docs/model-compatibility.md').read_text()
for family in ('DeepSeek-V2', 'DeepSeek-V3', 'DeepSeek-V4', 'Mixtral', 'Qwen3', 'Qwen3.5', 'GLM-5.2', 'GPT-OSS', 'DBRX', 'Jamba', 'OLMoE', 'NLLB-MoE', 'OPT'):
    assert family in matrix
assert 'no real-checkpoint serving validation recorded' in matrix
print('model matrix assertions passed')
PY
```

Checklist:

- [x] Conditional imports are named.
- [x] Unsupported OPT status is explicit.
- [x] Fixture coverage is not presented as production topology validation.

## Task 4: Document Configuration and Environment Controls

**Files:**
- Create `docs/configuration.md`.
- Create `docs/environment-variables.md`.
- Link both from `README.md` and `docs/README.md`.

- [x] Document all sixteen active `ArcherConfig` fields and exact defaults.
- [x] Explain derived fields, memory-ratio normalization, and store fingerprint checks.
- [x] Separate Python config names from serving config names.
- [x] Document runtime, acceleration, profiling, deterministic, model-specific, build, packaging, and standard third-party environment keys.
- [x] Exclude test-only variables from the production table while listing them separately.

**Validation:**

```bash
pytest -q tests/python/unit/test_utils_config.py tests/python/unit/test_kv_config_wiring.py
python - <<'PY'
from dataclasses import fields
from pathlib import Path
from moe_infinity.utils.config import ArcherConfig
doc = Path('docs/configuration.md').read_text()
missing = [f.name for f in fields(ArcherConfig) if f.name not in doc]
assert not missing, missing
print('ArcherConfig coverage passed')
PY
```

Checklist:

- [x] Defaults match source.
- [x] Reserved and scaffolded controls are identified.
- [x] Build-time controls are not described as runtime toggles.

## Task 5: Make DFlash Documentation Match Direct and Integrated Behavior

**Files:**
- Modify `docs/dflash.md`.
- Modify DFlash discovery text in `README.md`.
- Modify comments in `examples/dflash_gpt_oss_example.py`.

- [x] Separate experimental direct `DFlashSpeculator.generate` behavior from deprecated `MoE.generate` and serving gates.
- [x] Record batch-1 sampled direct support and greedy-gated integrations.
- [x] Record batch-greater-than-one constraints and the bare HuggingFace target requirement.
- [x] Explain draft, verify, commit, rollback, and cache truncation.
- [x] Explain route-ahead as scheduling-only executor behavior.
- [x] Add per-model route-ahead status: GPT-OSS not wired; Qwen3.5 synthetic/tiny coverage without real-checkpoint serving validation.
- [x] State that executor wiring for DeepSeek, Qwen, and Mixtral does not imply a validated drafter pair.

**Validation:**

```bash
pytest -q tests/python/dflash -m "not gpu"
python - <<'PY'
from pathlib import Path
text = Path('docs/dflash.md').read_text()
assert '| Route-ahead |' in text
assert text.count('Not wired: the GPT-OSS expert path') == 2
assert 'no real-checkpoint serving validation is recorded' in text
assert 'No additional drafter pair is implied.' in text
print('DFlash documentation assertions passed')
PY
```

Checklist:

- [x] No DeepSeek or Qwen drafter pair is invented.
- [x] Sampling scope is explicit.
- [x] Route-ahead metrics and fallback behavior are documented.

## Task 6: Document Serving, Multi-GPU Operation, and Troubleshooting

**Files:**
- Create `docs/serving.md`.
- Create `docs/multi-gpu.md`.
- Create `docs/troubleshooting.md`.

- [x] Document every API server CLI option and default.
- [x] Document completions, chat, streaming, auth, rate limiting, backpressure, and operational endpoints.
- [x] Explicitly route ContextPilot endpoints to `docs/contextpilot/README.md`.
- [x] State that the prefix-cache flag and data structure are not wired into OpenAI request execution.
- [x] Document DFlash serving delegation gates and watchdog behavior.
- [x] Document single-host visible-device ordering, round-robin expert ownership, host staging, model-specific tensor parallelism, and unsupported multi-node operation.
- [x] Add symptom-first troubleshooting entries linked to authoritative guides.

**Validation:**

```bash
python -m moe_infinity.entrypoints.openai.api_server_v2 --help
pytest -q tests/python/serving/test_api_routes.py tests/python/serving/test_cancellation.py tests/python/serving/test_hot_reload.py tests/python/serving/test_prefix_cache.py
pytest -q tests/python/unit/test_watchdog_config.py tests/python/unit/test_startup_watchdog.py tests/python/unit/test_decode_watchdog.py
pytest -q tests/python/unit/test_distributed_smoke.py tests/python/unit/test_multi_gpu.py
```

Checklist:

- [x] Core and optional subsystem routes have explicit ownership.
- [x] Prefix-cache wording is consistent across README, serving, multi-GPU, and troubleshooting.
- [x] No multi-node support is implied.

## Task 7: Rebuild the Contributor Architecture Map

**Files:**
- Modify `ARCHITECTURE.md`.

- [x] Map Python packages and native extension source trees.
- [x] Distinguish deprecated synchronous `MoE.generate` from async continuous serving.
- [x] Diagram DFlash delegation, route-ahead, scheduling, sampling, streaming, and termination.
- [x] Explain request lifecycle and committed-count bookkeeping.
- [x] Define documented package/server surfaces versus internal modules.
- [x] Add a symptom-to-module contributor lookup table.

**Validation:**

```bash
python - <<'PY'
from pathlib import Path
text = Path('ARCHITECTURE.md').read_text()
assert text.count('```mermaid') == 2
assert '## 5. API Stability Boundaries' in text
assert 'Path A - Synchronous' in text
assert 'Path B - Async continuous batching' in text
print('architecture assertions passed')
PY
```

Checklist:

- [x] Mermaid fences are balanced.
- [x] Live integration points are distinguished from contract helpers.
- [x] Internal exports are not elevated into stable public APIs accidentally.

## Task 8: Catalog Benchmarks and Reproduction Workflows

**Files:**
- Modify `docs/benchmarking.md`.
- Modify `benchmarks/expert_io_microbench/README.md`.
- Link existing `docs/benchmark_reproduction.md` and ContextPilot guides.

- [x] Define TTFT, ITL, decode throughput, end-to-end throughput, prefill, decode, and peak memory.
- [x] Catalog serving, ContextPilot, expert I/O, comparison, performance-model, kernel, and evaluation workflows.
- [x] List helpers and package markers as excluded entry points.
- [x] Add concrete serving commands and expert I/O profiler commands.
- [x] Add fairness rules and reproducibility metadata.
- [x] Label incomplete-provenance results as historical rather than canonical.

**Validation:**

```bash
python -m compileall -q benchmarks
bash -n benchmarks/comparison/run_all.sh
for f in benchmarks/expert_io_microbench/run_all.py benchmarks/expert_io_microbench/bench_routing.py benchmarks/expert_io_microbench/bench_transfer.py benchmarks/expert_io_microbench/bench_compute_evict.py benchmarks/expert_io_microbench/bench_bubble.py benchmarks/expert_io_microbench/compare_baseline.py benchmarks/expert_io_microbench/run_decision_profile.py benchmarks/expert_io_microbench/nsys_parser.py; do python "$f" --help >/dev/null; done
```

Checklist:

- [x] Every executable entry point is cataloged or intentionally classified.
- [x] Prerequisites and output metrics are stated.
- [x] DFlash tests are not mislabeled as a standalone benchmark CLI.

## Task 9: Add History, Contribution, Release, and Review Enforcement

**Files:**
- Create `CHANGELOG.md`.
- Modify `CONTRIBUTING.md`.
- Modify `.github/PULL_REQUEST_TEMPLATE.md`.
- Modify `RELEASE.md`.

- [x] Create an `Unreleased` changelog with Added, Changed, Fixed, and Known Limitations.
- [x] Map feature, model, config, architecture, performance, and bug changes to required docs.
- [x] Require evidence fields for performance, support, and model behavior changes.
- [x] Add release checks for links, contradictions, limitations, artifacts, and provenance.
- [x] Keep checks truthful as manual process requirements unless CI actually enforces them.

**Validation:**

```bash
python - <<'PY'
from pathlib import Path
assert '## [Unreleased]' in Path('CHANGELOG.md').read_text()
assert 'Documentation impact by change type' in Path('CONTRIBUTING.md').read_text()
assert '## Documentation impact' in Path('.github/PULL_REQUEST_TEMPLATE.md').read_text()
assert '## Release readiness checklist' in Path('RELEASE.md').read_text()
print('process documentation assertions passed')
PY
```

Checklist:

- [x] Changelog distinguishes history from roadmap.
- [x] PR evidence asks for model, hardware, software, workload, baseline, result, and limitations.
- [x] Release guidance verifies actual wheel or source artifacts before upload.

## Task 10: Recover Artifacts and Perform Cumulative Verification

**Files:**
- Create `docs/superpowers/specs/2026-08-09-repository-documentation-design.md`.
- Create `docs/superpowers/plans/2026-08-09-repository-documentation.md`.
- Reconcile `docs/dflash.md` and `docs/model-compatibility.md`.
- Review every changed documentation file in the exact worktree.

- [x] Recover the approved design as an accurate maintainable record of the implemented architecture.
- [x] Recover this ten-task plan with exact paths, commands, and checklists.
- [x] Reconcile Qwen3.5 serving language: tiny-fixture coverage exists, but no real-checkpoint serving validation is recorded.
- [x] Validate all repository-local Markdown links and practical anchors.
- [x] Validate DFlash and model matrix structure.
- [x] Parse the Qwen3.5 README example as Python syntax.
- [x] Scan for unresolved drafting markers and contradictory prefix-cache or DFlash route-ahead claims.
- [x] Run focused non-GPU assertions and classify environment-only failures.
- [x] Run `git diff --check`, inspect cumulative status, and confirm no runtime implementation edits were introduced.
- [x] Confirm the original checkout remains clean.

**Validation:**

```bash
pytest -q tests/python/unit/test_examples_smoke.py
pytest -q tests/python/dflash -m "not gpu"
git diff --check
git status --short
```

Artifact checks:

```bash
python - <<'PY'
from pathlib import Path
paths = [
    Path('docs/superpowers/specs/2026-08-09-repository-documentation-design.md'),
    Path('docs/superpowers/plans/2026-08-09-repository-documentation.md'),
]
for path in paths:
    text = path.read_text()
    assert text.startswith('# ')
    assert text.endswith('\n')
    assert len(text.splitlines()) >= 100
print({str(path): len(path.read_text().splitlines()) for path in paths})
PY
```

Final checklist:

- [x] Design covers layering, evidence, coverage, enforcement, verification, and risks.
- [x] Plan contains the required agentic-worker header, goal, architecture, tech stack, exact file map, and ten tasks.
- [x] Commands avoid GPU and checkpoint downloads by default.
- [x] No commit or staging action is required or performed.
- [x] Structural verification is not described as reader testing.

## Completion Criteria

The program is complete when:

- all authoritative guides agree on model, DFlash, prefix-cache, serving, and topology limits;
- source-derived fields, options, routes, model families, and benchmark entry points are covered;
- all local Markdown links and practical anchors resolve;
- fenced examples and documented safe commands pass syntax checks;
- focused non-GPU tests pass or any pre-existing/environment fixture failure is reproduced and classified;
- `git diff --check` succeeds;
- the cumulative diff contains documentation, example comments, and model subdirectory README changes only;
- the original checkout remains clean;
- no commit, stage, push, or Git configuration change occurs unless separately requested.
