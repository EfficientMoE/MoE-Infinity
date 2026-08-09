# Repository Documentation Design

**Date:** 2026-08-09  
**Status:** Approved and implemented  
**Scope:** Repository-wide documentation architecture, evidence standards, and maintenance controls

## 1. Purpose

MoE-Infinity needs documentation that helps four audiences without forcing each
reader through implementation details:

- users selecting a model and running the Python API;
- operators deploying the OpenAI-compatible server;
- contributors changing model, runtime, benchmark, or release behavior;
- maintainers reviewing project history and release readiness.

The design replaces a single overloaded README with a layered system. The root
README remains the discovery and quick-start surface. A documentation hub routes
readers to focused guides. Architecture and changelog files carry contributor
and project-history concerns. Every capability claim is scoped to code and test
evidence rather than inferred from a registry entry or an implementation seam.

## 2. Design Principles

### 2.1 Layer information by reader intent

The documentation has four layers:

1. **Discovery:** `README.md` identifies the project, lists major capabilities,
   gives minimal installation and usage examples, and links authoritative guides.
2. **Navigation:** `docs/README.md` groups focused guides by audience.
3. **Authority:** focused pages document configuration, environment variables,
   model compatibility, serving, DFlash, multi-GPU operation, benchmarking, and
   troubleshooting.
4. **Maintenance and history:** `ARCHITECTURE.md`, `CHANGELOG.md`,
   `CONTRIBUTING.md`, `RELEASE.md`, and the PR template define boundaries,
   history, and update expectations.

The same detailed subject should not be independently specified in several
places. Secondary pages summarize and link to the authoritative guide.

### 2.2 Prefer scoped claims over blanket support language

Registration, implementation, fixture coverage, real-checkpoint validation, and
production support are separate states. Documentation must say which state is
supported by evidence. Examples:

- a model registry entry does not prove end-to-end generation or serving;
- a tiny CPU fixture does not prove a real checkpoint or GPU topology;
- an executor hook does not establish a validated DFlash target/drafter pair;
- a feature flag and cache class do not establish an active request-path feature;
- historical benchmark numbers are not current performance guarantees.

### 2.3 Keep operational truth close to source

Source-derived references use exact names and defaults from implementation:

- `ArcherConfig` fields come from `moe_infinity/utils/config.py`;
- runtime and build environment keys come from production readers and build files;
- server options come from the `api_server_v2.py` argument parser;
- HTTP endpoints come from FastAPI route decorators;
- model families come from `moe_infinity/common/constants.py` and adapter wiring;
- benchmark entry points come from executable files under `benchmarks/`.

When code has scaffolding without a live integration, the guide names that limit.

## 3. Information Architecture

### 3.1 Root README

`README.md` owns:

- project overview and core value proposition;
- concise feature discovery with limitations linked inline;
- supported-model discovery table;
- installation prerequisites and source build commands;
- minimal Python examples, including model-specific examples that are safe to show;
- links to configuration, serving, DFlash, benchmarking, architecture, and history.

The README does not duplicate complete configuration tables, server route tables,
benchmark catalogs, or model validation matrices.

### 3.2 Documentation hub

`docs/README.md` is the stable index. It groups links under Users, Operators,
Contributors, and Project History. New focused guides must be linked from the hub
and, when user-facing, discovered from the root README.

### 3.3 Focused guides

The authoritative focused guides are:

| Guide | Responsibility |
| --- | --- |
| `docs/model-compatibility.md` | Registry families, maturity, checkpoint examples, versions, validation topology, and limitations. |
| `docs/configuration.md` | `ArcherConfig` fields, defaults, memory-ratio rules, store layout, and deprecated inputs. |
| `docs/environment-variables.md` | Production runtime, serving, acceleration, profiling, model, build, and packaging environment keys. |
| `docs/dflash.md` | Direct and integrated DFlash behavior, sampling and batch constraints, serving delegation, rollback, route-ahead, and per-model evidence. |
| `docs/serving.md` | CLI, Python startup, request fields, streaming, auth, backpressure, inactive prefix-cache scaffolding, DFlash gates, endpoints, and watchdogs. |
| `docs/multi-gpu.md` | Single-host topology, device numbering, expert ownership, transfers, model-specific tensor parallelism, and unsupported multi-node scope. |
| `docs/troubleshooting.md` | Symptom-first diagnosis tied to authoritative guides and focused tests. |
| `docs/benchmarking.md` | Metrics, methodology, benchmark catalog, exclusions, commands, and reproducibility metadata. |
| `docs/glm-5.2.md` | GLM-specific registration, FP8 residency, serving, speculative limitations, and validation. |
| `moe_infinity/models/deepseek_v4/README.md` | DeepSeek-V4 official loader, checkpoint conversion, FP4 offload, validation, and historical measurements. |

Existing ContextPilot, benchmark-reproduction, and C++ interface pages remain
specialized references linked from the hub or benchmarking guide.

### 3.4 Contributor architecture

`ARCHITECTURE.md` maps modules, native extensions, synchronous and asynchronous
execution paths, request lifecycle, DFlash integration, route-ahead behavior,
and API stability boundaries. It distinguishes documented package/server surfaces
from internal modules. Diagrams explain flow; prose records caveats that diagrams
cannot express safely.

### 3.5 Changelog and process documents

`CHANGELOG.md` separates shipped history from the `Unreleased` section and records
user-visible additions, changes, fixes, and known limitations. `RELEASE.md`
requires changelog closure, artifact inspection, link checks, support review, and
performance provenance. `CONTRIBUTING.md` and `.github/PULL_REQUEST_TEMPLATE.md`
make documentation impact and evidence part of normal review.

## 4. Evidence Policy

### 4.1 Evidence levels

Capability descriptions use the following ladder:

1. **Registered:** a family or option is recognized by source code.
2. **Implemented/experimental:** a path exists and has unit or synthetic coverage.
3. **Validated on tiny fixtures:** behavior is exercised without a real checkpoint.
4. **Validated on a real checkpoint:** a repository harness records that checkpoint.
5. **Validated topology:** hardware or process layout is explicitly recorded.
6. **Unsupported:** no runtime path exists or a fail-fast guard blocks the path.

The strongest phrase used in a guide must not exceed the strongest repository
evidence. Missing evidence is written as “not recorded” or “not validated,” not
silently converted into support.

### 4.2 Model and DFlash evidence

The model matrix covers every registry family, including conditional families and
registry-only unsupported entries. DFlash documentation separately records target
and drafter pairings. Route-ahead is described as an executor-path capability;
it does not create or imply a checkpoint pairing. GPT-OSS explicitly lacks the
executor wiring. Qwen3.5 serving and route-ahead may cite tiny/synthetic fixtures,
while stating that no real-checkpoint serving validation is recorded.

### 4.3 Performance evidence

Performance claims require model/checkpoint, hardware, software, workload,
baseline, metric definition, and limitations. Measurements without complete
provenance are labeled historical snapshots. TTFT, inter-token latency, decode
throughput, and end-to-end throughput remain distinct.

## 5. Coverage and Cross-Linking

The documentation must cover these source surfaces:

- all active `ArcherConfig` fields and defaults;
- all project-owned production environment variables and build controls;
- all server parser options and route decorators, with explicit exclusions for
  optional subsystem routes documented elsewhere;
- every registered model family and conditional-import caveat;
- every executable benchmark entry point or an explicit helper/exclusion class;
- all documented local scripts, tests, files, and anchors;
- shipped limitations for batch size, sampling, serving, topology, and inactive
  scaffolding.

Cross-links follow authority:

- README summaries link to focused guides;
- troubleshooting entries link to the guide that owns the behavior;
- model-specific pages link back to the compatibility matrix;
- benchmarking pages link to reproduction or subsystem runbooks;
- architecture links to serving and DFlash when user-facing contracts matter.

## 6. Enforcement

Documentation maintenance is enforced socially and mechanically:

- `CONTRIBUTING.md` maps change types to required documents;
- the PR template asks for README discovery, authoritative-guide updates,
  compatibility/architecture/benchmark impact, changelog impact, and evidence;
- `RELEASE.md` requires a final contradiction and link review;
- focused tests verify parser, config, model registry, serving, watchdog,
  distributed, and DFlash behavior;
- lightweight scripts validate Markdown links, anchors, table shape, examples,
  stale claims, and local paths.

Commits are optional execution checkpoints, not a documentation requirement.
No commits were created during the execution that recovered this artifact.

## 7. Verification Strategy

Verification is layered to avoid claiming more than the environment proves:

1. Run local Markdown link and anchor checks across repository docs.
2. Parse fenced Python examples with `ast.parse` and compile examples/benchmarks.
3. Check Markdown table column counts for compatibility and catalog tables.
4. Compare code-derived surfaces against their authoritative pages.
5. Run safe `--help` commands for documented CLIs.
6. Run `pytest -q tests/python/dflash -m "not gpu"` and focused non-GPU suites.
7. Run `git diff --check`, inspect `git status --short`, scan for secrets, and
   confirm documentation-only scope.
8. Record unavailable Markdown LSP or hardware-dependent checks as limitations.

GPU and model-download tests are never implied by a CPU-only run. Reader testing
is a separate review activity and is not claimed by structural verification.

## 8. Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| README and focused guide drift | Keep detailed truth in one focused guide and link from summaries. |
| Registry mistaken for support | Use the evidence ladder and explicit matrix limitations. |
| Fixture coverage overstated | Name tiny/synthetic coverage and separately state real-checkpoint status. |
| Scaffolding presented as active | Say whether request execution actually consumes the feature. |
| Historical numbers read as guarantees | Require provenance or label the data historical. |
| Local paths and anchors rot | Run repository-wide link, anchor, and path checks. |
| Generated or environment failures obscure docs defects | Diagnose and classify fixture, dependency, GPU, and model-download failures. |
| Process docs become ceremonial | Tie each checklist item to a specific authoritative file or validation command. |

## 9. Maintenance Rules

- Update the focused guide and changelog when user-visible behavior changes.
- Update README discovery only when the entry point or major capability changes.
- Update architecture when module ownership, execution flow, or API boundaries change.
- Add model rows only with explicit evidence and limitations.
- Add benchmark rows with prerequisites, outputs, status, and exclusions.
- Preserve explicit inactive-feature wording until a live integration and tests exist.
- Re-run structural and focused tests before release or major documentation review.

This design records the implemented documentation architecture. It is intended to
remain maintainable as source behavior changes, not to preserve historical wording.
