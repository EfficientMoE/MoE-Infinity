# Contributing to MoE-Infinity

Thank you for contributing to **MoE-Infinity**, a cost-effective MoE inference library for memory-constrained GPUs.

We welcome contributions across Python, C++/CUDA extensions, tests, docs, and examples.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
  - [Report Bugs](#report-bugs)
  - [Suggest Features](#suggest-features)
  - [Roadmap and Task Selection](#roadmap-and-task-selection)
- [Development Setup](#development-setup)
- [Code Style, Linting, and Formatting](#code-style-linting-and-formatting)
- [Running Tests](#running-tests)
- [Pull Request Process (Fork-and-Pull)](#pull-request-process-fork-and-pull)
  - [Documentation impact by change type](#documentation-impact-by-change-type)
- [Commit Message Convention](#commit-message-convention)
- [Contribution Tips for This Repo](#contribution-tips-for-this-repo)

## Code of Conduct

Please review and follow our [Code of Conduct](./CODE_OF_CONDUCT.md) in all interactions.

## How to Contribute

You can contribute by:

- Fixing bugs in Python or C++/CUDA code paths
- Improving performance/memory behavior for MoE inference
- Adding tests, examples, or documentation improvements
- Proposing and implementing new features

### Report Bugs

Use the GitHub **Bug Report** issue template:

- https://github.com/EfficientMoE/MoE-Infinity/issues/new?template=bug_report.yml

Before filing:

- Search existing issues: https://github.com/EfficientMoE/MoE-Infinity/issues
- Include environment details (OS, Python version, GPU, CUDA, PyTorch)
- Provide a minimal reproduction script/config and expected vs actual behavior

For security vulnerabilities, **do not** open a public issue. Use [SECURITY.md](./SECURITY.md).

### Suggest Features

Use the GitHub **Feature Request** template:

- https://github.com/EfficientMoE/MoE-Infinity/issues/new?template=feature_request.yml

Please include:

- Problem statement and why it matters for MoE inference users
- Proposed API or behavior changes
- Alternatives considered and expected trade-offs

For substantial changes (for example, new runtime behavior or model-family support), open an issue first to align on scope before implementing.

### Roadmap and Task Selection

- Start from the issue tracker: https://github.com/EfficientMoE/MoE-Infinity/issues
- New contributors should prioritize smaller scoped issues first.
- If available, issues labeled `good first issue` or `help wanted` are good entry points.

## Development Setup

MoE-Infinity builds Python and CUDA/C++ components. We recommend a fresh conda environment.

- Supported Python range in packaging metadata: `>=3.10` (some required dependencies, e.g. `sglang-kernel`, ship wheels for Python ≥ 3.10 only)
- Recommended local development version: Python `3.12`

```bash
git clone https://github.com/EfficientMoE/MoE-Infinity.git
cd MoE-Infinity

conda create -n moe-infinity-dev python=3.12
conda activate moe-infinity-dev

pip install -e .
conda install -c conda-forge libstdcxx-ng=12
```

Optional performance dependency:

```bash
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn
```

## Code Style, Linting, and Formatting

MoE-Infinity uses pre-commit hooks configured in [`.pre-commit-config.yaml`](./.pre-commit-config.yaml).

```bash
pip install -r requirements-lint.txt
pre-commit install --install-hooks
pre-commit run --all-files
```

`pre-commit install` registers both the **pre-commit** and **pre-push** hooks (configured via `default_install_hook_types`). The formatters therefore run again at `git push` time, so any drift that slipped past a commit is caught locally before it reaches CI.

Current lint/format stack includes:

- `ruff` + `ruff-format`
- `mypy` (configured via `pyproject.toml`)
- `clang-format` (for C++/CUDA sources)
- `codespell`

The pinned tool versions in `.pre-commit-config.yaml` are the source of truth. CI runs the exact same versions, so always run formatting through `pre-commit` (which installs those pinned versions in isolated environments) rather than a system-wide `ruff` or `clang-format`, whose version may differ and produce mismatched formatting.

Please run formatting/lint checks before opening a PR.

## Running Tests

We keep tests under `tests/` with Python suites and Docker/integration coverage.

Current Python test layout:

- `tests/python/unit/`: unit tests for core runtime and utilities
- `tests/python/integration/`: integration tests (including OpenAI-compatible API tests)
- `tests/python/serving/`: serving engine, scheduler, streaming, and cache tests
- `tests/python/ops/`: kernel/operator correctness tests (paged attention, fused ops, routing)
- `tests/python/e2e/`: end-to-end KV/offloading and serving scenarios

Recommended full local test command:

```bash
python tests/docker/run_tests.py
```

Useful targeted commands:

```bash
# Unit tests
python -m pytest -v --tb=short tests/python/unit/

# Integration tests
python -m pytest -v --tb=short tests/python/integration/

# Serving tests
python -m pytest -v --tb=short tests/python/serving/

# Operator/kernel tests
python -m pytest -v --tb=short tests/python/ops/

# End-to-end tests
python -m pytest -v --tb=short tests/python/e2e/

# Integration tests without CUDA
python -m pytest -v --tb=short -m "not cuda" tests/docker/test_io_integration.py

# CUDA-specific integration tests (when CUDA is available)
python -m pytest -v --tb=short -m "cuda" tests/docker/test_io_integration.py
```

If you change C++/CUDA logic, please run both unit and CUDA integration paths when hardware is available.

If your environment cannot run CUDA tests, call that out explicitly in your PR description so maintainers can run GPU validation.

## Pull Request Process (Fork-and-Pull)

We follow a standard fork-and-pull workflow:

1. Fork `EfficientMoE/MoE-Infinity`
2. Create a feature branch from `main`
3. Implement changes with tests/docs updates
4. Run local checks (`pre-commit run --all-files`, tests)
5. Open a PR against `main`

When opening a PR:

- Follow the repository PR template: [`.github/PULL_REQUEST_TEMPLATE.md`](./.github/PULL_REQUEST_TEMPLATE.md)
- Link related issues (for example, `Closes #123`)
- Explain impact on performance, memory usage, or compatibility when relevant

When applicable:

- Bug fix PRs should include a regression test.
- Feature PRs should include tests and an example or documentation update.
- Performance PRs should include benchmark context (hardware, model, and before/after observations).

### Documentation impact by change type

A change can match more than one row. Satisfy every row that applies before opening the PR, and update the authoritative guide for each surface you changed. Update the [Documentation hub](./docs/README.md) only when adding, removing, or renaming a guide, or when changing documentation navigation.

| Change type | Required docs | Concrete examples |
| --- | --- | --- |
| New user feature | [README](./README.md) discovery, the feature guide or API page, and [Changelog](./CHANGELOG.md) `Unreleased`. | Add a new section to `docs/serving.md`, link it from `README.md`, and add an `Unreleased` note. |
| Model support | [Model compatibility matrix](./docs/model-compatibility.md), the model family guide, and [Changelog](./CHANGELOG.md) `Unreleased`. | Add the checkpoint row in `docs/model-compatibility.md`, then update `moe_infinity/models/deepseek_v4/README.md` or `docs/glm-5.2.md`. |
| Config, CLI, or env | The config or env guide, [README](./README.md) discovery if the quick start changes, and [Changelog](./CHANGELOG.md) `Unreleased`. | Update `docs/configuration.md`, `docs/environment-variables.md`, and the matching README command. |
| Architecture | [Architecture](./ARCHITECTURE.md) and [Changelog](./CHANGELOG.md) `Unreleased`. | Update the module map, request lifecycle, or public API boundary notes. |
| Performance | [Benchmarking guide](./docs/benchmarking.md), the relevant reproduction or runbook page, and [Changelog](./CHANGELOG.md) `Unreleased`. | Add benchmark data with model, checkpoint, hardware, software, workload, baseline, and limitations. |
| User-visible bug | The guide for the affected surface and [Changelog](./CHANGELOG.md) `Unreleased`. Add [README](./README.md) discovery when the fix changes the user entry point. | Update troubleshooting, serving, or model docs if the fix changes what users must do or expect. |

If a change spans more than one row, satisfy every matching row. Do not rely on the README alone when a deeper guide exists.

## Commit Message Convention

Use Angular-style commit messages:

```text
<type>(optional-scope): <summary>

<body>
```

Recommended types:

- `feat`: new user-facing capability
- `fix`: bug fix
- `perf`: performance improvement
- `refactor`: internal restructuring without behavior change
- `test`: test-only updates
- `docs`: documentation changes
- `ci`: CI workflow changes
- `build`: build/dependency/toolchain changes
- `chore`: maintenance tasks

For non-trivial changes, include a short body explaining the rationale and impact (recommended minimum: one meaningful sentence).

Recommended for provenance tracking in open-source workflows:

```bash
git commit -s -m "feat(scope): concise summary"
```

Examples:

```text
feat(offload): add activation-aware prefetch tuning knob
fix(cuda): guard pinned-memory path for missing stream sync
docs(contributing): clarify local test matrix
```

## Contribution Tips for This Repo

- Keep HuggingFace compatibility intact when changing model-loading paths
- Call out any behavior differences between single-GPU and multi-GPU inference
- Include benchmark or latency notes when contributing performance-sensitive changes

Thanks for helping improve MoE-Infinity.
