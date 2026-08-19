# Changelog

All notable changes to MoE-Infinity will be documented in this file.

## [Unreleased]

### Added

- Documentation hub at `docs/README.md` for users, operators, contributors, and project-history readers.
- DFlash documentation that distinguishes direct batch-1 greedy and sampled draft/verify from the greedy-gated `MoE.generate` and serving integrations, explains the current batch>1 greedy-only constraint, and limits continuous-batching and route-ahead claims to validated paths.

### Changed

- Qwen3.5-MoE keeps its text backbone resident and offloads routed experts only on the text-only path.
- GLM-5.2-FP8 keeps routed FP8 experts in the host store while non-routed FP8 weights dequantize to BF16 on load.
- Root README is now a concise discovery surface and points readers to the docs hub, model compatibility, DFlash, serving, troubleshooting, architecture, and changelog.
- Release notes are split out of README and tracked here instead of being presented as shipped releases.
- Package version is now derived from git tags by setuptools-scm and written to `moe_infinity/_version.py` at build time, replacing the manual `MOEINF_VERSION`, `NIGHTLY_BASE_VERSION`, and hardcoded `setup.py`/`__init__.py` version strings.

### Deprecated

- `MoE.generate()` emits `DeprecationWarning` in the current code and is scheduled for removal. It remains documented as the current in-process synchronous transition path; `MoE.serve()` is recommended for continuous-batching HTTP serving and is not a drop-in in-process return API. No removal version is announced.

### Fixed

- Serving-path DFlash now truncates KV cache to the committed prefix after each verify step, so emitted and cached tokens stay aligned.
- GPT-OSS resident-load path now materializes MXFP4 blocks, scales, router, biases, and attention sinks instead of leaving placeholder tensors in place.
- GLM FP8 store and reload parity now stays stable across fresh stores and reloads.
- PyPI publishing for both stable (`publish.yml`) and nightly (`publish-test.yml`): stable releases now take their version from the pushed git tag instead of always publishing `0.0.1`, and nightly sdists carry their version in `PKG-INFO` so `pip install --pre moe-infinity` no longer fails with a `MetadataInconsistent` version mismatch on rebuild.

### Known Limitations

- Batch > 1 DFlash is greedy-only, requires a bare HuggingFace target, and sampled batch > 1 remains unsupported.
- Multi-node distributed inference is still unsupported.
