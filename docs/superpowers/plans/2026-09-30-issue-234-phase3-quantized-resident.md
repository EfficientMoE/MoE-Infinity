# Issue 234 Phase 3 — uniform FP8 residency

This is the spec for the first patch. The task list is `docs/superpowers/plans/2026-09-30-uniform-fp8-residency.md`. Mixed hot/warm/cold precision and INT4 are follow-ons, not this patch.

**Goal:** When adaptive precision is on and a released manifest is uniformly FP8, every routed expert that resides in GPU cache stays `fp8_e4m3_block128` for the life of the process. A fixed HBM budget therefore holds more experts than bf16 residency. Default serving stays bf16.

**Relationship to merged work:**

- Phase 1 (#235, moe-store#13) quantizes the store and the H2D copy. Routed FP8 weights already stay FP8 in the host store and dequantize on device through `fp8_dequant_bf16_gemm`. This patch does not add a kernel.
- #184 shipped variant keys, converters, and an empty `RELEASED_ADAPTIVE_ENTRIES` allowlist. This patch does not add a second cache and does not insert an allowlist row.

## Pinned checkpoint

1. **Kernel.** `ExpertFormat.FP8_E4M3_BLOCK128`, execution `fp8_dequant_bf16_gemm`, block size 128, output bfloat16, no new architecture floor, no `requires_extension`. One format per dispatch. This patch does not split a dispatch by precision.
2. **When precision changes.** Load time only. `uniform_fp8` sets targets once and creates no transitions and no evictions. It does not call `AdaptivePrecisionPolicy`.
3. **First tier.** FP8, not INT4. The capacity gate is `resident_fp8 >= 1.8 * resident_bf16` at the same HBM budget, the same bar as Phase 1. The 4–8× INT4 target is out of scope.
4. **Policy inputs.** Tracer frequency and router-norm sensitivity are not read. HOBBIT is not ported.

## Config

- `adaptive_expert_precision` stays default `False`. Unset means residency stays bf16.
- `adaptive_resident_mode` is `"legacy"` (default) or `"uniform_fp8"`.
- `"legacy"` keeps today's online policy once a manifest is approved.
- `"uniform_fp8"` is not an approval bypass. An unreleased manifest still resolves as `manifest_unapproved`.
- `RELEASED_ADAPTIVE_ENTRIES` stays empty. A row requires a later reviewed change that carries a measured fingerprint, converter version, and attestation digest.

## Quality gate

- Relative NLL `(nll_fp8 - nll_bf16) / nll_bf16 <= 0.01`.
- Resident-expert count at a fixed budget satisfies `resident_fp8 >= 1.8 * resident_bf16`.
- The harness records both and does not download weights in CI.

## Non-goals

Activation quantization, KV-cache quantization, permanent pruning, expert dropping, Marlin INT4, MXFP8, the MiniMax-M3 MXFP8 store, split dispatch, online tier changes, and any allowlist row.
