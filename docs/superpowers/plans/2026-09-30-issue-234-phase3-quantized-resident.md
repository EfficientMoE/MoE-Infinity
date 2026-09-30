# Issue 234 Phase 3 — quantized-resident experts

Draft for review. This does not add kernels, allowlist entries, or a serving path. #234 says this phase starts only after its own decision checkpoint, and the checkpoint is not done.

**Goal:** Keep experts quantized in GPU cache and execute them with a fused dequant-matmul kernel, so a fixed HBM budget holds more experts than bf16 residency. Importance-aware mixed precision (hot bf16 / warm FP8 / cold INT4) is a follow-on inside the same phase, not part of the first patch.

**Relationship to merged work:**

- Phase 1 (#235, moe-store#13) quantizes the *store and the H2D copy*, then dequantizes to bf16 at onboarding. GPU-resident experts are still bf16.
- #184 shipped `ExpertResidencyManager` variant keys `(format, generation)`, converters, and an empty `RELEASED_ADAPTIVE_ENTRIES` allowlist. Serving stays on the canonical format until a reviewed entry is added. This phase is "enable and extend #184", not a second residency system.

## Decision checkpoint (not yet accepted)

Do not write kernel or allowlist code until these are answered in a comment on this PR:

1. **Kernel.** One uniform precision per grouped dispatch. Mixed tiers mean split dispatches. Measure that split before choosing per-expert versus per-layer granularity.
2. **Where precision changes.** Time one-expert FP8 and INT4 quantization on GPU and on CPU. If that cost exceeds a cache-miss fetch, tier changes happen at load time only.
3. **First tier.** FP8 resident experts via the existing blockwise path, or INT4/HQQ. The issue's capacity target is 4–8× versus bf16. FP8 alone is about 2× and is the smaller step; INT4 is the step that can hit 4×.
4. **Policy inputs.** Tracer activation frequency plus router-norm sensitivity. HOBBIT is a reference, not code to port.

Recommended first patch, if the checkpoint agrees: load-time FP8 residency for one allowlisted checkpoint, uniform precision per dispatch, no online tier changes, no INT4. Hot/warm/cold mixed precision waits on the split-dispatch measurement.

## File map (after the checkpoint)

| File | Change |
|---|---|
| `moe_infinity/runtime/adaptive_precision_allowlist.py` | One `ReleasedAdaptiveEntry` only, with fingerprint, format, converter version, and attestation digest from a measured run. |
| `moe_infinity/runtime/expert_precision.py` | Reuse format selection. No second converter stack. |
| Grouped-GEMM dispatch | Split by precision tier only if the measurement says the split is cheaper than uniform precision. |
| `benchmarks/adaptive_precision/` | Release-gate numbers that justify the allowlist row. |

## QA gate before any allowlist row

- Default off: empty allowlist still refuses unlisted checkpoints, and adaptive precision unset leaves residency bf16.
- One recorded model: resident-expert count at a fixed HBM budget versus the bf16 store, plus perplexity delta versus bf16.
- Kernel matrix note in the PR: bit-width, group size, and architecture floor for the chosen kernel. MXFP8 and the MiniMax-M3 MXFP8 store stay deferred (the rest of #221).

## Non-goals

Activation quantization, KV-cache quantization, permanent pruning, expert dropping (Phase 2), and adding an allowlist row without the attestation digest.
