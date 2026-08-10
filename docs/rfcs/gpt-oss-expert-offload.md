# RFC: GPT-OSS Expert Offloading Backend

**Status:** Draft / Design-only (no implementation in this PR)
**Author:** (dflash working track)
**Related:** PR #84 (GPT-OSS support), `fix/gpt-oss-resident-experts` (resident-load correctness fix), PR #135 (native DFlash)

## Summary

GPT-OSS (`openai/gpt-oss-*`) currently runs in MoE-Infinity with **all expert
weights resident** (~64 GB of MXFP4 for `gpt-oss-120b`). It is the only supported
MoE architecture that is **excluded from the Archer expert-offload dispatcher**,
so it does not benefit from host/SSD expert offloading, activation-aware caching,
or just-in-time prefetch — the core value proposition of MoE-Infinity. This RFC
proposes wiring GPT-OSS into the dispatcher so its experts can be **offloaded and
prefetched** like Mixtral / DeepSeek / GLM, making `gpt-oss-120b` runnable on
memory-constrained GPUs while preserving output correctness.

## Background & Current State

GPT-OSS packs all experts of a layer into batched tensors on a single
`_PackedExperts` module (`moe_infinity/models/gpt_oss.py`):

- `experts.gate_up_proj_blocks` `[E=128, N=5760, K//2=1440]` uint8 (MXFP4)
- `experts.gate_up_proj_scales` `[E, N, K//32=90]` uint8
- `experts.{gate_up,down}_proj_bias` bf16
- `experts.down_proj_blocks` `[E, 2880, 1440]` uint8, `down_proj_scales` `[E, 2880, 90]`

Every other MoE arch registers **per-expert** tensors with the C++ dispatcher
(`register_expert`) and fetches them just-in-time. GPT-OSS is explicitly excluded
in three places (`moe_infinity/runtime/model_offload.py`):

1. `setup_archer_hooks` — `if not isinstance(module, SyncGptOssMLP): module.expert_executor = ...`
   → `SyncGptOssMLP.expert_executor` stays `None`.
2. `register_expert` loop — `if "expert" in key and self.config.model_type != "gpt_oss":`
   → GPT-OSS experts are never registered with the dispatcher.
3. `parse_expert_id` (`moe_infinity/utils/hf_config.py`) returns `(layer_id, None)`
   for GPT-OSS packed keys → `expert_tensor_map` never gets per-expert entries.

Consequently `SyncGptOssMLP.forward` takes the resident Python `else` branch
(`for expert_idx in range(num_experts): self._expert_forward(...)`), reading the
`_PackedExperts` params directly.

**Correctness prerequisite (already fixed):** the loader never materialized those
resident params (they were left as Archer `[1]` zero placeholders → zeros / NaN /
garbage → incoherent output). `fix/gpt-oss-resident-experts` adds
`_load_resident_gpt_oss` to load the real MXFP4 blocks/scales, biases, `router`,
and `self_attn.sinks` into the live params, and corrects the MXFP4 packed layout
(`[N, K//2]`, no transpose). Validated on real `gpt-oss-120b`: coherent output;
DFlash agreement `0.95 → 1.00`, mean acceptance `1.0 → 6.57`, ~21× decode speedup.
That fix makes GPT-OSS **correct but fully resident**. This RFC is the follow-on to
make it **offloadable**.

## Goals / Non-Goals

**Goals**
- Offload GPT-OSS routed experts to host memory (and optionally SSD), with
  activation-aware caching and JIT prefetch, via the existing Archer dispatcher.
- Make resident-vs-offload a tunable (`device_memory_ratio`), not hard-coded.
- Preserve bit-for-bit correctness parity with the resident path (agreement gate,
  MXFP4 kernel numerics unchanged).
- Keep the resident path as the default/fallback for GPUs with ample VRAM.

**Non-Goals**
- Changing DFlash speculative decoding (orthogonal; benefits automatically).
- Multi-node distributed experts.
- Sampled (non-greedy) speculative decoding.

## The Three Incompatibilities (root of the work)

| # | Area | Today (GPT-OSS) | Needed |
|---|------|-----------------|--------|
| 1 | `parse_expert_id` / `get_topology` | returns `expert_id=None`; no per-expert nodes | emit 128 per-expert identities per layer from the packed tensors |
| 2 | Expert execution in dispatcher | only NLLB/Mixtral/DeepSeek/GLM styles; no MXFP4 packed path | MXFP4 expert execution (dequant-on-copy or native uint8 kernel) |
| 3 | `SyncGptOssMLP` wiring | `expert_executor=None`, resident loop | dispatch_local / wait_dispatch_local wired, gated by tunable |

## Proposed Design (phased)

### Phase 1 — Per-expert identity for packed tensors
Give the dispatcher a per-expert view over the packed `[E, ...]` tensors:
- Extend `parse_expert_id` to yield `(layer_id, expert_idx)` for GPT-OSS by
  treating each slice `blocks[e]` / `scales[e]` / `bias[e]` as an expert tensor.
- Extend `get_topology` / the expert registration loop to register 128 per-expert
  entries per layer (blocks+scales+bias grouped), populating `expert_tensor_map`.
- Store layout stays MXFP4 output-major (`[N, K//2]` / `[N, K//32]`) per the
  correctness fix; per-expert slices are contiguous views.

### Phase 2 — MXFP4 expert execution in the dispatcher
Two options (recommend implementing A first for parity, then B for memory):

- **Option A — dequant-on-copy to bf16 (parity-first).** Mirror the GLM FP8
  path (`SetScales` + dequant-on-copy): keep MXFP4 in the host store, dequantize
  each fetched expert to bf16 on device before the GEMM, reuse the existing bf16
  expert execution. Host store ≈ 64 GB; device holds a small bf16 working set.
  Lowest risk; reuses proven machinery.
- **Option B — native MXFP4 expert kernel (memory-first).** Keep uint8 blocks
  in the store (≈ 64 GB host) and run the existing Triton `fused_mxfp4_gemm`
  (already verified, rel-err ~0.0015) — or a C++ equivalent — directly on the
  fetched packed weights. Smallest footprint; requires the dispatcher to carry
  blocks+scales together and call the MXFP4 path.

### Phase 3 — Wire `SyncGptOssMLP` + tunability
- When offload is enabled, inject `expert_executor` / `expert_prefetcher` into
  `SyncGptOssMLP` and route through `dispatch_local` / `wait_dispatch_local`.
- Make resident-vs-offload driven by `device_memory_ratio` (resident when VRAM
  is ample; offload when constrained) — consistent with the "Reconciliation
  Contract" tunable-offload direction already used elsewhere.
- Keep `_load_resident_gpt_oss` as the resident/fallback branch.

## Memory Analysis (`gpt-oss-120b`, 128 experts × 36 layers)

| Strategy | Host store | Resident GPU | Notes |
|---|---|---|---|
| Resident MXFP4 (current fix) | – | ~64 GB | correct today; needs a big GPU |
| Offload MXFP4, kernel-on-fetch (B) | ~64 GB | small cache | best memory; native MXFP4 exec |
| Offload bf16 dequant-on-copy (A) | ~218 GB | small cache | simplest; reuses GLM-style path |

## Validation Plan
- **Correctness:** reuse the GPU-gated 120B harness (`tests/.../test_gpu_120b.py`,
  `MOE_DFLASH_GPU`): base coherence + greedy agreement vs resident path (MXFP4
  agreement gate, not string identity).
- **DFlash parity:** agreement `≈1.0`, mean acceptance unchanged vs resident.
- **Memory:** measure resident GPU footprint at low `device_memory_ratio`; assert
  `gpt-oss-120b` runs under a target VRAM budget.
- **Regression:** existing `tests/test_gpt_oss_*.py`, `tests/test_mxfp4*.py`,
  and the 103 DFlash unit tests remain green.

## Risks & Mitigations
- **Layout/dequant drift** → lock the verified MXFP4 layout; golden-tensor test
  vs `mxfp4_dequantize` reference (rel-err 0.0 gate on `gate_up`).
- **Dispatcher assumes per-expert modules** → Phase 1 adapter maps packed slices
  to per-expert IDs without copying.
- **Prefetch/caching correctness under sliding-window + DFlash** → validate with
  the DFlash rollback tests already in place.
- **Scope creep** → Option A (dequant-on-copy) is a smaller first milestone that
  delivers offloading; Option B is a follow-up optimization.

## Alternatives Considered
- **Keep fully resident (status quo + fix):** correct, but excludes
  memory-constrained GPUs — contrary to MoE-Infinity's purpose.
- **Serve GPT-OSS via SGLang:** already the team's throughput path; does not give
  MoE-Infinity native offload parity.

## References (file:line)
- `moe_infinity/runtime/model_offload.py`: dispatcher exclusion (`register_expert`
  loop `model_type != "gpt_oss"`), `setup_archer_hooks` guard, `_load_resident_gpt_oss`.
- `moe_infinity/models/gpt_oss.py`: `SyncGptOssMLP`, `_PackedExperts`,
  `_expert_forward_mxfp4`.
- `moe_infinity/utils/hf_config.py`: `parse_expert_id` (GPT-OSS → `None`).
- `moe_infinity/kernel/mxfp4_gemm.py`: `fused_mxfp4_gemm`, `mxfp4_dequantize`.
- GLM FP8-in-store precedent: `deliver_fp8_scales_to_dispatcher`,
  `dequant_fp8_blockwise` (dequant-on-copy pattern to mirror).
