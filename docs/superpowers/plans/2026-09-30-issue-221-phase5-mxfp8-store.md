# Issue 221 Phase 5 — MiniMax-M3 MXFP8 expert host store

Draft for review. This does not dequantize a checkpoint. Phase 4's BF16 eager log is the numeric baseline this phase is measured against. #234 Phase 3 (quantized GPU residency) stays a different change.

**Goal:** Load checkpoint-native MXFP8 routed experts from the host store and dequantize them to bf16 at onboarding. Shared expert, vision tower, and multimodal projector stay resident in the dtype the checkpoint stored.

## Decision checkpoint (not yet accepted)

Do not write a dequantizer until these are answered on this PR:

1. **Predicate bug.** `_has_fp8_blockwise` in `moe_infinity/runtime/model_offload.py` is true when `quant_method` contains the substring `fp8`. A config value of `mxfp8` therefore enters the 128×128 blockwise path (`_scale_inv` pairs, fp32 scales) used by GLM-5.2, DeepSeek-V3, and #235. The first patch excludes `mxfp8` from that predicate, with a CPU test, before any MXFP8 tensor is read.
2. **On-disk layout.** The Sync block consumes unpacked `experts.{id}.{gate,up,down}_proj.weight`. The Hugging Face module state dict is packed `experts.gate_up_proj` / `experts.down_proj`. Record the real safetensors key names, scale tensor names, scale dtype, and group size from `MiniMaxAI/MiniMax-M3` (or the transformers converter for that revision) before choosing a kernel. OCP MX group-32 E8M0 is a hypothesis, not the layout.
3. **Which tensors.** Routed experts only. A key that `parse_expert_id` maps to `(None, None)` — shared expert, gate, vision tower, projector — is not converted and is not sent through the blockwise FP8 dispatcher.
4. **Where the bytes live.** moe-store owns identification and the store. This repo owns the onboarding dequant seam next to the existing blockwise-FP8 load in `model_offload.py`. No synthetic MXFP8 converter in this phase. No GPU-resident MXFP8 matmul (#234 Phase 3).

## File map (after the checkpoint)

| File | Change |
|---|---|
| `moe_infinity/runtime/model_offload.py` | `mxfp8` does not take `_has_fp8_blockwise`. Routed-expert onboarding calls the MXFP8 dequant, then the existing bf16 expert store. |
| `tests/python/unit/test_mxfp8_predicate.py` | `quant_method=mxfp8` is not blockwise FP8. `quant_method=fp8` still is. |
| moe-store companion PR | Key recognition and scale pairing for the recorded layout. This draft does not merge ahead of that PR. |

## QA gate

- CPU: the predicate test above, plus a tiny tensor round-trip once the group size is known (dequant matches a trusted reference within a recorded atol).
- Against the Phase 4 BF16 run: same cat prompt, greedy, keyword still matches, and the expert-store dtype for routed experts is MXFP8 on disk. No claim that MXFP8 matches BF16 logits bit-exact.

## Non-goals

Blockwise FP8 conversion of a BF16 M3 checkpoint, INT4, quantized residency, Lightning Indexer, and video input.
