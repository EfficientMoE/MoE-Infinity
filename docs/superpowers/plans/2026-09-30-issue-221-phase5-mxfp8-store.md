# Issue 221 Phase 5 — MiniMax-M3 MXFP8 expert host store

Draft for review. This does not dequantize a checkpoint. Phase 4's BF16 eager log is the numeric baseline this phase is measured against. #234 Phase 3 (quantized GPU residency) stays a different change.

**Goal:** Load checkpoint-native MXFP8 routed experts from the host store and dequantize them to bf16 at onboarding. Shared expert, vision tower, and multimodal projector stay resident in the dtype the checkpoint stored.

## Decision checkpoint (metadata accepted)

Implementation is bound to these recorded facts:

1. **Predicate bug.** `_has_fp8_blockwise` in `moe_infinity/runtime/model_offload.py` is true when `quant_method` contains the substring `fp8`. A config value of `mxfp8` therefore enters the 128×128 blockwise path (`_scale_inv` pairs, fp32 scales) used by GLM-5.2, DeepSeek-V3, and #235. The first patch excludes `mxfp8` from that predicate, with a CPU test, before any MXFP8 tensor is read.
2. **Accepted on-disk record (metadata only).** Discovery ran 2026-09-30 UTC: `list_repo_refs` found `MiniMaxAI/MiniMax-M3` main at `f0e1c1e04d40177e4673a22097036854f536e9c0`; `list_repo_commits` covered all 14 revisions; the MiniMaxAI author sweep and all-author searches for `MiniMax-M3`, `MXFP8`, `FP8`, and quantized variants produced 116 repo/revision candidates (90 safetensors configs/headers probed, 26 unavailable or non-safetensors). `FOUND` is `MiniMaxAI/MiniMax-M3-MXFP8@c5454eb03678d8710e54a4e0fc681b9f3b4a3dba` (repo update 2026-07-11). This immutable revision is the only checkpoint used below.

   Recorded routed payload pattern: `language_model.model.layers.<L>.block_sparse_moe.experts.<E>.{w1,w2,w3}.weight`, safetensors dtype `F8_E4M3`. Its paired scale is `<payload>_scale_inv`, dtype `U8` encoding E8M0, with `quantization_config={quant_method: mxfp8, weight_block_size: [1, 32]}`: one scale per 32 payload elements along K. Metadata contains 21,888 payloads and 21,888 scales, with zero missing or orphan scales. Concrete pair: `language_model.model.layers.3.block_sparse_moe.experts.0.w1.weight` `[3072,6144]` and `...w1.weight_scale_inv` `[3072,192]`.

   Recheck without weights: `HF_HOME=/tmp/m3-metadata python -c 'import json; from huggingface_hub import get_safetensors_metadata,hf_hub_download; r="MiniMaxAI/MiniMax-M3-MXFP8"; v="c5454eb03678d8710e54a4e0fc681b9f3b4a3dba"; q=json.load(open(hf_hub_download(r,"config.json",revision=v)))["quantization_config"]; m=get_safetensors_metadata(r,revision=v,timeout=30); t={k:str(x.dtype) for f in m.files_metadata.values() for k,x in f.tensors.items()}; p="language_model.model.layers.3.block_sparse_moe.experts.0.w1.weight"; assert q["quant_method"]=="mxfp8" and q["weight_block_size"]==[1,32] and t[p]=="F8_E4M3" and t[p+"_scale_inv"]=="U8"'`; expected exit 0.
3. **Which tensors.** Routed experts only. A key that `parse_expert_id` maps to `(None, None)` — shared expert, gate, vision tower, projector — is not converted and is not sent through the blockwise FP8 dispatcher.
4. **Companion contract and handoff.** This repo reads shards in `OffloadEngine.from_pretrained_decorator`, then `_offload_state_dict` hands each tensor to moe-store through `self.archer_engine.offload(tensor, tensor_id)`. The required moe-store companion PR adds public `moe_store.parsing.mxfp8.identify_mxfp8_pairs(names)` for the exact `<payload>_scale_inv` contract; makes `detect_quantization` accept `mxfp8`; makes `CastPolicy.keeps_raw_dtype` preserve E4M3/U8 pairs; and extends `V5Expansion.expand_names/spec/load` plus `parse_expert_id` to normalize the recorded `language_model.model.layers.*.block_sparse_moe` `w1/w2/w3` names into one `(layer, expert)` group. MoE-Infinity imports that API beside `identify_mxfp4_pairs`, dequantizes routed pairs to bf16 before `_offload_state_dict`, and never stores orphan scales. No synthetic converter or GPU-resident MXFP8 matmul.

## File map (after the checkpoint)

| File | Change |
|---|---|
| `moe_infinity/runtime/model_offload.py` | `mxfp8` does not take `_has_fp8_blockwise`. Routed-expert onboarding calls the MXFP8 dequant, then the existing bf16 expert store. |
| `tests/python/unit/test_mxfp8_predicate.py` | `quant_method=mxfp8` is not blockwise FP8. `quant_method=fp8` still is. |
| `tests/python/unit/test_mxfp8_dequant.py` | One recorded-group payload/scale fixture dequantizes to bf16 and round-trips against a float32 reference. |
| moe-store companion PR | Implement the four named APIs above and `tests/test_mxfp8_pairing.py`. This PR does not merge ahead of it or an accepted MXFP8 metadata record. |

## QA gate

- Predicate: `python -m pytest tests/python/unit/test_mxfp8_predicate.py -q`; expected exit 0 with `2 passed`, proving `mxfp8 -> False` and ordinary `fp8 -> True` for `_has_fp8_blockwise`.
- Tensor round-trip: `python -m pytest tests/python/unit/test_mxfp8_dequant.py -q`; expected exit 0 with `1 passed`. The fixture is `F8_E4M3` payload plus `U8` E8M0 inverse scale per 32 K-elements, compares bf16 output promoted to float32 against an independent float32 decode, and requires `torch.allclose(..., rtol=1e-2, atol=1e-2)`.
- moe-store companion, from the `moe-store` repo: `python -m pytest tests/test_mxfp8_pairing.py -q`; expected `1 passed`. Inputs are the concrete layer-3/expert-0 `w1.weight` and `w1.weight_scale_inv` keys above; `identify_mxfp8_pairs` must return exactly that pair, both members must share `StoreIndex.expert_group(3, 0)`, and the orphan-scale set must be empty.
- Pinned metadata: run the header-only recheck in checkpoint 2; expected exit 0 before downloading weights.
- Phase 4 comparison, on the ≥1TB-RAM/NVMe rig: `hf download MiniMaxAI/MiniMax-M3-MXFP8 --revision c5454eb03678d8710e54a4e0fc681b9f3b4a3dba --local-dir /nvme/m3-checkpoints/c5454eb03678d8710e54a4e0fc681b9f3b4a3dba && CUDA_VISIBLE_DEVICES=0 python examples/vl_example.py --checkpoint /nvme/m3-checkpoints/c5454eb03678d8710e54a4e0fc681b9f3b4a3dba --image tests/fixtures/images/cat.jpg --prompt "What animal is in the picture?" --offload_dir /nvme/m3-mxfp8-store --device_memory_ratio 0.8 --max_new_tokens 64 --expect cat`; expected exit 0, `ANSWER:` containing `cat`, non-zero `ROUTING_STATS`, and final `PASS`, matching Phase 4's BF16 keyword result. Store metadata must show bf16 routed tensors after onboarding; no bit-exact logit claim.

## Non-goals

Blockwise FP8 conversion of a BF16 M3 checkpoint, INT4, quantized residency, Lightning Indexer, and video input.
