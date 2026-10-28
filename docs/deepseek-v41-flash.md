# DeepSeek-V4.1-Flash (deepseek_v41) family guide

Tracking issue: [#281](https://github.com/EfficientMoE/MoE-Infinity/issues/281).
Status: **validated for the mp1 text-only greedy smoke scope below**
(2026-10-28 run of `tests/python/v41/e2e_offload.py`). No continuous-serving,
sampled, batch>1, vision, or multi-GPU claim is made.

## What is implemented

- **Path B (official reference code, mp1, text-only first).**
  `moe_store.wrappers.deepseek_v41.load_offloaded_v41_flash` builds the
  pinned reference model with null routed experts, streams the ~283 GB of
  FP4 routed experts through the V4-Flash `OfficialExpertHostStore`
  (identical tensor format: FP4 E2M1 packed + UE8M0 group-32 scales), and
  rebinds the two ~98 GB engram tables (`layers.{1,14}.engram.embed`) to a
  host row-gather store (`EngramHostStore`) so they are never allocated on
  GPU.
- **Precision protection.** `model_type: deepseek_v41` maps to
  `protected:deepseek_v4_fp4` in
  `moe_infinity/runtime/expert_precision.py`; the generic
  `quant_method: fp8` fallback (GLM 128×128 e4m3) can no longer mis-map
  this model. Adaptive expert conversion never touches V4.1 experts.
- **Path A (HF-native) is guarded, not active.** No released transformers
  (≤ 5.21.0) ships `DeepseekV41ForCausalLM`; the moe-store registry entry
  activates automatically when one does.

Excluded by design: dspark MTP (`mtp.*`, weights skipped; no speculative
decoding claim), DFlash pairing, multi-GPU TP, vision input validation
(`text_only=True` is the supported mode), serving-engine (Stage 4a/4b)
integration.

## Pinned official reference code

See `moe-store:moe_store/wrappers/deepseek_v41/OFFICIAL_PIN.md` for the
pinned revision (`314113f8a7dadb388215396ba5fbee2c610a04d6`), the security
note, and the "Verified reference API" record. Download:

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash \
    --revision 314113f8a7dadb388215396ba5fbee2c610a04d6 \
    --include "inference/*" --local-dir <official-dir>
```

The loader's `config_path` must point at `<official-dir>/inference/config.json`
(flat `ModelArgs` keys), not the HF root `config.json`.

## Running the gated end-to-end smoke

Requires the full 510 GB checkpoint on local SSD and a GPU with >= 40 GB
VRAM (resident remainder is ~30 GB):

```bash
MOE_V41_FLASH_CKPT=/ckpt/DeepSeek-V4.1-Flash \
MOE_V41_OFFICIAL_DIR=<official-dir>/inference \
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
    tests/python/v41/e2e_offload.py -q -s
```

Without the env vars the test skips; the suite never fails on machines
without the checkpoint.

## Hardware caveats found during validation

- **SM120 workstation Blackwell**: the reference tilelang `sparse_attn`
  kernel requests ~141 KB dynamic shared memory (64 heads × d=512) and
  cannot launch on GPUs with the ~99 KB opt-in limit; the adapter
  auto-substitutes the exact-math `sparse_attn_eager` fallback
  (unit-tested against a naive reference).
- The whole process must run under `torch.set_default_dtype(torch.bfloat16)`
  and set `torch.set_default_device(<cuda>)` *after* model load (reference
  kernels allocate outputs with the default dtype; cached attention index
  helpers build on the default device).
- Hub shards store `wo_a` quantized; the loader dequantizes it blockwise to
  bf16 exactly as official `convert.py` does (the module keeps `wo_a` bf16).

## Evidence

Recorded 2026-10-28 from `tests/python/v41/e2e_offload.py` (run log in PR
#282 discussion):

| Field | Value |
| --- | --- |
| Checkpoint revision | `314113f8a7dadb388215396ba5fbee2c610a04d6` |
| Hardware | 1× NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB, SM120), shared host, ~24 GB GPU free at run time |
| Software | Python 3.12, torch 2.11.0+cu130, tilelang 0.1.12, transformers 5.12.1 |
| Peak resident VRAM (text-only, experts+engram offloaded, `max_resident_experts=16`) | 14.8 GiB |
| Greedy determinism (16 tokens, 2 runs) | PASS — identical token ids |
| Greedy sanity ("The capital of France is") | `' Paris. The Eiffel Tower is in Paris. The Louvre is in Paris'` |
| Warm decode (16 greedy tokens, batch 1, eager sparse-attn fallback, python FP4 expert path) | 1.20 tok/s shared GPU / 3.67 tok/s exclusive GPU |

Additional exclusive-GPU datapoints (2026-10-29, same stack, warm 64-token
greedy decode):

| Config | Decode | Peak VRAM |
| --- | --- | --- |
| python FP4 path (tilelang `fp4_gemm`), `max_resident_experts=16` | 3.67 tok/s | 14.8 GiB |
| python FP4 path, `max_resident_experts=3584` | 3.49 tok/s | 63.3 GiB |
| native `_v4_fp4.v4_expert_forward`, `max_resident_experts=512` | 4.39 tok/s | 23.0 GiB |

Enlarging the GPU expert cache does **not** raise throughput: decode is
execution-bound (~240 sequential per-expert batch-1 forwards per token),
not PCIe-bound, because the store's copy-stream prefetch already overlaps
transfers. The native fused expert kernel (auto-selected on Blackwell when
`moe_infinity.runtime.engine_ops.register_engine_ops()` has run) gives
+20%; the real lever is batched/fused expert execution, tracked as
follow-up. Path A remains gated: transformers 5.21.0 (latest as of
2026-10-29) still has no `deepseek_v41`.
| A/B harness (`benchmarks/contextpilot/v41flash_ab.py`) | Executed 2026-10-28 (31 requests × 3 repeats, 64 new tokens): `results/v41flash_ab_20261028_044414.json`, verdict **NO-GO** — ContextPilot left prompt-token counts unchanged on every workload (e.g. 201→201), so the 2–9% TTFT deltas are cache warmth/noise, and no ContextPilot benefit is claimed for V4.1-Flash. TTFT p50 5.5–24.9 s, decode 1.2–2.6 tok/s at `max_resident_experts=16`. |

The 1.2 tok/s decode reflects ~240 host→GPU expert fetches per token
(6 active × 40 layers against a 16-expert LRU over 384 experts/layer); it
is an offload-correctness baseline, not a throughput claim.
