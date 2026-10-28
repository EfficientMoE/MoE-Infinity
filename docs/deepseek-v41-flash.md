# DeepSeek-V4.1-Flash (deepseek_v41) family guide

Tracking issue: [#281](https://github.com/EfficientMoE/MoE-Infinity/issues/281).
Status: **implemented/experimental, no real-checkpoint evidence recorded.**
Do not cite this page as a serving claim until the evidence table below is
filled from an actual run.

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

## Evidence

| Field | Value |
| --- | --- |
| Checkpoint revision | not recorded |
| Hardware | not recorded |
| Peak resident VRAM (text-only, experts+engram offloaded) | not recorded |
| Greedy determinism (16 tokens, 2 runs) | not recorded |
| Decode tokens/s | not recorded |
| A/B harness (`benchmarks/contextpilot/v41flash_ab.py`) | not recorded |

Fill this table from the first successful run of the smoke test and the
A/B harness; until then every support statement about V4.1-Flash must say
"implemented/experimental, not validated".
