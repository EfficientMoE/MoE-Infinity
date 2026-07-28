# Plan: Support DFlash Speculative Decoding for gpt-oss-120b in MoE-Infinity

**Status:** RFC — Phase 0 (SGLang reference validation) complete → **GO** for native port (see §0.1)
**Date:** 2026-07-28
**Goal:** Add **DFlash** (block-diffusion speculative decoding, arXiv 2602.06036) support to MoE-Infinity so `openai/gpt-oss-120b` (target) can be accelerated by the `z-lab/gpt-oss-120b-DFlash` 0.8B drafter, and produce **distribution-lossless** greedy output identical to plain gpt-oss-120b decoding.

---

## 0. Executive Summary

DFlash is a lightweight (~0.8B) **block-diffusion drafter** attached to a frozen autoregressive target. It extracts hidden states from 5 selected target layers, KV-injects them into every drafter layer, and predicts a whole masked block `[anchor, MASK×9]` in **one non-causal forward** → 9 draft tokens. The target then verifies the contiguous block in one forward; the longest matching prefix + 1 bonus token is committed and the rejected suffix's KV is rolled back. It is **lossless** speculative decoding (target is source of truth).

MoE-Infinity today: gpt-oss-120b is a supported target (offloaded MXFP4 experts), but there is **zero speculative-decoding scaffolding**. Two decode paths exist: async continuous-batching serving (`serving/engine.py`) and a synchronous native generate loop (`engine/generation_loop.py`).

**Strategic decision (Oracle-endorsed):** proceed **(c) → (b)**:
1. **Phase 0 — validate via the SGLang reference, fully resident (no offload)** to confirm the checkpoint works and to capture baseline parity/acceptance-length/tok-s on this box. If resident-mode gain is weak, **stop** (go/no-go gate).
2. **Then build natively into the SYNC path first** (`engine/generation_loop.py`) — it already has a usable non-paged HF `past_key_values` fallback and is structurally far closer than async serving. Async serving is a **later, optional** phase and is out of scope for v1.

**Resident-only for v1.** MoE-Infinity's differentiator is expert offloading, but a 9-token verify block activates many more experts per step; the current next-layer/top-2/mean-router prefetcher would thrash. Spec-decode + offload is deferred (adaptive-fallback design sketched, not built).

**Effort:** Phase 0 = Small–Medium (mostly downloads + a possibly-fragile sglang branch build). Native sync integration (Phases 1–4) = Medium. Async serving + offload-aware = Large (deferred).

---

## 0.1 Phase 0 Results — SGLang reference validation (2026-07-28) → GO

Validated `openai/gpt-oss-120b` + `z-lab/gpt-oss-120b-DFlash` via the SGLang reference on 6× RTX PRO 6000 Blackwell (SM120).

| Check | Result |
|---|---|
| DFlash worker init | PASS — `DFlashDraftModel`, `block_size=10`, `mask_token_id=200000`, greedy head folded into draft CUDA graph |
| Health | PASS (200) |
| Acceptance length | PASS — mean **3.66** (chat), ~2.6 (long technical); sanity band 3–5 (H200 ref 3.7–5.4) |
| Single-stream decode tok/s | **1.18–1.32×** speedup vs no-spec |
| Concurrency-8 | **0.65× (regression)** — verify blocks hurt under batching |
| Strict string-losslessness | FAIL by string-identity, but **not a DFlash bug** (see finding 2) |

**Two findings that refine this plan:**

1. **TP=4 is currently infeasible on SM120 for gpt-oss MXFP4 (sglang main).** No MXFP4 MoE backend supports it: Marlin fails the shape check (`intermediate 2880 / 4 = 720`, `720 % 32 ≠ 0`), `flashinfer_mxfp4` is SM90/SM100-only, and `triton_kernel` exceeds the RTX PRO 6000's 99 KB shared-mem/block (vs 227 KB on B200). Validation therefore ran **TP=2 + Marlin** (`2880 / 2 = 1440 % 32 = 0`). A fully-resident native port should target **TP=2 (or DP2×TP2)** on SM120, not TP=4, until an upstream MXFP4-at-TP=4 fix lands.

2. **Strict token-identity is unattainable on this stack — and not DFlash's fault.** The no-spec baseline is not string-identical to *itself* across repeated greedy runs (1/5 → 3/5 even with `--enable-deterministic-inference`); divergences are floating-point near-tie argmax flips (Marlin MXFP4 grouped-GEMM reduction ordering; 10-token verify forwards take a different-but-valid rounding path than 1-token decode). All outputs are coherent/correct. This is the standard "lossless in exact arithmetic ≠ bit-identical in FP" property of speculative decoding.

**Carried into Phases 1–3 (gates revised accordingly):**
- The parity gate (QA-3.1) compares DFlash vs plain decoding **in the same engine build/process** using a **token agreement-rate** metric, not string identity.
- **Disable/avoid spec-decode at high batch sizes** — it is a single/few-stream win, which is exactly MoE-Infinity's resident regime.
- Resident config target on SM120 is **TP=2** (not TP=4) until the upstream MXFP4-at-TP=4 gap closes.

---

## 1. Verified Facts (evidence base)

All facts verified against source (HF checkpoint `config.json`/`dflash.py`/`utils.py`, vLLM `qwen3_dflash.py` + `v1/spec_decode/dflash.py`, SGLang `dflash_worker_v2.py`/`dflash_utils.py`/`dflash_info.py`, and MoE-Infinity source via code exploration).

### 1.1 The DFlash drafter checkpoint (`z-lab/gpt-oss-120b-DFlash`)

- `config.json`: `architectures=["DFlashDraftModel"]`, `auto_map.AutoModel="dflash.DFlashDraftModel"` (**requires `trust_remote_code=True`**), `model_type=qwen3`.
- `block_size = 10` (**not 9**). Drafter predicts `block_size-1 = 9` mask positions per anchor (`sample_from_anchor=False`); per-step commit budget ≤ 10 (9 drafts + 1 target bonus).
- `dflash_config.mask_token_id = 200000`; `dflash_config.target_layer_ids = [1, 9, 17, 25, 33]` (5 layers, explicit override).
- Drafter dims: `num_hidden_layers=8`, `hidden_size=2880`, `num_attention_heads=64`, `num_key_value_heads=8` (GQA), `head_dim=64`, `intermediate_size=7680`, `num_target_layers=36`, `vocab_size=201088`, `attention_bias=true`, `tie_word_embeddings=false`, dtype bf16, RoPE yarn factor 32.
- Checkpoint contents: `config.json`, `dflash.py`, `utils.py`, single ~1.57 GB `model.safetensors`. It is the **drafter only** — no target transformer/experts/lm_head.
- **The drafter reuses the TARGET's `embed_tokens` and `lm_head`** at runtime: `noise_embedding = target.model.embed_tokens(block_ids)`; `draft_logits = target.lm_head(drafter(...))`.

### 1.2 DFlash mechanics (exact contract)

- **Context feature:** `extract_context_feature(hidden_states, layer_ids)` concatenates target hidden states at `hidden_states[layer_id + 1]` for `layer_id in [1,9,17,25,33]` along the last dim → `[B, L, 5*2880] = [B, L, 14400]`. (`+1` offset = output after that target layer; index 0 is the embedding.)
- **Fusion + KV injection:** `fc = Linear(14400 → 2880, bias=False)`, then `hidden_norm` (RMSNorm). The fused feature is projected to per-layer K/V and written into **every** drafter layer's KV cache (vLLM `precompute_and_store_context_kv`: RMSNorm → fused `[L*2*kv,H]` GEMM → K-RMSNorm → RoPE → per-layer cache insert). SGLang folds the greedy argmax over the target `lm_head` into the drafter CUDA graph.
- **Drafter attention is NON-CAUSAL** (`is_causal=False`): `k = cat([k_ctx, k_noise]); v = cat([v_ctx, v_noise])`; every mask query attends to all context + all queries.
- **spec_generate loop (HF reference):**
  1. Prefill target (`output_hidden_states=True, logits_to_keep=1`); anchor = `sample(target.logits)`; `target_hidden = extract_context_feature(...)`.
  2. Block = `[anchor, MASK×9]`; `noise_embedding = target.embed_tokens(block)`; `draft_logits = target.lm_head(drafter(...))[:, -block_size+1:, :]` (positions 1..9).
  3. `block[:,1:] = sample(draft_logits)`.
  4. Target forward over full block (`output_hidden_states=True`) → `posterior = sample(target.logits)`.
  5. `acceptance_length = cumprod(block[:,1:] == posterior[:,:-1]).sum()`; commit `acceptance_length+1` tokens + bonus `posterior[acceptance_length]`.
  6. `target_kv.crop(start)`, `draft_kv.crop(start)`; `target_hidden = extract_context_feature(...)[:, :acceptance_length+1, :]`.
- **Losslessness:** greedy acceptance compares argmax; final tokens are the target's — output must be argmax-identical to plain target greedy decoding.
- **Reference speedups (H200, SGLang, block 10):** acceptance length ~3.7–5.4; e2e speedup ~1.3–1.9×.

### 1.3 MoE-Infinity target engine (verified anchors)

- gpt-oss-120b supported: `MODEL_MAPPING_NAMES["gptoss"]=GptOssForCausalLM`, expert-type 4 (`common/constants.py`); `SyncGptOssMLP` (`models/gpt_oss.py`) monkey-patched over HF `GptOssMLP` in `runtime/model_offload.py` (~L547-550); config parse in `utils/hf_config.py` (~L114-118, L198-205). gpt-oss explicitly excluded from HF flash_attention_2 (`entrypoints/big_modeling.py` ~L204).
- **No speculative decoding anywhere.** `speculative_prefetch` (`memory/expert_prefetcher.py` L106) is router-logit→expert-id prefetch — unrelated and NOT repurposable.
- **Sync path (v1 target):** `engine/generation_loop.py::GenerationEngine.generate()` — prefill `_run_forward(prompt)` (~L107) → `_sample()` (~L120) → decode loop (~L123) `_run_forward([last])` (~L150) + `_sample()` (~L154). `_sample()` at ~L192 (greedy argmax ~L198). Invoked from `big_modeling.py` ~L647 via `_native_model_forward` (~L460). Has a usable non-paged HF `past_key_values` path.
- **Async path (deferred):** `serving/engine.py::ContinuousBatchingEngine.step()` (~L176) → `model_runner.execute()` → `_extract_last_token_logits` (~L195) → `sampler.sample()` (~L196, `serving/sampler.py` L26) returns 1 token/seq → `scheduler.update_after_step()` (~L246, hardcoded +1). Paged KV in `serving/kv_cache.py`. **Warning:** `ModelRunner._get_paged_attention_classes()` only recognizes DeepSeek paged-attention; `serving/cuda_graph.py` is scaffolded but uncalled — async gpt-oss is less ready.
- The model forward does not currently expose intermediate hidden states, BUT HF `GptOssModel` supports `output_hidden_states=True` (via `_can_record_outputs`/`@capture_outputs`), giving a clean side-channel with no second pass.

### 1.4 Environment / feasibility (verified on this machine)

- 6× RTX PRO 6000 Blackwell (SM120), 96 GB each, idle. gpt-oss-120b (MXFP4 ~60 GB) fits **fully resident** on 1–2 GPUs; the 0.8B drafter is trivial. Offloading NOT required here.
- venv: Python 3.12, torch 2.11.0+cu130, transformers 5.12.1, moe_infinity 0.0.1.
- **Missing / to-fetch:** `sglang`, `vllm`, `flashinfer`, `flash_attn` not installed; neither `openai/gpt-oss-120b` nor `z-lab/gpt-oss-120b-DFlash` is in the local HF cache (`HF_HOME=/mnt/raid0nvme0/public/huggingface`).

---

## 2. Scope

### In scope
- Phase 0: run the SGLang DFlash reference on this box (resident) to validate the checkpoint and record baseline parity + acceptance length + tok/s.
- Native DFlash in MoE-Infinity's **synchronous** `engine/generation_loop.py` decode path (greedy first), **fully resident** (no expert offload).
- Target hidden-state capture side-channel; drafter runtime (`DFlashDraftModel` via trust_remote_code, or a vendored reimplementation); block verify + paged/HF KV rollback.
- Losslessness proof vs plain gpt-oss-120b greedy; acceptance-length sanity vs the reference.

### Out of scope (v1)
- Async continuous-batching serving integration (Phase 4, deferred).
- Expert-offload + spec-decode co-existence (deferred; adaptive-fallback design only).
- Sampling (temperature > 0) lossless speculative sampling — greedy first; sampled path is a follow-up.
- Tree drafting (DFlash is linear/block — N/A), multi-node, training.

---

## 3. Critical Risks & Mitigations

| # | Risk | Evidence | Mitigation |
|---|---|---|---|
| R1 | **SGLang DFlash reference may not build** on torch 2.11/cu130/Blackwell from the PR branch. | §1.4; PR #20547 was spec-v2 on top of an earlier DFlash PR. | Phase 0 is time-boxed. If the branch won't build cleanly, fall back to the **HF `spec_generate` reference** (`dflash.py` + `openai/gpt-oss-120b`, plain transformers) to validate correctness, and defer throughput baselining. |
| R2 | **Hidden-state extraction** must not break the paged-attention/CUDA-graph path or trigger extra expert transfers. | §1.3 | Use `output_hidden_states=True` returned side-channel (HF-native, single pass), scoped to the 5 layers. NOT global forward hooks. Validate no extra expert I/O via cache-hit counters. |
| R3 | **KV rollback** on partial acceptance (slot_mapping/block-table/position-id traps). | §1.2, §1.3 | Beachhead on the **HF `past_key_values`** sync path first (`.crop(n)` like the reference), where rollback is trivial. Only add paged `truncate_to(seq,n)` if/when needed. Do NOT append draft tokens to committed output before acceptance — only KV is tentative. |
| R4 | **Drafter reuses target embed_tokens + lm_head** — dim/tokenizer/vocab must match exactly. | §1.1 | Assert `vocab_size==201088`, `hidden_size==2880`, `mask_token_id(200000) < vocab_size`, and `fc.in==5*2880`. Fail loudly on mismatch (mirror vLLM `combine_hidden_states` guard). |
| R5 | **Offload amplification** — a 9-token verify block routes to many experts, thrashing the cache. | §1.3, Oracle | Resident-only for v1. Later: adaptive disable when predicted expert-union/transfer-time exceeds a threshold. |
| R6 | **Losslessness regressions** hard to detect. | §1.2 | Golden test: greedy DFlash continuation must be **argmax-identical** to plain gpt-oss-120b greedy for a fixed prompt set (16-token continuation token-identical). |

---

## 4. Phased Plan

### Phase 0 — Reference validation, fully resident (DO FIRST; go/no-go gate)
1. Download `openai/gpt-oss-120b` + `z-lab/gpt-oss-120b-DFlash` into `$HF_HOME`.
2. Install SGLang from the DFlash-enabled ref; launch: `--model-path openai/gpt-oss-120b --speculative-algorithm DFLASH --speculative-draft-model-path z-lab/gpt-oss-120b-DFlash --tp-size 1 --dtype bfloat16 --mem-fraction-static 0.75 --trust-remote-code`.
3. Record: greedy output parity vs plain gpt-oss-120b, acceptance-length histogram, and decode tok/s (DFlash vs no-spec) on a fixed prompt set (GSM8K/HumanEval subset).
4. **Fallback (R1):** if SGLang won't build, run the HF `spec_generate` reference for correctness only.

**Exit / GO criteria:** reference runs, greedy output is lossless vs plain target, acceptance length ≥ ~3, and resident decode tok/s improves measurably. **If gain is not clearly positive → STOP and report** (native build not worth it).

### Phase 1 — Sync hidden-state capture side-channel (walking skeleton)
1. Add a returned side-channel to the sync forward (`_native_model_forward`/`GenerationEngine`): `execute_with_capture(capture_layer_ids=(1,9,17,25,33), token_selector="all")` using `output_hidden_states=True`; return only the 5 selected layers' states.
2. Assert shapes: captured concat = `[B, L, 14400]`.

**Exit:** capture returns correct 5-layer states with zero change to normal (non-spec) generation output and no extra expert transfers (cache-hit counters unchanged).

### Phase 2 — Sync greedy verify + rollback
1. Instantiate the DFlash drafter (trust_remote_code `DFlashDraftModel`, or vendored). Wire it to the target's `embed_tokens`/`lm_head`.
2. `SpeculativeVerifier`: build block `[anchor, MASK×9]`, drafter forward (KV-inject the 5-layer feature), target verify forward, accept via `cumprod(cand[:,1:]==tgt[:,:-1]).sum()`, commit `accept+1`, roll back KV (`past_key_values.crop(n)`).
3. Acceptance lives **before** `_sample()` in `generation_loop.py`.

**Exit:** integrated greedy loop produces coherent text; per-step committed tokens = accept+1.

### Phase 3 — Prove correctness (losslessness)
1. Golden parity: DFlash greedy vs plain gpt-oss-120b greedy — 16-token continuation **token-identical** on the fixed prompt set.
2. Acceptance-length stats within sanity range of Phase 0 reference.

**Exit:** parity test passes; acceptance length matches reference order-of-magnitude.

### Phase 4 — (DEFERRED) Async serving + offload-aware
- Explicit verify-batch builder for `[anchor+9]`; `Scheduler.update_after_step` `+1` → `kv_append_counts: dict[int,int]`; paged `truncate_to`. gpt-oss async paged-attention gaps must be closed first.
- Offload: adaptive spec-decode disable under expert-I/O pressure.

---

## 5. QA Scenarios (executable verification per phase)

Shell vars: `$PY=/mnt/raid0nvme0/leyang/MoE-Infinity/.venv/bin/python`; `$HF_HOME=/mnt/raid0nvme0/public/huggingface`; `$TGT=openai/gpt-oss-120b`; `$DRAFT=z-lab/gpt-oss-120b-DFlash`; `$PROMPTS` = a committed fixture of ~20 prompts. New tests under `tests/python/dflash/`.

**QA-0.1 (reference lossless + speedup)** — Launch SGLang DFlash; greedy continuation for `$PROMPTS` is token-identical to plain `$TGT` greedy; record acceptance-length histogram + tok/s (DFlash vs no-spec) to a committed results file. GO iff lossless AND tok/s improves.

**QA-1.1 (capture shapes)** — `pytest tests/python/dflash/test_capture.py -q`: for a short prompt, `execute_with_capture` returns 5 states, concat dim == 14400; normal generation output byte-identical with capture on vs off; expert cache-hit counter unchanged.

**QA-2.1 (drafter contract)** — `pytest tests/python/dflash/test_drafter_load.py -q`: drafter loads with `trust_remote_code`; asserts `hidden_size==2880`, `block_size==10`, `mask_token_id==200000<vocab`, `target_layer_ids==[1,9,17,25,33]`, `fc.in_features==14400`; drafter forward on random 5-layer feature emits `[B,9,vocab]` via target lm_head.

**QA-2.2 (accept rule parity)** — `pytest tests/python/dflash/test_accept_rule.py -q`: on synthetic `(candidates, target_predict)`, `acceptance_length` equals `cumprod(cand[:,1:]==tgt[:,:-1]).sum()` for hand-checked cases (full accept=9, first-mismatch=k, none=0).

**QA-3.1 (end-to-end parity — agreement-rate, same-process)** — `$PY tests/python/dflash/compare_greedy.py --prompt-file $PROMPTS`: run DFlash greedy vs plain greedy **in the same engine build/process** and compute the per-position **token agreement rate** over a 128-token continuation. Pass iff agreement ≥ the plain-decode **self-consistency** rate measured on the same box (i.e., DFlash disagrees no more than plain decoding disagrees with itself across repeat runs). Do NOT gate on exact string identity — Phase 0 (§0.1) showed FP near-tie argmax flips make plain decoding non-self-identical on this MXFP4 stack. Report first-divergence positions for inspection. Exit 0.

**QA-3.2 (acceptance sanity)** — mean acceptance length over `$PROMPTS` within ±30% of Phase-0 reference for block_size 10. Exit 0.

---

## 6. Open Questions (for review)
- Q1: ~~Phase 0 — is a real SGLang build required...~~ **Resolved (§0.1):** sglang git-main built and ran on SM120 (TP=2 Marlin; TP=4 infeasible on SM120 — see §0.1); HF `spec_generate` fallback was not needed.
- Q2: Reuse the drafter's shipped `dflash.py` via `trust_remote_code`, or vendor a clean reimplementation into `moe_infinity/spec_decode/dflash/`? (Lean: trust_remote_code for v1 correctness; vendor later for control/perf.)
- Q3: Does capturing `output_hidden_states=True` on gpt-oss interact badly with the `SyncGptOssMLP` monkey-patch or CUDA-graph capture? (Verify in Phase 1.)
- Q4: ~~Is native MoE-Infinity DFlash worth it...~~ **Resolved → GO (§0.1):** single/few-stream resident decode (MoE-Infinity's regime) shows a 1.18–1.32× win; proceed with the native sync port, keeping spec-decode off at high batch sizes.

## 7. Definition of Done (v1)
- Phase 0 reference validated: acceptance length + resident speedup recorded (done — §0.1: mean accept 3.66, single-stream 1.18–1.32×).
- Native sync DFlash path in MoE-Infinity matches plain gpt-oss-120b greedy **within the plain-decode self-consistency agreement rate** on the fixed prompt set (QA-3.1; strict string identity is not required — see §0.1).
- Acceptance length within sanity range of the reference (QA-3.2).
- Runs fully resident (TP=2 on SM120); existing gpt-oss tests still pass (no regression). Spec-decode disabled at high batch sizes. Offload + async serving explicitly deferred and documented.
