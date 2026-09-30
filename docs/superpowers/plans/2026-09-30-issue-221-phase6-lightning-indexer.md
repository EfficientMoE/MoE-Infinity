# Issue 221 Phase 6 — Lightning Indexer attention

Draft for review. This does not add a kernel and does not enable the native engine. Phase 4's eager run remains the correctness reference.

**Goal:** Serve MiniMax-M3 block-sparse attention with a kernel that matches the current Hugging Face eager path, while `use_native_engine` stays false for `minimax_m3_vl`.

**Why the native engine stays off:** `MoE.__init__` forces `use_native_engine = False` when `model_type` is `minimax_m3_vl`, and `arch == "minimaxm3"` forces `is_flash_attn_available = False`. Those guards exist because the native paged backend has no Lightning Indexer implementation. Re-enabling either flag is a separate change from landing a kernel inside the Hugging Face module.

## Decision checkpoint (not yet accepted)

Do not pick FlashInfer versus a local kernel until these are written on this PR from the installed `transformers` MiniMax-M3 attention module:

1. Block size, whether the mask is causal, how many blocks the indexer keeps, and which tensors are the index versus the attention scores.
2. Eager stays the default. A kernel path is opt-in until a tiny fixture (same shape style as `test_sync_block_matches_hf_block`) matches eager outputs at a recorded tolerance.
3. The first patch calls that kernel from the Hugging Face attention module. It does not flip `use_native_engine` and does not clear the `minimaxm3` flash-attn disable.
4. Qwen3.5 hybrid attention and GLM-DSA are not part of this patch. Their eager/native split has a different reason.

## File map (after the checkpoint)

| File | Change |
|---|---|
| New kernel module under `moe_infinity/kernel/` | Only the sparsity pattern recorded above. Eager fallback remains. |
| `moe_infinity/entrypoints/big_modeling.py` | Comment stays accurate: native engine still off. Wire the opt-in kernel flag here only after the tiny fixture passes. |
| `tests/python/unit/test_minimax_m3_indexer.py` | Tiny eager-versus-kernel comparison. Skip when `transformers` has no `minimax_m3_vl`. |

## QA gate

- CPU or single GPU: tiny fixture matches eager.
- Against the Phase 4 cat smoke: same prompt, greedy, keyword still matches, with the kernel flag on. The flag-off path is unchanged.

## Non-goals

Paged native KV for M3, flash-attn on `minimaxm3`, MXFP8, video, and multi-image.
