# Issue 221 Phase 4 — BF16 eager VL baseline

Draft for review. This does not run the 428B checkpoint and does not change the serving path. Later phases (MXFP8, Lightning Indexer) need a recorded BF16 eager result to compare against.

**Goal:** Write down what Stages 1–3 already shipped, and add a hardware-gated harness whose first successful run becomes the close-out evidence for #221.

**Shipped today:** #224 (multimodal batch plumbing), #229 (Qwen3.5 vision tower resident), #230 and moe-store#8 (MiniMax-M3, 128 routed experts offloaded, HF eager attention, native engine off). `examples/vl_example.py` already checks a non-empty answer, routed-expert dispatch stats, and an optional `--expect` substring.

## Decision checkpoint (proposed)

Accept or replace before any harness or docs diff:

1. MiniMax-M3 acceptance is the issue's keyword bar: at least 16 of 20 answers on a frozen image/question/keyword list contain the keyword. Greedy token-id parity against a full 428B Hugging Face reference is not the bar.
2. Record `torch.cuda.max_memory_allocated()` on every visible GPU. Do not fail the run when that value exceeds `device_memory_ratio * device_total`. The ratio sizes the expert cache. Resident weights (text backbone, shared expert, vision tower, projector, KV) are outside that cache. The first successful load reports those resident bytes; a numeric ceiling is chosen from that report, not before it.
3. The Qwen3.5 smoke (`Qwen/Qwen3.5-35B-A3B`, `tests/fixtures/images/cat.jpg`, `--expect cat`) stays the small proof that the vision path fetched routed experts. It does not stand in for the M3 run.
4. `docs/model-compatibility.md` describes shipped behavior only. Qwen3.5 keeps `visual.*` resident and still leaves MTP unused. MiniMax-M3 is BF16, eager attention, native engine off, no real-checkpoint evidence yet. The M3 row states the MiniMax Community License. No MXFP8 row and no indexer kernel claim.

## File map (after the checkpoint)

| File | Change |
|---|---|
| `docs/model-compatibility.md` | Replace the "Qwen3.5 is text-only / vision unused" lines. Add a MiniMax-M3 row with the license and the evidence boundary above. |
| `examples/vl_keyword_eval.py` | Read a JSON list of image, prompt, and keyword. Call the same generate path as `vl_example.py`. Print per-GPU max memory and the keyword score. Exit non-zero under 16/20. |
| Fixture list | Paths only, checked into the repo. Image bytes stay on the rig. Missing images skip with a message, so CI does not download M3. |

## QA gate

- CPU: the eval script exits 0 on a three-row fake list when the injected answers contain the keywords, and exits non-zero at 1/3.
- GPU, Qwen3.5: `vl_example.py --expect cat` still prints `PASS` and a non-zero dispatch counter.
- GPU, M3, ≥1TB host RAM + NVMe: one cat smoke, then the 20-pair score. Paste both logs on #221. That paste is what closes the checkpoint-evidence item.

## Non-goals

MXFP8 stores, Lightning Indexer kernels, video input, and turning `use_native_engine` back on for `minimax_m3_vl`.
