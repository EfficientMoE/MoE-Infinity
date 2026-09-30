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
| `tests/fixtures/vl/minimax_m3_keyword_eval.json` (new) | Frozen 20-row acceptance list. Paths only; image bytes stay on the M3 rig. |
| `tests/fixtures/vl/keyword_eval_cpu.json` (new) | Three-row path-only fixture for answer-injection tests; paths need not exist in fake mode. |
| `tests/python/unit/test_vl_keyword_eval.py` (new) | Subprocess coverage for 3/3 success, 1/3 failure, and model-load bypass. |
| `tests/python/unit/test_vl_model_compatibility_docs.py` (new) | Pin the Qwen3.5 vision/residency wording and MiniMax-M3 license/evidence boundary. |

## Fixture and test interface

- `tests/fixtures/vl/minimax_m3_keyword_eval.json` uses this complete schema; the CPU fixture uses the same `items` schema with exactly three rows:

```json
{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"array","minItems":20,"maxItems":20,"items":{"type":"object","additionalProperties":false,"required":["id","image","prompt","keyword"],"properties":{"id":{"type":"string","minLength":1},"image":{"type":"string","minLength":1},"prompt":{"type":"string","minLength":1},"keyword":{"type":"string","minLength":1}}}}
```

- Concrete acceptance row: `{"id":"m3-001","image":"tests/fixtures/vl/images/m3-001.jpg","prompt":"What animal is shown?","keyword":"cat"}`. Before the harness diff, the PR author and M3 rig operator jointly freeze the issue #221 acceptance set as exactly 20 such rows in `tests/fixtures/vl/minimax_m3_keyword_eval.json`; that checked-in file is the record of IDs, repo-relative rig-local image paths, prompts, keywords, and order. No image bytes enter git. The rig operator places the 20 untracked bytes at the listed paths on the M3 checkout. Real mode reports each missing image as `SKIP` and does not download it.
- `tests/fixtures/vl/keyword_eval_cpu.json` has the exact rows `cpu-1`/`tests/fixtures/vl/images/cpu-1.jpg`/`Name the animal.`/`cat`, `cpu-2`/`tests/fixtures/vl/images/cpu-2.jpg`/`Name the color.`/`red`, and `cpu-3`/`tests/fixtures/vl/images/cpu-3.jpg`/`How many objects?`/`two`, encoded with the four keys above.
- `examples/vl_keyword_eval.py --fixture <path> --fake-answers '<json-object>'` parses an `id`-to-answer JSON object. Fake mode validates the fixture and scores those strings, but performs no image existence check and imports/constructs neither `AutoProcessor` nor `MoE`; real mode lazily loads them and follows `vl_example.py`'s `apply_chat_template` plus greedy `model.generate` path.
- The score gate is `ceil(0.8 * row_count)`: 16/20 for acceptance and 3/3 for the CPU fixture. Missing fake-answer IDs count as misses. A met gate returns 0; malformed fixtures/answers or a score below the gate return non-zero.

## QA gate

- CPU, direct pass: `python examples/vl_keyword_eval.py --fixture tests/fixtures/vl/keyword_eval_cpu.json --fake-answers '{"cpu-1":"cat","cpu-2":"red","cpu-3":"two"}'`; expected score `3/3`, no model-load output, exit 0.
- CPU, direct fail: `python examples/vl_keyword_eval.py --fixture tests/fixtures/vl/keyword_eval_cpu.json --fake-answers '{"cpu-1":"cat","cpu-2":"wrong","cpu-3":"wrong"}'`; expected score `1/3`, exit non-zero.
- CPU, pytest: `python -m pytest tests/python/unit/test_vl_keyword_eval.py -q`; expected 3/3 and 1/3 subprocess assertions pass, exit 0.
- Docs: `tests/python/unit/test_vl_model_compatibility_docs.py` reads `docs/model-compatibility.md` via `Path(__file__).resolve().parents[3]` and asserts the exact substrings `visual.*`, `MTP`, `unused`, `MiniMax-M3`, `MiniMax Community License`, `BF16`, `eager attention`, `native engine off`, and `no real-checkpoint evidence`. Run `python -m pytest tests/python/unit/test_vl_model_compatibility_docs.py -q`; expected exit 0.
- GPU, Qwen3.5: `python examples/vl_example.py --image tests/fixtures/images/cat.jpg --expect cat` still prints `PASS` and a non-zero dispatch counter; exit 0.
- GPU, M3, ≥1TB host RAM + NVMe: one cat smoke, then the 20-pair score. Paste both logs on #221. That paste is what closes the checkpoint-evidence item.

## Non-goals

MXFP8 stores, Lightning Indexer kernels, video input, and turning `use_native_engine` back on for `minimax_m3_vl`.
