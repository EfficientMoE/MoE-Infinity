# Issue 221 Phase 4 — BF16 eager VL baseline

Draft for review. This does not run the 428B checkpoint and does not change the serving path. Later phases (MXFP8, Lightning Indexer) need a recorded BF16 eager result to compare against.

**Goal:** Write down what Stages 1–3 already shipped, and add a hardware-gated harness whose first successful run records correctness, memory, and performance close-out evidence for #221.

**Shipped today:** #224 (multimodal batch plumbing), #229 (Qwen3.5 vision tower resident), #230 and moe-store#8 (MiniMax-M3, 128 routed experts offloaded, HF eager attention, native engine off). `examples/vl_example.py` already checks a non-empty answer, routed-expert dispatch stats, and an optional `--expect` substring.

## Decision checkpoint (proposed)

Accept or replace before any harness or docs diff:

1. MiniMax-M3 acceptance is the issue's keyword bar: at least 16 of 20 answers on a frozen image/question/keyword list contain the keyword. Greedy token-id parity against a full 428B Hugging Face reference is not the bar.
2. Record `torch.cuda.max_memory_allocated()` on every visible GPU. Do not fail the run when that value exceeds `device_memory_ratio * device_total`. The ratio sizes the expert cache. Resident weights (text backbone, shared expert, vision tower, projector, KV) are outside that cache. The first successful load reports those resident bytes; a numeric ceiling is chosen from that report, not before it.
3. Record per-request time to first token (TTFT) and decode tokens/s from the same 20-pair run. Report p50/p99 only; these numbers are the BF16 baseline for Phases 5 and 6, not a pass/fail gate.
4. The Qwen3.5 smoke (`Qwen/Qwen3.5-35B-A3B`, `tests/fixtures/images/cat.jpg`, `--expect cat`) stays the small proof that the vision path fetched routed experts. It does not stand in for the M3 run.
5. `docs/model-compatibility.md` describes shipped behavior only. Qwen3.5 keeps `visual.*` resident and still leaves MTP unused. MiniMax-M3 is BF16, eager attention, native engine off, no real-checkpoint evidence yet. The M3 row states the MiniMax Community License. No MXFP8 row and no indexer kernel claim.

## File map (after the checkpoint)

| File | Change |
|---|---|
| `docs/model-compatibility.md` | Replace the "Qwen3.5 is text-only / vision unused" lines. Add a MiniMax-M3 row with the license and the evidence boundary above. |
| `examples/vl_keyword_eval.py` | Read a JSON list of image, prompt, and keyword. Call the same generate path as `vl_example.py`. Print per-GPU max memory, keyword score, and perf summary. Exit non-zero under 16/20 only. |
| `benchmarks/minimax_m3/perf_summary.py` (new) | Pure percentile and performance-summary math; no torch/model import. |
| `tests/fixtures/vl/minimax_m3_keyword_eval.json` (new) | Frozen 20-row acceptance list. Paths only; image bytes stay on the M3 rig. |
| `tests/fixtures/vl/keyword_eval_cpu.json` (new) | Three-row path-only fixture for answer-injection tests; paths need not exist in fake mode. |
| `tests/python/unit/test_vl_keyword_eval.py` (new) | Subprocess coverage for 3/3 success, 1/3 failure, model-load bypass, and deterministic fake timings. |
| `tests/python/unit/test_minimax_m3_perf_summary.py` (new) | CPU tests for interpolation, summary fields, and invalid samples. |
| `tests/python/unit/test_vl_model_compatibility_docs.py` (new) | Pin the Qwen3.5 vision/residency wording and MiniMax-M3 license/evidence boundary. |

## Fixture and test interface

- `tests/fixtures/vl/minimax_m3_keyword_eval.json` uses this complete schema; the CPU fixture uses the same `items` schema with exactly three rows:

```json
{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"array","minItems":20,"maxItems":20,"items":{"type":"object","additionalProperties":false,"required":["id","image","prompt","keyword"],"properties":{"id":{"type":"string","minLength":1},"image":{"type":"string","minLength":1},"prompt":{"type":"string","minLength":1},"keyword":{"type":"string","minLength":1}}}}
```

- Concrete acceptance row: `{"id":"m3-001","image":"tests/fixtures/vl/images/m3-001.jpg","prompt":"What animal is shown?","keyword":"cat"}`. Before the harness diff, the PR author and M3 rig operator jointly freeze the issue #221 acceptance set as exactly 20 such rows in `tests/fixtures/vl/minimax_m3_keyword_eval.json`; that checked-in file is the record of IDs, repo-relative rig-local image paths, prompts, keywords, and order. No image bytes enter git. The rig operator places the 20 untracked bytes at the listed paths on the M3 checkout. Real mode reports each missing image as `SKIP` and does not download it.
- `tests/fixtures/vl/keyword_eval_cpu.json` has the exact rows `cpu-1`/`tests/fixtures/vl/images/cpu-1.jpg`/`Name the animal.`/`cat`, `cpu-2`/`tests/fixtures/vl/images/cpu-2.jpg`/`Name the color.`/`red`, and `cpu-3`/`tests/fixtures/vl/images/cpu-3.jpg`/`How many objects?`/`two`, encoded with the four keys above.
- `examples/vl_keyword_eval.py --fixture <path> --fake-answers '<json-object>'` parses an `id`-to-answer JSON object. Fake mode validates the fixture and scores those strings, but performs no image existence check and imports/constructs neither `AutoProcessor` nor `MoE`; real mode lazily loads them and follows `vl_example.py`'s `apply_chat_template` plus greedy `model.generate` path.
- Fake mode accepts `--fake-timings '<json-object>'` only with `--fake-answers`; every fixture ID maps to `{"ttft_ms":<number>,"decode_tokens_per_s":<number>}`. Missing IDs or non-finite/negative values are malformed input and return non-zero.
- The score gate is `ceil(0.8 * row_count)`: 16/20 for acceptance and 3/3 for the CPU fixture. Missing fake-answer IDs count as misses. A met gate returns 0; malformed fixtures/answers or a score below the gate return non-zero.

## Performance recording (not a gate)

- `benchmarks/minimax_m3/perf_summary.py` exports `percentile(values: list[float], p: float) -> float` and `summarize_performance(ttft_ms: list[float], decode_tokens_per_s: list[float]) -> dict[str, float | int]`. Use the linear interpolation in `benchmarks/serving/latency.py`: position `(p / 100) * (n - 1)`. Require non-empty equal-length lists and `p` in `[0, 100]`; raise `ValueError` for non-finite or negative samples.
- Real mode uses `time.perf_counter()` and a generate streamer as in `benchmarks/serving/baseline_performance.py`: synchronize CUDA before the start and after generation, TTFT is start-to-first-generated-token in ms, and decode tokens/s is `(generated_tokens - 1) / (end - first_token_at)` (0 when fewer than two tokens are generated). These are per-request values from the same scored run, with no warmup removal and no threshold.
- Record each scored row as `REQUEST_PERF id=<id> ttft_ms=<value> decode_tokens_per_s=<value>` with three-decimal floats; those are the exact per-request field names.
- After `KEYWORD_SCORE`, print `PERF_SUMMARY sample_count=<n> ttft_p50_ms=<value> ttft_p99_ms=<value> decode_tok_s_p50=<value> decode_tok_s_p99=<value>`. Those five names are the exact machine-readable output fields; use fixed three-decimal formatting for the four floats.
- Unit tests pass TTFT `[100.0, 200.0, 300.0]` and decode rates `[10.0, 20.0, 30.0]`; expected summary is `sample_count=3`, `ttft_p50_ms=200.0`, `ttft_p99_ms=298.0`, `decode_tok_s_p50=20.0`, `decode_tok_s_p99=29.8`. Parametrize NaN, infinities, and negative values to expect `ValueError`.

## QA gate

- CPU, direct pass: `python examples/vl_keyword_eval.py --fixture tests/fixtures/vl/keyword_eval_cpu.json --fake-answers '{"cpu-1":"cat","cpu-2":"red","cpu-3":"two"}' --fake-timings '{"cpu-1":{"ttft_ms":100,"decode_tokens_per_s":10},"cpu-2":{"ttft_ms":200,"decode_tokens_per_s":20},"cpu-3":{"ttft_ms":300,"decode_tokens_per_s":30}}'`; expected score `3/3`, `PERF_SUMMARY sample_count=3 ttft_p50_ms=200.000 ttft_p99_ms=298.000 decode_tok_s_p50=20.000 decode_tok_s_p99=29.800`, no model-load output, exit 0.
- CPU, direct fail: `python examples/vl_keyword_eval.py --fixture tests/fixtures/vl/keyword_eval_cpu.json --fake-answers '{"cpu-1":"cat","cpu-2":"wrong","cpu-3":"wrong"}'`; expected score `1/3`, exit non-zero.
- CPU, pytest: `python -m pytest tests/python/unit/test_vl_keyword_eval.py -q`; expected 3/3 and 1/3 subprocess assertions, model-load bypass, and the exact deterministic `PERF_SUMMARY` line pass, exit 0.
- CPU, perf math: `python -m pytest tests/python/unit/test_minimax_m3_perf_summary.py -q`; expected deterministic percentile/summary assertions and invalid-input `ValueError` cases pass, exit 0.
- Docs: `tests/python/unit/test_vl_model_compatibility_docs.py` reads `docs/model-compatibility.md` via `Path(__file__).resolve().parents[3]` and asserts the exact substrings `visual.*`, `MTP`, `unused`, `MiniMax-M3`, `MiniMax Community License`, `BF16`, `eager attention`, `native engine off`, and `no real-checkpoint evidence`. Run `python -m pytest tests/python/unit/test_vl_model_compatibility_docs.py -q`; expected exit 0.
- GPU, Qwen3.5: `python examples/vl_example.py --image tests/fixtures/images/cat.jpg --expect cat` still prints `PASS` and a non-zero dispatch counter; exit 0.
- GPU, M3, ≥1TB host RAM + NVMe: one cat smoke, then the 20-pair score. Paste both logs on #221; the second log must include the complete `PERF_SUMMARY` block, explicitly labeled there as the numeric BF16 performance baseline for Phases 5 (MXFP8 store) and 6 (indexer kernel). No performance threshold applies. That paste is what closes the checkpoint-evidence item.

## Non-goals

MXFP8 stores, Lightning Indexer kernels, video input, and turning `use_native_engine` back on for `minimax_m3_vl`.
