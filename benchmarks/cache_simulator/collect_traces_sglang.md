# Agent prompt — collect Qwen3-235B-A22B expert-routing traces with SGLang 0.5.20

Copy everything below into a fresh agent that has a multi-GPU box (or 2-node
cluster) and network access. It tells the agent exactly what to capture, in
what format, and how to prove the traces are correct by running the simulator
on them.

---

## ROLE

You collect **expert-routing traces** for **`Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`**
served by **SGLang v0.5.20**, and save them in the exact format consumed by the
offline MoE expert-cache simulator in `benchmarks/cache_simulator/` (repo
`EfficientMoE/MoE-Infinity`, PR #288). Traces must cover **all input sequences**
of each target dataset so the simulator can run end-to-end on them.

Do not modify the simulator. Your output is trace files plus a short report.

## TARGET MODEL (verify constants from the HF `config.json`)

`Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`
- ~**94** decoder layers, **all MoE** (no dense layers) ⇒ simulator
  `--ignore-layers 0`.
- **128** routed experts per layer, **top_k = 8** activated. (This matches the
  recorder's hard-coded `_TOP_K_NUM = 8`, so there is no padding mismatch.)
- 235B total / 22B active params, **FP8** block-wise weights, context up to 262144.
- Cache-key space ≈ `94 × 128 = 12032` distinct `(layer, expert)` entries.

**FP8 note:** FP8 quantizes expert *weights*, not routing. `topk_ids` are the
same experts the bf16 model would pick, so the traces are valid for cache
analysis regardless of the FP8 checkpoint.

## BACKGROUND: what the simulator consumes

Two entry points, two formats (see `benchmarks/cache_simulator/README.md`):

1. `belady_cache_simulator.py` reads **`.pt`** via `torch.load`:
   ```
   { "records": [ record0, record1, ... ] }          # records[0]=prefill, records[1:]=decode
   record = { "topk_ids_of_layer": IntTensor[num_layers, seq_len, top_k], ... }  # -1 = unused slot
   ```
2. `cache_strategy.py` reads **`.pkl`** (`AccessSequence`):
   ```
   prefilling_<name>.pkl : { request_id: np.ndarray[num_layers, num_prompt_tokens] }
   decoding_<name>.pkl   : { request_id: [ np.ndarray[num_layers, 1, top_k], ... ] }  # one per decode step
   ```

**Key fact:** SGLang's `per_token` expert-distribution recorder already emits
format (1). Its `_DetailAccumulator.dump()` writes
`expert_distribution_recorder_<ts>_<rank>.pt` containing
`dict(records=[...], last_physical_to_logical_map=...)`, where each record has
`topk_ids_of_layer` of shape `[num_layers, num_tokens, top_k]`. So the primary
deliverable needs **zero conversion**.

## STEP 0 — SMOKE TEST ON A SMALL MODEL FIRST (do not skip)

Qwen3-235B is expensive. First validate the whole pipeline on a small MoE so
format/flags are correct before burning cluster time:

- `deepseek-ai/DeepSeek-V2-Lite-Chat` (1 dense layer ⇒ `--ignore-layers 1`) or
  `Qwen/Qwen1.5-MoE-A2.7B-Chat`.
- Run the capture + the ACCEPTANCE checks below. Only once a small-model `.pt`
  loads and `belady_cache_simulator.py` reports a sane hit rate do you proceed
  to the 235B run.

## SETUP

```bash
pip install "sglang[all]==0.5.20"     # pin this version
python -c "import sglang, inspect; from sglang.srt.eplb import expert_distribution as e; \
print(sglang.__version__); print(e.__file__)"
```
Confirm `get_global_expert_distribution_recorder`, the `per_token` detail
gatherer (`_DetailSinglePassGatherer`), and `_DetailAccumulator` exist. If
0.5.20's layout differs, locate the equivalents and note it in your report.

## ⚠️ KNOWN ISSUE — per_token dump (sgl-project/sglang #13206)

On this exact model, `--expert-distribution-recorder-mode stat` dumps fine, but
`per_pass`/`per_token` can raise `AssertionError: assert output_mode == "file"`
inside `_DetailAccumulator.dump`. The detail accumulator **only supports file
output** (it writes to `$SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR`); the error
means the dump was invoked in `object` mode.

Before the full run, **smoke-test the dump path** with one prompt and confirm a
`.pt` file actually appears on disk. If you hit the AssertionError:
- Ensure the dump is invoked with `output_mode="file"` (via the Engine method or
  endpoint variant that forces file mode), or
- Pick a 0.5.20 patch/commit where the detail dump endpoint passes file mode, or
- Apply a minimal local patch so the dump handler calls
  `dump_record(output_mode="file")` for detail accumulators.
Record exactly what you did in the report.

## CAPTURE — PRIMARY PATH (per_token recorder, multi-GPU)

### Config A (preferred for tracing): single node, TP, routing replicated

Keeping attention **non-DP** replicates the MoE gate across ranks, so routing
(`topk_ids`) is identical on every rank and **rank 0's dump is complete** — no
cross-rank token merge needed.

```bash
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/abs/path/traces_raw
export SGLANG_ENABLE_JIT_DEEPGEMM=1
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
    --trust-remote-code \
    --tp 8 --dp 1 \
    --expert-distribution-recorder-mode per_token \
    --disable-cuda-graph \
    --chunked-prefill-size 8192 \
    --mem-fraction-static 0.85 \
    --host 127.0.0.1 --port 30000
```
Needs ≈8×80GB (FP8 weights ≈235GB sharded over 8). If the recorder is a no-op
(no `ExpertLocationMetadata`), add `--enable-eplb --ep-size 8
--ep-num-redundant-experts 0` so metadata is built while keeping physical ==
logical, then re-verify (see pitfalls).

### Config B (only if A will not fit): EP / multi-node

Mirror the reference setup from #13206 (`--enable-eplb
--expert-distribution-recorder-mode per_token --enable-expert-distribution-metrics
--nnodes 2 --node-rank {0,1} --dist-init-addr <host:port> --disable-cuda-graph`).
With EP + DP attention you MUST handle both: physical→logical remap **and**
cross-rank token merge (see pitfalls). Prefer Config A.

### Driving the recorder + sending prompts

Per request (or per batch): start → generate → stop → dump.
```bash
curl -s 127.0.0.1:30000/start_expert_distribution_record
#   POST /generate or /v1/chat/completions with temperature=0 and fixed max_new_tokens
curl -s 127.0.0.1:30000/stop_expert_distribution_record
curl -s 127.0.0.1:30000/dump_expert_distribution_record     # writes the .pt (see KNOWN ISSUE)
```
Alternatively use the offline `sglang.Engine` and its
`start/stop/dump_expert_distribution_record` methods to script everything in one
Python process (still broadcasts to workers); ensure the dump uses file mode.

Rules that make dumps match the simulator's record model:
- **Batch size 1 for per-request traces:** start → one prompt → generate → stop
  → dump ⇒ that `.pt` has `records = [prefill, decode_1, …, decode_T]` for that
  request, exactly what the simulator expects.
- **One full-dataset serving-stream run** (natural batching, single dump at the
  end) ⇒ one large `.pt` of the interleaved forward-pass stream, for
  realistic aggregate-cache analysis. Keep both.
- Use `record["forward_mode"]` to confirm `records[0]` is prefill (`EXTEND`) and
  the rest are `DECODE`.

## CAPTURE — FALLBACK PATH (hook)

If per_token cannot be enabled/dumped for this model on 0.5.20, hook MoE top-k
instead: in `sglang/srt/layers/moe/topk.py`, `select_experts(...)` returns
`StandardTopKOutput(topk_weights, topk_ids, router_logits)` and the owning
`TopK` carries `layer_id`. Monkeypatch it (or
`sglang.srt.state_capturer.routed_experts.get_global_experts_capturer()`) to
append `(forward_pass_id, layer_id, topk_ids.cpu().int())`, then assemble the
same `{"records":[{"topk_ids_of_layer": [num_layers, seq_len, top_k]}]}`
yourself (pad empty slots with `-1`). Validate against the official recorder on
the small smoke-test model to prove identical schema and values.

## PITFALLS (handle these for 235B specifically)

- **Logical vs physical expert ids (mandatory check).** The cache key must be a
  stable `(layer, logical_expert)` with ids in `[0, 127]`. If you enabled EP/EPLB
  or any redundant experts, `topk_ids_of_layer` holds *physical* ids — remap them
  through the dumped `last_physical_to_logical_map` before saving. With
  `--ep-num-redundant-experts 0` the map is identity. **Verify** the max id ≤ 127
  per layer after processing.
- **DP attention token sharding (Config B).** With `--dp > 1`, each rank's record
  only covers the tokens it processed; merge per-rank `.pt` files by
  `forward_pass_id` + token position to reconstruct the full sequence. Config A
  (`--dp 1`) avoids this.
- **CUDA graph** suppresses per-token recording — keep `--disable-cuda-graph`.
- **Token buffer width.** The per-token gatherer sizes its buffer at
  `chunked_prefill_size * 8` tokens; set `--chunked-prefill-size` ≥ longest
  prompt, or cap prompt length, so prefill rows are not truncated.
- **Determinism.** `temperature=0`, fixed `max_new_tokens`, fixed prompt order.

## DATASETS / INPUT SEQUENCES

Cover the full set of prompts per dataset (goal: "all input sequences"). Use a
mix so routing diversity shows:
- **Chat / general**: ShareGPT first-turn user prompts (the `*_chat` lineage).
- **Knowledge / reasoning**: MMLU (per subject), GSM8K.
- Optionally FLAN / BIGBench subsets to mirror legacy trace names.

Per dataset: greedy (`temperature=0`), fixed `max_new_tokens` (record it, e.g.
128/256), iterate over **every** prompt in the chosen split. If a split is huge,
define a deterministic cap (e.g. first 2000 in fixed order) and document it.
Report prompt count, token-length distribution, and total decode steps.

## OUTPUT LAYOUT & NAMING

Trace data is git-ignored (`benchmarks/cache_simulator/.gitignore`). Put files
under a data dir (default `benchmarks/cache_simulator/data/`, or external via
`--data-dir`/`$MOE_SIM_DATA_DIR`).

```
<data_dir>/
  pt/qwen3-235b-a22b-fp8/<dataset>/req_00000.pt   # per-request dumps (batch=1)
  pt/qwen3-235b-a22b-fp8/<dataset>/stream.pt      # one full-dataset serving-stream dump
  prefilling_qwen3-235b_<dataset>.pkl             # optional, for cache_strategy.py
  decoding_qwen3-235b_<dataset>.pkl               # optional, for cache_strategy.py
  manifest.json
```
`manifest.json` per (dataset): SGLang version + launch flags, `num_layers`,
`top_k`, `num_experts`, validated `ignore_layers`, physical-vs-logical handling,
prompt count, `max_new_tokens`, produced files, and the dump workaround used.

### Conversion (use the provided converter, don't hand-roll)

`benchmarks/cache_simulator/convert_sglang_traces.py` turns raw SGLang dumps
into both simulator formats, including the physical→logical remap:

```bash
# Belady .pt inputs (remap + trim; per-file, or --merge into one access stream):
python -m benchmarks.cache_simulator.convert_sglang_traces \
    --inputs "<raw>/*.pt" --to belady --out <data_dir>/pt/qwen3-235b-a22b-fp8/<dataset>
python -m benchmarks.cache_simulator.convert_sglang_traces \
    --inputs "<raw>/*.pt" --to belady --out <data_dir>/pt/.../stream.pt --merge

# cache_strategy .pkl inputs (batch-size-1 dumps; one dump == one request):
python -m benchmarks.cache_simulator.convert_sglang_traces \
    --inputs "<raw>/req_*.pt" --to pkl --name qwen3-235b_<dataset> --out <data_dir>
```

- The converter remaps physical→logical via `last_physical_to_logical_map` by
  default; pass `--no-remap` only if ids are already logical.
- `--to pkl` requires batch-size-1 captures (no request ids are in the dump) and
  rejects multi-token decode passes.
- If you want a `cache_strategy.py` preset, add a `MODEL_CONFIG` entry with a
  `cache_size` chosen as a fraction of `94 × 128`.

## VALIDATION / ACCEPTANCE CRITERIA (include outputs in the report)

1. **Schema loads**:
   ```
   python -c "import torch; d=torch.load('<file>.pt', map_location='cpu'); \
   r=d['records']; print(len(r), r[0]['topk_ids_of_layer'].shape)"
   ```
   `topk_ids_of_layer` must be 3-D `[num_layers≈94, seq_len, top_k=8]`, integer,
   values in `[-1, 127]`.
2. **Belady simulator runs and is sane**:
   ```
   python -m benchmarks.cache_simulator.belady_cache_simulator \
       --files "<data_dir>/pt/qwen3-235b-a22b-fp8/<dataset>/*.pt" \
       --capacity <C> --ignore-layers 0
   ```
   Hit rate in `(0, 1)`. Sweep `--capacity` over e.g. `128,256,512,1024,2048,4096`
   and confirm hit rate is **non-decreasing** in capacity. Report the table.
3. **cache_strategy runs** (if pickles produced): decode hit rate in `(0, 1)`.
4. **Coverage**: distinct `(layer, expert)` keys approach `94 × 128` on a large
   dataset; decode steps per request match `max_new_tokens`.
5. **Reproducibility**: re-running one dataset with identical greedy flags yields
   identical prefill `topk_ids_of_layer`.

## DELIVERABLES

1. The trace tree under `<data_dir>` for every dataset you ran.
2. `manifest.json` as specified.
3. `REPORT.md`: environment (GPUs, driver, SGLang build), launch flags, the
   per_token dump workaround used, validated `ignore_layers`/`top_k`/`num_experts`,
   physical-vs-logical handling, the capacity→hit-rate tables, and anything
   skipped and why.

Validate the small-model smoke test first, then do one dataset end-to-end on
235B and validate before scaling to the rest. Ask for clarification only if the
model cannot be served at all in SGLang 0.5.20.
```
