# MoE expert-cache simulator

Offline simulator for the GPU **expert cache** used by MoE-Infinity's
offloading runtime. It replays recorded expert-activation traces through
different cache-replacement / prefetch policies and reports the resulting
cache **hit rate** — no GPU and no live model required. Use it to upper-bound
achievable hit rate (Belady optimal) and to compare practical policies before
implementing them in the runtime.

Ported from the Archer / MoE-Infinity prefetch study (`Archer-rel/simulation`).

## Contents

| File | What it does |
| --- | --- |
| `belady_cache_simulator.py` | Belady optimal hit-rate simulator over `.pt` recorder traces (CLI). Pure-Python core (`extract_access_sequence`, `belady_simulate`) reusable as a library. |
| `cache_strategy.py` | Belady optimal expert cache over pickled prefill/decode activation traces (CLI). Provides `ExpertIndex`, `ExpertCache` (reference scan) and `HeapExpertCache` (heap, for large traces). |
| `fetch_engines.py` | Prefetch/replacement policy library: `StreamFetch` (FIFO), `StreamFetchLRU` (LRU), `LFUFetch` (LFU), `LFUKFetch` (windowed LFU). |
| `policies.py` | `CachePolicy` ABC + baseline shortlist over a flat access sequence: LRU, LFU, LFU-K, model-tracing (BrainStorm), score-based, LHU (HOBBIT), plus trace-oracle prefetch variants (single-layer-ahead, stride, multi-layer, activation-matrix). One shared `simulate()` harness. |
| `research/run_policies.py` | Benchmark harness: sweeps capacity, runs every policy over synthetic MoE traces, emits a policy x capacity hit-rate table + MRC plots with Belady as "Ideal". |

The cache key throughout is `(layer_idx, expert_id)` and one cache entry holds
one expert's weights.

## Cache-policy baselines (`policies.py`)

`policies.py` adds a small `CachePolicy` ABC that replays a flat
`(layer, expert)` access sequence (the output of
`extract_access_sequence`) through a demand-paging cache and reports
`(hits, misses, hit_rate)` via a single shared `simulate()` harness. Each
policy's **demand** hit rate is directly comparable to `belady_simulate` on the
same trace and is bounded above by it on every trace.

| Policy | Class | Eviction / prefetch rule | Source lineage |
| --- | --- | --- | --- |
| LRU | `LRUPolicy` | evict least-recently-used | DeepSpeed-Inference / Mixtral base |
| LFU | `LFUPolicy` | evict least-frequent (LRU tie-break) | classic |
| LFU-K | `LFUKPolicy` | LFU over a sliding window of the last `k` accesses | windowed LFU |
| Model-tracing | `ModelTracingPolicy` | evict globally least-popular (offline profile counts) | BrainStorm (OSDI'23) |
| Score-based | `ScorePolicy` | evict smallest time-decayed accumulated activation score | ExpertFlow / Specter |
| LHU | `LHUPolicy` | evict coldest "hot unit" (aged frequency + recency) | HOBBIT (arXiv:2411.01433) |
| 1-layer-ahead* | `SingleLayerAheadPolicy` | prefetch the next layer's experts | Mixtral-Offloading / Pre-gated MoE |
| Stride* | `StridePrefetchPolicy` | prefetch the access `stride` steps ahead | ProMoE (arXiv:2410.22134) |
| Multi-layer* | `MultiLayerAheadPolicy` | prefetch the union over the next `horizon` layers | HOBBIT |
| Activation-matrix* | `ActivationMatrixPolicy` | prefetch top co-activated next-layer experts | MoE-Infinity EAM |

`*` = **trace-oracle** prefetch policy: the predictor is realised as a *perfect*
next-layer routing lookahead read from the trace, so the reported numbers are an
**upper bound** on any real gate/hidden-state predictor. Prefetch hides a
compulsory miss's fetch stall but does not remove the access, so a speculatively
prefetched expert counts as a (covered) miss on first demand. The demand hit
rate therefore stays bounded by Belady; the prefetch value shows up in
`policy.effective_hit_rate` (demand hits + prefetch-covered misses), reported
separately as the predictor's upper bound.

```python
from benchmarks.cache_simulator.policies import build_policy, simulate

accesses = [(3, 0), (3, 1), (4, 2), (3, 0)]
policy = build_policy("model_tracing", capacity=2)
hits, misses, hit_rate = simulate(policy, accesses)
```

### Benchmark harness

```bash
python -m benchmarks.cache_simulator.research.run_policies \
    --out benchmarks/cache_simulator/research
```

This sweeps cache capacity and runs every policy over three **synthetic** MoE
routing workloads (there are no real Qwen3-235B captures on this machine). It
writes a policy x capacity hit-rate table (`results_synthetic.md` /
`results_synthetic.csv`), appends rows to `research/EXPERIMENTS.md`, and renders
one hit-rate-vs-capacity (MRC-style) plot per model with Belady drawn as
"Ideal". Figures are git-ignored; the tables and `run_policies.py` are the
committed artifacts. Regenerate figures with the command above.

## Install

Only `torch`, `numpy`, and (optionally) `rich`/`tqdm` for progress bars:

```bash
pip install torch numpy rich tqdm
```

`rich` and `tqdm` are optional; the modules fall back to plain output when they
are missing.

## Usage

### Belady optimal over `.pt` recorder traces

```bash
python -m benchmarks.cache_simulator.belady_cache_simulator \
    --files "data/expert_distribution_recorder_*.pt" \
    --capacity 360 \
    --ignore-layers 3
```

- `--capacity`: cache size in `(layer, expert)` entries.
- `--ignore-layers`: skip leading dense (non-MoE) layers.
- `--include-prefill`: also count the prefill step (default: decode only).

### Belady optimal over pickled activation traces

```bash
export MOE_SIM_DATA_DIR=/path/to/traces
python -m benchmarks.cache_simulator.cache_strategy \
    --model DeepSeek-V2-Lite-Chat \
    --max-samples 100 \
    --strategy heap
```

`--data-dir` (or `$MOE_SIM_DATA_DIR`) is prepended to the trace filenames in
`MODEL_CONFIG`. Presets: `DeepSeek-V2-Lite-Chat`, `SwitchTransformers`,
`NLLB`, `Arctic`, `Mixtral`.

### As a library

```python
from benchmarks.cache_simulator import belady_simulate

accesses = [(3, 0), (3, 1), (3, 0), (4, 2), (3, 0)]
hits, misses, hit_rate = belady_simulate(accesses, capacity=2)
```

## Converting SGLang traces

If you capture routing with SGLang's `per_token` expert-distribution recorder
(`--expert-distribution-recorder-mode per_token`), its `.pt` dumps already use
the `records` schema below, so the Belady path is almost pass-through.
`convert_sglang_traces.py` does the two adaptations that remain:

- **physical → logical** expert-id remap via `last_physical_to_logical_map`
  (needed when EP/EPLB added redundant experts; a no-op otherwise), and
- the **`.pkl` export** for `cache_strategy.py` (one batch-size-1 dump per
  request).

```bash
# Belady inputs (remap + trim; one .pt per dump, or --merge into one stream):
python -m benchmarks.cache_simulator.convert_sglang_traces \
    --inputs "traces_raw/*.pt" --to belady --out data/belady

# cache_strategy inputs (capture with batch size 1; one dump == one request):
python -m benchmarks.cache_simulator.convert_sglang_traces \
    --inputs "traces_raw/req_*.pt" --to pkl --name qwen3-235b_mmlu --out data
```

See [`collect_traces_sglang.md`](./collect_traces_sglang.md) for the full
SGLang capture + conversion runbook.

## Runbooks (agent prompts)

Reusable prompts for driving this work with coding agents:

- [`collect_traces_sglang.md`](./collect_traces_sglang.md) — capture expert
  routing traces from SGLang 0.5.20 (Qwen3-235B-A22B-Instruct-2507-FP8).
- [`auto_research_policies.md`](./auto_research_policies.md) — orchestration hub
  for researching cache/prefetch policies into baselines and accelerating the
  simulator. It dispatches three sub-agent prompts in
  [`prompts/`](./prompts/): `01_research_baselines.md` (literature → survey),
  `02_implement_baselines.md` (implement + benchmark vs Belady), and
  `03_accelerate_simulator.md` (parallel / GPU acceleration).

## Trace formats

**`.pt` recorder trace** (`belady_cache_simulator.py`) — a dict with a
`records` list. `records[0]` is prefill, `records[1:]` are decode steps. Each
record has `topk_ids_of_layer`, a tensor of shape
`[num_layers, seq_len, topk]` holding the activated expert ids (`-1` = none).
These are produced by the expert-distribution recorder while serving a model;
they are regenerated per run and are **not** checked in.

**Pickled activation trace** (`cache_strategy.py`) — a pickled dict mapping a
request id to its per-layer activated-expert array. The prefill trace is
`[num_layers, ...]` and the decode trace is a list of per-iteration arrays of
shape `[num_layers, topk]` or `[num_layers, 1, topk]`.

> **Prefill accounting differs between the two tools.** `cache_strategy.py`
> de-duplicates experts per layer within a prefill step (`np.unique`), while
> `belady_cache_simulator.py` replays each activation as a separate access, so
> experts repeated within a step count as additional hits. Both tools skip
> `-1` padding. The two prefill hit-rate numbers are therefore not directly
> comparable.

## Trace data

Trace files are large and are **git-ignored** (see `.gitignore`). They are not
stored in this repo. The reference traces live on **gala1**:

```
xly@gala1:~/Archer-rel/simulation/
```

| File | Size | Notes |
| --- | --- | --- |
| `prefilling.pkl` | ~10.2 GB | prefill activation trace |
| `prefilling_chat.pkl` | ~4.2 GB | DeepSeek-V2-Lite-Chat prefill |
| `decoding.pkl` | ~312 MB | decode activation trace |
| `decoding_chat.pkl` | ~467 MB | DeepSeek-V2-Lite-Chat decode |
| `measurement_results/*_expert_activation.pkl` | 1–12 MB each | Switch / NLLB encoder+decoder traces across flan / mmlu / bigbench |

Fetch the small per-model traces (skip the multi-GB prefill dumps):

```bash
mkdir -p data
rsync -av \
    xly@gala1:~/Archer-rel/simulation/measurement_results/ \
    data/measurement_results/

# Or a single DeepSeek decode trace:
scp xly@gala1:~/Archer-rel/simulation/decoding_chat.pkl data/
```

Then point the simulator at them with `--data-dir data` (or
`export MOE_SIM_DATA_DIR=data`).
