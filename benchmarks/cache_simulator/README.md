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

The cache key throughout is `(layer_idx, expert_id)` and one cache entry holds
one expert's weights.

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
