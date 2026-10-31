# Accelerating the expert-cache simulator (Task C)

Part of tracking issue #289 / prompt
[`prompts/03_accelerate_simulator.md`](../prompts/03_accelerate_simulator.md).
Goal: speed up the pure-Python Belady / LRU simulator for 235B-scale traces
**without changing results**, and expose a one-pass miss-ratio-curve (MRC)
engine (`miss_ratio_curve`) that the capacity sweep in prompt B can call.

> **All numbers below are synthetic.** There are no real Qwen3-235B traces in
> the repo, so every measurement uses the reproducible synthetic generator in
> [`synthetic.py`](./synthetic.py) (Zipf-skewed expert popularity over
> `94 layers x 128 experts`, `topk=8`, fixed seed). Each table states the exact
> command and the hardware it ran on.

## Hardware / environment

```
python=3.12.3, cpu=x86_64, cores=128,
torch=2.11.0+cu130, gpu=NVIDIA RTX PRO 6000 Blackwell Server Edition (2x)
numpy=2.4.4
```

## 1. Profiling the reference first

Command:

```
python -m benchmarks.cache_simulator.research.profile_reference \
    --accesses 2000000 --capacity 2000 --seed 0
```

`cProfile` over `extract_access_sequence` + `belady_simulate` on a 2M-access
synthetic trace (one capacity), top functions by cumulative time:

| function | tottime (s) | ncalls | note |
| --- | ---: | ---: | --- |
| `belady_simulate` (per-access loop) | 1.22 | 1 | the dominant cost |
| `extract_access_sequence` (Python loop) | 0.36 | 1 | trace parsing |
| `heapq.heappush` | 0.41 | 2.0M | one push per access |
| `build_future_positions` | 0.25 | 1 | per-key future queues |
| `deque.popleft` | 0.19 | 2.0M | next-use advance |
| `heapq.heappop` | 0.18 | 0.39M | eviction |

**Hotspots:** (1) the per-access Python loop in `belady_simulate` plus its heap
operations, and (2) the per-element Python loop in `extract_access_sequence`.
A capacity *sweep* re-pays the whole `belady_simulate` cost once per capacity,
so the number of capacity points multiplies the reference cost.

## 2. What was accelerated

- **Vectorized access-sequence construction** (`vectorized_access_sequence`):
  encode keys as `layer * num_experts + expert` and flatten records with NumPy
  instead of a per-element Python loop. Bit-for-bit identical to
  `extract_access_sequence` (fuzz test). Measured **5.6x** on a 3.64M-access
  synthetic trace (`0.333s -> 0.060s`).
- **One-pass LRU MRC** (`lru_mrc`): exact LRU stack/reuse distance (Mattson et
  al. 1970) via a Fenwick tree in `O(N log N)` -> hit rate for *all* capacities
  from a single pass. Constant cost regardless of how many capacity points the
  sweep asks for.
- **OPT / Belady MRC** (`opt_mrc`, `opt_mrc_parallel`): the costly next-use
  precompute is vectorized once (stable argsort) and shared across capacities;
  each capacity then runs the exact heap-based Belady simulation, parallelized
  across capacities with `ProcessPoolExecutor`. Bit-for-bit identical to
  `belady_simulate` and to the brute-force OPT oracle.
- **Optional GPU path** (`lru_mrc_gpu`, torch, lazy import): the reuse-distance
  histogram/cumsum is reduced on GPU via `scatter_add_`; transparent CPU
  fallback when no CUDA device is present. The stack-distance pass is inherently
  sequential and stays on CPU, so the histogram offload is not the bottleneck at
  these trace sizes.
- **SHARDS approximate LRU MRC** (`shards_lru_mrc`, behind a flag): fixed-rate
  spatial hash sampling (Waldspurger et al., FAST'15). See the accuracy caveat
  below -- it is *not* recommended for this domain.

## 3. Measured speedups

### 3a. 2M-access trace, 12 capacities (reference feasible)

Command:

```
python -m benchmarks.cache_simulator.research.bench_speed \
    --accesses 2000000 --seed 0 \
    --capacities 64 128 192 256 384 512 768 1024 1536 2048 3072 4096
```

| method | wall (s) | peak RSS delta (MB) | speedup vs reference |
| --- | ---: | ---: | ---: |
| reference (pure-Python, per-capacity) | 20.02 | 7.1 | 1.0x |
| vectorized `opt_mrc` | 17.35 | 108.4 | 1.2x |
| **parallel `opt_mrc`** | **3.66** | ~0 | **5.5x** |
| `lru` one-pass (exact) | 8.78 | ~0 | 2.3x |
| `lru` one-pass (GPU histogram) | 8.89 | ~0 | 2.3x |
| `lru` SHARDS (1% sample, approx) | 0.07 | 11.9 | 305x |

Parity: `reference`, `vectorized opt_mrc` and `parallel opt_mrc` produce
**identical** hit rates for every capacity (checked inline by the benchmark and
by the equivalence tests).

The LRU one-pass cost is *constant* in the number of capacity points (one pass
builds the whole curve), so its advantage over the reference grows with the
width of the sweep; at 12 capacities it is already 2.3x and the reference keeps
scaling linearly with capacity count.

### 3b. 10M-access trace (reference impractical)

Command:

```
python -m benchmarks.cache_simulator.research.bench_speed \
    --accesses 10000000 --seed 0 --skip-reference \
    --capacities 64 128 256 512 1024 2048 4096
```

| method | wall (s) | peak RSS delta (MB) |
| --- | ---: | ---: |
| vectorized `opt_mrc` (7 caps) | 52.08 | 1016 |
| **parallel `opt_mrc` (7 caps)** | **11.75** | ~0 |
| `lru` one-pass (exact, whole curve) | 49.35 | ~0 |
| `lru` one-pass (GPU histogram) | 49.78 | ~0 |
| `lru` SHARDS (1% sample, approx) | 0.37 | ~0 |

The pure-Python reference was not run at 10M (its per-capacity cost makes the
full sweep take tens of minutes; this is exactly the pain the accelerated paths
remove). Against the vectorized serial `opt_mrc`, the parallel path is ~4.4x.

## 4. Parallel scaling

The capacities are independent, so `opt_mrc_parallel` forks one Belady
simulation per capacity point. On the 2M trace the serial `opt_mrc` is 17.4s and
the parallel version is 3.66s across 12 capacities (~4.7x), saturating once the
number of capacities approaches the number of cores. Parallelism across *trace
files* is equally independent and composes the same way (run one process per
file).

## 5. SHARDS accuracy caveat (honest)

SHARDS was designed for storage/block traces with **millions of distinct
addresses**. The expert-cache domain has a *tiny* key universe
(`num_layers * num_experts`, order 10^4), so at low sampling rates the rescaled
stack distances are badly quantized and the estimate degrades sharply at small
and mid capacities. On the synthetic traces here the fixed-rate SHARDS
hit-rate error is large (tens of percent) at low sampling rates, so **SHARDS is
not recommended for this domain**. It remains behind a flag, returns a
conservative `max(0.05, 4/sqrt(sample_size))` bound, and is exact at
`sampling_rate=1.0`. For MoE expert caches, prefer the exact one-pass `lru_mrc`
and the parallel `opt_mrc`.

## 6. Correctness gate (non-negotiable)

Equivalence tests live in
[`tests/python/unit/test_cache_simulator_mrc.py`](../../../tests/python/unit/test_cache_simulator_mrc.py):

- `opt_mrc` / `opt_mrc_parallel` == `belady_simulate` hit rate for every
  capacity, over 400 fuzz traces, and == the brute-force offline optimum over
  120 fuzz traces.
- `lru_mrc` / `lru_mrc_gpu` == an independent LRU reference over 400 fuzz
  traces; monotone non-decreasing in capacity; never beats OPT.
- `vectorized_access_sequence` == `extract_access_sequence` (NumPy and torch
  inputs).
- SHARDS is exact at full sampling and returns valid rates / a positive bound.

All `tests/python/unit/test_cache_simulator_*` tests stay green.
