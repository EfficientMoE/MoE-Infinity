# Sub-agent prompt C — accelerate the cache simulator (parallel / GPU)

Copy into a fresh performance agent. It speeds up the expert-cache simulator for
235B-scale traces **without changing results**, and exposes a one-pass
miss-ratio-curve (MRC) engine that prompt B's capacity sweep can call.

See `../auto_research_policies.md` for the overall plan and shared guardrails.

## CONTEXT

Work in `benchmarks/cache_simulator/` (repo `EfficientMoE/MoE-Infinity`, PR
#288). The reference implementations are pure-Python and single-threaded:
`belady_simulate` (`belady_cache_simulator.py`) and `ExpertCache` /
`HeapExpertCache` (`cache_strategy.py`). Traces are SGLang `per_token` captures
of `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` (≈94 layers × millions of decode
tokens), so single-threaded Python is slow. `test_cache_simulator_belady.py`
contains an independent brute-force OPT oracle and a fuzz cross-check — these are
your correctness ground truth.

This task is independent of prompts A/B and can run in parallel from the start.

## TASK

1. **Profile first.** On a large trace, run `cProfile`/`py-spy` and report the
   hotspots (heap operations, the per-access Python loop, trace parsing) before
   changing anything.
2. **Vectorize** access-sequence construction and counting with NumPy/torch:
   encode keys as `layer * num_experts + expert` and batch the transforms.
3. **One-pass miss-ratio curve** (the big win — gives hit rate for *all*
   capacities at once, which is exactly prompt B's output):
   - **LRU** via reuse / stack distance (Mattson et al. 1970) using a
     Fenwick / order-statistics tree → full MRC in `O(N log N)`.
   - **OPT / Belady** via the forward (Belady) reuse distance. If an exact
     OPT-MRC is impractical, at minimum parallelize the per-capacity OPT and
     vectorize its inner loop.
   - For very large traces, offer an **approximate MRC** behind a flag (SHARDS,
     Waldspurger et al. FAST'15; or Counter Stacks, Wires et al. OSDI'14) and
     report the error bound.
   - Expose `miss_ratio_curve(accesses, capacities) -> {cap: hit_rate}` as the
     shared entry point prompt B calls.
4. **Parallelism**: `concurrent.futures` / `multiprocessing` across trace files
   and across capacity points (traces are independent). Document the scaling.
5. **GPU (optional, behind a flag)**: torch-on-GPU for frequency/score policies
   (`scatter_add` histograms), next-use precomputation (sort / segmented ops),
   and reuse-distance histograms. Fall back to CPU when no GPU is present.
6. **Correctness gate (non-negotiable).** The pure-Python `belady_simulate` /
   `ExpertCache` remain the **reference oracle**. Every accelerated path must
   produce **identical** hit/miss counts to the reference on the existing fuzz
   suite and the brute-force OPT oracle; any approximate MRC must stay within its
   stated error bound. Add these equivalence tests under `tests/python/unit/`.
7. **Benchmark** `research/bench_speed.py`: wall-clock + speedup (e.g. `Nx` on a
   10M-access trace) and peak memory for reference vs vectorized vs parallel vs
   GPU.

## DELIVERABLES

- Accelerated implementation(s) behind clear entry points (incl.
  `miss_ratio_curve`).
- Equivalence tests proving bit-for-bit parity with the reference (and bounded
  error for approximations).
- `research/bench_speed.py` + a short speedup / memory report.
- `CHANGELOG.md` `Unreleased` note; PR per template.

## ACCEPTANCE

- Accelerated paths match the reference exactly on the fuzz / OPT oracle; any
  approximation stays within a documented error bound.
- A measured speedup is reported with the command + hardware; results
  (hit rates) are unchanged.
- `miss_ratio_curve` is available and used by prompt B's sweep.

Follow repo conventions (feature branch, Angular `-s` commits, pre-commit
ruff/ruff-format/codespell, torch imported lazily, GPU path optional). Ask
before adding heavy deps (prefer stdlib + the numpy/torch already in the repo).
Do not report any speedup you did not measure.
