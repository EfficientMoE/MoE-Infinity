# Sub-agent prompt B — implement baselines + benchmark vs Belady

Copy into a fresh implementation agent. It turns the baseline shortlist from
prompt A into runnable simulator policies, unit-tests them, and benchmarks them
against the Belady-optimal bound on the SGLang traces.

See `../auto_research_policies.md` for the overall plan and shared guardrails.

## CONTEXT

Work in `benchmarks/cache_simulator/` (repo `EfficientMoE/MoE-Infinity`, PR
#288). Existing pieces:
- `belady_cache_simulator.py` — Belady optimal = the **ceiling**
  (`belady_simulate(accesses, capacity)`), with the pure-Python
  `extract_access_sequence`.
- `cache_strategy.py` — `ExpertCache` / `HeapExpertCache` (Belady over `.pkl`),
  `ExpertIndex` key, `AccessSequence` parser.
- `fetch_engines.py` — `FetchEngineBase` + `StreamFetch` (FIFO),
  `StreamFetchLRU`, `LFUFetch`, `LFUKFetch` (prefetch via `_prefetch`, eviction
  via `_apply_replacement_policy`, frequency via `global_update`).
- `convert_sglang_traces.py` — makes `.pt`/`.pkl` traces.
- Tests in `tests/python/unit/test_cache_simulator_*.py`, including a
  brute-force OPT oracle you must keep green.

**Input:** the shortlist in `research/policy_survey.md` (prompt A). You may start
on the already-known policies (LRU, LFU, model-tracing, score-based,
single-layer-ahead speculative) before A finishes the long tail.

## TASK

1. **Policy interface.** Reuse/extend `FetchEngineBase` for prefetch+eviction
   policies. For pure admission/eviction policies (no prefetch), add a small
   `CachePolicy` ABC in a new `policies.py` with `on_access(key) -> hit: bool`
   plus `admit`/`evict` hooks, mirroring `ExpertCache`. Provide ONE shared
   harness that runs any policy over an access sequence and returns
   `(hits, misses, hit_rate)`.
2. **Implement the shortlist.** At minimum: model-tracing (BrainStorm),
   score-based eviction, LHU, and prefetch variants — single-layer-ahead
   speculative, stride-based (ProMoE-style), multi-layer-ahead (HOBBIT-style),
   activation-matrix (MoE-Infinity-style). Belady stays the bound; LRU/LFU/LFU-K
   already exist. For predictor-based policies, implement the **trace-oracle**
   form (use the recorded next-layer routing as a perfect predictor) and label
   it explicitly as an upper bound on that predictor.
3. **Tests** under `tests/python/unit/`: deterministic small hand-computed cases
   per policy, plus invariants on every trace: `hits + misses == len(accesses)`,
   capacity never exceeded, and `hit_rate(policy) <= hit_rate(Belady)`.
4. **Benchmark harness** `research/run_policies.py`: sweep cache capacity, run
   every policy over each model/dataset trace, emit a results table, and render
   a **hit-rate-vs-capacity (MRC-style) plot per model** using the
   `conference-plot` skill. Reproduce the MoE-Infinity-style comparison
   (policy × buffer slots → hit rate, Belady as "Ideal"). For large traces, call
   the one-pass MRC engine from prompt C if available (coordinate on its
   signature: `miss_ratio_curve(accesses, capacities) -> {cap: hit_rate}`);
   otherwise fall back to the per-capacity simulator.
5. Append every run to `research/EXPERIMENTS.md` (policy, model, dataset,
   capacity, hit rate, trace provenance, commit).

## DELIVERABLES

- The policy implementations + `CachePolicy` ABC/harness.
- Unit tests (all green, including the existing OPT oracle).
- `research/run_policies.py` + a results table and MRC plots.
- `CHANGELOG.md` `Unreleased` note; PR per template.

## ACCEPTANCE

- ≥6 baselines implemented and unit-tested; each satisfies
  `hit_rate(policy) <= hit_rate(Belady)` on every trace.
- `run_policies.py` reproduces a policy×capacity hit-rate table + MRC plots on
  the Qwen3-235B traces with Belady as the ceiling.
- All numbers come from real runs; attach the command + environment + trace
  provenance for each.

Follow repo conventions (feature branch, Angular `-s` commits, pre-commit
ruff/ruff-format/codespell, torch imported lazily). Ask before adding heavy
deps. Git-ignore traces and large figures; commit code, small result tables,
and the plot sources.
