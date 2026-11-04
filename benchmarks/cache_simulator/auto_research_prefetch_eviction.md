# Round 2: should expert prefetch be a separate policy from eviction?

Orchestration hub for the second cache-policy research round on the offline
expert-cache simulator in `benchmarks/cache_simulator/` (repo
`EfficientMoE/MoE-Infinity`, tracking issue #289). Round 1
(`auto_research_policies.md`) built eviction baselines; its real-trace
follow-up (`research/decode_policy_study.md`, `research/recommendations.md`)
fixed the decode-stream extraction and found:

- On batch-1 decode the stream is a cyclic scan over layers: LRU gets 0 hits
  below one token's working set (`moe_layers x top_k`); LCNU (layer-cycle
  next-use) is the best online evictor on real Qwen3 traces.
- Giving LCNU the **true** routing of the current and next 8 layers adds at
  most 0.7 points of *demand* hit rate, so cross-layer routing prediction is a
  prefetch signal, not an eviction signal.
- In the simulator every prefetch policy is hard-wired to LRU eviction and
  admits prefetched experts into the same cache with demand priority
  (`policies._OracleLRUPolicy`), and the predictors are trace oracles.

Round 2 asks whether prefetch and eviction should be **two independent,
composable policies**, which **combinations** work best on real traces, and
how prefetched experts should be **admitted** relative to the evictor, judged
by stall time and transfer volume, not only by hit rate.

## Research questions

1. **Decoupling.** Is the best (prefetcher, evictor) pair close to the best
   evictor alone plus the best prefetcher alone (separable), or do the two
   interact enough that they must be co-designed?
2. **Combinations.** Rank eviction {LRU, LFU, LIRS, S3-FIFO, W-TinyLFU, LCNU,
   per-layer LRU} x prefetch {none, online predictors, oracle bounds} across
   cache sizes from below one token's working set to half the key space.
3. **Admission of prefetched experts.** Same queue with demand priority
   (today), probation (evict first unless used), a separate prefetch buffer
   with its own budget, or "do no harm" admission (prefetch only if the
   candidate's predicted next use is sooner than the victim's, the integrated
   prefetching-and-caching rule of Cao et al., SIGMETRICS'95). LCNU already
   produces a next-use estimate that makes the last one implementable online.
4. **Timing.** With a PCIe transfer model (bytes per expert, link bandwidth,
   per-layer compute time, one shared link), which pair minimizes decode stall
   per token, and what does each prefetcher cost in wasted transfers?
5. **Engine knobs.** Should the engine expose `expert_eviction_policy` and
   `expert_prefetch_policy` separately? Propose names, defaults and the
   admission rule.

## Prompts

| Prompt | Task |
| --- | --- |
| [`prompts/04_research_prefetch_eviction.md`](./prompts/04_research_prefetch_eviction.md) | **D** - literature: MoE expert prefetchers (what they predict, from which signal, how far ahead) and prefetch-aware replacement / integrated prefetching-and-caching theory -> `research/prefetch_survey.md`. No simulator code. |
| [`prompts/05_implement_prefetch_eviction.md`](./prompts/05_implement_prefetch_eviction.md) | **E** - composable `PrefetchPolicy` x evictor harness, online predictors, admission modes, timing/bandwidth model, tests. |
| [`prompts/06_benchmark_prefetch_eviction.md`](./prompts/06_benchmark_prefetch_eviction.md) | **F** - real-trace combination sweep, analysis, recommendations, engine-knob proposal -> `research/prefetch_eviction_study.md`. |

## Dependency graph and dispatch

- **D || E**: E can start from the predictors already known (previous-token
  reuse, per-layer frequency, cross-layer co-activation / EAM) while D finishes
  the survey; D's shortlist adds the rest.
- **E -> F**: F needs the harness and the timing model.
- Fan out independent paper reads (D) and independent predictor
  implementations (E) with `dispatching-parallel-agents`; land everything on
  one branch / PR.

## Shared contracts

- **Access stream**: decode-only `(layer, expert)` streams from
  `research/run_policies_real.load_accesses(files, ignore_layers=0,
  prefill="skip")`; `prefill="union"` for the serving variant. Never use the
  old merged-stream extraction (it leaked prefill; see the study).
- **Evictors**: the `CachePolicy` classes in `policies.py`. Extend them with a
  victim *ranking* or "next eligible victim" hook if admission modes need it;
  keep every existing class's behaviour bit-identical (tests).
- **Prefetcher interface**: online, sees only past routing and the current
  token's routing up to the current layer; returns `(layer, expert, score)`
  candidates for the next `h` layer visits under a per-layer budget. Oracle
  variants read the trace and are labelled upper bounds.
- **Metrics** per run: demand hit rate (bounded by Belady), prefetch coverage
  (misses whose fetch a prefetch hid), wasted prefetches (fetched, evicted
  unused), total transfers, and timing-model stall per token.
- **Timing model** (`research/timing_model.py`): parameters per model and
  link, documented with sources; serial link, prefetch issued at layer `L` for
  layer `L + h`, demand miss stalls until its transfer completes.
- **Experiment log**: append to `research/EXPERIMENTS.md` with provenance
  `real` and the stream (`/decode`, `/decode+prefill-union`).

## Shared guardrails

- Cite primary sources; never fabricate titles, authors, numbers or
  algorithms; record the command and environment for every number.
- Demand hit rate must stay `<= Belady` for every policy; prefetch value is
  reported separately (coverage, stall). Keep the brute-force OPT and Belady
  tests green.
- Online predictors must not read the future: unit-test that permuting the
  future of a trace does not change a predictor's output at time `t`.
- Pick hyper-parameters on the first half of the requests, report on the
  second half (as `research/lcnu_ablation.py` does).
- Shared host: at most ~32 worker processes; no GPU needed.
- Repo conventions: feature branch, Angular commits with `-s`, pre-commit
  (ruff, ruff-format, codespell), tests under `tests/python/unit/`, torch
  imported lazily, `CHANGELOG.md` `Unreleased`. Ask before adding heavy
  dependencies. Traces stay git-ignored.

## Acceptance

1. `research/prefetch_survey.md` with verified citations and a shortlist of
   implementable prefetchers and admission rules (D).
2. Composable prefetch x eviction harness with at least 4 online prefetchers,
   3 admission modes and the timing model, unit-tested (E).
3. Real-trace sweep on Qwen3-235B and Qwen3-30B-A3B decode (plus the union
   stream) with a clear answer to questions 1-5 and a recommended default
   pair, in `research/prefetch_eviction_study.md` (F).
4. All numbers reproducible from committed code; CHANGELOG updated; tests
   green.

Trace locations (git-ignored, on the capture host):
`/mnt/raid0nvme0/leyang/moeinf-traces/belady_qwen3_235b/*.pt` (100 requests),
`belady_qwen3_30b/*.pt` (150), `belady_smoke/*.pt` (12), plus merged
`belady_<model>_stream/stream.pt`. Simulator venv:
`/mnt/raid0nvme0/leyang/MoE-Infinity/.venv/bin/python`.
