# Sub-agent prompt E - composable prefetch x eviction harness + timing model

Copy into a fresh implementation agent. It makes prefetch a policy of its own
that can be paired with any evictor, adds online predictors, admission modes
for prefetched experts and a transfer-timing model, and unit-tests them.

See `../auto_research_prefetch_eviction.md` for the plan, shared contracts and
guardrails.

## CONTEXT

`benchmarks/cache_simulator/policies.py` has the `CachePolicy` ABC, the
shared `simulate()` harness, on-demand evictors (LRU, LFU, LFU-K,
model-tracing, score, LHU, ARC, 2Q, LRU-K, LIRS, S3-FIFO, SIEVE, W-TinyLFU,
per-layer LRU, LCNU) and four trace-oracle prefetch policies that are
hard-wired to LRU eviction (`_OracleLRUPolicy`): prefetched keys are admitted
into the same cache, count as covered misses on first demand, and
`effective_hit_rate` reports demand hits plus covered misses. Real-trace
driver: `research/run_policies_real.py` (decode-only by default, parallel
workers). `research/lcnu_ablation.py` shows the tuning-half / held-out-half
protocol and an oracle-horizon LCNU. Prefetch survey (prompt D):
`research/prefetch_survey.md`.

## TASK

1. **`PrefetchPolicy` ABC** (new module, e.g. `prefetchers.py`): called once
   per access with the routed key and the layer-visit context; returns
   prefetch candidates `(key, score)` for the next `h` layer visits under a
   per-layer budget. It may read only the past plus the current token's
   routing up to the current layer.
2. **Online prefetchers** (at least 4): previous-token same-layer reuse;
   per-layer frequency / EMA top-k; cross-layer co-activation (an EAM-style
   matrix learned online from layer `L` -> `L+1..L+h` within a token, as
   MoE-Infinity does); a hybrid score combining them. Oracle bounds (true next
   `h` layers) for comparison, labelled as such.
3. **Combined policy**: `PrefetchCachePolicy(evictor, prefetcher,
   admission=...)` that composes any evictor with any prefetcher, without
   changing any existing class's behaviour. Admission modes: `same_queue`
   (current behaviour), `probation` (prefetched keys are first in line for
   eviction until their first demand use), `buffer` (separate prefetch buffer
   with its own slot budget, promoted into the cache on first use), and
   `do_no_harm` (prefetch only if the candidate's predicted next use is sooner
   than the evictor's chosen victim's; needs a next-use estimate, which LCNU
   provides and LRU/LFU can approximate). Evictors may need a "victim ranking"
   or "next eligible victim" hook; add it without changing results.
4. **Timing model** (`research/timing_model.py`): one serial host-to-GPU link;
   per-expert bytes (from the model config: expert params x bytes per param)
   and link bandwidth as parameters, per-layer compute time as a parameter;
   prefetches issued at layer `L` for layer `L + h` queue behind earlier
   transfers; a demand miss stalls compute until its transfer finishes.
   Report stall per token, total transfers, wasted prefetches. Defaults for
   Qwen3-235B-A22B-FP8 and Qwen3-30B-A3B on PCIe 5.0 x16, with the sources
   cited; keep them as parameters, not constants.
5. **Tests** under `tests/python/unit/`: demand hit rate <= Belady for every
   combination and admission mode; `hits + misses == len(accesses)`; capacity
   never exceeded (buffer included); online predictors are causal (changing
   the future of a trace does not change outputs at `t`); timing-model hand
   cases (zero prefetch = sum of miss stalls; perfect infinite-bandwidth
   prefetch = zero stall); existing policy tests unchanged.
6. Driver support in `research/run_policies_real.py` (or a sibling
   `run_prefetch_real.py`) to sweep evictor x prefetcher x admission x
   capacity in parallel workers, writing CSV + Markdown with every metric.

## DELIVERABLES

- Prefetcher module, combined policy, timing model, driver support, tests.
- `CHANGELOG.md` `Unreleased` entry.

## ACCEPTANCE

- `PrefetchCachePolicy(evictor, none)` reproduces each evictor's hit counts
  exactly on the real Qwen3-30B-A3B decode trace (bit-identical).
- At least 4 online prefetchers, 3 admission modes, oracle bounds, the timing
  model; all tests green; pre-commit clean.
