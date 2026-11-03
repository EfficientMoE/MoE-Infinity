# Expert-cache eviction: recommendations

What to run, based on the real-trace study in
[`decode_policy_study.md`](./decode_policy_study.md) (Qwen3-235B-A22B-FP8,
Qwen3-30B-A3B and Qwen1.5-MoE-A2.7B SGLang captures, batch-1 greedy decode).
"Working set" below means one decode step's experts, `moe_layers x top_k`
(752 for Qwen3-235B, 384 for Qwen3-30B-A3B); "key space" means
`moe_layers x experts`.

## Default: LCNU

Use `lcnu` (layer-cycle next-use) with `alpha = 0.3`, `beta = 0.5` and
`prior = top_k / experts`. It had the best decode hit rate of every online
policy at every capacity on Qwen3-235B and Qwen1.5-MoE-A2.7B, and on
Qwen3-30B-A3B up to half the key space. It needs per-expert counters and the
index of the layer being executed, both available to an offloading engine.
The parameters transferred across models: the Qwen3-235B tuning split picked
the same values as Qwen3-30B-A3B, and the ten best grid points are within 1.6
points of each other.

## By cache size (decode-dominated serving)

| Cache size | Use | Avoid | Why |
| --- | --- | --- | --- |
| At most the working set minus `top_k` | LCNU; else LIRS or W-TinyLFU | LRU, ARC, SIEVE, LFU-K (0 hits); 2Q, S3-FIFO (near 0) | decode is a cyclic scan over layers; recency evicts every expert just before its layer returns |
| 1-2x the working set | LCNU; else LIRS or LRU-2 | SIEVE, 2Q | loop-resistant policies beat LRU by up to 8 points here |
| 2x the working set to ~1/2 of the key space | LCNU; else S3-FIFO, per-layer LRU or LRU | LFU, offline popularity profiles, SIEVE | frequency alone misses token-to-token reuse; LCNU removes ~7-25% of LRU's misses |
| Above ~2/3 of the key space | anything recency-based (LRU, S3-FIFO, LIRS) | — | all within ~0.4 points; pick the cheapest |

## Workload adjustments

- **Prefill-heavy serving** (many requests, long prompts) **with a large cache**
  (above ~1/3 of the key space): every prefill pass touches thousands of
  experts once. W-TinyLFU and LFU filter that scan and beat LCNU by up to 1.9
  points there. Prefer keeping prefill fetches out of the cache (transient
  admission) and LCNU for decode; otherwise W-TinyLFU.
- **Weakly skewed routing** (Qwen1.5-MoE-A2.7B: top 20% of keys carry 30% of
  accesses, against ~59% on Qwen3): every policy stays far from Belady and the
  differences shrink to 0.2-1.7 points.
- **Batched decode** (several tokens per layer visit) weakens the cyclic-scan
  effect; the study did not measure it.

## Prefetch: a separate policy, joined by an admission rule

From the round-2 study, [`prefetch_eviction_study.md`](./prefetch_eviction_study.md),
on held-out requests with a PCIe 5.0 timing model:

- **Choose the evictor and the prefetcher independently.** The best
  evictor alone plus the best prefetcher alone is within 1.6% (Qwen3-30B-A3B)
  and 1.0% (Qwen3-235B) of the jointly best pair on decode. Prefetch does not
  change which evictor is best: keep LCNU.
- **Prefetcher: cross-layer co-activation, one layer ahead.** Score the
  experts of layer `L+1` by `P(e' at L+1 | experts just routed at L)`,
  learned online from routing. The `hybrid` variant adds previous-token and
  EMA terms with weights 1/1/16. Its recall@8 is 0.61 / 0.56, against 0.44 /
  0.39 for previous-token reuse. Budget per layer: 4 below ~1.5x the working
  set, 8 up to ~1/3 of the key space, 16 above.
- **Do not use the same-expert-id rule** (today's `speculative_prefetch`).
  Expert ids carry no meaning across layers; its precision is chance (0.06),
  and it adds 5-27% stall.
- **Admit prefetched experts outside the evictor.**
  - `probation` for LCNU, LRU and LFU: stage them, drop them at the target
    layer's gate if not routed, evict them first under pressure.
  - A fixed `buffer` for W-TinyLFU and S3-FIFO.
  - `do_no_harm` when the pair was not tuned together.
  - Never same-queue with LCNU or LFU: they evict the prefetch before it is
    used.
- **Expect modest gains.** LCNU + co-activation cuts stall per token by
  3-12% against LCNU alone (10-40% against LFU). One serial link can only
  hide the per-token compute (9 / 21 ms). A perfect predictor would remove
  54-94% at 5-8x the working set, where routing statistics capture under 10%
  of it.
- **Link scheduling matters as much as the policy.** Demand fetches must
  overtake queued prefetches. Queued prefetches go earliest layer first and
  are dropped once their layer has run.

| Cache size | Evictor | Prefetcher (h, budget) | Admission |
| --- | --- | --- | --- |
| below the working set | LCNU | co-activation / hybrid (1, 4) | probation |
| 1-3x the working set | LCNU | co-activation / hybrid (1, 4-8) | probation |
| 5x the working set to 1/2 of the key space | LCNU | co-activation / hybrid (1-2, 8-16) | probation or pin |
| prefill-heavy, above 1/3 of the key space | W-TinyLFU or LFU | previous-token / hybrid (1, 8-16) | probation, pin or buffer |

## In the simulator

```python
from benchmarks.cache_simulator.policies import build_policy, simulate
from benchmarks.cache_simulator.prefetchers import build_composed, build_prefetcher

policy = build_policy("lcnu", 1024, num_layers=48, prior=8 / 128)
hits, misses, hit_rate = simulate(policy, accesses)

prefetcher = build_prefetcher(
    "hybrid", horizon=1, budget=8, num_layers=48, num_experts=128,
    alpha=0.4, weights=(1, 1, 16),
)
composed = build_composed(
    "lcnu", 1024, prefetcher, admission="probation", num_layers=48, prior=8 / 128
)
simulate(composed, accesses)  # demand hits; composed.prefetch_covered, ...
```

Stall per token: `research/timing_model.py` (`replay_policy`). Sweeps:
`research/run_policies_real.py --policies lcnu lru lirs ...` (eviction) and
`research/run_prefetch_real.py` (prefetch x eviction x admission).
