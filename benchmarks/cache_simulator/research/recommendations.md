# Expert-cache policy: recommendations

What to run, based on the real-trace studies:

- [`decode_policy_study.md`](./decode_policy_study.md): eviction, round 1;
- [`prefetch_eviction_study.md`](./prefetch_eviction_study.md): prefetch x
  eviction, round 2;
- [`default_policy_study.md`](./default_policy_study.md): better predictors
  and one default, round 3.

The traces are Qwen3-235B-A22B-FP8, Qwen3-30B-A3B and Qwen1.5-MoE-A2.7B SGLang
captures, batch-1 greedy decode. "Working set" means one decode step's
experts, `moe_layers x top_k` (752 for Qwen3-235B, 384 for Qwen3-30B-A3B);
"key space" means `moe_layers x experts`.

## Default policy (one setting for every cache size)

| Component | Setting |
| --- | --- |
| Evictor | `lcnu_p`: LCNU (`alpha = 0.3`, `beta = 0.5`, `prior = top_k / experts`); prefill passes give no routing evidence |
| Prefetcher | `knn`: the 8 past decode tokens whose last 8 layers routed most like the current token vote for the next layer's experts; weight `overlap^8`, bank of 8192 tokens per layer |
| Horizon / budget | 1 layer ahead; 8 candidates; at most 2 transfers issued per layer |
| Admission | `probation`: prefetched experts are staged and charged to the cache, evicted first, dropped at their layer's gate if not routed, promoted on use |

Held-out results, over 2 models x {decode, decode + prefill} x 6 capacities
from 1/3 of the working set to half the key space:

- **Regret:** within 0.5% on average, and 6.2% at worst, of the best online
  configuration found for each cell in rounds 2 and 3.
- **Against LCNU alone (decode):** 15-31% less stall per token
  (Qwen3-30B-A3B) and 9-16% (Qwen3-235B).
- **Against LFU-like eviction with today's same-id prefetch:** 31-62% and
  18-50% less stall.
- **Link discipline:** the same with a FIFO link as with demand priority.

Parameters were selected on the first half of the Qwen3-30B-A3B requests and
reused unchanged.

## Why these choices

- **Eviction dominates below one token's working set.** Decode is a cyclic
  scan over layers: LRU, ARC, SIEVE and LFU-K get 0 hits there. LCNU keeps
  about 2/3 of Belady's hits and removes 21-35% of LRU's stall.
- **With a good predictor, prefetch matters as much or more above it.** kNN
  routing-trajectory matching predicts the next layer with recall@8 of
  0.78 / 0.71, against 0.61 / 0.56 for co-activation and 0.44 / 0.39 for
  previous-token reuse. Its prefetches remove 19-31% of LCNU's stall above
  1.3x the working set on Qwen3-30B-A3B. On Qwen3-235B, whose link moves
  fewer experts per layer of compute, they remove 9-16% at every size.
- **The two policies are separable.** The best evictor alone plus the best
  prefetcher alone is within 1.6% / 1.0% of the jointly best pair (round 2),
  as long as prefetched experts are staged outside the evictor. Same-queue
  admission lets LCNU and LFU evict prefetches before use.
- **Prefill passes must not count as routing evidence.**
  - For eviction: `lcnu_p` removes 9-24% of LCNU's stall in large caches when
    prefill is in the stream. It beats the frequency filters that round 1
    recommended there (W-TinyLFU, LFU) except on Qwen3-30B-A3B at half the
    key space (W-TinyLFU 4.65 ms vs 4.94 ms per token).
  - For prediction: kNN learns from decode passes only. Wide prefill
    signatures would otherwise crowd out the real neighbours.
- **Do not use the same-expert-id rule** (today's `speculative_prefetch`).
  Expert ids carry no meaning across layers; its precision is chance (0.06).

## When to deviate

- **Memory-constrained host:** kNN's bank is 1 MB per layer (48 / 94 MB).
  `coactivation` / `hybrid` (round 2) needs 64 KB per layer, but it cuts
  stall by only 3-12% against LCNU alone, where kNN cuts 9-31%.
- **Eviction only, no prefetch:** LCNU (`lcnu_p` with prefill in the stream).
  Without LCNU, round 1's next best are LIRS or W-TinyLFU below the working
  set, LIRS or LRU-2 at 1-2x, and S3-FIFO or LRU above. Above ~2/3 of the
  key space every recency-based policy is within ~0.4 points.
- **Weakly skewed routing** (Qwen1.5-MoE-A2.7B: top 20% of keys carry 30% of
  accesses, against ~59% on Qwen3): every policy stays far from Belady and the
  differences shrink to 0.2-1.7 points.
- **Batched decode** (several tokens per layer visit) weakens the cyclic-scan
  effect; no study has measured it.

## In the simulator

```python
from benchmarks.cache_simulator.policies import simulate
from benchmarks.cache_simulator.prefetchers import build_composed, build_prefetcher

prefetcher = build_prefetcher(
    "knn", horizon=1, budget=8, num_layers=48, num_experts=128,
    context=8, neighbors=8, power=8, bank=8192, decode_width=8,
)
policy = build_composed(
    "lcnu_p", 1024, prefetcher, admission="probation", issue_budget=2,
    num_layers=48, prior=8 / 128, top_k=8,
)
hits, misses, hit_rate = simulate(policy, accesses)  # demand hits only
```

Stall per token: `research/timing_model.py` (`replay_policy`). Sweeps:
`research/run_policies_real.py` (eviction) and `research/run_prefetch_real.py`
(prefetch x eviction x admission).
