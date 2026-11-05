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

## Prefetch is a separate decision

Giving LCNU the true routing of the current and next 8 layers adds at most
0.7 points of hit rate: cross-layer routing prediction does not help pick
eviction victims. Use it to prefetch, and choose the eviction policy
independently. Which prefetcher pairs best with which evictor is open; see
[`../auto_research_prefetch_eviction.md`](../auto_research_prefetch_eviction.md).

## In the simulator

```python
from benchmarks.cache_simulator.policies import build_policy, simulate

policy = build_policy("lcnu", 1024, num_layers=48, prior=8 / 128)
hits, misses, hit_rate = simulate(policy, accesses)
```

Sweeps: `research/run_policies_real.py --policies lcnu lru lirs ...`.
