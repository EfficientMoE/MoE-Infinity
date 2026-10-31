# Experiment log — expert-cache policies

Append-only log for the cache-policy study. One row per policy run (or seed row
for a policy not yet run). Task A (survey) seeds the policy / source / taxonomy
columns from `policy_survey.md`; Task B fills in hit rates and commits per the
shared contract in `../auto_research_policies.md`
(`run_policies.py -> (hits, misses, hit_rate)`, MRC engine owned by Task C).

Every on-line policy must satisfy `hit_rate(policy) <= hit_rate(Belady)` on
every trace (Belady is the "Ideal" ceiling from `belady_simulate`).

## Columns

`policy` · `source` · `taxonomy class (eviction / prefetch)` · `predictor` ·
`model` · `capacity` · `hit_rate` · `commit`

Runs use the Qwen3-235B-A22B-Instruct-2507-FP8 `per_token` traces
(`../collect_traces_sglang.md`) unless noted; `capacity` is in
`(layer, expert)` entries; `hit_rate`/`commit` are TBD until Task B runs them.

## Seed rows (from the Task A shortlist)

| policy | source | eviction class | prefetch class | predictor | model | capacity | hit_rate | commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Belady / OPT (Ideal) | belady_simulate / ExpertCache | OPT (bound) | — | oracle (bound) | Qwen3-235B | sweep | TBD | TBD |
| B1 LRU + on-demand | Mixtral-Offloading (2312.17238); DeepSpeed-Inference (2207.00032) | LRU | none | none | Qwen3-235B | sweep | TBD | TBD |
| B2 LFU (lifetime) + on-demand | MoE-Infinity (2401.14361); SwapMoE (2308.15030) | LFU | none | none | Qwen3-235B | sweep | TBD | TBD |
| B3 Windowed LFU (LFU-K) + on-demand | MoE-Infinity (2401.14361) | LFU-K / windowed | none | none | Qwen3-235B | sweep (k) | TBD | TBD |
| B4 Model-tracing keep-hottest | Brainstorm (OSDI'23); Lina (2210.17223); MoE-Infinity (2401.14361) | model-tracing (global counts) | none | none (offline-global = mild oracle) | Qwen3-235B | sweep | TBD | TBD |
| B5 Score-based (LHU-degraded, precision-free) | HOBBIT (2411.01433); SwapMoE (2308.15030); Specter (2511.14102) | score-based | none | none | Qwen3-235B | sweep | TBD | TBD |
| B6 Stride / FIFO-streaming prefetch | SE-MoE (2205.10034); ProMoE (2410.22134); DeepSpeed-Inference (2207.00032) | FIFO / LRU | stride / prefetch-all (positional) | none | Qwen3-235B | sweep (stride) | TBD | TBD |
| B7 Activation-matrix (EAM) prefetch | MoE-Infinity (2401.14361) | LRU/LFU | activation-matrix (trace-based) | self (trace prefix) | Qwen3-235B | sweep (m) | TBD | TBD |
| B8 One-layer-ahead exact-prefetch oracle | Pre-gated MoE (2308.12066); Mixtral-Offloading (2312.17238) | LRU | exact pre-gating / single-layer speculative | oracle (1 layer) | Qwen3-235B | sweep | TBD | TBD |
| B9 Multi-layer-ahead (h) prefetch oracle + cascade | ProMoE (2410.22134); HOBBIT (2411.01433); SpecMD (2602.03921) | LRU | multi-layer-ahead + fetch cascade | oracle (h layers) | Qwen3-235B | sweep (h) | TBD | TBD |
| B10 Data-aware per-input oracle | SiDA-MoE (2310.18859); ExpertFlow (2410.17954) | data-aware (predicted-inactive) | learned predictor (hash) | oracle (per-input) | Qwen3-235B | sweep | TBD | TBD |

Notes:
- B4 has two variants — online-prefix (faithful) and offline-global (uses whole
  trace counts, a mild oracle); label each run.
- B7 realizes the EAM causally from the trace prefix (faithful) or from the whole
  trace (oracle); label each run.
- B8/B9/B10 are trace oracles that upper-bound predictor-based prefetch; the
  deployed systems differ only in predictor accuracy, which the routing trace
  cannot measure (see `policy_survey.md` "oracle-only" notes).
- Precision tiers, expert substitution, sub-expert granularity, compute
  placement (CPU expert exec), and token reordering are out of scope for this
  cache model (see per-system flags in `policy_survey.md`).
