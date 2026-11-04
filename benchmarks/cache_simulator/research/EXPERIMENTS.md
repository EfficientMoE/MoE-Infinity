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

## Task B — synthetic baseline runs (#291)

Rows appended by `benchmarks/cache_simulator/research/run_policies.py`. All traces are synthetic (seeded); no real Qwen3-235B captures are used.

| policy | model | dataset | capacity | hit_rate | eff_hit_rate | provenance | commit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| belady | qwen3-235b-synthetic | synthetic | 256 | 0.3341 | 0.3341 | synthetic | b0aff90 |
| lru | qwen3-235b-synthetic | synthetic | 256 | 0.0000 | 0.0000 | synthetic | b0aff90 |
| lfu | qwen3-235b-synthetic | synthetic | 256 | 0.2251 | 0.2251 | synthetic | b0aff90 |
| lfu_k | qwen3-235b-synthetic | synthetic | 256 | 0.0000 | 0.0000 | synthetic | b0aff90 |
| model_tracing | qwen3-235b-synthetic | synthetic | 256 | 0.2434 | 0.2434 | synthetic | b0aff90 |
| score | qwen3-235b-synthetic | synthetic | 256 | 0.0000 | 0.0000 | synthetic | b0aff90 |
| lhu | qwen3-235b-synthetic | synthetic | 256 | 0.2249 | 0.2249 | synthetic | b0aff90 |
| single_layer_ahead | qwen3-235b-synthetic | synthetic | 256 | 0.0000 | 0.9999 | synthetic | b0aff90 |
| stride_prefetch | qwen3-235b-synthetic | synthetic | 256 | 0.0000 | 0.9999 | synthetic | b0aff90 |
| multi_layer_ahead | qwen3-235b-synthetic | synthetic | 256 | 0.0000 | 0.9999 | synthetic | b0aff90 |
| activation_matrix | qwen3-235b-synthetic | synthetic | 256 | 0.0000 | 0.5459 | synthetic | b0aff90 |
| belady | qwen3-235b-synthetic | synthetic | 512 | 0.5828 | 0.5828 | synthetic | b0aff90 |
| lru | qwen3-235b-synthetic | synthetic | 512 | 0.0000 | 0.0000 | synthetic | b0aff90 |
| lfu | qwen3-235b-synthetic | synthetic | 512 | 0.3536 | 0.3536 | synthetic | b0aff90 |
| lfu_k | qwen3-235b-synthetic | synthetic | 512 | 0.0000 | 0.0000 | synthetic | b0aff90 |
| model_tracing | qwen3-235b-synthetic | synthetic | 512 | 0.3731 | 0.3731 | synthetic | b0aff90 |
| score | qwen3-235b-synthetic | synthetic | 512 | 0.0000 | 0.0000 | synthetic | b0aff90 |
| lhu | qwen3-235b-synthetic | synthetic | 512 | 0.3536 | 0.3536 | synthetic | b0aff90 |
| single_layer_ahead | qwen3-235b-synthetic | synthetic | 512 | 0.0000 | 0.9999 | synthetic | b0aff90 |
| stride_prefetch | qwen3-235b-synthetic | synthetic | 512 | 0.0000 | 0.9999 | synthetic | b0aff90 |
| multi_layer_ahead | qwen3-235b-synthetic | synthetic | 512 | 0.0000 | 0.9999 | synthetic | b0aff90 |
| activation_matrix | qwen3-235b-synthetic | synthetic | 512 | 0.0000 | 0.5459 | synthetic | b0aff90 |
| belady | qwen3-235b-synthetic | synthetic | 1024 | 0.7181 | 0.7181 | synthetic | b0aff90 |
| lru | qwen3-235b-synthetic | synthetic | 1024 | 0.5332 | 0.5332 | synthetic | b0aff90 |
| lfu | qwen3-235b-synthetic | synthetic | 1024 | 0.4968 | 0.4968 | synthetic | b0aff90 |
| lfu_k | qwen3-235b-synthetic | synthetic | 1024 | 0.5332 | 0.5332 | synthetic | b0aff90 |
| model_tracing | qwen3-235b-synthetic | synthetic | 1024 | 0.5230 | 0.5230 | synthetic | b0aff90 |
| score | qwen3-235b-synthetic | synthetic | 1024 | 0.5332 | 0.5332 | synthetic | b0aff90 |
| lhu | qwen3-235b-synthetic | synthetic | 1024 | 0.4968 | 0.4968 | synthetic | b0aff90 |
| single_layer_ahead | qwen3-235b-synthetic | synthetic | 1024 | 0.5332 | 0.9999 | synthetic | b0aff90 |
| stride_prefetch | qwen3-235b-synthetic | synthetic | 1024 | 0.5332 | 0.9999 | synthetic | b0aff90 |
| multi_layer_ahead | qwen3-235b-synthetic | synthetic | 1024 | 0.5331 | 0.9999 | synthetic | b0aff90 |
| activation_matrix | qwen3-235b-synthetic | synthetic | 1024 | 0.0593 | 0.5647 | synthetic | b0aff90 |
| belady | qwen3-235b-synthetic | synthetic | 2048 | 0.8085 | 0.8085 | synthetic | b0aff90 |
| lru | qwen3-235b-synthetic | synthetic | 2048 | 0.6718 | 0.6718 | synthetic | b0aff90 |
| lfu | qwen3-235b-synthetic | synthetic | 2048 | 0.6382 | 0.6382 | synthetic | b0aff90 |
| lfu_k | qwen3-235b-synthetic | synthetic | 2048 | 0.6718 | 0.6718 | synthetic | b0aff90 |
| model_tracing | qwen3-235b-synthetic | synthetic | 2048 | 0.6770 | 0.6770 | synthetic | b0aff90 |
| score | qwen3-235b-synthetic | synthetic | 2048 | 0.6718 | 0.6718 | synthetic | b0aff90 |
| lhu | qwen3-235b-synthetic | synthetic | 2048 | 0.6382 | 0.6382 | synthetic | b0aff90 |
| single_layer_ahead | qwen3-235b-synthetic | synthetic | 2048 | 0.6718 | 0.9999 | synthetic | b0aff90 |
| stride_prefetch | qwen3-235b-synthetic | synthetic | 2048 | 0.6718 | 0.9999 | synthetic | b0aff90 |
| multi_layer_ahead | qwen3-235b-synthetic | synthetic | 2048 | 0.6717 | 0.9999 | synthetic | b0aff90 |
| activation_matrix | qwen3-235b-synthetic | synthetic | 2048 | 0.6254 | 0.7543 | synthetic | b0aff90 |
| belady | qwen3-235b-synthetic | synthetic | 4096 | 0.8694 | 0.8694 | synthetic | b0aff90 |
| lru | qwen3-235b-synthetic | synthetic | 4096 | 0.7884 | 0.7884 | synthetic | b0aff90 |
| lfu | qwen3-235b-synthetic | synthetic | 4096 | 0.7768 | 0.7768 | synthetic | b0aff90 |
| lfu_k | qwen3-235b-synthetic | synthetic | 4096 | 0.7884 | 0.7884 | synthetic | b0aff90 |
| model_tracing | qwen3-235b-synthetic | synthetic | 4096 | 0.8176 | 0.8176 | synthetic | b0aff90 |
| score | qwen3-235b-synthetic | synthetic | 4096 | 0.7884 | 0.7884 | synthetic | b0aff90 |
| lhu | qwen3-235b-synthetic | synthetic | 4096 | 0.7768 | 0.7768 | synthetic | b0aff90 |
| single_layer_ahead | qwen3-235b-synthetic | synthetic | 4096 | 0.7884 | 0.9999 | synthetic | b0aff90 |
| stride_prefetch | qwen3-235b-synthetic | synthetic | 4096 | 0.7884 | 0.9999 | synthetic | b0aff90 |
| multi_layer_ahead | qwen3-235b-synthetic | synthetic | 4096 | 0.7883 | 0.9999 | synthetic | b0aff90 |
| activation_matrix | qwen3-235b-synthetic | synthetic | 4096 | 0.7683 | 0.8330 | synthetic | b0aff90 |
| belady | qwen3-235b-synthetic | synthetic | 6016 | 0.8888 | 0.8888 | synthetic | b0aff90 |
| lru | qwen3-235b-synthetic | synthetic | 6016 | 0.8448 | 0.8448 | synthetic | b0aff90 |
| lfu | qwen3-235b-synthetic | synthetic | 6016 | 0.8420 | 0.8420 | synthetic | b0aff90 |
| lfu_k | qwen3-235b-synthetic | synthetic | 6016 | 0.8448 | 0.8448 | synthetic | b0aff90 |
| model_tracing | qwen3-235b-synthetic | synthetic | 6016 | 0.8708 | 0.8708 | synthetic | b0aff90 |
| score | qwen3-235b-synthetic | synthetic | 6016 | 0.8448 | 0.8448 | synthetic | b0aff90 |
| lhu | qwen3-235b-synthetic | synthetic | 6016 | 0.8420 | 0.8420 | synthetic | b0aff90 |
| single_layer_ahead | qwen3-235b-synthetic | synthetic | 6016 | 0.8448 | 0.9999 | synthetic | b0aff90 |
| stride_prefetch | qwen3-235b-synthetic | synthetic | 6016 | 0.8448 | 0.9999 | synthetic | b0aff90 |
| multi_layer_ahead | qwen3-235b-synthetic | synthetic | 6016 | 0.8448 | 0.9999 | synthetic | b0aff90 |
| activation_matrix | qwen3-235b-synthetic | synthetic | 6016 | 0.8382 | 0.8812 | synthetic | b0aff90 |
| belady | mixtral-8x7b-synthetic | synthetic | 16 | 0.2327 | 0.2327 | synthetic | b0aff90 |
| lru | mixtral-8x7b-synthetic | synthetic | 16 | 0.0000 | 0.0000 | synthetic | b0aff90 |
| lfu | mixtral-8x7b-synthetic | synthetic | 16 | 0.1250 | 0.1250 | synthetic | b0aff90 |
| lfu_k | mixtral-8x7b-synthetic | synthetic | 16 | 0.0000 | 0.0000 | synthetic | b0aff90 |
| model_tracing | mixtral-8x7b-synthetic | synthetic | 16 | 0.1303 | 0.1303 | synthetic | b0aff90 |
| score | mixtral-8x7b-synthetic | synthetic | 16 | 0.0000 | 0.0000 | synthetic | b0aff90 |
| lhu | mixtral-8x7b-synthetic | synthetic | 16 | 0.0323 | 0.0323 | synthetic | b0aff90 |
| single_layer_ahead | mixtral-8x7b-synthetic | synthetic | 16 | 0.0000 | 0.9999 | synthetic | b0aff90 |
| stride_prefetch | mixtral-8x7b-synthetic | synthetic | 16 | 0.0000 | 0.9997 | synthetic | b0aff90 |
| multi_layer_ahead | mixtral-8x7b-synthetic | synthetic | 16 | 0.0000 | 0.9999 | synthetic | b0aff90 |
| activation_matrix | mixtral-8x7b-synthetic | synthetic | 16 | 0.0000 | 0.7742 | synthetic | b0aff90 |
| belady | mixtral-8x7b-synthetic | synthetic | 32 | 0.4669 | 0.4669 | synthetic | b0aff90 |
| lru | mixtral-8x7b-synthetic | synthetic | 32 | 0.0000 | 0.0000 | synthetic | b0aff90 |
| lfu | mixtral-8x7b-synthetic | synthetic | 32 | 0.2582 | 0.2582 | synthetic | b0aff90 |
| lfu_k | mixtral-8x7b-synthetic | synthetic | 32 | 0.0000 | 0.0000 | synthetic | b0aff90 |
| model_tracing | mixtral-8x7b-synthetic | synthetic | 32 | 0.2669 | 0.2669 | synthetic | b0aff90 |
| score | mixtral-8x7b-synthetic | synthetic | 32 | 0.0000 | 0.0000 | synthetic | b0aff90 |
| lhu | mixtral-8x7b-synthetic | synthetic | 32 | 0.1255 | 0.1255 | synthetic | b0aff90 |
| single_layer_ahead | mixtral-8x7b-synthetic | synthetic | 32 | 0.0000 | 0.9999 | synthetic | b0aff90 |
| stride_prefetch | mixtral-8x7b-synthetic | synthetic | 32 | 0.0000 | 0.9997 | synthetic | b0aff90 |
| multi_layer_ahead | mixtral-8x7b-synthetic | synthetic | 32 | 0.0000 | 0.9999 | synthetic | b0aff90 |
| activation_matrix | mixtral-8x7b-synthetic | synthetic | 32 | 0.0000 | 0.7742 | synthetic | b0aff90 |
| belady | mixtral-8x7b-synthetic | synthetic | 64 | 0.7213 | 0.7213 | synthetic | b0aff90 |
| lru | mixtral-8x7b-synthetic | synthetic | 64 | 0.5679 | 0.5679 | synthetic | b0aff90 |
| lfu | mixtral-8x7b-synthetic | synthetic | 64 | 0.4374 | 0.4374 | synthetic | b0aff90 |
| lfu_k | mixtral-8x7b-synthetic | synthetic | 64 | 0.5679 | 0.5679 | synthetic | b0aff90 |
| model_tracing | mixtral-8x7b-synthetic | synthetic | 64 | 0.4487 | 0.4487 | synthetic | b0aff90 |
| score | mixtral-8x7b-synthetic | synthetic | 64 | 0.5679 | 0.5679 | synthetic | b0aff90 |
| lhu | mixtral-8x7b-synthetic | synthetic | 64 | 0.4688 | 0.4688 | synthetic | b0aff90 |
| single_layer_ahead | mixtral-8x7b-synthetic | synthetic | 64 | 0.3695 | 0.8053 | synthetic | b0aff90 |
| stride_prefetch | mixtral-8x7b-synthetic | synthetic | 64 | 0.0373 | 0.4406 | synthetic | b0aff90 |
| multi_layer_ahead | mixtral-8x7b-synthetic | synthetic | 64 | 0.1644 | 0.7652 | synthetic | b0aff90 |
| activation_matrix | mixtral-8x7b-synthetic | synthetic | 64 | 0.0012 | 0.7741 | synthetic | b0aff90 |
| belady | mixtral-8x7b-synthetic | synthetic | 96 | 0.8220 | 0.8220 | synthetic | b0aff90 |
| lru | mixtral-8x7b-synthetic | synthetic | 96 | 0.6582 | 0.6582 | synthetic | b0aff90 |
| lfu | mixtral-8x7b-synthetic | synthetic | 96 | 0.5702 | 0.5702 | synthetic | b0aff90 |
| lfu_k | mixtral-8x7b-synthetic | synthetic | 96 | 0.6582 | 0.6582 | synthetic | b0aff90 |
| model_tracing | mixtral-8x7b-synthetic | synthetic | 96 | 0.5842 | 0.5842 | synthetic | b0aff90 |
| score | mixtral-8x7b-synthetic | synthetic | 96 | 0.6582 | 0.6582 | synthetic | b0aff90 |
| lhu | mixtral-8x7b-synthetic | synthetic | 96 | 0.5866 | 0.5866 | synthetic | b0aff90 |
| single_layer_ahead | mixtral-8x7b-synthetic | synthetic | 96 | 0.6566 | 0.9989 | synthetic | b0aff90 |
| stride_prefetch | mixtral-8x7b-synthetic | synthetic | 96 | 0.6488 | 0.9896 | synthetic | b0aff90 |
| multi_layer_ahead | mixtral-8x7b-synthetic | synthetic | 96 | 0.6543 | 0.9988 | synthetic | b0aff90 |
| activation_matrix | mixtral-8x7b-synthetic | synthetic | 96 | 0.0074 | 0.7743 | synthetic | b0aff90 |
| belady | mixtral-8x7b-synthetic | synthetic | 128 | 0.8830 | 0.8830 | synthetic | b0aff90 |
| lru | mixtral-8x7b-synthetic | synthetic | 128 | 0.7352 | 0.7352 | synthetic | b0aff90 |
| lfu | mixtral-8x7b-synthetic | synthetic | 128 | 0.6830 | 0.6830 | synthetic | b0aff90 |
| lfu_k | mixtral-8x7b-synthetic | synthetic | 128 | 0.7352 | 0.7352 | synthetic | b0aff90 |
| model_tracing | mixtral-8x7b-synthetic | synthetic | 128 | 0.6964 | 0.6964 | synthetic | b0aff90 |
| score | mixtral-8x7b-synthetic | synthetic | 128 | 0.7352 | 0.7352 | synthetic | b0aff90 |
| lhu | mixtral-8x7b-synthetic | synthetic | 128 | 0.6901 | 0.6901 | synthetic | b0aff90 |
| single_layer_ahead | mixtral-8x7b-synthetic | synthetic | 128 | 0.7342 | 0.9996 | synthetic | b0aff90 |
| stride_prefetch | mixtral-8x7b-synthetic | synthetic | 128 | 0.7310 | 0.9953 | synthetic | b0aff90 |
| multi_layer_ahead | mixtral-8x7b-synthetic | synthetic | 128 | 0.7327 | 0.9991 | synthetic | b0aff90 |
| activation_matrix | mixtral-8x7b-synthetic | synthetic | 128 | 0.0486 | 0.7778 | synthetic | b0aff90 |
| belady | mixtral-8x7b-synthetic | synthetic | 192 | 0.9561 | 0.9561 | synthetic | b0aff90 |
| lru | mixtral-8x7b-synthetic | synthetic | 192 | 0.8783 | 0.8783 | synthetic | b0aff90 |
| lfu | mixtral-8x7b-synthetic | synthetic | 192 | 0.8588 | 0.8588 | synthetic | b0aff90 |
| lfu_k | mixtral-8x7b-synthetic | synthetic | 192 | 0.8783 | 0.8783 | synthetic | b0aff90 |
| model_tracing | mixtral-8x7b-synthetic | synthetic | 192 | 0.8700 | 0.8700 | synthetic | b0aff90 |
| score | mixtral-8x7b-synthetic | synthetic | 192 | 0.8783 | 0.8783 | synthetic | b0aff90 |
| lhu | mixtral-8x7b-synthetic | synthetic | 192 | 0.8597 | 0.8597 | synthetic | b0aff90 |
| single_layer_ahead | mixtral-8x7b-synthetic | synthetic | 192 | 0.8780 | 0.9997 | synthetic | b0aff90 |
| stride_prefetch | mixtral-8x7b-synthetic | synthetic | 192 | 0.8768 | 0.9982 | synthetic | b0aff90 |
| multi_layer_ahead | mixtral-8x7b-synthetic | synthetic | 192 | 0.8777 | 0.9998 | synthetic | b0aff90 |
| activation_matrix | mixtral-8x7b-synthetic | synthetic | 192 | 0.7347 | 0.9178 | synthetic | b0aff90 |
| belady | mixtral-8x7b-synthetic | synthetic | 256 | 0.9900 | 0.9900 | synthetic | b0aff90 |
| lru | mixtral-8x7b-synthetic | synthetic | 256 | 0.9900 | 0.9900 | synthetic | b0aff90 |
| lfu | mixtral-8x7b-synthetic | synthetic | 256 | 0.9900 | 0.9900 | synthetic | b0aff90 |
| lfu_k | mixtral-8x7b-synthetic | synthetic | 256 | 0.9900 | 0.9900 | synthetic | b0aff90 |
| model_tracing | mixtral-8x7b-synthetic | synthetic | 256 | 0.9900 | 0.9900 | synthetic | b0aff90 |
| score | mixtral-8x7b-synthetic | synthetic | 256 | 0.9900 | 0.9900 | synthetic | b0aff90 |
| lhu | mixtral-8x7b-synthetic | synthetic | 256 | 0.9900 | 0.9900 | synthetic | b0aff90 |
| single_layer_ahead | mixtral-8x7b-synthetic | synthetic | 256 | 0.9900 | 0.9999 | synthetic | b0aff90 |
| stride_prefetch | mixtral-8x7b-synthetic | synthetic | 256 | 0.9900 | 0.9997 | synthetic | b0aff90 |
| multi_layer_ahead | mixtral-8x7b-synthetic | synthetic | 256 | 0.9900 | 0.9999 | synthetic | b0aff90 |
| activation_matrix | mixtral-8x7b-synthetic | synthetic | 256 | 0.9900 | 0.9950 | synthetic | b0aff90 |
| belady | uniform-stress-synthetic | synthetic | 16 | 0.1827 | 0.1827 | synthetic | b0aff90 |
| lru | uniform-stress-synthetic | synthetic | 16 | 0.0000 | 0.0000 | synthetic | b0aff90 |
| lfu | uniform-stress-synthetic | synthetic | 16 | 0.0286 | 0.0286 | synthetic | b0aff90 |
| lfu_k | uniform-stress-synthetic | synthetic | 16 | 0.0000 | 0.0000 | synthetic | b0aff90 |
| model_tracing | uniform-stress-synthetic | synthetic | 16 | 0.0406 | 0.0406 | synthetic | b0aff90 |
| score | uniform-stress-synthetic | synthetic | 16 | 0.0000 | 0.0000 | synthetic | b0aff90 |
| lhu | uniform-stress-synthetic | synthetic | 16 | 0.0286 | 0.0286 | synthetic | b0aff90 |
| single_layer_ahead | uniform-stress-synthetic | synthetic | 16 | 0.0000 | 0.9998 | synthetic | b0aff90 |
| stride_prefetch | uniform-stress-synthetic | synthetic | 16 | 0.0000 | 0.9996 | synthetic | b0aff90 |
| multi_layer_ahead | uniform-stress-synthetic | synthetic | 16 | 0.0000 | 0.2504 | synthetic | b0aff90 |
| activation_matrix | uniform-stress-synthetic | synthetic | 16 | 0.0000 | 0.2744 | synthetic | b0aff90 |
| belady | uniform-stress-synthetic | synthetic | 32 | 0.2973 | 0.2973 | synthetic | b0aff90 |
| lru | uniform-stress-synthetic | synthetic | 32 | 0.0000 | 0.0000 | synthetic | b0aff90 |
| lfu | uniform-stress-synthetic | synthetic | 32 | 0.0606 | 0.0606 | synthetic | b0aff90 |
| lfu_k | uniform-stress-synthetic | synthetic | 32 | 0.0000 | 0.0000 | synthetic | b0aff90 |
| model_tracing | uniform-stress-synthetic | synthetic | 32 | 0.0802 | 0.0802 | synthetic | b0aff90 |
| score | uniform-stress-synthetic | synthetic | 32 | 0.0000 | 0.0000 | synthetic | b0aff90 |
| lhu | uniform-stress-synthetic | synthetic | 32 | 0.0606 | 0.0606 | synthetic | b0aff90 |
| single_layer_ahead | uniform-stress-synthetic | synthetic | 32 | 0.0000 | 0.9998 | synthetic | b0aff90 |
| stride_prefetch | uniform-stress-synthetic | synthetic | 32 | 0.0000 | 0.9996 | synthetic | b0aff90 |
| multi_layer_ahead | uniform-stress-synthetic | synthetic | 32 | 0.0000 | 0.9998 | synthetic | b0aff90 |
| activation_matrix | uniform-stress-synthetic | synthetic | 32 | 0.0000 | 0.4207 | synthetic | b0aff90 |
| belady | uniform-stress-synthetic | synthetic | 64 | 0.4398 | 0.4398 | synthetic | b0aff90 |
| lru | uniform-stress-synthetic | synthetic | 64 | 0.1801 | 0.1801 | synthetic | b0aff90 |
| lfu | uniform-stress-synthetic | synthetic | 64 | 0.1237 | 0.1237 | synthetic | b0aff90 |
| lfu_k | uniform-stress-synthetic | synthetic | 64 | 0.1801 | 0.1801 | synthetic | b0aff90 |
| model_tracing | uniform-stress-synthetic | synthetic | 64 | 0.1582 | 0.1582 | synthetic | b0aff90 |
| score | uniform-stress-synthetic | synthetic | 64 | 0.1801 | 0.1801 | synthetic | b0aff90 |
| lhu | uniform-stress-synthetic | synthetic | 64 | 0.1237 | 0.1237 | synthetic | b0aff90 |
| single_layer_ahead | uniform-stress-synthetic | synthetic | 64 | 0.0339 | 0.9016 | synthetic | b0aff90 |
| stride_prefetch | uniform-stress-synthetic | synthetic | 64 | 0.0029 | 0.8081 | synthetic | b0aff90 |
| multi_layer_ahead | uniform-stress-synthetic | synthetic | 64 | 0.0027 | 0.9909 | synthetic | b0aff90 |
| activation_matrix | uniform-stress-synthetic | synthetic | 64 | 0.0004 | 0.4207 | synthetic | b0aff90 |
| belady | uniform-stress-synthetic | synthetic | 128 | 0.6066 | 0.6066 | synthetic | b0aff90 |
| lru | uniform-stress-synthetic | synthetic | 128 | 0.2952 | 0.2952 | synthetic | b0aff90 |
| lfu | uniform-stress-synthetic | synthetic | 128 | 0.2506 | 0.2506 | synthetic | b0aff90 |
| lfu_k | uniform-stress-synthetic | synthetic | 128 | 0.2952 | 0.2952 | synthetic | b0aff90 |
| model_tracing | uniform-stress-synthetic | synthetic | 128 | 0.3026 | 0.3026 | synthetic | b0aff90 |
| score | uniform-stress-synthetic | synthetic | 128 | 0.2952 | 0.2952 | synthetic | b0aff90 |
| lhu | uniform-stress-synthetic | synthetic | 128 | 0.2506 | 0.2506 | synthetic | b0aff90 |
| single_layer_ahead | uniform-stress-synthetic | synthetic | 128 | 0.2949 | 0.9995 | synthetic | b0aff90 |
| stride_prefetch | uniform-stress-synthetic | synthetic | 128 | 0.2928 | 0.9971 | synthetic | b0aff90 |
| multi_layer_ahead | uniform-stress-synthetic | synthetic | 128 | 0.2895 | 0.9947 | synthetic | b0aff90 |
| activation_matrix | uniform-stress-synthetic | synthetic | 128 | 0.0071 | 0.4204 | synthetic | b0aff90 |
| belady | uniform-stress-synthetic | synthetic | 256 | 0.7995 | 0.7995 | synthetic | b0aff90 |
| lru | uniform-stress-synthetic | synthetic | 256 | 0.5096 | 0.5096 | synthetic | b0aff90 |
| lfu | uniform-stress-synthetic | synthetic | 256 | 0.5055 | 0.5055 | synthetic | b0aff90 |
| lfu_k | uniform-stress-synthetic | synthetic | 256 | 0.5096 | 0.5096 | synthetic | b0aff90 |
| model_tracing | uniform-stress-synthetic | synthetic | 256 | 0.5604 | 0.5604 | synthetic | b0aff90 |
| score | uniform-stress-synthetic | synthetic | 256 | 0.5096 | 0.5096 | synthetic | b0aff90 |
| lhu | uniform-stress-synthetic | synthetic | 256 | 0.5055 | 0.5055 | synthetic | b0aff90 |
| single_layer_ahead | uniform-stress-synthetic | synthetic | 256 | 0.5071 | 0.9983 | synthetic | b0aff90 |
| stride_prefetch | uniform-stress-synthetic | synthetic | 256 | 0.5066 | 0.9963 | synthetic | b0aff90 |
| multi_layer_ahead | uniform-stress-synthetic | synthetic | 256 | 0.5060 | 0.9991 | synthetic | b0aff90 |
| activation_matrix | uniform-stress-synthetic | synthetic | 256 | 0.2249 | 0.6118 | synthetic | b0aff90 |
| belady | uniform-stress-synthetic | synthetic | 384 | 0.9119 | 0.9119 | synthetic | b0aff90 |
| lru | uniform-stress-synthetic | synthetic | 384 | 0.7506 | 0.7506 | synthetic | b0aff90 |
| lfu | uniform-stress-synthetic | synthetic | 384 | 0.7476 | 0.7476 | synthetic | b0aff90 |
| lfu_k | uniform-stress-synthetic | synthetic | 384 | 0.7506 | 0.7506 | synthetic | b0aff90 |
| model_tracing | uniform-stress-synthetic | synthetic | 384 | 0.7910 | 0.7910 | synthetic | b0aff90 |
| score | uniform-stress-synthetic | synthetic | 384 | 0.7506 | 0.7506 | synthetic | b0aff90 |
| lhu | uniform-stress-synthetic | synthetic | 384 | 0.7476 | 0.7476 | synthetic | b0aff90 |
| single_layer_ahead | uniform-stress-synthetic | synthetic | 384 | 0.7489 | 0.9988 | synthetic | b0aff90 |
| stride_prefetch | uniform-stress-synthetic | synthetic | 384 | 0.7479 | 0.9967 | synthetic | b0aff90 |
| multi_layer_ahead | uniform-stress-synthetic | synthetic | 384 | 0.7472 | 0.9984 | synthetic | b0aff90 |
| activation_matrix | uniform-stress-synthetic | synthetic | 384 | 0.5021 | 0.8057 | synthetic | b0aff90 |

<!-- run: python -m benchmarks.cache_simulator.research.run_policies | env: python 3.12.3, numpy 2.4.4, Linux x86_64 | commit: b0aff90 -->

## Acceleration — Task C (#292)

Hardware H1: `python=3.12.3, x86_64, 128 cores, numpy=2.4.4,
torch=2.11.0+cu130, 2x NVIDIA RTX PRO 6000 Blackwell Server Edition`.

Base commit: `b0aff90`. Trace: synthetic Zipf (`skew=1.1`), `94 layers x 128
experts`, `topk=8`, `seed=0`. Metric: wall-clock seconds for the full
miss-ratio-curve sweep, and speedup vs the pure-Python reference.

| method | trace (accesses) | capacities | metric | value | hardware | source |
| --- | --- | --- | --- | --- | --- | --- |
| reference `belady_simulate` (per-cap) | 2,000,000 | 12 | wall_s | 20.02 | H1 | `bench_speed --accesses 2000000 --seed 0 --capacities 64..4096` |
| vectorized `opt_mrc` | 2,000,000 | 12 | wall_s / speedup | 17.35 / 1.2x | H1 | same |
| parallel `opt_mrc` | 2,000,000 | 12 | wall_s / speedup | 3.66 / 5.5x | H1 | same |
| exact `lru_mrc` (one pass) | 2,000,000 | 12 | wall_s / speedup | 8.78 / 2.3x | H1 | same |
| `lru_mrc_gpu` (GPU histogram) | 2,000,000 | 12 | wall_s / speedup | 8.89 / 2.3x | H1 | same |
| SHARDS `lru` (1% sample, approx) | 2,000,000 | 12 | wall_s / speedup | 0.07 / 305x | H1 | same |
| vectorized `opt_mrc` | 10,000,000 | 7 | wall_s | 52.08 | H1 | `bench_speed --accesses 10000000 --seed 0 --skip-reference --capacities 64..4096` |
| parallel `opt_mrc` | 10,000,000 | 7 | wall_s | 11.75 | H1 | same |
| exact `lru_mrc` (one pass) | 10,000,000 | 7 | wall_s | 49.35 | H1 | same |
| `lru_mrc_gpu` (GPU histogram) | 10,000,000 | 7 | wall_s | 49.78 | H1 | same |
| SHARDS `lru` (1% sample, approx) | 10,000,000 | 7 | wall_s | 0.37 | H1 | same |
| vectorized access-seq construction | 3,640,000 | n/a | wall_s / speedup | 0.060 / 5.6x | H1 | `vectorized_access_sequence` vs `extract_access_sequence` |

Parity (checked by `bench_speed` inline and by
`tests/python/unit/test_cache_simulator_mrc.py`): `reference`, vectorized
`opt_mrc`, and parallel `opt_mrc` give identical hit rates for every capacity;
`lru_mrc` matches an independent LRU reference. SHARDS abs-error bound reported
as `max(0.05, 4/sqrt(sample_size))`; not recommended for this small-key-universe
domain (see `research/report.md`).

## Real-trace runs — SGLang 0.5.20 capture (#289)

Rows appended by
`benchmarks/cache_simulator/research/run_policies_real.py` over **real**
SGLang `per_token` expert-routing captures (provenance = `real`, in contrast to
the `synthetic` rows above). Capture + conversion method and the full
policy x capacity table live in
`research/results_real_qwen3_235b.md` and `research/capture_report.md`.

Hardware H2: `6x NVIDIA RTX PRO 6000 Blackwell (sm_120, 96 GB)`; capture on the
2 free GPUs (Max-Q Workstation Edition, 300 W). SGLang `0.5.20`,
torch `2.13.0+cu130`, flashinfer `0.6.18`. Simulator run in the repo venv
(`python 3.12.3, numpy 2.4.4, torch 2.11.0+cu130`). `belady` is the
Belady-optimal ceiling; every on-line policy satisfies
`hit_rate(policy) <= hit_rate(belady)` on these real traces too (asserted at
runtime, and cross-checked against the vectorized `opt_mrc` MRC engine).

Model actually captured: `Qwen/Qwen1.5-MoE-A2.7B-Chat` (smoke-test MoE —
24 MoE layers, 60 experts, top-4; `ignore_layers=0`). The 235B target
(`Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`, `--tp 4`) was **blocked**: only 2 of
6 GPUs were free on this shared box (GPUs 0-3 held by another user), below the
>=4 GPUs required for TP4. See `capture_report.md`.

| policy | model | dataset | capacity | hit_rate | eff_hit_rate | provenance | commit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| belady | qwen1.5-moe-a2.7b | mixed20-greedy64 | 64 | 0.3012 | 0.3012 | real | 34717d3 |
| lru | qwen1.5-moe-a2.7b | mixed20-greedy64 | 64 | 0.0382 | 0.0382 | real | 34717d3 |
| lfu | qwen1.5-moe-a2.7b | mixed20-greedy64 | 64 | 0.0710 | 0.0710 | real | 34717d3 |
| lfu_k | qwen1.5-moe-a2.7b | mixed20-greedy64 | 64 | 0.0382 | 0.0382 | real | 34717d3 |
| model_tracing | qwen1.5-moe-a2.7b | mixed20-greedy64 | 64 | 0.0793 | 0.0793 | real | 34717d3 |
| score | qwen1.5-moe-a2.7b | mixed20-greedy64 | 64 | 0.0382 | 0.0382 | real | 34717d3 |
| lhu | qwen1.5-moe-a2.7b | mixed20-greedy64 | 64 | 0.0731 | 0.0731 | real | 34717d3 |
| single_layer_ahead | qwen1.5-moe-a2.7b | mixed20-greedy64 | 64 | 0.0382 | 0.9455 | real | 34717d3 |
| stride_prefetch | qwen1.5-moe-a2.7b | mixed20-greedy64 | 64 | 0.0382 | 0.9999 | real | 34717d3 |
| multi_layer_ahead | qwen1.5-moe-a2.7b | mixed20-greedy64 | 64 | 0.0310 | 0.9277 | real | 34717d3 |
| activation_matrix | qwen1.5-moe-a2.7b | mixed20-greedy64 | 64 | 0.0375 | 0.6259 | real | 34717d3 |
| belady | qwen1.5-moe-a2.7b | mixed20-greedy64 | 128 | 0.4239 | 0.4239 | real | 34717d3 |
| lru | qwen1.5-moe-a2.7b | mixed20-greedy64 | 128 | 0.1699 | 0.1699 | real | 34717d3 |
| lfu | qwen1.5-moe-a2.7b | mixed20-greedy64 | 128 | 0.1340 | 0.1340 | real | 34717d3 |
| lfu_k | qwen1.5-moe-a2.7b | mixed20-greedy64 | 128 | 0.1699 | 0.1699 | real | 34717d3 |
| model_tracing | qwen1.5-moe-a2.7b | mixed20-greedy64 | 128 | 0.1478 | 0.1478 | real | 34717d3 |
| score | qwen1.5-moe-a2.7b | mixed20-greedy64 | 128 | 0.1699 | 0.1699 | real | 34717d3 |
| lhu | qwen1.5-moe-a2.7b | mixed20-greedy64 | 128 | 0.1367 | 0.1367 | real | 34717d3 |
| single_layer_ahead | qwen1.5-moe-a2.7b | mixed20-greedy64 | 128 | 0.1694 | 0.9996 | real | 34717d3 |
| stride_prefetch | qwen1.5-moe-a2.7b | mixed20-greedy64 | 128 | 0.1698 | 0.9998 | real | 34717d3 |
| multi_layer_ahead | qwen1.5-moe-a2.7b | mixed20-greedy64 | 128 | 0.1693 | 0.9765 | real | 34717d3 |
| activation_matrix | qwen1.5-moe-a2.7b | mixed20-greedy64 | 128 | 0.0384 | 0.6653 | real | 34717d3 |
| belady | qwen1.5-moe-a2.7b | mixed20-greedy64 | 256 | 0.5776 | 0.5776 | real | 34717d3 |
| lru | qwen1.5-moe-a2.7b | mixed20-greedy64 | 256 | 0.2625 | 0.2625 | real | 34717d3 |
| lfu | qwen1.5-moe-a2.7b | mixed20-greedy64 | 256 | 0.2481 | 0.2481 | real | 34717d3 |
| lfu_k | qwen1.5-moe-a2.7b | mixed20-greedy64 | 256 | 0.2625 | 0.2625 | real | 34717d3 |
| model_tracing | qwen1.5-moe-a2.7b | mixed20-greedy64 | 256 | 0.2693 | 0.2693 | real | 34717d3 |
| score | qwen1.5-moe-a2.7b | mixed20-greedy64 | 256 | 0.2625 | 0.2625 | real | 34717d3 |
| lhu | qwen1.5-moe-a2.7b | mixed20-greedy64 | 256 | 0.2509 | 0.2509 | real | 34717d3 |
| single_layer_ahead | qwen1.5-moe-a2.7b | mixed20-greedy64 | 256 | 0.2587 | 0.9974 | real | 34717d3 |
| stride_prefetch | qwen1.5-moe-a2.7b | mixed20-greedy64 | 256 | 0.2579 | 0.9949 | real | 34717d3 |
| multi_layer_ahead | qwen1.5-moe-a2.7b | mixed20-greedy64 | 256 | 0.2559 | 0.9978 | real | 34717d3 |
| activation_matrix | qwen1.5-moe-a2.7b | mixed20-greedy64 | 256 | 0.0626 | 0.6704 | real | 34717d3 |
| belady | qwen1.5-moe-a2.7b | mixed20-greedy64 | 512 | 0.7573 | 0.7573 | real | 34717d3 |
| lru | qwen1.5-moe-a2.7b | mixed20-greedy64 | 512 | 0.4722 | 0.4722 | real | 34717d3 |
| lfu | qwen1.5-moe-a2.7b | mixed20-greedy64 | 512 | 0.4520 | 0.4520 | real | 34717d3 |
| lfu_k | qwen1.5-moe-a2.7b | mixed20-greedy64 | 512 | 0.4722 | 0.4722 | real | 34717d3 |
| model_tracing | qwen1.5-moe-a2.7b | mixed20-greedy64 | 512 | 0.4813 | 0.4813 | real | 34717d3 |
| score | qwen1.5-moe-a2.7b | mixed20-greedy64 | 512 | 0.4722 | 0.4722 | real | 34717d3 |
| lhu | qwen1.5-moe-a2.7b | mixed20-greedy64 | 512 | 0.4566 | 0.4566 | real | 34717d3 |
| single_layer_ahead | qwen1.5-moe-a2.7b | mixed20-greedy64 | 512 | 0.4687 | 0.9978 | real | 34717d3 |
| stride_prefetch | qwen1.5-moe-a2.7b | mixed20-greedy64 | 512 | 0.4694 | 0.9969 | real | 34717d3 |
| multi_layer_ahead | qwen1.5-moe-a2.7b | mixed20-greedy64 | 512 | 0.4662 | 0.9979 | real | 34717d3 |
| activation_matrix | qwen1.5-moe-a2.7b | mixed20-greedy64 | 512 | 0.2457 | 0.7643 | real | 34717d3 |
| belady | qwen1.5-moe-a2.7b | mixed20-greedy64 | 768 | 0.8639 | 0.8639 | real | 34717d3 |
| lru | qwen1.5-moe-a2.7b | mixed20-greedy64 | 768 | 0.6525 | 0.6525 | real | 34717d3 |
| lfu | qwen1.5-moe-a2.7b | mixed20-greedy64 | 768 | 0.6324 | 0.6324 | real | 34717d3 |
| lfu_k | qwen1.5-moe-a2.7b | mixed20-greedy64 | 768 | 0.6525 | 0.6525 | real | 34717d3 |
| model_tracing | qwen1.5-moe-a2.7b | mixed20-greedy64 | 768 | 0.6605 | 0.6605 | real | 34717d3 |
| score | qwen1.5-moe-a2.7b | mixed20-greedy64 | 768 | 0.6525 | 0.6525 | real | 34717d3 |
| lhu | qwen1.5-moe-a2.7b | mixed20-greedy64 | 768 | 0.6325 | 0.6325 | real | 34717d3 |
| single_layer_ahead | qwen1.5-moe-a2.7b | mixed20-greedy64 | 768 | 0.6498 | 0.9982 | real | 34717d3 |
| stride_prefetch | qwen1.5-moe-a2.7b | mixed20-greedy64 | 768 | 0.6508 | 0.9982 | real | 34717d3 |
| multi_layer_ahead | qwen1.5-moe-a2.7b | mixed20-greedy64 | 768 | 0.6481 | 0.9984 | real | 34717d3 |
| activation_matrix | qwen1.5-moe-a2.7b | mixed20-greedy64 | 768 | 0.4023 | 0.8245 | real | 34717d3 |
| belady | qwen1.5-moe-a2.7b | mixed20-greedy64 | 1024 | 0.9300 | 0.9300 | real | 34717d3 |
| lru | qwen1.5-moe-a2.7b | mixed20-greedy64 | 1024 | 0.8089 | 0.8089 | real | 34717d3 |
| lfu | qwen1.5-moe-a2.7b | mixed20-greedy64 | 1024 | 0.7899 | 0.7899 | real | 34717d3 |
| lfu_k | qwen1.5-moe-a2.7b | mixed20-greedy64 | 1024 | 0.8089 | 0.8089 | real | 34717d3 |
| model_tracing | qwen1.5-moe-a2.7b | mixed20-greedy64 | 1024 | 0.8122 | 0.8122 | real | 34717d3 |
| score | qwen1.5-moe-a2.7b | mixed20-greedy64 | 1024 | 0.8089 | 0.8089 | real | 34717d3 |
| lhu | qwen1.5-moe-a2.7b | mixed20-greedy64 | 1024 | 0.7899 | 0.7899 | real | 34717d3 |
| single_layer_ahead | qwen1.5-moe-a2.7b | mixed20-greedy64 | 1024 | 0.8077 | 0.9991 | real | 34717d3 |
| stride_prefetch | qwen1.5-moe-a2.7b | mixed20-greedy64 | 1024 | 0.8082 | 0.9991 | real | 34717d3 |
| multi_layer_ahead | qwen1.5-moe-a2.7b | mixed20-greedy64 | 1024 | 0.8068 | 0.9992 | real | 34717d3 |
| activation_matrix | qwen1.5-moe-a2.7b | mixed20-greedy64 | 1024 | 0.6264 | 0.8916 | real | 34717d3 |

<!-- run: python -m benchmarks.cache_simulator.research.run_policies_real --model qwen1.5-moe-a2.7b --dataset mixed20-greedy64 --ignore-layers 0 | env: python 3.12.3, numpy 2.4.4, Linux x86_64 | commit: 34717d3 -->

### Qwen3-30B-A3B — 235B-family proxy (#289 follow-up)

Real capture of `Qwen/Qwen3-30B-A3B` — the closest available proxy for the
235B target: same `Qwen3MoeForCausalLM` routing family, **128 experts, top-8,
48 MoE layers** (`ignore_layers=0`), identity physical->logical map
(`ep_num_redundant_experts=0`), expert ids in `[0, 127]`. This matches the
235B's 128-expert/top-8 routing geometry exactly (only the layer count and
weight precision differ). Captured on GPUs 4,5 (Max-Q Blackwell, sm_120) with
`--tp 2`, SGLang `0.5.20`, greedy, `max_new_tokens=64`, 150 MMLU
(`cais/mmlu[all][test]`, first 150 in fixed order) prompts, batch size 1 per
dump. TP0 and TP1 each dump the (replicated) routing; the 300 raw dumps are
de-duplicated to 150 by routing content before the merge. `belady` is the
ceiling; invariant checked at runtime and against `opt_mrc`. Full table in
`results_real_qwen3_30b.md`.

| policy | model | dataset | capacity | hit_rate | eff_hit_rate | provenance | commit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| belady | qwen3-30b-a3b | mmlu150-greedy64 | 256 | 0.7028 | 0.7028 | real | 963aab0 |
| lru | qwen3-30b-a3b | mmlu150-greedy64 | 256 | 0.4565 | 0.4565 | real | 963aab0 |
| lfu | qwen3-30b-a3b | mmlu150-greedy64 | 256 | 0.2021 | 0.2021 | real | 963aab0 |
| lfu_k | qwen3-30b-a3b | mmlu150-greedy64 | 256 | 0.4565 | 0.4565 | real | 963aab0 |
| model_tracing | qwen3-30b-a3b | mmlu150-greedy64 | 256 | 0.2083 | 0.2083 | real | 963aab0 |
| score | qwen3-30b-a3b | mmlu150-greedy64 | 256 | 0.4565 | 0.4565 | real | 963aab0 |
| lhu | qwen3-30b-a3b | mmlu150-greedy64 | 256 | 0.4485 | 0.4485 | real | 963aab0 |
| single_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64 | 256 | 0.4565 | 0.9974 | real | 963aab0 |
| stride_prefetch | qwen3-30b-a3b | mmlu150-greedy64 | 256 | 0.4565 | 1.0000 | real | 963aab0 |
| multi_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64 | 256 | 0.4559 | 0.9472 | real | 963aab0 |
| activation_matrix | qwen3-30b-a3b | mmlu150-greedy64 | 256 | 0.4565 | 0.8067 | real | 963aab0 |
| belady | qwen3-30b-a3b | mmlu150-greedy64 | 512 | 0.7898 | 0.7898 | real | 963aab0 |
| lru | qwen3-30b-a3b | mmlu150-greedy64 | 512 | 0.6617 | 0.6617 | real | 963aab0 |
| lfu | qwen3-30b-a3b | mmlu150-greedy64 | 512 | 0.3266 | 0.3266 | real | 963aab0 |
| lfu_k | qwen3-30b-a3b | mmlu150-greedy64 | 512 | 0.6617 | 0.6617 | real | 963aab0 |
| model_tracing | qwen3-30b-a3b | mmlu150-greedy64 | 512 | 0.3387 | 0.3387 | real | 963aab0 |
| score | qwen3-30b-a3b | mmlu150-greedy64 | 512 | 0.6617 | 0.6617 | real | 963aab0 |
| lhu | qwen3-30b-a3b | mmlu150-greedy64 | 512 | 0.6505 | 0.6505 | real | 963aab0 |
| single_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64 | 512 | 0.6614 | 0.9998 | real | 963aab0 |
| stride_prefetch | qwen3-30b-a3b | mmlu150-greedy64 | 512 | 0.6616 | 0.9999 | real | 963aab0 |
| multi_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64 | 512 | 0.6613 | 0.9999 | real | 963aab0 |
| activation_matrix | qwen3-30b-a3b | mmlu150-greedy64 | 512 | 0.4567 | 0.8068 | real | 963aab0 |
| belady | qwen3-30b-a3b | mmlu150-greedy64 | 1024 | 0.8705 | 0.8705 | real | 963aab0 |
| lru | qwen3-30b-a3b | mmlu150-greedy64 | 1024 | 0.7631 | 0.7631 | real | 963aab0 |
| lfu | qwen3-30b-a3b | mmlu150-greedy64 | 1024 | 0.5069 | 0.5069 | real | 963aab0 |
| lfu_k | qwen3-30b-a3b | mmlu150-greedy64 | 1024 | 0.7631 | 0.7631 | real | 963aab0 |
| model_tracing | qwen3-30b-a3b | mmlu150-greedy64 | 1024 | 0.5181 | 0.5181 | real | 963aab0 |
| score | qwen3-30b-a3b | mmlu150-greedy64 | 1024 | 0.7631 | 0.7631 | real | 963aab0 |
| lhu | qwen3-30b-a3b | mmlu150-greedy64 | 1024 | 0.7535 | 0.7535 | real | 963aab0 |
| single_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64 | 1024 | 0.7622 | 0.9994 | real | 963aab0 |
| stride_prefetch | qwen3-30b-a3b | mmlu150-greedy64 | 1024 | 0.7629 | 0.9997 | real | 963aab0 |
| multi_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64 | 1024 | 0.7615 | 0.9994 | real | 963aab0 |
| activation_matrix | qwen3-30b-a3b | mmlu150-greedy64 | 1024 | 0.7052 | 0.8724 | real | 963aab0 |
| belady | qwen3-30b-a3b | mmlu150-greedy64 | 2048 | 0.9462 | 0.9462 | real | 963aab0 |
| lru | qwen3-30b-a3b | mmlu150-greedy64 | 2048 | 0.8696 | 0.8696 | real | 963aab0 |
| lfu | qwen3-30b-a3b | mmlu150-greedy64 | 2048 | 0.7650 | 0.7650 | real | 963aab0 |
| lfu_k | qwen3-30b-a3b | mmlu150-greedy64 | 2048 | 0.8696 | 0.8696 | real | 963aab0 |
| model_tracing | qwen3-30b-a3b | mmlu150-greedy64 | 2048 | 0.7750 | 0.7750 | real | 963aab0 |
| score | qwen3-30b-a3b | mmlu150-greedy64 | 2048 | 0.8696 | 0.8696 | real | 963aab0 |
| lhu | qwen3-30b-a3b | mmlu150-greedy64 | 2048 | 0.8663 | 0.8663 | real | 963aab0 |
| single_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64 | 2048 | 0.8684 | 0.9993 | real | 963aab0 |
| stride_prefetch | qwen3-30b-a3b | mmlu150-greedy64 | 2048 | 0.8695 | 0.9999 | real | 963aab0 |
| multi_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64 | 2048 | 0.8677 | 0.9993 | real | 963aab0 |
| activation_matrix | qwen3-30b-a3b | mmlu150-greedy64 | 2048 | 0.8422 | 0.9222 | real | 963aab0 |
| belady | qwen3-30b-a3b | mmlu150-greedy64 | 3072 | 0.9810 | 0.9810 | real | 963aab0 |
| lru | qwen3-30b-a3b | mmlu150-greedy64 | 3072 | 0.9378 | 0.9378 | real | 963aab0 |
| lfu | qwen3-30b-a3b | mmlu150-greedy64 | 3072 | 0.9188 | 0.9188 | real | 963aab0 |
| lfu_k | qwen3-30b-a3b | mmlu150-greedy64 | 3072 | 0.9378 | 0.9378 | real | 963aab0 |
| model_tracing | qwen3-30b-a3b | mmlu150-greedy64 | 3072 | 0.9253 | 0.9253 | real | 963aab0 |
| score | qwen3-30b-a3b | mmlu150-greedy64 | 3072 | 0.9378 | 0.9378 | real | 963aab0 |
| lhu | qwen3-30b-a3b | mmlu150-greedy64 | 3072 | 0.9412 | 0.9412 | real | 963aab0 |
| single_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64 | 3072 | 0.9363 | 0.9991 | real | 963aab0 |
| stride_prefetch | qwen3-30b-a3b | mmlu150-greedy64 | 3072 | 0.9378 | 1.0000 | real | 963aab0 |
| multi_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64 | 3072 | 0.9353 | 0.9991 | real | 963aab0 |
| activation_matrix | qwen3-30b-a3b | mmlu150-greedy64 | 3072 | 0.9277 | 0.9571 | real | 963aab0 |
| belady | qwen3-30b-a3b | mmlu150-greedy64 | 4096 | 0.9961 | 0.9961 | real | 963aab0 |
| lru | qwen3-30b-a3b | mmlu150-greedy64 | 4096 | 0.9891 | 0.9891 | real | 963aab0 |
| lfu | qwen3-30b-a3b | mmlu150-greedy64 | 4096 | 0.9824 | 0.9824 | real | 963aab0 |
| lfu_k | qwen3-30b-a3b | mmlu150-greedy64 | 4096 | 0.9891 | 0.9891 | real | 963aab0 |
| model_tracing | qwen3-30b-a3b | mmlu150-greedy64 | 4096 | 0.9846 | 0.9846 | real | 963aab0 |
| score | qwen3-30b-a3b | mmlu150-greedy64 | 4096 | 0.9891 | 0.9891 | real | 963aab0 |
| lhu | qwen3-30b-a3b | mmlu150-greedy64 | 4096 | 0.9897 | 0.9897 | real | 963aab0 |
| single_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64 | 4096 | 0.9888 | 0.9999 | real | 963aab0 |
| stride_prefetch | qwen3-30b-a3b | mmlu150-greedy64 | 4096 | 0.9891 | 1.0000 | real | 963aab0 |
| multi_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64 | 4096 | 0.9886 | 0.9998 | real | 963aab0 |
| activation_matrix | qwen3-30b-a3b | mmlu150-greedy64 | 4096 | 0.9866 | 0.9886 | real | 963aab0 |
| belady | qwen3-30b-a3b | mmlu150-greedy64 | 6144 | 0.9992 | 0.9992 | real | 963aab0 |
| lru | qwen3-30b-a3b | mmlu150-greedy64 | 6144 | 0.9992 | 0.9992 | real | 963aab0 |
| lfu | qwen3-30b-a3b | mmlu150-greedy64 | 6144 | 0.9992 | 0.9992 | real | 963aab0 |
| lfu_k | qwen3-30b-a3b | mmlu150-greedy64 | 6144 | 0.9992 | 0.9992 | real | 963aab0 |
| model_tracing | qwen3-30b-a3b | mmlu150-greedy64 | 6144 | 0.9992 | 0.9992 | real | 963aab0 |
| score | qwen3-30b-a3b | mmlu150-greedy64 | 6144 | 0.9992 | 0.9992 | real | 963aab0 |
| lhu | qwen3-30b-a3b | mmlu150-greedy64 | 6144 | 0.9992 | 0.9992 | real | 963aab0 |
| single_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64 | 6144 | 0.9992 | 1.0000 | real | 963aab0 |
| stride_prefetch | qwen3-30b-a3b | mmlu150-greedy64 | 6144 | 0.9992 | 1.0000 | real | 963aab0 |
| multi_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64 | 6144 | 0.9992 | 1.0000 | real | 963aab0 |
| activation_matrix | qwen3-30b-a3b | mmlu150-greedy64 | 6144 | 0.9992 | 0.9994 | real | 963aab0 |

<!-- run: python -m benchmarks.cache_simulator.research.run_policies_real --model qwen3-30b-a3b --dataset mmlu150-greedy64 --ignore-layers 0 | env: python 3.12.3, numpy 2.4.4, Linux x86_64 | commit: 963aab0 -->

### Qwen3-235B-A22B-Instruct-2507-FP8 — the real target (#289)

Real capture of the **actual 235B target** at last. `Qwen3MoeForCausalLM`,
**94 MoE layers, 128 experts, top-8** (`ignore_layers=0`), identity
physical->logical map (`ep_num_redundant_experts=0`), expert ids in `[0, 127]`,
no `-1` padding (top-8 fills the recorder buffer). FP8 e4m3 block-wise weights;
gates are not quantized so `topk_ids` are the true routing. Captured on GPUs
0-3 (RTX PRO 6000 Blackwell, sm_120) with `--tp 4 --dp 1`, SGLang `0.5.20`,
greedy, `max_new_tokens=64`, 100 MMLU (`cais/mmlu[all][test]`, first 100 in
fixed order) prompts, batch size 1 per dump. All 4 TP ranks dump the
(replicated, non-DP) routing; the 397 raw dumps are de-duplicated by routing
content to 100.

**Decode-only numbers below.** The per-request dumps were merged into one
`.pt`; a merged stream holds one prefill pass *per request*
(100 prefill + 6400 decode records here), so a replay must drop **every**
prefill pass, not just the global `records[0]`. The correct decode-only stream
is **4,812,800** accesses, **10,551 / 12,032** distinct `(layer, expert)` keys.
Belady ceiling (decode-only), independently reproduced bit-for-bit by a
per-record decode filter (`seq_len==1`) and by the vectorized `opt_mrc`:

| policy | model | dataset | capacity | hit_rate | eff_hit_rate | provenance | commit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| belady | qwen3-235b | mmlu100-greedy64/decode | 256 | 0.3067 | 0.3067 | real | a44b9bd |
| belady | qwen3-235b | mmlu100-greedy64/decode | 512 | 0.4898 | 0.4898 | real | a44b9bd |
| belady | qwen3-235b | mmlu100-greedy64/decode | 1024 | 0.6581 | 0.6581 | real | a44b9bd |
| belady | qwen3-235b | mmlu100-greedy64/decode | 2048 | 0.8082 | 0.8082 | real | a44b9bd |
| belady | qwen3-235b | mmlu100-greedy64/decode | 4096 | 0.9320 | 0.9320 | real | a44b9bd |
| belady | qwen3-235b | mmlu100-greedy64/decode | 8192 | 0.9947 | 0.9947 | real | a44b9bd |

The **full decode-only policy x capacity study** for this capture (LRU/LFU/
model-tracing/LHU/ARC/2Q/LIRS/S3-FIFO/SIEVE/W-TinyLFU/per-layer-LRU/LCNU plus
prefetch oracles, Belady asserted as the ceiling) lives on branch
`feat/cache-policy-lcnu`
(`research/results_real_qwen3_235b_decode.md`, `research/decode_policy_study.md`),
which also carries the per-record decode-detection fix in
`run_policies_real.py` / `mrc.py`. These capture-provenance rows are recorded
here to avoid duplicating that in-flight study.

<!-- 235B capture: SGLang 0.5.20 TP4 on 4x RTX PRO 6000 Blackwell sm_120; decode-only Belady verified against opt_mrc; full policy study on feat/cache-policy-lcnu -->
## Decode-only re-runs, classic hybrids and LCNU

Rows appended by `run_policies_real.py` after the decode-stream fix. The
dataset column ends in `/decode` (decode passes only) or
`/decode+prefill-union` (each prefill pass as one fetch per distinct expert
per layer). The real-trace rows above this section were computed on merged
streams that still contained one prefill pass per request (52% of the
Qwen3-30B-A3B accesses), see `decode_policy_study.md`. Score-based eviction is
omitted (it decays every expert's score on every access: hours per capacity).
Commit `a44b9bd+dirty` = these changes on top of `a44b9bd`; 32 worker
processes on the 128-core host.

| policy | model | dataset | capacity | hit_rate | eff_hit_rate | provenance | commit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| belady | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 32 | 0.1856 | 0.1856 | real | a44b9bd+dirty |
| lru | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 32 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| lfu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 32 | 0.0367 | 0.0367 | real | a44b9bd+dirty |
| lfu_k | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 32 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| model_tracing | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 32 | 0.0419 | 0.0419 | real | a44b9bd+dirty |
| lhu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 32 | 0.0335 | 0.0335 | real | a44b9bd+dirty |
| arc | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 32 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| 2q | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 32 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| lru_k | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 32 | 0.0328 | 0.0328 | real | a44b9bd+dirty |
| lirs | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 32 | 0.0470 | 0.0470 | real | a44b9bd+dirty |
| s3fifo | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 32 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| sieve | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 32 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| w_tinylfu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 32 | 0.0458 | 0.0458 | real | a44b9bd+dirty |
| layer_lru | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 32 | 0.0090 | 0.0090 | real | a44b9bd+dirty |
| lcnu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 32 | 0.0526 | 0.0526 | real | a44b9bd+dirty |
| single_layer_ahead | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 32 | 0.0000 | 0.9999 | real | a44b9bd+dirty |
| stride_prefetch | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 32 | 0.0000 | 0.9999 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 32 | 0.0000 | 0.9999 | real | a44b9bd+dirty |
| activation_matrix | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 32 | 0.0000 | 0.6500 | real | a44b9bd+dirty |
| belady | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 64 | 0.2860 | 0.2860 | real | a44b9bd+dirty |
| lru | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 64 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| lfu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 64 | 0.0699 | 0.0699 | real | a44b9bd+dirty |
| lfu_k | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 64 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| model_tracing | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 64 | 0.0792 | 0.0792 | real | a44b9bd+dirty |
| lhu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 64 | 0.0739 | 0.0739 | real | a44b9bd+dirty |
| arc | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 64 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| 2q | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 64 | 0.0037 | 0.0037 | real | a44b9bd+dirty |
| lru_k | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 64 | 0.0773 | 0.0773 | real | a44b9bd+dirty |
| lirs | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 64 | 0.0865 | 0.0865 | real | a44b9bd+dirty |
| s3fifo | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 64 | 0.0495 | 0.0495 | real | a44b9bd+dirty |
| sieve | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 64 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| w_tinylfu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 64 | 0.0841 | 0.0841 | real | a44b9bd+dirty |
| layer_lru | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 64 | 0.0350 | 0.0350 | real | a44b9bd+dirty |
| lcnu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 64 | 0.0956 | 0.0956 | real | a44b9bd+dirty |
| single_layer_ahead | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 64 | 0.0000 | 0.9999 | real | a44b9bd+dirty |
| stride_prefetch | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 64 | 0.0000 | 0.9999 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 64 | 0.0000 | 0.9999 | real | a44b9bd+dirty |
| activation_matrix | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 64 | 0.0000 | 0.6501 | real | a44b9bd+dirty |
| belady | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 96 | 0.3581 | 0.3581 | real | a44b9bd+dirty |
| lru | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 96 | 0.0958 | 0.0958 | real | a44b9bd+dirty |
| lfu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 96 | 0.1020 | 0.1020 | real | a44b9bd+dirty |
| lfu_k | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 96 | 0.0958 | 0.0958 | real | a44b9bd+dirty |
| model_tracing | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 96 | 0.1146 | 0.1146 | real | a44b9bd+dirty |
| lhu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 96 | 0.1048 | 0.1048 | real | a44b9bd+dirty |
| arc | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 96 | 0.0946 | 0.0946 | real | a44b9bd+dirty |
| 2q | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 96 | 0.0557 | 0.0557 | real | a44b9bd+dirty |
| lru_k | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 96 | 0.1139 | 0.1139 | real | a44b9bd+dirty |
| lirs | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 96 | 0.1230 | 0.1230 | real | a44b9bd+dirty |
| s3fifo | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 96 | 0.1140 | 0.1140 | real | a44b9bd+dirty |
| sieve | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 96 | 0.0046 | 0.0046 | real | a44b9bd+dirty |
| w_tinylfu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 96 | 0.1128 | 0.1128 | real | a44b9bd+dirty |
| layer_lru | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 96 | 0.0958 | 0.0958 | real | a44b9bd+dirty |
| lcnu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 96 | 0.1346 | 0.1346 | real | a44b9bd+dirty |
| single_layer_ahead | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 96 | 0.0092 | 0.9685 | real | a44b9bd+dirty |
| stride_prefetch | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 96 | 0.0024 | 0.8805 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 96 | 0.0007 | 0.9983 | real | a44b9bd+dirty |
| activation_matrix | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 96 | 0.0000 | 0.6501 | real | a44b9bd+dirty |
| belady | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 128 | 0.4142 | 0.4142 | real | a44b9bd+dirty |
| lru | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 128 | 0.1485 | 0.1485 | real | a44b9bd+dirty |
| lfu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 128 | 0.1320 | 0.1320 | real | a44b9bd+dirty |
| lfu_k | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 128 | 0.1485 | 0.1485 | real | a44b9bd+dirty |
| model_tracing | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 128 | 0.1472 | 0.1472 | real | a44b9bd+dirty |
| lhu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 128 | 0.1350 | 0.1350 | real | a44b9bd+dirty |
| arc | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 128 | 0.1373 | 0.1373 | real | a44b9bd+dirty |
| 2q | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 128 | 0.1198 | 0.1198 | real | a44b9bd+dirty |
| lru_k | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 128 | 0.1514 | 0.1514 | real | a44b9bd+dirty |
| lirs | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 128 | 0.1566 | 0.1566 | real | a44b9bd+dirty |
| s3fifo | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 128 | 0.1445 | 0.1445 | real | a44b9bd+dirty |
| sieve | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 128 | 0.0533 | 0.0533 | real | a44b9bd+dirty |
| w_tinylfu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 128 | 0.1482 | 0.1482 | real | a44b9bd+dirty |
| layer_lru | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 128 | 0.1401 | 0.1401 | real | a44b9bd+dirty |
| lcnu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 128 | 0.1737 | 0.1737 | real | a44b9bd+dirty |
| single_layer_ahead | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 128 | 0.1484 | 0.9999 | real | a44b9bd+dirty |
| stride_prefetch | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 128 | 0.1484 | 0.9999 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 128 | 0.1484 | 0.9999 | real | a44b9bd+dirty |
| activation_matrix | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 128 | 0.0002 | 0.6501 | real | a44b9bd+dirty |
| belady | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 192 | 0.5035 | 0.5035 | real | a44b9bd+dirty |
| lru | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 192 | 0.2224 | 0.2224 | real | a44b9bd+dirty |
| lfu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 192 | 0.1893 | 0.1893 | real | a44b9bd+dirty |
| lfu_k | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 192 | 0.2224 | 0.2224 | real | a44b9bd+dirty |
| model_tracing | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 192 | 0.2097 | 0.2097 | real | a44b9bd+dirty |
| lhu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 192 | 0.1922 | 0.1922 | real | a44b9bd+dirty |
| arc | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 192 | 0.2090 | 0.2090 | real | a44b9bd+dirty |
| 2q | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 192 | 0.1796 | 0.1796 | real | a44b9bd+dirty |
| lru_k | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 192 | 0.2163 | 0.2163 | real | a44b9bd+dirty |
| lirs | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 192 | 0.2204 | 0.2204 | real | a44b9bd+dirty |
| s3fifo | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 192 | 0.2036 | 0.2036 | real | a44b9bd+dirty |
| sieve | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 192 | 0.1108 | 0.1108 | real | a44b9bd+dirty |
| w_tinylfu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 192 | 0.2030 | 0.2030 | real | a44b9bd+dirty |
| layer_lru | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 192 | 0.2110 | 0.2110 | real | a44b9bd+dirty |
| lcnu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 192 | 0.2369 | 0.2369 | real | a44b9bd+dirty |
| single_layer_ahead | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 192 | 0.2131 | 0.9926 | real | a44b9bd+dirty |
| stride_prefetch | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 192 | 0.2065 | 0.9835 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 192 | 0.2021 | 0.9914 | real | a44b9bd+dirty |
| activation_matrix | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 192 | 0.0016 | 0.6498 | real | a44b9bd+dirty |
| belady | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 256 | 0.5727 | 0.5727 | real | a44b9bd+dirty |
| lru | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 256 | 0.2494 | 0.2494 | real | a44b9bd+dirty |
| lfu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 256 | 0.2444 | 0.2444 | real | a44b9bd+dirty |
| lfu_k | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 256 | 0.2494 | 0.2494 | real | a44b9bd+dirty |
| model_tracing | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 256 | 0.2686 | 0.2686 | real | a44b9bd+dirty |
| lhu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 256 | 0.2472 | 0.2472 | real | a44b9bd+dirty |
| arc | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 256 | 0.2713 | 0.2713 | real | a44b9bd+dirty |
| 2q | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 256 | 0.2669 | 0.2669 | real | a44b9bd+dirty |
| lru_k | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 256 | 0.2662 | 0.2662 | real | a44b9bd+dirty |
| lirs | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 256 | 0.2775 | 0.2775 | real | a44b9bd+dirty |
| s3fifo | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 256 | 0.2621 | 0.2621 | real | a44b9bd+dirty |
| sieve | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 256 | 0.1737 | 0.1737 | real | a44b9bd+dirty |
| w_tinylfu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 256 | 0.2567 | 0.2567 | real | a44b9bd+dirty |
| layer_lru | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 256 | 0.2627 | 0.2627 | real | a44b9bd+dirty |
| lcnu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 256 | 0.2887 | 0.2887 | real | a44b9bd+dirty |
| single_layer_ahead | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 256 | 0.2459 | 0.9975 | real | a44b9bd+dirty |
| stride_prefetch | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 256 | 0.2443 | 0.9944 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 256 | 0.2431 | 0.9980 | real | a44b9bd+dirty |
| activation_matrix | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 256 | 0.0258 | 0.6546 | real | a44b9bd+dirty |
| belady | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 384 | 0.6768 | 0.6768 | real | a44b9bd+dirty |
| lru | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 384 | 0.3627 | 0.3627 | real | a44b9bd+dirty |
| lfu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 384 | 0.3498 | 0.3498 | real | a44b9bd+dirty |
| lfu_k | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 384 | 0.3627 | 0.3627 | real | a44b9bd+dirty |
| model_tracing | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 384 | 0.3795 | 0.3795 | real | a44b9bd+dirty |
| lhu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 384 | 0.3528 | 0.3528 | real | a44b9bd+dirty |
| arc | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 384 | 0.3830 | 0.3830 | real | a44b9bd+dirty |
| 2q | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 384 | 0.3870 | 0.3870 | real | a44b9bd+dirty |
| lru_k | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 384 | 0.3779 | 0.3779 | real | a44b9bd+dirty |
| lirs | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 384 | 0.3867 | 0.3867 | real | a44b9bd+dirty |
| s3fifo | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 384 | 0.3760 | 0.3760 | real | a44b9bd+dirty |
| sieve | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 384 | 0.2768 | 0.2768 | real | a44b9bd+dirty |
| w_tinylfu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 384 | 0.3566 | 0.3566 | real | a44b9bd+dirty |
| layer_lru | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 384 | 0.3736 | 0.3736 | real | a44b9bd+dirty |
| lcnu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 384 | 0.3894 | 0.3894 | real | a44b9bd+dirty |
| single_layer_ahead | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 384 | 0.3604 | 0.9985 | real | a44b9bd+dirty |
| stride_prefetch | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 384 | 0.3592 | 0.9962 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 384 | 0.3584 | 0.9983 | real | a44b9bd+dirty |
| activation_matrix | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 384 | 0.1530 | 0.7225 | real | a44b9bd+dirty |
| belady | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 512 | 0.7538 | 0.7538 | real | a44b9bd+dirty |
| lru | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 512 | 0.4698 | 0.4698 | real | a44b9bd+dirty |
| lfu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 512 | 0.4475 | 0.4475 | real | a44b9bd+dirty |
| lfu_k | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 512 | 0.4698 | 0.4698 | real | a44b9bd+dirty |
| model_tracing | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 512 | 0.4810 | 0.4810 | real | a44b9bd+dirty |
| lhu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 512 | 0.4481 | 0.4481 | real | a44b9bd+dirty |
| arc | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 512 | 0.4841 | 0.4841 | real | a44b9bd+dirty |
| 2q | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 512 | 0.4866 | 0.4866 | real | a44b9bd+dirty |
| lru_k | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 512 | 0.4811 | 0.4811 | real | a44b9bd+dirty |
| lirs | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 512 | 0.4819 | 0.4819 | real | a44b9bd+dirty |
| s3fifo | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 512 | 0.4794 | 0.4794 | real | a44b9bd+dirty |
| sieve | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 512 | 0.3721 | 0.3721 | real | a44b9bd+dirty |
| w_tinylfu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 512 | 0.4479 | 0.4479 | real | a44b9bd+dirty |
| layer_lru | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 512 | 0.4788 | 0.4788 | real | a44b9bd+dirty |
| lcnu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 512 | 0.4896 | 0.4896 | real | a44b9bd+dirty |
| single_layer_ahead | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 512 | 0.4680 | 0.9987 | real | a44b9bd+dirty |
| stride_prefetch | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 512 | 0.4672 | 0.9970 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 512 | 0.4666 | 0.9989 | real | a44b9bd+dirty |
| activation_matrix | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 512 | 0.2298 | 0.7597 | real | a44b9bd+dirty |
| belady | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 768 | 0.8622 | 0.8622 | real | a44b9bd+dirty |
| lru | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 768 | 0.6488 | 0.6488 | real | a44b9bd+dirty |
| lfu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 768 | 0.6286 | 0.6286 | real | a44b9bd+dirty |
| lfu_k | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 768 | 0.6488 | 0.6488 | real | a44b9bd+dirty |
| model_tracing | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 768 | 0.6588 | 0.6588 | real | a44b9bd+dirty |
| lhu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 768 | 0.6286 | 0.6286 | real | a44b9bd+dirty |
| arc | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 768 | 0.6492 | 0.6492 | real | a44b9bd+dirty |
| 2q | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 768 | 0.6492 | 0.6492 | real | a44b9bd+dirty |
| lru_k | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 768 | 0.6563 | 0.6563 | real | a44b9bd+dirty |
| lirs | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 768 | 0.6529 | 0.6529 | real | a44b9bd+dirty |
| s3fifo | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 768 | 0.6588 | 0.6588 | real | a44b9bd+dirty |
| sieve | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 768 | 0.5540 | 0.5540 | real | a44b9bd+dirty |
| w_tinylfu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 768 | 0.6153 | 0.6153 | real | a44b9bd+dirty |
| layer_lru | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 768 | 0.6518 | 0.6518 | real | a44b9bd+dirty |
| lcnu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 768 | 0.6668 | 0.6668 | real | a44b9bd+dirty |
| single_layer_ahead | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 768 | 0.6478 | 0.9992 | real | a44b9bd+dirty |
| stride_prefetch | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 768 | 0.6472 | 0.9982 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 768 | 0.6469 | 0.9992 | real | a44b9bd+dirty |
| activation_matrix | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 768 | 0.3868 | 0.8232 | real | a44b9bd+dirty |
| belady | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 1024 | 0.9286 | 0.9286 | real | a44b9bd+dirty |
| lru | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 1024 | 0.8068 | 0.8068 | real | a44b9bd+dirty |
| lfu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 1024 | 0.7847 | 0.7847 | real | a44b9bd+dirty |
| lfu_k | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 1024 | 0.8068 | 0.8068 | real | a44b9bd+dirty |
| model_tracing | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 1024 | 0.8103 | 0.8103 | real | a44b9bd+dirty |
| lhu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 1024 | 0.7847 | 0.7847 | real | a44b9bd+dirty |
| arc | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 1024 | 0.8073 | 0.8073 | real | a44b9bd+dirty |
| 2q | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 1024 | 0.8004 | 0.8004 | real | a44b9bd+dirty |
| lru_k | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 1024 | 0.8026 | 0.8026 | real | a44b9bd+dirty |
| lirs | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 1024 | 0.7993 | 0.7993 | real | a44b9bd+dirty |
| s3fifo | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 1024 | 0.8068 | 0.8068 | real | a44b9bd+dirty |
| sieve | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 1024 | 0.7375 | 0.7375 | real | a44b9bd+dirty |
| w_tinylfu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 1024 | 0.7701 | 0.7701 | real | a44b9bd+dirty |
| layer_lru | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 1024 | 0.8059 | 0.8059 | real | a44b9bd+dirty |
| lcnu | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 1024 | 0.8113 | 0.8113 | real | a44b9bd+dirty |
| single_layer_ahead | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 1024 | 0.8063 | 0.9996 | real | a44b9bd+dirty |
| stride_prefetch | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 1024 | 0.8061 | 0.9992 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 1024 | 0.8058 | 0.9996 | real | a44b9bd+dirty |
| activation_matrix | qwen1.5-moe-a2.7b | mixed20-greedy64/decode | 1024 | 0.6100 | 0.8912 | real | a44b9bd+dirty |

<!-- run: python -m benchmarks.cache_simulator.research.run_policies_real --model qwen1.5-moe-a2.7b --dataset mixed20-greedy64 --ignore-layers 0 --prefill skip | env: python 3.12.3, numpy 2.4.4, Linux x86_64 | commit: a44b9bd+dirty -->

| policy | model | dataset | capacity | hit_rate | eff_hit_rate | provenance | commit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| belady | qwen3-30b-a3b | mmlu150-greedy64/decode | 128 | 0.3068 | 0.3068 | real | a44b9bd+dirty |
| lru | qwen3-30b-a3b | mmlu150-greedy64/decode | 128 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| lfu | qwen3-30b-a3b | mmlu150-greedy64/decode | 128 | 0.1255 | 0.1255 | real | a44b9bd+dirty |
| lfu_k | qwen3-30b-a3b | mmlu150-greedy64/decode | 128 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| model_tracing | qwen3-30b-a3b | mmlu150-greedy64/decode | 128 | 0.1259 | 0.1259 | real | a44b9bd+dirty |
| lhu | qwen3-30b-a3b | mmlu150-greedy64/decode | 128 | 0.0044 | 0.0044 | real | a44b9bd+dirty |
| arc | qwen3-30b-a3b | mmlu150-greedy64/decode | 128 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| 2q | qwen3-30b-a3b | mmlu150-greedy64/decode | 128 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| lru_k | qwen3-30b-a3b | mmlu150-greedy64/decode | 128 | 0.0272 | 0.0272 | real | a44b9bd+dirty |
| lirs | qwen3-30b-a3b | mmlu150-greedy64/decode | 128 | 0.1830 | 0.1830 | real | a44b9bd+dirty |
| s3fifo | qwen3-30b-a3b | mmlu150-greedy64/decode | 128 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| sieve | qwen3-30b-a3b | mmlu150-greedy64/decode | 128 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-30b-a3b | mmlu150-greedy64/decode | 128 | 0.1712 | 0.1712 | real | a44b9bd+dirty |
| layer_lru | qwen3-30b-a3b | mmlu150-greedy64/decode | 128 | 0.0159 | 0.0159 | real | a44b9bd+dirty |
| lcnu | qwen3-30b-a3b | mmlu150-greedy64/decode | 128 | 0.2147 | 0.2147 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode | 128 | 0.0000 | 1.0000 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-30b-a3b | mmlu150-greedy64/decode | 128 | 0.0000 | 1.0000 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode | 128 | 0.0000 | 1.0000 | real | a44b9bd+dirty |
| activation_matrix | qwen3-30b-a3b | mmlu150-greedy64/decode | 128 | 0.0000 | 0.6585 | real | a44b9bd+dirty |
| belady | qwen3-30b-a3b | mmlu150-greedy64/decode | 256 | 0.5013 | 0.5013 | real | a44b9bd+dirty |
| lru | qwen3-30b-a3b | mmlu150-greedy64/decode | 256 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| lfu | qwen3-30b-a3b | mmlu150-greedy64/decode | 256 | 0.2151 | 0.2151 | real | a44b9bd+dirty |
| lfu_k | qwen3-30b-a3b | mmlu150-greedy64/decode | 256 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| model_tracing | qwen3-30b-a3b | mmlu150-greedy64/decode | 256 | 0.2176 | 0.2176 | real | a44b9bd+dirty |
| lhu | qwen3-30b-a3b | mmlu150-greedy64/decode | 256 | 0.0165 | 0.0165 | real | a44b9bd+dirty |
| arc | qwen3-30b-a3b | mmlu150-greedy64/decode | 256 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| 2q | qwen3-30b-a3b | mmlu150-greedy64/decode | 256 | 0.0528 | 0.0528 | real | a44b9bd+dirty |
| lru_k | qwen3-30b-a3b | mmlu150-greedy64/decode | 256 | 0.2743 | 0.2743 | real | a44b9bd+dirty |
| lirs | qwen3-30b-a3b | mmlu150-greedy64/decode | 256 | 0.3026 | 0.3026 | real | a44b9bd+dirty |
| s3fifo | qwen3-30b-a3b | mmlu150-greedy64/decode | 256 | 0.0897 | 0.0897 | real | a44b9bd+dirty |
| sieve | qwen3-30b-a3b | mmlu150-greedy64/decode | 256 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-30b-a3b | mmlu150-greedy64/decode | 256 | 0.2746 | 0.2746 | real | a44b9bd+dirty |
| layer_lru | qwen3-30b-a3b | mmlu150-greedy64/decode | 256 | 0.0837 | 0.0837 | real | a44b9bd+dirty |
| lcnu | qwen3-30b-a3b | mmlu150-greedy64/decode | 256 | 0.3415 | 0.3415 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode | 256 | 0.0000 | 1.0000 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-30b-a3b | mmlu150-greedy64/decode | 256 | 0.0000 | 1.0000 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode | 256 | 0.0000 | 1.0000 | real | a44b9bd+dirty |
| activation_matrix | qwen3-30b-a3b | mmlu150-greedy64/decode | 256 | 0.0000 | 0.6585 | real | a44b9bd+dirty |
| belady | qwen3-30b-a3b | mmlu150-greedy64/decode | 384 | 0.6026 | 0.6026 | real | a44b9bd+dirty |
| lru | qwen3-30b-a3b | mmlu150-greedy64/decode | 384 | 0.3053 | 0.3053 | real | a44b9bd+dirty |
| lfu | qwen3-30b-a3b | mmlu150-greedy64/decode | 384 | 0.2864 | 0.2864 | real | a44b9bd+dirty |
| lfu_k | qwen3-30b-a3b | mmlu150-greedy64/decode | 384 | 0.3053 | 0.3053 | real | a44b9bd+dirty |
| model_tracing | qwen3-30b-a3b | mmlu150-greedy64/decode | 384 | 0.2919 | 0.2919 | real | a44b9bd+dirty |
| lhu | qwen3-30b-a3b | mmlu150-greedy64/decode | 384 | 0.3970 | 0.3970 | real | a44b9bd+dirty |
| arc | qwen3-30b-a3b | mmlu150-greedy64/decode | 384 | 0.3352 | 0.3352 | real | a44b9bd+dirty |
| 2q | qwen3-30b-a3b | mmlu150-greedy64/decode | 384 | 0.2310 | 0.2310 | real | a44b9bd+dirty |
| lru_k | qwen3-30b-a3b | mmlu150-greedy64/decode | 384 | 0.3749 | 0.3749 | real | a44b9bd+dirty |
| lirs | qwen3-30b-a3b | mmlu150-greedy64/decode | 384 | 0.3832 | 0.3832 | real | a44b9bd+dirty |
| s3fifo | qwen3-30b-a3b | mmlu150-greedy64/decode | 384 | 0.3620 | 0.3620 | real | a44b9bd+dirty |
| sieve | qwen3-30b-a3b | mmlu150-greedy64/decode | 384 | 0.0838 | 0.0838 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-30b-a3b | mmlu150-greedy64/decode | 384 | 0.3447 | 0.3447 | real | a44b9bd+dirty |
| layer_lru | qwen3-30b-a3b | mmlu150-greedy64/decode | 384 | 0.3053 | 0.3053 | real | a44b9bd+dirty |
| lcnu | qwen3-30b-a3b | mmlu150-greedy64/decode | 384 | 0.4263 | 0.4263 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode | 384 | 0.0098 | 0.9203 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-30b-a3b | mmlu150-greedy64/decode | 384 | 0.0596 | 0.6797 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode | 384 | 0.0003 | 0.9984 | real | a44b9bd+dirty |
| activation_matrix | qwen3-30b-a3b | mmlu150-greedy64/decode | 384 | 0.0000 | 0.6585 | real | a44b9bd+dirty |
| belady | qwen3-30b-a3b | mmlu150-greedy64/decode | 512 | 0.6677 | 0.6677 | real | a44b9bd+dirty |
| lru | qwen3-30b-a3b | mmlu150-greedy64/decode | 512 | 0.4290 | 0.4290 | real | a44b9bd+dirty |
| lfu | qwen3-30b-a3b | mmlu150-greedy64/decode | 512 | 0.3471 | 0.3471 | real | a44b9bd+dirty |
| lfu_k | qwen3-30b-a3b | mmlu150-greedy64/decode | 512 | 0.4290 | 0.4290 | real | a44b9bd+dirty |
| model_tracing | qwen3-30b-a3b | mmlu150-greedy64/decode | 512 | 0.3573 | 0.3573 | real | a44b9bd+dirty |
| lhu | qwen3-30b-a3b | mmlu150-greedy64/decode | 512 | 0.4295 | 0.4295 | real | a44b9bd+dirty |
| arc | qwen3-30b-a3b | mmlu150-greedy64/decode | 512 | 0.4334 | 0.4334 | real | a44b9bd+dirty |
| 2q | qwen3-30b-a3b | mmlu150-greedy64/decode | 512 | 0.4047 | 0.4047 | real | a44b9bd+dirty |
| lru_k | qwen3-30b-a3b | mmlu150-greedy64/decode | 512 | 0.4371 | 0.4371 | real | a44b9bd+dirty |
| lirs | qwen3-30b-a3b | mmlu150-greedy64/decode | 512 | 0.4432 | 0.4432 | real | a44b9bd+dirty |
| s3fifo | qwen3-30b-a3b | mmlu150-greedy64/decode | 512 | 0.4203 | 0.4203 | real | a44b9bd+dirty |
| sieve | qwen3-30b-a3b | mmlu150-greedy64/decode | 512 | 0.1730 | 0.1730 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-30b-a3b | mmlu150-greedy64/decode | 512 | 0.4019 | 0.4019 | real | a44b9bd+dirty |
| layer_lru | qwen3-30b-a3b | mmlu150-greedy64/decode | 512 | 0.4366 | 0.4366 | real | a44b9bd+dirty |
| lcnu | qwen3-30b-a3b | mmlu150-greedy64/decode | 512 | 0.4916 | 0.4916 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode | 512 | 0.4287 | 0.9998 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-30b-a3b | mmlu150-greedy64/decode | 512 | 0.4288 | 0.9998 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode | 512 | 0.4286 | 0.9999 | real | a44b9bd+dirty |
| activation_matrix | qwen3-30b-a3b | mmlu150-greedy64/decode | 512 | 0.0004 | 0.6586 | real | a44b9bd+dirty |
| belady | qwen3-30b-a3b | mmlu150-greedy64/decode | 768 | 0.7534 | 0.7534 | real | a44b9bd+dirty |
| lru | qwen3-30b-a3b | mmlu150-greedy64/decode | 768 | 0.5551 | 0.5551 | real | a44b9bd+dirty |
| lfu | qwen3-30b-a3b | mmlu150-greedy64/decode | 768 | 0.4504 | 0.4504 | real | a44b9bd+dirty |
| lfu_k | qwen3-30b-a3b | mmlu150-greedy64/decode | 768 | 0.5551 | 0.5551 | real | a44b9bd+dirty |
| model_tracing | qwen3-30b-a3b | mmlu150-greedy64/decode | 768 | 0.4652 | 0.4652 | real | a44b9bd+dirty |
| lhu | qwen3-30b-a3b | mmlu150-greedy64/decode | 768 | 0.5559 | 0.5559 | real | a44b9bd+dirty |
| arc | qwen3-30b-a3b | mmlu150-greedy64/decode | 768 | 0.5527 | 0.5527 | real | a44b9bd+dirty |
| 2q | qwen3-30b-a3b | mmlu150-greedy64/decode | 768 | 0.5345 | 0.5345 | real | a44b9bd+dirty |
| lru_k | qwen3-30b-a3b | mmlu150-greedy64/decode | 768 | 0.5312 | 0.5312 | real | a44b9bd+dirty |
| lirs | qwen3-30b-a3b | mmlu150-greedy64/decode | 768 | 0.5321 | 0.5321 | real | a44b9bd+dirty |
| s3fifo | qwen3-30b-a3b | mmlu150-greedy64/decode | 768 | 0.5213 | 0.5213 | real | a44b9bd+dirty |
| sieve | qwen3-30b-a3b | mmlu150-greedy64/decode | 768 | 0.3131 | 0.3131 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-30b-a3b | mmlu150-greedy64/decode | 768 | 0.4990 | 0.4990 | real | a44b9bd+dirty |
| layer_lru | qwen3-30b-a3b | mmlu150-greedy64/decode | 768 | 0.5596 | 0.5596 | real | a44b9bd+dirty |
| lcnu | qwen3-30b-a3b | mmlu150-greedy64/decode | 768 | 0.5907 | 0.5907 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode | 768 | 0.5538 | 0.9991 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-30b-a3b | mmlu150-greedy64/decode | 768 | 0.5542 | 0.9990 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode | 768 | 0.5528 | 0.9992 | real | a44b9bd+dirty |
| activation_matrix | qwen3-30b-a3b | mmlu150-greedy64/decode | 768 | 0.4221 | 0.7608 | real | a44b9bd+dirty |
| belady | qwen3-30b-a3b | mmlu150-greedy64/decode | 1024 | 0.8122 | 0.8122 | real | a44b9bd+dirty |
| lru | qwen3-30b-a3b | mmlu150-greedy64/decode | 1024 | 0.6358 | 0.6358 | real | a44b9bd+dirty |
| lfu | qwen3-30b-a3b | mmlu150-greedy64/decode | 1024 | 0.5383 | 0.5383 | real | a44b9bd+dirty |
| lfu_k | qwen3-30b-a3b | mmlu150-greedy64/decode | 1024 | 0.6358 | 0.6358 | real | a44b9bd+dirty |
| model_tracing | qwen3-30b-a3b | mmlu150-greedy64/decode | 1024 | 0.5528 | 0.5528 | real | a44b9bd+dirty |
| lhu | qwen3-30b-a3b | mmlu150-greedy64/decode | 1024 | 0.6338 | 0.6338 | real | a44b9bd+dirty |
| arc | qwen3-30b-a3b | mmlu150-greedy64/decode | 1024 | 0.6297 | 0.6297 | real | a44b9bd+dirty |
| 2q | qwen3-30b-a3b | mmlu150-greedy64/decode | 1024 | 0.6199 | 0.6199 | real | a44b9bd+dirty |
| lru_k | qwen3-30b-a3b | mmlu150-greedy64/decode | 1024 | 0.6086 | 0.6086 | real | a44b9bd+dirty |
| lirs | qwen3-30b-a3b | mmlu150-greedy64/decode | 1024 | 0.6078 | 0.6078 | real | a44b9bd+dirty |
| s3fifo | qwen3-30b-a3b | mmlu150-greedy64/decode | 1024 | 0.6160 | 0.6160 | real | a44b9bd+dirty |
| sieve | qwen3-30b-a3b | mmlu150-greedy64/decode | 1024 | 0.4377 | 0.4377 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-30b-a3b | mmlu150-greedy64/decode | 1024 | 0.5819 | 0.5819 | real | a44b9bd+dirty |
| layer_lru | qwen3-30b-a3b | mmlu150-greedy64/decode | 1024 | 0.6361 | 0.6361 | real | a44b9bd+dirty |
| lcnu | qwen3-30b-a3b | mmlu150-greedy64/decode | 1024 | 0.6620 | 0.6620 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode | 1024 | 0.6350 | 0.9994 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-30b-a3b | mmlu150-greedy64/decode | 1024 | 0.6353 | 0.9994 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode | 1024 | 0.6343 | 0.9994 | real | a44b9bd+dirty |
| activation_matrix | qwen3-30b-a3b | mmlu150-greedy64/decode | 1024 | 0.5178 | 0.7927 | real | a44b9bd+dirty |
| belady | qwen3-30b-a3b | mmlu150-greedy64/decode | 1536 | 0.8888 | 0.8888 | real | a44b9bd+dirty |
| lru | qwen3-30b-a3b | mmlu150-greedy64/decode | 1536 | 0.7493 | 0.7493 | real | a44b9bd+dirty |
| lfu | qwen3-30b-a3b | mmlu150-greedy64/decode | 1536 | 0.6816 | 0.6816 | real | a44b9bd+dirty |
| lfu_k | qwen3-30b-a3b | mmlu150-greedy64/decode | 1536 | 0.7493 | 0.7493 | real | a44b9bd+dirty |
| model_tracing | qwen3-30b-a3b | mmlu150-greedy64/decode | 1536 | 0.6966 | 0.6966 | real | a44b9bd+dirty |
| lhu | qwen3-30b-a3b | mmlu150-greedy64/decode | 1536 | 0.7497 | 0.7497 | real | a44b9bd+dirty |
| arc | qwen3-30b-a3b | mmlu150-greedy64/decode | 1536 | 0.7427 | 0.7427 | real | a44b9bd+dirty |
| 2q | qwen3-30b-a3b | mmlu150-greedy64/decode | 1536 | 0.7470 | 0.7470 | real | a44b9bd+dirty |
| lru_k | qwen3-30b-a3b | mmlu150-greedy64/decode | 1536 | 0.7361 | 0.7361 | real | a44b9bd+dirty |
| lirs | qwen3-30b-a3b | mmlu150-greedy64/decode | 1536 | 0.7342 | 0.7342 | real | a44b9bd+dirty |
| s3fifo | qwen3-30b-a3b | mmlu150-greedy64/decode | 1536 | 0.7522 | 0.7522 | real | a44b9bd+dirty |
| sieve | qwen3-30b-a3b | mmlu150-greedy64/decode | 1536 | 0.5378 | 0.5378 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-30b-a3b | mmlu150-greedy64/decode | 1536 | 0.7226 | 0.7226 | real | a44b9bd+dirty |
| layer_lru | qwen3-30b-a3b | mmlu150-greedy64/decode | 1536 | 0.7465 | 0.7465 | real | a44b9bd+dirty |
| lcnu | qwen3-30b-a3b | mmlu150-greedy64/decode | 1536 | 0.7729 | 0.7729 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode | 1536 | 0.7488 | 0.9997 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-30b-a3b | mmlu150-greedy64/decode | 1536 | 0.7490 | 0.9996 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode | 1536 | 0.7484 | 0.9996 | real | a44b9bd+dirty |
| activation_matrix | qwen3-30b-a3b | mmlu150-greedy64/decode | 1536 | 0.6681 | 0.8477 | real | a44b9bd+dirty |
| belady | qwen3-30b-a3b | mmlu150-greedy64/decode | 2048 | 0.9354 | 0.9354 | real | a44b9bd+dirty |
| lru | qwen3-30b-a3b | mmlu150-greedy64/decode | 2048 | 0.8410 | 0.8410 | real | a44b9bd+dirty |
| lfu | qwen3-30b-a3b | mmlu150-greedy64/decode | 2048 | 0.7917 | 0.7917 | real | a44b9bd+dirty |
| lfu_k | qwen3-30b-a3b | mmlu150-greedy64/decode | 2048 | 0.8410 | 0.8410 | real | a44b9bd+dirty |
| model_tracing | qwen3-30b-a3b | mmlu150-greedy64/decode | 2048 | 0.8045 | 0.8045 | real | a44b9bd+dirty |
| lhu | qwen3-30b-a3b | mmlu150-greedy64/decode | 2048 | 0.8442 | 0.8442 | real | a44b9bd+dirty |
| arc | qwen3-30b-a3b | mmlu150-greedy64/decode | 2048 | 0.8432 | 0.8432 | real | a44b9bd+dirty |
| 2q | qwen3-30b-a3b | mmlu150-greedy64/decode | 2048 | 0.8349 | 0.8349 | real | a44b9bd+dirty |
| lru_k | qwen3-30b-a3b | mmlu150-greedy64/decode | 2048 | 0.8318 | 0.8318 | real | a44b9bd+dirty |
| lirs | qwen3-30b-a3b | mmlu150-greedy64/decode | 2048 | 0.8294 | 0.8294 | real | a44b9bd+dirty |
| s3fifo | qwen3-30b-a3b | mmlu150-greedy64/decode | 2048 | 0.8532 | 0.8532 | real | a44b9bd+dirty |
| sieve | qwen3-30b-a3b | mmlu150-greedy64/decode | 2048 | 0.6497 | 0.6497 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-30b-a3b | mmlu150-greedy64/decode | 2048 | 0.8290 | 0.8290 | real | a44b9bd+dirty |
| layer_lru | qwen3-30b-a3b | mmlu150-greedy64/decode | 2048 | 0.8323 | 0.8323 | real | a44b9bd+dirty |
| lcnu | qwen3-30b-a3b | mmlu150-greedy64/decode | 2048 | 0.8621 | 0.8621 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode | 2048 | 0.8407 | 0.9998 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-30b-a3b | mmlu150-greedy64/decode | 2048 | 0.8408 | 0.9998 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode | 2048 | 0.8405 | 0.9998 | real | a44b9bd+dirty |
| activation_matrix | qwen3-30b-a3b | mmlu150-greedy64/decode | 2048 | 0.7870 | 0.8903 | real | a44b9bd+dirty |
| belady | qwen3-30b-a3b | mmlu150-greedy64/decode | 3072 | 0.9808 | 0.9808 | real | a44b9bd+dirty |
| lru | qwen3-30b-a3b | mmlu150-greedy64/decode | 3072 | 0.9465 | 0.9465 | real | a44b9bd+dirty |
| lfu | qwen3-30b-a3b | mmlu150-greedy64/decode | 3072 | 0.9260 | 0.9260 | real | a44b9bd+dirty |
| lfu_k | qwen3-30b-a3b | mmlu150-greedy64/decode | 3072 | 0.9465 | 0.9465 | real | a44b9bd+dirty |
| model_tracing | qwen3-30b-a3b | mmlu150-greedy64/decode | 3072 | 0.9340 | 0.9340 | real | a44b9bd+dirty |
| lhu | qwen3-30b-a3b | mmlu150-greedy64/decode | 3072 | 0.9490 | 0.9490 | real | a44b9bd+dirty |
| arc | qwen3-30b-a3b | mmlu150-greedy64/decode | 3072 | 0.9468 | 0.9468 | real | a44b9bd+dirty |
| 2q | qwen3-30b-a3b | mmlu150-greedy64/decode | 3072 | 0.9413 | 0.9413 | real | a44b9bd+dirty |
| lru_k | qwen3-30b-a3b | mmlu150-greedy64/decode | 3072 | 0.9491 | 0.9491 | real | a44b9bd+dirty |
| lirs | qwen3-30b-a3b | mmlu150-greedy64/decode | 3072 | 0.9489 | 0.9489 | real | a44b9bd+dirty |
| s3fifo | qwen3-30b-a3b | mmlu150-greedy64/decode | 3072 | 0.9554 | 0.9554 | real | a44b9bd+dirty |
| sieve | qwen3-30b-a3b | mmlu150-greedy64/decode | 3072 | 0.8378 | 0.8378 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-30b-a3b | mmlu150-greedy64/decode | 3072 | 0.9495 | 0.9495 | real | a44b9bd+dirty |
| layer_lru | qwen3-30b-a3b | mmlu150-greedy64/decode | 3072 | 0.9360 | 0.9360 | real | a44b9bd+dirty |
| lcnu | qwen3-30b-a3b | mmlu150-greedy64/decode | 3072 | 0.9560 | 0.9560 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode | 3072 | 0.9465 | 1.0000 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-30b-a3b | mmlu150-greedy64/decode | 3072 | 0.9465 | 1.0000 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode | 3072 | 0.9464 | 1.0000 | real | a44b9bd+dirty |
| activation_matrix | qwen3-30b-a3b | mmlu150-greedy64/decode | 3072 | 0.9324 | 0.9524 | real | a44b9bd+dirty |
| belady | qwen3-30b-a3b | mmlu150-greedy64/decode | 4096 | 0.9956 | 0.9956 | real | a44b9bd+dirty |
| lru | qwen3-30b-a3b | mmlu150-greedy64/decode | 4096 | 0.9892 | 0.9892 | real | a44b9bd+dirty |
| lfu | qwen3-30b-a3b | mmlu150-greedy64/decode | 4096 | 0.9818 | 0.9818 | real | a44b9bd+dirty |
| lfu_k | qwen3-30b-a3b | mmlu150-greedy64/decode | 4096 | 0.9892 | 0.9892 | real | a44b9bd+dirty |
| model_tracing | qwen3-30b-a3b | mmlu150-greedy64/decode | 4096 | 0.9852 | 0.9852 | real | a44b9bd+dirty |
| lhu | qwen3-30b-a3b | mmlu150-greedy64/decode | 4096 | 0.9854 | 0.9854 | real | a44b9bd+dirty |
| arc | qwen3-30b-a3b | mmlu150-greedy64/decode | 4096 | 0.9894 | 0.9894 | real | a44b9bd+dirty |
| 2q | qwen3-30b-a3b | mmlu150-greedy64/decode | 4096 | 0.9854 | 0.9854 | real | a44b9bd+dirty |
| lru_k | qwen3-30b-a3b | mmlu150-greedy64/decode | 4096 | 0.9899 | 0.9899 | real | a44b9bd+dirty |
| lirs | qwen3-30b-a3b | mmlu150-greedy64/decode | 4096 | 0.9904 | 0.9904 | real | a44b9bd+dirty |
| s3fifo | qwen3-30b-a3b | mmlu150-greedy64/decode | 4096 | 0.9903 | 0.9903 | real | a44b9bd+dirty |
| sieve | qwen3-30b-a3b | mmlu150-greedy64/decode | 4096 | 0.9886 | 0.9886 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-30b-a3b | mmlu150-greedy64/decode | 4096 | 0.9901 | 0.9901 | real | a44b9bd+dirty |
| layer_lru | qwen3-30b-a3b | mmlu150-greedy64/decode | 4096 | 0.9825 | 0.9825 | real | a44b9bd+dirty |
| lcnu | qwen3-30b-a3b | mmlu150-greedy64/decode | 4096 | 0.9895 | 0.9895 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode | 4096 | 0.9892 | 1.0000 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-30b-a3b | mmlu150-greedy64/decode | 4096 | 0.9892 | 1.0000 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode | 4096 | 0.9892 | 1.0000 | real | a44b9bd+dirty |
| activation_matrix | qwen3-30b-a3b | mmlu150-greedy64/decode | 4096 | 0.9873 | 0.9890 | real | a44b9bd+dirty |

<!-- run: python -m benchmarks.cache_simulator.research.run_policies_real --model qwen3-30b-a3b --dataset mmlu150-greedy64 --ignore-layers 0 --prefill skip | env: python 3.12.3, numpy 2.4.4, Linux x86_64 | commit: a44b9bd+dirty -->

| policy | model | dataset | capacity | hit_rate | eff_hit_rate | provenance | commit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| belady | qwen3-235b | mmlu100-greedy64/decode | 256 | 0.3067 | 0.3067 | real | a44b9bd+dirty |
| lru | qwen3-235b | mmlu100-greedy64/decode | 256 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| lfu | qwen3-235b | mmlu100-greedy64/decode | 256 | 0.1577 | 0.1577 | real | a44b9bd+dirty |
| lfu_k | qwen3-235b | mmlu100-greedy64/decode | 256 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| model_tracing | qwen3-235b | mmlu100-greedy64/decode | 256 | 0.1584 | 0.1584 | real | a44b9bd+dirty |
| lhu | qwen3-235b | mmlu100-greedy64/decode | 256 | 0.0065 | 0.0065 | real | a44b9bd+dirty |
| arc | qwen3-235b | mmlu100-greedy64/decode | 256 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| 2q | qwen3-235b | mmlu100-greedy64/decode | 256 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| lru_k | qwen3-235b | mmlu100-greedy64/decode | 256 | 0.0396 | 0.0396 | real | a44b9bd+dirty |
| lirs | qwen3-235b | mmlu100-greedy64/decode | 256 | 0.1731 | 0.1731 | real | a44b9bd+dirty |
| s3fifo | qwen3-235b | mmlu100-greedy64/decode | 256 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| sieve | qwen3-235b | mmlu100-greedy64/decode | 256 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-235b | mmlu100-greedy64/decode | 256 | 0.1683 | 0.1683 | real | a44b9bd+dirty |
| layer_lru | qwen3-235b | mmlu100-greedy64/decode | 256 | 0.0171 | 0.0171 | real | a44b9bd+dirty |
| lcnu | qwen3-235b | mmlu100-greedy64/decode | 256 | 0.2051 | 0.2051 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-235b | mmlu100-greedy64/decode | 256 | 0.0000 | 1.0000 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-235b | mmlu100-greedy64/decode | 256 | 0.0000 | 1.0000 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-235b | mmlu100-greedy64/decode | 256 | 0.0000 | 1.0000 | real | a44b9bd+dirty |
| activation_matrix | qwen3-235b | mmlu100-greedy64/decode | 256 | 0.0000 | 0.6234 | real | a44b9bd+dirty |
| belady | qwen3-235b | mmlu100-greedy64/decode | 512 | 0.4898 | 0.4898 | real | a44b9bd+dirty |
| lru | qwen3-235b | mmlu100-greedy64/decode | 512 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| lfu | qwen3-235b | mmlu100-greedy64/decode | 512 | 0.2561 | 0.2561 | real | a44b9bd+dirty |
| lfu_k | qwen3-235b | mmlu100-greedy64/decode | 512 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| model_tracing | qwen3-235b | mmlu100-greedy64/decode | 512 | 0.2570 | 0.2570 | real | a44b9bd+dirty |
| lhu | qwen3-235b | mmlu100-greedy64/decode | 512 | 0.0248 | 0.0248 | real | a44b9bd+dirty |
| arc | qwen3-235b | mmlu100-greedy64/decode | 512 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| 2q | qwen3-235b | mmlu100-greedy64/decode | 512 | 0.1109 | 0.1109 | real | a44b9bd+dirty |
| lru_k | qwen3-235b | mmlu100-greedy64/decode | 512 | 0.2550 | 0.2550 | real | a44b9bd+dirty |
| lirs | qwen3-235b | mmlu100-greedy64/decode | 512 | 0.2820 | 0.2820 | real | a44b9bd+dirty |
| s3fifo | qwen3-235b | mmlu100-greedy64/decode | 512 | 0.1059 | 0.1059 | real | a44b9bd+dirty |
| sieve | qwen3-235b | mmlu100-greedy64/decode | 512 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-235b | mmlu100-greedy64/decode | 512 | 0.2725 | 0.2725 | real | a44b9bd+dirty |
| layer_lru | qwen3-235b | mmlu100-greedy64/decode | 512 | 0.0868 | 0.0868 | real | a44b9bd+dirty |
| lcnu | qwen3-235b | mmlu100-greedy64/decode | 512 | 0.3272 | 0.3272 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-235b | mmlu100-greedy64/decode | 512 | 0.0000 | 1.0000 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-235b | mmlu100-greedy64/decode | 512 | 0.0000 | 1.0000 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-235b | mmlu100-greedy64/decode | 512 | 0.0000 | 1.0000 | real | a44b9bd+dirty |
| activation_matrix | qwen3-235b | mmlu100-greedy64/decode | 512 | 0.0000 | 0.6234 | real | a44b9bd+dirty |
| belady | qwen3-235b | mmlu100-greedy64/decode | 768 | 0.5906 | 0.5906 | real | a44b9bd+dirty |
| lru | qwen3-235b | mmlu100-greedy64/decode | 768 | 0.3841 | 0.3841 | real | a44b9bd+dirty |
| lfu | qwen3-235b | mmlu100-greedy64/decode | 768 | 0.3317 | 0.3317 | real | a44b9bd+dirty |
| lfu_k | qwen3-235b | mmlu100-greedy64/decode | 768 | 0.3841 | 0.3841 | real | a44b9bd+dirty |
| model_tracing | qwen3-235b | mmlu100-greedy64/decode | 768 | 0.3327 | 0.3327 | real | a44b9bd+dirty |
| lhu | qwen3-235b | mmlu100-greedy64/decode | 768 | 0.3735 | 0.3735 | real | a44b9bd+dirty |
| arc | qwen3-235b | mmlu100-greedy64/decode | 768 | 0.3288 | 0.3288 | real | a44b9bd+dirty |
| 2q | qwen3-235b | mmlu100-greedy64/decode | 768 | 0.2114 | 0.2114 | real | a44b9bd+dirty |
| lru_k | qwen3-235b | mmlu100-greedy64/decode | 768 | 0.3568 | 0.3568 | real | a44b9bd+dirty |
| lirs | qwen3-235b | mmlu100-greedy64/decode | 768 | 0.3662 | 0.3662 | real | a44b9bd+dirty |
| s3fifo | qwen3-235b | mmlu100-greedy64/decode | 768 | 0.3481 | 0.3481 | real | a44b9bd+dirty |
| sieve | qwen3-235b | mmlu100-greedy64/decode | 768 | 0.1424 | 0.1424 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-235b | mmlu100-greedy64/decode | 768 | 0.3412 | 0.3412 | real | a44b9bd+dirty |
| layer_lru | qwen3-235b | mmlu100-greedy64/decode | 768 | 0.2787 | 0.2787 | real | a44b9bd+dirty |
| lcnu | qwen3-235b | mmlu100-greedy64/decode | 768 | 0.4130 | 0.4130 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-235b | mmlu100-greedy64/decode | 768 | 0.3841 | 1.0000 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-235b | mmlu100-greedy64/decode | 768 | 0.3841 | 1.0000 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-235b | mmlu100-greedy64/decode | 768 | 0.3620 | 0.9779 | real | a44b9bd+dirty |
| activation_matrix | qwen3-235b | mmlu100-greedy64/decode | 768 | 0.0001 | 0.6234 | real | a44b9bd+dirty |
| belady | qwen3-235b | mmlu100-greedy64/decode | 1024 | 0.6581 | 0.6581 | real | a44b9bd+dirty |
| lru | qwen3-235b | mmlu100-greedy64/decode | 1024 | 0.3850 | 0.3850 | real | a44b9bd+dirty |
| lfu | qwen3-235b | mmlu100-greedy64/decode | 1024 | 0.3941 | 0.3941 | real | a44b9bd+dirty |
| lfu_k | qwen3-235b | mmlu100-greedy64/decode | 1024 | 0.3850 | 0.3850 | real | a44b9bd+dirty |
| model_tracing | qwen3-235b | mmlu100-greedy64/decode | 1024 | 0.3954 | 0.3954 | real | a44b9bd+dirty |
| lhu | qwen3-235b | mmlu100-greedy64/decode | 1024 | 0.3884 | 0.3884 | real | a44b9bd+dirty |
| arc | qwen3-235b | mmlu100-greedy64/decode | 1024 | 0.4112 | 0.4112 | real | a44b9bd+dirty |
| 2q | qwen3-235b | mmlu100-greedy64/decode | 1024 | 0.3799 | 0.3799 | real | a44b9bd+dirty |
| lru_k | qwen3-235b | mmlu100-greedy64/decode | 1024 | 0.4237 | 0.4237 | real | a44b9bd+dirty |
| lirs | qwen3-235b | mmlu100-greedy64/decode | 1024 | 0.4315 | 0.4315 | real | a44b9bd+dirty |
| s3fifo | qwen3-235b | mmlu100-greedy64/decode | 1024 | 0.4099 | 0.4099 | real | a44b9bd+dirty |
| sieve | qwen3-235b | mmlu100-greedy64/decode | 1024 | 0.2302 | 0.2302 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-235b | mmlu100-greedy64/decode | 1024 | 0.3973 | 0.3973 | real | a44b9bd+dirty |
| layer_lru | qwen3-235b | mmlu100-greedy64/decode | 1024 | 0.3964 | 0.3964 | real | a44b9bd+dirty |
| lcnu | qwen3-235b | mmlu100-greedy64/decode | 1024 | 0.4808 | 0.4808 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-235b | mmlu100-greedy64/decode | 1024 | 0.3849 | 0.9999 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-235b | mmlu100-greedy64/decode | 1024 | 0.3849 | 0.9999 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-235b | mmlu100-greedy64/decode | 1024 | 0.3848 | 0.9999 | real | a44b9bd+dirty |
| activation_matrix | qwen3-235b | mmlu100-greedy64/decode | 1024 | 0.0005 | 0.6234 | real | a44b9bd+dirty |
| belady | qwen3-235b | mmlu100-greedy64/decode | 1536 | 0.7485 | 0.7485 | real | a44b9bd+dirty |
| lru | qwen3-235b | mmlu100-greedy64/decode | 1536 | 0.5253 | 0.5253 | real | a44b9bd+dirty |
| lfu | qwen3-235b | mmlu100-greedy64/decode | 1536 | 0.4966 | 0.4966 | real | a44b9bd+dirty |
| lfu_k | qwen3-235b | mmlu100-greedy64/decode | 1536 | 0.5253 | 0.5253 | real | a44b9bd+dirty |
| model_tracing | qwen3-235b | mmlu100-greedy64/decode | 1536 | 0.4982 | 0.4982 | real | a44b9bd+dirty |
| lhu | qwen3-235b | mmlu100-greedy64/decode | 1536 | 0.5270 | 0.5270 | real | a44b9bd+dirty |
| arc | qwen3-235b | mmlu100-greedy64/decode | 1536 | 0.5320 | 0.5320 | real | a44b9bd+dirty |
| 2q | qwen3-235b | mmlu100-greedy64/decode | 1536 | 0.5220 | 0.5220 | real | a44b9bd+dirty |
| lru_k | qwen3-235b | mmlu100-greedy64/decode | 1536 | 0.5293 | 0.5293 | real | a44b9bd+dirty |
| lirs | qwen3-235b | mmlu100-greedy64/decode | 1536 | 0.5304 | 0.5304 | real | a44b9bd+dirty |
| s3fifo | qwen3-235b | mmlu100-greedy64/decode | 1536 | 0.5128 | 0.5128 | real | a44b9bd+dirty |
| sieve | qwen3-235b | mmlu100-greedy64/decode | 1536 | 0.3893 | 0.3893 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-235b | mmlu100-greedy64/decode | 1536 | 0.4899 | 0.4899 | real | a44b9bd+dirty |
| layer_lru | qwen3-235b | mmlu100-greedy64/decode | 1536 | 0.5285 | 0.5285 | real | a44b9bd+dirty |
| lcnu | qwen3-235b | mmlu100-greedy64/decode | 1536 | 0.5840 | 0.5840 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-235b | mmlu100-greedy64/decode | 1536 | 0.5246 | 0.9996 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-235b | mmlu100-greedy64/decode | 1536 | 0.5249 | 0.9995 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-235b | mmlu100-greedy64/decode | 1536 | 0.5241 | 0.9996 | real | a44b9bd+dirty |
| activation_matrix | qwen3-235b | mmlu100-greedy64/decode | 1536 | 0.3853 | 0.7303 | real | a44b9bd+dirty |
| belady | qwen3-235b | mmlu100-greedy64/decode | 2048 | 0.8082 | 0.8082 | real | a44b9bd+dirty |
| lru | qwen3-235b | mmlu100-greedy64/decode | 2048 | 0.6171 | 0.6171 | real | a44b9bd+dirty |
| lfu | qwen3-235b | mmlu100-greedy64/decode | 2048 | 0.5805 | 0.5805 | real | a44b9bd+dirty |
| lfu_k | qwen3-235b | mmlu100-greedy64/decode | 2048 | 0.6171 | 0.6171 | real | a44b9bd+dirty |
| model_tracing | qwen3-235b | mmlu100-greedy64/decode | 2048 | 0.5824 | 0.5824 | real | a44b9bd+dirty |
| lhu | qwen3-235b | mmlu100-greedy64/decode | 2048 | 0.6172 | 0.6172 | real | a44b9bd+dirty |
| arc | qwen3-235b | mmlu100-greedy64/decode | 2048 | 0.6181 | 0.6181 | real | a44b9bd+dirty |
| 2q | qwen3-235b | mmlu100-greedy64/decode | 2048 | 0.6116 | 0.6116 | real | a44b9bd+dirty |
| lru_k | qwen3-235b | mmlu100-greedy64/decode | 2048 | 0.6098 | 0.6098 | real | a44b9bd+dirty |
| lirs | qwen3-235b | mmlu100-greedy64/decode | 2048 | 0.6087 | 0.6087 | real | a44b9bd+dirty |
| s3fifo | qwen3-235b | mmlu100-greedy64/decode | 2048 | 0.6111 | 0.6111 | real | a44b9bd+dirty |
| sieve | qwen3-235b | mmlu100-greedy64/decode | 2048 | 0.4929 | 0.4929 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-235b | mmlu100-greedy64/decode | 2048 | 0.5685 | 0.5685 | real | a44b9bd+dirty |
| layer_lru | qwen3-235b | mmlu100-greedy64/decode | 2048 | 0.6195 | 0.6195 | real | a44b9bd+dirty |
| lcnu | qwen3-235b | mmlu100-greedy64/decode | 2048 | 0.6608 | 0.6608 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-235b | mmlu100-greedy64/decode | 2048 | 0.6165 | 0.9996 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-235b | mmlu100-greedy64/decode | 2048 | 0.6167 | 0.9996 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-235b | mmlu100-greedy64/decode | 2048 | 0.6161 | 0.9996 | real | a44b9bd+dirty |
| activation_matrix | qwen3-235b | mmlu100-greedy64/decode | 2048 | 0.5024 | 0.7701 | real | a44b9bd+dirty |
| belady | qwen3-235b | mmlu100-greedy64/decode | 3072 | 0.8859 | 0.8859 | real | a44b9bd+dirty |
| lru | qwen3-235b | mmlu100-greedy64/decode | 3072 | 0.7463 | 0.7463 | real | a44b9bd+dirty |
| lfu | qwen3-235b | mmlu100-greedy64/decode | 3072 | 0.7147 | 0.7147 | real | a44b9bd+dirty |
| lfu_k | qwen3-235b | mmlu100-greedy64/decode | 3072 | 0.7463 | 0.7463 | real | a44b9bd+dirty |
| model_tracing | qwen3-235b | mmlu100-greedy64/decode | 3072 | 0.7169 | 0.7169 | real | a44b9bd+dirty |
| lhu | qwen3-235b | mmlu100-greedy64/decode | 3072 | 0.7442 | 0.7442 | real | a44b9bd+dirty |
| arc | qwen3-235b | mmlu100-greedy64/decode | 3072 | 0.7373 | 0.7373 | real | a44b9bd+dirty |
| 2q | qwen3-235b | mmlu100-greedy64/decode | 3072 | 0.7370 | 0.7370 | real | a44b9bd+dirty |
| lru_k | qwen3-235b | mmlu100-greedy64/decode | 3072 | 0.7319 | 0.7319 | real | a44b9bd+dirty |
| lirs | qwen3-235b | mmlu100-greedy64/decode | 3072 | 0.7286 | 0.7286 | real | a44b9bd+dirty |
| s3fifo | qwen3-235b | mmlu100-greedy64/decode | 3072 | 0.7465 | 0.7465 | real | a44b9bd+dirty |
| sieve | qwen3-235b | mmlu100-greedy64/decode | 3072 | 0.6174 | 0.6174 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-235b | mmlu100-greedy64/decode | 3072 | 0.6986 | 0.6986 | real | a44b9bd+dirty |
| layer_lru | qwen3-235b | mmlu100-greedy64/decode | 3072 | 0.7436 | 0.7436 | real | a44b9bd+dirty |
| lcnu | qwen3-235b | mmlu100-greedy64/decode | 3072 | 0.7765 | 0.7765 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-235b | mmlu100-greedy64/decode | 3072 | 0.7460 | 0.9998 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-235b | mmlu100-greedy64/decode | 3072 | 0.7461 | 0.9998 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-235b | mmlu100-greedy64/decode | 3072 | 0.7458 | 0.9998 | real | a44b9bd+dirty |
| activation_matrix | qwen3-235b | mmlu100-greedy64/decode | 3072 | 0.6692 | 0.8330 | real | a44b9bd+dirty |
| belady | qwen3-235b | mmlu100-greedy64/decode | 4096 | 0.9320 | 0.9320 | real | a44b9bd+dirty |
| lru | qwen3-235b | mmlu100-greedy64/decode | 4096 | 0.8334 | 0.8334 | real | a44b9bd+dirty |
| lfu | qwen3-235b | mmlu100-greedy64/decode | 4096 | 0.8150 | 0.8150 | real | a44b9bd+dirty |
| lfu_k | qwen3-235b | mmlu100-greedy64/decode | 4096 | 0.8334 | 0.8334 | real | a44b9bd+dirty |
| model_tracing | qwen3-235b | mmlu100-greedy64/decode | 4096 | 0.8173 | 0.8173 | real | a44b9bd+dirty |
| lhu | qwen3-235b | mmlu100-greedy64/decode | 4096 | 0.8336 | 0.8336 | real | a44b9bd+dirty |
| arc | qwen3-235b | mmlu100-greedy64/decode | 4096 | 0.8350 | 0.8350 | real | a44b9bd+dirty |
| 2q | qwen3-235b | mmlu100-greedy64/decode | 4096 | 0.8262 | 0.8262 | real | a44b9bd+dirty |
| lru_k | qwen3-235b | mmlu100-greedy64/decode | 4096 | 0.8211 | 0.8211 | real | a44b9bd+dirty |
| lirs | qwen3-235b | mmlu100-greedy64/decode | 4096 | 0.8180 | 0.8180 | real | a44b9bd+dirty |
| s3fifo | qwen3-235b | mmlu100-greedy64/decode | 4096 | 0.8413 | 0.8413 | real | a44b9bd+dirty |
| sieve | qwen3-235b | mmlu100-greedy64/decode | 4096 | 0.7132 | 0.7132 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-235b | mmlu100-greedy64/decode | 4096 | 0.8024 | 0.8024 | real | a44b9bd+dirty |
| layer_lru | qwen3-235b | mmlu100-greedy64/decode | 4096 | 0.8289 | 0.8289 | real | a44b9bd+dirty |
| lcnu | qwen3-235b | mmlu100-greedy64/decode | 4096 | 0.8602 | 0.8602 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-235b | mmlu100-greedy64/decode | 4096 | 0.8333 | 0.9999 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-235b | mmlu100-greedy64/decode | 4096 | 0.8334 | 0.9999 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-235b | mmlu100-greedy64/decode | 4096 | 0.8332 | 0.9999 | real | a44b9bd+dirty |
| activation_matrix | qwen3-235b | mmlu100-greedy64/decode | 4096 | 0.7881 | 0.8827 | real | a44b9bd+dirty |
| belady | qwen3-235b | mmlu100-greedy64/decode | 6144 | 0.9785 | 0.9785 | real | a44b9bd+dirty |
| lru | qwen3-235b | mmlu100-greedy64/decode | 6144 | 0.9407 | 0.9407 | real | a44b9bd+dirty |
| lfu | qwen3-235b | mmlu100-greedy64/decode | 6144 | 0.9389 | 0.9389 | real | a44b9bd+dirty |
| lfu_k | qwen3-235b | mmlu100-greedy64/decode | 6144 | 0.9407 | 0.9407 | real | a44b9bd+dirty |
| model_tracing | qwen3-235b | mmlu100-greedy64/decode | 6144 | 0.9405 | 0.9405 | real | a44b9bd+dirty |
| lhu | qwen3-235b | mmlu100-greedy64/decode | 6144 | 0.9432 | 0.9432 | real | a44b9bd+dirty |
| arc | qwen3-235b | mmlu100-greedy64/decode | 6144 | 0.9410 | 0.9410 | real | a44b9bd+dirty |
| 2q | qwen3-235b | mmlu100-greedy64/decode | 6144 | 0.9347 | 0.9347 | real | a44b9bd+dirty |
| lru_k | qwen3-235b | mmlu100-greedy64/decode | 6144 | 0.9402 | 0.9402 | real | a44b9bd+dirty |
| lirs | qwen3-235b | mmlu100-greedy64/decode | 6144 | 0.9390 | 0.9390 | real | a44b9bd+dirty |
| s3fifo | qwen3-235b | mmlu100-greedy64/decode | 6144 | 0.9490 | 0.9490 | real | a44b9bd+dirty |
| sieve | qwen3-235b | mmlu100-greedy64/decode | 6144 | 0.8849 | 0.8849 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-235b | mmlu100-greedy64/decode | 6144 | 0.9351 | 0.9351 | real | a44b9bd+dirty |
| layer_lru | qwen3-235b | mmlu100-greedy64/decode | 6144 | 0.9340 | 0.9340 | real | a44b9bd+dirty |
| lcnu | qwen3-235b | mmlu100-greedy64/decode | 6144 | 0.9553 | 0.9553 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-235b | mmlu100-greedy64/decode | 6144 | 0.9407 | 1.0000 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-235b | mmlu100-greedy64/decode | 6144 | 0.9407 | 1.0000 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-235b | mmlu100-greedy64/decode | 6144 | 0.9406 | 1.0000 | real | a44b9bd+dirty |
| activation_matrix | qwen3-235b | mmlu100-greedy64/decode | 6144 | 0.9304 | 0.9513 | real | a44b9bd+dirty |
| belady | qwen3-235b | mmlu100-greedy64/decode | 8192 | 0.9947 | 0.9947 | real | a44b9bd+dirty |
| lru | qwen3-235b | mmlu100-greedy64/decode | 8192 | 0.9876 | 0.9876 | real | a44b9bd+dirty |
| lfu | qwen3-235b | mmlu100-greedy64/decode | 8192 | 0.9892 | 0.9892 | real | a44b9bd+dirty |
| lfu_k | qwen3-235b | mmlu100-greedy64/decode | 8192 | 0.9876 | 0.9876 | real | a44b9bd+dirty |
| model_tracing | qwen3-235b | mmlu100-greedy64/decode | 8192 | 0.9898 | 0.9898 | real | a44b9bd+dirty |
| lhu | qwen3-235b | mmlu100-greedy64/decode | 8192 | 0.9892 | 0.9892 | real | a44b9bd+dirty |
| arc | qwen3-235b | mmlu100-greedy64/decode | 8192 | 0.9878 | 0.9878 | real | a44b9bd+dirty |
| 2q | qwen3-235b | mmlu100-greedy64/decode | 8192 | 0.9831 | 0.9831 | real | a44b9bd+dirty |
| lru_k | qwen3-235b | mmlu100-greedy64/decode | 8192 | 0.9875 | 0.9875 | real | a44b9bd+dirty |
| lirs | qwen3-235b | mmlu100-greedy64/decode | 8192 | 0.9881 | 0.9881 | real | a44b9bd+dirty |
| s3fifo | qwen3-235b | mmlu100-greedy64/decode | 8192 | 0.9888 | 0.9888 | real | a44b9bd+dirty |
| sieve | qwen3-235b | mmlu100-greedy64/decode | 8192 | 0.9865 | 0.9865 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-235b | mmlu100-greedy64/decode | 8192 | 0.9885 | 0.9885 | real | a44b9bd+dirty |
| layer_lru | qwen3-235b | mmlu100-greedy64/decode | 8192 | 0.9826 | 0.9826 | real | a44b9bd+dirty |
| lcnu | qwen3-235b | mmlu100-greedy64/decode | 8192 | 0.9911 | 0.9911 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-235b | mmlu100-greedy64/decode | 8192 | 0.9875 | 1.0000 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-235b | mmlu100-greedy64/decode | 8192 | 0.9875 | 1.0000 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-235b | mmlu100-greedy64/decode | 8192 | 0.9875 | 1.0000 | real | a44b9bd+dirty |
| activation_matrix | qwen3-235b | mmlu100-greedy64/decode | 8192 | 0.9868 | 0.9890 | real | a44b9bd+dirty |

<!-- run: python -m benchmarks.cache_simulator.research.run_policies_real --model qwen3-235b --dataset mmlu100-greedy64 --ignore-layers 0 --prefill skip | env: python 3.12.3, numpy 2.4.4, Linux x86_64 | commit: a44b9bd+dirty -->

| policy | model | dataset | capacity | hit_rate | eff_hit_rate | provenance | commit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| belady | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 128 | 0.2775 | 0.2775 | real | a44b9bd+dirty |
| lru | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 128 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| lfu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 128 | 0.1146 | 0.1146 | real | a44b9bd+dirty |
| lfu_k | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 128 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| model_tracing | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 128 | 0.1150 | 0.1150 | real | a44b9bd+dirty |
| lhu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 128 | 0.0037 | 0.0037 | real | a44b9bd+dirty |
| arc | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 128 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| 2q | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 128 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| lru_k | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 128 | 0.0233 | 0.0233 | real | a44b9bd+dirty |
| lirs | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 128 | 0.1647 | 0.1647 | real | a44b9bd+dirty |
| s3fifo | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 128 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| sieve | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 128 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 128 | 0.1544 | 0.1544 | real | a44b9bd+dirty |
| layer_lru | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 128 | 0.0141 | 0.0141 | real | a44b9bd+dirty |
| lcnu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 128 | 0.1940 | 0.1940 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 128 | 0.0000 | 0.8922 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 128 | 0.0000 | 1.0000 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 128 | 0.0000 | 0.8868 | real | a44b9bd+dirty |
| activation_matrix | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 128 | 0.0000 | 0.6146 | real | a44b9bd+dirty |
| belady | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 256 | 0.4564 | 0.4564 | real | a44b9bd+dirty |
| lru | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 256 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| lfu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 256 | 0.1971 | 0.1971 | real | a44b9bd+dirty |
| lfu_k | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 256 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| model_tracing | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 256 | 0.1982 | 0.1982 | real | a44b9bd+dirty |
| lhu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 256 | 0.0128 | 0.0128 | real | a44b9bd+dirty |
| arc | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 256 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| 2q | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 256 | 0.0502 | 0.0502 | real | a44b9bd+dirty |
| lru_k | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 256 | 0.2430 | 0.2430 | real | a44b9bd+dirty |
| lirs | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 256 | 0.2748 | 0.2748 | real | a44b9bd+dirty |
| s3fifo | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 256 | 0.0636 | 0.0636 | real | a44b9bd+dirty |
| sieve | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 256 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 256 | 0.2513 | 0.2513 | real | a44b9bd+dirty |
| layer_lru | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 256 | 0.0741 | 0.0741 | real | a44b9bd+dirty |
| lcnu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 256 | 0.3095 | 0.3095 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 256 | 0.0000 | 0.9955 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 256 | 0.0000 | 1.0000 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 256 | 0.0000 | 0.9082 | real | a44b9bd+dirty |
| activation_matrix | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 256 | 0.0000 | 0.6433 | real | a44b9bd+dirty |
| belady | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 384 | 0.5521 | 0.5521 | real | a44b9bd+dirty |
| lru | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 384 | 0.2680 | 0.2680 | real | a44b9bd+dirty |
| lfu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 384 | 0.2635 | 0.2635 | real | a44b9bd+dirty |
| lfu_k | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 384 | 0.2680 | 0.2680 | real | a44b9bd+dirty |
| model_tracing | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 384 | 0.2674 | 0.2674 | real | a44b9bd+dirty |
| lhu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 384 | 0.3390 | 0.3390 | real | a44b9bd+dirty |
| arc | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 384 | 0.3043 | 0.3043 | real | a44b9bd+dirty |
| 2q | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 384 | 0.2091 | 0.2091 | real | a44b9bd+dirty |
| lru_k | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 384 | 0.3410 | 0.3410 | real | a44b9bd+dirty |
| lirs | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 384 | 0.3494 | 0.3494 | real | a44b9bd+dirty |
| s3fifo | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 384 | 0.3288 | 0.3288 | real | a44b9bd+dirty |
| sieve | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 384 | 0.0795 | 0.0795 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 384 | 0.3150 | 0.3150 | real | a44b9bd+dirty |
| layer_lru | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 384 | 0.2702 | 0.2702 | real | a44b9bd+dirty |
| lcnu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 384 | 0.3870 | 0.3870 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 384 | 0.0086 | 0.9300 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 384 | 0.0524 | 0.7188 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 384 | 0.0002 | 0.9811 | real | a44b9bd+dirty |
| activation_matrix | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 384 | 0.0000 | 0.6433 | real | a44b9bd+dirty |
| belady | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 512 | 0.6154 | 0.6154 | real | a44b9bd+dirty |
| lru | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 512 | 0.3773 | 0.3773 | real | a44b9bd+dirty |
| lfu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 512 | 0.3201 | 0.3201 | real | a44b9bd+dirty |
| lfu_k | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 512 | 0.3773 | 0.3773 | real | a44b9bd+dirty |
| model_tracing | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 512 | 0.3250 | 0.3250 | real | a44b9bd+dirty |
| lhu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 512 | 0.3780 | 0.3780 | real | a44b9bd+dirty |
| arc | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 512 | 0.3927 | 0.3927 | real | a44b9bd+dirty |
| 2q | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 512 | 0.3677 | 0.3677 | real | a44b9bd+dirty |
| lru_k | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 512 | 0.4000 | 0.4000 | real | a44b9bd+dirty |
| lirs | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 512 | 0.4057 | 0.4057 | real | a44b9bd+dirty |
| s3fifo | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 512 | 0.3829 | 0.3829 | real | a44b9bd+dirty |
| sieve | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 512 | 0.1561 | 0.1561 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 512 | 0.3691 | 0.3691 | real | a44b9bd+dirty |
| layer_lru | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 512 | 0.3869 | 0.3869 | real | a44b9bd+dirty |
| lcnu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 512 | 0.4471 | 0.4471 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 512 | 0.3766 | 0.9996 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 512 | 0.3771 | 0.9998 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 512 | 0.3764 | 0.9998 | real | a44b9bd+dirty |
| activation_matrix | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 512 | 0.0000 | 0.6433 | real | a44b9bd+dirty |
| belady | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 768 | 0.7014 | 0.7014 | real | a44b9bd+dirty |
| lru | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 768 | 0.4897 | 0.4897 | real | a44b9bd+dirty |
| lfu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 768 | 0.4183 | 0.4183 | real | a44b9bd+dirty |
| lfu_k | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 768 | 0.4897 | 0.4897 | real | a44b9bd+dirty |
| model_tracing | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 768 | 0.4267 | 0.4267 | real | a44b9bd+dirty |
| lhu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 768 | 0.4911 | 0.4911 | real | a44b9bd+dirty |
| arc | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 768 | 0.5027 | 0.5027 | real | a44b9bd+dirty |
| 2q | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 768 | 0.4875 | 0.4875 | real | a44b9bd+dirty |
| lru_k | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 768 | 0.4918 | 0.4918 | real | a44b9bd+dirty |
| lirs | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 768 | 0.4918 | 0.4918 | real | a44b9bd+dirty |
| s3fifo | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 768 | 0.4769 | 0.4769 | real | a44b9bd+dirty |
| sieve | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 768 | 0.2878 | 0.2878 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 768 | 0.4607 | 0.4607 | real | a44b9bd+dirty |
| layer_lru | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 768 | 0.4981 | 0.4981 | real | a44b9bd+dirty |
| lcnu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 768 | 0.5386 | 0.5386 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 768 | 0.4878 | 0.9988 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 768 | 0.4889 | 0.9991 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 768 | 0.4865 | 0.9989 | real | a44b9bd+dirty |
| activation_matrix | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 768 | 0.3675 | 0.7342 | real | a44b9bd+dirty |
| belady | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1024 | 0.7623 | 0.7623 | real | a44b9bd+dirty |
| lru | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1024 | 0.5636 | 0.5636 | real | a44b9bd+dirty |
| lfu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1024 | 0.5035 | 0.5035 | real | a44b9bd+dirty |
| lfu_k | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1024 | 0.5636 | 0.5636 | real | a44b9bd+dirty |
| model_tracing | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1024 | 0.5132 | 0.5132 | real | a44b9bd+dirty |
| lhu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1024 | 0.5638 | 0.5638 | real | a44b9bd+dirty |
| arc | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1024 | 0.5763 | 0.5763 | real | a44b9bd+dirty |
| 2q | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1024 | 0.5676 | 0.5676 | real | a44b9bd+dirty |
| lru_k | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1024 | 0.5670 | 0.5670 | real | a44b9bd+dirty |
| lirs | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1024 | 0.5654 | 0.5654 | real | a44b9bd+dirty |
| s3fifo | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1024 | 0.5647 | 0.5647 | real | a44b9bd+dirty |
| sieve | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1024 | 0.4075 | 0.4075 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1024 | 0.5411 | 0.5411 | real | a44b9bd+dirty |
| layer_lru | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1024 | 0.5688 | 0.5688 | real | a44b9bd+dirty |
| lcnu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1024 | 0.6052 | 0.6052 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1024 | 0.5618 | 0.9989 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1024 | 0.5631 | 0.9994 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1024 | 0.5607 | 0.9989 | real | a44b9bd+dirty |
| activation_matrix | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1024 | 0.4571 | 0.7643 | real | a44b9bd+dirty |
| belady | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1536 | 0.8461 | 0.8461 | real | a44b9bd+dirty |
| lru | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1536 | 0.6706 | 0.6706 | real | a44b9bd+dirty |
| lfu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1536 | 0.6455 | 0.6455 | real | a44b9bd+dirty |
| lfu_k | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1536 | 0.6706 | 0.6706 | real | a44b9bd+dirty |
| model_tracing | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1536 | 0.6548 | 0.6548 | real | a44b9bd+dirty |
| lhu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1536 | 0.6748 | 0.6748 | real | a44b9bd+dirty |
| arc | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1536 | 0.6854 | 0.6854 | real | a44b9bd+dirty |
| 2q | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1536 | 0.6868 | 0.6868 | real | a44b9bd+dirty |
| lru_k | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1536 | 0.6934 | 0.6934 | real | a44b9bd+dirty |
| lirs | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1536 | 0.6910 | 0.6910 | real | a44b9bd+dirty |
| s3fifo | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1536 | 0.6966 | 0.6966 | real | a44b9bd+dirty |
| sieve | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1536 | 0.5072 | 0.5072 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1536 | 0.6797 | 0.6797 | real | a44b9bd+dirty |
| layer_lru | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1536 | 0.6726 | 0.6726 | real | a44b9bd+dirty |
| lcnu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1536 | 0.7127 | 0.7127 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1536 | 0.6686 | 0.9988 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1536 | 0.6702 | 0.9996 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1536 | 0.6674 | 0.9988 | real | a44b9bd+dirty |
| activation_matrix | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 1536 | 0.5955 | 0.8150 | real | a44b9bd+dirty |
| belady | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 2048 | 0.9009 | 0.9009 | real | a44b9bd+dirty |
| lru | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 2048 | 0.7593 | 0.7593 | real | a44b9bd+dirty |
| lfu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 2048 | 0.7573 | 0.7573 | real | a44b9bd+dirty |
| lfu_k | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 2048 | 0.7593 | 0.7593 | real | a44b9bd+dirty |
| model_tracing | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 2048 | 0.7661 | 0.7661 | real | a44b9bd+dirty |
| lhu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 2048 | 0.7684 | 0.7684 | real | a44b9bd+dirty |
| arc | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 2048 | 0.7665 | 0.7665 | real | a44b9bd+dirty |
| 2q | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 2048 | 0.7689 | 0.7689 | real | a44b9bd+dirty |
| lru_k | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 2048 | 0.7929 | 0.7929 | real | a44b9bd+dirty |
| lirs | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 2048 | 0.7880 | 0.7880 | real | a44b9bd+dirty |
| s3fifo | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 2048 | 0.7932 | 0.7932 | real | a44b9bd+dirty |
| sieve | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 2048 | 0.6329 | 0.6329 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 2048 | 0.7884 | 0.7884 | real | a44b9bd+dirty |
| layer_lru | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 2048 | 0.7561 | 0.7561 | real | a44b9bd+dirty |
| lcnu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 2048 | 0.8001 | 0.8001 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 2048 | 0.7572 | 0.9987 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 2048 | 0.7590 | 0.9997 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 2048 | 0.7559 | 0.9987 | real | a44b9bd+dirty |
| activation_matrix | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 2048 | 0.7089 | 0.8559 | real | a44b9bd+dirty |
| belady | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 3072 | 0.9650 | 0.9650 | real | a44b9bd+dirty |
| lru | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 3072 | 0.8850 | 0.8850 | real | a44b9bd+dirty |
| lfu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 3072 | 0.9029 | 0.9029 | real | a44b9bd+dirty |
| lfu_k | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 3072 | 0.8850 | 0.8850 | real | a44b9bd+dirty |
| model_tracing | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 3072 | 0.9091 | 0.9091 | real | a44b9bd+dirty |
| lhu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 3072 | 0.9015 | 0.9015 | real | a44b9bd+dirty |
| arc | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 3072 | 0.8853 | 0.8853 | real | a44b9bd+dirty |
| 2q | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 3072 | 0.8955 | 0.8955 | real | a44b9bd+dirty |
| lru_k | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 3072 | 0.9259 | 0.9259 | real | a44b9bd+dirty |
| lirs | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 3072 | 0.9251 | 0.9251 | real | a44b9bd+dirty |
| s3fifo | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 3072 | 0.9202 | 0.9202 | real | a44b9bd+dirty |
| sieve | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 3072 | 0.8174 | 0.8174 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 3072 | 0.9298 | 0.9298 | real | a44b9bd+dirty |
| layer_lru | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 3072 | 0.8794 | 0.8794 | real | a44b9bd+dirty |
| lcnu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 3072 | 0.9108 | 0.9108 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 3072 | 0.8821 | 0.9982 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 3072 | 0.8848 | 0.9998 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 3072 | 0.8804 | 0.9983 | real | a44b9bd+dirty |
| activation_matrix | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 3072 | 0.8663 | 0.9201 | real | a44b9bd+dirty |
| belady | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 4096 | 0.9928 | 0.9928 | real | a44b9bd+dirty |
| lru | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 4096 | 0.9797 | 0.9797 | real | a44b9bd+dirty |
| lfu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 4096 | 0.9730 | 0.9730 | real | a44b9bd+dirty |
| lfu_k | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 4096 | 0.9797 | 0.9797 | real | a44b9bd+dirty |
| model_tracing | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 4096 | 0.9764 | 0.9764 | real | a44b9bd+dirty |
| lhu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 4096 | 0.9814 | 0.9814 | real | a44b9bd+dirty |
| arc | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 4096 | 0.9802 | 0.9802 | real | a44b9bd+dirty |
| 2q | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 4096 | 0.9733 | 0.9733 | real | a44b9bd+dirty |
| lru_k | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 4096 | 0.9833 | 0.9833 | real | a44b9bd+dirty |
| lirs | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 4096 | 0.9841 | 0.9841 | real | a44b9bd+dirty |
| s3fifo | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 4096 | 0.9816 | 0.9816 | real | a44b9bd+dirty |
| sieve | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 4096 | 0.9747 | 0.9747 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 4096 | 0.9845 | 0.9845 | real | a44b9bd+dirty |
| layer_lru | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 4096 | 0.9633 | 0.9633 | real | a44b9bd+dirty |
| lcnu | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 4096 | 0.9825 | 0.9825 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 4096 | 0.9792 | 0.9997 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 4096 | 0.9796 | 0.9999 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 4096 | 0.9789 | 0.9997 | real | a44b9bd+dirty |
| activation_matrix | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union | 4096 | 0.9752 | 0.9786 | real | a44b9bd+dirty |

<!-- run: python -m benchmarks.cache_simulator.research.run_policies_real --model qwen3-30b-a3b --dataset mmlu150-greedy64 --ignore-layers 0 --prefill union | env: python 3.12.3, numpy 2.4.4, Linux x86_64 | commit: a44b9bd+dirty -->

| policy | model | dataset | capacity | hit_rate | eff_hit_rate | provenance | commit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| belady | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 256 | 0.2781 | 0.2781 | real | a44b9bd+dirty |
| lru | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 256 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| lfu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 256 | 0.1435 | 0.1435 | real | a44b9bd+dirty |
| lfu_k | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 256 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| model_tracing | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 256 | 0.1441 | 0.1441 | real | a44b9bd+dirty |
| lhu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 256 | 0.0053 | 0.0053 | real | a44b9bd+dirty |
| arc | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 256 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| 2q | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 256 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| lru_k | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 256 | 0.0344 | 0.0344 | real | a44b9bd+dirty |
| lirs | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 256 | 0.1571 | 0.1571 | real | a44b9bd+dirty |
| s3fifo | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 256 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| sieve | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 256 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 256 | 0.1520 | 0.1520 | real | a44b9bd+dirty |
| layer_lru | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 256 | 0.0152 | 0.0152 | real | a44b9bd+dirty |
| lcnu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 256 | 0.1860 | 0.1860 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 256 | 0.0000 | 0.9958 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 256 | 0.0000 | 1.0000 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 256 | 0.0000 | 0.8998 | real | a44b9bd+dirty |
| activation_matrix | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 256 | 0.0000 | 0.6109 | real | a44b9bd+dirty |
| belady | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 512 | 0.4480 | 0.4480 | real | a44b9bd+dirty |
| lru | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 512 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| lfu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 512 | 0.2344 | 0.2344 | real | a44b9bd+dirty |
| lfu_k | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 512 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| model_tracing | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 512 | 0.2353 | 0.2353 | real | a44b9bd+dirty |
| lhu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 512 | 0.0190 | 0.0190 | real | a44b9bd+dirty |
| arc | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 512 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| 2q | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 512 | 0.1051 | 0.1051 | real | a44b9bd+dirty |
| lru_k | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 512 | 0.2265 | 0.2265 | real | a44b9bd+dirty |
| lirs | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 512 | 0.2579 | 0.2579 | real | a44b9bd+dirty |
| s3fifo | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 512 | 0.0999 | 0.0999 | real | a44b9bd+dirty |
| sieve | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 512 | 0.0000 | 0.0000 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 512 | 0.2498 | 0.2498 | real | a44b9bd+dirty |
| layer_lru | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 512 | 0.0771 | 0.0771 | real | a44b9bd+dirty |
| lcnu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 512 | 0.2976 | 0.2976 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 512 | 0.0000 | 1.0000 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 512 | 0.0000 | 1.0000 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 512 | 0.0000 | 1.0000 | real | a44b9bd+dirty |
| activation_matrix | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 512 | 0.0000 | 0.6109 | real | a44b9bd+dirty |
| belady | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 768 | 0.5427 | 0.5427 | real | a44b9bd+dirty |
| lru | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 768 | 0.3375 | 0.3375 | real | a44b9bd+dirty |
| lfu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 768 | 0.3054 | 0.3054 | real | a44b9bd+dirty |
| lfu_k | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 768 | 0.3375 | 0.3375 | real | a44b9bd+dirty |
| model_tracing | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 768 | 0.3063 | 0.3063 | real | a44b9bd+dirty |
| lhu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 768 | 0.3307 | 0.3307 | real | a44b9bd+dirty |
| arc | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 768 | 0.2985 | 0.2985 | real | a44b9bd+dirty |
| 2q | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 768 | 0.1935 | 0.1935 | real | a44b9bd+dirty |
| lru_k | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 768 | 0.3264 | 0.3264 | real | a44b9bd+dirty |
| lirs | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 768 | 0.3363 | 0.3363 | real | a44b9bd+dirty |
| s3fifo | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 768 | 0.3162 | 0.3162 | real | a44b9bd+dirty |
| sieve | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 768 | 0.1340 | 0.1340 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 768 | 0.3137 | 0.3137 | real | a44b9bd+dirty |
| layer_lru | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 768 | 0.2469 | 0.2469 | real | a44b9bd+dirty |
| lcnu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 768 | 0.3757 | 0.3757 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 768 | 0.3374 | 0.9999 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 768 | 0.3375 | 1.0000 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 768 | 0.3185 | 0.9810 | real | a44b9bd+dirty |
| activation_matrix | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 768 | 0.0000 | 0.6109 | real | a44b9bd+dirty |
| belady | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1024 | 0.6079 | 0.6079 | real | a44b9bd+dirty |
| lru | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1024 | 0.3392 | 0.3392 | real | a44b9bd+dirty |
| lfu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1024 | 0.3644 | 0.3644 | real | a44b9bd+dirty |
| lfu_k | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1024 | 0.3392 | 0.3392 | real | a44b9bd+dirty |
| model_tracing | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1024 | 0.3657 | 0.3657 | real | a44b9bd+dirty |
| lhu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1024 | 0.3423 | 0.3423 | real | a44b9bd+dirty |
| arc | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1024 | 0.3729 | 0.3729 | real | a44b9bd+dirty |
| 2q | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1024 | 0.3458 | 0.3458 | real | a44b9bd+dirty |
| lru_k | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1024 | 0.3891 | 0.3891 | real | a44b9bd+dirty |
| lirs | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1024 | 0.3965 | 0.3965 | real | a44b9bd+dirty |
| s3fifo | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1024 | 0.3733 | 0.3733 | real | a44b9bd+dirty |
| sieve | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1024 | 0.2159 | 0.2159 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1024 | 0.3662 | 0.3662 | real | a44b9bd+dirty |
| layer_lru | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1024 | 0.3515 | 0.3515 | real | a44b9bd+dirty |
| lcnu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1024 | 0.4375 | 0.4375 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1024 | 0.3389 | 0.9998 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1024 | 0.3391 | 0.9999 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1024 | 0.3387 | 0.9998 | real | a44b9bd+dirty |
| activation_matrix | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1024 | 0.0000 | 0.6109 | real | a44b9bd+dirty |
| belady | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1536 | 0.6981 | 0.6981 | real | a44b9bd+dirty |
| lru | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1536 | 0.4646 | 0.4646 | real | a44b9bd+dirty |
| lfu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1536 | 0.4630 | 0.4630 | real | a44b9bd+dirty |
| lfu_k | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1536 | 0.4646 | 0.4646 | real | a44b9bd+dirty |
| model_tracing | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1536 | 0.4646 | 0.4646 | real | a44b9bd+dirty |
| lhu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1536 | 0.4675 | 0.4675 | real | a44b9bd+dirty |
| arc | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1536 | 0.4849 | 0.4849 | real | a44b9bd+dirty |
| 2q | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1536 | 0.4764 | 0.4764 | real | a44b9bd+dirty |
| lru_k | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1536 | 0.4901 | 0.4901 | real | a44b9bd+dirty |
| lirs | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1536 | 0.4912 | 0.4912 | real | a44b9bd+dirty |
| s3fifo | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1536 | 0.4692 | 0.4692 | real | a44b9bd+dirty |
| sieve | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1536 | 0.3656 | 0.3656 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1536 | 0.4551 | 0.4551 | real | a44b9bd+dirty |
| layer_lru | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1536 | 0.4696 | 0.4696 | real | a44b9bd+dirty |
| lcnu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1536 | 0.5317 | 0.5317 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1536 | 0.4636 | 0.9994 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1536 | 0.4642 | 0.9995 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1536 | 0.4630 | 0.9994 | real | a44b9bd+dirty |
| activation_matrix | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 1536 | 0.3386 | 0.7071 | real | a44b9bd+dirty |
| belady | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 2048 | 0.7600 | 0.7600 | real | a44b9bd+dirty |
| lru | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 2048 | 0.5479 | 0.5479 | real | a44b9bd+dirty |
| lfu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 2048 | 0.5452 | 0.5452 | real | a44b9bd+dirty |
| lfu_k | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 2048 | 0.5479 | 0.5479 | real | a44b9bd+dirty |
| model_tracing | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 2048 | 0.5471 | 0.5471 | real | a44b9bd+dirty |
| lhu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 2048 | 0.5500 | 0.5500 | real | a44b9bd+dirty |
| arc | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 2048 | 0.5674 | 0.5674 | real | a44b9bd+dirty |
| 2q | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 2048 | 0.5604 | 0.5604 | real | a44b9bd+dirty |
| lru_k | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 2048 | 0.5688 | 0.5688 | real | a44b9bd+dirty |
| lirs | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 2048 | 0.5681 | 0.5681 | real | a44b9bd+dirty |
| s3fifo | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 2048 | 0.5620 | 0.5620 | real | a44b9bd+dirty |
| sieve | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 2048 | 0.4629 | 0.4629 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 2048 | 0.5320 | 0.5320 | real | a44b9bd+dirty |
| layer_lru | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 2048 | 0.5517 | 0.5517 | real | a44b9bd+dirty |
| lcnu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 2048 | 0.6017 | 0.6017 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 2048 | 0.5469 | 0.9994 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 2048 | 0.5476 | 0.9996 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 2048 | 0.5463 | 0.9994 | real | a44b9bd+dirty |
| activation_matrix | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 2048 | 0.4458 | 0.7441 | real | a44b9bd+dirty |
| belady | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 3072 | 0.8439 | 0.8439 | real | a44b9bd+dirty |
| lru | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 3072 | 0.6686 | 0.6686 | real | a44b9bd+dirty |
| lfu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 3072 | 0.6797 | 0.6797 | real | a44b9bd+dirty |
| lfu_k | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 3072 | 0.6686 | 0.6686 | real | a44b9bd+dirty |
| model_tracing | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 3072 | 0.6820 | 0.6820 | real | a44b9bd+dirty |
| lhu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 3072 | 0.6724 | 0.6724 | real | a44b9bd+dirty |
| arc | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 3072 | 0.6848 | 0.6848 | real | a44b9bd+dirty |
| 2q | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 3072 | 0.6799 | 0.6799 | real | a44b9bd+dirty |
| lru_k | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 3072 | 0.6917 | 0.6917 | real | a44b9bd+dirty |
| lirs | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 3072 | 0.6889 | 0.6889 | real | a44b9bd+dirty |
| s3fifo | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 3072 | 0.6931 | 0.6931 | real | a44b9bd+dirty |
| sieve | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 3072 | 0.5964 | 0.5964 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 3072 | 0.6620 | 0.6620 | real | a44b9bd+dirty |
| layer_lru | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 3072 | 0.6673 | 0.6673 | real | a44b9bd+dirty |
| lcnu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 3072 | 0.7105 | 0.7105 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 3072 | 0.6676 | 0.9994 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 3072 | 0.6684 | 0.9998 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 3072 | 0.6670 | 0.9994 | real | a44b9bd+dirty |
| activation_matrix | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 3072 | 0.5992 | 0.8023 | real | a44b9bd+dirty |
| belady | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 4096 | 0.8985 | 0.8985 | real | a44b9bd+dirty |
| lru | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 4096 | 0.7557 | 0.7557 | real | a44b9bd+dirty |
| lfu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 4096 | 0.7835 | 0.7835 | real | a44b9bd+dirty |
| lfu_k | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 4096 | 0.7557 | 0.7557 | real | a44b9bd+dirty |
| model_tracing | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 4096 | 0.7857 | 0.7857 | real | a44b9bd+dirty |
| lhu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 4096 | 0.7642 | 0.7642 | real | a44b9bd+dirty |
| arc | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 4096 | 0.7647 | 0.7647 | real | a44b9bd+dirty |
| 2q | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 4096 | 0.7638 | 0.7638 | real | a44b9bd+dirty |
| lru_k | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 4096 | 0.7849 | 0.7849 | real | a44b9bd+dirty |
| lirs | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 4096 | 0.7817 | 0.7817 | real | a44b9bd+dirty |
| s3fifo | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 4096 | 0.7845 | 0.7845 | real | a44b9bd+dirty |
| sieve | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 4096 | 0.6944 | 0.6944 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 4096 | 0.7655 | 0.7655 | real | a44b9bd+dirty |
| layer_lru | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 4096 | 0.7527 | 0.7527 | real | a44b9bd+dirty |
| lcnu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 4096 | 0.7944 | 0.7944 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 4096 | 0.7546 | 0.9993 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 4096 | 0.7555 | 0.9998 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 4096 | 0.7540 | 0.9994 | real | a44b9bd+dirty |
| activation_matrix | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 4096 | 0.7125 | 0.8494 | real | a44b9bd+dirty |
| belady | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 6144 | 0.9626 | 0.9626 | real | a44b9bd+dirty |
| lru | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 6144 | 0.8828 | 0.8828 | real | a44b9bd+dirty |
| lfu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 6144 | 0.9195 | 0.9195 | real | a44b9bd+dirty |
| lfu_k | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 6144 | 0.8828 | 0.8828 | real | a44b9bd+dirty |
| model_tracing | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 6144 | 0.9212 | 0.9212 | real | a44b9bd+dirty |
| lhu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 6144 | 0.8997 | 0.8997 | real | a44b9bd+dirty |
| arc | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 6144 | 0.8833 | 0.8833 | real | a44b9bd+dirty |
| 2q | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 6144 | 0.8912 | 0.8912 | real | a44b9bd+dirty |
| lru_k | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 6144 | 0.9161 | 0.9161 | real | a44b9bd+dirty |
| lirs | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 6144 | 0.9160 | 0.9160 | real | a44b9bd+dirty |
| s3fifo | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 6144 | 0.9133 | 0.9133 | real | a44b9bd+dirty |
| sieve | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 6144 | 0.8714 | 0.8714 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 6144 | 0.9128 | 0.9128 | real | a44b9bd+dirty |
| layer_lru | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 6144 | 0.8793 | 0.8793 | real | a44b9bd+dirty |
| lcnu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 6144 | 0.9088 | 0.9088 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 6144 | 0.8815 | 0.9991 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 6144 | 0.8827 | 0.9999 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 6144 | 0.8806 | 0.9992 | real | a44b9bd+dirty |
| activation_matrix | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 6144 | 0.8699 | 0.9216 | real | a44b9bd+dirty |
| belady | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 8192 | 0.9918 | 0.9918 | real | a44b9bd+dirty |
| lru | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 8192 | 0.9768 | 0.9768 | real | a44b9bd+dirty |
| lfu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 8192 | 0.9834 | 0.9834 | real | a44b9bd+dirty |
| lfu_k | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 8192 | 0.9768 | 0.9768 | real | a44b9bd+dirty |
| model_tracing | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 8192 | 0.9843 | 0.9843 | real | a44b9bd+dirty |
| lhu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 8192 | 0.9835 | 0.9835 | real | a44b9bd+dirty |
| arc | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 8192 | 0.9774 | 0.9774 | real | a44b9bd+dirty |
| 2q | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 8192 | 0.9695 | 0.9695 | real | a44b9bd+dirty |
| lru_k | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 8192 | 0.9805 | 0.9805 | real | a44b9bd+dirty |
| lirs | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 8192 | 0.9817 | 0.9817 | real | a44b9bd+dirty |
| s3fifo | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 8192 | 0.9790 | 0.9790 | real | a44b9bd+dirty |
| sieve | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 8192 | 0.9801 | 0.9801 | real | a44b9bd+dirty |
| w_tinylfu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 8192 | 0.9829 | 0.9829 | real | a44b9bd+dirty |
| layer_lru | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 8192 | 0.9674 | 0.9674 | real | a44b9bd+dirty |
| lcnu | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 8192 | 0.9834 | 0.9834 | real | a44b9bd+dirty |
| single_layer_ahead | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 8192 | 0.9766 | 0.9998 | real | a44b9bd+dirty |
| stride_prefetch | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 8192 | 0.9768 | 1.0000 | real | a44b9bd+dirty |
| multi_layer_ahead | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 8192 | 0.9765 | 0.9999 | real | a44b9bd+dirty |
| activation_matrix | qwen3-235b | mmlu100-greedy64/decode+prefill-union | 8192 | 0.9750 | 0.9788 | real | a44b9bd+dirty |

<!-- run: python -m benchmarks.cache_simulator.research.run_policies_real --model qwen3-235b --dataset mmlu100-greedy64 --ignore-layers 0 --prefill union | env: python 3.12.3, numpy 2.4.4, Linux x86_64 | commit: a44b9bd+dirty -->

## Round 2 — prefetch x eviction composition (#289)

Provenance `real` throughout: SGLang 0.5.20 `per_token` captures, decode-only
(`/decode`) or prefill-union (`/decode+prefill-union`) streams, request files
split in capture order into a tuning half and a held-out half (`/tune`,
`/held`). Stall = timing-model stall per decode token (ms), nominal scenario
(measured PCIe 5.0 link 57.1 GB/s, measured per-layer compute, demand
priority). Study: `prefetch_eviction_study.md`.

Setup runs (commands in the study's Reproduce section):

| run | model | dataset | result | provenance | commit |
| --- | --- | --- | --- | --- | --- |
| composition identity (`--mode identity`, 7 evictors x {256, 1024}) | qwen3-30b-a3b | mmlu150-greedy64/decode/all | all 14 hit counts bit-identical (`results_prefetch_qwen3_30b_identity.md`) | real | db1ef24 |
| predictor stage 0 (`--mode predict`) | qwen3-30b-a3b | mmlu150-greedy64/decode/tune | topk_ema alpha=0.4, hybrid weights=1/1/16 (`results_prefetch_qwen3_30b_tune_predict.csv`) | real | db1ef24 |
| tuning grid (`--mode tune`, 3,972 runs, LRU + LCNU) | qwen3-30b-a3b | mmlu150-greedy64/decode/tune | `tuned_prefetch_qwen3_30b.json`, `results_prefetch_qwen3_30b_tune_grid.csv` | real | 1a64e8c |
| predictor accuracy (`--mode predict`) | qwen3-30b-a3b | mmlu150-greedy64/decode/held | `results_prefetch_qwen3_30b_held_predict.csv` | real | db1ef24 |
| predictor accuracy (`--mode predict`) | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | `results_prefetch_qwen3_235b_held_predict.csv` | real | db1ef24 |

Held-out sweeps below: compact rows (per capacity, every evictor alone, the
best online pair per admission mode, the best oracle pair); full rows in
`results_prefetch_<tag>.csv`.

| evictor | prefetcher | admission | model | dataset | capacity | hit_rate | covered | stall_ms_tok | provenance | commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lru | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 128 | 0.0000 | 0.0000 | 63.47 | real | db1ef24 |
| lfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 128 | 0.1303 | 0.0000 | 55.20 | real | db1ef24 |
| lirs | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 128 | 0.1859 | 0.0000 | 51.67 | real | db1ef24 |
| s3fifo | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 128 | 0.0000 | 0.0000 | 63.47 | real | db1ef24 |
| w_tinylfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 128 | 0.1721 | 0.0000 | 52.54 | real | db1ef24 |
| layer_lru | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 128 | 0.0157 | 0.0000 | 62.47 | real | db1ef24 |
| lcnu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 128 | 0.2179 | 0.0000 | 49.64 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=2,horizon=1,weights=1/1/16 | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 128 | 0.2175 | 0.0351 | 48.14 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | probation | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 128 | 0.2148 | 0.1951 | 45.62 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | pin | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 128 | 0.2148 | 0.1951 | 45.62 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | buffer | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 128 | 0.2126 | 0.1965 | 45.75 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | do_no_harm | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 128 | 0.2155 | 0.1656 | 45.65 | real | db1ef24 |
| lcnu | oracle:horizon=1 | probation | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 128 | 0.2095 | 0.7905 | 41.09 | real | db1ef24 |
| lru | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 256 | 0.0000 | 0.0000 | 63.47 | real | db1ef24 |
| lfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 256 | 0.2251 | 0.0000 | 49.18 | real | db1ef24 |
| lirs | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 256 | 0.3066 | 0.0000 | 44.01 | real | db1ef24 |
| s3fifo | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 256 | 0.0469 | 0.0000 | 60.49 | real | db1ef24 |
| w_tinylfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 256 | 0.2760 | 0.0000 | 45.95 | real | db1ef24 |
| layer_lru | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 256 | 0.0840 | 0.0000 | 58.13 | real | db1ef24 |
| lcnu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 256 | 0.3461 | 0.0000 | 41.50 | real | db1ef24 |
| lcnu | coactivation:budget=2,horizon=1 | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 256 | 0.3459 | 0.0208 | 40.57 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | probation | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 256 | 0.3459 | 0.1263 | 38.22 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | pin | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 256 | 0.3459 | 0.1263 | 38.22 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | buffer | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 256 | 0.3429 | 0.1274 | 38.40 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | do_no_harm | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 256 | 0.3460 | 0.1069 | 38.31 | real | db1ef24 |
| lcnu | oracle:horizon=1 | probation | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 256 | 0.3432 | 0.6568 | 32.63 | real | db1ef24 |
| lru | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 512 | 0.4398 | 0.0000 | 35.56 | real | db1ef24 |
| lfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 512 | 0.3577 | 0.0000 | 40.76 | real | db1ef24 |
| lirs | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 512 | 0.4481 | 0.0000 | 35.03 | real | db1ef24 |
| s3fifo | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 512 | 0.4245 | 0.0000 | 36.52 | real | db1ef24 |
| w_tinylfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 512 | 0.4042 | 0.0000 | 37.81 | real | db1ef24 |
| layer_lru | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 512 | 0.4472 | 0.0000 | 35.09 | real | db1ef24 |
| lcnu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 512 | 0.4976 | 0.0000 | 31.89 | real | db1ef24 |
| lcnu | coactivation:budget=4,horizon=1 | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 512 | 0.4977 | 0.0269 | 30.94 | real | db1ef24 |
| lcnu | coactivation:budget=4,horizon=1 | probation | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 512 | 0.4986 | 0.0877 | 29.34 | real | db1ef24 |
| lcnu | coactivation:budget=4,horizon=1 | pin | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 512 | 0.4986 | 0.0877 | 29.34 | real | db1ef24 |
| lcnu | coactivation:budget=4,horizon=1 | buffer | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 512 | 0.4957 | 0.0881 | 29.52 | real | db1ef24 |
| lcnu | coactivation:budget=4,horizon=1 | do_no_harm | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 512 | 0.4986 | 0.0732 | 29.48 | real | db1ef24 |
| lcnu | oracle:horizon=2 | probation | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 512 | 0.4965 | 0.5035 | 22.90 | real | db1ef24 |
| lru | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 1024 | 0.6432 | 0.0000 | 22.65 | real | db1ef24 |
| lfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 1024 | 0.5488 | 0.0000 | 28.64 | real | db1ef24 |
| lirs | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 1024 | 0.6138 | 0.0000 | 24.51 | real | db1ef24 |
| s3fifo | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 1024 | 0.6222 | 0.0000 | 23.98 | real | db1ef24 |
| w_tinylfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 1024 | 0.5875 | 0.0000 | 26.18 | real | db1ef24 |
| layer_lru | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 1024 | 0.6426 | 0.0000 | 22.68 | real | db1ef24 |
| lcnu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 1024 | 0.6697 | 0.0000 | 20.96 | real | db1ef24 |
| lcnu | coactivation:budget=4,horizon=1 | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 1024 | 0.6695 | 0.0161 | 20.13 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=8,horizon=1,weights=1/1/16 | probation | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 1024 | 0.6716 | 0.0972 | 18.52 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=8,horizon=1,weights=1/1/16 | pin | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 1024 | 0.6716 | 0.0972 | 18.52 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=8,horizon=1,weights=1/1/16 | buffer | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 1024 | 0.6677 | 0.0978 | 18.77 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=8,horizon=1,weights=1/1/16 | do_no_harm | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 1024 | 0.6712 | 0.0716 | 18.78 | real | db1ef24 |
| lcnu | oracle:horizon=4 | probation | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 1024 | 0.6705 | 0.3295 | 11.98 | real | db1ef24 |
| lru | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 2048 | 0.8490 | 0.0000 | 9.58 | real | db1ef24 |
| lfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 2048 | 0.8025 | 0.0000 | 12.53 | real | db1ef24 |
| lirs | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 2048 | 0.8347 | 0.0000 | 10.49 | real | db1ef24 |
| s3fifo | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 2048 | 0.8609 | 0.0000 | 8.83 | real | db1ef24 |
| w_tinylfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 2048 | 0.8377 | 0.0000 | 10.30 | real | db1ef24 |
| layer_lru | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 2048 | 0.8387 | 0.0000 | 10.24 | real | db1ef24 |
| lcnu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 2048 | 0.8662 | 0.0000 | 8.49 | real | db1ef24 |
| lcnu | coactivation:budget=8,horizon=1 | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 2048 | 0.8658 | 0.0101 | 7.98 | real | db1ef24 |
| lcnu | coactivation:budget=8,horizon=1 | probation | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 2048 | 0.8664 | 0.0173 | 7.79 | real | db1ef24 |
| lcnu | coactivation:budget=8,horizon=1 | pin | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 2048 | 0.8664 | 0.0173 | 7.79 | real | db1ef24 |
| lcnu | coactivation:budget=8,horizon=1 | buffer | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 2048 | 0.8651 | 0.0177 | 7.87 | real | db1ef24 |
| lcnu | coactivation:budget=8,horizon=1 | do_no_harm | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 2048 | 0.8664 | 0.0149 | 7.84 | real | db1ef24 |
| lcnu | oracle:horizon=4 | pin | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 2048 | 0.8664 | 0.1336 | 1.92 | real | db1ef24 |
| lru | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 3072 | 0.9506 | 0.0000 | 3.14 | real | db1ef24 |
| lfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 3072 | 0.9314 | 0.0000 | 4.36 | real | db1ef24 |
| lirs | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 3072 | 0.9551 | 0.0000 | 2.85 | real | db1ef24 |
| s3fifo | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 3072 | 0.9590 | 0.0000 | 2.60 | real | db1ef24 |
| w_tinylfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 3072 | 0.9536 | 0.0000 | 2.94 | real | db1ef24 |
| layer_lru | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 3072 | 0.9401 | 0.0000 | 3.80 | real | db1ef24 |
| lcnu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 3072 | 0.9558 | 0.0000 | 2.80 | real | db1ef24 |
| s3fifo | hybrid:alpha=0.4,budget=16,horizon=4,weights=1/1/16 | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 3072 | 0.9507 | 0.0133 | 2.57 | real | db1ef24 |
| s3fifo | hybrid:alpha=0.4,budget=16,horizon=1,weights=1/1/16 | probation | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 3072 | 0.9589 | 0.0057 | 2.58 | real | db1ef24 |
| lcnu | coactivation:budget=16,horizon=2 | pin | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 3072 | 0.9558 | 0.0065 | 2.53 | real | db1ef24 |
| s3fifo | prev_token:budget=2,horizon=1 | buffer | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 3072 | 0.9589 | 0.0000 | 2.61 | real | db1ef24 |
| s3fifo | topk_ema:alpha=0.4,budget=32,horizon=1 | do_no_harm | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 3072 | 0.9590 | 0.0002 | 2.59 | real | db1ef24 |
| s3fifo | oracle:horizon=4 | probation | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 3072 | 0.9589 | 0.0410 | 0.15 | real | db1ef24 |

<!-- run: python -m benchmarks.cache_simulator.research.run_prefetch_real --model qwen3-30b-a3b --split held --prefill skip --tag qwen3_30b_held | env: python 3.12.3, numpy 2.4.4, Linux x86_64 | commit: db1ef24 -->

| evictor | prefetcher | admission | model | dataset | capacity | hit_rate | covered | stall_ms_tok | provenance | commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lru | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 256 | 0.0000 | 0.0000 | 248.57 | real | db1ef24 |
| lfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 256 | 0.1586 | 0.0000 | 209.15 | real | db1ef24 |
| lirs | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 256 | 0.1749 | 0.0000 | 205.10 | real | db1ef24 |
| s3fifo | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 256 | 0.0000 | 0.0000 | 248.57 | real | db1ef24 |
| w_tinylfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 256 | 0.1704 | 0.0000 | 206.21 | real | db1ef24 |
| layer_lru | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 256 | 0.0171 | 0.0000 | 244.31 | real | db1ef24 |
| lcnu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 256 | 0.2064 | 0.0000 | 197.26 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=2,horizon=1,weights=1/1/16 | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 256 | 0.2062 | 0.0325 | 196.01 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | probation | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 256 | 0.2052 | 0.1709 | 188.09 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | pin | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 256 | 0.2052 | 0.1709 | 188.09 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | buffer | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 256 | 0.2040 | 0.1714 | 188.35 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | do_no_harm | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 256 | 0.2056 | 0.1439 | 188.19 | real | db1ef24 |
| lcnu | oracle:horizon=1 | probation | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 256 | 0.2028 | 0.7972 | 177.35 | real | db1ef24 |
| lru | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 512 | 0.0000 | 0.0000 | 248.57 | real | db1ef24 |
| lfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 512 | 0.2579 | 0.0000 | 184.46 | real | db1ef24 |
| lirs | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 512 | 0.2859 | 0.0000 | 177.52 | real | db1ef24 |
| s3fifo | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 512 | 0.1325 | 0.0000 | 215.63 | real | db1ef24 |
| w_tinylfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 512 | 0.2757 | 0.0000 | 180.03 | real | db1ef24 |
| layer_lru | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 512 | 0.0872 | 0.0000 | 226.90 | real | db1ef24 |
| lcnu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 512 | 0.3295 | 0.0000 | 166.67 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=2,horizon=1,weights=1/1/16 | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 512 | 0.3293 | 0.0168 | 166.31 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | probation | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 512 | 0.3295 | 0.1074 | 159.94 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | pin | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 512 | 0.3295 | 0.1074 | 159.94 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | buffer | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 512 | 0.3279 | 0.1078 | 160.30 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | do_no_harm | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 512 | 0.3295 | 0.0896 | 160.20 | real | db1ef24 |
| lcnu | oracle:horizon=1 | probation | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 512 | 0.3285 | 0.6715 | 146.12 | real | db1ef24 |
| lru | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 1024 | 0.3884 | 0.0000 | 152.02 | real | db1ef24 |
| lfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 1024 | 0.3971 | 0.0000 | 149.87 | real | db1ef24 |
| lirs | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 1024 | 0.4348 | 0.0000 | 140.50 | real | db1ef24 |
| s3fifo | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 1024 | 0.4139 | 0.0000 | 145.68 | real | db1ef24 |
| w_tinylfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 1024 | 0.4026 | 0.0000 | 148.49 | real | db1ef24 |
| layer_lru | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 1024 | 0.4001 | 0.0000 | 149.11 | real | db1ef24 |
| lcnu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 1024 | 0.4842 | 0.0000 | 128.21 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 1024 | 0.4846 | 0.0198 | 128.15 | real | db1ef24 |
| lcnu | coactivation:budget=4,horizon=1 | probation | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 1024 | 0.4852 | 0.0703 | 123.44 | real | db1ef24 |
| lcnu | coactivation:budget=4,horizon=1 | pin | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 1024 | 0.4852 | 0.0703 | 123.44 | real | db1ef24 |
| lcnu | coactivation:budget=4,horizon=1 | buffer | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 1024 | 0.4832 | 0.0705 | 123.93 | real | db1ef24 |
| lcnu | coactivation:budget=4,horizon=1 | do_no_harm | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 1024 | 0.4851 | 0.0568 | 123.90 | real | db1ef24 |
| lcnu | oracle:horizon=1 | probation | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 1024 | 0.4863 | 0.5137 | 107.04 | real | db1ef24 |
| lru | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 2048 | 0.6206 | 0.0000 | 94.32 | real | db1ef24 |
| lfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 2048 | 0.5848 | 0.0000 | 103.21 | real | db1ef24 |
| lirs | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 2048 | 0.6149 | 0.0000 | 95.73 | real | db1ef24 |
| s3fifo | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 2048 | 0.6163 | 0.0000 | 95.38 | real | db1ef24 |
| w_tinylfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 2048 | 0.5755 | 0.0000 | 105.52 | real | db1ef24 |
| layer_lru | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 2048 | 0.6228 | 0.0000 | 93.76 | real | db1ef24 |
| lcnu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 2048 | 0.6643 | 0.0000 | 83.45 | real | db1ef24 |
| lcnu | coactivation:budget=4,horizon=1 | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 2048 | 0.6645 | 0.0119 | 82.98 | real | db1ef24 |
| lcnu | coactivation:budget=8,horizon=1 | probation | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 2048 | 0.6663 | 0.0932 | 78.50 | real | db1ef24 |
| lcnu | coactivation:budget=8,horizon=1 | pin | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 2048 | 0.6663 | 0.0932 | 78.50 | real | db1ef24 |
| lcnu | coactivation:budget=8,horizon=1 | buffer | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 2048 | 0.6632 | 0.0936 | 79.26 | real | db1ef24 |
| lcnu | coactivation:budget=8,horizon=1 | do_no_harm | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 2048 | 0.6659 | 0.0645 | 79.37 | real | db1ef24 |
| lcnu | oracle:horizon=4 | probation | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 2048 | 0.6662 | 0.3338 | 62.22 | real | db1ef24 |
| lru | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 4096 | 0.8378 | 0.0000 | 40.33 | real | db1ef24 |
| lfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 4096 | 0.8190 | 0.0000 | 44.99 | real | db1ef24 |
| lirs | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 4096 | 0.8242 | 0.0000 | 43.71 | real | db1ef24 |
| s3fifo | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 4096 | 0.8458 | 0.0000 | 38.34 | real | db1ef24 |
| w_tinylfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 4096 | 0.8076 | 0.0000 | 47.82 | real | db1ef24 |
| layer_lru | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 4096 | 0.8329 | 0.0000 | 41.54 | real | db1ef24 |
| lcnu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 4096 | 0.8637 | 0.0000 | 33.87 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=8,horizon=2,weights=1/1/16 | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 4096 | 0.8636 | 0.0069 | 33.27 | real | db1ef24 |
| lcnu | coactivation:budget=8,horizon=1 | probation | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 4096 | 0.8639 | 0.0131 | 32.63 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=8,horizon=2,weights=1/1/16 | pin | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 4096 | 0.8639 | 0.0130 | 32.53 | real | db1ef24 |
| lcnu | coactivation:budget=8,horizon=1 | buffer | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 4096 | 0.8632 | 0.0132 | 32.80 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=8,horizon=2,weights=1/1/16 | do_no_harm | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 4096 | 0.8639 | 0.0109 | 32.64 | real | db1ef24 |
| lcnu | oracle:horizon=4 | probation | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 4096 | 0.8643 | 0.1357 | 15.69 | real | db1ef24 |
| lru | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 6016 | 0.9377 | 0.0000 | 15.48 | real | db1ef24 |
| lfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 6016 | 0.9352 | 0.0000 | 16.10 | real | db1ef24 |
| lirs | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 6016 | 0.9370 | 0.0000 | 15.66 | real | db1ef24 |
| s3fifo | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 6016 | 0.9470 | 0.0000 | 13.19 | real | db1ef24 |
| w_tinylfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 6016 | 0.9317 | 0.0000 | 16.98 | real | db1ef24 |
| layer_lru | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 6016 | 0.9315 | 0.0000 | 17.03 | real | db1ef24 |
| lcnu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 6016 | 0.9527 | 0.0000 | 11.75 | real | db1ef24 |
| lcnu | coactivation:budget=16,horizon=4 | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 6016 | 0.9523 | 0.0053 | 11.35 | real | db1ef24 |
| lcnu | coactivation:budget=16,horizon=1 | probation | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 6016 | 0.9528 | 0.0047 | 11.31 | real | db1ef24 |
| lcnu | coactivation:budget=16,horizon=2 | pin | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 6016 | 0.9528 | 0.0063 | 11.13 | real | db1ef24 |
| lcnu | coactivation:budget=16,horizon=1 | buffer | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 6016 | 0.9523 | 0.0048 | 11.43 | real | db1ef24 |
| lcnu | coactivation:budget=16,horizon=1 | do_no_harm | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 6016 | 0.9528 | 0.0031 | 11.43 | real | db1ef24 |
| lcnu | oracle:horizon=4 | pin | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 6016 | 0.9528 | 0.0472 | 2.47 | real | db1ef24 |

<!-- run: python -m benchmarks.cache_simulator.research.run_prefetch_real --model qwen3-235b-a22b-fp8 --split held --prefill skip --tag qwen3_235b_held | env: python 3.12.3, numpy 2.4.4, Linux x86_64 | commit: db1ef24 -->

| evictor | prefetcher | admission | model | dataset | capacity | hit_rate | covered | stall_ms_tok | provenance | commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lru | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 128 | 0.0000 | 0.0000 | 71.84 | real | db1ef24 |
| lfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 128 | 0.1189 | 0.0000 | 63.30 | real | db1ef24 |
| s3fifo | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 128 | 0.0000 | 0.0000 | 71.84 | real | db1ef24 |
| w_tinylfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 128 | 0.1556 | 0.0000 | 60.66 | real | db1ef24 |
| lcnu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 128 | 0.1973 | 0.0000 | 57.67 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=2,horizon=1,weights=1/1/16 | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 128 | 0.1970 | 0.0323 | 56.13 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | probation | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 128 | 0.1946 | 0.1738 | 53.60 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | pin | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 128 | 0.1946 | 0.1738 | 53.60 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | buffer | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 128 | 0.1926 | 0.1750 | 53.73 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | do_no_harm | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 128 | 0.1951 | 0.1505 | 53.62 | real | db1ef24 |
| lcnu | oracle:horizon=1 | pin | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 128 | 0.1877 | 0.8017 | 49.14 | real | db1ef24 |
| lru | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 256 | 0.0000 | 0.0000 | 71.84 | real | db1ef24 |
| lfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 256 | 0.2062 | 0.0000 | 57.03 | real | db1ef24 |
| s3fifo | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 256 | 0.0450 | 0.0000 | 68.61 | real | db1ef24 |
| w_tinylfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 256 | 0.2537 | 0.0000 | 53.62 | real | db1ef24 |
| lcnu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 256 | 0.3144 | 0.0000 | 49.26 | real | db1ef24 |
| lcnu | coactivation:budget=2,horizon=1 | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 256 | 0.3143 | 0.0196 | 48.28 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | probation | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 256 | 0.3140 | 0.1121 | 45.95 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | pin | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 256 | 0.3140 | 0.1121 | 45.95 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | buffer | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 256 | 0.3114 | 0.1130 | 46.12 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | do_no_harm | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 256 | 0.3141 | 0.0975 | 46.01 | real | db1ef24 |
| lcnu | oracle:horizon=1 | do_no_harm | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 256 | 0.3091 | 0.6908 | 40.44 | real | db1ef24 |
| lru | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 512 | 0.3877 | 0.0000 | 43.99 | real | db1ef24 |
| lfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 512 | 0.3301 | 0.0000 | 48.13 | real | db1ef24 |
| s3fifo | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 512 | 0.3873 | 0.0000 | 44.02 | real | db1ef24 |
| w_tinylfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 512 | 0.3715 | 0.0000 | 45.16 | real | db1ef24 |
| lcnu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 512 | 0.4530 | 0.0000 | 39.30 | real | db1ef24 |
| lcnu | coactivation:budget=4,horizon=1 | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 512 | 0.4528 | 0.0252 | 38.34 | real | db1ef24 |
| lcnu | coactivation:budget=4,horizon=1 | probation | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 512 | 0.4537 | 0.0801 | 36.71 | real | db1ef24 |
| lcnu | coactivation:budget=4,horizon=1 | pin | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 512 | 0.4537 | 0.0801 | 36.71 | real | db1ef24 |
| lcnu | coactivation:budget=4,horizon=1 | buffer | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 512 | 0.4514 | 0.0804 | 36.88 | real | db1ef24 |
| lcnu | coactivation:budget=4,horizon=1 | do_no_harm | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 512 | 0.4537 | 0.0696 | 36.81 | real | db1ef24 |
| lcnu | oracle:horizon=2 | probation | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 512 | 0.4500 | 0.5499 | 30.32 | real | db1ef24 |
| lru | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 1024 | 0.5714 | 0.0000 | 30.79 | real | db1ef24 |
| lfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 1024 | 0.5141 | 0.0000 | 34.91 | real | db1ef24 |
| s3fifo | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 1024 | 0.5706 | 0.0000 | 30.85 | real | db1ef24 |
| w_tinylfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 1024 | 0.5467 | 0.0000 | 32.57 | real | db1ef24 |
| lcnu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 1024 | 0.6121 | 0.0000 | 27.87 | real | db1ef24 |
| lcnu | coactivation:budget=4,horizon=1 | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 1024 | 0.6111 | 0.0155 | 27.06 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=8,horizon=1,weights=1/1/16 | probation | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 1024 | 0.6134 | 0.0834 | 25.49 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=8,horizon=1,weights=1/1/16 | pin | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 1024 | 0.6134 | 0.0834 | 25.49 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=8,horizon=1,weights=1/1/16 | buffer | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 1024 | 0.6102 | 0.0840 | 25.72 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=8,horizon=1,weights=1/1/16 | do_no_harm | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 1024 | 0.6132 | 0.0668 | 25.66 | real | db1ef24 |
| lcnu | oracle:horizon=4 | probation | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 1024 | 0.6127 | 0.3870 | 18.75 | real | db1ef24 |
| lru | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 2048 | 0.7699 | 0.0000 | 16.53 | real | db1ef24 |
| lfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 2048 | 0.7688 | 0.0000 | 16.61 | real | db1ef24 |
| s3fifo | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 2048 | 0.8012 | 0.0000 | 14.28 | real | db1ef24 |
| w_tinylfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 2048 | 0.7977 | 0.0000 | 14.53 | real | db1ef24 |
| lcnu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 2048 | 0.8087 | 0.0000 | 13.74 | real | db1ef24 |
| s3fifo | coactivation:budget=8,horizon=1 | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 2048 | 0.7943 | 0.0337 | 13.35 | real | db1ef24 |
| w_tinylfu | hybrid:alpha=0.4,budget=8,horizon=1,weights=1/1/16 | probation | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 2048 | 0.8066 | 0.0209 | 13.17 | real | db1ef24 |
| lcnu | coactivation:budget=8,horizon=1 | pin | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 2048 | 0.8088 | 0.0133 | 13.19 | real | db1ef24 |
| lcnu | coactivation:budget=8,horizon=1 | buffer | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 2048 | 0.8076 | 0.0135 | 13.27 | real | db1ef24 |
| lcnu | coactivation:budget=8,horizon=1 | do_no_harm | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 2048 | 0.8088 | 0.0117 | 13.22 | real | db1ef24 |
| w_tinylfu | oracle:horizon=4 | pin | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 2048 | 0.8075 | 0.1921 | 7.15 | real | db1ef24 |
| lru | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 3072 | 0.8919 | 0.0000 | 7.77 | real | db1ef24 |
| lfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 3072 | 0.9094 | 0.0000 | 6.51 | real | db1ef24 |
| s3fifo | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 3072 | 0.9262 | 0.0000 | 5.30 | real | db1ef24 |
| w_tinylfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 3072 | 0.9353 | 0.0000 | 4.65 | real | db1ef24 |
| lcnu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 3072 | 0.9174 | 0.0000 | 5.93 | real | db1ef24 |
| w_tinylfu | prev_token:budget=2,horizon=1 | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 3072 | 0.9350 | 0.0011 | 4.60 | real | db1ef24 |
| w_tinylfu | prev_token:budget=2,horizon=1 | probation | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 3072 | 0.9367 | 0.0005 | 4.51 | real | db1ef24 |
| w_tinylfu | prev_token:budget=2,horizon=1 | pin | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 3072 | 0.9367 | 0.0005 | 4.51 | real | db1ef24 |
| w_tinylfu | prev_token:budget=2,horizon=1 | buffer | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 3072 | 0.9350 | 0.0010 | 4.61 | real | db1ef24 |
| w_tinylfu | prev_token:budget=2,horizon=1 | do_no_harm | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 3072 | 0.9367 | 0.0005 | 4.51 | real | db1ef24 |
| w_tinylfu | oracle:horizon=4 | buffer | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 3072 | 0.9294 | 0.0702 | 2.00 | real | db1ef24 |

<!-- run: python -m benchmarks.cache_simulator.research.run_prefetch_real --model qwen3-30b-a3b --split held --prefill union --tag qwen3_30b_held_union | env: python 3.12.3, numpy 2.4.4, Linux x86_64 | commit: db1ef24 -->

| evictor | prefetcher | admission | model | dataset | capacity | hit_rate | covered | stall_ms_tok | provenance | commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lru | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 256 | 0.0000 | 0.0000 | 281.60 | real | db1ef24 |
| lfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 256 | 0.1444 | 0.0000 | 240.94 | real | db1ef24 |
| s3fifo | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 256 | 0.0000 | 0.0000 | 281.60 | real | db1ef24 |
| w_tinylfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 256 | 0.1533 | 0.0000 | 238.42 | real | db1ef24 |
| lcnu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 256 | 0.1873 | 0.0000 | 228.86 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=2,horizon=1,weights=1/1/16 | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 256 | 0.1871 | 0.0296 | 227.45 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | probation | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 256 | 0.1862 | 0.1492 | 219.82 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | pin | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 256 | 0.1862 | 0.1492 | 219.82 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | buffer | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 256 | 0.1851 | 0.1496 | 220.08 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | do_no_harm | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 256 | 0.1865 | 0.1291 | 219.88 | real | db1ef24 |
| lcnu | oracle:horizon=1 | probation | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 256 | 0.1824 | 0.8176 | 209.11 | real | db1ef24 |
| lru | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 512 | 0.0000 | 0.0000 | 281.60 | real | db1ef24 |
| lfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 512 | 0.2362 | 0.0000 | 215.09 | real | db1ef24 |
| s3fifo | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 512 | 0.1215 | 0.0000 | 247.37 | real | db1ef24 |
| w_tinylfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 512 | 0.2530 | 0.0000 | 210.36 | real | db1ef24 |
| lcnu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 512 | 0.2997 | 0.0000 | 197.20 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=2,horizon=1,weights=1/1/16 | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 512 | 0.2995 | 0.0151 | 196.72 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | probation | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 512 | 0.2995 | 0.0919 | 190.72 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | pin | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 512 | 0.2995 | 0.0919 | 190.72 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | buffer | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 512 | 0.2983 | 0.0922 | 191.05 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | do_no_harm | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 512 | 0.2996 | 0.0794 | 190.89 | real | db1ef24 |
| lcnu | oracle:horizon=1 | probation | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 512 | 0.2976 | 0.7024 | 176.67 | real | db1ef24 |
| lru | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 1024 | 0.3424 | 0.0000 | 185.18 | real | db1ef24 |
| lfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 1024 | 0.3671 | 0.0000 | 178.22 | real | db1ef24 |
| s3fifo | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 1024 | 0.3771 | 0.0000 | 175.40 | real | db1ef24 |
| w_tinylfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 1024 | 0.3704 | 0.0000 | 177.30 | real | db1ef24 |
| lcnu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 1024 | 0.4408 | 0.0000 | 157.48 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 1024 | 0.4409 | 0.0183 | 157.32 | real | db1ef24 |
| lcnu | coactivation:budget=4,horizon=1 | probation | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 1024 | 0.4414 | 0.0608 | 152.96 | real | db1ef24 |
| lcnu | coactivation:budget=4,horizon=1 | pin | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 1024 | 0.4414 | 0.0608 | 152.96 | real | db1ef24 |
| lcnu | coactivation:budget=4,horizon=1 | buffer | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 1024 | 0.4399 | 0.0610 | 153.36 | real | db1ef24 |
| lcnu | coactivation:budget=4,horizon=1 | do_no_harm | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 1024 | 0.4412 | 0.0515 | 153.29 | real | db1ef24 |
| lcnu | oracle:horizon=2 | probation | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 1024 | 0.4418 | 0.5581 | 136.05 | real | db1ef24 |
| lru | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 2048 | 0.5511 | 0.0000 | 126.40 | real | db1ef24 |
| lfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 2048 | 0.5488 | 0.0000 | 127.05 | real | db1ef24 |
| s3fifo | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 2048 | 0.5668 | 0.0000 | 121.98 | real | db1ef24 |
| w_tinylfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 2048 | 0.5373 | 0.0000 | 130.29 | real | db1ef24 |
| lcnu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 2048 | 0.6052 | 0.0000 | 111.17 | real | db1ef24 |
| lcnu | coactivation:budget=4,horizon=1 | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 2048 | 0.6053 | 0.0107 | 110.68 | real | db1ef24 |
| lcnu | coactivation:budget=8,horizon=1 | probation | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 2048 | 0.6067 | 0.0805 | 106.44 | real | db1ef24 |
| lcnu | coactivation:budget=8,horizon=1 | pin | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 2048 | 0.6067 | 0.0805 | 106.44 | real | db1ef24 |
| lcnu | coactivation:budget=8,horizon=1 | buffer | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 2048 | 0.6042 | 0.0808 | 107.13 | real | db1ef24 |
| lcnu | coactivation:budget=8,horizon=1 | do_no_harm | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 2048 | 0.6064 | 0.0605 | 107.04 | real | db1ef24 |
| lcnu | oracle:horizon=4 | probation | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 2048 | 0.6087 | 0.3912 | 89.12 | real | db1ef24 |
| lru | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 4096 | 0.7593 | 0.0000 | 67.77 | real | db1ef24 |
| lfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 4096 | 0.7872 | 0.0000 | 59.91 | real | db1ef24 |
| s3fifo | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 4096 | 0.7883 | 0.0000 | 59.62 | real | db1ef24 |
| w_tinylfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 4096 | 0.7705 | 0.0000 | 64.62 | real | db1ef24 |
| lcnu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 4096 | 0.7988 | 0.0000 | 56.66 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=8,horizon=2,weights=1/1/16 | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 4096 | 0.7984 | 0.0057 | 56.11 | real | db1ef24 |
| lcnu | coactivation:budget=8,horizon=1 | probation | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 4096 | 0.7989 | 0.0106 | 55.59 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=8,horizon=2,weights=1/1/16 | pin | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 4096 | 0.7989 | 0.0093 | 55.55 | real | db1ef24 |
| lcnu | coactivation:budget=8,horizon=1 | buffer | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 4096 | 0.7982 | 0.0107 | 55.77 | real | db1ef24 |
| lcnu | hybrid:alpha=0.4,budget=8,horizon=2,weights=1/1/16 | do_no_harm | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 4096 | 0.7989 | 0.0083 | 55.60 | real | db1ef24 |
| lcnu | oracle:horizon=4 | probation | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 4096 | 0.8008 | 0.1987 | 37.81 | real | db1ef24 |
| lru | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 6016 | 0.8777 | 0.0000 | 34.43 | real | db1ef24 |
| lfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 6016 | 0.9149 | 0.0000 | 23.96 | real | db1ef24 |
| s3fifo | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 6016 | 0.9097 | 0.0000 | 25.43 | real | db1ef24 |
| w_tinylfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 6016 | 0.9083 | 0.0000 | 25.82 | real | db1ef24 |
| lcnu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 6016 | 0.9047 | 0.0000 | 26.83 | real | db1ef24 |
| lfu | hybrid:alpha=0.4,budget=16,horizon=4,weights=1/1/16 | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 6016 | 0.9149 | 0.0046 | 23.71 | real | db1ef24 |
| lfu | hybrid:alpha=0.4,budget=16,horizon=1,weights=1/1/16 | probation | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 6016 | 0.9149 | 0.0037 | 23.54 | real | db1ef24 |
| lfu | hybrid:alpha=0.4,budget=16,horizon=2,weights=1/1/16 | pin | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 6016 | 0.9149 | 0.0048 | 23.35 | real | db1ef24 |
| lfu | hybrid:alpha=0.4,budget=16,horizon=1,weights=1/1/16 | buffer | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 6016 | 0.9141 | 0.0038 | 23.74 | real | db1ef24 |
| lfu | hybrid:alpha=0.4,budget=16,horizon=1,weights=1/1/16 | do_no_harm | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 6016 | 0.9149 | 0.0022 | 23.67 | real | db1ef24 |
| lfu | oracle:horizon=4 | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 6016 | 0.9147 | 0.0853 | 12.79 | real | db1ef24 |

<!-- run: python -m benchmarks.cache_simulator.research.run_prefetch_real --model qwen3-235b-a22b-fp8 --split held --prefill union --tag qwen3_235b_held_union | env: python 3.12.3, numpy 2.4.4, Linux x86_64 | commit: db1ef24 -->

## Round 3 — more predictors and one default policy (#289)

Same traces, protocol, streams and timing model as round 2. Study:
`default_policy_study.md`; default-policy regret:
`results_default_policy_analysis.md`.

Setup runs (commands in the study's Reproduce section; tuning-half runs are
summarized here, not logged row by row):

| run | model | dataset | result | provenance | commit |
| --- | --- | --- | --- | --- | --- |
| predictor stage A (`--mode predict`, 78 specs) | qwen3-30b-a3b | mmlu150-greedy64/decode/tune | mean recall@8 over distances 1-4: knn context 8, 8 neighbours, power 8 0.7717 (selected; context 12: 0.7727), logistic lr 0.1 0.6939, co-activation naive Bayes 0.6272, multi-visit co-activation 0.5834 (`results_prefetch_qwen3_30b_tune_predict_r3.csv`) | real | d3ed922 |
| eviction, no prefetch (8 evictors x 6 capacities) | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/tune | mean stall vs LCNU: lcnu_p 0.923; prefill weight 0.1 / 0.25 / 0.5: 0.936 / 0.958 / 0.980; lcnu_b 0.922; LFU 1.036; W-TinyLFU 1.063 (`results_prefetch_qwen3_30b_tune_union_evict.md`) | real | d3ed922 |
| bypass on prefill misses (2 evictors x 6 capacities) | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/tune | mean stall vs LCNU: `lcnu:bypass=prefill` 1.000, `lcnu_p:bypass=prefill` 0.923 (`results_prefetch_qwen3_30b_tune_union_evict_bypass.md`) | real | e0bdd04 |
| stage B (LCNU x 70 prefetch configurations x 6 capacities, 426 runs) | qwen3-30b-a3b | mmlu150-greedy64/decode/tune | knn budget 8 on `probation:issue=2`: mean stall 0.773 of LCNU alone, worst regret +4.6% vs the per-cell best (`results_prefetch_qwen3_30b_tune_r3_default.md`) | real | d3ed922 |
| predictor accuracy (`--mode predict`) | qwen3-30b-a3b | mmlu150-greedy64/decode/held | recall@8: knn 0.779, logistic 0.722, co-activation naive Bayes 0.655 (`results_prefetch_qwen3_30b_held_predict_r3.csv`) | real | d3ed922 |
| predictor accuracy (`--mode predict`) | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | recall@8: knn 0.714, logistic 0.661, co-activation naive Bayes 0.602 (`results_prefetch_qwen3_235b_held_predict_r3.csv`) | real | d3ed922 |
| default-policy regret (`default_policy_analysis.py` over the round-2 and round-3 held-out sweeps) | both | decode and decode+prefill-union, held | 24 cells: mean +0.5%, worst +6.2% vs the per-cell best online configuration (`results_default_policy_analysis.md`) | real | e0bdd04 |

Held-out sweeps below: compact rows (per capacity, every evictor alone, the
best online pair per admission mode, where `probation` includes its issue
caps, and the best oracle pair); full rows in `results_prefetch_<tag>.csv`.
The `*_held_union_r3` sweeps run kNN without `decode_width`; the
`*_held_union_r3d` sweeps rerun the prefill-union stream with the default's
decode-only kNN.

| evictor | prefetcher | admission | model | dataset | capacity | hit_rate | covered | stall_ms_tok | provenance | commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lcnu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 128 | 0.2179 | 0.0000 | 49.64 | real | d3ed922 |
| lru | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 128 | 0.0000 | 0.0000 | 63.47 | real | d3ed922 |
| lfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 128 | 0.1303 | 0.0000 | 55.20 | real | d3ed922 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 128 | 0.2163 | 0.2223 | 42.42 | real | d3ed922 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | do_no_harm | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 128 | 0.2124 | 0.4297 | 42.67 | real | d3ed922 |
| lcnu | oracle:horizon=1 | probation:issue=2 | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 128 | 0.2166 | 0.2500 | 40.64 | real | d3ed922 |
| lcnu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 256 | 0.3461 | 0.0000 | 41.50 | real | d3ed922 |
| lru | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 256 | 0.0000 | 0.0000 | 63.47 | real | d3ed922 |
| lfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 256 | 0.2251 | 0.0000 | 49.18 | real | d3ed922 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 256 | 0.3458 | 0.2106 | 34.83 | real | d3ed922 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | do_no_harm | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 256 | 0.3446 | 0.3548 | 34.88 | real | d3ed922 |
| lcnu | oracle:horizon=1 | probation:issue=2 | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 256 | 0.3463 | 0.2482 | 32.44 | real | d3ed922 |
| lcnu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 512 | 0.4976 | 0.0000 | 31.89 | real | d3ed922 |
| lru | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 512 | 0.4398 | 0.0000 | 35.56 | real | d3ed922 |
| lfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 512 | 0.3577 | 0.0000 | 40.76 | real | d3ed922 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 512 | 0.4992 | 0.1882 | 25.90 | real | d3ed922 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | do_no_harm | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 512 | 0.4993 | 0.2677 | 25.87 | real | d3ed922 |
| lcnu | oracle:horizon=1 | probation:issue=2 | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 512 | 0.4996 | 0.2377 | 22.90 | real | d3ed922 |
| lcnu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 1024 | 0.6697 | 0.0000 | 20.96 | real | d3ed922 |
| lru | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 1024 | 0.6432 | 0.0000 | 22.65 | real | d3ed922 |
| lfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 1024 | 0.5488 | 0.0000 | 28.64 | real | d3ed922 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 1024 | 0.6725 | 0.2205 | 16.01 | real | d3ed922 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | do_no_harm | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 1024 | 0.6721 | 0.1781 | 16.11 | real | d3ed922 |
| lcnu | oracle:horizon=4 | probation | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 1024 | 0.6705 | 0.3295 | 11.98 | real | d3ed922 |
| lcnu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 2048 | 0.8662 | 0.0000 | 8.49 | real | d3ed922 |
| lru | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 2048 | 0.8490 | 0.0000 | 9.58 | real | d3ed922 |
| lfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 2048 | 0.8025 | 0.0000 | 12.53 | real | d3ed922 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 2048 | 0.8667 | 0.0730 | 5.87 | real | d3ed922 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | do_no_harm | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 2048 | 0.8666 | 0.0613 | 6.04 | real | d3ed922 |
| lcnu | oracle:horizon=4 | do_no_harm | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 2048 | 0.8664 | 0.1336 | 1.92 | real | d3ed922 |
| lcnu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 3072 | 0.9558 | 0.0000 | 2.80 | real | d3ed922 |
| lru | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 3072 | 0.9506 | 0.0000 | 3.14 | real | d3ed922 |
| lfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 3072 | 0.9314 | 0.0000 | 4.36 | real | d3ed922 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | probation:issue=2 | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 3072 | 0.9559 | 0.0204 | 1.85 | real | d3ed922 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | do_no_harm | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 3072 | 0.9559 | 0.0172 | 1.95 | real | d3ed922 |
| lcnu | oracle:horizon=4 | do_no_harm | qwen3-30b-a3b | mmlu150-greedy64/decode/held | 3072 | 0.9558 | 0.0442 | 0.16 | real | d3ed922 |

<!-- run: python -m benchmarks.cache_simulator.research.run_prefetch_real --model qwen3-30b-a3b --split held --prefill skip --tag qwen3_30b_held_r3 | env: python 3.12.3, numpy 2.4.4, Linux x86_64 | commit: d3ed922 -->

| evictor | prefetcher | admission | model | dataset | capacity | hit_rate | covered | stall_ms_tok | provenance | commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lcnu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 256 | 0.2064 | 0.0000 | 197.26 | real | d3ed922 |
| lru | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 256 | 0.0000 | 0.0000 | 248.57 | real | d3ed922 |
| lfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 256 | 0.1586 | 0.0000 | 209.15 | real | d3ed922 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | probation:issue=2 | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 256 | 0.2057 | 0.2105 | 180.39 | real | d3ed922 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | do_no_harm | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 256 | 0.2044 | 0.3533 | 180.89 | real | d3ed922 |
| lcnu | oracle:horizon=1 | probation:issue=2 | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 256 | 0.2059 | 0.2500 | 176.58 | real | d3ed922 |
| lcnu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 512 | 0.3295 | 0.0000 | 166.67 | real | d3ed922 |
| lru | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 512 | 0.0000 | 0.0000 | 248.57 | real | d3ed922 |
| lfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 512 | 0.2579 | 0.0000 | 184.46 | real | d3ed922 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | probation:issue=2 | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 512 | 0.3296 | 0.1992 | 150.63 | real | d3ed922 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | do_no_harm | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 512 | 0.3292 | 0.2940 | 151.05 | real | d3ed922 |
| lcnu | oracle:horizon=1 | probation:issue=2 | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 512 | 0.3299 | 0.2494 | 145.76 | real | d3ed922 |
| lcnu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 1024 | 0.4842 | 0.0000 | 128.21 | real | d3ed922 |
| lru | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 1024 | 0.3884 | 0.0000 | 152.02 | real | d3ed922 |
| lfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 1024 | 0.3971 | 0.0000 | 149.87 | real | d3ed922 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 1024 | 0.4863 | 0.3213 | 113.58 | real | d3ed922 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | do_no_harm | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 1024 | 0.4862 | 0.2197 | 114.26 | real | d3ed922 |
| lcnu | oracle:horizon=1 | probation:issue=2 | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 1024 | 0.4863 | 0.2419 | 107.04 | real | d3ed922 |
| lcnu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 2048 | 0.6643 | 0.0000 | 83.45 | real | d3ed922 |
| lru | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 2048 | 0.6206 | 0.0000 | 94.32 | real | d3ed922 |
| lfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 2048 | 0.5848 | 0.0000 | 103.21 | real | d3ed922 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 2048 | 0.6671 | 0.1960 | 71.76 | real | d3ed922 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | do_no_harm | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 2048 | 0.6666 | 0.1455 | 72.62 | real | d3ed922 |
| lcnu | oracle:horizon=4 | probation | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 2048 | 0.6662 | 0.3338 | 62.22 | real | d3ed922 |
| lcnu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 4096 | 0.8637 | 0.0000 | 33.87 | real | d3ed922 |
| lru | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 4096 | 0.8378 | 0.0000 | 40.33 | real | d3ed922 |
| lfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 4096 | 0.8190 | 0.0000 | 44.99 | real | d3ed922 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 4096 | 0.8643 | 0.0628 | 28.59 | real | d3ed922 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | do_no_harm | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 4096 | 0.8642 | 0.0518 | 29.01 | real | d3ed922 |
| lcnu | oracle:horizon=4 | probation | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 4096 | 0.8643 | 0.1357 | 15.69 | real | d3ed922 |
| lcnu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 6016 | 0.9527 | 0.0000 | 11.75 | real | d3ed922 |
| lru | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 6016 | 0.9377 | 0.0000 | 15.48 | real | d3ed922 |
| lfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 6016 | 0.9352 | 0.0000 | 16.10 | real | d3ed922 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | probation:issue=2 | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 6016 | 0.9528 | 0.0179 | 10.02 | real | d3ed922 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | do_no_harm | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 6016 | 0.9528 | 0.0151 | 10.23 | real | d3ed922 |
| lcnu | oracle:horizon=4 | do_no_harm | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode/held | 6016 | 0.9528 | 0.0472 | 2.47 | real | d3ed922 |

<!-- run: python -m benchmarks.cache_simulator.research.run_prefetch_real --model qwen3-235b-a22b-fp8 --split held --prefill skip --tag qwen3_235b_held_r3 | env: python 3.12.3, numpy 2.4.4, Linux x86_64 | commit: d3ed922 -->

| evictor | prefetcher | admission | model | dataset | capacity | hit_rate | covered | stall_ms_tok | provenance | commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lcnu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 128 | 0.1973 | 0.0000 | 57.67 | real | d3ed922 |
| lcnu_p | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 128 | 0.1973 | 0.0000 | 57.67 | real | d3ed922 |
| w_tinylfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 128 | 0.1556 | 0.0000 | 60.66 | real | d3ed922 |
| lfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 128 | 0.1189 | 0.0000 | 63.30 | real | d3ed922 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 128 | 0.1961 | 0.0771 | 59.11 | real | d3ed922 |
| lcnu_p | oracle:horizon=4 | probation:issue=2 | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 128 | 0.1962 | 0.0000 | 64.64 | real | d3ed922 |
| lcnu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 256 | 0.3144 | 0.0000 | 49.26 | real | d3ed922 |
| lcnu_p | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 256 | 0.3147 | 0.0000 | 49.23 | real | d3ed922 |
| w_tinylfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 256 | 0.2537 | 0.0000 | 53.62 | real | d3ed922 |
| lfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 256 | 0.2062 | 0.0000 | 57.03 | real | d3ed922 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 256 | 0.3149 | 0.0698 | 51.07 | real | d3ed922 |
| lcnu_p | oracle:horizon=4 | probation:issue=2 | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 256 | 0.3151 | 0.0004 | 55.97 | real | d3ed922 |
| lcnu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 512 | 0.4530 | 0.0000 | 39.30 | real | d3ed922 |
| lcnu_p | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 512 | 0.4565 | 0.0000 | 39.05 | real | d3ed922 |
| w_tinylfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 512 | 0.3715 | 0.0000 | 45.16 | real | d3ed922 |
| lfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 512 | 0.3301 | 0.0000 | 48.13 | real | d3ed922 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 512 | 0.4584 | 0.0609 | 41.14 | real | d3ed922 |
| lcnu_p | oracle:horizon=4 | probation:issue=2 | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 512 | 0.4585 | 0.0087 | 44.43 | real | d3ed922 |
| lcnu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 1024 | 0.6121 | 0.0000 | 27.87 | real | d3ed922 |
| lcnu_p | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 1024 | 0.6230 | 0.0000 | 27.09 | real | d3ed922 |
| w_tinylfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 1024 | 0.5467 | 0.0000 | 32.57 | real | d3ed922 |
| lfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 1024 | 0.5141 | 0.0000 | 34.91 | real | d3ed922 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 1024 | 0.6251 | 0.0487 | 29.01 | real | d3ed922 |
| lcnu_p | oracle:horizon=4 | probation:issue=2 | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 1024 | 0.6252 | 0.0462 | 27.71 | real | d3ed922 |
| lcnu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 2048 | 0.8087 | 0.0000 | 13.74 | real | d3ed922 |
| lcnu_p | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 2048 | 0.8262 | 0.0000 | 12.49 | real | d3ed922 |
| w_tinylfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 2048 | 0.7977 | 0.0000 | 14.53 | real | d3ed922 |
| lfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 2048 | 0.7688 | 0.0000 | 16.61 | real | d3ed922 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 2048 | 0.8265 | 0.0227 | 13.19 | real | d3ed922 |
| lcnu_p | oracle:horizon=4 | probation:issue=2 | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 2048 | 0.8267 | 0.0746 | 8.01 | real | d3ed922 |
| lcnu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 3072 | 0.9174 | 0.0000 | 5.93 | real | d3ed922 |
| lcnu_p | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 3072 | 0.9313 | 0.0000 | 4.94 | real | d3ed922 |
| w_tinylfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 3072 | 0.9353 | 0.0000 | 4.65 | real | d3ed922 |
| lfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 3072 | 0.9094 | 0.0000 | 6.51 | real | d3ed922 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 3072 | 0.9313 | 0.0077 | 4.88 | real | d3ed922 |
| lcnu_p | oracle:horizon=4 | probation:issue=2 | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 3072 | 0.9313 | 0.0347 | 2.61 | real | d3ed922 |

<!-- run: python -m benchmarks.cache_simulator.research.run_prefetch_real --model qwen3-30b-a3b --split held --prefill union --tag qwen3_30b_held_union_r3 | env: python 3.12.3, numpy 2.4.4, Linux x86_64 | commit: d3ed922 -->

| evictor | prefetcher | admission | model | dataset | capacity | hit_rate | covered | stall_ms_tok | provenance | commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lcnu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 256 | 0.1873 | 0.0000 | 228.86 | real | d3ed922 |
| lcnu_p | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 256 | 0.1873 | 0.0000 | 228.87 | real | d3ed922 |
| w_tinylfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 256 | 0.1533 | 0.0000 | 238.42 | real | d3ed922 |
| lfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 256 | 0.1444 | 0.0000 | 240.94 | real | d3ed922 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 256 | 0.1868 | 0.0496 | 232.06 | real | d3ed922 |
| lcnu | oracle:horizon=4 | probation:issue=2 | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 256 | 0.1868 | 0.0000 | 239.41 | real | d3ed922 |
| lcnu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 512 | 0.2997 | 0.0000 | 197.20 | real | d3ed922 |
| lcnu_p | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 512 | 0.2999 | 0.0000 | 197.14 | real | d3ed922 |
| w_tinylfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 512 | 0.2530 | 0.0000 | 210.36 | real | d3ed922 |
| lfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 512 | 0.2362 | 0.0000 | 215.09 | real | d3ed922 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 512 | 0.3004 | 0.0410 | 201.44 | real | d3ed922 |
| lcnu_p | oracle:horizon=4 | probation:issue=2 | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 512 | 0.3004 | 0.0000 | 207.40 | real | d3ed922 |
| lcnu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 1024 | 0.4408 | 0.0000 | 157.48 | real | d3ed922 |
| lcnu_p | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 1024 | 0.4442 | 0.0000 | 156.51 | real | d3ed922 |
| w_tinylfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 1024 | 0.3704 | 0.0000 | 177.30 | real | d3ed922 |
| lfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 1024 | 0.3671 | 0.0000 | 178.22 | real | d3ed922 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 1024 | 0.4463 | 0.0332 | 161.50 | real | d3ed922 |
| lcnu_p | oracle:horizon=4 | probation:issue=2 | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 1024 | 0.4464 | 0.0050 | 164.89 | real | d3ed922 |
| lcnu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 2048 | 0.6052 | 0.0000 | 111.17 | real | d3ed922 |
| lcnu_p | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 2048 | 0.6175 | 0.0000 | 107.72 | real | d3ed922 |
| w_tinylfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 2048 | 0.5373 | 0.0000 | 130.29 | real | d3ed922 |
| lfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 2048 | 0.5488 | 0.0000 | 127.05 | real | d3ed922 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 2048 | 0.6198 | 0.0250 | 113.55 | real | d3ed922 |
| lcnu_p | oracle:horizon=4 | probation:issue=2 | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 2048 | 0.6199 | 0.0423 | 106.04 | real | d3ed922 |
| lcnu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 4096 | 0.7988 | 0.0000 | 56.66 | real | d3ed922 |
| lcnu_p | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 4096 | 0.8237 | 0.0000 | 49.63 | real | d3ed922 |
| w_tinylfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 4096 | 0.7705 | 0.0000 | 64.62 | real | d3ed922 |
| lfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 4096 | 0.7872 | 0.0000 | 59.91 | real | d3ed922 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 4096 | 0.8242 | 0.0115 | 54.80 | real | d3ed922 |
| lcnu_p | oracle:horizon=4 | probation:issue=2 | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 4096 | 0.8244 | 0.0755 | 34.87 | real | d3ed922 |
| lcnu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 6016 | 0.9047 | 0.0000 | 26.83 | real | d3ed922 |
| lcnu_p | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 6016 | 0.9277 | 0.0000 | 20.36 | real | d3ed922 |
| w_tinylfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 6016 | 0.9083 | 0.0000 | 25.82 | real | d3ed922 |
| lfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 6016 | 0.9149 | 0.0000 | 23.96 | real | d3ed922 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 6016 | 0.9277 | 0.0041 | 22.35 | real | d3ed922 |
| lcnu_p | oracle:horizon=4 | probation:issue=2 | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 6016 | 0.9278 | 0.0351 | 11.79 | real | d3ed922 |

<!-- run: python -m benchmarks.cache_simulator.research.run_prefetch_real --model qwen3-235b-a22b-fp8 --split held --prefill union --tag qwen3_235b_held_union_r3 | env: python 3.12.3, numpy 2.4.4, Linux x86_64 | commit: d3ed922 -->

| evictor | prefetcher | admission | model | dataset | capacity | hit_rate | covered | stall_ms_tok | provenance | commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lcnu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 128 | 0.1973 | 0.0000 | 57.67 | real | 2f8d8a4 |
| lcnu_p | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 128 | 0.1973 | 0.0000 | 57.67 | real | 2f8d8a4 |
| w_tinylfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 128 | 0.1556 | 0.0000 | 60.66 | real | 2f8d8a4 |
| lfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 128 | 0.1189 | 0.0000 | 63.30 | real | 2f8d8a4 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 128 | 0.1959 | 0.1966 | 50.43 | real | 2f8d8a4 |
| lcnu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 256 | 0.3144 | 0.0000 | 49.26 | real | 2f8d8a4 |
| lcnu_p | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 256 | 0.3147 | 0.0000 | 49.23 | real | 2f8d8a4 |
| w_tinylfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 256 | 0.2537 | 0.0000 | 53.62 | real | 2f8d8a4 |
| lfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 256 | 0.2062 | 0.0000 | 57.03 | real | 2f8d8a4 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 256 | 0.3147 | 0.1861 | 42.54 | real | 2f8d8a4 |
| lcnu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 512 | 0.4530 | 0.0000 | 39.30 | real | 2f8d8a4 |
| lcnu_p | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 512 | 0.4565 | 0.0000 | 39.05 | real | 2f8d8a4 |
| w_tinylfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 512 | 0.3715 | 0.0000 | 45.16 | real | 2f8d8a4 |
| lfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 512 | 0.3301 | 0.0000 | 48.13 | real | 2f8d8a4 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 512 | 0.4583 | 0.1661 | 33.04 | real | 2f8d8a4 |
| lcnu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 1024 | 0.6121 | 0.0000 | 27.87 | real | 2f8d8a4 |
| lcnu_p | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 1024 | 0.6230 | 0.0000 | 27.09 | real | 2f8d8a4 |
| w_tinylfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 1024 | 0.5467 | 0.0000 | 32.57 | real | 2f8d8a4 |
| lfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 1024 | 0.5141 | 0.0000 | 34.91 | real | 2f8d8a4 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 1024 | 0.6250 | 0.1280 | 22.17 | real | 2f8d8a4 |
| lcnu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 2048 | 0.8087 | 0.0000 | 13.74 | real | 2f8d8a4 |
| lcnu_p | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 2048 | 0.8262 | 0.0000 | 12.49 | real | 2f8d8a4 |
| w_tinylfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 2048 | 0.7977 | 0.0000 | 14.53 | real | 2f8d8a4 |
| lfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 2048 | 0.7688 | 0.0000 | 16.61 | real | 2f8d8a4 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 2048 | 0.8266 | 0.0552 | 9.90 | real | 2f8d8a4 |
| lcnu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 3072 | 0.9174 | 0.0000 | 5.93 | real | 2f8d8a4 |
| lcnu_p | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 3072 | 0.9313 | 0.0000 | 4.94 | real | 2f8d8a4 |
| w_tinylfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 3072 | 0.9353 | 0.0000 | 4.65 | real | 2f8d8a4 |
| lfu | none | same_queue | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 3072 | 0.9094 | 0.0000 | 6.51 | real | 2f8d8a4 |
| w_tinylfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | qwen3-30b-a3b | mmlu150-greedy64/decode+prefill-union/held | 3072 | 0.9351 | 0.0143 | 3.85 | real | 2f8d8a4 |

<!-- run: python -m benchmarks.cache_simulator.research.run_prefetch_real --model qwen3-30b-a3b --split held --prefill union --tag qwen3_30b_held_union_r3d | env: python 3.12.3, numpy 2.4.4, Linux x86_64 | commit: 2f8d8a4 -->

| evictor | prefetcher | admission | model | dataset | capacity | hit_rate | covered | stall_ms_tok | provenance | commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lcnu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 256 | 0.1873 | 0.0000 | 228.86 | real | 2f8d8a4 |
| lcnu_p | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 256 | 0.1873 | 0.0000 | 228.87 | real | 2f8d8a4 |
| w_tinylfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 256 | 0.1533 | 0.0000 | 238.42 | real | 2f8d8a4 |
| lfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 256 | 0.1444 | 0.0000 | 240.94 | real | 2f8d8a4 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 256 | 0.1867 | 0.1858 | 211.99 | real | 2f8d8a4 |
| lcnu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 512 | 0.2997 | 0.0000 | 197.20 | real | 2f8d8a4 |
| lcnu_p | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 512 | 0.2999 | 0.0000 | 197.14 | real | 2f8d8a4 |
| w_tinylfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 512 | 0.2530 | 0.0000 | 210.36 | real | 2f8d8a4 |
| lfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 512 | 0.2362 | 0.0000 | 215.09 | real | 2f8d8a4 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 512 | 0.3001 | 0.1755 | 181.07 | real | 2f8d8a4 |
| lcnu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 1024 | 0.4408 | 0.0000 | 157.48 | real | 2f8d8a4 |
| lcnu_p | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 1024 | 0.4442 | 0.0000 | 156.51 | real | 2f8d8a4 |
| w_tinylfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 1024 | 0.3704 | 0.0000 | 177.30 | real | 2f8d8a4 |
| lfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 1024 | 0.3671 | 0.0000 | 178.22 | real | 2f8d8a4 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 1024 | 0.4461 | 0.1546 | 141.91 | real | 2f8d8a4 |
| lcnu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 2048 | 0.6052 | 0.0000 | 111.17 | real | 2f8d8a4 |
| lcnu_p | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 2048 | 0.6175 | 0.0000 | 107.72 | real | 2f8d8a4 |
| w_tinylfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 2048 | 0.5373 | 0.0000 | 130.29 | real | 2f8d8a4 |
| lfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 2048 | 0.5488 | 0.0000 | 127.05 | real | 2f8d8a4 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 2048 | 0.6195 | 0.1169 | 96.16 | real | 2f8d8a4 |
| lcnu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 4096 | 0.7988 | 0.0000 | 56.66 | real | 2f8d8a4 |
| lcnu_p | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 4096 | 0.8237 | 0.0000 | 49.63 | real | 2f8d8a4 |
| w_tinylfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 4096 | 0.7705 | 0.0000 | 64.62 | real | 2f8d8a4 |
| lfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 4096 | 0.7872 | 0.0000 | 59.91 | real | 2f8d8a4 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 4096 | 0.8242 | 0.0476 | 44.43 | real | 2f8d8a4 |
| lcnu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 6016 | 0.9047 | 0.0000 | 26.83 | real | 2f8d8a4 |
| lcnu_p | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 6016 | 0.9277 | 0.0000 | 20.36 | real | 2f8d8a4 |
| w_tinylfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 6016 | 0.9083 | 0.0000 | 25.82 | real | 2f8d8a4 |
| lfu | none | same_queue | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 6016 | 0.9149 | 0.0000 | 23.96 | real | 2f8d8a4 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | qwen3-235b-a22b-fp8 | mmlu100-greedy64/decode+prefill-union/held | 6016 | 0.9278 | 0.0120 | 18.80 | real | 2f8d8a4 |

<!-- run: python -m benchmarks.cache_simulator.research.run_prefetch_real --model qwen3-235b-a22b-fp8 --split held --prefill union --tag qwen3_235b_held_union_r3d | env: python 3.12.3, numpy 2.4.4, Linux x86_64 | commit: 2f8d8a4 -->
