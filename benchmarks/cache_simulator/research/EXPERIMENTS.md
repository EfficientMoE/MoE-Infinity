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
