# Synthetic cache-policy benchmark results

All numbers are **SYNTHETIC** -- generated from seeded synthetic MoE routing traces (no real Qwen3-235B captures are available on this machine). Values are the demand **hit rate** (fraction of accesses served from the resident cache). `belady` is the Belady-optimal demand-paging ceiling ('Ideal'). Policies marked `*` are trace-oracle prefetch variants: their demand hit rate stays bounded by Belady, and their prefetch value shows up in `effective_hit_rate` (see the CSV), which is an **upper bound** on the underlying predictor.

Commit: `b0aff90`


## qwen3-235b-synthetic

Geometry: 94 layers x 128 experts, top-8, 120 decode steps, skew=1.1, reuse=0.35, seed=2026. Trace length = 90240 accesses.

| Policy | cap=256 | cap=512 | cap=1024 | cap=2048 | cap=4096 | cap=6016 |
| --- | --- | --- | --- | --- | --- | --- |
| Belady (Ideal) | 0.3341 | 0.5828 | 0.7181 | 0.8085 | 0.8694 | 0.8888 |
| LRU | 0.0000 | 0.0000 | 0.5332 | 0.6718 | 0.7884 | 0.8448 |
| LFU | 0.2251 | 0.3536 | 0.4968 | 0.6382 | 0.7768 | 0.8420 |
| LFU-K | 0.0000 | 0.0000 | 0.5332 | 0.6718 | 0.7884 | 0.8448 |
| Model-tracing | 0.2434 | 0.3731 | 0.5230 | 0.6770 | 0.8176 | 0.8708 |
| Score-based | 0.0000 | 0.0000 | 0.5332 | 0.6718 | 0.7884 | 0.8448 |
| LHU | 0.2249 | 0.3536 | 0.4968 | 0.6382 | 0.7768 | 0.8420 |
| 1-layer-ahead* | 0.0000 | 0.0000 | 0.5332 | 0.6718 | 0.7884 | 0.8448 |
| Stride* | 0.0000 | 0.0000 | 0.5332 | 0.6718 | 0.7884 | 0.8448 |
| Multi-layer* | 0.0000 | 0.0000 | 0.5331 | 0.6717 | 0.7883 | 0.8448 |
| Activation-matrix* | 0.0000 | 0.0000 | 0.0593 | 0.6254 | 0.7683 | 0.8382 |

## mixtral-8x7b-synthetic

Geometry: 32 layers x 8 experts, top-2, 400 decode steps, skew=0.8, reuse=0.4, seed=11. Trace length = 25600 accesses.

| Policy | cap=16 | cap=32 | cap=64 | cap=96 | cap=128 | cap=192 | cap=256 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Belady (Ideal) | 0.2327 | 0.4669 | 0.7213 | 0.8220 | 0.8830 | 0.9561 | 0.9900 |
| LRU | 0.0000 | 0.0000 | 0.5679 | 0.6582 | 0.7352 | 0.8783 | 0.9900 |
| LFU | 0.1250 | 0.2582 | 0.4374 | 0.5702 | 0.6830 | 0.8588 | 0.9900 |
| LFU-K | 0.0000 | 0.0000 | 0.5679 | 0.6582 | 0.7352 | 0.8783 | 0.9900 |
| Model-tracing | 0.1303 | 0.2669 | 0.4487 | 0.5842 | 0.6964 | 0.8700 | 0.9900 |
| Score-based | 0.0000 | 0.0000 | 0.5679 | 0.6582 | 0.7352 | 0.8783 | 0.9900 |
| LHU | 0.0323 | 0.1255 | 0.4688 | 0.5866 | 0.6901 | 0.8597 | 0.9900 |
| 1-layer-ahead* | 0.0000 | 0.0000 | 0.3695 | 0.6566 | 0.7342 | 0.8780 | 0.9900 |
| Stride* | 0.0000 | 0.0000 | 0.0373 | 0.6488 | 0.7310 | 0.8768 | 0.9900 |
| Multi-layer* | 0.0000 | 0.0000 | 0.1644 | 0.6543 | 0.7327 | 0.8777 | 0.9900 |
| Activation-matrix* | 0.0000 | 0.0000 | 0.0012 | 0.0074 | 0.0486 | 0.7347 | 0.9900 |

## uniform-stress-synthetic

Geometry: 16 layers x 32 experts, top-4, 300 decode steps, skew=0.0, reuse=0.1, seed=7. Trace length = 19200 accesses.

| Policy | cap=16 | cap=32 | cap=64 | cap=128 | cap=256 | cap=384 |
| --- | --- | --- | --- | --- | --- | --- |
| Belady (Ideal) | 0.1827 | 0.2973 | 0.4398 | 0.6066 | 0.7995 | 0.9119 |
| LRU | 0.0000 | 0.0000 | 0.1801 | 0.2952 | 0.5096 | 0.7506 |
| LFU | 0.0286 | 0.0606 | 0.1237 | 0.2506 | 0.5055 | 0.7476 |
| LFU-K | 0.0000 | 0.0000 | 0.1801 | 0.2952 | 0.5096 | 0.7506 |
| Model-tracing | 0.0406 | 0.0802 | 0.1582 | 0.3026 | 0.5604 | 0.7910 |
| Score-based | 0.0000 | 0.0000 | 0.1801 | 0.2952 | 0.5096 | 0.7506 |
| LHU | 0.0286 | 0.0606 | 0.1237 | 0.2506 | 0.5055 | 0.7476 |
| 1-layer-ahead* | 0.0000 | 0.0000 | 0.0339 | 0.2949 | 0.5071 | 0.7489 |
| Stride* | 0.0000 | 0.0000 | 0.0029 | 0.2928 | 0.5066 | 0.7479 |
| Multi-layer* | 0.0000 | 0.0000 | 0.0027 | 0.2895 | 0.5060 | 0.7472 |
| Activation-matrix* | 0.0000 | 0.0000 | 0.0004 | 0.0071 | 0.2249 | 0.5021 |

`*` = trace-oracle prefetch policy (perfect next-layer routing lookahead); reported demand hit rate is an upper bound on the achievable predictor. See `results_synthetic.csv` for `effective_hit_rate` (prefetch-coverage upper bound).
