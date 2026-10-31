# Real cache-policy benchmark — qwen1.5-moe-a2.7b / mixed20-greedy64

All numbers are **REAL** — computed over an actual SGLang `per_token` expert-routing capture (not synthetic). Values are the demand **hit rate**; `belady` is the Belady-optimal ceiling ('Ideal').

Trace stats: 1 file(s), 24 layers, max expert id 59, 1440 distinct (layer, expert) keys, 83328 decode accesses (ignore_layers=0).

Commit: `34717d3`

| Policy | cap=64 | cap=128 | cap=256 | cap=512 | cap=768 | cap=1024 |
| --- | --- | --- | --- | --- | --- | --- |
| Belady (Ideal) | 0.3012 | 0.4239 | 0.5776 | 0.7573 | 0.8639 | 0.9300 |
| LRU | 0.0382 | 0.1699 | 0.2625 | 0.4722 | 0.6525 | 0.8089 |
| LFU | 0.0710 | 0.1340 | 0.2481 | 0.4520 | 0.6324 | 0.7899 |
| LFU-K | 0.0382 | 0.1699 | 0.2625 | 0.4722 | 0.6525 | 0.8089 |
| Model-tracing | 0.0793 | 0.1478 | 0.2693 | 0.4813 | 0.6605 | 0.8122 |
| Score-based | 0.0382 | 0.1699 | 0.2625 | 0.4722 | 0.6525 | 0.8089 |
| LHU | 0.0731 | 0.1367 | 0.2509 | 0.4566 | 0.6325 | 0.7899 |
| 1-layer-ahead* | 0.0382 | 0.1694 | 0.2587 | 0.4687 | 0.6498 | 0.8077 |
| Stride* | 0.0382 | 0.1698 | 0.2579 | 0.4694 | 0.6508 | 0.8082 |
| Multi-layer* | 0.0310 | 0.1693 | 0.2559 | 0.4662 | 0.6481 | 0.8068 |
| Activation-matrix* | 0.0375 | 0.0384 | 0.0626 | 0.2457 | 0.4023 | 0.6264 |

`*` = trace-oracle prefetch policy (perfect next-layer routing lookahead); reported demand hit rate is an upper bound on an achievable predictor. See the CSV for `effective_hit_rate`.
