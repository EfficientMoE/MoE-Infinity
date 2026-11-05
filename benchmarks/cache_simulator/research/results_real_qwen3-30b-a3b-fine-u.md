# Real cache-policy benchmark — qwen3-30b-a3b / handcrafted8-greedy512/decode+prefill-union

All numbers are **REAL** — computed over an actual SGLang `per_token` expert-routing capture (not synthetic). Values are the demand **hit rate**; `belady` is the Belady-optimal ceiling ('Ideal').

Trace stats: 1 file(s), 48 layers, max expert id 127, top-k 8, 5982 distinct (layer, expert) keys, 1447031 accesses (decode + prefill as one fetch per distinct expert per layer; ignore_layers=0).

Commit: `b094474`

| Policy | cap=896 | cap=1024 | cap=1152 | cap=1280 | cap=1408 | cap=1536 | cap=1664 | cap=1792 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Belady (Ideal) | 0.7966 | 0.8263 | 0.8516 | 0.8731 | 0.8916 | 0.9073 | 0.9207 | 0.9320 |
| LRU | 0.5902 | 0.6320 | 0.6711 | 0.7095 | 0.7444 | 0.7776 | 0.8070 | 0.8346 |
| LFU | 0.4199 | 0.4633 | 0.5041 | 0.5424 | 0.5787 | 0.6125 | 0.6443 | 0.6743 |
| LFU-K | 0.5902 | 0.6320 | 0.6711 | 0.7095 | 0.7444 | 0.7776 | 0.8070 | 0.8346 |
| Model-tracing | 0.4306 | 0.4754 | 0.5175 | 0.5569 | 0.5930 | 0.6267 | 0.6589 | 0.6886 |
| Score-based | 0.5902 | 0.6320 | 0.6711 | 0.7095 | 0.7444 | 0.7776 | 0.8070 | 0.8346 |
| LHU | 0.5761 | 0.6183 | 0.6563 | 0.6918 | 0.7267 | 0.7564 | 0.7829 | 0.8060 |
| ARC | 0.5917 | 0.6352 | 0.6767 | 0.7153 | 0.7521 | 0.7847 | 0.8141 | 0.8402 |
| 2Q | 0.5948 | 0.6399 | 0.6794 | 0.7161 | 0.7487 | 0.7790 | 0.8076 | 0.8320 |
| LRU-2 | 0.5900 | 0.6372 | 0.6808 | 0.7222 | 0.7584 | 0.7910 | 0.8198 | 0.8437 |
| LIRS | 0.5921 | 0.6397 | 0.6818 | 0.7212 | 0.7575 | 0.7890 | 0.8172 | 0.8420 |
| S3-FIFO | 0.5960 | 0.6476 | 0.6927 | 0.7342 | 0.7701 | 0.8012 | 0.8278 | 0.8507 |
| SIEVE | 0.2470 | 0.2566 | 0.2973 | 0.3409 | 0.3809 | 0.4167 | 0.4608 | 0.4987 |
| W-TinyLFU | 0.5869 | 0.6286 | 0.6655 | 0.6990 | 0.7294 | 0.7580 | 0.7837 | 0.8040 |
| Per-layer LRU | 0.5922 | 0.6336 | 0.6723 | 0.7073 | 0.7409 | 0.7714 | 0.7989 | 0.8243 |
| LCNU (layer-cycle next-use) | 0.6253 | 0.6626 | 0.6971 | 0.7296 | 0.7595 | 0.7862 | 0.8101 | 0.8314 |
| 1-layer-ahead* | 0.5886 | 0.6310 | 0.6701 | 0.7086 | 0.7435 | 0.7769 | 0.8064 | 0.8342 |
| Stride* | 0.5892 | 0.6314 | 0.6705 | 0.7090 | 0.7439 | 0.7772 | 0.8067 | 0.8344 |
| Multi-layer* | 0.5875 | 0.6302 | 0.6694 | 0.7079 | 0.7429 | 0.7764 | 0.8059 | 0.8338 |
| Activation-matrix* | 0.4062 | 0.4520 | 0.5002 | 0.5422 | 0.5861 | 0.6269 | 0.6667 | 0.7050 |

`*` = trace-oracle prefetch policy (perfect next-layer routing lookahead); reported demand hit rate is an upper bound on an achievable predictor. See the CSV for `effective_hit_rate`.
