# Real cache-policy benchmark — granite3.1-moe-3b / handcrafted8-greedy512/decode

All numbers are **REAL** — computed over an actual llama.cpp eval-callback expert-routing capture (granite3.1-moe:3b, converted to the SGLang `per_token` record format; see `capture_report.md`). Values are the demand **hit rate**; `belady` is the Belady-optimal ceiling ('Ideal').

Trace stats: 1 file(s), 32 layers, max expert id 39, top-k 8, 1262 distinct (layer, expert) keys, 638208 accesses (decode passes only; ignore_layers=0).

Commit: `b2d1e8e`

| Policy | cap=32 | cap=64 | cap=128 | cap=256 | cap=512 | cap=768 | cap=1024 | cap=1280 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Belady (Ideal) | 0.1205 | 0.2360 | 0.4219 | 0.6458 | 0.8547 | 0.9501 | 0.9897 | 0.9980 |
| LRU | 0.0000 | 0.0000 | 0.0000 | 0.3503 | 0.6351 | 0.8530 | 0.9698 | 0.9980 |
| LFU | 0.0903 | 0.1545 | 0.2578 | 0.4235 | 0.6772 | 0.8518 | 0.9607 | 0.9980 |
| LFU-K | 0.0000 | 0.0000 | 0.0000 | 0.3503 | 0.6351 | 0.8530 | 0.9698 | 0.9980 |
| Model-tracing | 0.0913 | 0.1560 | 0.2605 | 0.4269 | 0.6813 | 0.8550 | 0.9631 | 0.9980 |
| Score-based | 0.0000 | 0.0000 | 0.0000 | 0.3503 | 0.6351 | 0.8530 | 0.9698 | 0.9980 |
| LHU | 0.0028 | 0.0072 | 0.0177 | 0.4149 | 0.6473 | 0.8579 | 0.9691 | 0.9980 |
| ARC | 0.0000 | 0.0000 | 0.0000 | 0.3429 | 0.6461 | 0.8532 | 0.9700 | 0.9980 |
| 2Q | 0.0000 | 0.0000 | 0.0000 | 0.3151 | 0.6521 | 0.8501 | 0.9635 | 0.9980 |
| LRU-2 | 0.0000 | 0.0121 | 0.1077 | 0.3741 | 0.6700 | 0.8705 | 0.9719 | 0.9980 |
| LIRS | 0.0694 | 0.1284 | 0.2347 | 0.4021 | 0.6788 | 0.8690 | 0.9713 | 0.9980 |
| S3-FIFO | 0.0000 | 0.0000 | 0.0000 | 0.3789 | 0.6765 | 0.8793 | 0.9732 | 0.9980 |
| SIEVE | 0.0000 | 0.0000 | 0.0000 | 0.2385 | 0.5348 | 0.7855 | 0.9291 | 0.9980 |
| W-TinyLFU | 0.0831 | 0.1496 | 0.2632 | 0.4468 | 0.7110 | 0.8762 | 0.9701 | 0.9980 |
| Per-layer LRU | 0.0033 | 0.0107 | 0.0433 | 0.3503 | 0.6489 | 0.8483 | 0.9588 | 0.9980 |
| LCNU (layer-cycle next-use) | 0.0957 | 0.1674 | 0.2838 | 0.4682 | 0.7287 | 0.8864 | 0.9708 | 0.9980 |
| 1-layer-ahead* | 0.0000 | 0.0000 | 0.0000 | 0.0240 | 0.6317 | 0.8518 | 0.9697 | 0.9980 |
| Stride* | 0.0000 | 0.0000 | 0.0000 | 0.0920 | 0.6328 | 0.8522 | 0.9697 | 0.9980 |
| Multi-layer* | 0.0000 | 0.0000 | 0.0000 | 0.0022 | 0.6297 | 0.8508 | 0.9696 | 0.9980 |
| Activation-matrix* | 0.0000 | 0.0000 | 0.0000 | 0.0004 | 0.5052 | 0.7978 | 0.9627 | 0.9980 |

`*` = trace-oracle prefetch policy (perfect next-layer routing lookahead); reported demand hit rate is an upper bound on an achievable predictor. See the CSV for `effective_hit_rate`.
