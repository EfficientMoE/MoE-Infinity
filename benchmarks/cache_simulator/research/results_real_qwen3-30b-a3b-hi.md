# Real cache-policy benchmark — qwen3-30b-a3b / handcrafted4-hi-greedy512/decode

All numbers are **REAL** — computed over an actual SGLang `per_token` expert-routing capture (not synthetic). Values are the demand **hit rate**; `belady` is the Belady-optimal ceiling ('Ideal').

Trace stats: 1 file(s), 48 layers, max expert id 127, top-k 8, 5097 distinct (layer, expert) keys, 786432 accesses (decode passes only; ignore_layers=0).

Commit: `48ca80c`

| Policy | cap=896 | cap=1024 | cap=1152 | cap=1280 | cap=1408 | cap=1536 | cap=1664 | cap=1792 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Belady (Ideal) | 0.8235 | 0.8525 | 0.8770 | 0.8977 | 0.9151 | 0.9298 | 0.9418 | 0.9517 |
| LRU | 0.6285 | 0.6718 | 0.7136 | 0.7524 | 0.7885 | 0.8202 | 0.8501 | 0.8784 |
| LFU | 0.5154 | 0.5648 | 0.6106 | 0.6529 | 0.6929 | 0.7302 | 0.7641 | 0.7942 |
| LFU-K | 0.6285 | 0.6718 | 0.7136 | 0.7524 | 0.7885 | 0.8202 | 0.8501 | 0.8784 |
| Model-tracing | 0.5313 | 0.5815 | 0.6274 | 0.6710 | 0.7120 | 0.7497 | 0.7837 | 0.8141 |
| Score-based | 0.6285 | 0.6718 | 0.7136 | 0.7524 | 0.7885 | 0.8202 | 0.8501 | 0.8784 |
| LHU | 0.6135 | 0.6570 | 0.6984 | 0.7382 | 0.7724 | 0.8031 | 0.8299 | 0.8510 |
| ARC | 0.6264 | 0.6709 | 0.7151 | 0.7551 | 0.7928 | 0.8269 | 0.8574 | 0.8819 |
| 2Q | 0.6324 | 0.6803 | 0.7218 | 0.7583 | 0.7916 | 0.8215 | 0.8479 | 0.8718 |
| LRU-2 | 0.6278 | 0.6765 | 0.7225 | 0.7656 | 0.8036 | 0.8357 | 0.8638 | 0.8851 |
| LIRS | 0.6293 | 0.6793 | 0.7237 | 0.7653 | 0.8022 | 0.8340 | 0.8613 | 0.8844 |
| S3-FIFO | 0.6381 | 0.6904 | 0.7372 | 0.7787 | 0.8145 | 0.8461 | 0.8721 | 0.8935 |
| SIEVE | 0.3056 | 0.3612 | 0.4062 | 0.4528 | 0.5079 | 0.5456 | 0.5938 | 0.6281 |
| W-TinyLFU | 0.6341 | 0.6769 | 0.7165 | 0.7532 | 0.7861 | 0.8127 | 0.8390 | 0.8630 |
| Per-layer LRU | 0.6300 | 0.6730 | 0.7129 | 0.7488 | 0.7831 | 0.8135 | 0.8405 | 0.8651 |
| LCNU (layer-cycle next-use) | 0.6656 | 0.7041 | 0.7385 | 0.7707 | 0.8004 | 0.8282 | 0.8534 | 0.8752 |
| 1-layer-ahead* | 0.6272 | 0.6709 | 0.7125 | 0.7516 | 0.7877 | 0.8196 | 0.8498 | 0.8782 |
| Stride* | 0.6277 | 0.6712 | 0.7129 | 0.7519 | 0.7880 | 0.8198 | 0.8499 | 0.8783 |
| Multi-layer* | 0.6262 | 0.6702 | 0.7117 | 0.7511 | 0.7873 | 0.8192 | 0.8495 | 0.8779 |
| Activation-matrix* | 0.4690 | 0.5257 | 0.5792 | 0.6271 | 0.6734 | 0.7155 | 0.7606 | 0.7982 |

`*` = trace-oracle prefetch policy (perfect next-layer routing lookahead); reported demand hit rate is an upper bound on an achievable predictor. See the CSV for `effective_hit_rate`.
