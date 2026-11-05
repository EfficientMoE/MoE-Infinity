# Real cache-policy benchmark — qwen3-30b-a3b / handcrafted8-greedy512/decode

All numbers are **REAL** — computed over an actual SGLang `per_token` expert-routing capture (not synthetic). Values are the demand **hit rate**; `belady` is the Belady-optimal ceiling ('Ideal').

Trace stats: 1 file(s), 48 layers, max expert id 127, top-k 8, 5948 distinct (layer, expert) keys, 1424640 accesses (decode passes only; ignore_layers=0).

Commit: `224034e`

| Policy | cap=896 | cap=1024 | cap=1152 | cap=1280 | cap=1408 | cap=1536 | cap=1664 | cap=1792 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Belady (Ideal) | 0.8033 | 0.8327 | 0.8577 | 0.8788 | 0.8970 | 0.9124 | 0.9253 | 0.9362 |
| LRU | 0.5990 | 0.6413 | 0.6808 | 0.7195 | 0.7544 | 0.7877 | 0.8172 | 0.8449 |
| LFU | 0.4233 | 0.4670 | 0.5079 | 0.5462 | 0.5827 | 0.6164 | 0.6483 | 0.6781 |
| LFU-K | 0.5990 | 0.6413 | 0.6808 | 0.7195 | 0.7544 | 0.7877 | 0.8172 | 0.8449 |
| Model-tracing | 0.4353 | 0.4807 | 0.5227 | 0.5615 | 0.5985 | 0.6320 | 0.6641 | 0.6938 |
| Score-based | 0.5990 | 0.6413 | 0.6808 | 0.7195 | 0.7544 | 0.7877 | 0.8172 | 0.8449 |
| LHU | 0.5832 | 0.6254 | 0.6633 | 0.6991 | 0.7338 | 0.7634 | 0.7896 | 0.8121 |
| ARC | 0.5986 | 0.6418 | 0.6829 | 0.7221 | 0.7585 | 0.7915 | 0.8220 | 0.8482 |
| 2Q | 0.6018 | 0.6474 | 0.6869 | 0.7238 | 0.7564 | 0.7865 | 0.8154 | 0.8401 |
| LRU-2 | 0.5953 | 0.6427 | 0.6866 | 0.7282 | 0.7646 | 0.7970 | 0.8252 | 0.8489 |
| LIRS | 0.5976 | 0.6453 | 0.6874 | 0.7270 | 0.7633 | 0.7947 | 0.8226 | 0.8470 |
| S3-FIFO | 0.6026 | 0.6547 | 0.7005 | 0.7420 | 0.7773 | 0.8083 | 0.8354 | 0.8581 |
| SIEVE | 0.2167 | 0.2514 | 0.3075 | 0.3472 | 0.3818 | 0.4235 | 0.4607 | 0.5044 |
| W-TinyLFU | 0.5918 | 0.6330 | 0.6714 | 0.7059 | 0.7360 | 0.7629 | 0.7879 | 0.8091 |
| Per-layer LRU | 0.6010 | 0.6430 | 0.6821 | 0.7174 | 0.7513 | 0.7819 | 0.8095 | 0.8348 |
| LCNU (layer-cycle next-use) | 0.6318 | 0.6685 | 0.7018 | 0.7328 | 0.7614 | 0.7871 | 0.8100 | 0.8301 |
| 1-layer-ahead* | 0.5975 | 0.6404 | 0.6799 | 0.7187 | 0.7537 | 0.7871 | 0.8167 | 0.8445 |
| Stride* | 0.5981 | 0.6407 | 0.6802 | 0.7190 | 0.7539 | 0.7873 | 0.8169 | 0.8447 |
| Multi-layer* | 0.5965 | 0.6397 | 0.6792 | 0.7180 | 0.7532 | 0.7867 | 0.8164 | 0.8443 |
| Activation-matrix* | 0.4133 | 0.4601 | 0.5076 | 0.5502 | 0.5940 | 0.6353 | 0.6752 | 0.7138 |

`*` = trace-oracle prefetch policy (perfect next-layer routing lookahead); reported demand hit rate is an upper bound on an achievable predictor. See the CSV for `effective_hit_rate`.
