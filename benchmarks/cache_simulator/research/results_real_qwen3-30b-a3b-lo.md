# Real cache-policy benchmark — qwen3-30b-a3b / handcrafted4-lo-greedy512/decode

All numbers are **REAL** — computed over an actual SGLang `per_token` expert-routing capture (not synthetic). Values are the demand **hit rate**; `belady` is the Belady-optimal ceiling ('Ideal').

Trace stats: 1 file(s), 48 layers, max expert id 127, top-k 8, 5842 distinct (layer, expert) keys, 638208 accesses (decode passes only; ignore_layers=0).

Commit: `48ca80c+dirty`

| Policy | cap=896 | cap=1024 | cap=1152 | cap=1280 | cap=1408 | cap=1536 | cap=1664 | cap=1792 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Belady (Ideal) | 0.7779 | 0.8077 | 0.8330 | 0.8547 | 0.8735 | 0.8894 | 0.9034 | 0.9154 |
| LRU | 0.5626 | 0.6036 | 0.6404 | 0.6787 | 0.7123 | 0.7477 | 0.7765 | 0.8035 |
| LFU | 0.3939 | 0.4357 | 0.4743 | 0.5108 | 0.5459 | 0.5788 | 0.6098 | 0.6390 |
| LFU-K | 0.5626 | 0.6036 | 0.6404 | 0.6787 | 0.7123 | 0.7477 | 0.7765 | 0.8035 |
| Model-tracing | 0.4126 | 0.4544 | 0.4930 | 0.5309 | 0.5653 | 0.5983 | 0.6294 | 0.6593 |
| Score-based | 0.5626 | 0.6036 | 0.6404 | 0.6787 | 0.7123 | 0.7477 | 0.7765 | 0.8035 |
| LHU | 0.5287 | 0.5696 | 0.6036 | 0.6333 | 0.6613 | 0.6859 | 0.7096 | 0.7296 |
| ARC | 0.5633 | 0.6066 | 0.6447 | 0.6836 | 0.7182 | 0.7495 | 0.7789 | 0.8065 |
| 2Q | 0.5655 | 0.6082 | 0.6443 | 0.6814 | 0.7133 | 0.7435 | 0.7743 | 0.7996 |
| LRU-2 | 0.5570 | 0.6031 | 0.6443 | 0.6835 | 0.7177 | 0.7505 | 0.7790 | 0.8050 |
| LIRS | 0.5601 | 0.6051 | 0.6444 | 0.6815 | 0.7168 | 0.7472 | 0.7757 | 0.8016 |
| S3-FIFO | 0.5599 | 0.6118 | 0.6567 | 0.6975 | 0.7323 | 0.7633 | 0.7909 | 0.8147 |
| SIEVE | 0.2520 | 0.2839 | 0.3361 | 0.3716 | 0.4049 | 0.4429 | 0.4752 | 0.5136 |
| W-TinyLFU | 0.5473 | 0.5879 | 0.6228 | 0.6554 | 0.6866 | 0.7112 | 0.7380 | 0.7623 |
| Per-layer LRU | 0.5653 | 0.6061 | 0.6441 | 0.6787 | 0.7119 | 0.7427 | 0.7710 | 0.7972 |
| LCNU (layer-cycle next-use) | 0.5908 | 0.6245 | 0.6560 | 0.6861 | 0.7144 | 0.7401 | 0.7636 | 0.7853 |
| 1-layer-ahead* | 0.5609 | 0.6027 | 0.6396 | 0.6779 | 0.7115 | 0.7470 | 0.7760 | 0.8030 |
| Stride* | 0.5615 | 0.6030 | 0.6398 | 0.6782 | 0.7118 | 0.7473 | 0.7761 | 0.8032 |
| Multi-layer* | 0.5597 | 0.6020 | 0.6390 | 0.6772 | 0.7110 | 0.7466 | 0.7755 | 0.8027 |
| Activation-matrix* | 0.3948 | 0.4357 | 0.4781 | 0.5211 | 0.5647 | 0.6043 | 0.6409 | 0.6764 |

`*` = trace-oracle prefetch policy (perfect next-layer routing lookahead); reported demand hit rate is an upper bound on an achievable predictor. See the CSV for `effective_hit_rate`.
