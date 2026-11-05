# Real cache-policy benchmark — qwen3-30b-a3b / handcrafted8-greedy64x8/decode

All numbers are **REAL** — computed over an actual SGLang `per_token` expert-routing capture (not synthetic). Values are the demand **hit rate**; `belady` is the Belady-optimal ceiling ('Ideal').

Trace stats: 1 file(s), 48 layers, max expert id 127, top-k 8, 5163 distinct (layer, expert) keys, 196608 accesses (decode passes only; ignore_layers=0).

Commit: `4f3cc78`

| Policy | cap=896 | cap=1024 | cap=1152 | cap=1280 | cap=1408 | cap=1536 | cap=1664 | cap=1792 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Belady (Ideal) | 0.7963 | 0.8234 | 0.8466 | 0.8671 | 0.8848 | 0.8990 | 0.9117 | 0.9221 |
| LRU | 0.6029 | 0.6456 | 0.6796 | 0.7123 | 0.7422 | 0.7721 | 0.7990 | 0.8274 |
| LFU | 0.4505 | 0.4966 | 0.5390 | 0.5798 | 0.6197 | 0.6553 | 0.6889 | 0.7210 |
| LFU-K | 0.6029 | 0.6456 | 0.6796 | 0.7123 | 0.7422 | 0.7721 | 0.7990 | 0.8274 |
| Model-tracing | 0.4830 | 0.5287 | 0.5742 | 0.6146 | 0.6544 | 0.6897 | 0.7254 | 0.7544 |
| Score-based | 0.6029 | 0.6456 | 0.6796 | 0.7123 | 0.7422 | 0.7721 | 0.7990 | 0.8274 |
| LHU | 0.4680 | 0.5101 | 0.5474 | 0.5837 | 0.6207 | 0.6553 | 0.6889 | 0.7210 |
| ARC | 0.6040 | 0.6356 | 0.6691 | 0.7037 | 0.7392 | 0.7710 | 0.8031 | 0.8288 |
| 2Q | 0.5900 | 0.6337 | 0.6699 | 0.7080 | 0.7371 | 0.7678 | 0.7978 | 0.8183 |
| LRU-2 | 0.5819 | 0.6219 | 0.6614 | 0.7026 | 0.7383 | 0.7687 | 0.7992 | 0.8237 |
| LIRS | 0.5813 | 0.6247 | 0.6628 | 0.6991 | 0.7343 | 0.7677 | 0.7956 | 0.8209 |
| S3-FIFO | 0.5870 | 0.6365 | 0.6779 | 0.7171 | 0.7523 | 0.7821 | 0.8088 | 0.8331 |
| SIEVE | 0.2782 | 0.3131 | 0.3698 | 0.4123 | 0.4487 | 0.4929 | 0.5292 | 0.5690 |
| W-TinyLFU | 0.5212 | 0.5540 | 0.5891 | 0.6220 | 0.6578 | 0.6878 | 0.7176 | 0.7424 |
| Per-layer LRU | 0.6009 | 0.6426 | 0.6786 | 0.7092 | 0.7382 | 0.7663 | 0.7920 | 0.8157 |
| LCNU (layer-cycle next-use) | 0.6257 | 0.6608 | 0.6910 | 0.7203 | 0.7482 | 0.7731 | 0.7967 | 0.8195 |
| 1-layer-ahead* | 0.6014 | 0.6449 | 0.6790 | 0.7116 | 0.7416 | 0.7716 | 0.7985 | 0.8270 |
| Stride* | 0.6019 | 0.6451 | 0.6792 | 0.7118 | 0.7418 | 0.7717 | 0.7986 | 0.8272 |
| Multi-layer* | 0.6002 | 0.6443 | 0.6785 | 0.7109 | 0.7412 | 0.7712 | 0.7981 | 0.8267 |
| Activation-matrix* | 0.4422 | 0.4775 | 0.5257 | 0.5731 | 0.6172 | 0.6554 | 0.6878 | 0.7258 |

`*` = trace-oracle prefetch policy (perfect next-layer routing lookahead); reported demand hit rate is an upper bound on an achievable predictor. See the CSV for `effective_hit_rate`.
