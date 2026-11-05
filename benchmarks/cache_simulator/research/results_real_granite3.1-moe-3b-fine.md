# Real cache-policy benchmark — granite3.1-moe-3b / handcrafted8-greedy512/decode

All numbers are **REAL** — computed over an actual SGLang `per_token` expert-routing capture (not synthetic). Values are the demand **hit rate**; `belady` is the Belady-optimal ceiling ('Ideal').

Trace stats: 1 file(s), 32 layers, max expert id 39, top-k 8, 1262 distinct (layer, expert) keys, 638208 accesses (decode passes only; ignore_layers=0).

Commit: `b18303c`

| Policy | cap=280 | cap=320 | cap=360 | cap=400 | cap=440 | cap=480 | cap=520 | cap=560 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Belady (Ideal) | 0.6739 | 0.7154 | 0.7515 | 0.7832 | 0.8114 | 0.8367 | 0.8589 | 0.8785 |
| LRU | 0.4677 | 0.4691 | 0.4826 | 0.5244 | 0.5659 | 0.6113 | 0.6431 | 0.6848 |
| LFU | 0.4513 | 0.4956 | 0.5381 | 0.5774 | 0.6153 | 0.6507 | 0.6838 | 0.7155 |
| LFU-K | 0.4677 | 0.4691 | 0.4826 | 0.5244 | 0.5659 | 0.6113 | 0.6431 | 0.6848 |
| Model-tracing | 0.4553 | 0.4998 | 0.5421 | 0.5817 | 0.6196 | 0.6550 | 0.6883 | 0.7193 |
| Score-based | 0.4677 | 0.4691 | 0.4826 | 0.5244 | 0.5659 | 0.6113 | 0.6431 | 0.6848 |
| LHU | 0.4615 | 0.4715 | 0.4945 | 0.5370 | 0.5789 | 0.6210 | 0.6550 | 0.6962 |
| ARC | 0.3968 | 0.4553 | 0.4993 | 0.5395 | 0.5768 | 0.6153 | 0.6536 | 0.6893 |
| 2Q | 0.3539 | 0.4193 | 0.4810 | 0.5338 | 0.5820 | 0.6230 | 0.6604 | 0.6955 |
| LRU-2 | 0.4067 | 0.4600 | 0.5030 | 0.5518 | 0.5935 | 0.6369 | 0.6783 | 0.7158 |
| LIRS | 0.4319 | 0.4801 | 0.5236 | 0.5676 | 0.6093 | 0.6483 | 0.6862 | 0.7218 |
| S3-FIFO | 0.4049 | 0.4475 | 0.4914 | 0.5399 | 0.5895 | 0.6393 | 0.6856 | 0.7265 |
| SIEVE | 0.2819 | 0.3330 | 0.3747 | 0.4126 | 0.4570 | 0.5034 | 0.5457 | 0.6036 |
| W-TinyLFU | 0.4773 | 0.5250 | 0.5687 | 0.6099 | 0.6480 | 0.6839 | 0.7176 | 0.7490 |
| Per-layer LRU | 0.4002 | 0.4582 | 0.5010 | 0.5402 | 0.5785 | 0.6184 | 0.6563 | 0.6916 |
| LCNU (layer-cycle next-use) | 0.4982 | 0.5449 | 0.5883 | 0.6291 | 0.6672 | 0.7024 | 0.7352 | 0.7654 |
| 1-layer-ahead* | 0.4677 | 0.4689 | 0.4801 | 0.5204 | 0.5609 | 0.6088 | 0.6399 | 0.6818 |
| Stride* | 0.4677 | 0.4690 | 0.4810 | 0.5219 | 0.5626 | 0.6097 | 0.6410 | 0.6828 |
| Multi-layer* | 0.4677 | 0.4688 | 0.4784 | 0.5172 | 0.5574 | 0.6068 | 0.6376 | 0.6797 |
| Activation-matrix* | 0.0008 | 0.0013 | 0.1102 | 0.4408 | 0.4691 | 0.4766 | 0.5178 | 0.5814 |

`*` = trace-oracle prefetch policy (perfect next-layer routing lookahead); reported demand hit rate is an upper bound on an achievable predictor. See the CSV for `effective_hit_rate`.
