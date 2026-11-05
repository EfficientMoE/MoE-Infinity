# Real cache-policy benchmark — granite3.1-moe-3b / handcrafted8-greedy512/decode+prefill-union

All numbers are **REAL** — computed over an actual llama.cpp eval-callback expert-routing capture (granite3.1-moe:3b, converted to the SGLang `per_token` record format; see `capture_report.md`). Values are the demand **hit rate**; `belady` is the Belady-optimal ceiling ('Ideal').

Trace stats: 1 file(s), 32 layers, max expert id 39, top-k 8, 1268 distinct (layer, expert) keys, 647312 accesses (decode + prefill as one fetch per distinct expert per layer; ignore_layers=0).

Commit: `b2d1e8e+dirty`

| Policy | cap=32 | cap=64 | cap=128 | cap=256 | cap=512 | cap=768 | cap=1024 | cap=1280 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Belady (Ideal) | 0.1192 | 0.2335 | 0.4177 | 0.6404 | 0.8495 | 0.9469 | 0.9885 | 0.9980 |
| LRU | 0.0000 | 0.0000 | 0.0000 | 0.3443 | 0.6263 | 0.8438 | 0.9634 | 0.9980 |
| LFU | 0.0894 | 0.1530 | 0.2555 | 0.4203 | 0.6733 | 0.8483 | 0.9587 | 0.9980 |
| LFU-K | 0.0000 | 0.0000 | 0.0000 | 0.3443 | 0.6263 | 0.8438 | 0.9634 | 0.9980 |
| Model-tracing | 0.0904 | 0.1545 | 0.2581 | 0.4233 | 0.6773 | 0.8514 | 0.9608 | 0.9980 |
| Score-based | 0.0000 | 0.0000 | 0.0000 | 0.3443 | 0.6263 | 0.8438 | 0.9634 | 0.9980 |
| LHU | 0.0027 | 0.0070 | 0.0172 | 0.4042 | 0.6386 | 0.8500 | 0.9647 | 0.9980 |
| ARC | 0.0000 | 0.0000 | 0.0000 | 0.3389 | 0.6382 | 0.8439 | 0.9635 | 0.9980 |
| 2Q | 0.0000 | 0.0000 | 0.0000 | 0.3124 | 0.6457 | 0.8415 | 0.9589 | 0.9980 |
| LRU-2 | 0.0000 | 0.0118 | 0.1061 | 0.3713 | 0.6651 | 0.8657 | 0.9699 | 0.9980 |
| LIRS | 0.0686 | 0.1270 | 0.2323 | 0.3988 | 0.6743 | 0.8640 | 0.9691 | 0.9980 |
| S3-FIFO | 0.0000 | 0.0000 | 0.0000 | 0.3754 | 0.6698 | 0.8728 | 0.9699 | 0.9980 |
| SIEVE | 0.0000 | 0.0000 | 0.0000 | 0.2374 | 0.5250 | 0.7781 | 0.9294 | 0.9980 |
| W-TinyLFU | 0.0820 | 0.1484 | 0.2606 | 0.4429 | 0.7064 | 0.8711 | 0.9664 | 0.9980 |
| Per-layer LRU | 0.0032 | 0.0106 | 0.0427 | 0.3449 | 0.6403 | 0.8390 | 0.9529 | 0.9980 |
| LCNU (layer-cycle next-use) | 0.0946 | 0.1656 | 0.2810 | 0.4641 | 0.7230 | 0.8809 | 0.9675 | 0.9980 |
| 1-layer-ahead* | 0.0000 | 0.0000 | 0.0000 | 0.0236 | 0.6227 | 0.8422 | 0.9628 | 0.9980 |
| Stride* | 0.0000 | 0.0000 | 0.0000 | 0.0904 | 0.6240 | 0.8430 | 0.9633 | 0.9980 |
| Multi-layer* | 0.0000 | 0.0000 | 0.0000 | 0.0022 | 0.6206 | 0.8410 | 0.9624 | 0.9980 |
| Activation-matrix* | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4976 | 0.7883 | 0.9552 | 0.9980 |

`*` = trace-oracle prefetch policy (perfect next-layer routing lookahead); reported demand hit rate is an upper bound on an achievable predictor. See the CSV for `effective_hit_rate`.
