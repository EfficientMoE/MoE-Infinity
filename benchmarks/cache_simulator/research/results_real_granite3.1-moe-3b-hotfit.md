# Real cache-policy benchmark — granite3.1-moe-3b / handcrafted8-greedy512/decode

All numbers are **REAL** — computed over an actual SGLang `per_token` expert-routing capture (not synthetic). Values are the demand **hit rate**; `belady` is the Belady-optimal ceiling ('Ideal').

Trace stats: 1 file(s), 32 layers, max expert id 39, top-k 8, 1262 distinct (layer, expert) keys, 638208 accesses (decode passes only; ignore_layers=0).

Commit: `1d210a4+dirty`

| Policy | cap=128 | cap=256 | cap=384 | cap=512 | cap=768 | cap=1024 | cap=1536 | cap=2048 | cap=3072 | cap=4096 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Belady (Ideal) | 0.4219 | 0.6458 | 0.7709 | 0.8547 | 0.9501 | 0.9897 | 0.9980 | 0.9980 | 0.9980 | 0.9980 |
| HotFit (hot-set-fit adaptive) | 0.2838 | 0.4682 | 0.6082 | 0.6779 | 0.8794 | 0.9732 | 0.9980 | 0.9980 | 0.9980 | 0.9980 |
