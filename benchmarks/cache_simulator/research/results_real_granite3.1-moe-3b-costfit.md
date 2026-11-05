# Real cache-policy benchmark — granite3.1-moe-3b / belady/decode

All numbers are **REAL** — computed over an actual SGLang `per_token` expert-routing capture (not synthetic). Values are the demand **hit rate**; `belady` is the Belady-optimal ceiling ('Ideal').

Trace stats: 1 file(s), 32 layers, max expert id 39, top-k 8, 1262 distinct (layer, expert) keys, 638208 accesses (decode passes only; ignore_layers=0).

Commit: `fc168b7+dirty`

| Policy | cap=128 | cap=256 | cap=512 | cap=768 | cap=1024 |
| --- | --- | --- | --- | --- | --- |
| Belady (Ideal) | 0.4219 | 0.6458 | 0.8547 | 0.9501 | 0.9897 |
| CostFit (self-limiting switch-cost arbitration) | 0.2838 | 0.4682 | 0.7287 | 0.8864 | 0.9721 |
