# Real cache-policy benchmark — qwen3-30b-a3b / belady/decode/decode

All numbers are **REAL** — computed over an actual SGLang `per_token` expert-routing capture (not synthetic). Values are the demand **hit rate**; `belady` is the Belady-optimal ceiling ('Ideal').

Trace stats: 1 file(s), 48 layers, max expert id 127, top-k 8, 5948 distinct (layer, expert) keys, 1424640 accesses (decode passes only; ignore_layers=0).

Commit: `b48438c+dirty`

| Policy | cap=1024 |
| --- | --- |
| Belady (Ideal) | 0.8327 |
| CostFit (self-limiting switch-cost arbitration) | 0.6674 |
| CostFit3 (3-candidate self-limiting arbitration) | 0.6674 |
| SharedCostFit (CostFit, shared per-key metadata table) | 0.6674 |
| SharedCostFit3 (CostFit3, shared per-key metadata table) | 0.6674 |
