# Real cache-policy benchmark — qwen3-30b-a3b / handcrafted8-greedy512/decode

All numbers are **REAL** — computed over an actual SGLang `per_token` expert-routing capture (not synthetic). Values are the demand **hit rate**; `belady` is the Belady-optimal ceiling ('Ideal').

Trace stats: 1 file(s), 48 layers, max expert id 127, top-k 8, 5948 distinct (layer, expert) keys, 1424640 accesses (decode passes only; ignore_layers=0).

Commit: `1d210a4+dirty`

| Policy | cap=128 | cap=256 | cap=384 | cap=512 | cap=768 | cap=1024 | cap=1536 | cap=2048 | cap=3072 | cap=4096 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Belady (Ideal) | 0.3067 | 0.4977 | 0.6012 | 0.6722 | 0.7682 | 0.8327 | 0.9124 | 0.9532 | 0.9856 | 0.9935 |
| HotFit (hot-set-fit adaptive) | 0.2083 | 0.3273 | 0.4148 | 0.4829 | 0.5591 | 0.6547 | 0.8083 | 0.8956 | 0.9685 | 0.9889 |
