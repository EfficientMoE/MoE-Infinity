# Prefetch x eviction - qwen3-30b-a3b / mmlu150-greedy64/decode+prefill-union/tune

REAL SGLang `per_token` routing, 75 requests, 2095125 accesses (4800 decode tokens). Commit `d3ed922`.
Stall = timing-model stall per decode token (nominal: measured per-layer compute, measured PCIe 5.0 link, demand priority).

### capacity 128

Belady hit rate 0.2759, stall floor 43.02 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.1925 | 0.0000 | 58.26 | 58.26 | 58.26 | 352.5 | 0.0 |
| lcnu_p | none | same_queue | 0.1925 | 0.0000 | 58.25 | 58.25 | 58.25 | 352.4 | 0.0 |
| lcnu_p:prefill_weight=0.1 | none | same_queue | 0.1925 | 0.0000 | 58.25 | 58.25 | 58.25 | 352.4 | 0.0 |
| lcnu_p:prefill_weight=0.25 | none | same_queue | 0.1926 | 0.0000 | 58.25 | 58.25 | 58.25 | 352.4 | 0.0 |
| lcnu_p:prefill_weight=0.5 | none | same_queue | 0.1926 | 0.0000 | 58.25 | 58.25 | 58.25 | 352.4 | 0.0 |
| lcnu_b | none | same_queue | 0.1937 | 0.0000 | 58.17 | 58.17 | 58.17 | 352.0 | 0.0 |
| w_tinylfu | none | same_queue | 0.1527 | 0.0000 | 61.13 | 61.13 | 61.13 | 369.8 | 0.0 |
| lfu | none | same_queue | 0.1412 | 0.0000 | 61.96 | 61.96 | 61.96 | 374.9 | 0.0 |

### capacity 256

Belady hit rate 0.4520, stall floor 30.31 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.3076 | 0.0000 | 49.95 | 49.95 | 49.95 | 302.2 | 0.0 |
| lcnu_p | none | same_queue | 0.3081 | 0.0000 | 49.92 | 49.92 | 49.92 | 302.0 | 0.0 |
| lcnu_p:prefill_weight=0.1 | none | same_queue | 0.3082 | 0.0000 | 49.91 | 49.91 | 49.91 | 302.0 | 0.0 |
| lcnu_p:prefill_weight=0.25 | none | same_queue | 0.3082 | 0.0000 | 49.91 | 49.91 | 49.91 | 302.0 | 0.0 |
| lcnu_p:prefill_weight=0.5 | none | same_queue | 0.3082 | 0.0000 | 49.90 | 49.90 | 49.90 | 301.9 | 0.0 |
| lcnu_b | none | same_queue | 0.3085 | 0.0000 | 49.89 | 49.89 | 49.89 | 301.8 | 0.0 |
| w_tinylfu | none | same_queue | 0.2492 | 0.0000 | 54.16 | 54.16 | 54.16 | 327.7 | 0.0 |
| lfu | none | same_queue | 0.2354 | 0.0000 | 55.16 | 55.16 | 55.16 | 333.7 | 0.0 |

### capacity 512

Belady hit rate 0.6105, stall floor 18.88 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.4449 | 0.0000 | 40.04 | 40.04 | 40.04 | 242.3 | 0.0 |
| lcnu_p | none | same_queue | 0.4491 | 0.0000 | 39.74 | 39.74 | 39.74 | 240.4 | 0.0 |
| lcnu_p:prefill_weight=0.1 | none | same_queue | 0.4493 | 0.0000 | 39.73 | 39.73 | 39.73 | 240.4 | 0.0 |
| lcnu_p:prefill_weight=0.25 | none | same_queue | 0.4492 | 0.0000 | 39.74 | 39.74 | 39.74 | 240.4 | 0.0 |
| lcnu_p:prefill_weight=0.5 | none | same_queue | 0.4480 | 0.0000 | 39.82 | 39.82 | 39.82 | 240.9 | 0.0 |
| lcnu_b | none | same_queue | 0.4491 | 0.0000 | 39.74 | 39.74 | 39.74 | 240.4 | 0.0 |
| w_tinylfu | none | same_queue | 0.3668 | 0.0000 | 45.68 | 45.68 | 45.68 | 276.4 | 0.0 |
| lfu | none | same_queue | 0.3689 | 0.0000 | 45.53 | 45.53 | 45.53 | 275.5 | 0.0 |

### capacity 1024

Belady hit rate 0.7566, stall floor 8.34 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.6028 | 0.0000 | 28.65 | 28.65 | 28.65 | 173.4 | 0.0 |
| lcnu_p | none | same_queue | 0.6167 | 0.0000 | 27.65 | 27.65 | 27.65 | 167.3 | 0.0 |
| lcnu_p:prefill_weight=0.1 | none | same_queue | 0.6163 | 0.0000 | 27.68 | 27.68 | 27.68 | 167.5 | 0.0 |
| lcnu_p:prefill_weight=0.25 | none | same_queue | 0.6144 | 0.0000 | 27.82 | 27.82 | 27.82 | 168.3 | 0.0 |
| lcnu_p:prefill_weight=0.5 | none | same_queue | 0.6099 | 0.0000 | 28.14 | 28.14 | 28.14 | 170.3 | 0.0 |
| lcnu_b | none | same_queue | 0.6168 | 0.0000 | 27.65 | 27.65 | 27.65 | 167.3 | 0.0 |
| w_tinylfu | none | same_queue | 0.5353 | 0.0000 | 33.52 | 33.52 | 33.52 | 202.8 | 0.0 |
| lfu | none | same_queue | 0.5519 | 0.0000 | 32.32 | 32.32 | 32.32 | 195.6 | 0.0 |

### capacity 2048

Belady hit rate 0.8961, stall floor 0.00 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.7961 | 0.0000 | 14.71 | 14.71 | 14.71 | 89.0 | 0.0 |
| lcnu_p | none | same_queue | 0.8235 | 0.0000 | 12.73 | 12.73 | 12.73 | 77.0 | 0.0 |
| lcnu_p:prefill_weight=0.1 | none | same_queue | 0.8200 | 0.0000 | 12.99 | 12.99 | 12.99 | 78.6 | 0.0 |
| lcnu_p:prefill_weight=0.25 | none | same_queue | 0.8137 | 0.0000 | 13.44 | 13.44 | 13.44 | 81.3 | 0.0 |
| lcnu_p:prefill_weight=0.5 | none | same_queue | 0.8050 | 0.0000 | 14.07 | 14.07 | 14.07 | 85.1 | 0.0 |
| lcnu_b | none | same_queue | 0.8235 | 0.0000 | 12.73 | 12.73 | 12.73 | 77.0 | 0.0 |
| w_tinylfu | none | same_queue | 0.7776 | 0.0000 | 16.04 | 16.04 | 16.04 | 97.1 | 0.0 |
| lfu | none | same_queue | 0.7982 | 0.0000 | 14.56 | 14.56 | 14.56 | 88.1 | 0.0 |

### capacity 3072

Belady hit rate 0.9619, stall floor 0.00 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.9076 | 0.0000 | 6.67 | 6.67 | 6.67 | 40.3 | 0.0 |
| lcnu_p | none | same_queue | 0.9340 | 0.0000 | 4.76 | 4.76 | 4.76 | 28.8 | 0.0 |
| lcnu_p:prefill_weight=0.1 | none | same_queue | 0.9281 | 0.0000 | 5.19 | 5.19 | 5.19 | 31.4 | 0.0 |
| lcnu_p:prefill_weight=0.25 | none | same_queue | 0.9196 | 0.0000 | 5.80 | 5.80 | 5.80 | 35.1 | 0.0 |
| lcnu_p:prefill_weight=0.5 | none | same_queue | 0.9123 | 0.0000 | 6.32 | 6.32 | 6.32 | 38.3 | 0.0 |
| lcnu_b | none | same_queue | 0.9340 | 0.0000 | 4.76 | 4.76 | 4.76 | 28.8 | 0.0 |
| w_tinylfu | none | same_queue | 0.9221 | 0.0000 | 5.62 | 5.62 | 5.62 | 34.0 | 0.0 |
| lfu | none | same_queue | 0.9265 | 0.0000 | 5.30 | 5.30 | 5.30 | 32.1 | 0.0 |

`*` = trace-oracle prefetcher (upper bound). Full metrics: CSV.
