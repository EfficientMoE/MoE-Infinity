# Prefetch x eviction - qwen3-30b-a3b / mmlu150-greedy64/decode+prefill-union/tune

REAL SGLang `per_token` routing, 75 requests, 2095125 accesses (4800 decode tokens). Commit `e0bdd04`.
Stall = timing-model stall per decode token (nominal: measured per-layer compute, measured PCIe 5.0 link, demand priority).

### capacity 128

Belady hit rate 0.2759, stall floor 42.91 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu:bypass=prefill | none | same_queue | 0.1925 | 0.0000 | 58.25 | 58.25 | 58.25 | 352.5 | 0.0 |
| lcnu_p:bypass=prefill | none | same_queue | 0.1925 | 0.0000 | 58.25 | 58.25 | 58.25 | 352.4 | 0.0 |

### capacity 256

Belady hit rate 0.4520, stall floor 30.27 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu:bypass=prefill | none | same_queue | 0.3076 | 0.0000 | 49.95 | 49.95 | 49.95 | 302.2 | 0.0 |
| lcnu_p:bypass=prefill | none | same_queue | 0.3081 | 0.0000 | 49.91 | 49.91 | 49.91 | 302.0 | 0.0 |

### capacity 512

Belady hit rate 0.6105, stall floor 18.86 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu:bypass=prefill | none | same_queue | 0.4449 | 0.0000 | 40.05 | 40.05 | 40.05 | 242.3 | 0.0 |
| lcnu_p:bypass=prefill | none | same_queue | 0.4492 | 0.0000 | 39.74 | 39.74 | 39.74 | 240.4 | 0.0 |

### capacity 1024

Belady hit rate 0.7566, stall floor 8.33 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu:bypass=prefill | none | same_queue | 0.6028 | 0.0000 | 28.65 | 28.65 | 28.65 | 173.4 | 0.0 |
| lcnu_p:bypass=prefill | none | same_queue | 0.6168 | 0.0000 | 27.65 | 27.65 | 27.65 | 167.3 | 0.0 |

### capacity 2048

Belady hit rate 0.8961, stall floor 0.00 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu:bypass=prefill | none | same_queue | 0.7961 | 0.0000 | 14.71 | 14.71 | 14.71 | 89.0 | 0.0 |
| lcnu_p:bypass=prefill | none | same_queue | 0.8235 | 0.0000 | 12.73 | 12.73 | 12.73 | 77.0 | 0.0 |

### capacity 3072

Belady hit rate 0.9619, stall floor 0.00 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu:bypass=prefill | none | same_queue | 0.9076 | 0.0000 | 6.67 | 6.67 | 6.67 | 40.3 | 0.0 |
| lcnu_p:bypass=prefill | none | same_queue | 0.9340 | 0.0000 | 4.76 | 4.76 | 4.76 | 28.8 | 0.0 |

`*` = trace-oracle prefetcher (upper bound). Full metrics: CSV.
