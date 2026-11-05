# Prefetch x eviction - qwen3-30b-a3b / mmlu150-greedy64/decode+prefill-union/held

REAL SGLang `per_token` routing, 75 requests, 2086506 accesses (4800 decode tokens). Commit `2f8d8a4`.
Stall = timing-model stall per decode token (nominal: measured per-layer compute, measured PCIe 5.0 link, demand priority).

### capacity 128

Belady hit rate 0.2790, stall floor 42.58 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.1973 | 0.0000 | 57.67 | 57.67 | 57.67 | 348.9 | 0.0 |
| lcnu_p | none | same_queue | 0.1973 | 0.0000 | 57.67 | 57.67 | 57.67 | 348.9 | 0.0 |
| w_tinylfu | none | same_queue | 0.1556 | 0.0000 | 60.66 | 60.66 | 60.66 | 367.0 | 0.0 |
| lfu | none | same_queue | 0.1189 | 0.0000 | 63.30 | 63.30 | 63.30 | 383.0 | 0.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.1959 | 0.1966 | 50.43 | 50.43 | 55.38 | 360.0 | 10.5 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.1959 | 0.1966 | 50.43 | 50.43 | 55.38 | 360.0 | 10.5 |
| lfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.1174 | 0.2047 | 55.49 | 55.49 | 60.80 | 390.6 | 7.0 |
| w_tinylfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.0162 | 0.2080 | 62.52 | 62.52 | 68.00 | 433.2 | 5.5 |

### capacity 256

Belady hit rate 0.4607, stall floor 29.52 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.3144 | 0.0000 | 49.26 | 49.26 | 49.26 | 298.0 | 0.0 |
| lcnu_p | none | same_queue | 0.3147 | 0.0000 | 49.23 | 49.23 | 49.23 | 297.9 | 0.0 |
| w_tinylfu | none | same_queue | 0.2537 | 0.0000 | 53.62 | 53.62 | 53.62 | 324.4 | 0.0 |
| lfu | none | same_queue | 0.2062 | 0.0000 | 57.03 | 57.03 | 57.03 | 345.1 | 0.0 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.3147 | 0.1861 | 42.54 | 42.54 | 47.09 | 312.1 | 14.2 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.3142 | 0.1862 | 42.57 | 42.57 | 47.13 | 312.3 | 14.2 |
| lfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.2051 | 0.2003 | 49.50 | 49.50 | 54.60 | 354.4 | 8.8 |
| w_tinylfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.0833 | 0.2059 | 57.85 | 57.85 | 63.22 | 404.9 | 6.4 |

### capacity 512

Belady hit rate 0.6200, stall floor 18.08 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.4530 | 0.0000 | 39.30 | 39.30 | 39.30 | 237.8 | 0.0 |
| lcnu_p | none | same_queue | 0.4565 | 0.0000 | 39.05 | 39.05 | 39.05 | 236.2 | 0.0 |
| w_tinylfu | none | same_queue | 0.3715 | 0.0000 | 45.16 | 45.16 | 45.16 | 273.2 | 0.0 |
| lfu | none | same_queue | 0.3301 | 0.0000 | 48.13 | 48.13 | 48.13 | 291.2 | 0.0 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.4583 | 0.1661 | 33.04 | 33.04 | 37.16 | 253.0 | 17.5 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.4542 | 0.1660 | 33.34 | 33.34 | 37.45 | 254.8 | 17.6 |
| lfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.3294 | 0.1891 | 41.13 | 41.13 | 45.87 | 303.2 | 11.7 |
| w_tinylfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.2314 | 0.1987 | 47.64 | 47.64 | 52.74 | 342.9 | 8.8 |

### capacity 1024

Belady hit rate 0.7677, stall floor 7.47 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.6121 | 0.0000 | 27.87 | 27.87 | 27.87 | 168.6 | 0.0 |
| lcnu_p | none | same_queue | 0.6230 | 0.0000 | 27.09 | 27.09 | 27.09 | 163.9 | 0.0 |
| w_tinylfu | none | same_queue | 0.5467 | 0.0000 | 32.57 | 32.57 | 32.57 | 197.0 | 0.0 |
| lfu | none | same_queue | 0.5141 | 0.0000 | 34.91 | 34.91 | 34.91 | 211.2 | 0.0 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.6250 | 0.1280 | 22.17 | 22.17 | 25.66 | 180.8 | 17.8 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.6137 | 0.1272 | 23.02 | 23.02 | 26.49 | 185.9 | 17.9 |
| w_tinylfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.5567 | 0.1431 | 26.61 | 26.61 | 30.39 | 210.4 | 17.7 |
| lfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.5136 | 0.1587 | 29.03 | 29.03 | 33.11 | 226.8 | 15.4 |

### capacity 2048

Belady hit rate 0.9048, stall floor 0.00 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.8087 | 0.0000 | 13.74 | 13.74 | 13.74 | 83.2 | 0.0 |
| lcnu_p | none | same_queue | 0.8262 | 0.0000 | 12.49 | 12.49 | 12.49 | 75.6 | 0.0 |
| w_tinylfu | none | same_queue | 0.7977 | 0.0000 | 14.53 | 14.53 | 14.53 | 87.9 | 0.0 |
| lfu | none | same_queue | 0.7688 | 0.0000 | 16.61 | 16.61 | 16.61 | 100.5 | 0.0 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.8266 | 0.0552 | 9.90 | 9.90 | 11.93 | 87.6 | 12.2 |
| w_tinylfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.8163 | 0.0607 | 10.55 | 10.55 | 12.68 | 94.3 | 14.5 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.8090 | 0.0502 | 11.37 | 11.37 | 13.27 | 95.2 | 12.2 |
| lfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.7686 | 0.0770 | 13.37 | 13.37 | 15.90 | 114.7 | 14.1 |

### capacity 3072

Belady hit rate 0.9667, stall floor 0.00 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.9174 | 0.0000 | 5.93 | 5.93 | 5.93 | 35.9 | 0.0 |
| lcnu_p | none | same_queue | 0.9313 | 0.0000 | 4.94 | 4.94 | 4.94 | 29.9 | 0.0 |
| w_tinylfu | none | same_queue | 0.9353 | 0.0000 | 4.65 | 4.65 | 4.65 | 28.1 | 0.0 |
| lfu | none | same_queue | 0.9094 | 0.0000 | 6.51 | 6.51 | 6.51 | 39.4 | 0.0 |
| w_tinylfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.9351 | 0.0143 | 3.85 | 3.85 | 4.52 | 34.0 | 5.8 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.9313 | 0.0146 | 4.09 | 4.09 | 4.76 | 34.1 | 4.3 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.9174 | 0.0139 | 5.11 | 5.11 | 5.76 | 40.5 | 4.6 |
| lfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.9093 | 0.0243 | 5.24 | 5.24 | 6.26 | 45.3 | 5.9 |

`*` = trace-oracle prefetcher (upper bound). Full metrics: CSV.
