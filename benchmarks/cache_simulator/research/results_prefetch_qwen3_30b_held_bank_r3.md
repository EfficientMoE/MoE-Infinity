# Prefetch x eviction - qwen3-30b-a3b / mmlu150-greedy64/decode/held

REAL SGLang `per_token` routing, 75 requests, 1843200 accesses (4800 decode tokens). Commit `8e2d550`.
Stall = timing-model stall per decode token (nominal: measured per-layer compute, measured PCIe 5.0 link, demand priority).

### capacity 128

Belady hit rate 0.3079, stall floor 34.85 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.2179 | 0.0000 | 49.64 | 49.64 | 49.64 | 300.3 | 0.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=8192 | probation:issue=2 | 0.2163 | 0.2223 | 42.42 | 42.42 | 47.35 | 311.5 | 10.6 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=2048 | probation:issue=2 | 0.2163 | 0.2220 | 42.43 | 42.43 | 47.36 | 311.6 | 10.7 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=1024 | probation:issue=2 | 0.2163 | 0.2204 | 42.53 | 42.53 | 47.40 | 312.3 | 11.3 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=512 | probation:issue=2 | 0.2163 | 0.2159 | 42.82 | 42.82 | 47.52 | 314.0 | 13.1 |

### capacity 256

Belady hit rate 0.5052, stall floor 22.33 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.3461 | 0.0000 | 41.50 | 41.50 | 41.50 | 251.1 | 0.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=8192 | probation:issue=2 | 0.3458 | 0.2106 | 34.83 | 34.83 | 39.38 | 265.5 | 14.3 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=2048 | probation:issue=2 | 0.3458 | 0.2103 | 34.85 | 34.85 | 39.39 | 265.6 | 14.4 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=1024 | probation:issue=2 | 0.3459 | 0.2083 | 34.97 | 34.97 | 39.44 | 266.3 | 15.1 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=512 | probation:issue=2 | 0.3459 | 0.2024 | 35.34 | 35.34 | 39.60 | 268.5 | 17.3 |

### capacity 512

Belady hit rate 0.6718, stall floor 11.76 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.4976 | 0.0000 | 31.89 | 31.89 | 31.89 | 192.9 | 0.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=8192 | probation:issue=2 | 0.4992 | 0.1882 | 25.90 | 25.90 | 30.02 | 209.8 | 17.5 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=2048 | probation:issue=2 | 0.4992 | 0.1878 | 25.92 | 25.92 | 30.03 | 209.9 | 17.6 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=1024 | probation:issue=2 | 0.4992 | 0.1853 | 26.06 | 26.06 | 30.10 | 210.7 | 18.4 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=512 | probation:issue=2 | 0.4992 | 0.1785 | 26.46 | 26.46 | 30.29 | 212.9 | 20.6 |

### capacity 1024

Belady hit rate 0.8158, stall floor 2.61 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.6697 | 0.0000 | 20.96 | 20.96 | 20.96 | 126.8 | 0.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=8192 | probation:issue=2 | 0.6719 | 0.1450 | 16.05 | 16.05 | 19.54 | 143.8 | 17.8 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=2048 | probation:issue=2 | 0.6719 | 0.1446 | 16.07 | 16.07 | 19.55 | 143.9 | 17.9 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=1024 | probation:issue=2 | 0.6718 | 0.1418 | 16.22 | 16.22 | 19.63 | 144.5 | 18.5 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=512 | probation:issue=2 | 0.6718 | 0.1347 | 16.57 | 16.57 | 19.81 | 146.0 | 19.9 |

### capacity 2048

Belady hit rate 0.9380, stall floor 0.00 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.8662 | 0.0000 | 8.49 | 8.49 | 8.49 | 51.4 | 0.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=8192 | probation:issue=2 | 0.8666 | 0.0634 | 5.88 | 5.88 | 7.92 | 63.4 | 12.2 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=2048 | probation:issue=2 | 0.8666 | 0.0632 | 5.88 | 5.88 | 7.92 | 63.4 | 12.2 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=1024 | probation:issue=2 | 0.8666 | 0.0611 | 5.99 | 5.99 | 7.98 | 63.7 | 12.5 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=512 | probation:issue=2 | 0.8666 | 0.0561 | 6.25 | 6.25 | 8.10 | 64.3 | 13.0 |

### capacity 3072

Belady hit rate 0.9818, stall floor 0.00 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.9558 | 0.0000 | 2.80 | 2.80 | 2.80 | 17.0 | 0.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=8192 | probation:issue=2 | 0.9559 | 0.0168 | 1.94 | 1.94 | 2.61 | 21.2 | 4.2 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=2048 | probation:issue=2 | 0.9559 | 0.0168 | 1.94 | 1.94 | 2.62 | 21.2 | 4.2 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=1024 | probation:issue=2 | 0.9559 | 0.0159 | 1.99 | 1.99 | 2.64 | 21.2 | 4.3 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=512 | probation:issue=2 | 0.9559 | 0.0139 | 2.10 | 2.10 | 2.69 | 21.4 | 4.4 |

`*` = trace-oracle prefetcher (upper bound). Full metrics: CSV.
