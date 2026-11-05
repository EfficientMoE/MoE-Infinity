# Prefetch x eviction - qwen3-235b-a22b-fp8 / mmlu100-greedy64/decode/held

REAL SGLang `per_token` routing, 50 requests, 2406400 accesses (3200 decode tokens). Commit `8e2d550`.
Stall = timing-model stall per decode token (nominal: measured per-layer compute, measured PCIe 5.0 link, demand priority).

### capacity 256

Belady hit rate 0.3077, stall floor 151.28 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.2064 | 0.0000 | 197.26 | 197.26 | 197.26 | 596.8 | 0.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=8192 | probation:issue=2 | 0.2057 | 0.2105 | 180.39 | 180.39 | 189.48 | 608.7 | 11.4 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=2048 | probation:issue=2 | 0.2057 | 0.2099 | 180.44 | 180.44 | 189.54 | 608.8 | 11.5 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=1024 | probation:issue=2 | 0.2057 | 0.2068 | 180.77 | 180.77 | 189.87 | 609.8 | 12.5 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=512 | probation:issue=2 | 0.2057 | 0.2014 | 181.34 | 181.34 | 190.44 | 611.6 | 14.3 |

### capacity 512

Belady hit rate 0.4921, stall floor 105.44 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.3295 | 0.0000 | 166.67 | 166.67 | 166.67 | 504.2 | 0.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=8192 | probation:issue=2 | 0.3296 | 0.1989 | 150.64 | 150.64 | 159.72 | 518.6 | 14.4 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=2048 | probation:issue=2 | 0.3296 | 0.1982 | 150.70 | 150.70 | 159.78 | 518.8 | 14.6 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=1024 | probation:issue=2 | 0.3296 | 0.1944 | 151.11 | 151.11 | 160.20 | 520.0 | 15.9 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=512 | probation:issue=2 | 0.3296 | 0.1880 | 151.85 | 151.85 | 160.93 | 522.2 | 18.1 |

### capacity 1024

Belady hit rate 0.6609, stall floor 63.48 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.4842 | 0.0000 | 128.21 | 128.21 | 128.21 | 387.9 | 0.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=8192 | probation:issue=2 | 0.4860 | 0.1755 | 113.67 | 113.67 | 122.52 | 405.4 | 18.9 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=2048 | probation:issue=2 | 0.4860 | 0.1746 | 113.76 | 113.76 | 122.61 | 405.7 | 19.2 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=1024 | probation:issue=2 | 0.4860 | 0.1700 | 114.28 | 114.28 | 123.11 | 407.1 | 20.6 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=512 | probation:issue=2 | 0.4860 | 0.1623 | 115.18 | 115.18 | 123.96 | 409.5 | 23.0 |

### capacity 2048

Belady hit rate 0.8109, stall floor 26.20 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.6643 | 0.0000 | 83.45 | 83.45 | 83.45 | 252.5 | 0.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=8192 | probation:issue=2 | 0.6665 | 0.1326 | 71.91 | 71.91 | 79.50 | 272.2 | 21.4 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=2048 | probation:issue=2 | 0.6665 | 0.1318 | 72.01 | 72.01 | 79.58 | 272.4 | 21.6 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=1024 | probation:issue=2 | 0.6665 | 0.1271 | 72.56 | 72.56 | 80.03 | 273.5 | 22.7 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=512 | probation:issue=2 | 0.6664 | 0.1188 | 73.51 | 73.51 | 80.77 | 274.9 | 24.0 |

### capacity 4096

Belady hit rate 0.9334, stall floor 0.00 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.8637 | 0.0000 | 33.87 | 33.87 | 33.87 | 102.5 | 0.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=8192 | probation:issue=2 | 0.8642 | 0.0547 | 28.60 | 28.60 | 32.57 | 119.5 | 17.4 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=2048 | probation:issue=2 | 0.8642 | 0.0541 | 28.68 | 28.68 | 32.62 | 119.6 | 17.5 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=1024 | probation:issue=2 | 0.8642 | 0.0512 | 29.06 | 29.06 | 32.87 | 120.0 | 17.9 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=512 | probation:issue=2 | 0.8641 | 0.0461 | 29.73 | 29.73 | 33.30 | 120.4 | 18.3 |

### capacity 6016

Belady hit rate 0.9765, stall floor 0.00 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.9527 | 0.0000 | 11.75 | 11.75 | 11.75 | 35.5 | 0.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=8192 | probation:issue=2 | 0.9528 | 0.0140 | 10.14 | 10.14 | 11.31 | 42.7 | 7.2 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=2048 | probation:issue=2 | 0.9528 | 0.0138 | 10.18 | 10.18 | 11.33 | 42.7 | 7.2 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=1024 | probation:issue=2 | 0.9528 | 0.0128 | 10.33 | 10.33 | 11.42 | 42.8 | 7.3 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,budget=8,bank=512 | probation:issue=2 | 0.9528 | 0.0107 | 10.60 | 10.60 | 11.55 | 42.6 | 7.1 |

`*` = trace-oracle prefetcher (upper bound). Full metrics: CSV.
