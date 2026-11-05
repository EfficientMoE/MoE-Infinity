# Prefetch x eviction - qwen3-235b-a22b-fp8 / mmlu100-greedy64/decode+prefill-union/held

REAL SGLang `per_token` routing, 50 requests, 2726105 accesses (3200 decode tokens). Commit `2f8d8a4`.
Stall = timing-model stall per decode token (nominal: measured per-layer compute, measured PCIe 5.0 link, demand priority).

### capacity 256

Belady hit rate 0.2790, stall floor 181.89 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.1873 | 0.0000 | 228.86 | 228.86 | 228.86 | 692.4 | 0.0 |
| lcnu_p | none | same_queue | 0.1873 | 0.0000 | 228.87 | 228.87 | 228.87 | 692.4 | 0.0 |
| w_tinylfu | none | same_queue | 0.1533 | 0.0000 | 238.42 | 238.42 | 238.42 | 721.3 | 0.0 |
| lfu | none | same_queue | 0.1444 | 0.0000 | 240.94 | 240.94 | 240.94 | 728.9 | 0.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.1867 | 0.1858 | 211.99 | 211.99 | 221.08 | 704.3 | 11.4 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.1866 | 0.1858 | 212.00 | 212.00 | 221.09 | 704.3 | 11.4 |
| lfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.1436 | 0.1913 | 223.44 | 223.44 | 232.53 | 738.9 | 9.3 |
| w_tinylfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.0158 | 0.1993 | 258.66 | 258.66 | 267.75 | 845.5 | 7.0 |

### capacity 512

Belady hit rate 0.4501, stall floor 133.71 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.2997 | 0.0000 | 197.20 | 197.20 | 197.20 | 596.6 | 0.0 |
| lcnu_p | none | same_queue | 0.2999 | 0.0000 | 197.14 | 197.14 | 197.14 | 596.4 | 0.0 |
| w_tinylfu | none | same_queue | 0.2530 | 0.0000 | 210.36 | 210.36 | 210.36 | 636.4 | 0.0 |
| lfu | none | same_queue | 0.2362 | 0.0000 | 215.09 | 215.09 | 215.09 | 650.7 | 0.0 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.3001 | 0.1755 | 181.07 | 181.07 | 190.16 | 610.7 | 14.4 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.2997 | 0.1755 | 181.20 | 181.20 | 190.28 | 611.1 | 14.4 |
| lfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.2356 | 0.1859 | 198.02 | 198.02 | 207.12 | 662.0 | 10.8 |
| w_tinylfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.1079 | 0.1957 | 233.00 | 233.00 | 242.09 | 767.8 | 7.8 |

### capacity 1024

Belady hit rate 0.6105, stall floor 88.56 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.4408 | 0.0000 | 157.48 | 157.48 | 157.48 | 476.4 | 0.0 |
| lcnu_p | none | same_queue | 0.4442 | 0.0000 | 156.51 | 156.51 | 156.51 | 473.5 | 0.0 |
| w_tinylfu | none | same_queue | 0.3704 | 0.0000 | 177.30 | 177.30 | 177.30 | 536.4 | 0.0 |
| lfu | none | same_queue | 0.3671 | 0.0000 | 178.22 | 178.22 | 178.22 | 539.2 | 0.0 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.4461 | 0.1546 | 141.91 | 141.91 | 150.76 | 490.9 | 19.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.4420 | 0.1546 | 143.08 | 143.08 | 151.93 | 494.4 | 19.0 |
| lfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.3668 | 0.1728 | 162.23 | 162.23 | 171.27 | 553.4 | 14.0 |
| w_tinylfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.2190 | 0.1877 | 202.41 | 202.41 | 211.47 | 675.1 | 9.7 |

### capacity 2048

Belady hit rate 0.7627, stall floor 45.70 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.6052 | 0.0000 | 111.17 | 111.17 | 111.17 | 336.3 | 0.0 |
| lcnu_p | none | same_queue | 0.6175 | 0.0000 | 107.72 | 107.72 | 107.72 | 325.9 | 0.0 |
| w_tinylfu | none | same_queue | 0.5373 | 0.0000 | 130.29 | 130.29 | 130.29 | 394.2 | 0.0 |
| lfu | none | same_queue | 0.5488 | 0.0000 | 127.05 | 127.05 | 127.05 | 384.4 | 0.0 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.6195 | 0.1169 | 96.16 | 96.16 | 103.74 | 345.5 | 21.4 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.6069 | 0.1165 | 99.77 | 99.77 | 107.35 | 356.5 | 21.6 |
| w_tinylfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.5668 | 0.1308 | 109.97 | 109.97 | 118.13 | 390.8 | 21.7 |
| lfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.5486 | 0.1385 | 113.80 | 113.80 | 122.26 | 403.6 | 19.1 |

### capacity 4096

Belady hit rate 0.8999, stall floor 7.05 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.7988 | 0.0000 | 56.66 | 56.66 | 56.66 | 171.4 | 0.0 |
| lcnu_p | none | same_queue | 0.8237 | 0.0000 | 49.63 | 49.63 | 49.63 | 150.2 | 0.0 |
| w_tinylfu | none | same_queue | 0.7705 | 0.0000 | 64.62 | 64.62 | 64.62 | 195.5 | 0.0 |
| lfu | none | same_queue | 0.7872 | 0.0000 | 59.91 | 59.91 | 59.91 | 181.2 | 0.0 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.8242 | 0.0476 | 44.43 | 44.43 | 48.36 | 167.2 | 17.4 |
| w_tinylfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.8024 | 0.0578 | 49.87 | 49.87 | 54.53 | 189.5 | 21.1 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.7991 | 0.0464 | 51.65 | 51.65 | 55.55 | 189.2 | 18.1 |
| lfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.7872 | 0.0614 | 53.49 | 53.49 | 58.38 | 199.9 | 18.6 |

### capacity 6016

Belady hit rate 0.9600, stall floor 0.00 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.9047 | 0.0000 | 26.83 | 26.83 | 26.83 | 81.2 | 0.0 |
| lcnu_p | none | same_queue | 0.9277 | 0.0000 | 20.36 | 20.36 | 20.36 | 61.6 | 0.0 |
| w_tinylfu | none | same_queue | 0.9083 | 0.0000 | 25.82 | 25.82 | 25.82 | 78.1 | 0.0 |
| lfu | none | same_queue | 0.9149 | 0.0000 | 23.96 | 23.96 | 23.96 | 72.5 | 0.0 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.9278 | 0.0120 | 18.80 | 18.80 | 19.94 | 68.6 | 7.1 |
| w_tinylfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.9177 | 0.0167 | 21.18 | 21.18 | 22.80 | 81.2 | 11.1 |
| lfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.9149 | 0.0184 | 21.74 | 21.74 | 23.43 | 81.5 | 9.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8 | probation:issue=2 | 0.9047 | 0.0136 | 25.11 | 25.11 | 26.43 | 90.4 | 9.2 |

`*` = trace-oracle prefetcher (upper bound). Full metrics: CSV.
