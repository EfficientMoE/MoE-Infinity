# Prefetch x eviction - qwen3-30b-a3b / mmlu150-greedy64/decode/held

REAL SGLang `per_token` routing, 75 requests, 1843200 accesses (4800 decode tokens). Commit `d3ed922`.
Stall = timing-model stall per decode token (nominal: measured per-layer compute, measured PCIe 5.0 link, demand priority).

### capacity 128

Belady hit rate 0.3079, stall floor 34.85 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.2179 | 0.0000 | 49.64 | 49.64 | 49.64 | 300.3 | 0.0 |
| lru | none | same_queue | 0.0000 | 0.0000 | 63.47 | 63.47 | 63.47 | 384.0 | 0.0 |
| lfu | none | same_queue | 0.1303 | 0.0000 | 55.20 | 55.20 | 55.20 | 334.0 | 0.0 |
| lcnu | oracle:horizon=1* | probation:issue=2 | 0.2166 | 0.2500 | 40.64 | 40.64 | 46.67 | 300.8 | 0.0 |
| lcnu | oracle:horizon=1* | probation | 0.2095 | 0.7905 | 41.09 | 41.09 | 47.12 | 303.5 | 0.0 |
| lcnu | oracle:horizon=1* | do_no_harm | 0.2095 | 0.7905 | 41.09 | 41.09 | 47.12 | 303.5 | 0.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.2163 | 0.2223 | 42.42 | 42.42 | 47.35 | 311.5 | 10.6 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | probation:issue=2 | 0.2163 | 0.2223 | 42.42 | 42.42 | 47.35 | 311.6 | 10.6 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | do_no_harm | 0.2124 | 0.4297 | 42.67 | 42.67 | 47.61 | 312.9 | 10.4 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | do_no_harm | 0.2121 | 0.4428 | 42.70 | 42.70 | 47.62 | 313.2 | 10.6 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation | 0.2096 | 0.5872 | 42.82 | 42.82 | 47.77 | 314.0 | 10.4 |
| lcnu | logistic:horizon=1,lr=0.1,budget=8 | probation:issue=2 | 0.2163 | 0.2143 | 42.92 | 42.92 | 47.50 | 314.6 | 13.7 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | probation | 0.2042 | 0.6549 | 43.14 | 43.14 | 48.10 | 315.9 | 10.3 |
| lcnu | oracle:horizon=4* | probation | 0.1768 | 0.8232 | 43.17 | 43.17 | 49.20 | 316.1 | 0.0 |
| lcnu | oracle:horizon=4* | do_no_harm | 0.1768 | 0.8232 | 43.17 | 43.17 | 49.20 | 316.1 | 0.0 |

### capacity 256

Belady hit rate 0.5052, stall floor 22.33 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.3461 | 0.0000 | 41.50 | 41.50 | 41.50 | 251.1 | 0.0 |
| lru | none | same_queue | 0.0000 | 0.0000 | 63.47 | 63.47 | 63.47 | 384.0 | 0.0 |
| lfu | none | same_queue | 0.2251 | 0.0000 | 49.18 | 49.18 | 49.18 | 297.6 | 0.0 |
| lcnu | oracle:horizon=1* | probation:issue=2 | 0.3463 | 0.2482 | 32.44 | 32.44 | 38.45 | 251.0 | 0.0 |
| lcnu | oracle:horizon=1* | probation | 0.3432 | 0.6568 | 32.63 | 32.63 | 38.64 | 252.2 | 0.0 |
| lcnu | oracle:horizon=1* | do_no_harm | 0.3432 | 0.6568 | 32.63 | 32.63 | 38.64 | 252.2 | 0.0 |
| lcnu | oracle:horizon=4* | probation | 0.3259 | 0.6741 | 33.70 | 33.70 | 39.73 | 258.8 | 0.0 |
| lcnu | oracle:horizon=4* | do_no_harm | 0.3259 | 0.6741 | 33.70 | 33.70 | 39.73 | 258.8 | 0.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.3458 | 0.2106 | 34.83 | 34.83 | 39.38 | 265.5 | 14.3 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | probation:issue=2 | 0.3459 | 0.2109 | 34.87 | 34.87 | 39.38 | 265.8 | 14.7 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | do_no_harm | 0.3446 | 0.3548 | 34.88 | 34.88 | 39.47 | 265.0 | 13.3 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | do_no_harm | 0.3444 | 0.3674 | 34.97 | 34.97 | 39.49 | 265.9 | 14.1 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation | 0.3431 | 0.4760 | 34.98 | 34.98 | 39.54 | 266.4 | 14.2 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | probation | 0.3408 | 0.5316 | 35.16 | 35.16 | 39.69 | 267.6 | 14.5 |
| lcnu | logistic:horizon=1,lr=0.1,budget=8 | probation:issue=2 | 0.3459 | 0.1962 | 35.60 | 35.60 | 39.67 | 269.8 | 18.7 |

### capacity 512

Belady hit rate 0.6718, stall floor 11.76 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.4976 | 0.0000 | 31.89 | 31.89 | 31.89 | 192.9 | 0.0 |
| lru | none | same_queue | 0.4398 | 0.0000 | 35.56 | 35.56 | 35.56 | 215.1 | 0.0 |
| lfu | none | same_queue | 0.3577 | 0.0000 | 40.76 | 40.76 | 40.76 | 246.6 | 0.0 |
| lcnu | oracle:horizon=1* | probation:issue=2 | 0.4996 | 0.2377 | 22.90 | 22.90 | 28.76 | 192.2 | 0.0 |
| lcnu | oracle:horizon=1* | probation | 0.4991 | 0.5009 | 22.93 | 22.93 | 28.79 | 192.4 | 0.0 |
| lcnu | oracle:horizon=1* | do_no_harm | 0.4991 | 0.5009 | 22.93 | 22.93 | 28.79 | 192.4 | 0.0 |
| lcnu | oracle:horizon=4* | probation | 0.4917 | 0.5083 | 23.18 | 23.18 | 29.21 | 195.2 | 0.0 |
| lcnu | oracle:horizon=4* | do_no_harm | 0.4917 | 0.5083 | 23.18 | 23.18 | 29.21 | 195.2 | 0.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | do_no_harm | 0.4993 | 0.2677 | 25.87 | 25.87 | 30.03 | 207.1 | 14.9 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.4992 | 0.1882 | 25.90 | 25.90 | 30.02 | 209.8 | 17.5 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation | 0.4990 | 0.3519 | 25.91 | 25.91 | 30.03 | 209.9 | 17.5 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | do_no_harm | 0.4993 | 0.2788 | 26.03 | 26.03 | 30.08 | 209.4 | 17.1 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | probation:issue=2 | 0.4993 | 0.1905 | 26.09 | 26.09 | 30.05 | 212.1 | 19.8 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | probation | 0.4979 | 0.3944 | 26.16 | 26.16 | 30.12 | 212.5 | 19.7 |
| lru | oracle:horizon=4* | probation | 0.4391 | 0.5607 | 26.52 | 26.52 | 32.55 | 215.4 | 0.0 |

### capacity 1024

Belady hit rate 0.8158, stall floor 2.61 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.6697 | 0.0000 | 20.96 | 20.96 | 20.96 | 126.8 | 0.0 |
| lru | none | same_queue | 0.6432 | 0.0000 | 22.65 | 22.65 | 22.65 | 137.0 | 0.0 |
| lfu | none | same_queue | 0.5488 | 0.0000 | 28.64 | 28.64 | 28.64 | 173.3 | 0.0 |
| lcnu | oracle:horizon=4* | probation | 0.6705 | 0.3295 | 11.98 | 11.98 | 17.87 | 126.5 | 0.0 |
| lcnu | oracle:horizon=4* | do_no_harm | 0.6705 | 0.3295 | 11.98 | 11.98 | 17.87 | 126.5 | 0.0 |
| lcnu | oracle:horizon=1* | probation | 0.6728 | 0.3272 | 12.74 | 12.74 | 17.99 | 125.6 | 0.0 |
| lcnu | oracle:horizon=1* | do_no_harm | 0.6728 | 0.3272 | 12.74 | 12.74 | 17.99 | 125.6 | 0.0 |
| lcnu | oracle:horizon=1* | probation:issue=2 | 0.6722 | 0.2018 | 12.79 | 12.79 | 18.03 | 125.9 | 0.0 |
| lru | oracle:horizon=4* | probation | 0.6410 | 0.3582 | 13.93 | 13.93 | 19.75 | 137.9 | 0.0 |
| lru | oracle:horizon=4* | do_no_harm | 0.6409 | 0.3584 | 13.93 | 13.94 | 19.75 | 137.9 | 0.0 |
| lru | oracle:horizon=1* | probation:issue=2 | 0.6430 | 0.2039 | 14.61 | 14.61 | 19.88 | 137.1 | 0.0 |
| lru | oracle:horizon=1* | probation | 0.6429 | 0.3568 | 14.62 | 14.62 | 19.89 | 137.1 | 0.0 |
| lru | oracle:horizon=1* | do_no_harm | 0.6429 | 0.3568 | 14.62 | 14.62 | 19.89 | 137.1 | 0.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation | 0.6725 | 0.2205 | 16.01 | 16.01 | 19.50 | 143.6 | 17.8 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.6719 | 0.1450 | 16.05 | 16.05 | 19.54 | 143.8 | 17.8 |

### capacity 2048

Belady hit rate 0.9380, stall floor 0.00 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.8662 | 0.0000 | 8.49 | 8.49 | 8.49 | 51.4 | 0.0 |
| lru | none | same_queue | 0.8490 | 0.0000 | 9.58 | 9.58 | 9.58 | 58.0 | 0.0 |
| lfu | none | same_queue | 0.8025 | 0.0000 | 12.53 | 12.53 | 12.53 | 75.8 | 0.0 |
| lcnu | oracle:horizon=4* | do_no_harm | 0.8664 | 0.1336 | 1.92 | 1.92 | 5.70 | 51.3 | 0.0 |
| lcnu | oracle:horizon=4* | probation | 0.8664 | 0.1336 | 1.92 | 1.92 | 5.70 | 51.3 | 0.0 |
| lru | oracle:horizon=4* | do_no_harm | 0.8483 | 0.1515 | 3.06 | 3.07 | 6.89 | 58.3 | 0.0 |
| lru | oracle:horizon=4* | probation | 0.8483 | 0.1515 | 3.06 | 3.07 | 6.89 | 58.3 | 0.0 |
| lcnu | oracle:horizon=1* | probation | 0.8669 | 0.1331 | 3.25 | 3.25 | 6.57 | 51.1 | 0.0 |
| lcnu | oracle:horizon=1* | do_no_harm | 0.8669 | 0.1331 | 3.25 | 3.25 | 6.57 | 51.1 | 0.0 |
| lcnu | oracle:horizon=1* | probation:issue=2 | 0.8668 | 0.1118 | 3.26 | 3.26 | 6.58 | 51.1 | 0.0 |
| lcnu | oracle:horizon=4* | probation:issue=2 | 0.8666 | 0.0851 | 3.91 | 4.13 | 6.30 | 61.3 | 10.1 |
| lru | oracle:horizon=1* | probation:issue=2 | 0.8490 | 0.1190 | 4.21 | 4.21 | 7.66 | 58.0 | 0.0 |
| lru | oracle:horizon=1* | probation | 0.8489 | 0.1509 | 4.22 | 4.22 | 7.67 | 58.0 | 0.0 |
| lru | oracle:horizon=1* | do_no_harm | 0.8489 | 0.1509 | 4.22 | 4.22 | 7.67 | 58.0 | 0.0 |
| lfu | oracle:horizon=4* | do_no_harm | 0.8012 | 0.1987 | 5.17 | 5.17 | 9.74 | 76.3 | 0.0 |

### capacity 3072

Belady hit rate 0.9818, stall floor 0.00 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.9558 | 0.0000 | 2.80 | 2.80 | 2.80 | 17.0 | 0.0 |
| lru | none | same_queue | 0.9506 | 0.0000 | 3.14 | 3.14 | 3.14 | 19.0 | 0.0 |
| lfu | none | same_queue | 0.9314 | 0.0000 | 4.36 | 4.36 | 4.36 | 26.4 | 0.0 |
| lcnu | oracle:horizon=4* | do_no_harm | 0.9558 | 0.0442 | 0.16 | 0.16 | 1.12 | 17.0 | 0.0 |
| lcnu | oracle:horizon=4* | probation | 0.9558 | 0.0442 | 0.16 | 0.16 | 1.12 | 17.0 | 0.0 |
| lru | oracle:horizon=4* | do_no_harm | 0.9504 | 0.0495 | 0.24 | 0.24 | 1.38 | 19.0 | 0.0 |
| lru | oracle:horizon=4* | probation | 0.9504 | 0.0495 | 0.24 | 0.24 | 1.38 | 19.1 | 0.0 |
| lcnu | oracle:horizon=4* | probation:issue=2 | 0.9558 | 0.0400 | 0.34 | 0.37 | 1.16 | 17.8 | 0.8 |
| lru | oracle:horizon=4* | probation:issue=2 | 0.9504 | 0.0436 | 0.49 | 0.53 | 1.43 | 20.2 | 1.2 |
| lcnu | oracle:horizon=1* | probation | 0.9559 | 0.0441 | 0.57 | 0.57 | 1.97 | 16.9 | 0.0 |
| lcnu | oracle:horizon=1* | do_no_harm | 0.9559 | 0.0441 | 0.57 | 0.57 | 1.97 | 16.9 | 0.0 |
| lcnu | oracle:horizon=1* | probation:issue=2 | 0.9559 | 0.0417 | 0.57 | 0.57 | 1.97 | 16.9 | 0.0 |
| lfu | oracle:horizon=4* | do_no_harm | 0.9315 | 0.0685 | 0.64 | 0.64 | 2.35 | 26.3 | 0.0 |
| lfu | oracle:horizon=4* | probation | 0.9315 | 0.0685 | 0.64 | 0.64 | 2.35 | 26.3 | 0.0 |
| lru | oracle:horizon=1* | probation:issue=2 | 0.9506 | 0.0462 | 0.70 | 0.70 | 2.23 | 19.0 | 0.0 |

`*` = trace-oracle prefetcher (upper bound). Full metrics: CSV.
