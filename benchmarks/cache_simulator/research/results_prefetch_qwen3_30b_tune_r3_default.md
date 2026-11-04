# Prefetch x eviction - qwen3-30b-a3b / mmlu150-greedy64/decode/tune

REAL SGLang `per_token` routing, 75 requests, 1843200 accesses (4800 decode tokens). Commit `d3ed922`.
Stall = timing-model stall per decode token (nominal: measured per-layer compute, measured PCIe 5.0 link, demand priority).

### capacity 128

Belady hit rate 0.3057, stall floor 34.99 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.2130 | 0.0000 | 49.95 | 49.95 | 49.95 | 302.2 | 0.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=4 | probation:issue=2 | 0.2115 | 0.2127 | 42.56 | 42.56 | 47.63 | 310.7 | 7.9 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.5 | probation:issue=2 | 0.2115 | 0.2239 | 42.56 | 42.56 | 47.61 | 312.3 | 9.5 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.2114 | 0.2243 | 42.59 | 42.59 | 47.61 | 312.6 | 9.8 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | probation:issue=2 | 0.2114 | 0.2243 | 42.59 | 42.59 | 47.61 | 312.6 | 9.8 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=16 | probation:issue=2 | 0.2114 | 0.2243 | 42.60 | 42.60 | 47.61 | 312.6 | 9.8 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32 | probation:issue=2 | 0.2114 | 0.2243 | 42.60 | 42.60 | 47.61 | 312.6 | 9.8 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=4 | probation | 0.2097 | 0.3163 | 42.67 | 42.67 | 47.74 | 311.4 | 7.9 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=4 | probation:issue=4 | 0.2097 | 0.3163 | 42.67 | 42.67 | 47.74 | 311.4 | 7.9 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=4 | probation:issue=8 | 0.2097 | 0.3163 | 42.67 | 42.67 | 47.74 | 311.4 | 7.9 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=4 | do_no_harm | 0.2101 | 0.2845 | 42.68 | 42.68 | 47.73 | 311.0 | 7.7 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.5 | probation:issue=4 | 0.2092 | 0.4084 | 42.70 | 42.70 | 47.75 | 313.1 | 9.5 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=4 | 0.2092 | 0.4141 | 42.73 | 42.73 | 47.75 | 313.4 | 9.8 |

### capacity 256

Belady hit rate 0.4973, stall floor 22.83 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.3394 | 0.0000 | 41.92 | 41.92 | 41.92 | 253.7 | 0.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.5 | probation:issue=2 | 0.3390 | 0.2121 | 34.97 | 34.97 | 39.71 | 265.8 | 12.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.5 | probation:issue=4 | 0.3381 | 0.3646 | 35.03 | 35.03 | 39.77 | 266.2 | 12.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.5 | do_no_harm | 0.3377 | 0.3544 | 35.07 | 35.07 | 39.81 | 265.8 | 11.5 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.3391 | 0.2138 | 35.08 | 35.08 | 39.72 | 267.0 | 13.2 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.5 | probation | 0.3367 | 0.4561 | 35.11 | 35.11 | 39.85 | 266.7 | 12.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.5 | probation:issue=8 | 0.3367 | 0.4558 | 35.11 | 35.11 | 39.86 | 266.7 | 12.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | probation:issue=2 | 0.3391 | 0.2141 | 35.11 | 35.11 | 39.72 | 267.3 | 13.5 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=4 | probation:issue=2 | 0.3390 | 0.1858 | 35.11 | 35.11 | 39.77 | 261.6 | 7.8 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=16 | probation:issue=2 | 0.3391 | 0.2141 | 35.12 | 35.12 | 39.73 | 267.4 | 13.6 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32 | probation:issue=2 | 0.3391 | 0.2141 | 35.12 | 35.12 | 39.73 | 267.4 | 13.6 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=4 | 0.3380 | 0.3749 | 35.13 | 35.13 | 39.78 | 267.3 | 13.2 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=4 | probation | 0.3385 | 0.2517 | 35.14 | 35.14 | 39.80 | 261.8 | 7.8 |

### capacity 512

Belady hit rate 0.6635, stall floor 12.28 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.4906 | 0.0000 | 32.33 | 32.33 | 32.33 | 195.6 | 0.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.5 | probation:issue=4 | 0.4924 | 0.2938 | 25.97 | 25.97 | 30.29 | 208.7 | 13.8 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.5 | probation:issue=2 | 0.4921 | 0.1879 | 25.99 | 25.99 | 30.31 | 208.8 | 13.8 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.5 | probation | 0.4920 | 0.3381 | 25.99 | 25.99 | 30.32 | 208.9 | 13.8 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.5 | probation:issue=8 | 0.4920 | 0.3381 | 25.99 | 25.99 | 30.32 | 208.9 | 13.8 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.5 | do_no_harm | 0.4922 | 0.2695 | 26.05 | 26.05 | 30.34 | 207.4 | 12.4 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=4 | 0.4924 | 0.3067 | 26.17 | 26.17 | 30.35 | 211.7 | 16.8 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | do_no_harm | 0.4923 | 0.2708 | 26.18 | 26.18 | 30.39 | 209.3 | 14.4 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.4922 | 0.1917 | 26.19 | 26.19 | 30.37 | 211.9 | 16.9 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation | 0.4919 | 0.3600 | 26.21 | 26.21 | 30.38 | 212.0 | 16.8 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=8 | 0.4919 | 0.3600 | 26.21 | 26.21 | 30.38 | 212.0 | 16.8 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | do_no_harm | 0.4923 | 0.2813 | 26.33 | 26.33 | 30.44 | 211.5 | 16.5 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | probation:issue=4 | 0.4924 | 0.3176 | 26.34 | 26.34 | 30.38 | 213.7 | 18.8 |

### capacity 1024

Belady hit rate 0.8081, stall floor 3.10 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.6643 | 0.0000 | 21.31 | 21.31 | 21.31 | 128.9 | 0.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.5 | probation | 0.6667 | 0.2129 | 16.03 | 16.03 | 19.67 | 140.7 | 12.8 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.5 | probation:issue=8 | 0.6667 | 0.2129 | 16.03 | 16.03 | 19.67 | 140.7 | 12.8 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.5 | probation:issue=4 | 0.6667 | 0.1978 | 16.04 | 16.04 | 19.67 | 140.8 | 12.8 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.5 | probation:issue=2 | 0.6662 | 0.1431 | 16.07 | 16.07 | 19.71 | 140.9 | 12.8 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.5 | do_no_harm | 0.6665 | 0.1800 | 16.20 | 16.20 | 19.73 | 139.0 | 10.9 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation | 0.6669 | 0.2273 | 16.21 | 16.21 | 19.78 | 145.4 | 17.5 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=8 | 0.6669 | 0.2273 | 16.21 | 16.21 | 19.78 | 145.4 | 17.5 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=4 | 0.6669 | 0.2092 | 16.21 | 16.21 | 19.78 | 145.4 | 17.5 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.6663 | 0.1487 | 16.25 | 16.25 | 19.81 | 145.6 | 17.5 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | do_no_harm | 0.6666 | 0.1820 | 16.32 | 16.32 | 19.82 | 141.8 | 13.8 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | do_no_harm | 0.6667 | 0.1891 | 16.47 | 16.47 | 19.91 | 145.5 | 17.5 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | probation:issue=4 | 0.6670 | 0.2239 | 16.52 | 16.52 | 19.87 | 150.9 | 23.0 |

### capacity 2048

Belady hit rate 0.9318, stall floor 0.00 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.8647 | 0.0000 | 8.59 | 8.59 | 8.59 | 51.9 | 0.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=4 | 0.8653 | 0.0744 | 5.87 | 5.87 | 7.97 | 63.6 | 11.8 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation | 0.8652 | 0.0756 | 5.87 | 5.87 | 7.97 | 63.6 | 11.8 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=8 | 0.8652 | 0.0756 | 5.87 | 5.87 | 7.97 | 63.6 | 11.8 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.8652 | 0.0651 | 5.87 | 5.87 | 7.97 | 63.6 | 11.8 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.5 | probation | 0.8652 | 0.0692 | 5.90 | 5.90 | 7.86 | 59.3 | 7.5 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.5 | probation:issue=8 | 0.8652 | 0.0692 | 5.90 | 5.90 | 7.86 | 59.3 | 7.5 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.5 | probation:issue=4 | 0.8652 | 0.0682 | 5.90 | 5.90 | 7.86 | 59.3 | 7.5 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.5 | probation:issue=2 | 0.8652 | 0.0602 | 5.90 | 5.90 | 7.87 | 59.3 | 7.5 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | probation:issue=4 | 0.8653 | 0.0851 | 6.02 | 6.02 | 8.13 | 72.0 | 20.3 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | probation | 0.8652 | 0.0877 | 6.02 | 6.02 | 8.14 | 72.1 | 20.3 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | probation:issue=8 | 0.8652 | 0.0876 | 6.02 | 6.02 | 8.14 | 72.1 | 20.3 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | probation:issue=2 | 0.8652 | 0.0722 | 6.02 | 6.02 | 8.14 | 72.0 | 20.3 |

### capacity 3072

Belady hit rate 0.9782, stall floor 0.00 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.9583 | 0.0000 | 2.64 | 2.64 | 2.64 | 16.0 | 0.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | probation:issue=2 | 0.9583 | 0.0189 | 1.75 | 1.75 | 2.52 | 24.4 | 8.4 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | probation | 0.9583 | 0.0198 | 1.75 | 1.75 | 2.52 | 24.4 | 8.4 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | probation:issue=8 | 0.9583 | 0.0198 | 1.75 | 1.75 | 2.52 | 24.4 | 8.4 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | probation:issue=4 | 0.9583 | 0.0198 | 1.75 | 1.75 | 2.52 | 24.4 | 8.4 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation | 0.9583 | 0.0161 | 1.83 | 1.83 | 2.47 | 19.9 | 3.9 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=4 | 0.9583 | 0.0161 | 1.83 | 1.83 | 2.47 | 19.9 | 3.9 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=8 | 0.9583 | 0.0161 | 1.83 | 1.83 | 2.47 | 19.9 | 3.9 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.9583 | 0.0156 | 1.83 | 1.83 | 2.47 | 19.9 | 3.9 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | do_no_harm | 0.9583 | 0.0160 | 1.83 | 1.83 | 2.52 | 22.1 | 6.1 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=16 | do_no_harm | 0.9583 | 0.0183 | 1.84 | 1.84 | 2.67 | 33.1 | 17.1 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.5 | probation | 0.9583 | 0.0144 | 1.90 | 1.90 | 2.44 | 18.3 | 2.3 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.5 | probation:issue=4 | 0.9583 | 0.0144 | 1.90 | 1.90 | 2.44 | 18.3 | 2.3 |

`*` = trace-oracle prefetcher (upper bound). Full metrics: CSV.
