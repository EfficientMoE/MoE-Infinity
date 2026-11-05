# Prefetch x eviction - qwen3-235b-a22b-fp8 / mmlu100-greedy64/decode/held

REAL SGLang `per_token` routing, 50 requests, 2406400 accesses (3200 decode tokens). Commit `d3ed922`.
Stall = timing-model stall per decode token (nominal: measured per-layer compute, measured PCIe 5.0 link, demand priority).

### capacity 256

Belady hit rate 0.3077, stall floor 151.28 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.2064 | 0.0000 | 197.26 | 197.26 | 197.26 | 596.8 | 0.0 |
| lru | none | same_queue | 0.0000 | 0.0000 | 248.57 | 248.57 | 248.57 | 752.0 | 0.0 |
| lfu | none | same_queue | 0.1586 | 0.0000 | 209.15 | 209.15 | 209.15 | 632.7 | 0.0 |
| lcnu | oracle:horizon=1* | probation:issue=2 | 0.2059 | 0.2500 | 176.58 | 176.58 | 185.67 | 597.2 | 0.0 |
| lcnu | oracle:horizon=1* | probation | 0.2028 | 0.7972 | 177.35 | 177.35 | 186.44 | 599.5 | 0.0 |
| lcnu | oracle:horizon=1* | do_no_harm | 0.2028 | 0.7972 | 177.35 | 177.35 | 186.44 | 599.5 | 0.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | probation:issue=2 | 0.2057 | 0.2105 | 180.39 | 180.39 | 189.48 | 608.7 | 11.4 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.2057 | 0.2105 | 180.39 | 180.39 | 189.48 | 608.7 | 11.4 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | do_no_harm | 0.2044 | 0.3533 | 180.89 | 180.89 | 189.96 | 610.0 | 11.8 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | do_no_harm | 0.2042 | 0.3704 | 180.94 | 180.94 | 190.02 | 610.3 | 11.8 |
| lcnu | oracle:horizon=4* | probation | 0.1878 | 0.8122 | 181.07 | 181.07 | 190.17 | 610.8 | 0.0 |
| lcnu | oracle:horizon=4* | do_no_harm | 0.1878 | 0.8122 | 181.07 | 181.07 | 190.17 | 610.8 | 0.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation | 0.2028 | 0.5354 | 181.09 | 181.09 | 190.18 | 610.8 | 11.3 |
| lcnu | logistic:horizon=1,lr=0.1,budget=8 | probation:issue=2 | 0.2058 | 0.1975 | 181.59 | 181.59 | 190.68 | 612.3 | 15.1 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | probation | 0.1999 | 0.6153 | 181.80 | 181.80 | 190.89 | 612.9 | 11.2 |

### capacity 512

Belady hit rate 0.4921, stall floor 105.44 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.3295 | 0.0000 | 166.67 | 166.67 | 166.67 | 504.2 | 0.0 |
| lru | none | same_queue | 0.0000 | 0.0000 | 248.57 | 248.57 | 248.57 | 752.0 | 0.0 |
| lfu | none | same_queue | 0.2579 | 0.0000 | 184.46 | 184.46 | 184.46 | 558.0 | 0.0 |
| lcnu | oracle:horizon=1* | probation:issue=2 | 0.3299 | 0.2494 | 145.76 | 145.76 | 154.86 | 503.9 | 0.0 |
| lcnu | oracle:horizon=1* | probation | 0.3285 | 0.6715 | 146.12 | 146.12 | 155.21 | 505.0 | 0.0 |
| lcnu | oracle:horizon=1* | do_no_harm | 0.3285 | 0.6715 | 146.12 | 146.12 | 155.21 | 505.0 | 0.0 |
| lcnu | oracle:horizon=4* | probation | 0.3192 | 0.6808 | 148.41 | 148.41 | 157.50 | 511.9 | 0.0 |
| lcnu | oracle:horizon=4* | do_no_harm | 0.3192 | 0.6808 | 148.41 | 148.41 | 157.50 | 511.9 | 0.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | probation:issue=2 | 0.3296 | 0.1992 | 150.63 | 150.63 | 159.72 | 518.6 | 14.5 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.3296 | 0.1989 | 150.64 | 150.64 | 159.72 | 518.6 | 14.4 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation | 0.3284 | 0.4375 | 150.90 | 150.90 | 159.98 | 519.4 | 14.4 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | do_no_harm | 0.3292 | 0.2940 | 151.05 | 151.05 | 160.04 | 519.2 | 14.8 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | do_no_harm | 0.3291 | 0.3103 | 151.07 | 151.07 | 160.11 | 519.6 | 15.1 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | probation | 0.3270 | 0.5050 | 151.26 | 151.26 | 160.35 | 520.5 | 14.4 |
| lcnu | logistic:horizon=1,lr=0.1,budget=8 | probation:issue=2 | 0.3297 | 0.1807 | 152.48 | 152.48 | 161.55 | 524.1 | 20.0 |

### capacity 1024

Belady hit rate 0.6609, stall floor 63.48 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.4842 | 0.0000 | 128.21 | 128.21 | 128.21 | 387.9 | 0.0 |
| lru | none | same_queue | 0.3884 | 0.0000 | 152.02 | 152.02 | 152.02 | 459.9 | 0.0 |
| lfu | none | same_queue | 0.3971 | 0.0000 | 149.87 | 149.87 | 149.87 | 453.4 | 0.0 |
| lcnu | oracle:horizon=1* | probation:issue=2 | 0.4863 | 0.2419 | 107.04 | 107.04 | 116.06 | 386.3 | 0.0 |
| lcnu | oracle:horizon=1* | probation | 0.4863 | 0.5137 | 107.04 | 107.04 | 116.06 | 386.3 | 0.0 |
| lcnu | oracle:horizon=1* | do_no_harm | 0.4863 | 0.5137 | 107.04 | 107.04 | 116.06 | 386.3 | 0.0 |
| lcnu | oracle:horizon=4* | probation | 0.4818 | 0.5182 | 108.00 | 108.00 | 117.09 | 389.7 | 0.0 |
| lcnu | oracle:horizon=4* | do_no_harm | 0.4818 | 0.5182 | 108.00 | 108.00 | 117.09 | 389.7 | 0.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation | 0.4863 | 0.3213 | 113.58 | 113.58 | 122.43 | 405.2 | 18.9 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.4860 | 0.1755 | 113.67 | 113.67 | 122.52 | 405.4 | 18.9 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | probation:issue=2 | 0.4861 | 0.1779 | 113.68 | 113.68 | 122.64 | 406.4 | 19.9 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | probation | 0.4857 | 0.3730 | 113.74 | 113.74 | 122.70 | 406.6 | 19.8 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | do_no_harm | 0.4862 | 0.2197 | 114.26 | 114.26 | 122.71 | 404.4 | 18.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | do_no_harm | 0.4863 | 0.2335 | 114.26 | 114.26 | 122.94 | 406.1 | 19.8 |
| lcnu | logistic:horizon=1,lr=0.1,budget=8 | probation | 0.4862 | 0.2533 | 116.20 | 116.20 | 124.83 | 411.6 | 25.2 |

### capacity 2048

Belady hit rate 0.8109, stall floor 26.20 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.6643 | 0.0000 | 83.45 | 83.45 | 83.45 | 252.5 | 0.0 |
| lru | none | same_queue | 0.6206 | 0.0000 | 94.32 | 94.32 | 94.32 | 285.3 | 0.0 |
| lfu | none | same_queue | 0.5848 | 0.0000 | 103.21 | 103.21 | 103.21 | 312.2 | 0.0 |
| lcnu | oracle:horizon=4* | probation | 0.6662 | 0.3338 | 62.22 | 62.22 | 71.27 | 251.0 | 0.0 |
| lcnu | oracle:horizon=4* | do_no_harm | 0.6662 | 0.3338 | 62.22 | 62.22 | 71.27 | 251.0 | 0.0 |
| lcnu | oracle:horizon=1* | probation | 0.6675 | 0.3325 | 63.32 | 63.32 | 71.77 | 250.0 | 0.0 |
| lcnu | oracle:horizon=1* | do_no_harm | 0.6675 | 0.3325 | 63.32 | 63.32 | 71.77 | 250.0 | 0.0 |
| lcnu | oracle:horizon=1* | probation:issue=2 | 0.6669 | 0.2079 | 63.48 | 63.48 | 71.92 | 250.5 | 0.0 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation | 0.6671 | 0.1960 | 71.76 | 71.76 | 79.35 | 271.8 | 21.4 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | probation | 0.6671 | 0.2292 | 71.88 | 71.88 | 79.88 | 276.6 | 26.3 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.6665 | 0.1326 | 71.91 | 71.91 | 79.50 | 272.2 | 21.4 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | probation:issue=2 | 0.6666 | 0.1395 | 72.01 | 72.01 | 80.01 | 277.0 | 26.3 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | do_no_harm | 0.6666 | 0.1455 | 72.62 | 72.62 | 79.54 | 269.3 | 18.5 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=32,min_score=0.25 | do_no_harm | 0.6667 | 0.1548 | 72.73 | 72.73 | 80.07 | 273.7 | 23.1 |
| lru | oracle:horizon=4* | probation | 0.6194 | 0.3802 | 73.90 | 73.90 | 82.93 | 286.2 | 0.0 |

### capacity 4096

Belady hit rate 0.9334, stall floor 0.00 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.8637 | 0.0000 | 33.87 | 33.87 | 33.87 | 102.5 | 0.0 |
| lru | none | same_queue | 0.8378 | 0.0000 | 40.33 | 40.33 | 40.33 | 122.0 | 0.0 |
| lfu | none | same_queue | 0.8190 | 0.0000 | 44.99 | 44.99 | 44.99 | 136.1 | 0.0 |
| lcnu | oracle:horizon=4* | probation | 0.8643 | 0.1357 | 15.69 | 15.69 | 22.92 | 102.1 | 0.0 |
| lcnu | oracle:horizon=4* | do_no_harm | 0.8643 | 0.1357 | 15.69 | 15.69 | 22.92 | 102.1 | 0.0 |
| lcnu | oracle:horizon=4* | probation:issue=2 | 0.8643 | 0.0864 | 18.94 | 20.65 | 25.39 | 110.9 | 8.9 |
| lcnu | oracle:horizon=1* | probation | 0.8645 | 0.1355 | 20.65 | 20.65 | 26.35 | 101.9 | 0.0 |
| lcnu | oracle:horizon=1* | do_no_harm | 0.8645 | 0.1355 | 20.65 | 20.65 | 26.35 | 101.9 | 0.0 |
| lcnu | oracle:horizon=1* | probation:issue=2 | 0.8644 | 0.1140 | 20.67 | 20.67 | 26.36 | 102.0 | 0.0 |
| lru | oracle:horizon=4* | probation | 0.8374 | 0.1625 | 22.03 | 22.04 | 29.52 | 122.3 | 0.0 |
| lru | oracle:horizon=4* | do_no_harm | 0.8374 | 0.1625 | 22.04 | 22.04 | 29.52 | 122.3 | 0.0 |
| lfu | oracle:horizon=4* | do_no_harm | 0.8182 | 0.1818 | 25.94 | 25.94 | 33.94 | 136.7 | 0.0 |
| lfu | oracle:horizon=4* | probation | 0.8182 | 0.1818 | 25.94 | 25.94 | 33.94 | 136.7 | 0.0 |
| lru | oracle:horizon=1* | probation:issue=2 | 0.8377 | 0.1284 | 26.30 | 26.30 | 32.44 | 122.0 | 0.0 |
| lru | oracle:horizon=1* | probation | 0.8377 | 0.1622 | 26.31 | 26.31 | 32.44 | 122.0 | 0.0 |

### capacity 6016

Belady hit rate 0.9765, stall floor 0.00 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.9527 | 0.0000 | 11.75 | 11.75 | 11.75 | 35.5 | 0.0 |
| lru | none | same_queue | 0.9377 | 0.0000 | 15.48 | 15.48 | 15.48 | 46.8 | 0.0 |
| lfu | none | same_queue | 0.9352 | 0.0000 | 16.10 | 16.10 | 16.10 | 48.7 | 0.0 |
| lcnu | oracle:horizon=4* | do_no_harm | 0.9528 | 0.0472 | 2.47 | 2.48 | 5.02 | 35.5 | 0.0 |
| lcnu | oracle:horizon=4* | probation | 0.9528 | 0.0472 | 2.47 | 2.47 | 5.02 | 35.5 | 0.0 |
| lcnu | oracle:horizon=4* | probation:issue=2 | 0.9528 | 0.0414 | 2.82 | 3.12 | 5.23 | 36.4 | 0.9 |
| lru | oracle:horizon=4* | do_no_harm | 0.9376 | 0.0623 | 4.15 | 4.16 | 7.68 | 46.9 | 0.0 |
| lru | oracle:horizon=4* | probation | 0.9376 | 0.0623 | 4.16 | 4.16 | 7.68 | 46.9 | 0.0 |
| lru | oracle:horizon=4* | probation:issue=2 | 0.9376 | 0.0518 | 4.80 | 5.28 | 8.13 | 48.6 | 1.7 |
| lfu | oracle:horizon=4* | do_no_harm | 0.9351 | 0.0649 | 5.17 | 5.17 | 8.56 | 48.8 | 0.0 |
| lfu | oracle:horizon=4* | probation | 0.9351 | 0.0649 | 5.17 | 5.17 | 8.56 | 48.8 | 0.0 |
| lcnu | oracle:horizon=1* | probation | 0.9528 | 0.0472 | 5.82 | 5.82 | 8.40 | 35.5 | 0.0 |
| lcnu | oracle:horizon=1* | do_no_harm | 0.9528 | 0.0472 | 5.82 | 5.82 | 8.40 | 35.5 | 0.0 |
| lcnu | oracle:horizon=1* | probation:issue=2 | 0.9528 | 0.0439 | 5.82 | 5.82 | 8.40 | 35.5 | 0.0 |
| lfu | oracle:horizon=4* | probation:issue=2 | 0.9351 | 0.0492 | 6.19 | 6.74 | 9.32 | 51.6 | 2.8 |

`*` = trace-oracle prefetcher (upper bound). Full metrics: CSV.
