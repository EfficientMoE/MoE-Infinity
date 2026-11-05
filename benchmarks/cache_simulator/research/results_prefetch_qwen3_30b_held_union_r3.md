# Prefetch x eviction - qwen3-30b-a3b / mmlu150-greedy64/decode+prefill-union/held

REAL SGLang `per_token` routing, 75 requests, 2086506 accesses (4800 decode tokens). Commit `d3ed922`.
Stall = timing-model stall per decode token (nominal: measured per-layer compute, measured PCIe 5.0 link, demand priority).

### capacity 128

Belady hit rate 0.2790, stall floor 42.58 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.1973 | 0.0000 | 57.67 | 57.67 | 57.67 | 348.9 | 0.0 |
| lcnu_p | none | same_queue | 0.1973 | 0.0000 | 57.67 | 57.67 | 57.67 | 348.9 | 0.0 |
| w_tinylfu | none | same_queue | 0.1556 | 0.0000 | 60.66 | 60.66 | 60.66 | 367.0 | 0.0 |
| lfu | none | same_queue | 0.1189 | 0.0000 | 63.30 | 63.30 | 63.30 | 383.0 | 0.0 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.1961 | 0.0771 | 59.11 | 59.11 | 59.90 | 413.4 | 63.9 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.1961 | 0.0771 | 59.11 | 59.11 | 59.90 | 413.4 | 64.0 |
| lfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.1176 | 0.0864 | 64.09 | 64.09 | 65.27 | 443.5 | 59.9 |
| lcnu_p | oracle:horizon=4* | probation:issue=2 | 0.1962 | 0.0000 | 64.64 | 64.64 | 62.71 | 446.9 | 97.5 |
| lcnu | oracle:horizon=4* | probation:issue=2 | 0.1962 | 0.0000 | 64.64 | 64.64 | 62.71 | 446.9 | 97.5 |
| lfu | oracle:horizon=4* | probation:issue=2 | 0.1182 | 0.0000 | 70.25 | 70.25 | 68.32 | 480.8 | 97.5 |
| w_tinylfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.0014 | 0.0990 | 71.52 | 71.52 | 73.10 | 488.5 | 54.4 |
| w_tinylfu | oracle:horizon=4* | probation:issue=2 | 0.0072 | 0.0000 | 78.22 | 78.22 | 76.29 | 529.0 | 97.5 |

### capacity 256

Belady hit rate 0.4607, stall floor 29.52 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.3144 | 0.0000 | 49.26 | 49.26 | 49.26 | 298.0 | 0.0 |
| lcnu_p | none | same_queue | 0.3147 | 0.0000 | 49.23 | 49.23 | 49.23 | 297.9 | 0.0 |
| w_tinylfu | none | same_queue | 0.2537 | 0.0000 | 53.62 | 53.62 | 53.62 | 324.4 | 0.0 |
| lfu | none | same_queue | 0.2062 | 0.0000 | 57.03 | 57.03 | 57.03 | 345.1 | 0.0 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.3149 | 0.0698 | 51.07 | 51.07 | 51.62 | 364.7 | 66.9 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.3143 | 0.0698 | 51.12 | 51.12 | 51.67 | 365.0 | 67.0 |
| lcnu_p | oracle:horizon=4* | probation:issue=2 | 0.3151 | 0.0004 | 55.97 | 55.97 | 54.12 | 394.3 | 96.6 |
| lcnu | oracle:horizon=4* | probation:issue=2 | 0.3146 | 0.0004 | 56.01 | 56.01 | 54.15 | 394.5 | 96.6 |
| lfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.2052 | 0.0800 | 58.23 | 58.23 | 59.19 | 408.1 | 62.6 |
| lfu | oracle:horizon=4* | probation:issue=2 | 0.2056 | 0.0000 | 63.96 | 63.96 | 62.03 | 442.8 | 97.4 |
| w_tinylfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.0525 | 0.0976 | 67.96 | 67.96 | 69.46 | 466.9 | 55.1 |
| w_tinylfu | oracle:horizon=4* | probation:issue=2 | 0.0733 | 0.0000 | 73.47 | 73.47 | 71.54 | 500.3 | 97.5 |

### capacity 512

Belady hit rate 0.6200, stall floor 18.08 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.4530 | 0.0000 | 39.30 | 39.30 | 39.30 | 237.8 | 0.0 |
| lcnu_p | none | same_queue | 0.4565 | 0.0000 | 39.05 | 39.05 | 39.05 | 236.2 | 0.0 |
| w_tinylfu | none | same_queue | 0.3715 | 0.0000 | 45.16 | 45.16 | 45.16 | 273.2 | 0.0 |
| lfu | none | same_queue | 0.3301 | 0.0000 | 48.13 | 48.13 | 48.13 | 291.2 | 0.0 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.4584 | 0.0609 | 41.14 | 41.14 | 41.49 | 304.3 | 68.9 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.4544 | 0.0609 | 41.44 | 41.44 | 41.78 | 306.1 | 68.9 |
| lcnu_p | oracle:horizon=4* | probation:issue=2 | 0.4585 | 0.0087 | 44.43 | 44.46 | 43.02 | 323.6 | 88.2 |
| lcnu | oracle:horizon=4* | probation:issue=2 | 0.4548 | 0.0086 | 44.70 | 44.73 | 43.29 | 325.2 | 88.2 |
| lfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.3295 | 0.0726 | 49.67 | 49.67 | 50.44 | 355.9 | 64.4 |
| lfu | oracle:horizon=4* | probation:issue=2 | 0.3296 | 0.0016 | 54.75 | 54.75 | 52.95 | 386.7 | 95.3 |
| w_tinylfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.1825 | 0.0916 | 58.97 | 58.97 | 60.36 | 412.3 | 57.0 |
| w_tinylfu | oracle:horizon=4* | probation:issue=2 | 0.1831 | 0.0015 | 65.37 | 65.38 | 63.53 | 451.2 | 96.1 |

### capacity 1024

Belady hit rate 0.7677, stall floor 7.47 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.6121 | 0.0000 | 27.87 | 27.87 | 27.87 | 168.6 | 0.0 |
| lcnu_p | none | same_queue | 0.6230 | 0.0000 | 27.09 | 27.09 | 27.09 | 163.9 | 0.0 |
| w_tinylfu | none | same_queue | 0.5467 | 0.0000 | 32.57 | 32.57 | 32.57 | 197.0 | 0.0 |
| lfu | none | same_queue | 0.5141 | 0.0000 | 34.91 | 34.91 | 34.91 | 211.2 | 0.0 |
| lcnu_p | oracle:horizon=4* | probation:issue=2 | 0.6252 | 0.0462 | 27.71 | 27.97 | 28.18 | 219.9 | 57.0 |
| lcnu | oracle:horizon=4* | probation:issue=2 | 0.6140 | 0.0468 | 28.43 | 28.70 | 28.93 | 224.2 | 56.5 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.6251 | 0.0487 | 29.01 | 29.01 | 29.36 | 229.0 | 66.1 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.6139 | 0.0484 | 29.83 | 29.83 | 30.17 | 234.0 | 66.2 |
| w_tinylfu | oracle:horizon=4* | probation:issue=2 | 0.5765 | 0.0338 | 32.75 | 32.94 | 32.60 | 251.1 | 67.0 |
| w_tinylfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.5575 | 0.0564 | 33.77 | 33.77 | 34.18 | 258.8 | 66.5 |
| lfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.5137 | 0.0585 | 36.59 | 36.59 | 37.33 | 274.9 | 63.5 |
| lfu | oracle:horizon=4* | probation:issue=2 | 0.5135 | 0.0185 | 39.22 | 39.32 | 38.31 | 291.4 | 79.9 |

### capacity 2048

Belady hit rate 0.9048, stall floor 0.00 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.8087 | 0.0000 | 13.74 | 13.74 | 13.74 | 83.2 | 0.0 |
| lcnu_p | none | same_queue | 0.8262 | 0.0000 | 12.49 | 12.49 | 12.49 | 75.6 | 0.0 |
| w_tinylfu | none | same_queue | 0.7977 | 0.0000 | 14.53 | 14.53 | 14.53 | 87.9 | 0.0 |
| lfu | none | same_queue | 0.7688 | 0.0000 | 16.61 | 16.61 | 16.61 | 100.5 | 0.0 |
| lcnu_p | oracle:horizon=4* | probation:issue=2 | 0.8267 | 0.0746 | 8.01 | 8.23 | 10.36 | 86.8 | 11.5 |
| lcnu | oracle:horizon=4* | probation:issue=2 | 0.8091 | 0.0731 | 9.23 | 9.43 | 11.54 | 92.5 | 9.5 |
| w_tinylfu | oracle:horizon=4* | probation:issue=2 | 0.8137 | 0.0705 | 9.43 | 9.65 | 11.55 | 95.6 | 14.6 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.8265 | 0.0227 | 13.19 | 13.19 | 13.90 | 122.9 | 47.5 |
| lfu | oracle:horizon=4* | probation:issue=2 | 0.7681 | 0.0659 | 14.02 | 14.27 | 15.77 | 129.0 | 28.1 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.8090 | 0.0219 | 14.38 | 14.38 | 15.07 | 129.7 | 46.7 |
| w_tinylfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.8085 | 0.0275 | 14.63 | 14.63 | 15.25 | 134.7 | 51.5 |
| lfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.7686 | 0.0294 | 17.36 | 17.36 | 18.32 | 147.4 | 46.8 |

### capacity 3072

Belady hit rate 0.9667, stall floor 0.00 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.9174 | 0.0000 | 5.93 | 5.93 | 5.93 | 35.9 | 0.0 |
| lcnu_p | none | same_queue | 0.9313 | 0.0000 | 4.94 | 4.94 | 4.94 | 29.9 | 0.0 |
| w_tinylfu | none | same_queue | 0.9353 | 0.0000 | 4.65 | 4.65 | 4.65 | 28.1 | 0.0 |
| lfu | none | same_queue | 0.9094 | 0.0000 | 6.51 | 6.51 | 6.51 | 39.4 | 0.0 |
| lcnu_p | oracle:horizon=4* | probation:issue=2 | 0.9313 | 0.0347 | 2.61 | 2.63 | 3.37 | 32.1 | 2.3 |
| w_tinylfu | oracle:horizon=4* | probation:issue=2 | 0.9319 | 0.0326 | 2.71 | 2.73 | 3.38 | 31.8 | 2.2 |
| lfu | oracle:horizon=4* | probation:issue=2 | 0.9093 | 0.0466 | 3.56 | 3.63 | 4.77 | 44.5 | 5.1 |
| lcnu | oracle:horizon=4* | probation:issue=2 | 0.9174 | 0.0315 | 3.80 | 3.82 | 4.43 | 37.8 | 1.9 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.9313 | 0.0077 | 4.88 | 4.88 | 5.38 | 57.5 | 27.6 |
| w_tinylfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.9304 | 0.0084 | 4.94 | 4.94 | 5.41 | 58.8 | 28.6 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.9174 | 0.0087 | 5.74 | 5.74 | 6.25 | 60.6 | 24.7 |
| lfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.9093 | 0.0107 | 6.44 | 6.44 | 7.13 | 65.7 | 26.3 |

`*` = trace-oracle prefetcher (upper bound). Full metrics: CSV.
