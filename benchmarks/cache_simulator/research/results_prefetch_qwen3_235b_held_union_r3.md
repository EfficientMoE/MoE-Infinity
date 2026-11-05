# Prefetch x eviction - qwen3-235b-a22b-fp8 / mmlu100-greedy64/decode+prefill-union/held

REAL SGLang `per_token` routing, 50 requests, 2726105 accesses (3200 decode tokens). Commit `d3ed922`.
Stall = timing-model stall per decode token (nominal: measured per-layer compute, measured PCIe 5.0 link, demand priority).

### capacity 256

Belady hit rate 0.2790, stall floor 181.89 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.1873 | 0.0000 | 228.86 | 228.86 | 228.86 | 692.4 | 0.0 |
| lcnu_p | none | same_queue | 0.1873 | 0.0000 | 228.87 | 228.87 | 228.87 | 692.4 | 0.0 |
| w_tinylfu | none | same_queue | 0.1533 | 0.0000 | 238.42 | 238.42 | 238.42 | 721.3 | 0.0 |
| lfu | none | same_queue | 0.1444 | 0.0000 | 240.94 | 240.94 | 240.94 | 728.9 | 0.0 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.1868 | 0.0496 | 232.06 | 232.06 | 241.29 | 766.0 | 73.2 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.1868 | 0.0495 | 232.07 | 232.07 | 241.31 | 766.0 | 73.2 |
| lcnu | oracle:horizon=4* | probation:issue=2 | 0.1868 | 0.0000 | 239.41 | 239.41 | 248.64 | 788.2 | 95.5 |
| lcnu_p | oracle:horizon=4* | probation:issue=2 | 0.1868 | 0.0000 | 239.41 | 239.41 | 248.65 | 788.2 | 95.5 |
| lfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.1437 | 0.0534 | 243.66 | 243.66 | 252.89 | 801.1 | 71.5 |
| lfu | oracle:horizon=4* | probation:issue=2 | 0.1440 | 0.0000 | 251.48 | 251.48 | 260.72 | 824.7 | 95.5 |
| w_tinylfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.0004 | 0.0790 | 280.13 | 280.13 | 289.36 | 911.4 | 59.8 |
| w_tinylfu | oracle:horizon=4* | probation:issue=2 | 0.0077 | 0.0000 | 289.85 | 289.85 | 299.09 | 940.8 | 95.5 |

### capacity 512

Belady hit rate 0.4501, stall floor 133.71 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.2997 | 0.0000 | 197.20 | 197.20 | 197.20 | 596.6 | 0.0 |
| lcnu_p | none | same_queue | 0.2999 | 0.0000 | 197.14 | 197.14 | 197.14 | 596.4 | 0.0 |
| w_tinylfu | none | same_queue | 0.2530 | 0.0000 | 210.36 | 210.36 | 210.36 | 636.4 | 0.0 |
| lfu | none | same_queue | 0.2362 | 0.0000 | 215.09 | 215.09 | 215.09 | 650.7 | 0.0 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.3004 | 0.0410 | 201.44 | 201.44 | 210.66 | 673.3 | 77.3 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.2998 | 0.0409 | 201.61 | 201.61 | 210.83 | 673.8 | 77.3 |
| lcnu_p | oracle:horizon=4* | probation:issue=2 | 0.3004 | 0.0000 | 207.40 | 207.41 | 216.62 | 691.4 | 95.4 |
| lcnu | oracle:horizon=4* | probation:issue=2 | 0.3001 | 0.0000 | 207.50 | 207.50 | 216.72 | 691.7 | 95.4 |
| lfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.2357 | 0.0458 | 218.99 | 218.99 | 228.23 | 726.4 | 75.3 |
| lfu | oracle:horizon=4* | probation:issue=2 | 0.2359 | 0.0000 | 225.60 | 225.60 | 234.84 | 746.4 | 95.5 |
| w_tinylfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.0599 | 0.0734 | 264.19 | 264.19 | 273.42 | 863.2 | 62.3 |
| w_tinylfu | oracle:horizon=4* | probation:issue=2 | 0.0932 | 0.0000 | 265.78 | 265.78 | 275.02 | 868.0 | 95.4 |

### capacity 1024

Belady hit rate 0.6105, stall floor 88.56 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.4408 | 0.0000 | 157.48 | 157.48 | 157.48 | 476.4 | 0.0 |
| lcnu_p | none | same_queue | 0.4442 | 0.0000 | 156.51 | 156.51 | 156.51 | 473.5 | 0.0 |
| w_tinylfu | none | same_queue | 0.3704 | 0.0000 | 177.30 | 177.30 | 177.30 | 536.4 | 0.0 |
| lfu | none | same_queue | 0.3671 | 0.0000 | 178.22 | 178.22 | 178.22 | 539.2 | 0.0 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.4463 | 0.0332 | 161.50 | 161.50 | 170.50 | 552.4 | 80.8 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.4422 | 0.0331 | 162.67 | 162.67 | 171.67 | 556.0 | 80.8 |
| lcnu_p | oracle:horizon=4* | probation:issue=2 | 0.4464 | 0.0050 | 164.89 | 165.23 | 173.63 | 562.4 | 90.8 |
| lcnu | oracle:horizon=4* | probation:issue=2 | 0.4426 | 0.0051 | 165.96 | 166.30 | 174.69 | 565.6 | 90.7 |
| lfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.3668 | 0.0382 | 183.26 | 183.26 | 192.44 | 618.3 | 78.9 |
| lfu | oracle:horizon=4* | probation:issue=2 | 0.3668 | 0.0004 | 188.59 | 188.62 | 197.72 | 634.4 | 95.0 |
| w_tinylfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.2099 | 0.0699 | 222.19 | 222.19 | 231.35 | 735.9 | 62.9 |
| w_tinylfu | oracle:horizon=4* | probation:issue=2 | 0.2105 | 0.0014 | 232.39 | 232.46 | 241.55 | 766.9 | 94.3 |

### capacity 2048

Belady hit rate 0.7627, stall floor 45.70 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.6052 | 0.0000 | 111.17 | 111.17 | 111.17 | 336.3 | 0.0 |
| lcnu_p | none | same_queue | 0.6175 | 0.0000 | 107.72 | 107.72 | 107.72 | 325.9 | 0.0 |
| w_tinylfu | none | same_queue | 0.5373 | 0.0000 | 130.29 | 130.29 | 130.29 | 394.2 | 0.0 |
| lfu | none | same_queue | 0.5488 | 0.0000 | 127.05 | 127.05 | 127.05 | 384.4 | 0.0 |
| lcnu_p | oracle:horizon=4* | probation:issue=2 | 0.6199 | 0.0423 | 106.04 | 108.19 | 113.54 | 383.3 | 59.5 |
| lcnu | oracle:horizon=4* | probation:issue=2 | 0.6074 | 0.0424 | 109.54 | 111.71 | 117.02 | 393.8 | 59.4 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.6198 | 0.0250 | 113.55 | 113.55 | 121.01 | 406.7 | 82.8 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.6072 | 0.0249 | 117.12 | 117.12 | 124.57 | 417.5 | 82.8 |
| w_tinylfu | oracle:horizon=4* | probation:issue=2 | 0.5689 | 0.0276 | 124.31 | 125.85 | 132.07 | 438.8 | 71.5 |
| w_tinylfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.5643 | 0.0301 | 128.51 | 128.51 | 136.55 | 452.2 | 81.1 |
| lfu | oracle:horizon=4* | probation:issue=2 | 0.5484 | 0.0191 | 132.34 | 133.58 | 140.23 | 463.4 | 78.6 |
| lfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.5487 | 0.0280 | 133.11 | 133.11 | 141.43 | 465.6 | 81.1 |

### capacity 4096

Belady hit rate 0.8999, stall floor 7.05 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.7988 | 0.0000 | 56.66 | 56.66 | 56.66 | 171.4 | 0.0 |
| lcnu_p | none | same_queue | 0.8237 | 0.0000 | 49.63 | 49.63 | 49.63 | 150.2 | 0.0 |
| w_tinylfu | none | same_queue | 0.7705 | 0.0000 | 64.62 | 64.62 | 64.62 | 195.5 | 0.0 |
| lfu | none | same_queue | 0.7872 | 0.0000 | 59.91 | 59.91 | 59.91 | 181.2 | 0.0 |
| lcnu_p | oracle:horizon=4* | probation:issue=2 | 0.8244 | 0.0755 | 34.87 | 36.57 | 41.43 | 159.9 | 10.3 |
| lcnu | oracle:horizon=4* | probation:issue=2 | 0.7994 | 0.0755 | 41.76 | 43.40 | 48.31 | 180.7 | 9.8 |
| w_tinylfu | oracle:horizon=4* | probation:issue=2 | 0.8012 | 0.0743 | 42.44 | 44.38 | 49.09 | 183.9 | 14.5 |
| lfu | oracle:horizon=4* | probation:issue=2 | 0.7869 | 0.0734 | 47.58 | 49.57 | 54.69 | 202.0 | 20.4 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.8242 | 0.0115 | 54.80 | 54.80 | 57.58 | 222.0 | 72.3 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.7992 | 0.0115 | 61.78 | 61.78 | 64.49 | 242.7 | 71.7 |
| w_tinylfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.7960 | 0.0154 | 62.89 | 62.89 | 66.26 | 248.5 | 74.8 |
| lfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.7872 | 0.0126 | 65.29 | 65.29 | 68.91 | 251.5 | 70.1 |

### capacity 6016

Belady hit rate 0.9600, stall floor 0.00 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lcnu | none | same_queue | 0.9047 | 0.0000 | 26.83 | 26.83 | 26.83 | 81.2 | 0.0 |
| lcnu_p | none | same_queue | 0.9277 | 0.0000 | 20.36 | 20.36 | 20.36 | 61.6 | 0.0 |
| w_tinylfu | none | same_queue | 0.9083 | 0.0000 | 25.82 | 25.82 | 25.82 | 78.1 | 0.0 |
| lfu | none | same_queue | 0.9149 | 0.0000 | 23.96 | 23.96 | 23.96 | 72.5 | 0.0 |
| lcnu_p | oracle:horizon=4* | probation:issue=2 | 0.9278 | 0.0351 | 11.79 | 12.10 | 14.23 | 63.8 | 2.3 |
| w_tinylfu | oracle:horizon=4* | probation:issue=2 | 0.9159 | 0.0399 | 14.06 | 14.43 | 16.95 | 74.2 | 2.5 |
| lfu | oracle:horizon=4* | probation:issue=2 | 0.9148 | 0.0430 | 14.17 | 14.74 | 17.42 | 76.7 | 4.2 |
| lcnu | oracle:horizon=4* | probation:issue=2 | 0.9048 | 0.0370 | 17.76 | 18.07 | 20.34 | 83.3 | 2.2 |
| lcnu_p | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.9277 | 0.0041 | 22.35 | 22.35 | 23.11 | 113.3 | 51.7 |
| lfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.9149 | 0.0043 | 26.28 | 26.28 | 27.31 | 122.1 | 49.6 |
| w_tinylfu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.9133 | 0.0054 | 26.57 | 26.57 | 27.55 | 125.4 | 51.6 |
| lcnu | knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8 | probation:issue=2 | 0.9047 | 0.0044 | 28.70 | 28.70 | 29.54 | 129.5 | 48.4 |

`*` = trace-oracle prefetcher (upper bound). Full metrics: CSV.
