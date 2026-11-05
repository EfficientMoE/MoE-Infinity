# Prefetch x eviction - qwen3-30b-a3b / mmlu150-greedy64/decode/held

REAL SGLang `per_token` routing, 75 requests, 1843200 accesses (4800 decode tokens). Commit `db1ef24`.
Stall = timing-model stall per decode token (nominal: measured per-layer compute, measured PCIe 5.0 link, demand priority).

### capacity 128

Belady hit rate 0.3079, stall floor 34.85 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lru | none | same_queue | 0.0000 | 0.0000 | 63.47 | 63.47 | 63.47 | 384.0 | 0.0 |
| lfu | none | same_queue | 0.1303 | 0.0000 | 55.20 | 55.20 | 55.20 | 334.0 | 0.0 |
| lirs | none | same_queue | 0.1859 | 0.0000 | 51.67 | 51.67 | 51.67 | 312.6 | 0.0 |
| s3fifo | none | same_queue | 0.0000 | 0.0000 | 63.47 | 63.47 | 63.47 | 384.0 | 0.0 |
| w_tinylfu | none | same_queue | 0.1721 | 0.0000 | 52.54 | 52.54 | 52.54 | 317.9 | 0.0 |
| layer_lru | none | same_queue | 0.0157 | 0.0000 | 62.47 | 62.47 | 62.47 | 378.0 | 0.0 |
| lcnu | none | same_queue | 0.2179 | 0.0000 | 49.64 | 49.64 | 49.64 | 300.3 | 0.0 |
| lcnu | oracle:horizon=1* | probation | 0.2095 | 0.7905 | 41.09 | 41.09 | 47.12 | 303.5 | 0.0 |
| lcnu | oracle:horizon=1* | pin | 0.2095 | 0.7905 | 41.09 | 41.09 | 47.12 | 303.5 | 0.0 |
| lcnu | oracle:horizon=1* | do_no_harm | 0.2095 | 0.7905 | 41.09 | 41.09 | 47.12 | 303.5 | 0.0 |
| lcnu | oracle:horizon=1* | buffer | 0.1962 | 0.8037 | 41.93 | 41.93 | 47.96 | 308.6 | 0.0 |
| lirs | oracle:horizon=1* | probation | 0.1859 | 0.1250 | 43.74 | 43.74 | 48.62 | 312.6 | 0.0 |
| lirs | oracle:horizon=1* | pin | 0.1859 | 0.1250 | 43.74 | 43.74 | 48.62 | 312.6 | 0.0 |
| lirs | oracle:horizon=1* | buffer | 0.1641 | 0.8215 | 43.97 | 43.97 | 50.00 | 321.0 | 0.0 |
| w_tinylfu | oracle:horizon=1* | buffer | 0.1545 | 0.8451 | 44.59 | 44.59 | 50.61 | 324.7 | 0.0 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | probation | 0.2148 | 0.1951 | 45.62 | 45.62 | 49.05 | 325.4 | 23.9 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | pin | 0.2148 | 0.1951 | 45.62 | 45.62 | 49.05 | 325.4 | 23.9 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | do_no_harm | 0.2155 | 0.1656 | 45.65 | 45.65 | 49.05 | 324.8 | 23.5 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | buffer | 0.2126 | 0.1965 | 45.75 | 45.75 | 49.18 | 326.2 | 23.9 |

### capacity 256

Belady hit rate 0.5052, stall floor 22.33 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lru | none | same_queue | 0.0000 | 0.0000 | 63.47 | 63.47 | 63.47 | 384.0 | 0.0 |
| lfu | none | same_queue | 0.2251 | 0.0000 | 49.18 | 49.18 | 49.18 | 297.6 | 0.0 |
| lirs | none | same_queue | 0.3066 | 0.0000 | 44.01 | 44.01 | 44.01 | 266.3 | 0.0 |
| s3fifo | none | same_queue | 0.0469 | 0.0000 | 60.49 | 60.49 | 60.49 | 366.0 | 0.0 |
| w_tinylfu | none | same_queue | 0.2760 | 0.0000 | 45.95 | 45.95 | 45.95 | 278.0 | 0.0 |
| layer_lru | none | same_queue | 0.0840 | 0.0000 | 58.13 | 58.13 | 58.13 | 351.7 | 0.0 |
| lcnu | none | same_queue | 0.3461 | 0.0000 | 41.50 | 41.50 | 41.50 | 251.1 | 0.0 |
| lcnu | oracle:horizon=1* | probation | 0.3432 | 0.6568 | 32.63 | 32.63 | 38.64 | 252.2 | 0.0 |
| lcnu | oracle:horizon=1* | pin | 0.3432 | 0.6568 | 32.63 | 32.63 | 38.64 | 252.2 | 0.0 |
| lcnu | oracle:horizon=1* | do_no_harm | 0.3432 | 0.6568 | 32.63 | 32.63 | 38.64 | 252.2 | 0.0 |
| lcnu | oracle:horizon=1* | buffer | 0.3331 | 0.6657 | 33.27 | 33.27 | 39.28 | 256.1 | 0.0 |
| lirs | oracle:horizon=1* | probation | 0.3061 | 0.3691 | 34.97 | 34.97 | 40.99 | 266.5 | 0.0 |
| lirs | oracle:horizon=1* | pin | 0.3061 | 0.3691 | 34.97 | 34.97 | 40.99 | 266.5 | 0.0 |
| lirs | oracle:horizon=1* | do_no_harm | 0.3060 | 0.3691 | 34.98 | 34.98 | 41.00 | 266.5 | 0.0 |
| lirs | oracle:horizon=1* | buffer | 0.2957 | 0.7032 | 35.63 | 35.63 | 41.65 | 270.5 | 0.0 |
| w_tinylfu | oracle:horizon=1* | buffer | 0.2663 | 0.7335 | 37.50 | 37.50 | 43.52 | 281.7 | 0.0 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | probation | 0.3459 | 0.1263 | 38.22 | 38.22 | 41.14 | 269.9 | 18.7 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | pin | 0.3459 | 0.1263 | 38.22 | 38.22 | 41.14 | 269.9 | 18.7 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | do_no_harm | 0.3460 | 0.1069 | 38.31 | 38.31 | 41.17 | 269.4 | 18.3 |

### capacity 512

Belady hit rate 0.6718, stall floor 11.76 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lru | none | same_queue | 0.4398 | 0.0000 | 35.56 | 35.56 | 35.56 | 215.1 | 0.0 |
| lfu | none | same_queue | 0.3577 | 0.0000 | 40.76 | 40.76 | 40.76 | 246.6 | 0.0 |
| lirs | none | same_queue | 0.4481 | 0.0000 | 35.03 | 35.03 | 35.03 | 211.9 | 0.0 |
| s3fifo | none | same_queue | 0.4245 | 0.0000 | 36.52 | 36.52 | 36.52 | 221.0 | 0.0 |
| w_tinylfu | none | same_queue | 0.4042 | 0.0000 | 37.81 | 37.81 | 37.81 | 228.8 | 0.0 |
| layer_lru | none | same_queue | 0.4472 | 0.0000 | 35.09 | 35.09 | 35.09 | 212.3 | 0.0 |
| lcnu | none | same_queue | 0.4976 | 0.0000 | 31.89 | 31.89 | 31.89 | 192.9 | 0.0 |
| lcnu | oracle:horizon=2* | probation | 0.4965 | 0.5035 | 22.90 | 22.90 | 28.91 | 193.3 | 0.0 |
| lcnu | oracle:horizon=2* | pin | 0.4965 | 0.5035 | 22.90 | 22.90 | 28.91 | 193.3 | 0.0 |
| lcnu | oracle:horizon=2* | do_no_harm | 0.4965 | 0.5035 | 22.90 | 22.90 | 28.91 | 193.3 | 0.0 |
| lcnu | oracle:horizon=1* | probation | 0.4991 | 0.5009 | 22.93 | 22.93 | 28.79 | 192.4 | 0.0 |
| lcnu | oracle:horizon=1* | pin | 0.4991 | 0.5009 | 22.93 | 22.93 | 28.79 | 192.4 | 0.0 |
| lcnu | oracle:horizon=1* | do_no_harm | 0.4991 | 0.5009 | 22.93 | 22.93 | 28.79 | 192.4 | 0.0 |
| lcnu | oracle:horizon=1* | buffer | 0.4901 | 0.5065 | 23.49 | 23.49 | 29.36 | 195.8 | 0.0 |
| lirs | oracle:horizon=2* | pin | 0.4474 | 0.4865 | 26.01 | 26.01 | 32.02 | 212.2 | 0.0 |
| lirs | oracle:horizon=2* | do_no_harm | 0.4474 | 0.4864 | 26.01 | 26.01 | 32.02 | 212.2 | 0.0 |
| lirs | oracle:horizon=2* | probation | 0.4474 | 0.4865 | 26.01 | 26.01 | 32.02 | 212.2 | 0.0 |
| lirs | oracle:horizon=1* | probation | 0.4478 | 0.4864 | 26.13 | 26.13 | 32.03 | 212.1 | 0.0 |
| lirs | oracle:horizon=1* | pin | 0.4478 | 0.4864 | 26.13 | 26.13 | 32.03 | 212.1 | 0.0 |

### capacity 1024

Belady hit rate 0.8158, stall floor 2.61 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lru | none | same_queue | 0.6432 | 0.0000 | 22.65 | 22.65 | 22.65 | 137.0 | 0.0 |
| lfu | none | same_queue | 0.5488 | 0.0000 | 28.64 | 28.64 | 28.64 | 173.3 | 0.0 |
| lirs | none | same_queue | 0.6138 | 0.0000 | 24.51 | 24.51 | 24.51 | 148.3 | 0.0 |
| s3fifo | none | same_queue | 0.6222 | 0.0000 | 23.98 | 23.98 | 23.98 | 145.1 | 0.0 |
| w_tinylfu | none | same_queue | 0.5875 | 0.0000 | 26.18 | 26.18 | 26.18 | 158.4 | 0.0 |
| layer_lru | none | same_queue | 0.6426 | 0.0000 | 22.68 | 22.68 | 22.68 | 137.2 | 0.0 |
| lcnu | none | same_queue | 0.6697 | 0.0000 | 20.96 | 20.96 | 20.96 | 126.8 | 0.0 |
| lcnu | oracle:horizon=4* | probation | 0.6705 | 0.3295 | 11.98 | 11.98 | 17.87 | 126.5 | 0.0 |
| lcnu | oracle:horizon=4* | pin | 0.6705 | 0.3295 | 11.98 | 11.98 | 17.87 | 126.5 | 0.0 |
| lcnu | oracle:horizon=4* | do_no_harm | 0.6705 | 0.3295 | 11.98 | 11.98 | 17.87 | 126.5 | 0.0 |
| lcnu | oracle:horizon=1* | probation | 0.6728 | 0.3272 | 12.74 | 12.74 | 17.99 | 125.6 | 0.0 |
| lcnu | oracle:horizon=1* | pin | 0.6728 | 0.3272 | 12.74 | 12.74 | 17.99 | 125.6 | 0.0 |
| lcnu | oracle:horizon=1* | do_no_harm | 0.6728 | 0.3272 | 12.74 | 12.74 | 17.99 | 125.6 | 0.0 |
| lcnu | oracle:horizon=2* | buffer | 0.6617 | 0.3343 | 12.78 | 12.80 | 18.49 | 129.9 | 0.0 |
| lcnu | oracle:horizon=1* | buffer | 0.6657 | 0.3303 | 13.17 | 13.17 | 18.44 | 128.4 | 0.0 |
| layer_lru | oracle:horizon=4* | probation | 0.6408 | 0.3465 | 13.93 | 14.06 | 19.77 | 137.9 | 0.0 |
| layer_lru | oracle:horizon=4* | pin | 0.6408 | 0.3465 | 13.93 | 14.06 | 19.77 | 137.9 | 0.0 |
| layer_lru | oracle:horizon=4* | do_no_harm | 0.6408 | 0.3465 | 13.93 | 14.06 | 19.77 | 137.9 | 0.0 |
| lru | oracle:horizon=4* | probation | 0.6410 | 0.3582 | 13.93 | 13.93 | 19.75 | 137.9 | 0.0 |

### capacity 2048

Belady hit rate 0.9380, stall floor 0.00 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lru | none | same_queue | 0.8490 | 0.0000 | 9.58 | 9.58 | 9.58 | 58.0 | 0.0 |
| lfu | none | same_queue | 0.8025 | 0.0000 | 12.53 | 12.53 | 12.53 | 75.8 | 0.0 |
| lirs | none | same_queue | 0.8347 | 0.0000 | 10.49 | 10.49 | 10.49 | 63.5 | 0.0 |
| s3fifo | none | same_queue | 0.8609 | 0.0000 | 8.83 | 8.83 | 8.83 | 53.4 | 0.0 |
| w_tinylfu | none | same_queue | 0.8377 | 0.0000 | 10.30 | 10.30 | 10.30 | 62.3 | 0.0 |
| layer_lru | none | same_queue | 0.8387 | 0.0000 | 10.24 | 10.24 | 10.24 | 61.9 | 0.0 |
| lcnu | none | same_queue | 0.8662 | 0.0000 | 8.49 | 8.49 | 8.49 | 51.4 | 0.0 |
| lcnu | oracle:horizon=4* | pin | 0.8664 | 0.1336 | 1.92 | 1.92 | 5.70 | 51.3 | 0.0 |
| lcnu | oracle:horizon=4* | do_no_harm | 0.8664 | 0.1336 | 1.92 | 1.92 | 5.70 | 51.3 | 0.0 |
| lcnu | oracle:horizon=4* | probation | 0.8664 | 0.1336 | 1.92 | 1.92 | 5.70 | 51.3 | 0.0 |
| s3fifo | oracle:horizon=4* | pin | 0.8602 | 0.1396 | 2.23 | 2.24 | 6.08 | 53.7 | 0.0 |
| s3fifo | oracle:horizon=4* | do_no_harm | 0.8602 | 0.1396 | 2.23 | 2.24 | 6.08 | 53.7 | 0.0 |
| s3fifo | oracle:horizon=4* | probation | 0.8601 | 0.1397 | 2.23 | 2.24 | 6.09 | 53.8 | 0.0 |
| lcnu | oracle:horizon=4* | buffer | 0.8569 | 0.1421 | 2.27 | 2.32 | 6.26 | 55.0 | 0.0 |
| w_tinylfu | oracle:horizon=4* | pin | 0.8571 | 0.1427 | 2.48 | 2.48 | 6.29 | 54.9 | 0.0 |
| w_tinylfu | oracle:horizon=4* | do_no_harm | 0.8571 | 0.1427 | 2.48 | 2.48 | 6.29 | 54.9 | 0.0 |
| w_tinylfu | oracle:horizon=4* | probation | 0.8571 | 0.1426 | 2.48 | 2.48 | 6.29 | 54.9 | 0.0 |
| s3fifo | oracle:horizon=4* | same_queue | 0.8533 | 0.1464 | 2.58 | 2.59 | 6.51 | 56.3 | 0.0 |
| s3fifo | oracle:horizon=4* | buffer | 0.8500 | 0.1499 | 2.59 | 2.60 | 6.68 | 57.6 | 0.0 |

### capacity 3072

Belady hit rate 0.9818, stall floor 0.00 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lru | none | same_queue | 0.9506 | 0.0000 | 3.14 | 3.14 | 3.14 | 19.0 | 0.0 |
| lfu | none | same_queue | 0.9314 | 0.0000 | 4.36 | 4.36 | 4.36 | 26.4 | 0.0 |
| lirs | none | same_queue | 0.9551 | 0.0000 | 2.85 | 2.85 | 2.85 | 17.3 | 0.0 |
| s3fifo | none | same_queue | 0.9590 | 0.0000 | 2.60 | 2.60 | 2.60 | 15.7 | 0.0 |
| w_tinylfu | none | same_queue | 0.9536 | 0.0000 | 2.94 | 2.94 | 2.94 | 17.8 | 0.0 |
| layer_lru | none | same_queue | 0.9401 | 0.0000 | 3.80 | 3.80 | 3.80 | 23.0 | 0.0 |
| lcnu | none | same_queue | 0.9558 | 0.0000 | 2.80 | 2.80 | 2.80 | 17.0 | 0.0 |
| s3fifo | oracle:horizon=4* | probation | 0.9589 | 0.0410 | 0.15 | 0.15 | 1.01 | 15.8 | 0.0 |
| s3fifo | oracle:horizon=4* | pin | 0.9590 | 0.0410 | 0.15 | 0.16 | 1.01 | 15.8 | 0.0 |
| s3fifo | oracle:horizon=4* | do_no_harm | 0.9590 | 0.0410 | 0.15 | 0.16 | 1.01 | 15.8 | 0.0 |
| w_tinylfu | oracle:horizon=4* | pin | 0.9581 | 0.0419 | 0.16 | 0.16 | 1.05 | 16.1 | 0.0 |
| w_tinylfu | oracle:horizon=4* | do_no_harm | 0.9581 | 0.0419 | 0.16 | 0.16 | 1.05 | 16.1 | 0.0 |
| lcnu | oracle:horizon=4* | pin | 0.9558 | 0.0442 | 0.16 | 0.16 | 1.12 | 17.0 | 0.0 |
| lcnu | oracle:horizon=4* | do_no_harm | 0.9558 | 0.0442 | 0.16 | 0.16 | 1.12 | 17.0 | 0.0 |
| lcnu | oracle:horizon=4* | probation | 0.9558 | 0.0442 | 0.16 | 0.16 | 1.12 | 17.0 | 0.0 |
| w_tinylfu | oracle:horizon=4* | probation | 0.9581 | 0.0419 | 0.16 | 0.16 | 1.05 | 16.1 | 0.0 |
| s3fifo | oracle:horizon=4* | same_queue | 0.9568 | 0.0432 | 0.17 | 0.17 | 1.09 | 16.6 | 0.0 |
| s3fifo | oracle:horizon=4* | buffer | 0.9555 | 0.0445 | 0.18 | 0.18 | 1.14 | 17.1 | 0.0 |
| lcnu | oracle:horizon=4* | buffer | 0.9525 | 0.0474 | 0.19 | 0.19 | 1.26 | 18.2 | 0.0 |

`*` = trace-oracle prefetcher (upper bound). Full metrics: CSV.
