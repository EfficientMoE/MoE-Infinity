# Prefetch x eviction - qwen3-235b-a22b-fp8 / mmlu100-greedy64/decode/held

REAL SGLang `per_token` routing, 50 requests, 2406400 accesses (3200 decode tokens). Commit `db1ef24`.
Stall = timing-model stall per decode token (nominal: measured per-layer compute, measured PCIe 5.0 link, demand priority).

### capacity 256

Belady hit rate 0.3077, stall floor 151.28 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lru | none | same_queue | 0.0000 | 0.0000 | 248.57 | 248.57 | 248.57 | 752.0 | 0.0 |
| lfu | none | same_queue | 0.1586 | 0.0000 | 209.15 | 209.15 | 209.15 | 632.7 | 0.0 |
| lirs | none | same_queue | 0.1749 | 0.0000 | 205.10 | 205.10 | 205.10 | 620.5 | 0.0 |
| s3fifo | none | same_queue | 0.0000 | 0.0000 | 248.57 | 248.57 | 248.57 | 752.0 | 0.0 |
| w_tinylfu | none | same_queue | 0.1704 | 0.0000 | 206.21 | 206.21 | 206.21 | 623.8 | 0.0 |
| layer_lru | none | same_queue | 0.0171 | 0.0000 | 244.31 | 244.31 | 244.31 | 739.1 | 0.0 |
| lcnu | none | same_queue | 0.2064 | 0.0000 | 197.26 | 197.26 | 197.26 | 596.8 | 0.0 |
| lcnu | oracle:horizon=1* | probation | 0.2028 | 0.7972 | 177.35 | 177.35 | 186.44 | 599.5 | 0.0 |
| lcnu | oracle:horizon=1* | pin | 0.2028 | 0.7972 | 177.35 | 177.35 | 186.44 | 599.5 | 0.0 |
| lcnu | oracle:horizon=1* | do_no_harm | 0.2028 | 0.7972 | 177.35 | 177.35 | 186.44 | 599.5 | 0.0 |
| lcnu | oracle:horizon=1* | buffer | 0.1967 | 0.8032 | 178.86 | 178.86 | 187.95 | 604.1 | 0.0 |
| lirs | oracle:horizon=1* | probation | 0.1748 | 0.3746 | 184.31 | 184.31 | 193.40 | 620.5 | 0.0 |
| lirs | oracle:horizon=1* | pin | 0.1748 | 0.3746 | 184.31 | 184.31 | 193.40 | 620.5 | 0.0 |
| lirs | oracle:horizon=1* | do_no_harm | 0.1740 | 0.3746 | 184.52 | 184.52 | 193.61 | 621.2 | 0.0 |
| lirs | oracle:horizon=1* | buffer | 0.1639 | 0.8306 | 187.01 | 187.01 | 196.11 | 628.7 | 0.0 |
| lcnu | oracle:horizon=1* | same_queue | 0.2058 | 0.1357 | 187.13 | 187.13 | 196.22 | 629.1 | 31.9 |
| w_tinylfu | oracle:horizon=1* | buffer | 0.1626 | 0.8372 | 187.35 | 187.35 | 196.44 | 629.7 | 0.0 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | probation | 0.2052 | 0.1709 | 188.09 | 188.09 | 196.51 | 627.3 | 29.6 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | pin | 0.2052 | 0.1709 | 188.09 | 188.09 | 196.51 | 627.3 | 29.6 |

### capacity 512

Belady hit rate 0.4921, stall floor 105.44 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lru | none | same_queue | 0.0000 | 0.0000 | 248.57 | 248.57 | 248.57 | 752.0 | 0.0 |
| lfu | none | same_queue | 0.2579 | 0.0000 | 184.46 | 184.46 | 184.46 | 558.0 | 0.0 |
| lirs | none | same_queue | 0.2859 | 0.0000 | 177.52 | 177.52 | 177.52 | 537.0 | 0.0 |
| s3fifo | none | same_queue | 0.1325 | 0.0000 | 215.63 | 215.63 | 215.63 | 652.3 | 0.0 |
| w_tinylfu | none | same_queue | 0.2757 | 0.0000 | 180.03 | 180.03 | 180.03 | 544.6 | 0.0 |
| layer_lru | none | same_queue | 0.0872 | 0.0000 | 226.90 | 226.90 | 226.90 | 686.4 | 0.0 |
| lcnu | none | same_queue | 0.3295 | 0.0000 | 166.67 | 166.67 | 166.67 | 504.2 | 0.0 |
| lcnu | oracle:horizon=1* | probation | 0.3285 | 0.6715 | 146.12 | 146.12 | 155.21 | 505.0 | 0.0 |
| lcnu | oracle:horizon=1* | pin | 0.3285 | 0.6715 | 146.12 | 146.12 | 155.21 | 505.0 | 0.0 |
| lcnu | oracle:horizon=1* | do_no_harm | 0.3285 | 0.6715 | 146.12 | 146.12 | 155.21 | 505.0 | 0.0 |
| lcnu | oracle:horizon=1* | buffer | 0.3231 | 0.6758 | 147.44 | 147.44 | 156.54 | 509.0 | 0.0 |
| lirs | oracle:horizon=1* | probation | 0.2850 | 0.5837 | 156.92 | 156.92 | 166.02 | 537.7 | 0.0 |
| lirs | oracle:horizon=1* | pin | 0.2850 | 0.5837 | 156.92 | 156.92 | 166.02 | 537.7 | 0.0 |
| lirs | oracle:horizon=1* | do_no_harm | 0.2850 | 0.5837 | 156.93 | 156.93 | 166.02 | 537.7 | 0.0 |
| lirs | oracle:horizon=1* | buffer | 0.2789 | 0.7204 | 158.43 | 158.43 | 167.52 | 542.2 | 0.0 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | probation | 0.3295 | 0.1074 | 159.94 | 159.94 | 166.54 | 529.6 | 25.3 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | pin | 0.3295 | 0.1074 | 159.94 | 159.94 | 166.54 | 529.6 | 25.3 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | do_no_harm | 0.3295 | 0.0896 | 160.20 | 160.20 | 166.67 | 529.4 | 25.2 |
| lcnu | coactivation:budget=4,horizon=1 | probation | 0.3295 | 0.1190 | 160.30 | 160.30 | 167.66 | 535.9 | 31.7 |

### capacity 1024

Belady hit rate 0.6609, stall floor 63.48 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lru | none | same_queue | 0.3884 | 0.0000 | 152.02 | 152.02 | 152.02 | 459.9 | 0.0 |
| lfu | none | same_queue | 0.3971 | 0.0000 | 149.87 | 149.87 | 149.87 | 453.4 | 0.0 |
| lirs | none | same_queue | 0.4348 | 0.0000 | 140.50 | 140.50 | 140.50 | 425.1 | 0.0 |
| s3fifo | none | same_queue | 0.4139 | 0.0000 | 145.68 | 145.68 | 145.68 | 440.7 | 0.0 |
| w_tinylfu | none | same_queue | 0.4026 | 0.0000 | 148.49 | 148.49 | 148.49 | 449.2 | 0.0 |
| layer_lru | none | same_queue | 0.4001 | 0.0000 | 149.11 | 149.11 | 149.11 | 451.1 | 0.0 |
| lcnu | none | same_queue | 0.4842 | 0.0000 | 128.21 | 128.21 | 128.21 | 387.9 | 0.0 |
| lcnu | oracle:horizon=1* | probation | 0.4863 | 0.5137 | 107.04 | 107.04 | 116.06 | 386.3 | 0.0 |
| lcnu | oracle:horizon=1* | pin | 0.4863 | 0.5137 | 107.04 | 107.04 | 116.06 | 386.3 | 0.0 |
| lcnu | oracle:horizon=1* | do_no_harm | 0.4863 | 0.5137 | 107.04 | 107.04 | 116.06 | 386.3 | 0.0 |
| lcnu | oracle:horizon=2* | probation | 0.4848 | 0.5152 | 107.25 | 107.25 | 116.34 | 387.4 | 0.0 |
| lcnu | oracle:horizon=2* | pin | 0.4848 | 0.5152 | 107.25 | 107.25 | 116.34 | 387.4 | 0.0 |
| lcnu | oracle:horizon=2* | do_no_harm | 0.4848 | 0.5152 | 107.25 | 107.25 | 116.34 | 387.4 | 0.0 |
| lcnu | oracle:horizon=1* | buffer | 0.4803 | 0.5165 | 108.54 | 108.54 | 117.56 | 390.8 | 0.0 |
| lirs | oracle:horizon=2* | probation | 0.4342 | 0.5654 | 119.84 | 119.84 | 128.93 | 425.5 | 0.0 |
| lirs | oracle:horizon=1* | probation | 0.4346 | 0.5652 | 119.84 | 119.84 | 128.89 | 425.2 | 0.0 |
| lirs | oracle:horizon=1* | pin | 0.4346 | 0.5652 | 119.84 | 119.84 | 128.89 | 425.2 | 0.0 |
| lirs | oracle:horizon=1* | do_no_harm | 0.4346 | 0.5652 | 119.84 | 119.84 | 128.89 | 425.2 | 0.0 |
| lirs | oracle:horizon=2* | pin | 0.4342 | 0.5654 | 119.84 | 119.84 | 128.93 | 425.5 | 0.0 |

### capacity 2048

Belady hit rate 0.8109, stall floor 26.20 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lru | none | same_queue | 0.6206 | 0.0000 | 94.32 | 94.32 | 94.32 | 285.3 | 0.0 |
| lfu | none | same_queue | 0.5848 | 0.0000 | 103.21 | 103.21 | 103.21 | 312.2 | 0.0 |
| lirs | none | same_queue | 0.6149 | 0.0000 | 95.73 | 95.73 | 95.73 | 289.6 | 0.0 |
| s3fifo | none | same_queue | 0.6163 | 0.0000 | 95.38 | 95.38 | 95.38 | 288.5 | 0.0 |
| w_tinylfu | none | same_queue | 0.5755 | 0.0000 | 105.52 | 105.52 | 105.52 | 319.2 | 0.0 |
| layer_lru | none | same_queue | 0.6228 | 0.0000 | 93.76 | 93.76 | 93.76 | 283.7 | 0.0 |
| lcnu | none | same_queue | 0.6643 | 0.0000 | 83.45 | 83.45 | 83.45 | 252.5 | 0.0 |
| lcnu | oracle:horizon=4* | probation | 0.6662 | 0.3338 | 62.22 | 62.22 | 71.27 | 251.0 | 0.0 |
| lcnu | oracle:horizon=4* | pin | 0.6662 | 0.3338 | 62.22 | 62.22 | 71.27 | 251.0 | 0.0 |
| lcnu | oracle:horizon=4* | do_no_harm | 0.6662 | 0.3338 | 62.22 | 62.22 | 71.27 | 251.0 | 0.0 |
| lcnu | oracle:horizon=1* | probation | 0.6675 | 0.3325 | 63.32 | 63.32 | 71.77 | 250.0 | 0.0 |
| lcnu | oracle:horizon=1* | pin | 0.6675 | 0.3325 | 63.32 | 63.32 | 71.77 | 250.0 | 0.0 |
| lcnu | oracle:horizon=1* | do_no_harm | 0.6675 | 0.3325 | 63.32 | 63.32 | 71.77 | 250.0 | 0.0 |
| lcnu | oracle:horizon=2* | buffer | 0.6601 | 0.3362 | 64.01 | 64.03 | 72.94 | 255.6 | 0.0 |
| lcnu | oracle:horizon=1* | buffer | 0.6622 | 0.3341 | 64.63 | 64.63 | 73.09 | 254.1 | 0.0 |
| layer_lru | oracle:horizon=4* | probation | 0.6216 | 0.3624 | 73.35 | 73.48 | 82.38 | 284.6 | 0.0 |
| layer_lru | oracle:horizon=4* | pin | 0.6216 | 0.3624 | 73.35 | 73.48 | 82.38 | 284.6 | 0.0 |
| layer_lru | oracle:horizon=4* | do_no_harm | 0.6216 | 0.3624 | 73.35 | 73.48 | 82.38 | 284.6 | 0.0 |
| lru | oracle:horizon=2* | same_queue | 0.6208 | 0.3792 | 73.79 | 73.79 | 82.71 | 285.2 | 0.0 |

### capacity 4096

Belady hit rate 0.9334, stall floor 0.00 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lru | none | same_queue | 0.8378 | 0.0000 | 40.33 | 40.33 | 40.33 | 122.0 | 0.0 |
| lfu | none | same_queue | 0.8190 | 0.0000 | 44.99 | 44.99 | 44.99 | 136.1 | 0.0 |
| lirs | none | same_queue | 0.8242 | 0.0000 | 43.71 | 43.71 | 43.71 | 132.2 | 0.0 |
| s3fifo | none | same_queue | 0.8458 | 0.0000 | 38.34 | 38.34 | 38.34 | 116.0 | 0.0 |
| w_tinylfu | none | same_queue | 0.8076 | 0.0000 | 47.82 | 47.82 | 47.82 | 144.7 | 0.0 |
| layer_lru | none | same_queue | 0.8329 | 0.0000 | 41.54 | 41.54 | 41.54 | 125.7 | 0.0 |
| lcnu | none | same_queue | 0.8637 | 0.0000 | 33.87 | 33.87 | 33.87 | 102.5 | 0.0 |
| lcnu | oracle:horizon=4* | probation | 0.8643 | 0.1357 | 15.69 | 15.69 | 22.92 | 102.1 | 0.0 |
| lcnu | oracle:horizon=4* | pin | 0.8643 | 0.1357 | 15.69 | 15.69 | 22.92 | 102.1 | 0.0 |
| lcnu | oracle:horizon=4* | do_no_harm | 0.8643 | 0.1357 | 15.69 | 15.69 | 22.92 | 102.1 | 0.0 |
| lcnu | oracle:horizon=4* | buffer | 0.8593 | 0.1398 | 16.75 | 16.81 | 24.10 | 105.8 | 0.0 |
| s3fifo | oracle:horizon=4* | probation | 0.8454 | 0.1545 | 19.78 | 19.78 | 27.42 | 116.3 | 0.0 |
| s3fifo | oracle:horizon=4* | pin | 0.8453 | 0.1546 | 19.78 | 19.79 | 27.42 | 116.3 | 0.0 |
| s3fifo | oracle:horizon=4* | do_no_harm | 0.8453 | 0.1546 | 19.78 | 19.79 | 27.42 | 116.3 | 0.0 |
| w_tinylfu | oracle:horizon=4* | pin | 0.8449 | 0.1550 | 20.12 | 20.12 | 27.61 | 116.6 | 0.0 |
| w_tinylfu | oracle:horizon=4* | do_no_harm | 0.8449 | 0.1550 | 20.12 | 20.12 | 27.61 | 116.6 | 0.0 |
| w_tinylfu | oracle:horizon=4* | probation | 0.8450 | 0.1549 | 20.12 | 20.12 | 27.61 | 116.6 | 0.0 |
| s3fifo | oracle:horizon=4* | same_queue | 0.8431 | 0.1568 | 20.40 | 20.41 | 28.01 | 118.0 | 0.0 |
| lcnu | oracle:horizon=1* | probation | 0.8645 | 0.1355 | 20.65 | 20.65 | 26.35 | 101.9 | 0.0 |

### capacity 6016

Belady hit rate 0.9765, stall floor 0.00 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lru | none | same_queue | 0.9377 | 0.0000 | 15.48 | 15.48 | 15.48 | 46.8 | 0.0 |
| lfu | none | same_queue | 0.9352 | 0.0000 | 16.10 | 16.10 | 16.10 | 48.7 | 0.0 |
| lirs | none | same_queue | 0.9370 | 0.0000 | 15.66 | 15.66 | 15.66 | 47.4 | 0.0 |
| s3fifo | none | same_queue | 0.9470 | 0.0000 | 13.19 | 13.19 | 13.19 | 39.9 | 0.0 |
| w_tinylfu | none | same_queue | 0.9317 | 0.0000 | 16.98 | 16.98 | 16.98 | 51.4 | 0.0 |
| layer_lru | none | same_queue | 0.9315 | 0.0000 | 17.03 | 17.03 | 17.03 | 51.5 | 0.0 |
| lcnu | none | same_queue | 0.9527 | 0.0000 | 11.75 | 11.75 | 11.75 | 35.5 | 0.0 |
| lcnu | oracle:horizon=4* | pin | 0.9528 | 0.0472 | 2.47 | 2.48 | 5.02 | 35.5 | 0.0 |
| lcnu | oracle:horizon=4* | do_no_harm | 0.9528 | 0.0472 | 2.47 | 2.48 | 5.02 | 35.5 | 0.0 |
| lcnu | oracle:horizon=4* | probation | 0.9528 | 0.0472 | 2.47 | 2.47 | 5.02 | 35.5 | 0.0 |
| lcnu | oracle:horizon=4* | buffer | 0.9509 | 0.0490 | 2.66 | 2.68 | 5.33 | 37.0 | 0.0 |
| s3fifo | oracle:horizon=4* | pin | 0.9469 | 0.0531 | 2.97 | 2.98 | 5.94 | 39.9 | 0.0 |
| s3fifo | oracle:horizon=4* | do_no_harm | 0.9469 | 0.0531 | 2.97 | 2.98 | 5.94 | 39.9 | 0.0 |
| s3fifo | oracle:horizon=4* | probation | 0.9469 | 0.0531 | 2.97 | 2.97 | 5.94 | 39.9 | 0.0 |
| s3fifo | oracle:horizon=4* | same_queue | 0.9450 | 0.0550 | 3.14 | 3.15 | 6.24 | 41.4 | 0.0 |
| s3fifo | oracle:horizon=4* | buffer | 0.9447 | 0.0553 | 3.17 | 3.17 | 6.28 | 41.6 | 0.0 |
| w_tinylfu | oracle:horizon=4* | pin | 0.9440 | 0.0560 | 3.29 | 3.30 | 6.46 | 42.1 | 0.0 |
| w_tinylfu | oracle:horizon=4* | do_no_harm | 0.9440 | 0.0560 | 3.29 | 3.30 | 6.46 | 42.1 | 0.0 |
| w_tinylfu | oracle:horizon=4* | probation | 0.9440 | 0.0560 | 3.30 | 3.30 | 6.46 | 42.1 | 0.0 |

`*` = trace-oracle prefetcher (upper bound). Full metrics: CSV.
