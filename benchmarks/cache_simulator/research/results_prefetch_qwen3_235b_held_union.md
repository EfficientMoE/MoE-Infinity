# Prefetch x eviction - qwen3-235b-a22b-fp8 / mmlu100-greedy64/decode+prefill-union/held

REAL SGLang `per_token` routing, 50 requests, 2726105 accesses (3200 decode tokens). Commit `db1ef24`.
Stall = timing-model stall per decode token (nominal: measured per-layer compute, measured PCIe 5.0 link, demand priority).

### capacity 256

Belady hit rate 0.2790, stall floor 181.89 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lru | none | same_queue | 0.0000 | 0.0000 | 281.60 | 281.60 | 281.60 | 851.9 | 0.0 |
| lfu | none | same_queue | 0.1444 | 0.0000 | 240.94 | 240.94 | 240.94 | 728.9 | 0.0 |
| s3fifo | none | same_queue | 0.0000 | 0.0000 | 281.60 | 281.60 | 281.60 | 851.9 | 0.0 |
| w_tinylfu | none | same_queue | 0.1533 | 0.0000 | 238.42 | 238.42 | 238.42 | 721.3 | 0.0 |
| lcnu | none | same_queue | 0.1873 | 0.0000 | 228.86 | 228.86 | 228.86 | 692.4 | 0.0 |
| lcnu | oracle:horizon=1* | probation | 0.1824 | 0.8176 | 209.11 | 209.11 | 218.35 | 696.6 | 0.0 |
| lcnu | oracle:horizon=1* | pin | 0.1824 | 0.8176 | 209.11 | 209.11 | 218.35 | 696.6 | 0.0 |
| lcnu | oracle:horizon=1* | do_no_harm | 0.1824 | 0.8176 | 209.11 | 209.11 | 218.35 | 696.6 | 0.0 |
| lcnu | oracle:horizon=1* | buffer | 0.1784 | 0.7356 | 210.22 | 210.22 | 219.46 | 699.9 | 0.0 |
| lcnu | oracle:horizon=1* | same_queue | 0.1866 | 0.1315 | 217.88 | 217.88 | 227.12 | 723.1 | 30.2 |
| w_tinylfu | oracle:horizon=1* | buffer | 0.1466 | 0.7667 | 219.17 | 219.17 | 228.41 | 727.0 | 0.0 |
| lfu | oracle:horizon=1* | same_queue | 0.1445 | 0.8555 | 219.78 | 219.78 | 229.02 | 728.8 | 0.0 |
| lfu | oracle:horizon=1* | probation | 0.1445 | 0.8555 | 219.78 | 219.78 | 229.02 | 728.8 | 0.0 |
| lfu | oracle:horizon=1* | pin | 0.1445 | 0.8555 | 219.78 | 219.78 | 229.02 | 728.8 | 0.0 |
| lfu | oracle:horizon=1* | do_no_harm | 0.1445 | 0.8555 | 219.78 | 219.78 | 229.02 | 728.8 | 0.0 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | probation | 0.1862 | 0.1492 | 219.82 | 219.82 | 228.34 | 724.0 | 30.7 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | pin | 0.1862 | 0.1492 | 219.82 | 219.82 | 228.34 | 724.0 | 30.7 |

### capacity 512

Belady hit rate 0.4501, stall floor 133.71 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lru | none | same_queue | 0.0000 | 0.0000 | 281.60 | 281.60 | 281.60 | 851.9 | 0.0 |
| lfu | none | same_queue | 0.2362 | 0.0000 | 215.09 | 215.09 | 215.09 | 650.7 | 0.0 |
| s3fifo | none | same_queue | 0.1215 | 0.0000 | 247.37 | 247.37 | 247.37 | 748.4 | 0.0 |
| w_tinylfu | none | same_queue | 0.2530 | 0.0000 | 210.36 | 210.36 | 210.36 | 636.4 | 0.0 |
| lcnu | none | same_queue | 0.2997 | 0.0000 | 197.20 | 197.20 | 197.20 | 596.6 | 0.0 |
| lcnu | oracle:horizon=1* | probation | 0.2976 | 0.7024 | 176.67 | 176.67 | 185.90 | 598.4 | 0.0 |
| lcnu | oracle:horizon=1* | pin | 0.2976 | 0.7024 | 176.67 | 176.67 | 185.90 | 598.4 | 0.0 |
| lcnu | oracle:horizon=1* | do_no_harm | 0.2976 | 0.7024 | 176.67 | 176.67 | 185.90 | 598.4 | 0.0 |
| lcnu | oracle:horizon=1* | buffer | 0.2940 | 0.6226 | 177.69 | 177.69 | 186.92 | 601.5 | 0.0 |
| lcnu | oracle:horizon=1* | same_queue | 0.3001 | 0.0949 | 189.88 | 189.88 | 199.12 | 638.4 | 42.1 |
| w_tinylfu | oracle:horizon=1* | buffer | 0.2478 | 0.6692 | 190.69 | 190.69 | 199.93 | 640.8 | 0.0 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | probation | 0.2995 | 0.0919 | 190.72 | 190.72 | 197.33 | 622.7 | 26.0 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | pin | 0.2995 | 0.0919 | 190.72 | 190.72 | 197.33 | 622.7 | 26.0 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | do_no_harm | 0.2996 | 0.0794 | 190.89 | 190.89 | 197.40 | 622.6 | 25.9 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | buffer | 0.2983 | 0.0922 | 191.05 | 191.05 | 197.68 | 623.9 | 26.1 |
| lcnu | coactivation:budget=4,horizon=1 | probation | 0.2995 | 0.1035 | 191.12 | 191.12 | 198.58 | 629.9 | 33.1 |
| lcnu | coactivation:budget=4,horizon=1 | pin | 0.2995 | 0.1035 | 191.12 | 191.12 | 198.58 | 629.9 | 33.1 |

### capacity 1024

Belady hit rate 0.6105, stall floor 88.56 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lru | none | same_queue | 0.3424 | 0.0000 | 185.18 | 185.18 | 185.18 | 560.2 | 0.0 |
| lfu | none | same_queue | 0.3671 | 0.0000 | 178.22 | 178.22 | 178.22 | 539.2 | 0.0 |
| s3fifo | none | same_queue | 0.3771 | 0.0000 | 175.40 | 175.40 | 175.40 | 530.6 | 0.0 |
| w_tinylfu | none | same_queue | 0.3704 | 0.0000 | 177.30 | 177.30 | 177.30 | 536.4 | 0.0 |
| lcnu | none | same_queue | 0.4408 | 0.0000 | 157.48 | 157.48 | 157.48 | 476.4 | 0.0 |
| lcnu | oracle:horizon=2* | probation | 0.4418 | 0.5581 | 136.05 | 136.05 | 145.29 | 475.5 | 0.0 |
| lcnu | oracle:horizon=2* | pin | 0.4418 | 0.5581 | 136.06 | 136.06 | 145.29 | 475.5 | 0.0 |
| lcnu | oracle:horizon=2* | do_no_harm | 0.4418 | 0.5581 | 136.06 | 136.06 | 145.29 | 475.5 | 0.0 |
| lcnu | oracle:horizon=1* | buffer | 0.4372 | 0.4814 | 137.51 | 137.51 | 146.68 | 479.4 | 0.0 |
| lcnu | oracle:horizon=2* | same_queue | 0.4416 | 0.0741 | 151.33 | 155.14 | 160.09 | 521.7 | 45.9 |
| lcnu | coactivation:budget=4,horizon=1 | probation | 0.4414 | 0.0608 | 152.96 | 152.96 | 158.00 | 497.8 | 21.8 |
| lcnu | coactivation:budget=4,horizon=1 | pin | 0.4414 | 0.0608 | 152.96 | 152.96 | 158.00 | 497.8 | 21.8 |
| lcnu | coactivation:budget=4,horizon=1 | do_no_harm | 0.4412 | 0.0515 | 153.29 | 153.29 | 158.14 | 497.4 | 21.4 |
| lcnu | coactivation:budget=4,horizon=1 | buffer | 0.4399 | 0.0610 | 153.36 | 153.36 | 158.41 | 499.0 | 21.9 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | probation | 0.4412 | 0.0474 | 153.40 | 153.40 | 157.13 | 489.9 | 13.9 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | pin | 0.4412 | 0.0474 | 153.40 | 153.40 | 157.13 | 489.9 | 13.9 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | do_no_harm | 0.4411 | 0.0413 | 153.62 | 153.62 | 157.23 | 489.8 | 13.7 |

### capacity 2048

Belady hit rate 0.7627, stall floor 45.70 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lru | none | same_queue | 0.5511 | 0.0000 | 126.40 | 126.40 | 126.40 | 382.4 | 0.0 |
| lfu | none | same_queue | 0.5488 | 0.0000 | 127.05 | 127.05 | 127.05 | 384.4 | 0.0 |
| s3fifo | none | same_queue | 0.5668 | 0.0000 | 121.98 | 121.98 | 121.98 | 369.0 | 0.0 |
| w_tinylfu | none | same_queue | 0.5373 | 0.0000 | 130.29 | 130.29 | 130.29 | 394.2 | 0.0 |
| lcnu | none | same_queue | 0.6052 | 0.0000 | 111.17 | 111.17 | 111.17 | 336.3 | 0.0 |
| lcnu | oracle:horizon=4* | probation | 0.6087 | 0.3912 | 89.12 | 89.12 | 98.31 | 333.4 | 0.0 |
| lcnu | oracle:horizon=4* | pin | 0.6086 | 0.3912 | 89.14 | 89.14 | 98.33 | 333.4 | 0.0 |
| lcnu | oracle:horizon=4* | do_no_harm | 0.6086 | 0.3912 | 89.14 | 89.14 | 98.33 | 333.4 | 0.0 |
| lcnu | oracle:horizon=2* | buffer | 0.6013 | 0.3498 | 91.48 | 91.50 | 100.54 | 339.7 | 0.0 |
| s3fifo | oracle:horizon=2* | same_queue | 0.5665 | 0.4326 | 101.18 | 101.18 | 110.28 | 369.3 | 0.0 |
| s3fifo | oracle:horizon=4* | pin | 0.5618 | 0.4374 | 102.30 | 102.30 | 111.51 | 373.3 | 0.0 |
| s3fifo | oracle:horizon=4* | do_no_harm | 0.5618 | 0.4374 | 102.30 | 102.30 | 111.51 | 373.3 | 0.0 |
| s3fifo | oracle:horizon=4* | probation | 0.5618 | 0.4374 | 102.30 | 102.30 | 111.52 | 373.3 | 0.0 |
| s3fifo | oracle:horizon=2* | buffer | 0.5617 | 0.3968 | 102.48 | 102.48 | 111.62 | 373.4 | 0.0 |
| lru | oracle:horizon=2* | same_queue | 0.5515 | 0.4485 | 105.57 | 105.57 | 114.60 | 382.1 | 0.0 |
| lcnu | oracle:horizon=2* | same_queue | 0.6065 | 0.0631 | 105.76 | 107.04 | 112.73 | 382.3 | 47.1 |
| lru | oracle:horizon=4* | probation | 0.5490 | 0.4502 | 105.98 | 105.98 | 115.13 | 384.2 | 0.0 |

### capacity 4096

Belady hit rate 0.8999, stall floor 7.05 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lru | none | same_queue | 0.7593 | 0.0000 | 67.77 | 67.77 | 67.77 | 205.0 | 0.0 |
| lfu | none | same_queue | 0.7872 | 0.0000 | 59.91 | 59.91 | 59.91 | 181.2 | 0.0 |
| s3fifo | none | same_queue | 0.7883 | 0.0000 | 59.62 | 59.62 | 59.62 | 180.4 | 0.0 |
| w_tinylfu | none | same_queue | 0.7705 | 0.0000 | 64.62 | 64.62 | 64.62 | 195.5 | 0.0 |
| lcnu | none | same_queue | 0.7988 | 0.0000 | 56.66 | 56.66 | 56.66 | 171.4 | 0.0 |
| lcnu | oracle:horizon=4* | probation | 0.8008 | 0.1987 | 37.81 | 37.81 | 45.11 | 169.7 | 0.0 |
| lcnu | oracle:horizon=4* | pin | 0.8007 | 0.1987 | 37.86 | 37.86 | 45.16 | 169.8 | 0.0 |
| lcnu | oracle:horizon=4* | do_no_harm | 0.8007 | 0.1987 | 37.86 | 37.86 | 45.16 | 169.8 | 0.0 |
| w_tinylfu | oracle:horizon=4* | pin | 0.7953 | 0.2045 | 39.00 | 39.01 | 46.58 | 174.4 | 0.0 |
| w_tinylfu | oracle:horizon=4* | do_no_harm | 0.7953 | 0.2045 | 39.00 | 39.01 | 46.58 | 174.4 | 0.0 |
| w_tinylfu | oracle:horizon=4* | probation | 0.7953 | 0.2045 | 39.01 | 39.01 | 46.58 | 174.4 | 0.0 |
| lcnu | oracle:horizon=4* | buffer | 0.7942 | 0.2008 | 39.50 | 39.55 | 46.91 | 175.3 | 0.0 |
| lfu | oracle:horizon=4* | same_queue | 0.7870 | 0.2129 | 40.47 | 40.47 | 48.57 | 181.4 | 0.0 |
| lfu | oracle:horizon=4* | probation | 0.7868 | 0.2131 | 40.53 | 40.53 | 48.63 | 181.6 | 0.0 |
| lfu | oracle:horizon=4* | pin | 0.7868 | 0.2131 | 40.53 | 40.53 | 48.63 | 181.6 | 0.0 |
| lfu | oracle:horizon=4* | do_no_harm | 0.7868 | 0.2131 | 40.53 | 40.53 | 48.63 | 181.6 | 0.0 |
| s3fifo | oracle:horizon=4* | probation | 0.7851 | 0.2145 | 41.48 | 41.48 | 49.30 | 183.1 | 0.0 |

### capacity 6016

Belady hit rate 0.9600, stall floor 0.00 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lru | none | same_queue | 0.8777 | 0.0000 | 34.43 | 34.43 | 34.43 | 104.2 | 0.0 |
| lfu | none | same_queue | 0.9149 | 0.0000 | 23.96 | 23.96 | 23.96 | 72.5 | 0.0 |
| s3fifo | none | same_queue | 0.9097 | 0.0000 | 25.43 | 25.43 | 25.43 | 76.9 | 0.0 |
| w_tinylfu | none | same_queue | 0.9083 | 0.0000 | 25.82 | 25.82 | 25.82 | 78.1 | 0.0 |
| lcnu | none | same_queue | 0.9047 | 0.0000 | 26.83 | 26.83 | 26.83 | 81.2 | 0.0 |
| lfu | oracle:horizon=4* | same_queue | 0.9147 | 0.0853 | 12.79 | 12.79 | 16.29 | 72.7 | 0.0 |
| lfu | oracle:horizon=4* | pin | 0.9145 | 0.0854 | 12.82 | 12.82 | 16.32 | 72.8 | 0.0 |
| lfu | oracle:horizon=4* | do_no_harm | 0.9145 | 0.0854 | 12.82 | 12.82 | 16.32 | 72.8 | 0.0 |
| lfu | oracle:horizon=4* | probation | 0.9145 | 0.0854 | 12.82 | 12.82 | 16.32 | 72.8 | 0.0 |
| lfu | oracle:horizon=4* | buffer | 0.9119 | 0.0879 | 13.18 | 13.18 | 16.86 | 75.1 | 0.0 |
| w_tinylfu | oracle:horizon=4* | same_queue | 0.9078 | 0.0721 | 14.03 | 18.26 | 18.06 | 79.5 | 0.9 |
| w_tinylfu | oracle:horizon=4* | buffer | 0.9049 | 0.0946 | 14.29 | 14.29 | 18.42 | 81.0 | 0.0 |
| s3fifo | oracle:horizon=4* | pin | 0.9079 | 0.0918 | 15.12 | 15.12 | 18.33 | 78.5 | 0.0 |
| s3fifo | oracle:horizon=4* | do_no_harm | 0.9079 | 0.0918 | 15.12 | 15.12 | 18.33 | 78.5 | 0.0 |
| s3fifo | oracle:horizon=4* | probation | 0.9079 | 0.0919 | 15.12 | 15.12 | 18.34 | 78.5 | 0.0 |
| s3fifo | oracle:horizon=4* | buffer | 0.9067 | 0.0929 | 15.21 | 15.21 | 18.53 | 79.5 | 0.0 |
| w_tinylfu | oracle:horizon=4* | pin | 0.9056 | 0.0941 | 15.91 | 15.91 | 19.09 | 80.4 | 0.0 |

`*` = trace-oracle prefetcher (upper bound). Full metrics: CSV.
