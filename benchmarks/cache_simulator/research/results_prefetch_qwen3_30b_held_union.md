# Prefetch x eviction - qwen3-30b-a3b / mmlu150-greedy64/decode+prefill-union/held

REAL SGLang `per_token` routing, 75 requests, 2086506 accesses (4800 decode tokens). Commit `db1ef24`.
Stall = timing-model stall per decode token (nominal: measured per-layer compute, measured PCIe 5.0 link, demand priority).

### capacity 128

Belady hit rate 0.2790, stall floor 42.58 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lru | none | same_queue | 0.0000 | 0.0000 | 71.84 | 71.84 | 71.84 | 434.7 | 0.0 |
| lfu | none | same_queue | 0.1189 | 0.0000 | 63.30 | 63.30 | 63.30 | 383.0 | 0.0 |
| s3fifo | none | same_queue | 0.0000 | 0.0000 | 71.84 | 71.84 | 71.84 | 434.7 | 0.0 |
| w_tinylfu | none | same_queue | 0.1556 | 0.0000 | 60.66 | 60.66 | 60.66 | 367.0 | 0.0 |
| lcnu | none | same_queue | 0.1973 | 0.0000 | 57.67 | 57.67 | 57.67 | 348.9 | 0.0 |
| lcnu | oracle:horizon=1* | pin | 0.1877 | 0.8017 | 49.14 | 49.14 | 55.27 | 353.1 | 0.0 |
| lcnu | oracle:horizon=1* | do_no_harm | 0.1867 | 0.8132 | 49.21 | 49.21 | 55.34 | 353.6 | 0.0 |
| lcnu | oracle:horizon=1* | probation | 0.1867 | 0.8133 | 49.22 | 49.22 | 55.34 | 353.6 | 0.0 |
| lcnu | oracle:horizon=1* | buffer | 0.1776 | 0.7370 | 49.87 | 49.87 | 55.99 | 357.5 | 0.0 |
| w_tinylfu | oracle:horizon=1* | buffer | 0.1390 | 0.7746 | 52.64 | 52.64 | 58.76 | 374.3 | 0.0 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | probation | 0.1946 | 0.1738 | 53.60 | 53.60 | 57.09 | 374.4 | 24.3 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | pin | 0.1946 | 0.1738 | 53.60 | 53.60 | 57.09 | 374.4 | 24.3 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | do_no_harm | 0.1951 | 0.1505 | 53.62 | 53.62 | 57.08 | 373.8 | 23.9 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | buffer | 0.1926 | 0.1750 | 53.73 | 53.73 | 57.22 | 375.2 | 24.3 |
| lcnu | coactivation:budget=4,horizon=1 | do_no_harm | 0.1953 | 0.1450 | 54.03 | 54.03 | 57.40 | 377.4 | 27.6 |
| lcnu | coactivation:budget=4,horizon=1 | probation | 0.1945 | 0.1790 | 54.05 | 54.05 | 57.40 | 378.7 | 28.6 |
| lcnu | coactivation:budget=4,horizon=1 | pin | 0.1945 | 0.1790 | 54.05 | 54.05 | 57.40 | 378.7 | 28.6 |

### capacity 256

Belady hit rate 0.4607, stall floor 29.52 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lru | none | same_queue | 0.0000 | 0.0000 | 71.84 | 71.84 | 71.84 | 434.7 | 0.0 |
| lfu | none | same_queue | 0.2062 | 0.0000 | 57.03 | 57.03 | 57.03 | 345.1 | 0.0 |
| s3fifo | none | same_queue | 0.0450 | 0.0000 | 68.61 | 68.61 | 68.61 | 415.1 | 0.0 |
| w_tinylfu | none | same_queue | 0.2537 | 0.0000 | 53.62 | 53.62 | 53.62 | 324.4 | 0.0 |
| lcnu | none | same_queue | 0.3144 | 0.0000 | 49.26 | 49.26 | 49.26 | 298.0 | 0.0 |
| lcnu | oracle:horizon=1* | do_no_harm | 0.3091 | 0.6908 | 40.44 | 40.44 | 46.54 | 300.3 | 0.0 |
| lcnu | oracle:horizon=1* | probation | 0.3091 | 0.6908 | 40.44 | 40.44 | 46.54 | 300.3 | 0.0 |
| lcnu | oracle:horizon=1* | pin | 0.3091 | 0.6908 | 40.44 | 40.44 | 46.54 | 300.3 | 0.0 |
| lcnu | oracle:horizon=1* | buffer | 0.3025 | 0.6144 | 40.91 | 40.91 | 47.02 | 303.2 | 0.0 |
| w_tinylfu | oracle:horizon=1* | buffer | 0.2441 | 0.6736 | 45.10 | 45.10 | 51.22 | 328.6 | 0.0 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | probation | 0.3140 | 0.1121 | 45.95 | 45.95 | 48.90 | 317.0 | 18.8 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | pin | 0.3140 | 0.1121 | 45.95 | 45.95 | 48.90 | 317.0 | 18.8 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | do_no_harm | 0.3141 | 0.0975 | 46.01 | 46.01 | 48.91 | 316.6 | 18.4 |
| lcnu | hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 | buffer | 0.3114 | 0.1130 | 46.12 | 46.12 | 49.08 | 318.2 | 18.9 |
| lcnu | coactivation:budget=4,horizon=1 | probation | 0.3139 | 0.1264 | 46.15 | 46.15 | 49.17 | 323.1 | 24.8 |
| lcnu | coactivation:budget=4,horizon=1 | pin | 0.3139 | 0.1264 | 46.15 | 46.15 | 49.17 | 323.1 | 24.8 |
| lcnu | coactivation:budget=4,horizon=1 | do_no_harm | 0.3141 | 0.1044 | 46.21 | 46.21 | 49.20 | 322.1 | 24.0 |

### capacity 512

Belady hit rate 0.6200, stall floor 18.08 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lru | none | same_queue | 0.3877 | 0.0000 | 43.99 | 43.99 | 43.99 | 266.2 | 0.0 |
| lfu | none | same_queue | 0.3301 | 0.0000 | 48.13 | 48.13 | 48.13 | 291.2 | 0.0 |
| s3fifo | none | same_queue | 0.3873 | 0.0000 | 44.02 | 44.02 | 44.02 | 266.3 | 0.0 |
| w_tinylfu | none | same_queue | 0.3715 | 0.0000 | 45.16 | 45.16 | 45.16 | 273.2 | 0.0 |
| lcnu | none | same_queue | 0.4530 | 0.0000 | 39.30 | 39.30 | 39.30 | 237.8 | 0.0 |
| lcnu | oracle:horizon=2* | probation | 0.4500 | 0.5499 | 30.32 | 30.32 | 36.42 | 239.1 | 0.0 |
| lcnu | oracle:horizon=2* | pin | 0.4500 | 0.5499 | 30.32 | 30.32 | 36.42 | 239.1 | 0.0 |
| lcnu | oracle:horizon=2* | do_no_harm | 0.4500 | 0.5499 | 30.32 | 30.32 | 36.42 | 239.1 | 0.0 |
| lcnu | oracle:horizon=1* | buffer | 0.4463 | 0.4730 | 30.77 | 30.77 | 36.73 | 240.7 | 0.0 |
| lru | oracle:horizon=2* | same_queue | 0.3879 | 0.6120 | 34.77 | 34.77 | 40.88 | 266.1 | 0.0 |
| lru | oracle:horizon=2* | pin | 0.3870 | 0.6125 | 34.84 | 34.84 | 40.95 | 266.5 | 0.0 |
| lru | oracle:horizon=2* | do_no_harm | 0.3870 | 0.6125 | 34.84 | 34.84 | 40.95 | 266.5 | 0.0 |
| lru | oracle:horizon=2* | probation | 0.3870 | 0.6125 | 34.84 | 34.84 | 40.95 | 266.5 | 0.0 |
| lru | oracle:horizon=1* | buffer | 0.3866 | 0.5247 | 34.95 | 34.95 | 40.99 | 266.6 | 0.0 |
| s3fifo | oracle:horizon=1* | buffer | 0.3815 | 0.5415 | 35.32 | 35.32 | 41.36 | 268.9 | 0.0 |
| s3fifo | oracle:horizon=2* | pin | 0.3779 | 0.6211 | 35.48 | 35.48 | 41.60 | 270.4 | 0.0 |
| s3fifo | oracle:horizon=2* | do_no_harm | 0.3779 | 0.6211 | 35.48 | 35.48 | 41.60 | 270.4 | 0.0 |

### capacity 1024

Belady hit rate 0.7677, stall floor 7.47 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lru | none | same_queue | 0.5714 | 0.0000 | 30.79 | 30.79 | 30.79 | 186.3 | 0.0 |
| lfu | none | same_queue | 0.5141 | 0.0000 | 34.91 | 34.91 | 34.91 | 211.2 | 0.0 |
| s3fifo | none | same_queue | 0.5706 | 0.0000 | 30.85 | 30.85 | 30.85 | 186.6 | 0.0 |
| w_tinylfu | none | same_queue | 0.5467 | 0.0000 | 32.57 | 32.57 | 32.57 | 197.0 | 0.0 |
| lcnu | none | same_queue | 0.6121 | 0.0000 | 27.87 | 27.87 | 27.87 | 168.6 | 0.0 |
| lcnu | oracle:horizon=4* | probation | 0.6127 | 0.3870 | 18.75 | 18.75 | 24.74 | 168.4 | 0.0 |
| lcnu | oracle:horizon=4* | pin | 0.6126 | 0.3870 | 18.76 | 18.76 | 24.74 | 168.4 | 0.0 |
| lcnu | oracle:horizon=4* | do_no_harm | 0.6126 | 0.3870 | 18.76 | 18.76 | 24.74 | 168.4 | 0.0 |
| lcnu | oracle:horizon=2* | buffer | 0.6045 | 0.3474 | 19.59 | 19.60 | 25.38 | 171.9 | 0.0 |
| lru | oracle:horizon=2* | same_queue | 0.5719 | 0.4280 | 22.03 | 22.03 | 27.74 | 186.1 | 0.0 |
| lru | oracle:horizon=4* | probation | 0.5674 | 0.4313 | 22.10 | 22.11 | 28.00 | 188.1 | 0.0 |
| lru | oracle:horizon=4* | do_no_harm | 0.5673 | 0.4315 | 22.11 | 22.11 | 28.01 | 188.1 | 0.0 |
| lru | oracle:horizon=4* | pin | 0.5673 | 0.4315 | 22.11 | 22.11 | 28.01 | 188.1 | 0.0 |
| s3fifo | oracle:horizon=2* | same_queue | 0.5669 | 0.4089 | 22.31 | 22.41 | 28.13 | 188.6 | 0.4 |
| s3fifo | oracle:horizon=4* | probation | 0.5609 | 0.4375 | 22.42 | 22.43 | 28.45 | 190.9 | 0.0 |
| s3fifo | oracle:horizon=4* | do_no_harm | 0.5608 | 0.4377 | 22.43 | 22.44 | 28.46 | 190.9 | 0.0 |
| s3fifo | oracle:horizon=4* | pin | 0.5608 | 0.4377 | 22.43 | 22.44 | 28.46 | 190.9 | 0.0 |

### capacity 2048

Belady hit rate 0.9048, stall floor 0.00 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lru | none | same_queue | 0.7699 | 0.0000 | 16.53 | 16.53 | 16.53 | 100.0 | 0.0 |
| lfu | none | same_queue | 0.7688 | 0.0000 | 16.61 | 16.61 | 16.61 | 100.5 | 0.0 |
| s3fifo | none | same_queue | 0.8012 | 0.0000 | 14.28 | 14.28 | 14.28 | 86.4 | 0.0 |
| w_tinylfu | none | same_queue | 0.7977 | 0.0000 | 14.53 | 14.53 | 14.53 | 87.9 | 0.0 |
| lcnu | none | same_queue | 0.8087 | 0.0000 | 13.74 | 13.74 | 13.74 | 83.2 | 0.0 |
| w_tinylfu | oracle:horizon=4* | pin | 0.8075 | 0.1921 | 7.15 | 7.16 | 11.02 | 83.7 | 0.0 |
| w_tinylfu | oracle:horizon=4* | do_no_harm | 0.8075 | 0.1921 | 7.15 | 7.16 | 11.02 | 83.7 | 0.0 |
| w_tinylfu | oracle:horizon=4* | probation | 0.8075 | 0.1920 | 7.15 | 7.15 | 11.02 | 83.7 | 0.0 |
| lcnu | oracle:horizon=4* | probation | 0.8089 | 0.1903 | 7.38 | 7.38 | 10.95 | 83.1 | 0.0 |
| lcnu | oracle:horizon=4* | pin | 0.8087 | 0.1903 | 7.40 | 7.40 | 10.97 | 83.2 | 0.0 |
| lcnu | oracle:horizon=4* | do_no_harm | 0.8087 | 0.1903 | 7.40 | 7.40 | 10.97 | 83.2 | 0.0 |
| w_tinylfu | oracle:horizon=4* | buffer | 0.7853 | 0.2144 | 7.76 | 7.76 | 12.49 | 93.3 | 0.0 |
| lcnu | oracle:horizon=4* | buffer | 0.7994 | 0.1955 | 7.79 | 7.82 | 11.58 | 87.2 | 0.0 |
| s3fifo | oracle:horizon=4* | pin | 0.7935 | 0.2057 | 7.86 | 7.87 | 11.97 | 89.8 | 0.0 |
| s3fifo | oracle:horizon=4* | do_no_harm | 0.7935 | 0.2057 | 7.86 | 7.87 | 11.97 | 89.8 | 0.0 |
| s3fifo | oracle:horizon=4* | probation | 0.7934 | 0.2057 | 7.86 | 7.87 | 11.98 | 89.8 | 0.0 |
| w_tinylfu | oracle:horizon=4* | same_queue | 0.7966 | 0.1361 | 7.92 | 9.45 | 12.08 | 93.1 | 4.6 |

### capacity 3072

Belady hit rate 0.9667, stall floor 0.00 ms/token (nominal timing).

| Evictor | Prefetcher | Admission | Hit rate | Covered | Stall ms/tok | FIFO | Roofline | Transfers/tok | Wasted/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lru | none | same_queue | 0.8919 | 0.0000 | 7.77 | 7.77 | 7.77 | 47.0 | 0.0 |
| lfu | none | same_queue | 0.9094 | 0.0000 | 6.51 | 6.51 | 6.51 | 39.4 | 0.0 |
| s3fifo | none | same_queue | 0.9262 | 0.0000 | 5.30 | 5.30 | 5.30 | 32.1 | 0.0 |
| w_tinylfu | none | same_queue | 0.9353 | 0.0000 | 4.65 | 4.65 | 4.65 | 28.1 | 0.0 |
| lcnu | none | same_queue | 0.9174 | 0.0000 | 5.93 | 5.93 | 5.93 | 35.9 | 0.0 |
| w_tinylfu | oracle:horizon=4* | buffer | 0.9294 | 0.0702 | 2.00 | 2.01 | 3.26 | 30.7 | 0.0 |
| w_tinylfu | oracle:horizon=4* | same_queue | 0.9346 | 0.0447 | 2.04 | 2.22 | 3.07 | 29.7 | 1.2 |
| lfu | oracle:horizon=4* | same_queue | 0.9099 | 0.0901 | 2.66 | 2.66 | 4.42 | 39.2 | 0.0 |
| lfu | oracle:horizon=4* | pin | 0.9095 | 0.0904 | 2.68 | 2.68 | 4.44 | 39.3 | 0.0 |
| lfu | oracle:horizon=4* | do_no_harm | 0.9095 | 0.0904 | 2.68 | 2.68 | 4.44 | 39.3 | 0.0 |
| lfu | oracle:horizon=4* | probation | 0.9095 | 0.0904 | 2.68 | 2.68 | 4.44 | 39.3 | 0.0 |
| lfu | oracle:horizon=4* | buffer | 0.9032 | 0.0965 | 2.78 | 2.78 | 4.78 | 42.1 | 0.0 |
| s3fifo | oracle:horizon=4* | pin | 0.9232 | 0.0764 | 2.79 | 2.79 | 3.80 | 33.4 | 0.0 |
| s3fifo | oracle:horizon=4* | do_no_harm | 0.9232 | 0.0764 | 2.79 | 2.79 | 3.80 | 33.4 | 0.0 |
| s3fifo | oracle:horizon=4* | probation | 0.9232 | 0.0764 | 2.79 | 2.79 | 3.80 | 33.4 | 0.0 |
| s3fifo | oracle:horizon=4* | buffer | 0.9210 | 0.0785 | 2.79 | 2.79 | 3.88 | 34.3 | 0.0 |
| s3fifo | oracle:horizon=4* | same_queue | 0.9187 | 0.0808 | 2.88 | 2.88 | 4.02 | 35.3 | 0.0 |

`*` = trace-oracle prefetcher (upper bound). Full metrics: CSV.
