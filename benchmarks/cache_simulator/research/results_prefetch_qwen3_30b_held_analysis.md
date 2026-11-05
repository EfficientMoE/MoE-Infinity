## qwen3-30b-a3b / mmlu150-greedy64/decode/held

### Per capacity (nominal timing, ms of stall per decode token)

| Capacity | Belady hit | Floor | Best evictor alone | Stall | Best online pair | Stall | vs alone | Oracle bound | Stall |
| ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | --- | ---: |
| 128 | 0.3079 | 34.85 | lcnu (0.2179) | 49.64 | lcnu + hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 / probation | 45.62 | -8.1% | lcnu + oracle-h1 / probation | 41.09 |
| 256 | 0.5052 | 22.33 | lcnu (0.3461) | 41.50 | lcnu + hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 / probation | 38.22 | -7.9% | lcnu + oracle-h1 / probation | 32.63 |
| 512 | 0.6718 | 11.76 | lcnu (0.4976) | 31.89 | lcnu + coactivation:budget=4,horizon=1 / probation | 29.34 | -8.0% | lcnu + oracle-h2 / probation | 22.90 |
| 1024 | 0.8158 | 2.61 | lcnu (0.6697) | 20.96 | lcnu + hybrid:alpha=0.4,budget=8,horizon=1,weights=1/1/16 / probation | 18.52 | -11.6% | lcnu + oracle-h4 / probation | 11.98 |
| 2048 | 0.9379 | 0.00 | lcnu (0.8662) | 8.49 | lcnu + coactivation:budget=8,horizon=1 / probation | 7.79 | -8.2% | lcnu + oracle-h4 / pin | 1.92 |
| 3072 | 0.9818 | 0.00 | s3fifo (0.9590) | 2.60 | lcnu + coactivation:budget=16,horizon=2 / pin | 2.53 | -2.7% | s3fifo + oracle-h4 / probation | 0.15 |

### Best online pair per admission mode (nominal)

| Capacity | same_queue | probation | pin | buffer | do_no_harm |
| ---: | --- | --- | --- | --- | --- |
| 128 | 48.14 lcnu+hybrid | 45.62 lcnu+hybrid | 45.62 lcnu+hybrid | 45.75 lcnu+hybrid | 45.65 lcnu+hybrid |
| 256 | 40.57 lcnu+coactivation | 38.22 lcnu+hybrid | 38.22 lcnu+hybrid | 38.40 lcnu+hybrid | 38.31 lcnu+hybrid |
| 512 | 30.94 lcnu+coactivation | 29.34 lcnu+coactivation | 29.34 lcnu+coactivation | 29.52 lcnu+coactivation | 29.48 lcnu+coactivation |
| 1024 | 20.13 lcnu+coactivation | 18.52 lcnu+hybrid | 18.52 lcnu+hybrid | 18.77 lcnu+hybrid | 18.78 lcnu+hybrid |
| 2048 | 7.98 lcnu+coactivation | 7.79 lcnu+coactivation | 7.79 lcnu+coactivation | 7.87 lcnu+coactivation | 7.84 lcnu+coactivation |
| 3072 | 2.57 s3fifo+hybrid | 2.58 s3fifo+hybrid | 2.53 lcnu+coactivation | 2.61 s3fifo+prev_token | 2.59 s3fifo+topk_ema |

### Separability (nominal)

| Capacity | Admission | Joint optimum | Stall | Independent (evictor alone + prefetcher on lru) | Regret | Sequential (best prefetcher for that evictor) | Regret | Evictor / prefetcher / interaction share of log-stall variance |
| ---: | --- | --- | ---: | --- | ---: | --- | ---: | --- |
| 128 | best per pair | lcnu + hybrid | 45.62 | lcnu + hybrid = 45.62 | +0.0% | lcnu + hybrid = 45.62 | +0.0% | 83% / 15% / 2% |
| 128 | same_queue | lcnu + hybrid | 48.14 | lcnu + hybrid = 48.14 | +0.0% | lcnu + hybrid = 48.14 | +0.0% | 84% / 11% / 6% |
| 128 | probation | lcnu + hybrid | 45.62 | lcnu + hybrid = 45.62 | +0.0% | lcnu + hybrid = 45.62 | +0.0% | 63% / 30% / 7% |
| 128 | pin | lcnu + hybrid | 45.62 | lcnu + hybrid = 45.62 | +0.0% | lcnu + hybrid = 45.62 | +0.0% | 63% / 30% / 7% |
| 128 | buffer | lcnu + hybrid | 45.75 | lcnu + hybrid = 45.75 | +0.0% | lcnu + hybrid = 45.75 | +0.0% | 65% / 35% / 0% |
| 128 | do_no_harm | lcnu + hybrid | 45.65 | lcnu + hybrid = 45.65 | +0.0% | lcnu + hybrid = 45.65 | +0.0% | 83% / 9% / 8% |
| 256 | best per pair | lcnu + hybrid | 38.22 | lcnu + hybrid = 38.22 | +0.0% | lcnu + hybrid = 38.22 | +0.0% | 90% / 8% / 2% |
| 256 | same_queue | lcnu + coactivation | 40.57 | lcnu + hybrid = 40.63 | +0.1% | lcnu + coactivation = 40.57 | +0.0% | 84% / 13% / 4% |
| 256 | probation | lcnu + hybrid | 38.22 | lcnu + hybrid = 38.22 | +0.0% | lcnu + hybrid = 38.22 | +0.0% | 78% / 16% / 5% |
| 256 | pin | lcnu + hybrid | 38.22 | lcnu + hybrid = 38.22 | +0.0% | lcnu + hybrid = 38.22 | +0.0% | 78% / 16% / 5% |
| 256 | buffer | lcnu + hybrid | 38.40 | lcnu + hybrid = 38.40 | +0.0% | lcnu + hybrid = 38.40 | +0.0% | 79% / 20% / 1% |
| 256 | do_no_harm | lcnu + hybrid | 38.31 | lcnu + hybrid = 38.31 | +0.0% | lcnu + hybrid = 38.31 | +0.0% | 89% / 4% / 7% |
| 512 | best per pair | lcnu + coactivation | 29.34 | lcnu + hybrid = 29.55 | +0.7% | lcnu + coactivation = 29.34 | +0.0% | 71% / 25% / 4% |
| 512 | same_queue | lcnu + coactivation | 30.94 | lcnu + hybrid = 30.98 | +0.1% | lcnu + coactivation = 30.94 | +0.0% | 60% / 33% / 7% |
| 512 | probation | lcnu + coactivation | 29.34 | lcnu + hybrid = 29.55 | +0.7% | lcnu + coactivation = 29.34 | +0.0% | 60% / 33% / 7% |
| 512 | pin | lcnu + coactivation | 29.34 | lcnu + hybrid = 29.55 | +0.7% | lcnu + coactivation = 29.34 | +0.0% | 60% / 33% / 7% |
| 512 | buffer | lcnu + coactivation | 29.52 | lcnu + hybrid = 29.71 | +0.7% | lcnu + coactivation = 29.52 | +0.0% | 43% / 57% / 0% |
| 512 | do_no_harm | lcnu + coactivation | 29.48 | lcnu + hybrid = 29.66 | +0.6% | lcnu + coactivation = 29.48 | +0.0% | 77% / 11% / 12% |
| 1024 | best per pair | lcnu + hybrid | 18.52 | lcnu + coactivation = 18.60 | +0.4% | lcnu + hybrid = 18.52 | +0.0% | 79% / 20% / 1% |
| 1024 | same_queue | lcnu + coactivation | 20.13 | lcnu + coactivation = 20.13 | +0.0% | lcnu + coactivation = 20.13 | +0.0% | 59% / 29% / 12% |
| 1024 | probation | lcnu + hybrid | 18.52 | lcnu + hybrid = 18.52 | +0.0% | lcnu + hybrid = 18.52 | +0.0% | 61% / 35% / 5% |
| 1024 | pin | lcnu + hybrid | 18.52 | lcnu + hybrid = 18.52 | +0.0% | lcnu + hybrid = 18.52 | +0.0% | 62% / 33% / 5% |
| 1024 | buffer | lcnu + hybrid | 18.77 | lcnu + hybrid = 18.77 | +0.0% | lcnu + hybrid = 18.77 | +0.0% | 52% / 47% / 1% |
| 1024 | do_no_harm | lcnu + hybrid | 18.78 | lcnu + hybrid = 18.78 | +0.0% | lcnu + hybrid = 18.78 | +0.0% | 83% / 15% / 2% |
| 2048 | best per pair | lcnu + coactivation | 7.79 | lcnu + coactivation = 7.79 | +0.0% | lcnu + coactivation = 7.79 | +0.0% | 82% / 16% / 2% |
| 2048 | same_queue | lcnu + coactivation | 7.98 | lcnu + coactivation = 7.98 | +0.0% | lcnu + coactivation = 7.98 | +0.0% | 53% / 34% / 12% |
| 2048 | probation | lcnu + coactivation | 7.79 | lcnu + hybrid = 7.93 | +1.7% | lcnu + coactivation = 7.79 | +0.0% | 63% / 35% / 2% |
| 2048 | pin | lcnu + coactivation | 7.79 | lcnu + coactivation = 7.79 | +0.0% | lcnu + coactivation = 7.79 | +0.0% | 63% / 35% / 2% |
| 2048 | buffer | lcnu + coactivation | 7.87 | lcnu + hybrid = 8.00 | +1.7% | lcnu + coactivation = 7.87 | +0.0% | 62% / 37% / 1% |
| 2048 | do_no_harm | lcnu + coactivation | 7.84 | lcnu + coactivation = 7.84 | +0.0% | lcnu + coactivation = 7.84 | +0.0% | 88% / 10% / 2% |
| 3072 | best per pair | lcnu + coactivation | 2.53 | s3fifo + hybrid = 2.57 | +1.6% | s3fifo + hybrid = 2.57 | +1.6% | 94% / 4% / 2% |
| 3072 | same_queue | s3fifo + hybrid | 2.57 | s3fifo + hybrid = 2.57 | +0.0% | s3fifo + hybrid = 2.57 | +0.0% | 54% / 29% / 17% |
| 3072 | probation | s3fifo + hybrid | 2.58 | s3fifo + hybrid = 2.58 | +0.0% | s3fifo + hybrid = 2.58 | +0.0% | 84% / 15% / 1% |
| 3072 | pin | lcnu + coactivation | 2.53 | s3fifo + none = 2.60 | +2.8% | s3fifo + prev_token = 2.60 | +2.8% | 84% / 14% / 2% |
| 3072 | buffer | s3fifo + none | 2.60 | s3fifo + hybrid = 2.63 | +0.9% | s3fifo + none = 2.60 | +0.0% | 84% / 15% / 1% |
| 3072 | do_no_harm | s3fifo + topk_ema | 2.59 | s3fifo + hybrid = 2.60 | +0.3% | s3fifo + topk_ema = 2.59 | +0.0% | 96% / 3% / 1% |

### Admission modes over all online pairs (nominal)

| Admission | Pairs | Best mode for | Mean stall vs best mode | Worse than no prefetch | Mean demand hit-rate change (points) | Mean wasted transfers / token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same_queue | 210 | 43 | +7.7% | 87 (41%) | -0.79 | 27.6 |
| probation | 210 | 73 | +5.1% | 66 (31%) | -1.08 | 30.7 |
| pin | 210 | 72 | +5.2% | 71 (34%) | -1.10 | 31.0 |
| buffer | 210 | 34 | +3.7% | 79 (38%) | +0.08 | 28.2 |
| do_no_harm | 210 | 97 | +1.5% | 45 (21%) | -0.43 | 18.6 |

### Online combinations worse than the evictor alone (nominal)

| Admission | cap=128 | cap=256 | cap=512 | cap=1024 | cap=2048 | cap=3072 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same_queue | 14/35 | 13/35 | 17/35 | 16/35 | 10/35 | 17/35 |
| probation | 10/35 | 10/35 | 11/35 | 17/35 | 9/35 | 9/35 |
| pin | 10/35 | 10/35 | 11/35 | 17/35 | 9/35 | 14/35 |
| buffer | 7/35 | 8/35 | 12/35 | 14/35 | 16/35 | 22/35 |
| do_no_harm | 6/35 | 6/35 | 4/35 | 9/35 | 8/35 | 12/35 |

### Online combinations worse than the evictor alone (FIFO link)

| Admission | cap=128 | cap=256 | cap=512 | cap=1024 | cap=2048 | cap=3072 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same_queue | 14/35 | 13/35 | 17/35 | 16/35 | 10/35 | 22/35 |
| probation | 10/35 | 10/35 | 11/35 | 17/35 | 9/35 | 9/35 |
| pin | 10/35 | 10/35 | 11/35 | 18/35 | 11/35 | 17/35 |
| buffer | 7/35 | 8/35 | 12/35 | 14/35 | 16/35 | 22/35 |
| do_no_harm | 6/35 | 6/35 | 4/35 | 11/35 | 10/35 | 12/35 |

### Does prefetch change the best evictor? (nominal)

| Capacity | Best evictor, no prefetch | Best evictor per prefetcher (best admission) | Kendall tau of evictor ranking vs no prefetch |
| ---: | --- | --- | --- |
| 128 | lcnu | coactivation: lcnu; hybrid: lcnu; prev_token: lcnu; same_id: lcnu; topk_ema: lcnu | coactivation +1.00; hybrid +1.00; prev_token +0.81; same_id +0.81; topk_ema +1.00 |
| 256 | lcnu | coactivation: lcnu; hybrid: lcnu; prev_token: lcnu; same_id: lcnu; topk_ema: lcnu | coactivation +0.90; hybrid +0.90; prev_token +0.90; same_id +0.90; topk_ema +0.90 |
| 512 | lcnu | coactivation: lcnu; hybrid: lcnu; prev_token: lcnu; same_id: lcnu; topk_ema: lcnu | coactivation +1.00; hybrid +1.00; prev_token +1.00; same_id +0.71; topk_ema +0.90 |
| 1024 | lcnu | coactivation: lcnu; hybrid: lcnu; prev_token: lcnu; same_id: lcnu; topk_ema: lcnu | coactivation +1.00; hybrid +1.00; prev_token +1.00; same_id +1.00; topk_ema +1.00 |
| 2048 | lcnu | coactivation: lcnu; hybrid: lcnu; prev_token: lcnu; same_id: lcnu; topk_ema: lcnu | coactivation +0.81; hybrid +0.71; prev_token +0.81; same_id +0.81; topk_ema +0.90 |
| 3072 | s3fifo | coactivation: lcnu; hybrid: lcnu; prev_token: s3fifo; same_id: lcnu; topk_ema: s3fifo | coactivation +0.81; hybrid +0.81; prev_token +0.81; same_id +0.90; topk_ema +0.81 |

### Timing sensitivity

| Capacity | best online pair (nominal) | best online pair (fifo) | best online pair (roofline) | gain vs best alone (nominal) | gain vs best alone (fifo) | gain vs best alone (roofline) |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 128 | lcnu + hybrid / probation | lcnu + hybrid / probation | lcnu + hybrid / do_no_harm | -8.1% | -8.1% | -1.2% |
| 256 | lcnu + hybrid / probation | lcnu + hybrid / probation | lcnu + hybrid / probation | -7.9% | -7.9% | -0.9% |
| 512 | lcnu + coactivation / probation | lcnu + coactivation / probation | lcnu + hybrid / probation | -8.0% | -8.0% | -1.0% |
| 1024 | lcnu + hybrid / probation | lcnu + hybrid / probation | lcnu + hybrid / probation | -11.6% | -11.6% | -1.1% |
| 2048 | lcnu + coactivation / probation | lcnu + coactivation / probation | lcnu + hybrid / pin | -8.2% | -8.2% | -1.6% |
| 3072 | lcnu + coactivation / pin | s3fifo + hybrid / probation | s3fifo + prev_token / probation | -2.7% | -1.0% | -0.0% |

### Evictor x prefetcher at capacity 128 (nominal stall, best admission)

| Evictor | none | coactivation | hybrid | prev_token | same_id | topk_ema |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| layer_lru | 62.47 | 56.70 prob | 56.29 prob | 60.52 do_n | 67.17 do_n | 57.98 buff |
| lcnu | 49.64 | 46.05 do_n | 45.62 prob | 48.23 do_n | 48.97 do_n | 48.84 do_n |
| lfu | 55.20 | 49.78 prob | 49.47 prob | 52.51 do_n | 54.30 do_n | 52.29 do_n |
| lirs | 51.67 | 47.78 buff | 47.41 buff | 50.67 buff | 51.55 do_n | 50.40 buff |
| lru | 63.47 | 57.50 prob | 57.13 same | 60.32 do_n | 62.84 do_n | 59.73 do_n |
| s3fifo | 63.47 | 57.50 prob | 57.13 same | 60.32 do_n | 62.84 do_n | 59.73 do_n |
| w_tinylfu | 52.54 | 48.36 buff | 47.98 buff | 51.42 buff | 52.52 same | 50.97 buff |

### Evictor x prefetcher at capacity 256 (nominal stall, best admission)

| Evictor | none | coactivation | hybrid | prev_token | same_id | topk_ema |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| layer_lru | 58.13 | 53.10 prob | 52.60 prob | 56.20 do_n | 62.49 do_n | 54.14 buff |
| lcnu | 41.50 | 38.43 prob | 38.22 prob | 40.83 do_n | 41.10 do_n | 41.29 do_n |
| lfu | 49.18 | 44.39 prob | 44.15 prob | 47.22 do_n | 48.44 do_n | 47.03 buff |
| lirs | 44.01 | 40.62 prob | 40.33 prob | 43.27 prob | 43.70 do_n | 43.55 do_n |
| lru | 63.47 | 57.50 prob | 57.13 same | 60.37 do_n | 62.93 do_n | 59.53 do_n |
| s3fifo | 60.49 | 49.51 do_n | 49.47 buff | 53.27 buff | 54.14 do_n | 51.07 prob |
| w_tinylfu | 45.95 | 42.28 buff | 41.98 buff | 44.99 buff | 46.13 do_n | 44.97 buff |

### Evictor x prefetcher at capacity 512 (nominal stall, best admission)

| Evictor | none | coactivation | hybrid | prev_token | same_id | topk_ema |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| layer_lru | 35.09 | 32.55 prob | 32.52 prob | 35.09 same | 39.04 do_n | 34.93 do_n |
| lcnu | 31.89 | 29.34 prob | 29.55 prob | 31.50 prob | 31.72 do_n | 31.88 prob |
| lfu | 40.76 | 37.19 prob | 37.06 prob | 39.43 do_n | 40.25 do_n | 39.63 do_n |
| lirs | 35.03 | 32.22 same | 32.10 same | 34.45 prob | 34.81 do_n | 34.96 do_n |
| lru | 35.56 | 32.67 same | 32.61 same | 35.56 same | 35.38 do_n | 35.29 same |
| s3fifo | 36.52 | 33.64 prob | 33.51 prob | 35.89 do_n | 36.35 do_n | 36.37 do_n |
| w_tinylfu | 37.81 | 34.70 buff | 34.55 buff | 36.96 buff | 37.02 do_n | 37.22 buff |

### Evictor x prefetcher at capacity 1024 (nominal stall, best admission)

| Evictor | none | coactivation | hybrid | prev_token | same_id | topk_ema |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| layer_lru | 22.68 | 21.41 same | 21.14 do_n | 22.68 same | 23.49 do_n | 22.68 same |
| lcnu | 20.96 | 18.60 prob | 18.52 prob | 20.96 prob | 20.94 do_n | 20.96 pin |
| lfu | 28.64 | 25.35 prob | 25.16 prob | 27.67 buff | 28.57 do_n | 27.64 buff |
| lirs | 24.51 | 22.21 same | 22.33 same | 24.04 buff | 24.47 do_n | 24.55 buff |
| lru | 22.65 | 21.01 same | 21.13 same | 22.65 same | 22.64 do_n | 22.65 same |
| s3fifo | 23.98 | 21.97 same | 21.90 prob | 23.71 buff | 23.97 do_n | 23.95 pin |
| w_tinylfu | 26.18 | 23.35 do_n | 23.39 do_n | 25.37 buff | 25.56 do_n | 25.52 do_n |

### Evictor x prefetcher at capacity 2048 (nominal stall, best admission)

| Evictor | none | coactivation | hybrid | prev_token | same_id | topk_ema |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| layer_lru | 10.24 | 9.44 do_n | 9.36 prob | 10.24 same | 11.10 do_n | 10.24 same |
| lcnu | 8.49 | 7.79 prob | 7.80 pin | 8.49 same | 8.51 same | 8.49 same |
| lfu | 12.53 | 11.72 prob | 11.38 pin | 11.96 do_n | 12.53 do_n | 12.05 do_n |
| lirs | 10.49 | 9.28 same | 9.15 same | 10.22 prob | 10.53 do_n | 10.51 buff |
| lru | 9.58 | 8.74 same | 8.74 prob | 9.58 same | 9.84 do_n | 9.58 same |
| s3fifo | 8.83 | 8.15 prob | 8.14 prob | 8.81 same | 9.17 do_n | 8.79 do_n |
| w_tinylfu | 10.30 | 8.77 prob | 8.64 pin | 9.69 prob | 10.43 do_n | 9.63 prob |

### Evictor x prefetcher at capacity 3072 (nominal stall, best admission)

| Evictor | none | coactivation | hybrid | prev_token | same_id | topk_ema |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| layer_lru | 3.80 | 3.73 prob | 3.65 prob | 3.80 same | 4.16 do_n | 3.80 same |
| lcnu | 2.80 | 2.53 pin | 2.57 pin | 2.80 same | 2.82 same | 2.80 same |
| lfu | 4.36 | 3.92 pin | 3.79 pin | 4.19 prob | 4.38 do_n | 3.94 do_n |
| lirs | 2.85 | 2.87 do_n | 2.81 prob | 2.82 same | 2.91 do_n | 2.85 prob |
| lru | 3.14 | 3.11 prob | 3.04 same | 3.14 same | 3.32 do_n | 3.14 same |
| s3fifo | 2.60 | 2.60 same | 2.57 same | 2.60 prob | 2.83 do_n | 2.59 do_n |
| w_tinylfu | 2.94 | 2.70 prob | 2.65 prob | 2.75 do_n | 3.01 do_n | 2.76 do_n |
