## qwen3-30b-a3b / mmlu150-greedy64/decode+prefill-union/held

### Per capacity (nominal timing, ms of stall per decode token)

| Capacity | Belady hit | Floor | Best evictor alone | Stall | Best online pair | Stall | vs alone | Oracle bound | Stall |
| ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | --- | ---: |
| 128 | 0.2790 | 42.58 | lcnu (0.1973) | 57.67 | lcnu + hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 / probation | 53.60 | -7.1% | lcnu + oracle-h1 / pin | 49.14 |
| 256 | 0.4607 | 29.52 | lcnu (0.3144) | 49.26 | lcnu + hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 / probation | 45.95 | -6.7% | lcnu + oracle-h1 / do_no_harm | 40.44 |
| 512 | 0.6200 | 18.08 | lcnu (0.4530) | 39.30 | lcnu + coactivation:budget=4,horizon=1 / probation | 36.71 | -6.6% | lcnu + oracle-h2 / probation | 30.32 |
| 1024 | 0.7677 | 7.47 | lcnu (0.6121) | 27.87 | lcnu + hybrid:alpha=0.4,budget=8,horizon=1,weights=1/1/16 / probation | 25.49 | -8.5% | lcnu + oracle-h4 / probation | 18.75 |
| 2048 | 0.9048 | 0.00 | lcnu (0.8087) | 13.74 | w_tinylfu + hybrid:alpha=0.4,budget=8,horizon=1,weights=1/1/16 / probation | 13.17 | -4.2% | w_tinylfu + oracle-h4 / pin | 7.15 |
| 3072 | 0.9667 | 0.00 | w_tinylfu (0.9353) | 4.65 | w_tinylfu + prev_token:budget=2,horizon=1 / probation | 4.51 | -2.9% | w_tinylfu + oracle-h4 / buffer | 2.00 |

### Best online pair per admission mode (nominal)

| Capacity | same_queue | probation | pin | buffer | do_no_harm |
| ---: | --- | --- | --- | --- | --- |
| 128 | 56.13 lcnu+hybrid | 53.60 lcnu+hybrid | 53.60 lcnu+hybrid | 53.73 lcnu+hybrid | 53.62 lcnu+hybrid |
| 256 | 48.28 lcnu+coactivation | 45.95 lcnu+hybrid | 45.95 lcnu+hybrid | 46.12 lcnu+hybrid | 46.01 lcnu+hybrid |
| 512 | 38.34 lcnu+coactivation | 36.71 lcnu+coactivation | 36.71 lcnu+coactivation | 36.88 lcnu+coactivation | 36.81 lcnu+coactivation |
| 1024 | 27.06 lcnu+coactivation | 25.49 lcnu+hybrid | 25.49 lcnu+hybrid | 25.72 lcnu+hybrid | 25.66 lcnu+hybrid |
| 2048 | 13.35 s3fifo+coactivation | 13.17 w_tinylfu+hybrid | 13.19 lcnu+coactivation | 13.27 lcnu+coactivation | 13.22 lcnu+coactivation |
| 3072 | 4.60 w_tinylfu+prev_token | 4.51 w_tinylfu+prev_token | 4.51 w_tinylfu+prev_token | 4.61 w_tinylfu+prev_token | 4.51 w_tinylfu+prev_token |

### Separability (nominal)

| Capacity | Admission | Joint optimum | Stall | Independent (evictor alone + prefetcher on lru) | Regret | Sequential (best prefetcher for that evictor) | Regret | Evictor / prefetcher / interaction share of log-stall variance |
| ---: | --- | --- | ---: | --- | ---: | --- | ---: | --- |
| 128 | best per pair | lcnu + hybrid | 53.60 | lcnu + hybrid = 53.60 | +0.0% | lcnu + hybrid = 53.60 | +0.0% | 85% / 15% / 0% |
| 128 | same_queue | lcnu + hybrid | 56.13 | lcnu + hybrid = 56.13 | +0.0% | lcnu + hybrid = 56.13 | +0.0% | 79% / 14% / 7% |
| 128 | probation | lcnu + hybrid | 53.60 | lcnu + hybrid = 53.60 | +0.0% | lcnu + hybrid = 53.60 | +0.0% | 59% / 35% / 6% |
| 128 | pin | lcnu + hybrid | 53.60 | lcnu + hybrid = 53.60 | +0.0% | lcnu + hybrid = 53.60 | +0.0% | 59% / 35% / 6% |
| 128 | buffer | lcnu + hybrid | 53.73 | lcnu + hybrid = 53.73 | +0.0% | lcnu + hybrid = 53.73 | +0.0% | 63% / 36% / 0% |
| 128 | do_no_harm | lcnu + hybrid | 53.62 | lcnu + hybrid = 53.62 | +0.0% | lcnu + hybrid = 53.62 | +0.0% | 84% / 7% / 9% |
| 256 | best per pair | lcnu + hybrid | 45.95 | lcnu + hybrid = 45.95 | +0.0% | lcnu + hybrid = 45.95 | +0.0% | 87% / 11% / 2% |
| 256 | same_queue | lcnu + coactivation | 48.28 | lcnu + hybrid = 48.33 | +0.1% | lcnu + coactivation = 48.28 | +0.0% | 80% / 14% / 6% |
| 256 | probation | lcnu + hybrid | 45.95 | lcnu + hybrid = 45.95 | +0.0% | lcnu + hybrid = 45.95 | +0.0% | 74% / 15% / 10% |
| 256 | pin | lcnu + hybrid | 45.95 | lcnu + hybrid = 45.95 | +0.0% | lcnu + hybrid = 45.95 | +0.0% | 74% / 15% / 10% |
| 256 | buffer | lcnu + hybrid | 46.12 | lcnu + hybrid = 46.12 | +0.0% | lcnu + hybrid = 46.12 | +0.0% | 78% / 21% / 1% |
| 256 | do_no_harm | lcnu + hybrid | 46.01 | lcnu + hybrid = 46.01 | +0.0% | lcnu + hybrid = 46.01 | +0.0% | 90% / 3% / 7% |
| 512 | best per pair | lcnu + coactivation | 36.71 | lcnu + hybrid = 36.96 | +0.7% | lcnu + coactivation = 36.71 | +0.0% | 79% / 20% / 0% |
| 512 | same_queue | lcnu + coactivation | 38.34 | lcnu + hybrid = 38.38 | +0.1% | lcnu + coactivation = 38.34 | +0.0% | 65% / 29% / 6% |
| 512 | probation | lcnu + coactivation | 36.71 | lcnu + hybrid = 36.96 | +0.7% | lcnu + coactivation = 36.71 | +0.0% | 58% / 32% / 10% |
| 512 | pin | lcnu + coactivation | 36.71 | lcnu + hybrid = 36.96 | +0.7% | lcnu + coactivation = 36.71 | +0.0% | 58% / 32% / 10% |
| 512 | buffer | lcnu + coactivation | 36.88 | lcnu + hybrid = 37.11 | +0.6% | lcnu + coactivation = 36.88 | +0.0% | 45% / 55% / 0% |
| 512 | do_no_harm | lcnu + coactivation | 36.81 | lcnu + hybrid = 37.03 | +0.6% | lcnu + coactivation = 36.81 | +0.0% | 85% / 7% / 8% |
| 1024 | best per pair | lcnu + hybrid | 25.49 | lcnu + coactivation = 25.59 | +0.4% | lcnu + hybrid = 25.49 | +0.0% | 78% / 21% / 1% |
| 1024 | same_queue | lcnu + coactivation | 27.06 | lcnu + coactivation = 27.06 | +0.0% | lcnu + coactivation = 27.06 | +0.0% | 59% / 32% / 9% |
| 1024 | probation | lcnu + hybrid | 25.49 | lcnu + hybrid = 25.49 | +0.0% | lcnu + hybrid = 25.49 | +0.0% | 56% / 38% / 6% |
| 1024 | pin | lcnu + hybrid | 25.49 | lcnu + hybrid = 25.49 | +0.0% | lcnu + hybrid = 25.49 | +0.0% | 57% / 36% / 7% |
| 1024 | buffer | lcnu + hybrid | 25.72 | lcnu + hybrid = 25.72 | +0.0% | lcnu + hybrid = 25.72 | +0.0% | 46% / 53% / 1% |
| 1024 | do_no_harm | lcnu + hybrid | 25.66 | lcnu + hybrid = 25.66 | +0.0% | lcnu + hybrid = 25.66 | +0.0% | 81% / 17% / 1% |
| 2048 | best per pair | w_tinylfu + hybrid | 13.17 | lcnu + coactivation = 13.19 | +0.2% | lcnu + coactivation = 13.19 | +0.2% | 86% / 12% / 2% |
| 2048 | same_queue | s3fifo + coactivation | 13.35 | lcnu + coactivation = 13.45 | +0.7% | lcnu + coactivation = 13.45 | +0.7% | 55% / 37% / 8% |
| 2048 | probation | w_tinylfu + hybrid | 13.17 | lcnu + hybrid = 13.32 | +1.2% | lcnu + coactivation = 13.19 | +0.2% | 60% / 39% / 1% |
| 2048 | pin | lcnu + coactivation | 13.19 | lcnu + coactivation = 13.19 | +0.0% | lcnu + coactivation = 13.19 | +0.0% | 61% / 38% / 1% |
| 2048 | buffer | lcnu + coactivation | 13.27 | lcnu + hybrid = 13.40 | +1.0% | lcnu + coactivation = 13.27 | +0.0% | 58% / 41% / 1% |
| 2048 | do_no_harm | lcnu + coactivation | 13.22 | lcnu + coactivation = 13.22 | +0.0% | lcnu + coactivation = 13.22 | +0.0% | 90% / 9% / 1% |
| 3072 | best per pair | w_tinylfu + prev_token | 4.51 | w_tinylfu + hybrid = 4.67 | +3.4% | w_tinylfu + prev_token = 4.51 | +0.0% | 98% / 2% / 1% |
| 3072 | same_queue | w_tinylfu + prev_token | 4.60 | w_tinylfu + hybrid = 4.92 | +7.1% | w_tinylfu + prev_token = 4.60 | +0.0% | 80% / 15% / 5% |
| 3072 | probation | w_tinylfu + prev_token | 4.51 | w_tinylfu + hybrid = 4.77 | +5.7% | w_tinylfu + prev_token = 4.51 | +0.0% | 92% / 7% / 1% |
| 3072 | pin | w_tinylfu + prev_token | 4.51 | w_tinylfu + hybrid = 5.12 | +13.5% | w_tinylfu + prev_token = 4.51 | +0.0% | 90% / 7% / 3% |
| 3072 | buffer | w_tinylfu + prev_token | 4.61 | w_tinylfu + hybrid = 4.67 | +1.3% | w_tinylfu + prev_token = 4.61 | +0.0% | 94% / 5% / 0% |
| 3072 | do_no_harm | w_tinylfu + prev_token | 4.51 | w_tinylfu + hybrid = 4.77 | +5.6% | w_tinylfu + prev_token = 4.51 | +0.0% | 98% / 1% / 1% |

### Admission modes over all online pairs (nominal)

| Admission | Pairs | Best mode for | Mean stall vs best mode | Worse than no prefetch | Mean demand hit-rate change (points) | Mean wasted transfers / token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same_queue | 120 | 20 | +4.8% | 44 (37%) | -0.08 | 28.7 |
| probation | 120 | 37 | +5.7% | 46 (38%) | -1.63 | 33.1 |
| pin | 120 | 35 | +6.0% | 49 (41%) | -1.67 | 33.7 |
| buffer | 120 | 20 | +3.5% | 40 (33%) | +0.11 | 32.5 |
| do_no_harm | 120 | 49 | +1.5% | 24 (20%) | -0.57 | 15.8 |

### Online combinations worse than the evictor alone (nominal)

| Admission | cap=128 | cap=256 | cap=512 | cap=1024 | cap=2048 | cap=3072 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same_queue | 6/20 | 7/20 | 6/20 | 10/20 | 6/20 | 9/20 |
| probation | 8/20 | 7/20 | 8/20 | 10/20 | 6/20 | 7/20 |
| pin | 8/20 | 7/20 | 8/20 | 11/20 | 6/20 | 9/20 |
| buffer | 5/20 | 5/20 | 5/20 | 6/20 | 7/20 | 12/20 |
| do_no_harm | 4/20 | 4/20 | 2/20 | 3/20 | 4/20 | 7/20 |

### Online combinations worse than the evictor alone (FIFO link)

| Admission | cap=128 | cap=256 | cap=512 | cap=1024 | cap=2048 | cap=3072 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same_queue | 6/20 | 7/20 | 6/20 | 10/20 | 6/20 | 11/20 |
| probation | 8/20 | 7/20 | 8/20 | 10/20 | 6/20 | 7/20 |
| pin | 8/20 | 7/20 | 8/20 | 12/20 | 7/20 | 12/20 |
| buffer | 5/20 | 5/20 | 5/20 | 6/20 | 7/20 | 12/20 |
| do_no_harm | 4/20 | 4/20 | 2/20 | 5/20 | 5/20 | 7/20 |

### Does prefetch change the best evictor? (nominal)

| Capacity | Best evictor, no prefetch | Best evictor per prefetcher (best admission) | Kendall tau of evictor ranking vs no prefetch |
| ---: | --- | --- | --- |
| 128 | lcnu | coactivation: lcnu; hybrid: lcnu; prev_token: lcnu; same_id: lcnu | coactivation +1.00; hybrid +1.00; prev_token +1.00; same_id +1.00 |
| 256 | lcnu | coactivation: lcnu; hybrid: lcnu; prev_token: lcnu; same_id: lcnu | coactivation +1.00; hybrid +1.00; prev_token +1.00; same_id +1.00 |
| 512 | lcnu | coactivation: lcnu; hybrid: lcnu; prev_token: lcnu; same_id: lcnu | coactivation +0.80; hybrid +0.80; prev_token +0.80; same_id +1.00 |
| 1024 | lcnu | coactivation: lcnu; hybrid: lcnu; prev_token: lcnu; same_id: lcnu | coactivation +0.80; hybrid +0.80; prev_token +0.80; same_id +1.00 |
| 2048 | lcnu | coactivation: lcnu; hybrid: w_tinylfu; prev_token: w_tinylfu; same_id: lcnu | coactivation +0.80; hybrid +0.40; prev_token +0.40; same_id +0.60 |
| 3072 | w_tinylfu | coactivation: w_tinylfu; hybrid: w_tinylfu; prev_token: w_tinylfu; same_id: w_tinylfu | coactivation +1.00; hybrid +1.00; prev_token +1.00; same_id +1.00 |

### Timing sensitivity

| Capacity | best online pair (nominal) | best online pair (fifo) | best online pair (roofline) | gain vs best alone (nominal) | gain vs best alone (fifo) | gain vs best alone (roofline) |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 128 | lcnu + hybrid / probation | lcnu + hybrid / probation | lcnu + hybrid / do_no_harm | -7.1% | -7.1% | -1.0% |
| 256 | lcnu + hybrid / probation | lcnu + hybrid / probation | lcnu + hybrid / probation | -6.7% | -6.7% | -0.7% |
| 512 | lcnu + coactivation / probation | lcnu + coactivation / probation | lcnu + hybrid / probation | -6.6% | -6.6% | -0.8% |
| 1024 | lcnu + hybrid / probation | lcnu + hybrid / probation | lcnu + hybrid / probation | -8.5% | -8.5% | -0.6% |
| 2048 | w_tinylfu + hybrid / probation | w_tinylfu + hybrid / probation | lcnu + hybrid / probation | -4.2% | -4.2% | +0.1% |
| 3072 | w_tinylfu + prev_token / probation | w_tinylfu + prev_token / probation | w_tinylfu + prev_token / probation | -2.9% | -2.9% | -2.0% |

### Evictor x prefetcher at capacity 128 (nominal stall, best admission)

| Evictor | none | coactivation | hybrid | prev_token | same_id |
| --- | ---: | ---: | ---: | ---: | ---: |
| lcnu | 57.67 | 54.03 do_n | 53.60 prob | 56.28 do_n | 56.93 do_n |
| lfu | 63.30 | 57.87 do_n | 57.55 prob | 60.61 do_n | 62.33 do_n |
| lru | 71.84 | 65.87 prob | 65.46 same | 68.60 do_n | 71.17 do_n |
| s3fifo | 71.84 | 65.87 prob | 65.46 same | 68.60 do_n | 71.17 do_n |
| w_tinylfu | 60.66 | 56.48 buff | 56.08 buff | 59.46 buff | 60.67 same |

### Evictor x prefetcher at capacity 256 (nominal stall, best admission)

| Evictor | none | coactivation | hybrid | prev_token | same_id |
| --- | ---: | ---: | ---: | ---: | ---: |
| lcnu | 49.26 | 46.15 prob | 45.95 prob | 48.53 do_n | 48.82 do_n |
| lfu | 57.03 | 52.30 prob | 52.04 prob | 55.04 do_n | 56.23 do_n |
| lru | 71.84 | 65.87 prob | 65.46 same | 68.62 do_n | 71.28 do_n |
| s3fifo | 68.61 | 56.20 same | 54.83 same | 60.80 prob | 63.50 do_n |
| w_tinylfu | 53.62 | 49.89 buff | 49.58 buff | 52.59 buff | 53.96 do_n |

### Evictor x prefetcher at capacity 512 (nominal stall, best admission)

| Evictor | none | coactivation | hybrid | prev_token | same_id |
| --- | ---: | ---: | ---: | ---: | ---: |
| lcnu | 39.30 | 36.71 prob | 36.96 prob | 38.92 do_n | 39.11 do_n |
| lfu | 48.13 | 44.61 prob | 44.45 prob | 46.74 do_n | 47.57 do_n |
| lru | 43.99 | 41.10 same | 41.02 same | 43.91 do_n | 43.80 do_n |
| s3fifo | 44.02 | 41.08 prob | 40.95 prob | 43.34 do_n | 43.84 do_n |
| w_tinylfu | 45.16 | 42.00 buff | 41.85 buff | 44.25 buff | 44.42 do_n |

### Evictor x prefetcher at capacity 1024 (nominal stall, best admission)

| Evictor | none | coactivation | hybrid | prev_token | same_id |
| --- | ---: | ---: | ---: | ---: | ---: |
| lcnu | 27.87 | 25.59 prob | 25.49 prob | 27.90 prob | 27.84 do_n |
| lfu | 34.91 | 31.72 prob | 31.51 prob | 33.86 buff | 34.82 do_n |
| lru | 30.79 | 29.11 same | 29.17 prob | 30.74 prob | 30.76 do_n |
| s3fifo | 30.85 | 28.72 same | 28.67 prob | 30.56 buff | 30.81 do_n |
| w_tinylfu | 32.57 | 30.02 do_n | 30.04 do_n | 31.60 buff | 31.88 do_n |

### Evictor x prefetcher at capacity 2048 (nominal stall, best admission)

| Evictor | none | coactivation | hybrid | prev_token | same_id |
| --- | ---: | ---: | ---: | ---: | ---: |
| lcnu | 13.74 | 13.19 prob | 13.26 pin | 13.75 prob | 13.89 do_n |
| lfu | 16.61 | 15.88 prob | 15.56 pin | 16.00 do_n | 16.59 do_n |
| lru | 16.53 | 15.62 same | 15.64 prob | 16.49 do_n | 16.71 do_n |
| s3fifo | 14.28 | 13.35 same | 13.36 same | 14.26 do_n | 14.53 do_n |
| w_tinylfu | 14.53 | 13.29 prob | 13.17 prob | 13.72 do_n | 14.45 do_n |

### Evictor x prefetcher at capacity 3072 (nominal stall, best admission)

| Evictor | none | coactivation | hybrid | prev_token | same_id |
| --- | ---: | ---: | ---: | ---: | ---: |
| lcnu | 5.93 | 5.78 prob | 5.78 prob | 5.93 same | 6.07 do_n |
| lfu | 6.51 | 6.23 pin | 6.08 pin | 6.33 prob | 6.52 do_n |
| lru | 7.77 | 7.39 same | 7.31 same | 7.76 same | 7.89 do_n |
| s3fifo | 5.30 | 5.20 same | 5.20 same | 5.30 same | 5.46 do_n |
| w_tinylfu | 4.65 | 4.71 buff | 4.67 buff | 4.51 prob | 4.99 do_n |
