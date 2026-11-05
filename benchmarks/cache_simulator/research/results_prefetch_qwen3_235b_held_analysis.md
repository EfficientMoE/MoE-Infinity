## qwen3-235b-a22b-fp8 / mmlu100-greedy64/decode/held

### Per capacity (nominal timing, ms of stall per decode token)

| Capacity | Belady hit | Floor | Best evictor alone | Stall | Best online pair | Stall | vs alone | Oracle bound | Stall |
| ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | --- | ---: |
| 256 | 0.3077 | 151.28 | lcnu (0.2064) | 197.26 | lcnu + hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 / probation | 188.09 | -4.6% | lcnu + oracle-h1 / probation | 177.35 |
| 512 | 0.4921 | 105.44 | lcnu (0.3295) | 166.67 | lcnu + hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 / probation | 159.94 | -4.0% | lcnu + oracle-h1 / probation | 146.12 |
| 1024 | 0.6609 | 63.48 | lcnu (0.4842) | 128.21 | lcnu + coactivation:budget=4,horizon=1 / probation | 123.44 | -3.7% | lcnu + oracle-h1 / probation | 107.04 |
| 2048 | 0.8109 | 26.20 | lcnu (0.6643) | 83.45 | lcnu + coactivation:budget=8,horizon=1 / probation | 78.50 | -5.9% | lcnu + oracle-h4 / probation | 62.22 |
| 4096 | 0.9334 | 0.00 | lcnu (0.8637) | 33.87 | lcnu + hybrid:alpha=0.4,budget=8,horizon=2,weights=1/1/16 / pin | 32.53 | -4.0% | lcnu + oracle-h4 / probation | 15.69 |
| 6016 | 0.9765 | 0.00 | lcnu (0.9527) | 11.75 | lcnu + coactivation:budget=16,horizon=2 / pin | 11.13 | -5.2% | lcnu + oracle-h4 / pin | 2.47 |

### Best online pair per admission mode (nominal)

| Capacity | same_queue | probation | pin | buffer | do_no_harm |
| ---: | --- | --- | --- | --- | --- |
| 256 | 196.01 lcnu+hybrid | 188.09 lcnu+hybrid | 188.09 lcnu+hybrid | 188.35 lcnu+hybrid | 188.19 lcnu+hybrid |
| 512 | 166.31 lcnu+hybrid | 159.94 lcnu+hybrid | 159.94 lcnu+hybrid | 160.30 lcnu+hybrid | 160.20 lcnu+hybrid |
| 1024 | 128.15 lcnu+hybrid | 123.44 lcnu+coactivation | 123.44 lcnu+coactivation | 123.93 lcnu+coactivation | 123.90 lcnu+coactivation |
| 2048 | 82.98 lcnu+coactivation | 78.50 lcnu+coactivation | 78.50 lcnu+coactivation | 79.27 lcnu+coactivation | 79.37 lcnu+coactivation |
| 4096 | 33.27 lcnu+hybrid | 32.63 lcnu+coactivation | 32.53 lcnu+hybrid | 32.80 lcnu+coactivation | 32.64 lcnu+hybrid |
| 6016 | 11.35 lcnu+coactivation | 11.31 lcnu+coactivation | 11.13 lcnu+coactivation | 11.43 lcnu+coactivation | 11.43 lcnu+coactivation |

### Separability (nominal)

| Capacity | Admission | Joint optimum | Stall | Independent (evictor alone + prefetcher on lru) | Regret | Sequential (best prefetcher for that evictor) | Regret | Evictor / prefetcher / interaction share of log-stall variance |
| ---: | --- | --- | ---: | --- | ---: | --- | ---: | --- |
| 256 | best per pair | lcnu + hybrid | 188.09 | lcnu + hybrid = 188.09 | +0.0% | lcnu + hybrid = 188.09 | +0.0% | 92% / 7% / 1% |
| 256 | same_queue | lcnu + hybrid | 196.01 | lcnu + hybrid = 196.01 | +0.0% | lcnu + hybrid = 196.01 | +0.0% | 95% / 5% / 0% |
| 256 | probation | lcnu + hybrid | 188.09 | lcnu + hybrid = 188.09 | +0.0% | lcnu + hybrid = 188.09 | +0.0% | 82% / 13% / 4% |
| 256 | pin | lcnu + hybrid | 188.09 | lcnu + hybrid = 188.09 | +0.0% | lcnu + hybrid = 188.09 | +0.0% | 82% / 13% / 4% |
| 256 | buffer | lcnu + hybrid | 188.35 | lcnu + hybrid = 188.35 | +0.0% | lcnu + hybrid = 188.35 | +0.0% | 88% / 12% / 0% |
| 256 | do_no_harm | lcnu + hybrid | 188.19 | lcnu + hybrid = 188.19 | +0.0% | lcnu + hybrid = 188.19 | +0.0% | 91% / 4% / 5% |
| 512 | best per pair | lcnu + hybrid | 159.94 | lcnu + hybrid = 159.94 | +0.0% | lcnu + hybrid = 159.94 | +0.0% | 96% / 3% / 1% |
| 512 | same_queue | lcnu + hybrid | 166.31 | lcnu + hybrid = 166.31 | +0.0% | lcnu + hybrid = 166.31 | +0.0% | 95% / 4% / 1% |
| 512 | probation | lcnu + hybrid | 159.94 | lcnu + hybrid = 159.94 | +0.0% | lcnu + hybrid = 159.94 | +0.0% | 90% / 6% / 4% |
| 512 | pin | lcnu + hybrid | 159.94 | lcnu + hybrid = 159.94 | +0.0% | lcnu + hybrid = 159.94 | +0.0% | 90% / 6% / 4% |
| 512 | buffer | lcnu + hybrid | 160.30 | lcnu + hybrid = 160.30 | +0.0% | lcnu + hybrid = 160.30 | +0.0% | 94% / 5% / 0% |
| 512 | do_no_harm | lcnu + hybrid | 160.20 | lcnu + hybrid = 160.20 | +0.0% | lcnu + hybrid = 160.20 | +0.0% | 94% / 1% / 5% |
| 1024 | best per pair | lcnu + coactivation | 123.44 | lcnu + coactivation = 123.44 | +0.0% | lcnu + coactivation = 123.44 | +0.0% | 86% / 12% / 2% |
| 1024 | same_queue | lcnu + hybrid | 128.15 | lcnu + hybrid = 128.15 | +0.0% | lcnu + hybrid = 128.15 | +0.0% | 68% / 25% / 7% |
| 1024 | probation | lcnu + coactivation | 123.44 | lcnu + coactivation = 123.44 | +0.0% | lcnu + coactivation = 123.44 | +0.0% | 77% / 15% / 8% |
| 1024 | pin | lcnu + coactivation | 123.44 | lcnu + coactivation = 123.44 | +0.0% | lcnu + coactivation = 123.44 | +0.0% | 77% / 15% / 8% |
| 1024 | buffer | lcnu + coactivation | 123.93 | lcnu + coactivation = 123.93 | +0.0% | lcnu + coactivation = 123.93 | +0.0% | 70% / 30% / 0% |
| 1024 | do_no_harm | lcnu + coactivation | 123.90 | lcnu + hybrid = 124.17 | +0.2% | lcnu + coactivation = 123.90 | +0.0% | 79% / 2% / 19% |
| 2048 | best per pair | lcnu + coactivation | 78.50 | lcnu + hybrid = 78.73 | +0.3% | lcnu + coactivation = 78.50 | +0.0% | 82% / 15% / 3% |
| 2048 | same_queue | lcnu + coactivation | 82.98 | lcnu + hybrid = 83.00 | +0.0% | lcnu + coactivation = 82.98 | +0.0% | 45% / 32% / 24% |
| 2048 | probation | lcnu + coactivation | 78.50 | lcnu + hybrid = 78.73 | +0.3% | lcnu + coactivation = 78.50 | +0.0% | 74% / 21% / 5% |
| 2048 | pin | lcnu + coactivation | 78.50 | lcnu + hybrid = 78.73 | +0.3% | lcnu + coactivation = 78.50 | +0.0% | 74% / 19% / 7% |
| 2048 | buffer | lcnu + coactivation | 79.27 | lcnu + hybrid = 79.37 | +0.1% | lcnu + coactivation = 79.27 | +0.0% | 69% / 31% / 0% |
| 2048 | do_no_harm | lcnu + coactivation | 79.37 | lcnu + hybrid = 79.50 | +0.2% | lcnu + coactivation = 79.37 | +0.0% | 84% / 11% / 4% |
| 4096 | best per pair | lcnu + hybrid | 32.53 | lcnu + hybrid = 32.53 | +0.0% | lcnu + hybrid = 32.53 | +0.0% | 71% / 22% / 6% |
| 4096 | same_queue | lcnu + hybrid | 33.27 | lcnu + hybrid = 33.27 | +0.0% | lcnu + hybrid = 33.27 | +0.0% | 47% / 42% / 11% |
| 4096 | probation | lcnu + coactivation | 32.63 | lcnu + hybrid = 32.91 | +0.9% | lcnu + coactivation = 32.63 | +0.0% | 61% / 34% / 4% |
| 4096 | pin | lcnu + hybrid | 32.53 | lcnu + hybrid = 32.53 | +0.0% | lcnu + hybrid = 32.53 | +0.0% | 60% / 36% / 5% |
| 4096 | buffer | lcnu + coactivation | 32.80 | lcnu + hybrid = 33.08 | +0.8% | lcnu + coactivation = 32.80 | +0.0% | 68% / 32% / 0% |
| 4096 | do_no_harm | lcnu + hybrid | 32.64 | lcnu + coactivation = 32.74 | +0.3% | lcnu + hybrid = 32.64 | +0.0% | 76% / 17% / 7% |
| 6016 | best per pair | lcnu + coactivation | 11.13 | lcnu + hybrid = 11.24 | +1.0% | lcnu + coactivation = 11.13 | +0.0% | 68% / 26% / 5% |
| 6016 | same_queue | lcnu + coactivation | 11.35 | lcnu + hybrid = 11.39 | +0.3% | lcnu + coactivation = 11.35 | +0.0% | 45% / 45% / 10% |
| 6016 | probation | lcnu + coactivation | 11.31 | lcnu + hybrid = 11.40 | +0.8% | lcnu + coactivation = 11.31 | +0.0% | 61% / 36% / 3% |
| 6016 | pin | lcnu + coactivation | 11.13 | lcnu + hybrid = 11.24 | +1.0% | lcnu + coactivation = 11.13 | +0.0% | 61% / 36% / 3% |
| 6016 | buffer | lcnu + coactivation | 11.43 | lcnu + hybrid = 11.52 | +0.8% | lcnu + coactivation = 11.43 | +0.0% | 66% / 34% / 0% |
| 6016 | do_no_harm | lcnu + coactivation | 11.43 | lcnu + hybrid = 11.49 | +0.6% | lcnu + coactivation = 11.43 | +0.0% | 73% / 21% / 6% |

### Admission modes over all online pairs (nominal)

| Admission | Pairs | Best mode for | Mean stall vs best mode | Worse than no prefetch | Mean demand hit-rate change (points) | Mean wasted transfers / token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same_queue | 210 | 31 | +6.9% | 111 (53%) | -0.96 | 39.3 |
| probation | 210 | 90 | +3.1% | 77 (37%) | -0.97 | 35.4 |
| pin | 210 | 97 | +3.1% | 77 (37%) | -1.00 | 35.8 |
| buffer | 210 | 28 | +2.4% | 81 (39%) | -0.10 | 32.8 |
| do_no_harm | 210 | 101 | +1.3% | 74 (35%) | -0.42 | 23.7 |

### Online combinations worse than the evictor alone (nominal)

| Admission | cap=256 | cap=512 | cap=1024 | cap=2048 | cap=4096 | cap=6016 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same_queue | 23/35 | 20/35 | 21/35 | 17/35 | 14/35 | 16/35 |
| probation | 11/35 | 13/35 | 17/35 | 17/35 | 11/35 | 8/35 |
| pin | 11/35 | 13/35 | 17/35 | 17/35 | 11/35 | 8/35 |
| buffer | 7/35 | 11/35 | 15/35 | 16/35 | 16/35 | 16/35 |
| do_no_harm | 9/35 | 12/35 | 15/35 | 14/35 | 12/35 | 12/35 |

### Online combinations worse than the evictor alone (FIFO link)

| Admission | cap=256 | cap=512 | cap=1024 | cap=2048 | cap=4096 | cap=6016 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same_queue | 23/35 | 22/35 | 25/35 | 18/35 | 16/35 | 22/35 |
| probation | 11/35 | 13/35 | 17/35 | 17/35 | 11/35 | 8/35 |
| pin | 11/35 | 13/35 | 17/35 | 17/35 | 15/35 | 16/35 |
| buffer | 7/35 | 11/35 | 15/35 | 16/35 | 16/35 | 16/35 |
| do_no_harm | 9/35 | 12/35 | 15/35 | 16/35 | 16/35 | 12/35 |

### Does prefetch change the best evictor? (nominal)

| Capacity | Best evictor, no prefetch | Best evictor per prefetcher (best admission) | Kendall tau of evictor ranking vs no prefetch |
| ---: | --- | --- | --- |
| 256 | lcnu | coactivation: lcnu; hybrid: lcnu; prev_token: lcnu; same_id: lcnu; topk_ema: lcnu | coactivation +1.00; hybrid +1.00; prev_token +1.00; same_id +0.81; topk_ema +1.00 |
| 512 | lcnu | coactivation: lcnu; hybrid: lcnu; prev_token: lcnu; same_id: lcnu; topk_ema: lcnu | coactivation +1.00; hybrid +1.00; prev_token +1.00; same_id +1.00; topk_ema +1.00 |
| 1024 | lcnu | coactivation: lcnu; hybrid: lcnu; prev_token: lcnu; same_id: lcnu; topk_ema: lcnu | coactivation +1.00; hybrid +1.00; prev_token +0.90; same_id +0.81; topk_ema +0.90 |
| 2048 | lcnu | coactivation: lcnu; hybrid: lcnu; prev_token: lcnu; same_id: lcnu; topk_ema: lcnu | coactivation +0.90; hybrid +0.81; prev_token +0.90; same_id +0.71; topk_ema +0.90 |
| 4096 | lcnu | coactivation: lcnu; hybrid: lcnu; prev_token: lcnu; same_id: lcnu; topk_ema: lcnu | coactivation +0.71; hybrid +0.62; prev_token +0.90; same_id +0.62; topk_ema +0.81 |
| 6016 | lcnu | coactivation: lcnu; hybrid: lcnu; prev_token: lcnu; same_id: lcnu; topk_ema: lcnu | coactivation +0.71; hybrid +0.71; prev_token +0.90; same_id +0.62; topk_ema +0.90 |

### Timing sensitivity

| Capacity | best online pair (nominal) | best online pair (fifo) | best online pair (roofline) | gain vs best alone (nominal) | gain vs best alone (fifo) | gain vs best alone (roofline) |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 256 | lcnu + hybrid / probation | lcnu + hybrid / probation | lcnu + hybrid / probation | -4.6% | -4.6% | -0.4% |
| 512 | lcnu + hybrid / probation | lcnu + hybrid / probation | lcnu + hybrid / probation | -4.0% | -4.0% | -0.1% |
| 1024 | lcnu + coactivation / probation | lcnu + coactivation / probation | lcnu + hybrid / probation | -3.7% | -3.7% | -0.4% |
| 2048 | lcnu + coactivation / probation | lcnu + coactivation / probation | lcnu + hybrid / probation | -5.9% | -5.9% | -0.2% |
| 4096 | lcnu + hybrid / pin | lcnu + coactivation / probation | lcnu + hybrid / pin | -4.0% | -3.7% | -1.2% |
| 6016 | lcnu + coactivation / pin | lcnu + coactivation / probation | lcnu + hybrid / pin | -5.2% | -3.7% | -0.7% |

### Evictor x prefetcher at capacity 256 (nominal stall, best admission)

| Evictor | none | coactivation | hybrid | prev_token | same_id | topk_ema |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| layer_lru | 244.31 | 230.01 prob | 229.24 prob | 238.57 do_n | 252.69 do_n | 234.21 prob |
| lcnu | 197.26 | 188.99 prob | 188.09 prob | 196.06 do_n | 197.84 do_n | 196.84 do_n |
| lfu | 209.15 | 197.25 prob | 196.40 prob | 205.69 do_n | 209.25 do_n | 204.48 do_n |
| lirs | 205.10 | 194.72 prob | 194.12 prob | 203.78 do_n | 205.40 do_n | 202.78 prob |
| lru | 248.57 | 233.91 prob | 233.22 prob | 241.56 do_n | 247.92 do_n | 238.38 do_n |
| s3fifo | 248.57 | 233.91 prob | 233.22 prob | 241.56 do_n | 247.92 do_n | 238.38 do_n |
| w_tinylfu | 206.21 | 196.38 buff | 195.66 buff | 204.23 buff | 208.26 do_n | 203.65 buff |

### Evictor x prefetcher at capacity 512 (nominal stall, best admission)

| Evictor | none | coactivation | hybrid | prev_token | same_id | topk_ema |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| layer_lru | 226.90 | 214.24 prob | 213.31 prob | 221.16 do_n | 235.35 do_n | 218.05 prob |
| lcnu | 166.67 | 160.30 prob | 159.94 prob | 167.03 do_n | 167.61 do_n | 167.13 buff |
| lfu | 184.46 | 175.07 prob | 174.35 prob | 182.36 do_n | 184.88 do_n | 181.26 do_n |
| lirs | 177.52 | 169.41 prob | 169.02 prob | 177.42 prob | 178.19 do_n | 176.97 do_n |
| lru | 248.57 | 233.91 prob | 233.22 prob | 242.10 do_n | 247.94 do_n | 238.43 prob |
| s3fifo | 215.63 | 200.83 prob | 200.24 prob | 211.89 prob | 215.47 do_n | 200.98 do_n |
| w_tinylfu | 180.03 | 172.20 buff | 171.57 buff | 179.22 buff | 181.38 do_n | 179.18 buff |

### Evictor x prefetcher at capacity 1024 (nominal stall, best admission)

| Evictor | none | coactivation | hybrid | prev_token | same_id | topk_ema |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| layer_lru | 149.11 | 143.14 prob | 143.22 prob | 149.11 same | 157.84 do_n | 148.68 prob |
| lcnu | 128.21 | 123.44 prob | 123.83 prob | 128.72 prob | 128.99 do_n | 128.36 prob |
| lfu | 149.87 | 144.45 prob | 144.05 prob | 149.00 do_n | 150.46 do_n | 148.44 do_n |
| lirs | 140.50 | 135.15 prob | 134.92 prob | 140.74 do_n | 141.60 do_n | 140.64 do_n |
| lru | 152.02 | 145.23 prob | 145.30 prob | 152.02 same | 152.87 do_n | 150.99 same |
| s3fifo | 145.68 | 139.99 prob | 139.78 prob | 145.94 do_n | 146.44 do_n | 145.91 do_n |
| w_tinylfu | 148.49 | 142.55 buff | 142.09 buff | 147.63 do_n | 146.65 do_n | 147.89 buff |

### Evictor x prefetcher at capacity 2048 (nominal stall, best admission)

| Evictor | none | coactivation | hybrid | prev_token | same_id | topk_ema |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| layer_lru | 93.76 | 91.14 prob | 90.89 prob | 93.76 same | 100.47 do_n | 93.76 same |
| lcnu | 83.45 | 78.50 prob | 78.73 prob | 83.45 prob | 83.74 do_n | 83.45 pin |
| lfu | 103.21 | 97.40 prob | 96.94 prob | 103.15 buff | 103.39 do_n | 102.67 buff |
| lirs | 95.73 | 92.11 same | 91.83 prob | 95.95 buff | 97.05 do_n | 95.81 buff |
| lru | 94.32 | 91.52 prob | 91.29 prob | 94.32 same | 95.72 do_n | 94.32 same |
| s3fifo | 95.38 | 91.58 prob | 91.18 prob | 95.29 buff | 96.34 do_n | 95.48 buff |
| w_tinylfu | 105.52 | 93.54 do_n | 93.21 do_n | 98.78 do_n | 103.40 do_n | 99.69 do_n |

### Evictor x prefetcher at capacity 4096 (nominal stall, best admission)

| Evictor | none | coactivation | hybrid | prev_token | same_id | topk_ema |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| layer_lru | 41.54 | 40.12 prob | 40.03 prob | 41.54 same | 48.04 do_n | 41.54 same |
| lcnu | 33.87 | 32.63 prob | 32.53 pin | 33.87 same | 38.21 do_n | 33.88 do_n |
| lfu | 44.99 | 43.76 prob | 43.02 pin | 44.71 do_n | 45.36 do_n | 44.66 buff |
| lirs | 43.71 | 40.67 same | 40.17 same | 43.58 same | 45.66 do_n | 43.74 buff |
| lru | 40.33 | 39.04 prob | 38.93 prob | 40.33 same | 44.51 do_n | 40.33 same |
| s3fifo | 38.34 | 36.72 same | 36.42 same | 38.33 same | 42.36 do_n | 38.38 buff |
| w_tinylfu | 47.82 | 39.83 prob | 38.47 pin | 43.83 prob | 47.17 prob | 42.45 prob |

### Evictor x prefetcher at capacity 6016 (nominal stall, best admission)

| Evictor | none | coactivation | hybrid | prev_token | same_id | topk_ema |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| layer_lru | 17.03 | 16.45 do_n | 16.32 prob | 17.03 same | 20.79 do_n | 17.03 same |
| lcnu | 11.75 | 11.13 pin | 11.24 pin | 11.75 same | 14.38 do_n | 11.75 prob |
| lfu | 16.10 | 15.25 pin | 15.03 pin | 15.93 prob | 16.49 do_n | 15.95 prob |
| lirs | 15.66 | 15.11 same | 14.64 same | 15.60 same | 17.13 do_n | 15.66 prob |
| lru | 15.48 | 14.70 same | 14.02 same | 15.48 same | 18.49 do_n | 15.48 same |
| s3fifo | 13.19 | 12.64 pin | 12.31 same | 13.18 same | 16.21 do_n | 13.19 prob |
| w_tinylfu | 16.98 | 13.68 pin | 13.54 pin | 15.84 prob | 17.75 do_n | 15.80 do_n |
