## qwen3-235b-a22b-fp8 / mmlu100-greedy64/decode+prefill-union/held

### Per capacity (nominal timing, ms of stall per decode token)

| Capacity | Belady hit | Floor | Best evictor alone | Stall | Best online pair | Stall | vs alone | Oracle bound | Stall |
| ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | --- | ---: |
| 256 | 0.2790 | 181.89 | lcnu (0.1873) | 228.86 | lcnu + hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 / probation | 219.82 | -3.9% | lcnu + oracle-h1 / probation | 209.11 |
| 512 | 0.4501 | 133.71 | lcnu (0.2997) | 197.20 | lcnu + hybrid:alpha=0.4,budget=4,horizon=1,weights=1/1/16 / probation | 190.72 | -3.3% | lcnu + oracle-h1 / probation | 176.66 |
| 1024 | 0.6105 | 88.56 | lcnu (0.4408) | 157.48 | lcnu + coactivation:budget=4,horizon=1 / probation | 152.96 | -2.9% | lcnu + oracle-h2 / probation | 136.06 |
| 2048 | 0.7627 | 45.70 | lcnu (0.6052) | 111.17 | lcnu + coactivation:budget=8,horizon=1 / probation | 106.44 | -4.2% | lcnu + oracle-h4 / probation | 89.12 |
| 4096 | 0.8999 | 7.05 | lcnu (0.7988) | 56.66 | lcnu + hybrid:alpha=0.4,budget=8,horizon=2,weights=1/1/16 / pin | 55.55 | -2.0% | lcnu + oracle-h4 / probation | 37.81 |
| 6016 | 0.9600 | 0.00 | lfu (0.9149) | 23.96 | lfu + hybrid:alpha=0.4,budget=16,horizon=2,weights=1/1/16 / pin | 23.35 | -2.5% | lfu + oracle-h4 / same_queue | 12.79 |

### Best online pair per admission mode (nominal)

| Capacity | same_queue | probation | pin | buffer | do_no_harm |
| ---: | --- | --- | --- | --- | --- |
| 256 | 227.44 lcnu+hybrid | 219.82 lcnu+hybrid | 219.82 lcnu+hybrid | 220.08 lcnu+hybrid | 219.88 lcnu+hybrid |
| 512 | 196.72 lcnu+hybrid | 190.72 lcnu+hybrid | 190.72 lcnu+hybrid | 191.05 lcnu+hybrid | 190.89 lcnu+hybrid |
| 1024 | 157.32 lcnu+hybrid | 152.96 lcnu+coactivation | 152.96 lcnu+coactivation | 153.36 lcnu+coactivation | 153.29 lcnu+coactivation |
| 2048 | 110.68 lcnu+coactivation | 106.44 lcnu+coactivation | 106.44 lcnu+coactivation | 107.13 lcnu+coactivation | 107.04 lcnu+coactivation |
| 4096 | 56.11 lcnu+hybrid | 55.59 lcnu+coactivation | 55.55 lcnu+hybrid | 55.77 lcnu+coactivation | 55.60 lcnu+hybrid |
| 6016 | 23.71 lfu+hybrid | 23.54 lfu+hybrid | 23.35 lfu+hybrid | 23.74 lfu+hybrid | 23.67 lfu+hybrid |

### Separability (nominal)

| Capacity | Admission | Joint optimum | Stall | Independent (evictor alone + prefetcher on lru) | Regret | Sequential (best prefetcher for that evictor) | Regret | Evictor / prefetcher / interaction share of log-stall variance |
| ---: | --- | --- | ---: | --- | ---: | --- | ---: | --- |
| 256 | best per pair | lcnu + hybrid | 219.82 | lcnu + hybrid = 219.82 | +0.0% | lcnu + hybrid = 219.82 | +0.0% | 93% / 6% / 1% |
| 256 | same_queue | lcnu + hybrid | 227.44 | lcnu + hybrid = 227.44 | +0.0% | lcnu + hybrid = 227.44 | +0.0% | 94% / 3% / 3% |
| 256 | probation | lcnu + hybrid | 219.82 | lcnu + hybrid = 219.82 | +0.0% | lcnu + hybrid = 219.82 | +0.0% | 81% / 11% / 8% |
| 256 | pin | lcnu + hybrid | 219.82 | lcnu + hybrid = 219.82 | +0.0% | lcnu + hybrid = 219.82 | +0.0% | 81% / 11% / 8% |
| 256 | buffer | lcnu + hybrid | 220.08 | lcnu + hybrid = 220.08 | +0.0% | lcnu + hybrid = 220.08 | +0.0% | 88% / 12% / 0% |
| 256 | do_no_harm | lcnu + hybrid | 219.88 | lcnu + hybrid = 219.88 | +0.0% | lcnu + hybrid = 219.88 | +0.0% | 92% / 2% / 6% |
| 512 | best per pair | lcnu + hybrid | 190.72 | lcnu + hybrid = 190.72 | +0.0% | lcnu + hybrid = 190.72 | +0.0% | 95% / 4% / 1% |
| 512 | same_queue | lcnu + hybrid | 196.72 | lcnu + hybrid = 196.72 | +0.0% | lcnu + hybrid = 196.72 | +0.0% | 95% / 4% / 1% |
| 512 | probation | lcnu + hybrid | 190.72 | lcnu + hybrid = 190.72 | +0.0% | lcnu + hybrid = 190.72 | +0.0% | 86% / 5% / 9% |
| 512 | pin | lcnu + hybrid | 190.72 | lcnu + hybrid = 190.72 | +0.0% | lcnu + hybrid = 190.72 | +0.0% | 86% / 5% / 9% |
| 512 | buffer | lcnu + hybrid | 191.05 | lcnu + hybrid = 191.05 | +0.0% | lcnu + hybrid = 191.05 | +0.0% | 92% / 8% / 1% |
| 512 | do_no_harm | lcnu + hybrid | 190.89 | lcnu + hybrid = 190.89 | +0.0% | lcnu + hybrid = 190.89 | +0.0% | 94% / 1% / 5% |
| 1024 | best per pair | lcnu + coactivation | 152.96 | lcnu + coactivation = 152.96 | +0.0% | lcnu + coactivation = 152.96 | +0.0% | 91% / 8% / 0% |
| 1024 | same_queue | lcnu + hybrid | 157.32 | lcnu + hybrid = 157.32 | +0.0% | lcnu + hybrid = 157.32 | +0.0% | 78% / 21% / 1% |
| 1024 | probation | lcnu + coactivation | 152.96 | lcnu + coactivation = 152.96 | +0.0% | lcnu + coactivation = 152.96 | +0.0% | 76% / 15% / 10% |
| 1024 | pin | lcnu + coactivation | 152.96 | lcnu + coactivation = 152.96 | +0.0% | lcnu + coactivation = 152.96 | +0.0% | 76% / 15% / 10% |
| 1024 | buffer | lcnu + coactivation | 153.36 | lcnu + coactivation = 153.36 | +0.0% | lcnu + coactivation = 153.36 | +0.0% | 75% / 25% / 0% |
| 1024 | do_no_harm | lcnu + coactivation | 153.29 | lcnu + hybrid = 153.62 | +0.2% | lcnu + coactivation = 153.29 | +0.0% | 79% / 0% / 21% |
| 2048 | best per pair | lcnu + coactivation | 106.44 | lcnu + hybrid = 106.69 | +0.2% | lcnu + coactivation = 106.44 | +0.0% | 84% / 14% / 2% |
| 2048 | same_queue | lcnu + coactivation | 110.68 | lcnu + hybrid = 110.72 | +0.0% | lcnu + coactivation = 110.68 | +0.0% | 52% / 37% / 11% |
| 2048 | probation | lcnu + coactivation | 106.44 | lcnu + hybrid = 106.69 | +0.2% | lcnu + coactivation = 106.44 | +0.0% | 73% / 19% / 8% |
| 2048 | pin | lcnu + coactivation | 106.44 | lcnu + hybrid = 106.69 | +0.2% | lcnu + coactivation = 106.44 | +0.0% | 73% / 18% / 9% |
| 2048 | buffer | lcnu + coactivation | 107.13 | lcnu + hybrid = 107.26 | +0.1% | lcnu + coactivation = 107.13 | +0.0% | 65% / 34% / 0% |
| 2048 | do_no_harm | lcnu + coactivation | 107.04 | lcnu + hybrid = 107.23 | +0.2% | lcnu + coactivation = 107.04 | +0.0% | 87% / 10% / 3% |
| 4096 | best per pair | lcnu + hybrid | 55.55 | lcnu + hybrid = 55.55 | +0.0% | lcnu + hybrid = 55.55 | +0.0% | 78% / 17% / 6% |
| 4096 | same_queue | lcnu + hybrid | 56.11 | lcnu + hybrid = 56.11 | +0.0% | lcnu + hybrid = 56.11 | +0.0% | 44% / 51% / 5% |
| 4096 | probation | lcnu + coactivation | 55.59 | lcnu + hybrid = 55.86 | +0.5% | lcnu + coactivation = 55.59 | +0.0% | 57% / 39% / 4% |
| 4096 | pin | lcnu + hybrid | 55.55 | lcnu + hybrid = 55.55 | +0.0% | lcnu + hybrid = 55.55 | +0.0% | 56% / 40% / 4% |
| 4096 | buffer | lcnu + coactivation | 55.77 | lcnu + hybrid = 56.03 | +0.5% | lcnu + coactivation = 55.77 | +0.0% | 60% / 40% / 0% |
| 4096 | do_no_harm | lcnu + hybrid | 55.60 | lcnu + coactivation = 55.64 | +0.1% | lcnu + hybrid = 55.60 | +0.0% | 80% / 14% / 6% |
| 6016 | best per pair | lfu + hybrid | 23.35 | lfu + hybrid = 23.35 | +0.0% | lfu + hybrid = 23.35 | +0.0% | 89% / 9% / 2% |
| 6016 | same_queue | lfu + hybrid | 23.71 | lfu + hybrid = 23.71 | +0.0% | lfu + hybrid = 23.71 | +0.0% | 59% / 37% / 4% |
| 6016 | probation | lfu + hybrid | 23.54 | lfu + hybrid = 23.54 | +0.0% | lfu + hybrid = 23.54 | +0.0% | 83% / 16% / 1% |
| 6016 | pin | lfu + hybrid | 23.35 | lfu + hybrid = 23.35 | +0.0% | lfu + hybrid = 23.35 | +0.0% | 84% / 15% / 1% |
| 6016 | buffer | lfu + hybrid | 23.74 | lfu + hybrid = 23.74 | +0.0% | lfu + hybrid = 23.74 | +0.0% | 83% / 17% / 0% |
| 6016 | do_no_harm | lfu + hybrid | 23.67 | lfu + hybrid = 23.67 | +0.0% | lfu + hybrid = 23.67 | +0.0% | 93% / 5% / 1% |

### Admission modes over all online pairs (nominal)

| Admission | Pairs | Best mode for | Mean stall vs best mode | Worse than no prefetch | Mean demand hit-rate change (points) | Mean wasted transfers / token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same_queue | 120 | 14 | +4.4% | 55 (46%) | -0.28 | 41.1 |
| probation | 120 | 42 | +4.0% | 53 (44%) | -1.65 | 37.9 |
| pin | 120 | 44 | +4.0% | 54 (45%) | -1.67 | 38.5 |
| buffer | 120 | 18 | +2.3% | 40 (33%) | -0.07 | 37.5 |
| do_no_harm | 120 | 49 | +1.2% | 41 (34%) | -0.54 | 20.6 |

### Online combinations worse than the evictor alone (nominal)

| Admission | cap=256 | cap=512 | cap=1024 | cap=2048 | cap=4096 | cap=6016 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same_queue | 9/20 | 11/20 | 11/20 | 10/20 | 8/20 | 6/20 |
| probation | 8/20 | 10/20 | 10/20 | 11/20 | 7/20 | 7/20 |
| pin | 8/20 | 10/20 | 10/20 | 11/20 | 7/20 | 8/20 |
| buffer | 5/20 | 7/20 | 7/20 | 6/20 | 7/20 | 8/20 |
| do_no_harm | 5/20 | 8/20 | 8/20 | 7/20 | 6/20 | 7/20 |

### Online combinations worse than the evictor alone (FIFO link)

| Admission | cap=256 | cap=512 | cap=1024 | cap=2048 | cap=4096 | cap=6016 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same_queue | 10/20 | 11/20 | 14/20 | 10/20 | 11/20 | 12/20 |
| probation | 8/20 | 10/20 | 10/20 | 11/20 | 7/20 | 7/20 |
| pin | 8/20 | 10/20 | 10/20 | 12/20 | 9/20 | 12/20 |
| buffer | 5/20 | 7/20 | 7/20 | 6/20 | 7/20 | 8/20 |
| do_no_harm | 5/20 | 8/20 | 8/20 | 9/20 | 8/20 | 7/20 |

### Does prefetch change the best evictor? (nominal)

| Capacity | Best evictor, no prefetch | Best evictor per prefetcher (best admission) | Kendall tau of evictor ranking vs no prefetch |
| ---: | --- | --- | --- |
| 256 | lcnu | coactivation: lcnu; hybrid: lcnu; prev_token: lcnu; same_id: lcnu | coactivation +1.00; hybrid +1.00; prev_token +1.00; same_id +0.80 |
| 512 | lcnu | coactivation: lcnu; hybrid: lcnu; prev_token: lcnu; same_id: lcnu | coactivation +1.00; hybrid +1.00; prev_token +1.00; same_id +1.00 |
| 1024 | lcnu | coactivation: lcnu; hybrid: lcnu; prev_token: lcnu; same_id: lcnu | coactivation +1.00; hybrid +1.00; prev_token +1.00; same_id +0.80 |
| 2048 | lcnu | coactivation: lcnu; hybrid: lcnu; prev_token: lcnu; same_id: lcnu | coactivation +0.40; hybrid +0.40; prev_token +0.60; same_id +0.80 |
| 4096 | lcnu | coactivation: lcnu; hybrid: lcnu; prev_token: lcnu; same_id: lfu | coactivation +0.80; hybrid +0.60; prev_token +0.80; same_id +0.60 |
| 6016 | lfu | coactivation: lfu; hybrid: lfu; prev_token: lfu; same_id: lfu | coactivation +0.80; hybrid +0.80; prev_token +0.80; same_id +0.80 |

### Timing sensitivity

| Capacity | best online pair (nominal) | best online pair (fifo) | best online pair (roofline) | gain vs best alone (nominal) | gain vs best alone (fifo) | gain vs best alone (roofline) |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 256 | lcnu + hybrid / probation | lcnu + hybrid / probation | lcnu + hybrid / probation | -3.9% | -3.9% | -0.2% |
| 512 | lcnu + hybrid / probation | lcnu + hybrid / probation | lcnu + hybrid / probation | -3.3% | -3.3% | +0.1% |
| 1024 | lcnu + coactivation / probation | lcnu + coactivation / probation | lcnu + hybrid / probation | -2.9% | -2.9% | -0.2% |
| 2048 | lcnu + coactivation / probation | lcnu + coactivation / probation | lcnu + hybrid / probation | -4.2% | -4.2% | -0.0% |
| 4096 | lcnu + hybrid / pin | lcnu + coactivation / probation | lcnu + hybrid / pin | -2.0% | -1.9% | -0.5% |
| 6016 | lfu + hybrid / pin | lfu + hybrid / probation | lfu + hybrid / pin | -2.5% | -1.8% | -0.5% |

### Evictor x prefetcher at capacity 256 (nominal stall, best admission)

| Evictor | none | coactivation | hybrid | prev_token | same_id |
| --- | ---: | ---: | ---: | ---: | ---: |
| lcnu | 228.86 | 220.82 prob | 219.82 prob | 227.49 do_n | 229.34 do_n |
| lfu | 240.94 | 229.23 prob | 228.27 prob | 237.26 do_n | 240.93 do_n |
| lru | 281.60 | 266.92 prob | 266.13 prob | 274.31 do_n | 280.81 do_n |
| s3fifo | 281.60 | 266.92 prob | 266.13 prob | 274.31 do_n | 269.43 same |
| w_tinylfu | 238.42 | 228.57 buff | 227.77 buff | 235.97 buff | 240.90 do_n |

### Evictor x prefetcher at capacity 512 (nominal stall, best admission)

| Evictor | none | coactivation | hybrid | prev_token | same_id |
| --- | ---: | ---: | ---: | ---: | ---: |
| lcnu | 197.20 | 191.12 prob | 190.72 prob | 197.46 do_n | 198.09 do_n |
| lfu | 215.09 | 206.04 prob | 205.21 prob | 212.78 do_n | 215.44 do_n |
| lru | 281.60 | 266.92 prob | 266.13 prob | 274.85 do_n | 280.85 do_n |
| s3fifo | 247.38 | 228.12 same | 228.31 buff | 247.29 do_n | 252.18 do_n |
| w_tinylfu | 210.36 | 202.89 buff | 202.22 buff | 209.51 buff | 212.43 do_n |

### Evictor x prefetcher at capacity 1024 (nominal stall, best admission)

| Evictor | none | coactivation | hybrid | prev_token | same_id |
| --- | ---: | ---: | ---: | ---: | ---: |
| lcnu | 157.48 | 152.96 prob | 153.40 prob | 158.00 do_n | 158.35 do_n |
| lfu | 178.22 | 173.17 prob | 172.69 prob | 177.17 do_n | 178.77 do_n |
| lru | 185.18 | 178.47 prob | 178.51 prob | 185.00 do_n | 186.00 do_n |
| s3fifo | 175.40 | 169.80 prob | 169.57 prob | 175.50 do_n | 176.14 do_n |
| w_tinylfu | 177.31 | 171.55 buff | 171.04 buff | 176.50 buff | 175.75 do_n |

### Evictor x prefetcher at capacity 2048 (nominal stall, best admission)

| Evictor | none | coactivation | hybrid | prev_token | same_id |
| --- | ---: | ---: | ---: | ---: | ---: |
| lcnu | 111.17 | 106.44 prob | 106.69 prob | 111.23 prob | 111.55 do_n |
| lfu | 127.05 | 121.69 prob | 121.16 prob | 126.84 buff | 127.20 do_n |
| lru | 126.40 | 123.74 prob | 123.50 prob | 126.25 prob | 127.71 do_n |
| s3fifo | 121.98 | 118.33 prob | 117.92 prob | 121.85 buff | 122.84 do_n |
| w_tinylfu | 130.29 | 120.04 do_n | 120.15 do_n | 123.70 do_n | 127.99 do_n |

### Evictor x prefetcher at capacity 4096 (nominal stall, best admission)

| Evictor | none | coactivation | hybrid | prev_token | same_id |
| --- | ---: | ---: | ---: | ---: | ---: |
| lcnu | 56.66 | 55.59 prob | 55.55 pin | 56.70 prob | 60.28 do_n |
| lfu | 59.91 | 59.00 prob | 58.32 pin | 59.59 do_n | 60.20 do_n |
| lru | 67.77 | 66.75 same | 66.42 same | 67.69 do_n | 71.30 do_n |
| s3fifo | 59.62 | 57.72 same | 57.45 same | 59.63 do_n | 63.21 do_n |
| w_tinylfu | 64.62 | 57.92 prob | 57.39 pin | 59.85 prob | 63.51 do_n |

### Evictor x prefetcher at capacity 6016 (nominal stall, best admission)

| Evictor | none | coactivation | hybrid | prev_token | same_id |
| --- | ---: | ---: | ---: | ---: | ---: |
| lcnu | 26.83 | 26.06 same | 26.11 same | 26.84 prob | 29.34 do_n |
| lfu | 23.96 | 23.52 pin | 23.35 pin | 23.82 prob | 24.26 do_n |
| lru | 34.43 | 32.47 same | 32.06 same | 34.41 same | 36.97 do_n |
| s3fifo | 25.43 | 24.43 same | 24.34 same | 25.43 same | 28.14 do_n |
| w_tinylfu | 25.82 | 23.67 prob | 23.55 prob | 24.46 prob | 26.63 do_n |
