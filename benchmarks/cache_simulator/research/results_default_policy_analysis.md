Default: `lcnu_p` + `knn:horizon=1,context=8,neighbors=8,power=8,bank=8192,budget=8,decode_width=8` / `probation:issue=2`. Stall = ms per decode token (nominal timing).

### qwen3-235b-a22b-fp8 / decode

| Slots | LRU | LFU | LCNU | Default | vs LCNU | vs LFU | Per-cell best online | Stall | Default regret | Oracle bound |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 256 | 248.57 | 209.15 | 197.26 | 180.39 | -8.6% | -13.7% | lcnu + knn / probation:issue=2 | 180.39 | +0.0% | 176.58 |
| 512 | 248.57 | 184.46 | 166.67 | 150.63 | -9.6% | -18.3% | lcnu + knn / probation:issue=2 | 150.63 | +0.0% | 145.76 |
| 1024 | 152.02 | 149.87 | 128.21 | 113.67 | -11.3% | -24.2% | lcnu + knn / probation | 113.58 | +0.1% | 107.04 |
| 2048 | 94.32 | 103.21 | 83.45 | 71.91 | -13.8% | -30.3% | lcnu + knn / probation | 71.76 | +0.2% | 62.22 |
| 4096 | 40.33 | 44.99 | 33.87 | 28.60 | -15.6% | -36.4% | lcnu + knn / probation | 28.59 | +0.0% | 15.69 |
| 6016 | 15.48 | 16.10 | 11.75 | 10.14 | -13.6% | -37.0% | lcnu + knn / probation:issue=2 | 10.02 | +1.2% | 2.47 |

Default regret vs the per-cell best online configuration: mean +0.3%, worst +1.2%.

### qwen3-235b-a22b-fp8 / decode+prefill-union

| Slots | LRU | LFU | LCNU | Default | vs LCNU | vs LFU | Per-cell best online | Stall | Default regret | Oracle bound |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 256 | 281.60 | 240.94 | 228.87 | 212.00 | -7.4% | -12.0% | lcnu + knn / probation:issue=2 | 211.99 | +0.0% | 209.11 |
| 512 | 281.60 | 215.09 | 197.14 | 181.07 | -8.2% | -15.8% | lcnu_p + knn / probation:issue=2 | 181.07 | +0.0% | 176.66 |
| 1024 | 185.18 | 178.22 | 156.51 | 141.91 | -9.3% | -20.4% | lcnu_p + knn / probation:issue=2 | 141.91 | +0.0% | 136.06 |
| 2048 | 126.40 | 127.05 | 107.72 | 96.16 | -10.7% | -24.3% | lcnu_p + knn / probation:issue=2 | 96.16 | +0.0% | 89.12 |
| 4096 | 67.77 | 59.91 | 49.63 | 44.43 | -10.5% | -25.8% | lcnu_p + knn / probation:issue=2 | 44.43 | +0.0% | 34.87 |
| 6016 | 34.43 | 23.96 | 20.36 | 18.80 | -7.6% | -21.5% | lcnu_p + knn / probation:issue=2 | 18.80 | +0.0% | 11.79 |

Default regret vs the per-cell best online configuration: mean +0.0%, worst +0.0%.

### qwen3-30b-a3b / decode

| Slots | LRU | LFU | LCNU | Default | vs LCNU | vs LFU | Per-cell best online | Stall | Default regret | Oracle bound |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 128 | 63.47 | 55.20 | 49.64 | 42.42 | -14.6% | -23.2% | lcnu + knn / probation:issue=2 | 42.42 | +0.0% | 40.64 |
| 256 | 63.47 | 49.18 | 41.50 | 34.83 | -16.1% | -29.2% | lcnu + knn / probation:issue=2 | 34.83 | +0.0% | 32.44 |
| 512 | 35.56 | 40.76 | 31.89 | 25.90 | -18.8% | -36.5% | lcnu + knn / do_no_harm | 25.87 | +0.1% | 22.90 |
| 1024 | 22.65 | 28.64 | 20.96 | 16.05 | -23.4% | -44.0% | lcnu + knn / probation | 16.01 | +0.3% | 11.98 |
| 2048 | 9.58 | 12.53 | 8.49 | 5.88 | -30.8% | -53.1% | lcnu + knn / probation | 5.87 | +0.1% | 1.92 |
| 3072 | 3.14 | 4.36 | 2.80 | 1.94 | -30.8% | -55.5% | lcnu + knn / probation:issue=2 | 1.85 | +4.6% | 0.15 |

Default regret vs the per-cell best online configuration: mean +0.8%, worst +4.6%.

### qwen3-30b-a3b / decode+prefill-union

| Slots | LRU | LFU | LCNU | Default | vs LCNU | vs LFU | Per-cell best online | Stall | Default regret | Oracle bound |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 128 | 71.84 | 63.30 | 57.67 | 50.43 | -12.5% | -20.3% | lcnu + knn / probation:issue=2 | 50.43 | +0.0% | 49.14 |
| 256 | 71.84 | 57.03 | 49.23 | 42.54 | -13.6% | -25.4% | lcnu_p + knn / probation:issue=2 | 42.54 | +0.0% | 40.44 |
| 512 | 43.99 | 48.13 | 39.05 | 33.04 | -15.4% | -31.3% | lcnu_p + knn / probation:issue=2 | 33.04 | +0.0% | 30.32 |
| 1024 | 30.79 | 34.91 | 27.09 | 22.17 | -18.1% | -36.5% | lcnu_p + knn / probation:issue=2 | 22.17 | +0.0% | 18.75 |
| 2048 | 16.53 | 16.61 | 12.49 | 9.90 | -20.7% | -40.4% | lcnu_p + knn / probation:issue=2 | 9.90 | +0.0% | 7.15 |
| 3072 | 7.77 | 6.51 | 4.94 | 4.09 | -17.1% | -37.2% | w_tinylfu + knn / probation:issue=2 | 3.85 | +6.2% | 2.00 |

Default regret vs the per-cell best online configuration: mean +1.0%, worst +6.2%.

Over all 24 cells: mean regret +0.5%, worst +6.2%.
