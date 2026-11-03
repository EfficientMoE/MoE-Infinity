# Composition identity check - qwen3-30b-a3b / mmlu150-greedy64/decode/all

Demand hits of each evictor alone vs `PrefetchCachePolicy` with no prefetcher and with a never-predicting prefetcher under each shared admission mode. Commit `db1ef24`.

| Evictor | Capacity | Alone | composed (none) | composed (same_queue) | composed (probation) | composed (pin) | composed (do_no_harm) | Identical |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| lru | 256 | 0 | 0 | 0 | 0 | 0 | 0 | yes |
| lru | 1024 | 2343938 | 2343938 | 2343938 | 2343938 | 2343938 | 2343938 | yes |
| lfu | 256 | 792775 | 792775 | 792775 | 792775 | 792775 | 792775 | yes |
| lfu | 1024 | 1984419 | 1984419 | 1984419 | 1984419 | 1984419 | 1984419 | yes |
| lirs | 256 | 1115455 | 1115455 | 1115455 | 1115455 | 1115455 | 1115455 | yes |
| lirs | 1024 | 2240719 | 2240719 | 2240719 | 2240719 | 2240719 | 2240719 | yes |
| s3fifo | 256 | 330616 | 330616 | 330616 | 330616 | 330616 | 330616 | yes |
| s3fifo | 1024 | 2270899 | 2270899 | 2270899 | 2270899 | 2270899 | 2270899 | yes |
| w_tinylfu | 256 | 1012299 | 1012299 | 1012299 | 1012299 | 1012299 | 1012299 | yes |
| w_tinylfu | 1024 | 2145150 | 2145150 | 2145150 | 2145150 | 2145150 | 2145150 | yes |
| layer_lru | 256 | 308465 | 308465 | 308465 | 308465 | 308465 | 308465 | yes |
| layer_lru | 1024 | 2345072 | 2345072 | 2345072 | 2345072 | 2345072 | 2345072 | yes |
| lcnu | 256 | 1259085 | 1259085 | 1259085 | 1259085 | 1259085 | 1259085 | yes |
| lcnu | 1024 | 2440333 | 2440333 | 2440333 | 2440333 | 2440333 | 2440333 | yes |
