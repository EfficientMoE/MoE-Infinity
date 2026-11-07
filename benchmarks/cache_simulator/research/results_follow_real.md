# FOLLOW: a realizable LCNU / S3-FIFO arbiter on real decode traces

`follow` (`research/policies_follow.py`) is a single-resident-set version of
`costfit`. CostFit runs LCNU and S3-FIFO side by side at full capacity and is
credited with the hits of whichever inner it currently chooses. That is an
optimistic bound, because a real cache would need both expert sets resident
and an instant migration at every switch. FOLLOW keeps one real set of `C`
expert slots and runs the unchanged CostFit control law over key-only ghosts.
On a miss with the set full, it evicts a resident expert that the current
leader ghost does not hold. The real set converges to the leader's set through
demand misses only: no extra fetches and no bulk migration.

## Why

LCNU is the engine default (#339/#341). Replaying the same three real decode
streams through the C++ `SimulateEvictionPolicy` port gives the same hit counts
as this simulator for all 13 shipped policies at 27/27 capacities. Under that
check, LCNU is within 0.5pp of the best shipped policy in only **17/27** cells.
S3-FIFO wins a contiguous upper-capacity tail on every trace, by up to
3.27pp. S3-FIFO alone is within 0.5pp in 14/27. The best of {LCNU, S3-FIFO} equals
the best of all 13 in 27/27 cells, so an arbiter between the two is enough.

## Result (3 traces x 9 capacities, decode-only, `--prefill skip`)

- **within 0.5pp of the best of 13 shipped policies: FOLLOW 27/27**, LCNU 17/27.
  Worst FOLLOW gap 0.191pp (qwen3-30b-a3b (lo) @ 2048).
- **never materially below LCNU:** worst FOLLOW - LCNU -0.099pp (qwen3-30b-a3b (lo) @ 1024),
  best +3.076pp (qwen3-30b-a3b (lo) @ 2048).
- **The single-set realization costs almost nothing against the optimistic bound:** FOLLOW - CostFit
  min -0.020 / median +0.000 / max +0.127pp.
- In 10/27 cells FOLLOW is above every individual policy (max
  +0.479pp, qwen3-30b-a3b (lo) @ 1152). During convergence the real set is a hybrid of the two ghosts.
- 0 fallbacks. The `D = R - leader` invariant was checked every 65536 accesses.
  Reruns are bit-identical. An independent brute-force reimplementation, which recomputes
  `R - leader` at every step, matches the hit counts.
- **Static capacity rules do not transfer.** As an alternative, we tried choosing S3-FIFO above a fixed
  fraction of the working set, fitted on two traces and tested on the held-out third.
  It is within 0.5pp in only 15/27 (fraction of distinct experts) and 17/27 (fraction of the 20%-tail working set),
  and loses up to 9.25pp. The crossover fraction is 0.19-0.66x distinct across traces, so an online arbiter is needed.

| trace @ capacity | best of 13 (policy) | LCNU | S3-FIFO | CostFit (bound) | **FOLLOW** | switches | mean D size |
|---|---|---|---|---|---|---|---|
| granite3.1-moe-3b @ 128 | 28.38 (lcnu) | 28.38 | 0.00 | 28.38 | **28.38** | 0 | 0.0 |
| granite3.1-moe-3b @ 192 | 38.15 (lcnu) | 38.15 | 28.85 | 38.15 | **38.15** | 0 | 0.0 |
| granite3.1-moe-3b @ 256 | 46.82 (lcnu) | 46.82 | 37.89 | 46.82 | **46.82** | 0 | 0.0 |
| granite3.1-moe-3b @ 384 | 61.32 (lcnu) | 61.32 | 52.07 | 61.32 | **61.32** | 0 | 0.0 |
| granite3.1-moe-3b @ 512 | 72.87 (lcnu) | 72.87 | 67.65 | 72.87 | **72.87** | 0 | 0.0 |
| granite3.1-moe-3b @ 640 | 81.86 (lcnu) | 81.86 | 79.58 | 81.86 | **81.86** | 0 | 0.0 |
| granite3.1-moe-3b @ 768 | 88.64 (lcnu) | 88.64 | 87.93 | 88.64 | **88.64** | 0 | 0.0 |
| granite3.1-moe-3b @ 896 | 93.60 (s3fifo) | 93.57 | 93.60 | 93.68 | **93.68** | 1 | 0.2 |
| granite3.1-moe-3b @ 1024 | 97.32 (s3fifo) | 97.08 | 97.32 | 97.21 | **97.21** | 1 | 0.4 |
| qwen3-30b-a3b (hi) @ 768 | 62.24 (lcnu) | 62.24 | 57.52 | 62.24 | **62.24** | 0 | 0.0 |
| qwen3-30b-a3b (hi) @ 896 | 66.56 (lcnu) | 66.56 | 63.81 | 66.56 | **66.56** | 0 | 0.0 |
| qwen3-30b-a3b (hi) @ 1024 | 70.41 (lcnu) | 70.41 | 69.04 | 70.32 | **70.45** | 2 | 3.6 |
| qwen3-30b-a3b (hi) @ 1152 | 73.85 (lcnu) | 73.85 | 73.72 | 74.18 | **74.18** | 3 | 0.4 |
| qwen3-30b-a3b (hi) @ 1280 | 77.87 (s3fifo) | 77.07 | 77.87 | 78.27 | **78.26** | 1 | 0.5 |
| qwen3-30b-a3b (hi) @ 1408 | 81.45 (s3fifo) | 80.04 | 81.45 | 81.70 | **81.69** | 1 | 1.0 |
| qwen3-30b-a3b (hi) @ 1536 | 84.61 (s3fifo) | 82.82 | 84.61 | 84.63 | **84.61** | 3 | 2.3 |
| qwen3-30b-a3b (hi) @ 1792 | 89.35 (s3fifo) | 87.52 | 89.35 | 89.31 | **89.30** | 3 | 2.8 |
| qwen3-30b-a3b (hi) @ 2048 | 92.83 (s3fifo) | 91.10 | 92.83 | 92.81 | **92.82** | 1 | 0.7 |
| qwen3-30b-a3b (lo) @ 768 | 55.20 (lcnu) | 55.20 | 50.33 | 55.20 | **55.20** | 0 | 0.0 |
| qwen3-30b-a3b (lo) @ 896 | 59.08 (lcnu) | 59.08 | 55.99 | 59.08 | **59.08** | 0 | 0.0 |
| qwen3-30b-a3b (lo) @ 1024 | 62.45 (lcnu) | 62.45 | 61.18 | 62.22 | **62.35** | 4 | 7.0 |
| qwen3-30b-a3b (lo) @ 1152 | 65.67 (s3fifo) | 65.60 | 65.67 | 66.16 | **66.15** | 3 | 0.6 |
| qwen3-30b-a3b (lo) @ 1280 | 69.75 (s3fifo) | 68.61 | 69.75 | 70.09 | **70.07** | 3 | 1.3 |
| qwen3-30b-a3b (lo) @ 1408 | 73.23 (s3fifo) | 71.44 | 73.23 | 73.58 | **73.57** | 1 | 0.8 |
| qwen3-30b-a3b (lo) @ 1536 | 76.33 (s3fifo) | 74.01 | 76.33 | 76.54 | **76.53** | 1 | 0.6 |
| qwen3-30b-a3b (lo) @ 1792 | 81.47 (s3fifo) | 78.53 | 81.47 | 81.53 | **81.53** | 1 | 1.1 |
| qwen3-30b-a3b (lo) @ 2048 | 85.56 (s3fifo) | 82.29 | 85.56 | 85.36 | **85.37** | 3 | 5.7 |

Hit rates are in %. Streams are the decode-only real captures behind `results_real_granite3_1_moe_3b_decode.md`
and `results_real_qwen3-30b-a3b-hi.md` / `-lo.md`. Distinct (layer, expert) keys: 1262 / 5097 / 5842.
Accesses: 638208 / 786432 / 638208. LCNU uses alpha 0.3, beta 0.5 and prior top_k / num_experts.

## Cost

Per access, FOLLOW does one LCNU and one S3-FIFO metadata update, plus O(1) ordered-set work on the resident set and on `D`.
Ghost state is keys only, with no expert weights. The cost over LCNU is therefore the S3-FIFO metadata update and the
CostFit scoreboard. The C++ LCNU port measured <=685 ns/access worst-case median on these streams. The C++ FOLLOW port
and its engine benchmark are not part of this PR.

## Reproduce

```bash
python -m benchmarks.cache_simulator.research.run_policies_real \
    --files <stream>.pt --prefill skip --ignore-layers 0 \
    --model <model> --dataset <name> \
    --policies lcnu s3fifo costfit follow --capacities 768 1024 1280 1536 2048
```

Scope: decode-stream simulation only. Expert fetch latency and overlap are not modeled.
