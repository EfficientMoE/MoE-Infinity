# Expert-cache policies on real decode traces: correction + LCNU

Follow-up to #289 and the real-trace results in `results_real_qwen3_30b.md` /
`results_real_qwen3_235b.md`. It fixes how the decode access stream is built,
re-runs every baseline on real captures, adds seven classic
recency/frequency hybrids and a per-layer LRU, and proposes **LCNU**
(layer-cycle next-use), an online policy built for batch-1 decode. Which
policy to pick when: [`recommendations.md`](./recommendations.md).

> **Summary**
>
> 1. **The earlier "decode" tables counted prefill.** For merged multi-request
>    streams, `extract_access_sequence(include_prefill=False)` skipped only the
>    first record, so the prefill pass of every other request stayed in the
>    stream: 52% of the accesses of the published Qwen3-30B-A3B and Qwen3-235B
>    sweeps. Their small-capacity LRU and Belady numbers are mostly intra-prefill
>    reuse. Fixed: non-decode records are now dropped by `forward_mode`.
> 2. **On real decode, LRU gets zero hits below one token's working set.**
>    Batch-1 decode visits every MoE layer once per token, in order, so the
>    stream is a cyclic scan of `layers x top_k` keys per token (384 for
>    Qwen3-30B-A3B, 752 for Qwen3-235B). With at most `layers x top_k - top_k`
>    slots, every key is evicted before its layer runs again: LRU, ARC, SIEVE
>    and LFU-K all score exactly 0.
> 3. **LCNU beats LRU, LFU and all seven hybrids** on Qwen3-235B at every
>    capacity tested (+1.9 to +5.2 points over the best other online policy
>    for 256-4096 slots) and on Qwen3-30B-A3B at every capacity up to 3072 of
>    6144 slots; at 4096 slots LIRS edges it by 0.09 points. Below the LRU
>    cliff it gets about 2/3 of Belady's hits; above it, it removes a median
>    12% of LRU's misses. Its two parameters, picked on the first half of the
>    Qwen3-30B-A3B requests, are also what the Qwen3-235B tuning half picks.
> 4. **The classic hybrids help mostly next to the cliff.** Of the seven only
>    LIRS and W-TinyLFU hold up at every size below it, and LIRS / LRU-2 beat
>    LRU by up to 8 points just above it. Over all sizes above the cliff none
>    is consistently better than LRU: S3-FIFO, the best at large sizes, ranges
>    from 3.6 points below LRU to 2.5 points above it.
> 5. **Routing prediction is a prefetch tool, not an eviction tool.** Giving
>    LCNU the *true* routing of the current and next 8 layers adds at most 0.7
>    points (0.3 beyond the current layer's gate output); the eviction decision
>    that matters is whether an expert is reused one *token* later, and recency
>    + frequency already carry most of that signal.

All numbers are real: SGLang 0.5.20 `per_token` captures replayed through
the pure-Python simulator, Belady asserted as the ceiling for every policy and
cross-checked against the vectorized `opt_mrc` engine. Hardware for capture
and capture method: `capture_report.md`. Simulation: `python 3.12.3,
numpy 2.4.4, Linux x86_64`, 128-core host.

## 1. Traces and streams

| Model | MoE layers x experts, top-k | Prompts (greedy, 64 new tokens, batch 1) | Decode accesses | Keys per token | Key space |
| --- | --- | --- | ---: | ---: | ---: |
| Qwen3-235B-A22B-Instruct-2507-FP8 | 94 x 128, top-8 | 100 MMLU | 4,812,800 | 752 | 12,032 |
| Qwen3-30B-A3B | 48 x 128, top-8 | 150 MMLU | 3,686,400 | 384 | 6,144 |
| Qwen1.5-MoE-A2.7B-Chat | 24 x 60, top-4 | 12 mixed | 73,728 | 96 | 1,440 |

Traces live on disk under `/mnt/raid0nvme0/leyang/moeinf-traces/`
(git-ignored): `belady_<model>/` holds one converted dump per request,
`belady_<model>_stream/stream.pt` the merged stream.

**Decode-only** (`--prefill skip`, the default) is the primary stream: one
access per routed `(token, layer, expert)`, requests concatenated in capture
order, the cache persisting across requests. Batch-1 decode is the
latency-critical phase of offloaded MoE inference. Section 8 repeats the sweep
with prefill passes included the way an engine fetches them.

### What the old merged stream contained

| Model | Accesses in the published sweep | Decode | Leaked prefill |
| --- | ---: | ---: | ---: |
| Qwen3-235B | 10,006,864 | 4,812,800 | 5,194,064 (99 passes, 52%) |
| Qwen3-30B-A3B | 7,686,912 | 3,686,400 | 4,000,512 (149 passes, 52%) |
| Qwen1.5-MoE-A2.7B | 83,328 | 73,728 | 9,600 (11 passes, 12%) |

A prefill record is `[layers, prompt_tokens, top_k]`, flattened layer-major,
so each layer contributes a few hundred back-to-back accesses over ~70
distinct experts: hits for any recency policy, and hits no engine would pay
for (a layer's expert is fetched once per forward pass, not once per token).
Rebuilding the old stream reproduces the published tables exactly (e.g.
Qwen3-30B-A3B at 256 slots: LRU 0.4565, Belady 0.7028). On decode only:

| Qwen3-30B-A3B, hit rate | cap=256 | cap=512 | cap=1024 | cap=2048 |
| --- | --- | --- | --- | --- |
| Belady, published (prefill leaked) | 0.7028 | 0.7898 | 0.8705 | 0.9462 |
| Belady, decode only | 0.5013 | 0.6677 | 0.8122 | 0.9354 |
| LRU, published | 0.4565 | 0.6617 | 0.7631 | 0.8696 |
| LRU, decode only | 0.0000 | 0.4290 | 0.6358 | 0.8410 |
| LFU, published | 0.2021 | 0.3266 | 0.5069 | 0.7650 |
| LFU, decode only | 0.2151 | 0.3471 | 0.5383 | 0.7917 |

The published conclusion "LRU beats LFU 2x at 256 slots" came from the leaked
prefill passes; on decode LRU gets nothing there.

## 2. Why LRU collapses: decode is a cyclic scan

```
token t:    L0 [8 experts] -> L1 [8] -> ... -> L47 [8]      (384 accesses)
token t+1:  L0 [8 experts] -> L1 [8] -> ... -> L47 [8]
            ^ the earliest an expert of L0 can be reused
```

An expert routed at layer `L` for token `t` can next be used at layer `L` for
token `t+1`. In between, the other layers of the cycle touch at least
`layers x top_k - top_k` other distinct keys, so with that many slots or fewer
LRU evicts every expert just before its layer comes back: exactly 0 hits up to
376 slots on Qwen3-30B-A3B and 744 on Qwen3-235B. It is the classic LRU
failure on loops longer than the cache. ARC, SIEVE and LFU-K (64-access
window) also score exactly 0 there, 2Q and S3-FIFO nearly so.

What makes the stream cacheable is skew and burstiness (decode only,
`lcnu_ablation.py --stats-only`):

| | Qwen3-235B | Qwen3-30B-A3B | Qwen1.5-MoE-A2.7B |
| --- | --- | --- | --- |
| Share of accesses to the 20% most-used keys | 0.59 | 0.58 | 0.30 |
| P(expert routed again at the same layer on the next token) | 0.39 | 0.43 | 0.15 |
| ... if routing were independent per token at each key's frequency | 0.19 | 0.16 | 0.07 |

An expert just used is 2-3x more likely to be used on the next token than
its frequency alone predicts. Qwen1.5-MoE-A2.7B has much weaker skew and
burstiness, so every policy is further from Belady there.

## 3. Policies

On-demand (no prefetch) policies in `policies.py`:

| Policy | Rule | Source |
| --- | --- | --- |
| LRU, LFU, LFU-K, LHU | as before (`README.md`) | |
| Model-tracing | evict the expert with the lowest whole-trace count (offline profile; mild oracle) | BrainStorm |
| ARC | adaptive split between a recency and a frequency list, tuned by ghost hits | Megiddo & Modha, FAST'03 |
| 2Q | FIFO probation (25%), ghost list (50%), LRU main | Johnson & Shasha, VLDB'94 |
| LRU-2 | evict the largest backward 2-distance; history kept for evicted keys | O'Neil et al., SIGMOD'93 |
| LIRS | keep keys with low inter-reference recency; 1% of slots for the rest | Jiang & Zhang, SIGMETRICS'02 |
| S3-FIFO | 10% small FIFO, main FIFO with 2-bit counters, ghost FIFO (Algorithm 1) | Yang et al., SOSP'23 |
| SIEVE | FIFO with a visited bit and a moving hand | Zhang et al., NSDI'24 |
| W-TinyLFU | 1% LRU window, 80/20 SLRU main, admit only if more frequent than the victim | Einziger et al., ACM ToS 2017 |
| Per-layer LRU | `capacity / layers` slots per layer, LRU inside a layer | Mixtral-Offloading |
| **LCNU** | evict the farthest *expected* next use (below) | this study |

The four `*` prefetch policies of the earlier study are still swept; their
predictors read the future routing from the trace, so they are upper bounds
and their demand hit rates are listed only in the full tables.

## 4. LCNU: layer-cycle next-use

Belady evicts the key whose next use is farthest away. In batch-1 decode half
of that is known online: the engine knows which layer runs now, so the time
until layer `L` runs again is exact. LCNU estimates the other half, how many
more tokens until the expert is routed again:

```
score(L, e) = d(L) + layers * (1 / p(L, e) - 1)        evict max score

d(L)  layer-steps until layer L runs again:
      (L - current) mod layers; for the current layer, `layers` if the expert
      was already used in this visit, else 0
p     estimated probability that the expert is routed on a visit of L:
      p = beta * ema + (1 - beta) * (count + 4 * prior) / (visits + 4)
ema   EMA (rate alpha) over the visits of L of "routed on this visit"
      -> recency / burstiness
count/visits   routing frequency since the expert was first seen
      (smoothed toward prior = top_k / experts) -> long-run popularity
```

`layers * (1/p - 1)` is the expected wait in layer-steps if routing at each
visit were a coin flip with probability `p`. Defaults `alpha = 0.3`,
`beta = 0.5` were picked on the first half of the Qwen3-30B-A3B requests
(section 6) and used unchanged everywhere else.

Runtime cost: per-expert `(ema, count, first visit, last visit)` and the
layer cursor. Between two runs of a layer the scores of its experts do not
change, so each layer's best victim is cached and recomputed only after that
layer runs or loses its candidate; an eviction compares one candidate per
layer (`O(layers)`). Replaying the 4.8M-access Qwen3-235B trace takes 0.5-1.5
minutes of pure Python per capacity (LRU: 2-4 s).

## 5. Results (decode only)

Hit rate = fraction of decode expert accesses served from the cache.
"Best other" is the best online policy except LCNU and the offline
model-tracing profile. Full tables with every policy:
`results_real_qwen3_235b_decode.md`, `results_real_qwen3_30b_decode.md`,
`results_real_qwen1_5_moe_a2_7b_decode.md` (and `.csv`).

### Qwen3-235B-A22B-FP8 (94 x 128, 752 keys per token)

| Slots | Belady | LRU | LFU | Best other | LCNU | LCNU vs best | Misses vs LRU |
| ---: | --- | --- | --- | --- | --- | ---: | ---: |
| 256 | 0.3067 | 0.0000 | 0.1577 | 0.1731 LIRS | **0.2051** | +3.2 | -20.5% |
| 512 | 0.4898 | 0.0000 | 0.2561 | 0.2820 LIRS | **0.3272** | +4.5 | -32.7% |
| 768 | 0.5906 | 0.3841 | 0.3317 | 0.3841 LRU | **0.4130** | +2.9 | -4.7% |
| 1024 | 0.6581 | 0.3850 | 0.3941 | 0.4315 LIRS | **0.4808** | +4.9 | -15.6% |
| 1536 | 0.7485 | 0.5253 | 0.4966 | 0.5320 ARC | **0.5840** | +5.2 | -12.4% |
| 2048 | 0.8082 | 0.6171 | 0.5805 | 0.6195 per-layer LRU | **0.6608** | +4.1 | -11.4% |
| 3072 | 0.8859 | 0.7463 | 0.7147 | 0.7465 S3-FIFO | **0.7765** | +3.0 | -11.9% |
| 4096 | 0.9320 | 0.8334 | 0.8150 | 0.8413 S3-FIFO | **0.8602** | +1.9 | -16.1% |
| 6144 | 0.9785 | 0.9407 | 0.9389 | 0.9490 S3-FIFO | **0.9553** | +0.6 | -24.5% |
| 8192 | 0.9947 | 0.9876 | 0.9892 | 0.9892 LFU | **0.9911** | +0.2 | -28.4% |

### Qwen3-30B-A3B (48 x 128, 384 keys per token)

| Slots | Belady | LRU | LFU | Best other | LCNU | LCNU vs best | Misses vs LRU |
| ---: | --- | --- | --- | --- | --- | ---: | ---: |
| 128 | 0.3068 | 0.0000 | 0.1255 | 0.1830 LIRS | **0.2147** | +3.2 | -21.5% |
| 256 | 0.5013 | 0.0000 | 0.2151 | 0.3026 LIRS | **0.3415** | +3.9 | -34.2% |
| 384 | 0.6026 | 0.3053 | 0.2864 | 0.3970 LHU | **0.4263** | +2.9 | -17.4% |
| 512 | 0.6677 | 0.4290 | 0.3471 | 0.4432 LIRS | **0.4916** | +4.8 | -11.0% |
| 768 | 0.7534 | 0.5551 | 0.4504 | 0.5596 per-layer LRU | **0.5907** | +3.1 | -8.0% |
| 1024 | 0.8122 | 0.6358 | 0.5383 | 0.6361 per-layer LRU | **0.6620** | +2.6 | -7.2% |
| 1536 | 0.8888 | 0.7493 | 0.6816 | 0.7522 S3-FIFO | **0.7729** | +2.1 | -9.4% |
| 2048 | 0.9354 | 0.8410 | 0.7917 | 0.8532 S3-FIFO | **0.8621** | +0.9 | -13.2% |
| 3072 | 0.9808 | 0.9465 | 0.9260 | 0.9554 S3-FIFO | **0.9560** | +0.1 | -17.7% |
| 4096 | 0.9956 | 0.9892 | 0.9818 | **0.9904** LIRS | 0.9895 | -0.1 | -3.5% |

### Qwen1.5-MoE-A2.7B (24 x 60, 96 keys per token)

| Slots | Belady | LRU | LFU | Best other | LCNU | LCNU vs best |
| ---: | --- | --- | --- | --- | --- | ---: |
| 32 | 0.1856 | 0.0000 | 0.0367 | 0.0470 LIRS | **0.0526** | +0.6 |
| 64 | 0.2860 | 0.0000 | 0.0699 | 0.0865 LIRS | **0.0956** | +0.9 |
| 128 | 0.4142 | 0.1485 | 0.1320 | 0.1566 LIRS | **0.1737** | +1.7 |
| 256 | 0.5727 | 0.2494 | 0.2444 | 0.2775 LIRS | **0.2887** | +1.1 |
| 512 | 0.7538 | 0.4698 | 0.4475 | 0.4866 2Q | **0.4896** | +0.3 |
| 1024 | 0.9286 | 0.8068 | 0.7847 | 0.8073 ARC | **0.8113** | +0.4 |

Observations:

- **Below the LRU cliff** (up to 744 / 376 / 92 slots), LCNU reaches 67-70%
  of Belady's hits on both Qwen3 models. Of the baselines only LIRS,
  W-TinyLFU, LFU and model-tracing keep hitting at every size there.
- **Above the cliff** LCNU removes 4-28% of LRU's misses (median 12%; each
  miss is an expert transfer over PCIe) and closes 6-50% of the LRU-to-Belady
  gap.
- **Classic hybrids are not a consistent win over LRU** once the cache holds
  a token's working set. Right above the cliff the loop-resistant ones help
  (Qwen3-30B-A3B at 384 slots: LIRS +7.8, LRU-2 +7.0 points; Qwen3-235B at
  1024: LIRS +4.7). S3-FIFO is the best of them at large sizes, but over all
  sizes above the cliff it ranges from -3.6 to +2.5 points against LRU, and
  SIEVE stays far below it.
- **Frequency alone loses to LRU at most sizes above the cliff:** LFU and
  the offline whole-trace profile (model-tracing) miss the token-to-token
  reuse, and the profile is only 0-1.5 points better than online LFU.
- The gains shrink on Qwen1.5-MoE-A2.7B (+0.2 to +1.7 points), whose routing
  has half the skew and burstiness of Qwen3.

## 6. Held-out selection and ablations (Qwen3-30B-A3B)

`lcnu_ablation.py` splits the 150 requests in capture order: requests 0-74 to
pick `(alpha, beta)` from a 5 x 4 grid, requests 75-149 to report. The
selected setting, `alpha = 0.3, beta = 0.5`, is the shipped default. The top
ten grid points are within 1.6 points of each other at every capacity, so the
result is not sensitive to the exact choice. Full output:
`results_lcnu_ablation_qwen3_30b.md`.

Held-out half (1,843,200 decode accesses):

| Policy | cap=128 | cap=256 | cap=384 | cap=512 | cap=1024 | cap=2048 |
| --- | --- | --- | --- | --- | --- | --- |
| Belady | 0.3079 | 0.5052 | 0.6065 | 0.6718 | 0.8158 | 0.9380 |
| LRU | 0.0000 | 0.0000 | 0.3151 | 0.4398 | 0.6432 | 0.8490 |
| LFU | 0.1303 | 0.2251 | 0.2970 | 0.3577 | 0.5488 | 0.8025 |
| ARC | 0.0000 | 0.0000 | 0.3403 | 0.4404 | 0.6368 | 0.8517 |
| LIRS | 0.1859 | 0.3066 | 0.3875 | 0.4481 | 0.6138 | 0.8347 |
| S3-FIFO | 0.0000 | 0.0469 | 0.3655 | 0.4245 | 0.6222 | 0.8609 |
| W-TinyLFU | 0.1721 | 0.2760 | 0.3467 | 0.4042 | 0.5875 | 0.8377 |
| **LCNU** | **0.2179** | **0.3461** | **0.4317** | **0.4976** | **0.6697** | **0.8662** |
| LCNU without the cycle term `d(L)` | 0.1893 | 0.3151 | 0.4048 | 0.4742 | 0.6623 | 0.8652 |
| LCNU, EMA only (`beta = 1`) | 0.2139 | 0.3396 | 0.4193 | 0.4906 | 0.6568 | 0.8523 |
| LCNU, frequency only (`beta = 0`) | 0.1486 | 0.2455 | 0.3174 | 0.3762 | 0.5625 | 0.8103 |

- The **cycle term** is worth 2.3-3.1 points up to 512 slots, where choosing
  which layer to give up matters most, and fades to 0.1 points at 2048.
- The **recency EMA** carries most of the gain over frequency-based policies;
  adding long-run **frequency** is worth another 0.4-1.4 points.

The same protocol on Qwen3-235B (requests 0-49 / 50-99,
`results_lcnu_ablation_qwen3_235b.md`) independently selects the same
`alpha = 0.3, beta = 0.5`, and on its held-out half LCNU again leads at every
capacity (256-4096 slots: +3.2, +4.4, +2.8, +4.9, +4.2, +1.8 points over the
best of LRU, LFU, ARC, LIRS, S3-FIFO and W-TinyLFU), with the same ablation
pattern.

## 7. Would routing prediction help eviction?

Cross-layer predictors (gate look-ahead, activation matrices, hidden-state
similarity; Pre-gated MoE, ProMoE, HOBBIT) predict the *next few layers* of
the current token. To bound what that could add to eviction, LCNU was given
the true routing of the current layer and of the next `h` layer visits, read
from the trace (first 10 Qwen3-30B-A3B requests, 245,760 accesses):

| Policy | cap=128 | cap=256 | cap=512 | cap=1024 | cap=2048 |
| --- | --- | --- | --- | --- | --- |
| Belady | 0.3030 | 0.4958 | 0.6600 | 0.8007 | 0.9228 |
| LCNU (online) | 0.2122 | 0.3386 | 0.4858 | 0.6560 | 0.8433 |
| + true routing, current layer only | 0.2131 | 0.3407 | 0.4901 | 0.6599 | 0.8444 |
| + next 1 layer | 0.2131 | 0.3410 | 0.4903 | 0.6602 | 0.8445 |
| + next 8 layers | 0.2154 | 0.3434 | 0.4927 | 0.6615 | 0.8451 |
| + next 16 layers | 0.2214 | 0.3496 | 0.4975 | 0.6635 | 0.8456 |
| + next 47 layers (whole next cycle) | 0.2816 | 0.4188 | 0.5489 | 0.6803 | 0.8502 |

Even a perfect predictor of the current and next 8 layers adds at most 0.7
points, 0.3 of them beyond what the current layer's gate output already gives.
The eviction victims are experts whose layer just ran; what decides whether to
keep them is reuse one *token* later, which near-term routing prediction does
not see. A perfect one-token look-ahead (h = 47) would add up to 8 points;
Belady gets the rest by also seeing further ahead. Cross-layer predictors
should drive **prefetch** (hiding transfer time), not eviction.

## 8. Prefill included (as an engine fetches it)

`--prefill union`: each prefill pass contributes one access per distinct
expert per layer, followed by the decode accesses; all are counted. Full
tables: `results_real_qwen3_235b_union.md`, `results_real_qwen3_30b_union.md`.

| Qwen3-235B slots | Belady | LRU | LFU | Best other | LCNU | LCNU vs best |
| ---: | --- | --- | --- | --- | --- | ---: |
| 256 | 0.2781 | 0.0000 | 0.1435 | 0.1571 LIRS | **0.1860** | +2.9 |
| 512 | 0.4480 | 0.0000 | 0.2344 | 0.2579 LIRS | **0.2976** | +4.0 |
| 1024 | 0.6079 | 0.3392 | 0.3644 | 0.3965 LIRS | **0.4375** | +4.1 |
| 2048 | 0.7600 | 0.5479 | 0.5452 | 0.5688 LRU-2 | **0.6017** | +3.3 |
| 4096 | 0.8985 | 0.7557 | 0.7835 | 0.7849 LRU-2 | **0.7944** | +0.9 |
| 6144 | 0.9626 | 0.8828 | **0.9195** | **0.9195** LFU | 0.9088 | -1.1 |
| 8192 | 0.9918 | 0.9768 | 0.9834 | **0.9835** LHU | 0.9834 | -0.0 |

| Qwen3-30B-A3B slots | Belady | LRU | LFU | Best other | LCNU | LCNU vs best |
| ---: | --- | --- | --- | --- | --- | ---: |
| 128 | 0.2775 | 0.0000 | 0.1146 | 0.1647 LIRS | **0.1940** | +2.9 |
| 256 | 0.4564 | 0.0000 | 0.1971 | 0.2748 LIRS | **0.3095** | +3.5 |
| 512 | 0.6154 | 0.3773 | 0.3201 | 0.4057 LIRS | **0.4471** | +4.1 |
| 1024 | 0.7623 | 0.5636 | 0.5035 | 0.5763 ARC | **0.6052** | +2.9 |
| 2048 | 0.9009 | 0.7593 | 0.7573 | 0.7932 S3-FIFO | **0.8001** | +0.7 |
| 3072 | 0.9650 | 0.8850 | 0.9029 | **0.9298** W-TinyLFU | 0.9108 | -1.9 |
| 4096 | 0.9928 | 0.9797 | 0.9730 | **0.9845** W-TinyLFU | 0.9825 | -0.2 |

With prefill in the stream LCNU still leads up to about a third of the key
space (2048 of 6144 slots on Qwen3-30B-A3B, 4096 of 12,032 on Qwen3-235B) but
falls up to 1.9 points behind frequency-filtering policies (W-TinyLFU, LFU)
in larger caches. A prefill pass touches every expert its prompt routes to,
on average 3,300 distinct keys on Qwen3-30B-A3B and 6,400 on Qwen3-235B.
LCNU treats each touch like a decode routing event and lets that scan
displace hot experts, whereas admission filters keep it out. In an
exploratory run (not in the repo),
weighting a prefill visit's evidence by `top_k / experts touched` recovered
about 1 point at 2048-3072 slots on Qwen3-30B-A3B but lost 0.2 at 4096, so it
is not shipped. A real engine knows when it runs prefill and could route those
fetches around the cache or through an admission filter.

## 9. Limitations

- Batch-1 greedy decode of 64 tokens on MMLU-style prompts. Longer outputs,
  sampling, other tasks or batched decode (several tokens per layer visit)
  change the reuse pattern; batched decode weakens the cyclic-scan effect.
- Hit rate only: no transfer-time or overlap model, and no prefetch.
- LCNU's two parameters were selected on one model's tuning half. They
  transferred to the other models here, but on an exploratory synthetic trace
  with much stronger token-to-token reuse (60%, against 39-43% on Qwen3) LIRS
  beat it below one token's working set and plain LRU above it.
- The Qwen3-235B capture is 100 prompts on TP4; the Qwen1.5 smoke trace is 12
  prompts, so its differences of a few tenths of a point are within noise.

## 10. Reproduce

```bash
TD=/mnt/raid0nvme0/leyang/moeinf-traces
POLICIES="lru lfu lfu_k model_tracing lhu arc 2q lru_k lirs s3fifo sieve \
  w_tinylfu layer_lru lcnu single_layer_ahead stride_prefetch \
  multi_layer_ahead activation_matrix"

# Decode-only sweeps (score-based is omitted: it decays every expert's score
# on every access, hours per capacity on these traces).
python -m benchmarks.cache_simulator.research.run_policies_real \
    --files "$TD/belady_qwen3_235b_stream/stream.pt" --model qwen3-235b \
    --dataset mmlu100-greedy64 --policies $POLICIES --workers 32 \
    --capacities 256 512 768 1024 1536 2048 3072 4096 6144 8192 \
    --tag qwen3_235b_decode
python -m benchmarks.cache_simulator.research.run_policies_real \
    --files "$TD/belady_qwen3_30b_stream/stream.pt" --model qwen3-30b-a3b \
    --dataset mmlu150-greedy64 --policies $POLICIES --workers 32 \
    --capacities 128 256 384 512 768 1024 1536 2048 3072 4096 \
    --tag qwen3_30b_decode
python -m benchmarks.cache_simulator.research.run_policies_real \
    --files "$TD/belady_smoke_stream/stream.pt" --model qwen1.5-moe-a2.7b \
    --dataset mixed20-greedy64 --policies $POLICIES --workers 32 \
    --capacities 32 64 96 128 192 256 384 512 768 1024 \
    --tag qwen1_5_moe_a2_7b_decode
# Prefill included: add --prefill union (tags qwen3_30b_union, qwen3_235b_union).

# Selection, ablations, routing-oracle bound, routing statistics.
python -m benchmarks.cache_simulator.research.lcnu_ablation \
    --files "$TD/belady_qwen3_30b/*.pt" --workers 16 \
    --capacities 128 256 384 512 1024 2048 \
    --oracle-requests 10 --horizons 0 1 2 4 8 16 47
python -m benchmarks.cache_simulator.research.lcnu_ablation \
    --files "$TD/belady_qwen3_235b/*.pt" --workers 16 \
    --capacities 256 512 768 1024 2048 4096
python -m benchmarks.cache_simulator.research.lcnu_ablation \
    --files "$TD/belady_qwen3_30b/*.pt" --stats-only
```

## References

- L. A. Belady. A study of replacement algorithms for a virtual-storage
  computer. IBM Systems Journal 5(2), 1966.
- N. Megiddo, D. S. Modha. ARC: A Self-Tuning, Low Overhead Replacement
  Cache. FAST 2003.
- T. Johnson, D. Shasha. 2Q: A Low Overhead High Performance Buffer
  Management Replacement Algorithm. VLDB 1994.
- E. J. O'Neil, P. E. O'Neil, G. Weikum. The LRU-K Page Replacement Algorithm
  for Database Disk Buffering. SIGMOD 1993.
- S. Jiang, X. Zhang. LIRS: An Efficient Low Inter-reference Recency Set
  Replacement Policy to Improve Buffer Cache Performance. SIGMETRICS 2002.
- J. Yang, Y. Zhang, Z. Qiu, Y. Yue, K. V. Rashmi. FIFO Queues are All You
  Need for Cache Eviction. SOSP 2023.
- Y. Zhang, J. Yang, Y. Yue, Y. Vigfusson, K. V. Rashmi. SIEVE is Simpler
  than LRU: an Efficient Turn-Key Eviction Algorithm for Web Caches.
  NSDI 2024.
- G. Einziger, R. Friedman, B. Manes. TinyLFU: A Highly Efficient Cache
  Admission Policy. ACM Transactions on Storage 13(4), 2017.
- A. Eliseev, D. Mazur. Fast Inference of Mixture-of-Experts Language Models
  with Offloading. arXiv:2312.17238, 2023.
