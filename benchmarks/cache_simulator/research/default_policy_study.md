# Round 3: better online predictors and one default policy

Follow-up to round 2 ([`prefetch_eviction_study.md`](./prefetch_eviction_study.md))
for #289. Round 2 found that prefetch and eviction can be chosen
independently. With its best online predictor, however, prefetch cut stall by
only 3-12%, so the eviction policy mattered more. This round does three
things:

1. It tries stronger routing-only predictors.
2. It revisits eviction, which still dominated.
3. It fixes **one default policy** for all cache sizes, both models and both
   streams, replacing round 2's per-regime table.

> **Summary**
>
> 1. **One default policy:**
>    - **LCNU with decode-only routing statistics** (`lcnu_p`);
>    - **nearest-neighbour routing-trajectory prefetch** (`knn`): one layer
>      ahead, 8 candidates, at most 2 transfers per layer, decode passes
>      only;
>    - **probation** admission.
>
>    The same parameters hold at every capacity, for both models and both
>    streams. Across all 24 held-out cells (2 models x 2 streams x 6
>    capacities) it is on average within 0.5% of the best online
>    configuration found for that cell in rounds 2 and 3, and never more
>    than 6.2% off.
> 2. **A better predictor makes prefetch matter.** The new predictor is the
>    next-layer routing of the past tokens whose last 8 layers routed most
>    like the current token's. Its recall@8 is 0.78 on Qwen3-30B-A3B and
>    0.71 on Qwen3-235B, against 0.61 / 0.56 for round 2's best (hybrid
>    co-activation). Online logistic regression reaches 0.72 / 0.66.
>    On held-out decode the default cuts stall per token by 15-31%
>    (Qwen3-30B-A3B) and 9-16% (Qwen3-235B) against LCNU alone; round 2's
>    best pair cut 7-12% and 3-6%. That is 17-82% of a perfect predictor's
>    gain, depending on the cache size, against 4-47% for round 2's pair.
> 3. **Eviction still dominates below one token's working set; prefetch
>    matters as much or more above it.** On Qwen3-30B-A3B, LRU -> LCNU
>    removes 22-35% of the stall below the working set, where kNN prefetch
>    removes 15-16%; above 1.3x the working set prefetch removes 19-31% and
>    eviction 7-11%. On Qwen3-235B, whose link moves fewer experts per layer
>    of compute (0.67 vs 1.14), eviction is the equal or larger lever at 5
>    of 6 sizes (prefetch is ahead only at 2,048 slots: -13.8% vs -11.5%).
>    Both levers compose: the default needs both.
> 4. **Eviction fix for prefill-heavy serving.** Round 1 found W-TinyLFU or
>    LFU best in large caches when prefill passes are in the stream. Ignoring
>    prefill passes as routing evidence (`lcnu_p`) removes 9-24% of LCNU's
>    stall there; on Qwen3-235B it now beats LFU and W-TinyLFU at every
>    size. Admission control on top (not caching a miss whose expected next
>    use is later than the victim's) adds nothing (0.04% on the tuning
>    half).
> 5. **Against today's engine**, approximated as LFU-like eviction plus the
>    same-expert-id `speculative_prefetch`, the default cuts stall per decode
>    token by 31-62% (Qwen3-30B-A3B) and 18-50% (Qwen3-235B).

All numbers are real SGLang 0.5.20 `per_token` routing, replayed with the
round-2 harness and timing model: the nominal scenario uses the measured
PCIe 5.0 link and the measured per-layer compute, with demand priority.
Parameters were selected on requests 0-74 of Qwen3-30B-A3B and reported on
requests 75-149 and on Qwen3-235B requests 50-99 without retuning. Demand hit
rate is asserted `<= Belady` in every run (`<=` Belady with bypass for
policies that may bypass the cache).

## 1. More prediction algorithms

All new predictors are online (unit-tested for causality), use routing only
and are queried once per layer visit after the gate, as in round 2
([`../prefetchers.py`](../prefetchers.py)).

| Predictor | Rule | State per layer | Selected parameters (tuning half) |
| --- | --- | --- | --- |
| `coactivation`, depth / decay | round 2's `P(e' at L+d \| e at L)`, with evidence from the last `depth` visits, weighted `decay^j` (Klotski uses several preceding layers) | E x E counts per (layer, distance) | depth 4, decay 0.5 |
| `coactivation`, naive Bayes | `log pi(e') + sum_j w_j sum_{e in S_j} log(P(e' \| e) / pi(e'))`, smoothed toward the layer's routing frequency `pi` | same | depth 3, decay 0.5, smoothing 0.25 |
| `logistic` | online logistic regression per (layer, distance): `sigmoid(b + sum_{e in S} W[e] + w_prev * prev + w_ema * ema)`, SGD on the log-loss | E x E weights + E bias | learning rate 0.1 |
| **`knn`** | the 8 past tokens whose last `context` = 8 layers share the most (layer, expert) pairs with the current token vote, weight `overlap^8`, for the next layer's experts; a bank of the last 8192 tokens per layer (MoE-Infinity's activation-matrix matching at token, not request, granularity) | 8192 x (8 x 128) bits | context 8, neighbours 8, power 8 |

Selection was by mean recall@8 over distances 1-4 on the tuning half,
from 78 grid points (`results_prefetch_qwen3_30b_tune_predict_r3.csv`).
kNN improves with longer contexts up to 8-12 layers, a handful of
neighbours (4-8 rather than 16-128) and sharper weights (power 8 rather than
1-4). Context 12 scored 0.7727 and context 8 scored 0.7717; context 8 was
kept because it is within 0.1 point with two thirds of the signature.

Held-out accuracy for the next visit
(`results_prefetch_qwen3_{30b,235b}_held_predict{,_r3}.csv`). With 8
candidates and 8 routed experts, precision equals recall.

| Predictor | 30B recall@8 | 30B precision@2 | 30B recall@16 | 235B recall@8 | 235B precision@2 | 235B recall@16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| previous token (r2) | 0.438 | 0.551 | - | 0.387 | 0.494 | - |
| co-activation (r2) | 0.593 | 0.804 | 0.793 | 0.551 | 0.750 | 0.763 |
| hybrid (r2) | 0.607 | 0.827 | 0.808 | 0.563 | 0.771 | 0.776 |
| co-activation, naive Bayes | 0.655 | - | - | 0.602 | - | - |
| logistic | 0.722 | 0.936 | 0.894 | 0.661 | 0.889 | 0.853 |
| **knn** | **0.779** | **0.943** | **0.912** | **0.714** | **0.907** | **0.879** |
| same expert id (engine today) | 0.058 | 0.055 | - | 0.059 | 0.057 | - |

Routing is a function of the hidden state. Tokens whose last 8 layers routed
alike are in similar hidden states, so the top-2 kNN candidates are routed
94% (Qwen3-30B-A3B) and 91% (Qwen3-235B) of the time. Pairwise co-activation
counts cannot express this: they score each source expert independently.

## 2. Eviction: prefill-aware LCNU

Two LCNU options, both off by default (plain LCNU is bit-identical):

- `prefill_weight` (`lcnu_p` = 0): a visit routing more experts than one token
  can (more than `top_k`) is a prefill pass, and its routing counts as
  `prefill_weight` of a routing event. The engine knows the forward mode at
  the gate.
- `bypass` (`lcnu_b`): a missed expert whose LCNU expected next use is no
  sooner than the victim's is served from a transient slot instead of being
  cached. The engine already has `transient_on_pressure` admission. Its
  bound is `belady_simulate(..., bypass=True)`, the optimum when admission
  is optional, checked against a brute-force optimum.

Tuning half, prefill-union stream, Qwen3-30B-A3B, no prefetch, stall ms per
token (`results_prefetch_qwen3_30b_tune_union_evict.md`):

| Evictor | 128 | 256 | 512 | 1024 | 2048 | 3072 | Mean vs LCNU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LCNU | 58.26 | 49.95 | 40.04 | 28.65 | 14.71 | 6.67 | 1.000 |
| `lcnu_p` (prefill_weight 0) | 58.25 | 49.92 | 39.74 | 27.65 | 12.73 | 4.76 | **0.923** |
| prefill_weight 0.1 / 0.25 / 0.5 | | | | | | | 0.936 / 0.958 / 0.980 |
| `lcnu_b` (+ bypass every miss) | 58.17 | 49.89 | 39.74 | 27.65 | 12.73 | 4.76 | 0.922 |
| W-TinyLFU | 61.13 | 54.16 | 45.68 | 33.52 | 16.04 | 5.62 | 1.063 |
| LFU | 61.96 | 55.16 | 45.53 | 32.32 | 14.56 | 5.30 | 1.036 |

Each prefill pass touches about 70 experts per layer once. Counted as
routing evidence, those touches displace decode-hot experts. Ignored, they
leave the cache through LCNU's normal ranking. Bypass changes nothing
measurable once the evidence is right: `lcnu_p` with bypass on prefill
misses scores 0.923, and LCNU with that bypass but full prefill evidence
scores 1.000, identical to LCNU
(`results_prefetch_qwen3_30b_tune_union_evict_bypass.csv`). On decode-only
streams `lcnu_p` is LCNU exactly.

## 3. Choosing one default

Stage B ran on the tuning half: LCNU with each candidate prefetch
configuration, nominal stall relative to LCNU alone at the six capacities
(`results_prefetch_qwen3_30b_tune_r3_default.md`; 70 configurations, 426
runs). "issue" caps the prefetches issued per target layer, so a deeper
candidate list costs at most that many transfers.

| Prefetcher (h = 1) | Admission | 128 | 256 | 512 | 1024 | 2048 | 3072 | Mean | Worst regret vs per-cell best |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **knn, 8 candidates** | **probation, issue 2** | 0.853 | 0.837 | 0.810 | 0.763 | 0.684 | 0.690 | **0.773** | +4.6% |
| knn, 8 candidates | probation, issue 4 | 0.855 | 0.838 | 0.809 | 0.761 | 0.684 | 0.690 | 0.773 | +4.6% |
| knn, 32 candidates, score >= 0.25 | probation, issue 2 | 0.853 | 0.837 | 0.815 | 0.777 | 0.701 | 0.660 | 0.774 | +3.3% |
| logistic, 8 candidates | probation, issue 2 | 0.867 | 0.861 | 0.847 | 0.819 | 0.807 | 0.890 | 0.848 | +34.9% |
| hybrid (r2), 16 candidates | probation, issue 2 | 0.924 | 0.934 | 0.944 | 0.938 | 0.874 | 0.946 | 0.927 | +43.3% |
| best per cell (any of the 70) | | 0.852 | 0.834 | 0.803 | 0.753 | 0.684 | 0.660 | | |

The rule was the lowest mean over the six capacities. Round 2 needed the
budget to grow with the cache (4 -> 16 candidates). kNN does not: one
setting is within 4.6% of the best per cell on the tuning half. An adaptive
count (up to 32 candidates scoring at least 0.25) is better at the largest
cache (0.660 vs 0.690) but worse at 1-5x the working set; on the mean the
two are within 0.2% (0.774 vs 0.773). The `issue 2`
cap also makes the policy indifferent to the link discipline: FIFO and
demand-priority stall are identical in every held-out cell (section 4).

**The default policy**, one setting for every cache size and model:

| Component | Setting |
| --- | --- |
| Evictor | `lcnu_p`: LCNU (alpha 0.3, beta 0.5, prior top_k / experts), prefill passes give no routing evidence |
| Prefetcher | `knn`: context 8 layers, 8 neighbours, weight `overlap^8`, bank 8192 tokens per layer; decode passes only (`decode_width = top_k`) |
| Horizon / candidates / transfers | 1 layer ahead / 8 per layer / at most 2 issued per layer |
| Admission | `probation`: staged outside the evictor, evicted first under pressure, dropped at the gate if not routed, promoted on use |

Without `decode_width` the kNN bank also stores prefill passes. Their wide
signatures win every overlap comparison, so prefill noise crowds out the
real neighbours. On the prefill-union stream, prefetch then made stall worse
than LCNU-P alone in 11 of 12 cells, by up to 10%
(`results_prefetch_*_held_union_r3.md`). Restricting the predictor to decode
passes fixes it.

## 4. Held-out results

Stall per decode token (ms), nominal timing
([`results_default_policy_analysis.md`](./results_default_policy_analysis.md);
sweeps `results_prefetch_*_held{,_r3}`, `*_held_union{,_r3d}`).
"Per-cell best" is the best online configuration of every round-2 and
round-3 sweep at that cell. On the union stream the baseline is `lcnu_p`
alone (LCNU-P).

### Qwen3-30B-A3B

| Slots (x WS) | LFU | LCNU | Default | vs LCNU | vs LFU | Regret vs per-cell best | Oracle bound | Union: LCNU-P | Union: default | Union vs LCNU-P |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 (0.33) | 55.20 | 49.64 | 42.42 | -14.6% | -23.2% | 0.0% | 40.64 | 57.67 | 50.43 | -12.5% |
| 256 (0.67) | 49.18 | 41.50 | 34.83 | -16.1% | -29.2% | 0.0% | 32.44 | 49.23 | 42.54 | -13.6% |
| 512 (1.33) | 40.76 | 31.89 | 25.90 | -18.8% | -36.5% | +0.1% | 22.90 | 39.05 | 33.04 | -15.4% |
| 1024 (2.67) | 28.64 | 20.96 | 16.05 | -23.4% | -44.0% | +0.3% | 11.98 | 27.09 | 22.17 | -18.1% |
| 2048 (5.3) | 12.53 | 8.49 | 5.88 | -30.8% | -53.1% | +0.1% | 1.92 | 12.49 | 9.90 | -20.7% |
| 3072 (8.0) | 4.36 | 2.80 | 1.94 | -30.8% | -55.5% | +4.6% | 0.15 | 4.94 | 4.09 | -17.1% |

### Qwen3-235B-A22B-FP8

| Slots (x WS) | LFU | LCNU | Default | vs LCNU | vs LFU | Regret vs per-cell best | Oracle bound | Union: LCNU-P | Union: default | Union vs LCNU-P |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 (0.34) | 209.15 | 197.26 | 180.39 | -8.6% | -13.7% | 0.0% | 176.58 | 228.87 | 212.00 | -7.4% |
| 512 (0.68) | 184.46 | 166.67 | 150.63 | -9.6% | -18.3% | 0.0% | 145.76 | 197.14 | 181.07 | -8.2% |
| 1024 (1.36) | 149.87 | 128.21 | 113.67 | -11.3% | -24.2% | +0.1% | 107.04 | 156.51 | 141.91 | -9.3% |
| 2048 (2.72) | 103.21 | 83.45 | 71.91 | -13.8% | -30.3% | +0.2% | 62.22 | 107.72 | 96.16 | -10.7% |
| 4096 (5.4) | 44.99 | 33.87 | 28.60 | -15.6% | -36.4% | 0.0% | 15.69 | 49.63 | 44.43 | -10.5% |
| 6016 (8.0) | 16.10 | 11.75 | 10.14 | -13.6% | -37.0% | +1.2% | 2.47 | 20.36 | 18.80 | -7.6% |

On decode the default is within 0.3% of the per-cell best up to about 5x
the working set. At 8x the adaptive variant (up to 32 candidates scoring at
least 0.25) is better: 1.85 vs 1.94 ms (Qwen3-30B-A3B) and 10.02 vs 10.14 ms
(Qwen3-235B).

On the prefill-union stream the default is the best online configuration, or
within 0.01 ms of it, in 11 of 12 cells. The exception is Qwen3-30B-A3B at
3,072 slots, where W-TinyLFU + kNN is 3.85 ms against 4.09 (+6.2%). Over all
24 cells the default's regret is +0.5% on average and +6.2% at worst.

Against plain LCNU on the prefill-union stream (round 2's evictor), the
default cuts stall by 13-31% (Qwen3-30B-A3B; 5.93 -> 4.09 ms at 3,072) and
7-30% (Qwen3-235B; 26.83 -> 18.80 ms at 6,016).

Against round 2's per-regime recommendation (LCNU + hybrid on probation,
tuned per capacity), held-out decode stall is lower by 7-26% (30B: 45.62 ->
42.42 ... 2.62 -> 1.94 ms) and 4-13% (235B: 188.09 -> 180.39 ... 11.40 ->
10.14 ms).

Against today's engine path (LFU + same-id prefetch on the same queue, round
2 sweep), stall is lower by 31-62% (30B) and 18-50% (235B).

Robustness of the default (decode, held-out):

- **FIFO link:** identical stall in every one of the 24 cells. At most 2
  transfers per layer never build a queue that demand fetches would wait
  behind. Without the cap, FIFO gains move by up to 0.9 points.
- **Roofline compute (least overlap):** gains shrink to 4.6-6.8% (30B) and
  3.7-4.7% (235B), still positive everywhere.
- **Cost:**
  - Transfers per token rise by 2-25% (e.g. 30B at 1,024 slots: 126.8 ->
    143.8).
  - Wasted transfers are 4-21 per token.
  - Demand hit rate drops by at most 0.16 points (30B) and 0.07 points
    (235B) and rises by up to 0.22 points; it stays within Belady.

## 5. Which lever dominates?

Held-out decode, nominal timing. "Eviction" is LRU -> LCNU without prefetch;
"prefetch" is the default's kNN prefetch added to LCNU (and, for
comparison, to LRU).

| Model | Slots (x WS) | Eviction: LRU -> LCNU | Prefetch on LCNU | Prefetch on LRU |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-30B-A3B | 128 (0.33) | -21.8% | -14.6% | -12.9% |
| | 256 (0.67) | -34.6% | -16.1% | -12.9% |
| | 512 (1.33) | -10.3% | -18.8% | -17.3% |
| | 1024 (2.67) | -7.4% | -23.4% | -21.3% |
| | 2048 (5.3) | -11.3% | -30.8% | -29.2% |
| | 3072 (8.0) | -10.7% | -30.8% | -33.4% |
| Qwen3-235B | 256 (0.34) | -20.6% | -8.6% | -7.5% |
| | 512 (0.68) | -32.9% | -9.6% | -7.5% |
| | 1024 (1.36) | -15.7% | -11.3% | -9.9% |
| | 2048 (2.72) | -11.5% | -13.8% | -12.0% |
| | 4096 (5.4) | -16.0% | -15.6% | -15.0% |
| | 6016 (8.0) | -24.1% | -13.6% | -16.2% |

- **Below one token's working set, eviction is the lever.** The link is
  saturated and LCNU's choice of which layers to keep removes the most
  transfers.
- **Above it, a good predictor makes prefetch as important or more:** -19% to
  -31% on Qwen3-30B-A3B, whose link moves 1.14 experts per layer of compute.
  On Qwen3-235B the link moves only 0.67, so prefetch can hide less. It
  stays at or below eviction except at 2,048 slots.
- **The two compose.** The prefetch gain is nearly the same on LRU and on
  LCNU (round 2's separability), so the default takes the best of each.
- **Share of a perfect predictor's gain.** As in round 2, this is
  `(LCNU - default) / (LCNU - oracle bound)`: 74-82% below one token's
  working set, 54-69% at 1.3-2.7x and 17-40% at 5-8x. Round 2's pair reached
  33-47%, 21-27% and 4-9%. Headroom remains in large caches (30B at 3,072
  slots: 1.94 ms against the oracle's 0.15).

## 6. Engine knobs: one default

| Knob | Default | Notes |
| --- | --- | --- |
| `expert_eviction_policy` | `lcnu` | with decode-only routing statistics (prefill passes counted as weight 0); alpha 0.3, beta 0.5 |
| `expert_prefetch_policy` | `knn` | decode passes only; `coactivation` (round 2) as the low-memory fallback; `same_id` (today's `speculative_prefetch`) deprecated |
| `expert_prefetch_horizon` | 1 | |
| `expert_prefetch_budget` | 8 | candidates per layer, best first |
| `expert_prefetch_issue` | 2 | transfers issued per layer; keeps the link queue short (FIFO-safe) |
| `expert_prefetch_admission` | `probation` | prefetched experts charged to the cache budget, evicted first, dropped at the layer's gate if not routed |

kNN at runtime, per MoE layer:

- **Bank:** the last 8192 decode tokens' routing at the preceding 8 layers
  (8 x 128 bits = 128 B per token), 1 MB per layer: 48 MB for
  Qwen3-30B-A3B, 94 MB for Qwen3-235B, in host memory. It also keeps the
  next layer's 8 routed ids per token (16 B).
- **Query:** after the gate, sum the bank's bit-rows for the token's 64
  (layer, expert) features (8192-bit overlap counts), take the top 8 and
  count their votes over 128 experts. That is a few hundred thousand bit
  operations on the CPU, overlapped with the layer's 189-221 us of compute;
  one small matrix product on the GPU would also do.
- **Update:** write the token's 64 feature bits into one bank column per
  layer.

## 7. Limitations

- **Same simulator and timing model as round 2:** one serial link,
  non-preemptive transfers, constant per-layer compute, batch-1 greedy MMLU
  decode.
- **kNN learns online within a run** of 75 (tuning) or 50 (Qwen3-235B)
  requests; its bank never fills (8192 tokens per layer). Longer or more
  diverse workloads may need a larger bank or an eviction rule for it. Its
  accuracy depends on recurring routing trajectories, which MMLU's shared
  prompt format may favour.
- **Prefetch is decode-only in the default.** Prefill passes are fetched on
  demand; the oracle bound shows what prefill-time prefetch could add.
- **Single tuning model.** Parameters were selected on Qwen3-30B-A3B only;
  Qwen3-235B is a transfer test.
- **Options not adopted.** Bypass admission, prefill weights between 0 and
  1 and multi-visit co-activation were evaluated and did not beat the
  default; they stay available as options.

## 8. Reproduce

```bash
TD=/mnt/raid0nvme0/leyang/moeinf-traces
R="python -m benchmarks.cache_simulator.research.run_prefetch_real"
F30="$TD/belady_qwen3_30b/*.pt"; F235="$TD/belady_qwen3_235b/*.pt"
KNN="knn:horizon=1,context=8,neighbors=8,power=8,bank=8192"
L="logistic:horizon=1,lr=0.1"; H="hybrid:horizon=1,alpha=0.4,weights=1/1/16"

# Stage A (tuning half): predictor grid (78 specs; see the CSV for the list).
$R --files "$F30" --model qwen3-30b-a3b --dataset mmlu150-greedy64 --split tune \
   --mode predict --tag qwen3_30b_tune_predict_r3 --prefetchers \
   knn:horizon=4,context={1,2,3,4},neighbors={8,16,32,64,128},bank=8192,power=2 \
   knn:horizon=4,context={4,6,8},neighbors={2,4,8},bank=8192,power=2 \
   knn:horizon=4,context={8,12,16},neighbors={4,8},bank=8192,power={4,8} \
   logistic:horizon=4,lr={0.01,0.02,0.03,0.05,0.1,0.2,0.3,0.5} \
   coactivation:horizon=4,scoring=bayes,smoothing={0.25,1,4,16},depth={1,2,3},decay={1,0.5} \
   coactivation:horizon=4,depth={2,3,4},decay={1,0.5}

# Eviction (tuning half, prefill-union stream), then bypass on prefill misses.
$R --files "$F30" --model qwen3-30b-a3b --dataset mmlu150-greedy64 --split tune \
   --prefill union --evictors lcnu lcnu_p lcnu_p:prefill_weight={0.1,0.25,0.5} \
   lcnu_b w_tinylfu lfu --prefetchers none \
   --capacities 128 256 512 1024 2048 3072 --tag qwen3_30b_tune_union_evict
$R --files "$F30" --model qwen3-30b-a3b --dataset mmlu150-greedy64 --split tune \
   --prefill union --evictors lcnu:bypass=prefill lcnu_p:bypass=prefill \
   --prefetchers none --capacities 128 256 512 1024 2048 3072 \
   --tag qwen3_30b_tune_union_evict_bypass

# Stage B (tuning half): one default prefetch configuration.
$R --files "$F30" --model qwen3-30b-a3b --dataset mmlu150-greedy64 --split tune \
   --evictors lcnu --prefetchers none $KNN,budget={4,8,16,32} \
   $KNN,budget=32,min_score={0.25,0.5} $L,budget={4,8,16,32} $H,budget={4,8,16,32} \
   --admissions probation do_no_harm probation:issue={2,4,8} \
   --capacities 128 256 512 1024 2048 3072 --tag qwen3_30b_tune_r3_default

# Held-out, decode (same for qwen3-235b-a22b-fp8 / $F235 / mmlu100-greedy64 /
# capacities 256 512 1024 2048 4096 6016 / tag qwen3_235b_held_r3).
$R --files "$F30" --model qwen3-30b-a3b --dataset mmlu150-greedy64 --split held \
   --evictors lcnu lru lfu --prefetchers none $KNN,budget=8 \
   $KNN,budget=32,min_score=0.25 $L,budget=8 oracle:horizon=1 oracle:horizon=4 \
   --admissions probation:issue=2 probation do_no_harm \
   --capacities 128 256 512 1024 2048 3072 --tag qwen3_30b_held_r3

# Held-out, prefill-union, decode-only kNN (tag *_held_union_r3d; the
# *_held_union_r3 runs use $KNN,budget=8 without decode_width and add
# oracle:horizon=4).
$R --files "$F30" --model qwen3-30b-a3b --dataset mmlu150-greedy64 --split held \
   --prefill union --evictors lcnu lcnu_p w_tinylfu lfu \
   --prefetchers none $KNN,budget=8,decode_width=8 --admissions probation:issue=2 \
   --capacities 128 256 512 1024 2048 3072 --tag qwen3_30b_held_union_r3d

# Held-out predictor accuracy (same for Qwen3-235B).
$R --files "$F30" --model qwen3-30b-a3b --dataset mmlu150-greedy64 --split held \
   --mode predict --tag qwen3_30b_held_predict_r3 --prefetchers \
   knn:horizon=4,context=8,neighbors=8,power=8,bank=8192,budget={8,2,16,32} \
   logistic:horizon=4,lr=0.1,budget={8,2,16,32} \
   coactivation:horizon=4,scoring=bayes,smoothing=0.25,depth=3,decay=0.5

# Default-policy regret over all held-out sweeps (rounds 2 and 3); writes
# results_default_policy_analysis.md byte-for-byte.
D=benchmarks/cache_simulator/research
python -m benchmarks.cache_simulator.research.default_policy_analysis \
   --csv $D/results_prefetch_qwen3_{30b,235b}_held{,_r3,_union,_union_r3,_union_r3d}.csv \
   --evictor lcnu_p lcnu \
   --prefetcher "$KNN,budget=8,decode_width=8" "$KNN,budget=8" \
   --admission probation:issue=2 --out $D/results_default_policy_analysis.md
```

Runtimes on the shared host with 28 workers: stage A 6 min, stage B 17 min,
held-out decode 18 min (30B) and 26 min (235B), union 3-10 min.

## References

- L. Xue, Y. Fu, Z. Lu, L. Mai, M. Marina. MoE-Infinity: Efficient MoE
  Inference on Personal Machines with Sparsity-Aware Expert Cache.
  arXiv:2401.14361.
- K. Fang et al. Klotski: Efficient Mixture-of-Expert Inference via
  Expert-Aware Multi-Batch Pipeline. arXiv:2502.06888.
- L. A. Belady. A study of replacement algorithms for a virtual-storage
  computer. IBM Systems Journal 5(2), 1966.
- Round 2 study and its references:
  [`prefetch_eviction_study.md`](./prefetch_eviction_study.md); survey:
  [`prefetch_survey.md`](./prefetch_survey.md).
