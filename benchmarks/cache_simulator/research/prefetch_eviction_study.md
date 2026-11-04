# Round 2: prefetch and eviction as separate, composable policies

Real-trace study for #289, round 2 (plan:
[`../auto_research_prefetch_eviction.md`](../auto_research_prefetch_eviction.md)).
Round 1 ([`decode_policy_study.md`](./decode_policy_study.md)) found LCNU to
be the best online evictor on batch-1 decode and showed that routing
prediction is a prefetch signal, not an eviction signal. This round
separates the two decisions. A prefetcher x evictor x admission-mode
harness ([`../prefetchers.py`](../prefetchers.py)) is replayed against a
transfer-timing model ([`timing_model.py`](./timing_model.py)) on the
held-out halves of the real Qwen3 captures. The prefetcher literature and
caching theory behind it are in [`prefetch_survey.md`](./prefetch_survey.md).

> **Superseded in part by round 3**
> ([`default_policy_study.md`](./default_policy_study.md)): a kNN
> routing-trajectory predictor replaces co-activation, and one default policy
> replaces the per-regime recommendation of item 5 and section 9. The
> separability and admission findings below still hold.

> **Summary**
>
> 1. **Yes, keep prefetch and eviction as two policies, connected by an
>    admission rule.** On held-out requests, choosing the evictor alone (no
>    prefetch) and the prefetcher alone (on LRU) comes within 1.6% of the
>    jointly optimal pair on Qwen3-30B-A3B and within 1.0% on Qwen3-235B, at
>    every capacity of the decode stream; on the prefill-union stream the
>    worst case is 3.4% (Qwen3-30B-A3B at half the key space). The evictor x
>    prefetcher interaction explains 1-6% of the variance of log-stall.
>    Prefetch never changes the best evictor below half the key space on
>    decode: LCNU stays best with every prefetcher.
> 2. **The admission rule is where the two policies meet.** Naive
>    same-queue admission is the worst mode. LCNU and LFU rank a freshly
>    prefetched expert near the eviction end, so with LCNU and the hybrid
>    prefetcher it covers 1.2% of accesses where staged admission covers
>    9.7%. It also raises the interaction share to 24%. Staging prefetched
>    experts outside the evictor fixes this. `probation` is best for the
>    recommended pair. `do_no_harm` (Cao et al.'s rule 3 with LCNU's
>    next-use estimate) is the most robust over all 210 pairs: it is the best
>    mode for the most pairs, averages 1.3-1.5% from each pair's best mode,
>    and has the fewest harmful combinations and wasted transfers. Evictors
>    with internal partitions (W-TinyLFU, S3-FIFO) need a fixed `buffer`:
>    shared staging bypasses their admission filter.
> 3. **Cross-layer co-activation is the online predictor to use; the
>    engine's current one is at chance.** It is learned from routing alone
>    and predicts the next layer's experts with recall@8 of 0.61 (hybrid) and
>    0.59 (co-activation) on Qwen3-30B-A3B, and 0.56 / 0.55 on Qwen3-235B.
>    Previous-token reuse and per-layer frequency reach 0.39-0.44. The
>    engine's `speculative_prefetch` predicts the same expert ids at the next
>    layer: precision 0.055-0.059, where chance is 0.0625. With today's
>    LFU-like eviction it adds 5-27% stall.
> 4. **Prefetch can only hide the compute it overlaps with.** With one serial
>    PCIe link, a token's 9.1 ms (Qwen3-30B-A3B) or 20.8 ms (Qwen3-235B) of
>    compute is all a prefetcher can hide. Below the working set eviction
>    dominates: LCNU removes 21-35% of LRU's stall where the best prefetcher
>    removes 4-8%. At 5-8x the working set a perfect predictor would remove
>    54-94% of the remaining stall. Online prefetchers capture 33-47% of the
>    oracle's gain below one token's working set, 21-27% at 1.3-2.7x and only
>    4-9% beyond 5x: the experts that still miss in a large cache are the
>    ones routing history ranks low.
> 5. **Recommended defaults (both models, decode).** LCNU + co-activation
>    prefetch (hybrid scoring), one layer ahead, admitted on `probation`.
>    Budget per layer: 4 experts below ~1.5x the working set, 8 up to ~1/3
>    of the key space, 16 above. This cuts stall per token by 3-12% against
>    LCNU alone, by 10-40% against LFU (today's demand path), and by 14-48%
>    against LFU with today's same-id prefetch. Engine knobs: section 9.

Every number below comes from a committed command (section 11) on the real
SGLang 0.5.20 `per_token` captures (`capture_report.md`). Prefetcher and
admission parameters were selected on the first half of the Qwen3-30B-A3B
requests and reported on the second half. Qwen3-235B uses the same
parameters without retuning. Demand hit rate is asserted `<= Belady` in
every run. Simulation: `python 3.12.3, numpy 2.4.4, Linux x86_64`, 128-core
shared host, at most 28 workers.

## 1. Setup

### Traces and protocol

| Model | MoE layers x experts, top-k | Working set (layers x k) | Key space | Tuning half | Held-out half |
| --- | --- | ---: | ---: | --- | --- |
| Qwen3-30B-A3B | 48 x 128, top-8 | 384 | 6,144 | requests 0-74 (1,843,200 decode accesses) | requests 75-149 (1,843,200; 4,800 tokens) |
| Qwen3-235B-A22B-Instruct-2507-FP8 | 94 x 128, top-8 | 752 | 12,032 | not used (no retuning) | requests 50-99 (2,406,400; 3,200 tokens) |

The streams are the same as round 1. `decode` holds one access per routed
(token, layer, expert), with the cache persisting across requests.
`decode+prefill-union` adds each prefill pass as one access per distinct
expert per layer. Capacities run from 1/3 of the working set to half the key
space. The two models' capacities are at the same multiples of the working
set (and the same fractions of the key space), so tuned settings carry over
by fraction of the key space.

Selection was done in two stages, both on the tuning half:

1. **Predictor parameters**, by mean recall@8 over distances 1-4
   (`results_prefetch_qwen3_30b_tune_predict.csv`): EMA rate 0.4 for
   `topk_ema` (0.4238; 0.3: 0.4153, 0.5: 0.4174) and hybrid weights 1/1/16
   (0.5885).
2. **Horizon x budget** for every (prefetcher, admission, capacity): a grid
   of h in {1, 2, 4} and m in {2, 4, 8, 16, 32}, 3,972 runs with LRU and
   LCNU (`results_prefetch_qwen3_30b_tune_grid.csv`). The selection
   minimizes the mean of `stall / stall of the evictor alone` over the two
   evictors (`tuned_prefetch_qwen3_30b.json`). Every evictor then uses the
   selected (h, m).

### What composes

`PrefetchCachePolicy(evictor, prefetcher, admission)` wraps any evictor of
round 1. A prefetcher is consulted **once per layer visit, after the gate**,
and proposes up to `m` experts for each of the next `h` visits. At most `m`
prefetches are issued per target visit. Online prefetchers learn only from
completed visits; a unit test checks this by changing the future of a trace.

| Prefetcher | Prediction for layer `L+d` | State |
| --- | --- | --- |
| `prev_token` | experts routed at `L+d` on its previous visit (previous token) | last routed set per layer |
| `topk_ema` | top-`m` of a per-layer EMA of "routed on this visit" (rate 0.4) | E floats per layer |
| `coactivation` | top-`m` of `mean_{e in S_L} P(e' at L+d | e at L)`, counts learned online over the layer cycle (EAM-style) | E x E counts per (layer, d) |
| `hybrid` | `1 x prev_token + 1 x ema + 16 x coactivation` scores | the three above |
| `same_id` | the same expert ids as layer `L`'s top experts, at `L+1` (MoE-Infinity's `speculative_prefetch`, `moe_infinity/memory/expert_prefetcher.py`) | none |
| `oracle` (bound) | the true routing of the next `h` visits | the trace |

| Admission | Where prefetched experts live | Under demand pressure | Source |
| --- | --- | --- | --- |
| `same_queue` | inside the evictor, ranked by its own rule | the evictor's rule | today's simulator / engine |
| `probation` | staging area sharing the capacity | staged experts go first (farthest target first) | PACMan-M, Mixtral-Offloading |
| `pin` | staging area sharing the capacity (at most half) | the evictor's victim goes first | HOBBIT's mask |
| `buffer` | separate buffer of `h x min(m, 16)` slots carved out of the capacity | never touched by demand misses | ExpertFlow-style buffer |
| `do_no_harm` | staging area sharing the capacity | the larger expected next use goes first | Cao et al., SIGMETRICS'95, rule 3 |

In every staging mode a staged expert is *use it or lose it*. It is dropped
at the gate of its target layer if that layer does not route it, and
promoted into the evictor on its first use otherwise. A prefetch's first use counts as a covered **miss**, never a
demand hit, so the demand hit rate stays bounded by Belady.

`do_no_harm` uses LCNU's model for both sides. A resident's expected next use
is LCNU's score `d(L) + layers (1 - p) / p`. A candidate at distance `d`
gets the same form with its first-visit probability replaced by the
predictor's measured precision `q` at that distance and rank:
`d + (1 - q) layers / p`. The prefetch is admitted into a full cache only if
the victim's estimate is larger.

`PrefetchCachePolicy(evictor, None)` and the composed demand path, with a
never-predicting prefetcher under every shared admission mode, reproduce
each of the 7 evictors' hit counts **bit for bit** on the full Qwen3-30B-A3B
decode trace (3,686,400 accesses) at 256 and 1,024 slots
([`results_prefetch_qwen3_30b_identity.md`](./results_prefetch_qwen3_30b_identity.md)).

### Timing model

The model has one serial host-to-GPU link and replays the per-visit event
log of each run:

- A layer visit starts at `T_v`. Its demand misses queue at `T_v` and
  compute waits for all of them.
- Prefetches decided at visit `v` queue behind `v`'s misses. A later demand
  miss overtakes queued prefetches, which are served earliest target first
  (`nominal`). The alternative is one FIFO queue (`fifo`).
- Transfers are never preempted. A prefetch still queued at its use becomes
  a demand transfer. A queued prefetch whose expert was dropped or evicted
  is cancelled.
- `T_{v+1} = ready_v + layer compute`. Stall is reported per decode token.

| Parameter | Qwen3-30B-A3B | Qwen3-235B-A22B-FP8 | Source |
| --- | ---: | ---: | --- |
| Bytes per expert | 9,437,184 (BF16) | 18,874,368 (FP8) | `3 x hidden x moe_intermediate x bytes`, HF `config.json` |
| Link | 57.1 GB/s | 57.1 GB/s | measured, PCIe 5.0 x16, pinned H2D (`measure_link.py`: 166 / 330 us per expert, 57.07-57.39 GB/s) |
| Transfer per expert | 165.3 us | 330.6 us | |
| Layer compute, `nominal` / `fifo` | 189.1 us | 221.4 us | measured batch-1 GEMVs of one layer's attention projections + router + 8 experts, CUDA graph, RTX PRO 6000 Blackwell (`measure_layer.py`; FP8 scaled from BF16 by bytes) |
| Layer compute, `roofline` | 63.5 us | 124.6 us | active layer bytes / 1.792 TB/s |
| Compute per token (nominal) | 9.08 ms | 20.81 ms | the most stall prefetch can hide |
| Experts moved per layer of compute (nominal) | 1.14 | 0.67 | |

The **stall floor** is a bound no policy can beat: any policy with `C`
slots transfers at least Belady's misses, so stall is at least that link
time minus all the compute.

## 2. How well can routing be predicted online?

Held-out precision / recall of the predictions for the next visit
(`results_prefetch_qwen3_{30b,235b}_held_predict.csv`). With `m = 8` and 8
routed experts per visit, precision equals recall.

| Predictor | 30B recall@8, d=1 / d=4 | 30B precision@2 | 30B recall@16 / @32 | 235B recall@8, d=1 / d=4 | 235B precision@2 | 235B recall@16 / @32 |
| --- | --- | ---: | --- | --- | ---: | --- |
| `prev_token` | 0.438 / 0.438 | 0.551 | - | 0.387 / 0.387 | 0.494 | - |
| `topk_ema` | 0.440 / 0.440 | 0.652 | 0.597 / 0.764 | 0.401 / 0.401 | 0.600 | 0.568 / 0.754 |
| `coactivation` | 0.593 / 0.560 | 0.804 | 0.793 / 0.920 | 0.551 / 0.528 | 0.750 | 0.763 / 0.912 |
| `hybrid` | **0.607 / 0.576** | **0.827** | **0.808 / 0.929** | **0.563 / 0.541** | **0.771** | **0.776 / 0.920** |
| `same_id` (engine) | 0.058 / 0.064 | 0.055 | - | 0.059 / 0.061 | 0.057 | - |
| chance (any expert) | 0.0625 | 0.0625 | 0.125 / 0.25 | 0.0625 | 0.0625 | 0.125 / 0.25 |

- Knowing which experts the current token just used at layer `L` predicts
  layer `L+1` better than the previous token's routing of `L+1` does (0.59
  vs 0.44). Accuracy decays slowly with distance: 0.56 four layers ahead.
- Expert ids carry no meaning across layers. The engine's same-id rule is no
  better than a random pick.
- 32 candidates cover 92% of a layer's routing, at precision 0.23. Section 3
  shows that in large caches even this captures little of the oracle's
  gain.

## 3. What prefetch can buy at all

Stall per decode token, held-out half, nominal timing. The evictors are shown
alone; the recommended pair is LCNU + hybrid on probation with the tuned
(h, m); the bound is the best oracle prefetcher with LCNU.

### Qwen3-30B-A3B (decode, `results_prefetch_qwen3_30b_held.md`)

| Slots | x WS | Belady hit | Floor | LFU | LRU | LCNU | LCNU + hybrid (h, m) | vs LCNU | Oracle bound | vs LCNU | Share of oracle gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 128 | 0.33 | 0.3079 | 34.85 | 55.20 | 63.47 | 49.64 | 45.62 (1, 4) | -8.1% | 41.09 | -17.2% | 47% |
| 256 | 0.67 | 0.5052 | 22.33 | 49.18 | 63.47 | 41.50 | 38.22 (1, 4) | -7.9% | 32.63 | -21.4% | 37% |
| 512 | 1.33 | 0.6718 | 11.76 | 40.76 | 35.56 | 31.89 | 29.55 (1, 4) | -7.3% | 22.90 | -28.2% | 26% |
| 1024 | 2.67 | 0.8158 | 2.61 | 28.64 | 22.65 | 20.96 | 18.52 (1, 8) | -11.6% | 11.98 | -42.8% | 27% |
| 2048 | 5.3 | 0.9379 | 0 | 12.53 | 9.58 | 8.49 | 7.93 (1, 8) | -6.6% | 1.92 | -77.4% | 9% |
| 3072 | 8.0 | 0.9818 | 0 | 4.36 | 3.14 | 2.80 | 2.62 (1, 16) | -6.4% | 0.16 | -94.3% | 7% |

### Qwen3-235B-A22B-FP8 (decode, `results_prefetch_qwen3_235b_held.md`)

| Slots | x WS | Belady hit | Floor | LFU | LRU | LCNU | LCNU + hybrid (h, m) | vs LCNU | Oracle bound | vs LCNU | Share of oracle gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 256 | 0.34 | 0.3077 | 151.28 | 209.15 | 248.57 | 197.26 | 188.09 (1, 4) | -4.6% | 177.35 | -10.1% | 46% |
| 512 | 0.68 | 0.4921 | 105.44 | 184.46 | 248.57 | 166.67 | 159.94 (1, 4) | -4.0% | 146.12 | -12.3% | 33% |
| 1024 | 1.36 | 0.6609 | 63.48 | 149.87 | 152.02 | 128.21 | 123.83 (1, 4) | -3.4% | 107.04 | -16.5% | 21% |
| 2048 | 2.72 | 0.8109 | 26.20 | 103.21 | 94.32 | 83.45 | 78.73 (1, 8) | -5.7% | 62.22 | -25.4% | 22% |
| 4096 | 5.4 | 0.9334 | 0 | 44.99 | 40.33 | 33.87 | 32.91 (1, 8) | -2.8% | 15.69 | -53.7% | 5% |
| 6016 | 8.0 | 0.9765 | 0 | 16.10 | 15.48 | 11.75 | 11.40 (1, 16) | -3.0% | 2.47 | -79.0% | 4% |

Stall in ms per token; total time per token adds the compute (9.08 / 20.81
ms). "Share of oracle gain" is `(LCNU - pair) / (LCNU - oracle)`.

- **Below one token's working set the link is saturated.** At 128 slots on
  Qwen3-30B-A3B, LCNU alone moves 300 experts per token, 49.6 ms of link
  time against 9.1 ms of compute. A perfect prefetcher (41.1 ms) only gets
  close to `stall - compute` (40.6 ms). Eviction is the lever here: LCNU vs
  LRU is -22% at 128 slots and -35% at 256 slots.
- **Above it, prefetch is the lever, but online predictors reach little of
  it.** At 8x the working set the oracle removes 79-94% of LCNU's stall.
  The best online pair removes 3-10%. The remaining misses fall on rarely
  routed experts; catching them takes 16-32 candidates per layer, most of
  them wrong. A gate-lookahead predictor, which needs hidden states
  (Pre-gated MoE, HOBBIT, ProMoE, AdapMoE), could fill this gap. Routing
  statistics cannot.
- **One layer ahead is enough** with this link. The tuned horizon is 1 for
  the recommended pair at every capacity; a few `pin` / `do_no_harm` cells
  at large caches pick 2, and the oracle bound up to 4. The link moves only
  0.7-1.1 experts per layer of compute, so deeper prefetch only queues
  transfers. With a FIFO link, deeper prefetch also hurts: on the tuning
  half, LCNU + hybrid (m = 8, pin) at 1,024 slots gives 18.77 ms at h = 1
  under both link orders, and 19.33 / 22.48 ms (priority / FIFO) at h = 4.

Best online pair overall at each capacity (any evictor, prefetcher and mode):
Qwen3-30B-A3B LCNU + hybrid on probation at 128-256 and 1,024; LCNU +
co-activation on probation at 512 (29.34) and 2,048 (7.79); LCNU +
co-activation on pin, h = 2, at 3,072 (2.53, -2.7% against S3-FIFO alone,
the best evictor alone there). Qwen3-235B: LCNU + hybrid or co-activation
(probation, or pin at 4,096-6,016) everywhere, -3.7% to -5.9%. Hybrid and
co-activation are within 2% of each other in every cell: hybrid is ahead
below the working set, co-activation slightly ahead above it.

## 4. Q1: are the two policies separable?

For each capacity and admission mode, `S[evictor][prefetcher]` holds the
stall over 7 evictors x {none + 5 online prefetchers} (5 evictors x {none +
4} on the prefill-union stream).

- **Independent choice**: the best evictor alone (no prefetch) plus the best
  prefetcher on LRU (how prefetchers were evaluated so far).
- **Sequential choice**: the best evictor alone, then the best prefetcher
  for it.
- **Decomposition**: two-way, `log S = mu + a_e + b_p + interaction`.

Full tables: `results_prefetch_*_analysis.md`.

Regret of the independent choice vs the joint optimum, best admission per
pair:

| Stream | Model | per capacity, smallest to largest | max |
| --- | --- | --- | ---: |
| decode | Qwen3-30B-A3B | 0.0, 0.0, +0.7, +0.4, 0.0, +1.6% | 1.6% |
| decode | Qwen3-235B | 0.0, 0.0, 0.0, +0.3, 0.0, +1.0% | 1.0% |
| prefill-union | Qwen3-30B-A3B | 0.0, 0.0, +0.7, +0.4, +0.2, +3.4% | 3.4% |
| prefill-union | Qwen3-235B | 0.0, 0.0, 0.0, +0.2, 0.0, 0.0% | 0.2% |

The sequential choice reaches the joint optimum exactly in 22 of the 24
cells; the other two are 30B decode at 3,072 (+1.6%) and 30B prefill-union
at 2,048 (+0.2%).

Share of log-stall variance (evictor / prefetcher / interaction) at the
smallest to the largest capacity:

| Admission | Qwen3-30B-A3B decode | Qwen3-235B decode |
| --- | --- | --- |
| best per pair | 83/15/2 ... 94/4/2; interaction 1-4% | 92/7/1 ... 68/26/5; interaction 1-6% |
| `probation` | interaction 1-7% | interaction 3-8% |
| `same_queue` | interaction 4-17% | interaction 0-24% |

Answers:

- **Separable: yes, through a staged admission rule.** The best evictor
  without prefetch is the best evictor with prefetch, and the best predictor
  on LRU is the best predictor on LCNU. The independent choice loses at most
  1.6% on decode and 3.4% with prefill.
- **Same-queue admission is what couples them.** Its interaction share is
  up to 17% (30B) and 24% (235B), because the evictor decides how long a
  prefetched expert survives (section 5).
- **Prefetch does not change the best evictor on decode.** LCNU is best at
  every capacity for every prefetcher on Qwen3-235B. On Qwen3-30B-A3B it is
  best everywhere except 3,072 slots, where S3-FIFO is best alone (2.60 vs
  2.80) but LCNU + co-activation is best with prefetch. Kendall tau of the
  evictor ranking with vs without prefetch is 0.6-1.0.
- **With prefill passes in the stream**, round 1's large-cache ranking
  returns with or without prefetch: W-TinyLFU on Qwen3-30B-A3B from 2,048
  slots, LFU on Qwen3-235B at 6,016.

## 5. Q3: admission of prefetched experts

All 210 online (evictor, prefetcher, capacity) triples per model, decode,
nominal timing (`results_prefetch_*_held_analysis.md`). "Best for" counts
ties (`pin` equals `probation` when h = 1).

| Admission | 30B: best for | 30B: mean vs best mode | 30B: worse than no prefetch | 30B: wasted xfers/tok | 235B: best for | 235B: mean vs best mode | 235B: worse than no prefetch | 235B: wasted xfers/tok |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `same_queue` | 43 | +7.7% | 41% | 27.6 | 31 | +6.9% | 53% | 39.3 |
| `probation` | 73 | +5.1% | 31% | 30.7 | 90 | +3.1% | 37% | 35.4 |
| `pin` | 72 | +5.2% | 34% | 31.0 | 97 | +3.1% | 37% | 35.8 |
| `buffer` | 34 | +3.7% | 38% | 28.2 | 28 | +2.4% | 39% | 32.8 |
| `do_no_harm` | **97** | **+1.5%** | **21%** | **18.6** | **101** | **+1.3%** | **35%** | **23.7** |

The "worse than no prefetch" column includes the weak predictors
(`same_id`; `prev_token` and `topk_ema` above the working set, where their
candidates are already cached). For the recommended pair (LCNU + hybrid),
`probation` beats `do_no_harm` by 0.1-1.9% in 10 of 12 decode cells and
loses by 0.8-0.9% in 2.

Why the modes differ (Qwen3-30B-A3B decode, hybrid prefetcher):

- **Same queue lets the evictor throw prefetches away.** LCNU scores a new
  expert by its routing history, and LFU by its count. Both rank a prefetched
  expert near the eviction end, and the next admission evicts it. At 1,024
  slots, prefetches cover 1.2% of LCNU's accesses (9.7% staged) and 3.3% of
  LFU's (14.4% staged). Stall is 20.32 vs 18.52 ms (LCNU) and 27.69 vs 25.16
  ms (LFU). With LRU the problem is the reverse: prefetched experts enter at
  MRU and push demand residents out. LRU's demand hit rate drops by 2.1
  points at 3,072 slots, which here the covered misses make up for.
- **Staging is evictor-agnostic for priority evictors** (LCNU, LRU, LFU,
  LIRS, per-layer LRU). The evictor only ever sees demand admissions.
  Staged experts are ranked by their target layer, not by an evictor that
  cannot value them.
- **Shared staging breaks partitioned evictors.** When staged experts take
  slots, W-TinyLFU's main region is no longer full, and its window victim
  enters without the frequency test. At 256 slots, W-TinyLFU's demand hit
  rate falls from 0.276 to 0.086 on probation, and stall rises from 45.95 to
  52.05 ms. A fixed `buffer` keeps the filter intact (0.273, 41.98 ms).
- **`do_no_harm` issues fewer prefetches** (up to 45% fewer than
  `probation` with the hybrid prefetcher at 256 and 1,024 slots) and admits
  only those it expects to beat the victim. It wastes 28-40% fewer
  transfers than the other modes on average. That makes it the safest
  mode for an evictor or predictor that was not tuned together. It guards
  cache contents, not the link, so a near-random predictor still costs
  transfers. With LCNU at 6,016 slots on Qwen3-235B, `same_id` adds 28% of
  stall on `probation` (15.05 vs 11.75 ms) and still 22% on `do_no_harm`
  (14.38 ms).

## 6. Q4: timing, and when prefetching hurts

- **Wasted transfers are cheap only when demand fetches have priority.**
  Under `fifo` the best pairs keep their gains at h = 1. The stall is
  identical at 128-2,048 slots on Qwen3-30B-A3B; at the largest capacities
  the gain shrinks from -2.7% to -1.0% (30B) and from -5.2% to -3.7%
  (235B). Deeper horizons lose: 22.48 vs 19.33 ms at h = 4 above. The
  engine's demand fetcher runs on its own stream and shares the link with
  prefetch copies without priority
  (`core/parallel/expert_dispatcher.cpp`). Neither scenario models that
  sharing exactly; `fifo` bounds its cost.
- **Less compute means less to hide.** At the bandwidth roofline (63.5 /
  124.6 us per layer) the best online gain shrinks to 0-1.6% (30B) and
  0.1-1.2% (235B). Prefetch pays off in proportion to the per-layer compute
  window.
- **When prefetching hurts** (held-out decode, nominal):
  - the same-id predictor: LFU + same-id on the same queue is +5% to +27%
    against LFU alone, worst at large caches;
  - same-queue admission with LCNU or LFU: transfers are wasted on experts
    the evictor drops before use;
  - shared staging with W-TinyLFU / S3-FIFO: their filters are bypassed;
  - `buffer` at large caches: carving slots out costs more than the
    prefetches return, and 16-22 of 35 combinations at the two largest
    capacities are worse than the evictor alone.
- **`do_no_harm` removes most of it.** It has the fewest harmful
  combinations (30B: 21% vs 31-41% for the other modes; 235B: 35% vs
  37-53%) and cuts wasted transfers by about a third. It does not remove
  harm that comes from the link (section 5).

## 7. Prefill-union stream

With every prefill pass fetched once per distinct expert and layer
(`results_prefetch_qwen3_{30b,235b}_held_union.md`, 5 evictors). The timing
model gives prefill visits the decode compute time, so stall per decode
token here includes prefill fetches that real prefill compute would partly
hide.

| Model | LCNU + hybrid (probation) vs LCNU alone, smallest to largest capacity | Best at the largest capacity |
| --- | --- | --- |
| Qwen3-30B-A3B | -7.1, -6.7, -6.0, -8.5, -3.1, -2.5% | W-TinyLFU alone 4.65 ms; W-TinyLFU + prev_token (probation) 4.51 ms |
| Qwen3-235B | -4.0, -3.3, -2.6, -4.0, -1.4, -1.5% | LFU alone 23.96 ms; LFU + hybrid (pin) 23.35 ms |

Separability holds (regret at most 3.4%), and the admission ranking is
unchanged: `do_no_harm` is +1.2-1.5% from the per-pair best mode with the
fewest wasted transfers. The large-cache prefill effect of round 1 remains.
A prefill pass is a scan that frequency filters keep out of the cache. For
prefill-heavy serving with a cache above ~1/3 of the key space, W-TinyLFU or
LFU with a light prefetcher (previous-token or hybrid, on `probation` or
`pin`) is the better pair. Prefetch adds only 2.5-3% there.

## 8. Recommended pair and admission per regime

Decode-dominated serving, both models (capacities as multiples of one
token's working set `WS = layers x top_k`):

| Regime | Evictor | Prefetcher | h | m | Admission | Stall vs LCNU alone (30B / 235B) | vs today's LFU path (30B / 235B) |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| below WS (<= 0.7x) | LCNU | hybrid (co-activation) | 1 | 4 | `probation` | -8% / -4 to -5% | -17 to -22% / -10 to -13% |
| 1-3x WS | LCNU | hybrid or co-activation | 1 | 4 (1.3x), 8 (2.7x) | `probation` | -7 to -12% / -3 to -6% | -28 to -35% / -17 to -24% |
| 5x WS to 1/2 key space | LCNU | co-activation or hybrid | 1-2 | 8 (5x), 16 (8x) | `probation` or `pin` | -6 to -10% / -3 to -5% | -37 to -42% / -27 to -31% |
| any, evictor not LCNU or predictor untuned | (as configured) | (as configured) | 1 | as above | `do_no_harm` | within 1.3-1.5% of the best mode on average | |
| W-TinyLFU / S3-FIFO (prefill-heavy, large cache) | W-TinyLFU | hybrid or prev_token | 1 | as above | `buffer` / `probation` | | |

Against LFU with today's same-id prefetch on the same queue, the
recommended pair cuts stall by 26-48% (30B) and 14-44% (235B).

## 9. Q5: engine knobs

**Recommendation: expose eviction and prefetch as two policies, plus the
admission rule that joins them.** The research shows the two choices can be
made independently once prefetched experts are staged outside the evictor.
None of these knobs exists today: `expert_eviction_policy` and
`expert_prefetch_policy` appear only in the research prompts. The engine
currently has:

- demand eviction by lowest `incache_visit_count`
  (`ExpertDispatcher::FindExpertEvict`, LFU-like);
- `speculative_prefetch` (off by default), the same-id rule with top-2
  experts one layer ahead;
- prefetched experts that are not charged to the cache budget
  (`cached_experts_` holds demand fetches only).

| Knob | Values | Default | Notes |
| --- | --- | --- | --- |
| `expert_eviction_policy` | `lcnu`, `lru`, `lfu`, `w_tinylfu` | `lcnu` | `lfu` reproduces today's demand path; `w_tinylfu` for prefill-heavy serving with a cache above ~1/3 of the key space (round 1) |
| `expert_prefetch_policy` | `off`, `coactivation` (hybrid scoring), `prev_token`, `same_id` | `coactivation` | `same_id` = today's `speculative_prefetch`; recommend deprecating it (chance-level precision) |
| `expert_prefetch_horizon` | int | 1 | deeper prefetch only queues transfers on one PCIe link |
| `expert_prefetch_budget` | int or `auto` | `auto`: 4 below 1.5x WS, 8 up to 1/3 of the key space, 16 above | per target layer |
| `expert_prefetch_admission` | `probation`, `do_no_harm`, `buffer`, `pin`, `same_queue` | `probation` with LCNU / LRU / LFU; `buffer` with W-TinyLFU / S3-FIFO; `do_no_harm` for untuned pairs | `same_queue` only for legacy behaviour |

What the engine needs at runtime:

| Component | Signals | State | Cost |
| --- | --- | --- | --- |
| LCNU eviction | the layer being executed; routed expert ids | per expert: EMA, count, first and last visit; per-layer visit counters | O(layers) victim search per eviction, with per-layer candidates cached |
| co-activation prefetch (hybrid) | the routed ids of the layer that just gated | for h = 1: an E x E count matrix + E counts per layer (48 x 128 x 128 x 4 B = 3.1 MB for Qwen3-30B-A3B, 6.2 MB for Qwen3-235B); previous routed set and E-float EMA per layer | per layer: 64 counter increments, a sum of 8 rows of 128, a top-m; microseconds on CPU, overlapped with the layer |
| `probation` / `pin` | the target layer's routed ids at its gate | a staging set (expert -> target layer, at most h x m entries) charged to the cache budget | drop unrouted staged experts at the gate; promote routed ones; staged experts evicted first under demand pressure |
| `do_no_harm` | LCNU's next-use estimate; whether each prediction was routed | (h + 1) x m precision counters | one comparison per admission |
| link scheduling (not a knob) | demand vs prefetch | a prefetch queue ordered by target layer | demand fetches must overtake queued prefetches; drop queued prefetches whose layer has run or whose slot was reclaimed (`StartExec` already drops earlier-layer tasks) |

Two engine changes matter even without new knobs:

1. Put demand fetches ahead of prefetch copies on the link.
2. Charge prefetched experts to the cache budget, so that admission is a
   real decision.

## 10. Limitations

- **Timing model.** It has one serial link, non-preemptive per-expert
  transfers and constant per-layer compute. It has no CPU-side expert
  compute, multi-GPU or batching. The per-layer compute is a measured
  GEMV-only time of one synthetic layer (FP8 scaled by bytes); a real engine
  adds attention over the KV cache and routing, and fuses experts. Section
  6 shows how the gains scale with this window.
- **Predictors are routing-only.** Gate-lookahead predictors need hidden
  states and appear only through the oracle bound. Their headroom is largest
  where routing statistics are weakest, in large caches.
- **Tuning.** Parameters were tuned with LRU and LCNU only; the other
  evictors use the same (h, m).
- **Use-it-or-lose-it staging** drops unused staged experts at their target
  layer. Keeping them as lowest-priority residents, as PACMan does, may help
  in large caches. On the tuning half, same-queue admission with LRU beat
  staging at 3,072 slots (3.02 vs 3.29 ms).
- **`do_no_harm` guards the cache, not the link.** A bandwidth-aware rule
  (TIP's cost-benefit, Patterson et al. SOSP'95) is the natural extension.
- **Workload.** Batch-1 greedy decode of 64 tokens on MMLU prompts, as in
  round 1. 100 (Qwen3-235B) and 150 (Qwen3-30B-A3B) requests.

## 11. Reproduce

```bash
TD=/mnt/raid0nvme0/leyang/moeinf-traces
R="python -m benchmarks.cache_simulator.research.run_prefetch_real"
F30="$TD/belady_qwen3_30b/*.pt"; F235="$TD/belady_qwen3_235b/*.pt"
EV="lru lfu lirs s3fifo w_tinylfu layer_lru lcnu"
ADM="same_queue probation pin buffer do_no_harm"
TUNED=benchmarks/cache_simulator/research/tuned_prefetch_qwen3_30b.json

# Link and per-layer compute on the capture host (one idle GPU).
CUDA_VISIBLE_DEVICES=2 python -m benchmarks.cache_simulator.research.measure_link
CUDA_VISIBLE_DEVICES=2 python -m benchmarks.cache_simulator.research.measure_layer

# Composition identity on the full Qwen3-30B-A3B decode trace.
$R --files "$F30" --model qwen3-30b-a3b --dataset mmlu150-greedy64 --split all \
   --mode identity --evictors $EV --capacities 256 1024 --tag qwen3_30b_identity

# Stage 0 (tuning half): predictor parameters by recall (bash brace expansion).
$R --files "$F30" --model qwen3-30b-a3b --dataset mmlu150-greedy64 --split tune \
   --mode predict --tag qwen3_30b_tune_predict --prefetchers prev_token:horizon=4 \
   same_id:horizon=4 coactivation:horizon=4 \
   topk_ema:horizon=4,alpha={0.1,0.2,0.3,0.35,0.4,0.45,0.5,0.7} \
   hybrid:horizon=4,alpha=0.4,weights={1/1/1,1/1/4,1/0.5/4,0.5/1/4,0/1/4,1/0/4,1/1/8,0.5/1/8,1/1/16,1/1/24,1/2/16,2/1/16} \
   hybrid:horizon=4,alpha=0.3,weights=1/1/16

# Stage 1 (tuning half): horizon x budget per (prefetcher, admission, capacity).
$R --files "$F30" --model qwen3-30b-a3b --dataset mmlu150-greedy64 --split tune \
   --mode tune --evictors lru lcnu --admissions $ADM \
   --prefetchers prev_token topk_ema:alpha=0.4 coactivation \
     hybrid:alpha=0.4,weights=1/1/16 same_id oracle \
   --horizons 1 2 4 --budgets 2 4 8 16 32 \
   --capacities 128 256 512 1024 2048 3072 --tag qwen3_30b_tune_grid

# Held-out sweeps, decode.
$R --files "$F30" --model qwen3-30b-a3b --dataset mmlu150-greedy64 --split held \
   --evictors $EV --admissions $ADM --tuned $TUNED \
   --prefetchers none prev_token topk_ema coactivation hybrid same_id \
     oracle:horizon=1 oracle \
   --capacities 128 256 512 1024 2048 3072 --tag qwen3_30b_held
$R --files "$F235" --model qwen3-235b-a22b-fp8 --dataset mmlu100-greedy64 \
   --split held --evictors $EV --admissions $ADM --tuned $TUNED \
   --prefetchers none prev_token topk_ema coactivation hybrid same_id \
     oracle:horizon=1 oracle \
   --capacities 256 512 1024 2048 4096 6016 --tag qwen3_235b_held

# Held-out sweeps, prefill-union (same for qwen3-235b-a22b-fp8 with $F235,
# mmlu100-greedy64, capacities 256 ... 6016, tag qwen3_235b_held_union).
$R --files "$F30" --model qwen3-30b-a3b --dataset mmlu150-greedy64 --split held \
   --prefill union --evictors lru lfu s3fifo w_tinylfu lcnu --admissions $ADM \
   --tuned $TUNED --prefetchers none prev_token coactivation hybrid same_id oracle \
   --capacities 128 256 512 1024 2048 3072 --tag qwen3_30b_held_union

# Held-out predictor accuracy (same for qwen3-235b-a22b-fp8 / $F235).
$R --files "$F30" --model qwen3-30b-a3b --dataset mmlu150-greedy64 --split held \
   --mode predict --tag qwen3_30b_held_predict --prefetchers \
   prev_token:horizon=4,budget={8,2} same_id:horizon=4,budget={8,2} \
   topk_ema:horizon=4,budget={8,2,16,32},alpha=0.4 \
   coactivation:horizon=4,budget={8,2,16,32} \
   hybrid:horizon=4,budget={8,2,16,32},alpha=0.4,weights=1/1/16

# Analysis tables.
python -m benchmarks.cache_simulator.research.prefetch_analysis \
   --csv benchmarks/cache_simulator/research/results_prefetch_qwen3_30b_held.csv
```

Runtimes on the shared 128-core host with 28 workers: tuning grid 69 min,
Qwen3-30B-A3B held-out 23 min, Qwen3-235B held-out 66 min, union streams 17
and 39 min.

## References

- P. Cao, E. W. Felten, A. R. Karlin, K. Li. A Study of Integrated
  Prefetching and Caching Strategies. SIGMETRICS 1995.
- R. H. Patterson, G. A. Gibson, E. Ginting, D. Stodolsky, J. Zelenka.
  Informed Prefetching and Caching. SOSP 1995.
- C.-J. Wu, A. Jaleel, M. Martonosi, S. C. Steely Jr., J. Emer. PACMan:
  Prefetch-Aware Cache Management for High Performance Caching. MICRO 2011.
- L. Xue, Y. Fu, Z. Lu, L. Mai, M. Marina. MoE-Infinity: Efficient MoE
  Inference on Personal Machines with Sparsity-Aware Expert Cache.
  arXiv:2401.14361.
- P. Tang, J. Liu, X. Hou, Y. Pu, J. Wang, P.-A. Heng, C. Li, M. Guo.
  HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference.
  arXiv:2411.01433.
- A. Eliseev, D. Mazur. Fast Inference of Mixture-of-Experts Language Models
  with Offloading. arXiv:2312.17238.
- Full prefetch bibliography: [`prefetch_survey.md`](./prefetch_survey.md).
