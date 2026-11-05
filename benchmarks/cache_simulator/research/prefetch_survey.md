# MoE expert prefetchers and prefetch-aware eviction: survey

Round-2 Task D of the prefetch/eviction research plan
(`../auto_research_prefetch_eviction.md`,
`../prompts/04_research_prefetch_eviction.md`). Round 1
([`policy_survey.md`](./policy_survey.md)) covered these systems from the
**eviction** angle; this survey covers the **prefetch** angle — what each
system predicts, from which signal, how far ahead, under what budget, where
prefetched experts are admitted, and whether eviction knows about prefetches —
plus the classic integrated prefetching-and-caching theory, and a shortlist of
prefetchers and admission modes implementable in the simulator for prompt E.
**No simulator code is added here.**

Context from round 1 ([`decode_policy_study.md`](./decode_policy_study.md),
[`recommendations.md`](./recommendations.md)): on real batch-1 decode traces
LCNU is the best online evictor, and giving it perfect routing of the next 8
layers adds at most 0.7 points of demand hit rate — cross-layer routing
prediction is a **prefetch** signal, not an eviction signal. The question for
round 2 is which prefetcher, under which admission rule, pairs best with which
evictor.

Every identifier below was re-verified against the arXiv API on 2026-11-02;
claims are quoted or paraphrased from the papers' arXiv HTML/PDF full texts
(see Verification notes). Exact numbers that appear only inside math images in
the arXiv HTML (and so could not be extracted verbatim) are flagged rather
than guessed.

## What an online prefetcher may see (simulator contract)

Per the shared contract, an **online** prefetcher sees only past routing and
the current token's routing up to the current layer, and returns
`(layer, expert, score)` candidates for the next `h` layer visits under a
per-layer budget. The trace records routed `(layer, expert)` sets only — no
hidden states, no gate logits, no token IDs. Hence:

- **Rebuildable online from routing traces**: predictors that are statistics
  of past routing — previous-token persistence, per-layer frequency,
  cross-layer co-activation tables (Klotski), request-level activation-matrix
  matching (MoE-Infinity's EAM/EAMC), hot-expert profiles.
- **Oracle-only in the simulator**: predictors that consume hidden states or
  gate logits (Mixtral-Offloading's skip-gate, HOBBIT's stacked gates,
  AdapMoE's gate reuse, ProMoE's learned MLP, SpecMD's score-percentile
  budget), token/prompt content (SiDA's LSTM hash, ExpertFlow's T5 RPP, fMoE's
  semantic search), or a modified model (Pre-gated MoE). These are realized
  as trace oracles (perfect prediction `h` ahead) and labelled upper bounds.

## 1. Per-system prefetch mechanisms

### Summary table

| System | Predictor input | Horizon | Budget | Admission of prefetches | Eviction prefetch-aware? | Reported accuracy (metric) | Online from traces? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Mixtral-Offloading (arXiv:2312.17238) | next layer's gate applied to current hidden state | 1 (2, 10 measured) | fixed number of most-likely experts | staged separately; replaces an LRU entry only if actually used | yes (probation-style) | "expert recall" vs experts loaded, Fig. 2; falls with distance | no (oracle) |
| MoE-Infinity (arXiv:2401.14361) | iteration-level EAM matched against stored request-level EAMs (routing counts only) | next layer(s) of current token | top-likelihood entries of matched EAM rows | same cache; prediction likelihood also drives eviction | yes (shared pEAM priority + layer decay) | TPOT 3.1-16.7x vs baselines; no standalone recall@k | **yes** |
| Pre-gated MoE (arXiv:2308.12066; ISCA'24) | relocated (trained) gate at block `l` selects block `l+1` | 1, exact | exactly the activated set | double-buffer: current + next block resident | trivially (drop finished layer) | exact by construction; model quality preserved after fine-tuning | no (model change; oracle) |
| ProMoE (arXiv:2410.22134; EuroSys'25) | per-layer MLP on the layer's gating input (offline-trained) | layer-wise +1; stride adds +1 (≈2) | same count as the model's top-k per layer | per-layer LRU cache; prefetch touches refresh recency; 2-priority queue | partially (touch-refresh protects predicted experts) | 84.7% avg accuracy (−5% under stride); skip-based 66.9%, token-based 58.3% | no (oracle); its *stride schedule* is positional |
| HOBBIT (arXiv:2411.01433) | all remaining gates stacked, applied to current gating input | adaptive 1..n; recommend 1-3 | predicted top-k per layer, mixed precision | same cache; predicted experts **masked** from eviction | **yes** (mask/pin + LRU/LFU/LHU/FLD weighted priority) | next-1 top-1 96%, next-2/3 ≈90% (Mixtral-8x7B, Fig. 7) | no (oracle) |
| AdapMoE (arXiv:2408.10284; ICCAD'24) | next layers' own gates on current activations; trained aux gate for layer 0 | 1-3 (extends while predicted set already cached) | adaptive top-k (sensitivity-gated) | LRU cache; DP allocates per-layer cache size from profiled prefetch accuracy | capacity-level only | ≈90% per-layer prefetch accuracy (Fig. 9b) | no (oracle) |
| SiDA-MoE (arXiv:2310.18859; MLSys'24) | offline-trained LSTM+sparse-attention hash on token embeddings | whole forward pass, one batch ahead | exactly predicted experts per layer | load predicted set, offload rest; FIFO when budget exhausted | yes (predicted-inactive are evicted) | top-3 "hash hits rate" (Table 5; values in math, not extracted); >90% per Klotski's citation | no (oracle) |
| ExpertFlow (arXiv:2410.17954; DAC'26) | T5-style encoder-decoder on the input sequence, all layers at once | entire forward pass | predicted experts; slots allocated per layer by predicted demand | PLEC slot plan + prioritized correction swaps on mispredictions | yes (slot plan from predictions; freed slots reused) | up to 95% routing-path accuracy; ECE hit ratio 91.96%, up to +61.15 points vs LRU | no (oracle) |
| fMoE / FineMoE (arXiv:2502.05370) | expert-map store searched by semantic (prompt) + trajectory (gate distributions) similarity | prefetch distance `d` layers (swept) | dynamic threshold: smallest set whose summed probability ≥ θ(score), ≥ top-k | same cache; prefetch priority = probability / layer distance | **yes** (eviction priority = LFU count combined with map probability) | −47% latency, +39% expert hit rate vs SOTA | partially (trajectory part degrades to EAM matching on routed sets; semantic part no) |
| Klotski (arXiv:2502.06888) | cross-layer expert correlation table from warm-up, updated online (routing only) | next layer (during attention) | top-K hot experts, K = top-k | prefetch hot experts; others fetched at gate time; computation reordered around them | schedule-driven (offload immediately after use) | hot-expert prediction 58.89% avg; 100% of prefetched hot experts used (multi-batch) | **yes** |
| SpecMD (arXiv:2602.03921) | next layer's gate on current activations (framework); budgets: top-k x overfetch or score percentile | 1 | top-k x overfetch factor (e.g. 1.5x), or all experts ≥ 80th score percentile | async watchdog enqueues when space free; prefetched marked "current" | **yes** (Least-Stale: stale-first eviction; current/prefetched protected) | 88-92% hit rate at 5% capacity; collision misses −85x vs LRU; TTFT −10.7-34.7% (OLMoE) | eviction yes; score budget no (oracle) |
| MoE-Lightning (arXiv:2411.11217) | none (positional) | next layer | whole layer (paged weights) | pipelined paged streaming (CGOPipe) | n/a (schedule-driven) | up to 10.3x throughput vs offloading baselines | yes (positional prefetch-all) |
| SwapMoE (arXiv:2308.15030; ACL'24) | expert importance profile (gating-derived) | asynchronous, amortized over tokens | top-importance "Virtual Experts" set | masked gating routes **only** to resident experts (changes outputs) | n/a (no speculative prefetch) | e.g. Switch: 14.2→4.7 GiB, −50% latency, −0.041 Rouge-2 | no (changes routing; out of scope) |

DeepSpeed-Inference (layer prefetch-all) and SE-MoE (ring streaming) are
covered in round 1 (`policy_survey.md`) and unchanged: positional
prefetch-all, no routing prediction, modelable as `StreamFetch`.

### 1.1 Mixtral-Offloading — Eliseev & Mazur (arXiv:2312.17238)

- **Predictor**: apply the **next layer's gating function to the current
  layer's hidden states** ("the same hidden states that are used by previous
  MoE layer's gating function"); works because residual layers make
  consecutive hidden states similar (Sec. 3.2).
- **Horizon**: speculative loading is triggered right after the current
  layer's experts finish loading, for the next layer; the evaluation also
  measures guessing 2 and 10 layers ahead (recall drops with distance,
  Fig. 2 right — dashed/dotted lines).
- **Budget**: "fetches [the] most likely experts" — a fixed count (the value
  is rendered as math in the arXiv HTML and not extractable; see Fig. 2).
- **Admission — the notable part**: "The newly loaded experts **do not
  replace the currently cached experts**. If a speculatively loaded expert
  was later used during next layer inference, it will replace the
  least-recently used expert from the next layer's cache" (Sec. 4.1). I.e. a
  **probation/staging admission**: unused speculations never pollute the LRU
  cache.
- **Eviction**: per-layer LRU with an equal expert quota per layer.
- **Metric**: "expert recall" — the fraction of times a needed expert was
  already on GPU — for LRU caching and for speculative loading separately
  (Fig. 2); plus tokens/s on consumer GPUs.
- **Simulator**: predictor needs hidden states + gates → **one-layer-ahead
  trace oracle**; the *admission rule* is online and is shortlist mode A2.

### 1.2 MoE-Infinity — Xue et al. (arXiv:2401.14361)

- **Predictor input**: routing counts only. An **iteration-level EAM (iEAM)**
  (`layers x experts`, tokens routed per expert, updated per layer) is
  matched by **cosine distance** against a store of historical
  **request-level EAMs (EAMC)**; matched EAMs are summed, row-normalized
  into activation likelihoods (pEAM), then a **layer-proximity decay**
  scales each cell by distance from the current layer (Sec. 4.5, Fig. 5).
- **Horizon**: prefetch targets "the experts that are likely to be activated
  for the next layer" as execution proceeds (Sec. 4.6); the pEAM covers all
  remaining layers, so deeper prefetch is the same statistic.
- **Budget**: top-likelihood entries; the paper reports K-means evidence
  that requests fall into 10-30 activation groups, so one matched EAM is
  predictive within a group (Table 2), while **group transitions are
  unpredictable** (max transition probability ≈0.3, mostly <0.12, Fig. 4) —
  the stated reason to reject post-hoc learned predictors.
- **Admission / eviction coupling**: the same pEAM likelihood, with a layer
  decay, is the **eviction priority** (Algorithm 1: evict the lowest
  activation-likelihood resident); initial layers get extra caching priority
  because prediction is least confident at request start (Sec. 4.6). So
  prefetch and eviction share one predictor — co-designed, prefetch-aware.
- **Metric**: 3.1-16.7x per-token latency vs vLLM/Ollama/DeepSpeed/
  BrainStorm (abstract); no isolated recall@k for the predictor.
- **Simulator**: EAMs are routing statistics → **fully online** (build the
  EAMC from the trace prefix). Shortlist P4.

### 1.3 Pre-gated MoE — Hwang et al. (arXiv:2308.12066; ISCA'24)

- **Predictor**: none at inference time — the **pre-gate function of block
  `l` selects the experts of block `l+1`** (the model is fine-tuned with
  this function, Sec. IV-B), removing the selection→execution dependency.
- **Horizon**: exactly 1 block; **budget**: exactly the activated set (no
  speculation waste); **admission**: current + next block resident
  (double-buffer), eviction is "drop the finished block".
- **Accuracy**: exact by construction; the paper reports comparable or
  higher model accuracy than the original MoE after fine-tuning, and
  explicitly warns that for unmodified models "predicting active experts
  does not always ensure accuracy, and mispredictions can lead to penalties
  such as waste of GPU memory and ... serialization" (Sec. III-B).
- **Simulator**: model change → **one-layer-ahead exact oracle** is its
  ceiling (round-1 baseline B8).

### 1.4 ProMoE — Song et al. (arXiv:2410.22134; EuroSys'25)

- **Predictor**: a small per-layer **MLP trained offline** on collected
  traces, mapping a layer's gating input to that layer's expert selection;
  runs on CPU (~200 us) concurrently with inference (Sec. 4.3). Defines
  **GoodPred = combination of Accuracy and FetchRate** (fraction of
  predicted experts actually fetched in time) to score predictors.
- **Horizon**: layer-wise prefetch (+1); **stride prefetching** increases
  the prediction distance by 1 and pipelines prediction with fetching, so
  prediction for layer `l+2` runs while layer `l+1`'s fetch is in flight
  (Sec. 4.4). Accuracy declines only ~5% at the longer distance.
- **Budget**: "selects the same number of experts that the model would
  activate for one token" per layer (e.g. 6 for DS-1).
- **Accuracy (Fig. 8)**: learned predictor **84.7% average**; the skip-based
  predictor (Mixtral-style gate reuse) 66.9% average on QW-2 with drops at
  head/tail layers; a token-based predictor (SiDA-style, whole iteration
  ahead) 58.3%.
- **Admission / coordination**: per-layer **LRU** cache; a 2-priority task
  queue (low = speculative, high = precise, i.e. demand); **chunked
  prefetching** (expert split into its 3 linear layers so a demand fetch
  preempts within one chunk); **early preemption** (at gate completion, the
  precise expert list preempts and cancels that layer's speculative tasks;
  misses become waits on in-flight prefetches); **reordered inference**
  (compute cached experts first, then in-flight, then missing — "the
  prefetching of the missing expert might evict other soon-to-be-accessed
  experts, causing cache thrashing"). Prefetch tasks *touch* already-cached
  experts to refresh their LRU position ("thereby delaying their eviction",
  Sec. 6) — a weak form of prefetch-aware eviction.
- **Simulator**: MLP needs hidden states → **multi-layer oracle** for the
  predictor; the queue-priority, early-preemption and touch-refresh ideas
  are online admission mechanics (shortlist A1/A2 variants).

### 1.5 HOBBIT — Tang et al. (arXiv:2411.01433)

- **Predictor**: **stack all remaining gating modules** and apply them to
  the current layer's gating input in one batched matmul (Stacking
  Computer); justified by high cosine similarity of gating inputs across
  consecutive layers (Fig. 7a).
- **Horizon**: **adaptive** — predict layer `l+1`; if all predicted experts
  are already cached, predict `l+2`, and so on until a predicted expert is
  missing or all gates are consumed (Sec. 3.3); evaluation recommends
  limiting depth to **1-3** layers (Sec. 5.5.2, Fig. 17).
- **Budget**: predicted top-k per layer, fetched in **mixed precision** (low
  precision when a full-precision load would not finish in time), which
  caps the misprediction penalty since a wrong low-precision fetch is ~4x
  cheaper (int4 vs fp16).
- **Admission / eviction coupling**: "we will **mask all predicted experts
  to prevent them from being evicted** from the expert cache" — explicit
  prefetch-aware pinning. Eviction itself is a weighted sum of **LRU +
  sequence-level LFU + LHU (high-precision-use LFU) + FLD (farthest layer
  distance)** priorities (Eq. 3), with separate high-/low-precision caches
  and per-sequence record resets. FLD — prioritizing experts of nearer
  upcoming layers — is a static cousin of LCNU's cycle term.
- **Accuracy (Fig. 7b, Mixtral-8x7B)**: next-layer top-1 prediction ≈**96%**
  average; next-2/next-3 ≈**90%**. Prefill prefetch is ~100% accurate (all
  experts used) and cuts prefill latency ~10%; decode-time gains need the
  mixed-precision loader (1.05x; without it 1.01x or slightly negative) —
  i.e. **naive same-queue prefetch barely helps when mispredictions stall
  the link**, HOBBIT's own ablation.
- **Simulator**: gating-input predictor → **`h`-layer oracle** (B9); the
  pin-mask is online admission (shortlist A5); FLD is an online eviction
  feature already subsumed by LCNU.

### 1.6 AdapMoE — Zhong et al. (arXiv:2408.10284; ICCAD'24)

- **Predictor**: reuse the **next layers' own gate functions on the current
  activations** (same residual-similarity argument); for the *first* layer
  of the next token, an **auxiliary predictive gate** (a `hidden_dim x
  n_expert` linear layer) is trained to predict layer-0 gating from the
  previous token's last-layer activation (KL loss; training optional).
- **Horizon**: next layer; **extends to the next-next layer (2-3 ahead)
  whenever the predicted set for `l+1` is already cached** — the same
  adaptive-depth idea as HOBBIT.
- **Budget**: the (adaptively reduced) number of experts chosen by its
  sensitivity-based gating — AdapMoE also *skips* low-sensitivity experts
  (−25% activations), which changes outputs and is out of scope here.
- **Admission / eviction**: **LRU** eviction ("All the methods take LRU as
  its cache elimination algorithm", Sec. 6); plus a **DP/knapsack per-layer
  cache-size allocation** minimizing expected on-demand loads, where each
  layer's miss cost is computed from its profiled **prefetching accuracy**
  and cache hit probability (Sec. 4.4) — prefetch-awareness at the capacity
  level rather than per-decision.
- **Accuracy**: "most layers maintain ... around **90%**" prefetch accuracy
  (Fig. 9b), including the trained layer-0 gate.
- **Simulator**: gate reuse needs activations → oracle; the **per-layer
  capacity split informed by per-layer predictor accuracy** is an online
  idea worth a note in prompt E's admission work.

### 1.7 SiDA-MoE — Du et al. (arXiv:2310.18859; MLSys'24)

- **Predictor**: an offline-trained **2-layer LSTM + SparseMax attention**
  "hash function" over token embeddings predicts **all activated experts
  for every token at every layer**, trained with truncated knowledge
  distillation + cross-entropy (Sec. 3.4).
- **Horizon**: the **entire forward pass, one batch ahead** — a
  hash-building thread processes batch `i` while inference runs batch
  `i-1`.
- **Budget / admission**: load exactly the predicted-active experts, offload
  predicted-inactive ones; "A first-in-first-out (FIFO) scheme is applied on
  experts if no memory budgets remain" (Sec. 3.3).
- **Accuracy**: top-3 "hash hits rate" per model/dataset in Table 5 — the
  numeric cells are math images in the arXiv HTML and were not extractable;
  the abstract claims up to 3.93x throughput, up to 80% GPU memory saving,
  down to 1% accuracy drop. Klotski (arXiv:2502.06888, Sec. 1) summarizes
  SiDA as "achieving a prefetching accuracy of over 90%".
- **Simulator**: needs embeddings → **per-input oracle** (round-1 B10;
  over-states SiDA since real recall < 1).

### 1.8 ExpertFlow — He et al. (arXiv:2410.17954; DAC'26)

- **Predictor**: **Routing Path Predictor (RPP)** — a T5-style
  encoder-decoder with per-layer heads that outputs the **whole routing
  plan (all tokens x all layers) in one forward pass** before the first MoE
  layer, trained as multi-label classification on logged routing paths
  (Sec. 3.2).
- **Horizon**: the entire forward pass; **budget**: predicted experts, with
  **PLEC** allocating cache slots **across layers proportionally to
  predicted demand** when the predicted total exceeds capacity, and reusing
  slots freed by finished early layers (Sec. 3.4.1).
- **Admission / correction**: prefetch the most probable experts
  (step-ordered); runtime detects mispredictions and performs
  **prioritized swaps** while other experts compute (Real-time Correction);
  a token scheduler regroups tokens with similar predicted paths
  (out of scope for the simulator).
- **Accuracy**: RPP "up to **95%** expert prediction accuracy" (in-domain);
  ECE cache hit ratio **91.96%**, "outperforming LRU by up to **61.15%**";
  TS adds up to 16.19% throughput (Sec. 1 contributions).
- **Simulator**: needs the input sequence → **per-input oracle**; PLEC's
  predicted-demand slot allocation is an admission idea (a per-layer-budget
  variant of A3).

### 1.9 fMoE / FineMoE — Yu et al. (arXiv:2502.05370)

v1 was titled "fMoE: Fine-Grained Expert Offloading for Large
Mixture-of-Experts Serving"; v2 (the current record) is "Taming
Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert
Offloading" with the system renamed FineMoE. Same arXiv ID; both names in
circulation.

- **Predictor**: an **expert-map store** of historical iteration-level gate
  **probability distributions** ("expert maps"), searched by two
  similarities: **semantic** (prompt embedding) and **trajectory** (cosine
  over the current request's distributions through layers); the searched
  map supplies per-expert activation probabilities (Secs. 4.1-4.2).
- **Horizon**: a configurable **prefetch distance `d`** — "the number of
  layers ahead that a prefetch instruction is issued before the target
  layer activates its experts" (Sec. 2, Fig. 4 sweeps it).
- **Budget — adaptive**: compute a threshold θ from the search similarity
  score and take experts by descending probability until the summed
  probability exceeds θ, with at least the model's top-k per layer
  (Eqs. 6-8): low-confidence matches prefetch more, high-confidence fewer.
- **Admission**: prefetch priority = map probability scaled down by layer
  distance from the current layer; on a miss, all prefetch tasks pause and
  the missed expert loads immediately (Sec. 4.5).
- **Eviction — prefetch-aware**: "FineMoE's expert caching is based on the
  LFU caching algorithm. We integrate the searched map to jointly determine
  the eviction priority" — evict the expert with the lowest combination of
  cache-visit frequency and predicted activation probability; recency is
  deliberately excluded because "expert usage is layer-wise sequential ...
  prioritizing recently used experts is against the nature of sequential
  forward computation" (Sec. 4.5) — independent confirmation of round 1's
  LRU-collapse finding.
- **Metric**: −47% inference latency, +39% expert hit rate vs
  state-of-the-art baselines (abstract).
- **Simulator**: semantic search needs prompts (oracle); the **trajectory
  side degrades gracefully** to matching routed-set histories — i.e. an
  EAMC variant — and the **LFU x predicted-probability eviction** and the
  **similarity-scaled budget** are both implementable online (P4/A4 notes).

### 1.10 Klotski — Fang et al. (arXiv:2502.06888)

- **Predictor — routing-only**: a **data-aware expert correlation table**
  recording "the correlations (i.e., frequency relationships) between
  experts activated by tokens at different layers", built during a warm-up
  pre-run and **updated online during inference**; at layer `l`, each
  token's tendency is looked up from its expert selections in the previous
  `m` layers, tendencies are aggregated across all tokens in the
  multi-batch, and the **top-K experts (K = the model's top-k)** are
  prefetched as "hot experts" during the attention computation
  (Secs. 5, 6.2, Fig. 8).
- **Horizon**: next layer (issued one sublayer early, during attention);
  **budget**: K experts per layer plus the gate weights (Eq. 1 bounds what
  fits under the attention window).
- **Admission / schedule**: non-hot gate-selected experts start transferring
  at gate time; expert **computation is reordered** (hot/loaded experts
  first) so transfers overlap; "experts that have completed all computations
  are offloaded immediately" — residency is schedule-driven rather than
  cache-policy-driven.
- **Accuracy (Fig. 13)**: hot-expert prediction **58.89%** average across
  layers; **100%** of prefetched hot experts participate in computation
  under multi-batch (vs 42.24% single-sequence) — batch aggregation, not
  predictor quality, is what removes I/O waste.
- **Simulator**: the correlation table is a pure routing statistic →
  **online** (shortlist P3); the multi-batch scheduling itself is out of
  scope (our traces are batch-1).

### 1.11 SpecMD — Hoang et al. (arXiv:2602.03921)

The seed list flagged this ID as suspicious; it resolves cleanly
(submitted 2026-02-03, Apple/UT Austin authors: Hoang, Jaiswal, Samragh,
Cho). It is the closest published analogue to this repo's round-2 agenda: a
**policy-first benchmarking framework** decomposing MoE cache management
into **routing / prefetching / eviction / miss-handling** axes and sweeping
their combinations across 4 models and 3 capacity regimes (1%/5%/25%).

- **Prefetch**: next layer only, via "the gate from an upcoming layer
  together with the activations of the current layer" (Sec. 3.1.4), run by
  an asynchronous watchdog that enqueues prefetches when capacity is free.
  Two budgets: **top-k x overfetch factor** (e.g. 1.5x → 12 experts
  instead of 8 on OLMoE) and **score-based** — prefetch *all* experts above
  a softmax-score percentile (default 80th), following Specter
  (arXiv:2511.14102), so the per-layer count adapts to the score
  distribution.
- **Key findings**: (i) "MoE expert access is not consistent with temporal
  locality assumptions (e.g. LRU, LFU)" — wrongful evictions arise from the
  deterministic sequential layer pattern (matches round 1's cyclic-scan
  analysis); (ii) **score-based prefetching beats top-k on hit rate and
  synchronous overhead despite lower prediction accuracy**, because it
  reallocates bandwidth from high-risk to high-reward layers — except on
  Mixtral-class models whose huge experts leave no bandwidth headroom;
  (iii) policy rankings shift across capacity/bandwidth regimes; no single
  combination dominates.
- **Least-Stale eviction (prefetch-aware)**: partition residents into a
  **stale** queue (used only in previous forward passes) and a **current**
  queue (used this pass **or selected by prefetching**); evict stale first,
  FIFO by layer position within a queue; current entries are evicted only
  to prevent OOM (Sec. 4.2). Reduces collision misses (evicted then needed
  again in the same pass) by up to 85x vs LRU; 88-92% hit rates at 1-5%
  capacity; TTFT −10.7-34.7% on OLMoE as a drop-in.
- **Simulator**: Least-Stale is **routing-only and online** — it is
  essentially per-pass staleness + layer-cycle distance, a coarse cousin of
  LCNU (which additionally models per-expert reuse probability). The
  score-percentile budget needs gate scores → oracle; its *shape* (adaptive
  per-layer budget) is reproducible online from predicted probabilities
  (P3/P4 scores).

### 1.12 MoE-Lightning (arXiv:2411.11217) and SwapMoE (arXiv:2308.15030)

- **MoE-Lightning**: throughput-oriented **CGOPipe** pipelining with paged
  weights — weights stream positionally; attention runs on CPU; no
  routing-based expert prediction. Prefetch-all baseline for the timing
  model (its HRM roofline analysis is a useful reference for prompt E's
  PCIe model), not a candidate prefetcher.
- **SwapMoE**: maintains a resident "Virtual Experts" set by a
  gating-derived importance score, updated **asynchronously and amortized**
  across tokens; **masked gating** routes only to resident experts, so
  outputs change — out of scope for trace replay (importance ≈ score-based
  eviction, covered in round 1). No speculative per-layer prefetch.

## 2. Integrated prefetching and caching theory

### 2.1 Cao, Felten, Karlin & Li, SIGMETRICS'95 — the four rules

"A Study of Integrated Prefetching and Caching Strategies" (SIGMETRICS '95,
pp. 188-197). Model: single resource, unit-time accesses, fetch latency `F`,
serialized fetches, a block being replaced becomes unavailable the moment the
prefetch is issued; goal is minimal elapsed time (stall), with full knowledge
of the reference stream. Any optimal integrated strategy can be transformed
to satisfy four rules (Sec. 2.2, quoted verbatim):

1. **Optimal prefetching**: "Every prefetch should bring into the cache the
   next block in the reference stream that is not in the cache."
2. **Optimal replacement**: "Every prefetch should discard the block whose
   next reference is furthest in the future."
3. **Do no harm**: "Never discard block A to prefetch block B when A will be
   referenced before B."
4. **First opportunity**: "Never perform a prefetch-and-replace operation
   when the same operations ... could have been performed previously."

Two concrete policies built on the rules: **conservative** (perform exactly
MIN's replacements, each fetch moved to the earliest opportunity consistent
with the rules — same fetch count as demand MIN, elapsed time within 2x of
optimal) and **aggressive** (always prefetch the next missing block at the
earliest moment the rules allow, replacing the furthest-next-reference
resident; "it at least waits until there is a block in cache whose next
reference is after the first reference to the block that will be fetched";
fetch count ≤ LRU demand paging, elapsed time within a factor
`min(1 + F/k, 2)` of optimal). The follow-up implementation paper (Cao et
al., "Implementation and Performance of Integrated Application-Controlled
File Caching, Prefetching, and Disk Scheduling", ACM TOCS 14(4), 1996)
defines **controlled-aggressive**: "always answers 'yes' whenever Rules 3
and 4 allow it".

**Assumptions vs our setting**: known future (oracle), uniform fetch cost,
one serialized fetch channel — the serial-link assumption matches the
shared PCIe link in the round-2 timing model, and the known-future variant
is exactly what the per-layer routing oracle provides. **Rule 3 is directly
implementable online** by replacing true next-use with an estimate: LCNU
already computes an expected next use for every resident, and any shortlist
prefetcher supplies a predicted next use (`d(L)` of the target layer) for
the candidate. This is shortlist admission mode A4.

### 2.2 Patterson et al., SOSP'95 — TIP, cost-benefit allocation

"Informed Prefetching and Caching" (Patterson, Gibson, Ginting, Stodolsky,
Zelenka; SOSP '95, pp. 79-95). Applications **disclose** future accesses as
hints; the buffer manager runs a **cost-benefit analysis in a common
currency** (change in I/O service time per buffer-access) to split buffers
among three consumers: prefetching hinted blocks, caching hinted blocks for
reuse, and an LRU cache for unhinted accesses.

- **Prefetch horizon**: with inter-access compute `T_CPU`, hit cost `T_hit`,
  driver cost `T_driver`, disk latency `T_disk`, the worst-case stall of a
  prefetch issued `x` accesses early is `T_disk − x(T_CPU + T_hit +
  T_driver)` (Eq. 6), so prefetching beyond `P(T_CPU) = T_disk / (T_CPU +
  T_hit + T_driver)` accesses ahead yields **zero additional benefit**
  (Eq. 7): "There is no benefit from prefetching further ahead than the
  prefetch horizon." The marginal benefit of one more prefetch buffer at
  depth `x` is `ΔT_pf(x) ∝ T_disk/(x(x+1))` (Eq. 9) — steeply diminishing.
- **Replacement**: take the globally cheapest buffer — the LRU tail's
  estimated cost (hit-ratio profile) or the **hinted block whose next use is
  furthest** (its cost is the stall its re-fetch will cause, which is small
  if the re-fetch can itself be prefetched in time) — whenever a consumer's
  marginal benefit exceeds that cost.
- **Mapping**: the prefetch horizon translates directly to the round-2
  timing model — with per-layer compute `t_layer` and per-expert transfer
  `t_xfer` on one link, prefetch issued at layer `L` for layer `L+h` hides
  the transfer iff `h * t_layer ≥ queue + t_xfer`; deeper prefetch buys
  nothing and only occupies slots (bounds the useful horizon sweep in
  prompt F). The common-currency idea motivates reporting **stall per
  token** rather than hit rate alone.

### 2.3 Wu et al., MICRO-44 2011 — PACMan

"PACMan: Prefetch-Aware Cache Management for High Performance Caching" (Wu,
Jaleel, Martonosi, Steely Jr., Emer; MICRO-44, pp. 442-453). Built on RRIP
(2-bit re-reference prediction value per line; insert at RRPV 2/3, promote
to 0 on hit, evict RRPV 3): treat **prefetch requests differently from
demand requests** at insertion and at promotion:

- **PACMan-M** (misses): demand misses insert per baseline (RRPV 2, or 3
  for the bimodal fraction); **prefetch misses always insert with RRPV 3**
  (nearest eviction, "LRU position") — shortens the lifetime of unused
  prefetches and prioritizes demand over prefetch.
- **PACMan-H** (hits): demand hits promote to RRPV 0; **prefetch hits do
  not update the RRPV** — a line kept alive only by the prefetcher is
  "prefetchable" and need not occupy cache, while hard-to-prefetch demand
  lines are retained.
- **PACMan-HM**: both; **PACMan-DYN**: set-dueling picks among them per
  workload, since static demotion hurts prefetch-friendly workloads.

**Mapping**: in expert-cache terms, PACMan-M = *admit prefetched experts at
the victim end of the evictor's order* (probation, A2) and PACMan-H =
*a prefetch touching a resident expert must not refresh the evictor's
recency/frequency state* — the exact opposite of ProMoE's touch-refresh;
both arms are testable online in prompt E. The caveat: PACMan assumes
prefetches target the *same* reuse stream; in MoE decode, prefetches target
the next layer visit (one cycle ahead), so demotion may be too aggressive —
which is what admission-mode sweeps should measure.

### 2.4 Seshadri et al., ACM TACO 2015 — ICP

"Mitigating Prefetcher-Caused Pollution Using Informed Caching Policies for
Prefetched Blocks" (Seshadri, Yedkar, Xin, Mutlu, Gibbons, Kozuch, Mowry;
TACO 11(4), Art. 51). Two orthogonal rules: **ICP-D (demotion)** — "over
95% of useful prefetches are not reused after the first demand hit", so a
prefetched block is **demoted to lowest priority on its first demand hit**
(one bit per entry); **ICP-AP** — a self-tuning **prefetch-accuracy
predictor** gates insertion: predicted-accurate prefetches insert with high
priority, predicted-inaccurate with lowest. Mapping: ICP-D is the
"use-once-then-demote" admission (a stronger probation: promote on hit,
then immediately demote); ICP-AP corresponds to scaling admission priority
by the online predictor's measured per-layer accuracy (cf. AdapMoE's
per-layer accuracy profile). Caution: the ≥95% single-use statistic is for
CPU cache lines; MoE experts routinely see multi-token reuse, so ICP-D
needs testing rather than assuming.

### 2.5 Jain & Lin, ISCA 2018 — Demand-MIN and Flex-MIN

"Rethinking Belady's Algorithm to Accommodate Prefetching" (45th ISCA).
With prefetches in the stream, Belady's MIN minimizes *total* misses but
not *demand* misses. **Demand-MIN**: "Unlike MIN, which evicts the line
that is reused furthest in the future, Demand-MIN evicts the line that is
**prefetched furthest in the future** — and then falling back on MIN if no
such line exists." It preferentially evicts lines whose next access is a
prefetch (prefetch-friendly lines, repeated prefetches, dead intervals),
trading extra traffic for demand hits. **Flex-MIN** interpolates: evict the
furthest-prefetched *unprotected* line, where "protected" prefetches are
those whose eviction would cost traffic without hit-rate benefit.
PA-Hawkeye learns Flex-MIN's decisions online.

**Mapping — important for prompts E/F's bounds**: once a prefetcher is
active, the demand-hit-rate ceiling is **not** the demand-only Belady the
simulator asserts today; a (prefetcher, evictor) pair can legitimately
exceed it. The clean upper bound for a given oracle prefetch schedule is a
Demand-MIN-style variant (evict the resident whose next *demand* use is
farthest, treating prefetchable next-uses as free). Keep the existing
`<= Belady` assertion for the *no-prefetch demand path* and report prefetch
value separately (coverage/stall), as the plan already specifies; a
brute-force Demand-MIN on small traces would make a good additional test.

## 3. Shortlist for prompt E

### 3.1 Online prefetchers (routing-trace inputs only)

All see only past routing + the current token's routing up to the current
layer `L`, and emit `(layer, expert, score)` for the next `h` layer visits
under per-layer budget `m`. Defaults to sweep: `h ∈ {1, 2, 4, 8}`,
`m ∈ {k/2, k, 2k}` (`k` = model top-k), tuned on the first half of requests.

- **P1 — Previous-token persistence.** For target layer `L+d` (wrapping to
  the next token's early layers), predict the experts routed at that layer
  **on the previous token**, scored by 1 (or by LCNU's per-expert EMA).
  Rationale: P(expert routed again at the same layer next token) = 0.39-0.43
  on Qwen3 — 2-3x its frequency baseline (decode study §2); this is the
  signal Mixtral/HOBBIT's hidden-state similarity proxies. Inputs: last
  token's per-layer routed sets. Budget: that set is exactly `k` per layer;
  with `m > k` extend by P2 scores.
- **P2 — Per-layer frequency (EAM-lite).** Score expert `e` at layer `l` by
  a decayed routing frequency `p(l, e)` — either a sliding-window count or
  exactly LCNU's blended estimate `beta * ema + (1-beta) * smoothed count`
  (no new state if the evictor is LCNU). Prefetch the top-`m` per upcoming
  layer. This is the online core of MoE-Infinity's activation-awareness and
  Klotski's hot experts (whose measured hot-expert accuracy is 58.89%).
- **P3 — Cross-layer co-activation (Klotski table / first-order EAM).**
  Maintain counts `C_d[e, e']` = times expert `e'` was routed at layer
  `l+d` given `e` routed at layer `l` (accumulated causally over the trace
  prefix; optionally per (l, d) to keep layer identity). On finishing layer
  `L` with routed set `S_L`, score `e'` at layer `L+d` by
  `sum_{e in S_L} C_d[e, e'] / count(e)` and take the top-`m`. Variants:
  condition on the union of the last `j` layers' sets (Klotski uses `m`
  previous layers); normalize per column to de-bias hot experts. Memory:
  `d_max * E^2` counters per layer pair or shared.
- **P4 — Request-matching EAMC (MoE-Infinity, faithful).** Keep up to `C`
  historical request-level EAMs (row-normalized routing frequencies,
  `layers x experts`). At layer `L` of the current token, flatten the
  current request's accumulated iteration/request-level EAM prefix, find the `n` nearest
  stored EAMs by cosine distance, average them row-normalized, apply a
  layer-proximity decay, and prefetch the top-`m` per upcoming layer.
  Update/insert the request's EAM at request end (fMoE's dedup — drop the
  most redundant map when full — bounds the store). Purely routing-based;
  the whole-trace (non-causal) variant is its oracle.
- **P5 — Least-Stale-guided refresh (SpecMD-flavored, optional).** Not a
  separate predictor: schedule P1-P4's candidates in **layer-cycle order**
  (nearest upcoming layer first — Cao's first-opportunity + HOBBIT's FLD),
  and only for layers whose candidate is not already resident (HOBBIT's
  adaptive-depth rule: spend the budget on the first `h` layers with
  missing predictions). This is the issue-order policy all four above
  should share.

Oracle counterparts (upper bounds, already partly in the simulator):
one-layer exact (Pre-gated/skip-gate ceiling), `h`-layer cascade
(HOBBIT/ProMoE/SpecMD ceiling), per-request set (SiDA/ExpertFlow ceiling),
plus a **score-threshold oracle** (SpecMD's percentile budget: prefetch all
true next-layer experts, i.e. adaptive `m`) to bound adaptive budgets.

### 3.2 Admission modes for prefetched experts

Each mode composes with any evictor via a victim-ranking / next-victim
hook; all are precise enough to implement directly.

- **A1 — Same-queue, demand priority** (status quo;
  ProMoE/`_OracleLRUPolicy`). Prefetched experts are admitted exactly like
  demand fetches (full eviction protection/recency update); demand misses
  preempt pending prefetches on the link. Baseline. Sub-knob (PACMan-H):
  whether a prefetch of an already-resident expert refreshes the evictor's
  state (ProMoE: yes; PACMan-H: no) — implement as a boolean
  `prefetch_touch_refreshes`.
- **A2 — Probation / victim-first insertion** (Mixtral-Offloading,
  PACMan-M, ICP-D). Prefetched experts are inserted at the evictor's
  victim end (first to be evicted; they never displace a resident that the
  evictor ranks above any probationary entry). On first demand hit the
  entry is promoted to a normal admission (Mixtral: "if a speculatively
  loaded expert was later used ... it will replace the least recently used
  expert"). Optional ICP-D flavor: demote again after the layer that used
  it completes.
- **A3 — Separate prefetch buffer.** A side buffer of `B` slots with its own
  replacement (FIFO, or drop entries whose target layer has passed); demand
  hits in the buffer move the expert into the main cache as a normal
  admission; buffer entries never evict main-cache residents; unused
  entries die in the buffer. Sweep `B` as a fraction of capacity;
  ExpertFlow's PLEC suggests apportioning `B` across layers by predicted
  demand.
- **A4 — Do-no-harm (Cao Rule 3 + first opportunity, online via LCNU).**
  Admit a prefetch of candidate `c` targeted at layer `l_c` only if the
  evictor's current victim `v` has an **estimated next use later than
  `c`'s**: with LCNU, require `score_LCNU(v) > d(l_c) + layers *
  (1/p(l_c, c) − 1)` (or, simpler first cut, `score_LCNU(v) > d(l_c)`,
  i.e. the victim's expected reuse is beyond the target layer's visit).
  Issue at the earliest time the test passes (first opportunity); demand
  misses always admit. This is the only mode that provably never hurts the
  evictor under a perfect next-use estimate (Rules 1-4), and LCNU supplies
  the estimate for free.
- **A5 — Pin-until-use (HOBBIT mask), optional.** Admit same-queue but mark
  prefetched experts non-evictable until their target layer executes (then
  unpin, whether used or not). Must cap pinned slots (e.g. `≤ h * k`) to
  avoid starving demand admissions. Measures the value of HOBBIT's
  "mask all predicted experts to prevent them from being evicted".

### 3.3 Notes for the harness and metrics

- SpecMD's central result — adaptive per-layer budgets beat fixed top-k on
  hit rate *despite worse precision*, except when expert transfer time
  dwarfs the per-layer compute window — predicts that budget shape will
  matter more than predictor choice at small horizons; sweep budgets, not
  just horizons.
- HOBBIT's ablation (prefetch ≈1.01x without the cheap-misprediction
  loader) and Pre-gated's misprediction warning both say the same thing:
  on a saturated serial link, **wasted prefetches are the dominant cost**,
  so report wasted transfers and timing-model stall, not only coverage
  (already in the shared metrics contract).
- TIP's prefetch horizon bounds the useful `h` once the timing model
  exists: `h* ≈ ceil((queueing + t_xfer) / t_layer)`; deeper prefetch only
  costs slots.
- With prefetching enabled, demand hit rate may legitimately exceed
  demand-only Belady (Demand-MIN, §2.5); keep the Belady assertion on the
  no-prefetch path and consider a small brute-force Demand-MIN test as the
  prefetch-mode ceiling.

## 4. BibTeX

Entries already in `policy_survey.md` (MoE-Infinity, Mixtral-Offloading,
Pre-gated, ProMoE, HOBBIT, SiDA, FineMoE/fMoE, SwapMoE, MoE-Lightning,
ExpertFlow, Klotski, SpecMD, Specter, DeepSpeed-Inference, SE-MoE) keep the
same keys there; repeated below only where this survey cites them directly,
plus the new theory and AdapMoE entries.

```bibtex
@article{eliseev2023mixtraloffloading,
  title   = {Fast Inference of Mixture-of-Experts Language Models with Offloading},
  author  = {Eliseev, Artyom and Mazur, Denis},
  journal = {arXiv preprint arXiv:2312.17238},
  year    = {2023},
  doi     = {10.48550/arXiv.2312.17238}
}

@article{xue2024moeinfinity,
  title   = {{MoE-Infinity}: Efficient {MoE} Inference on Personal Machines with Sparsity-Aware Expert Cache},
  author  = {Xue, Leyang and Fu, Yao and Lu, Zhan and Mai, Luo and Marina, Mahesh},
  journal = {arXiv preprint arXiv:2401.14361},
  year    = {2024},
  doi     = {10.48550/arXiv.2401.14361}
}

@inproceedings{hwang2024pregated,
  title     = {Pre-gated {MoE}: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference},
  author    = {Hwang, Ranggi and Wei, Jianyu and Cao, Shijie and Hwang, Changho and Tang, Xiaohu and Cao, Ting and Yang, Mao},
  booktitle = {Proceedings of the 51st ACM/IEEE Annual International Symposium on Computer Architecture (ISCA '24)},
  year      = {2024},
  note      = {arXiv:2308.12066},
  doi       = {10.48550/arXiv.2308.12066}
}

@inproceedings{song2024promoe,
  title     = {{ProMoE}: Fast {MoE}-based {LLM} Serving using Proactive Caching},
  author    = {Song, Xiaoniu and Zhong, Zihang and Chen, Rong and Chen, Haibo},
  booktitle = {Proceedings of the Twentieth European Conference on Computer Systems (EuroSys '25)},
  year      = {2025},
  note      = {arXiv:2410.22134},
  doi       = {10.48550/arXiv.2410.22134}
}

@article{tang2024hobbit,
  title   = {{HOBBIT}: A Mixed Precision Expert Offloading System for Fast {MoE} Inference},
  author  = {Tang, Peng and Liu, Jiacheng and Hou, Xiaofeng and Pu, Yifei and Wang, Jing and Heng, Pheng-Ann and Li, Chao and Guo, Minyi},
  journal = {arXiv preprint arXiv:2411.01433},
  year    = {2024},
  doi     = {10.48550/arXiv.2411.01433}
}

@inproceedings{zhong2024adapmoe,
  title     = {{AdapMoE}: Adaptive Sensitivity-based Expert Gating and Management for Efficient {MoE} Inference},
  author    = {Zhong, Shuzhang and Liang, Ling and Wang, Yuan and Wang, Runsheng and Huang, Ru and Li, Meng},
  booktitle = {Proceedings of the 43rd IEEE/ACM International Conference on Computer-Aided Design (ICCAD '24)},
  year      = {2024},
  note      = {arXiv:2408.10284},
  doi       = {10.1145/3676536.3676741}
}

@inproceedings{du2024sidamoe,
  title     = {{SiDA-MoE}: Sparsity-Inspired Data-Aware Serving for Efficient and Scalable Large Mixture-of-Experts Models},
  author    = {Du, Zhixu and Li, Shiyu and Wu, Yuhao and Jiang, Xiangyu and Sun, Jingwei and Zheng, Qilin and Wu, Yongkai and Li, Ang and Li, Hai and Chen, Yiran},
  booktitle = {Proceedings of Machine Learning and Systems (MLSys '24)},
  year      = {2024},
  note      = {arXiv:2310.18859},
  doi       = {10.48550/arXiv.2310.18859}
}

@inproceedings{he2024expertflow,
  title     = {{ExpertFlow}: Efficient Mixture-of-Experts Inference via Predictive Expert Caching and Token Scheduling},
  author    = {He, Xin and Zhang, Shunkang and Tang, Kaijie and Shi, Shaohuai and Wang, Yuxin and Zeng, Zihao and Tang, Zhenheng and Chu, Xiaowen and Yin, Haiyan and Tsang, Ivor W. and Ong, Yew Soon},
  booktitle = {Proceedings of the 63rd ACM/IEEE Design Automation Conference (DAC '26)},
  year      = {2026},
  note      = {arXiv:2410.17954},
  doi       = {10.1145/3770743.3804292}
}

@article{yu2025finemoe,
  title   = {Taming Latency-Memory Trade-Off in {MoE}-Based {LLM} Serving via Fine-Grained Expert Offloading},
  author  = {Yu, Hanfei and Cui, Xingqi and Zhang, Hong and Wang, Hao and Wang, Hao},
  journal = {arXiv preprint arXiv:2502.05370},
  year    = {2025},
  note    = {v1 title: ``{fMoE}: Fine-Grained Expert Offloading for Large Mixture-of-Experts Serving''},
  doi     = {10.48550/arXiv.2502.05370}
}

@article{fang2025klotski,
  title   = {{Klotski}: Efficient Mixture-of-Expert Inference via Expert-Aware Multi-Batch Pipeline},
  author  = {Fang, Zhiyuan and Huang, Yuegui and Hong, Zicong and Lyu, Yufeng and Chen, Wuhui and Yu, Yue and Yu, Fan and Zheng, Zibin},
  journal = {arXiv preprint arXiv:2502.06888},
  year    = {2025},
  doi     = {10.48550/arXiv.2502.06888}
}

@article{hoang2026specmd,
  title   = {{SpecMD}: A Comprehensive Study On Speculative Expert Prefetching},
  author  = {Hoang, Duc and Jaiswal, Ajay and Samragh, Mohammad and Cho, Minsik},
  journal = {arXiv preprint arXiv:2602.03921},
  year    = {2026},
  doi     = {10.48550/arXiv.2602.03921}
}

@article{cao2024moelightning,
  title   = {{MoE-Lightning}: High-Throughput {MoE} Inference on Memory-constrained GPUs},
  author  = {Cao, Shiyi and Liu, Shu and Griggs, Tyler and Schafhalter, Peter and Liu, Xiaoxuan and Sheng, Ying and Gonzalez, Joseph E. and Zaharia, Matei and Stoica, Ion},
  journal = {arXiv preprint arXiv:2411.11217},
  year    = {2024},
  doi     = {10.48550/arXiv.2411.11217}
}

@inproceedings{kong2024swapmoe,
  title     = {{SwapMoE}: Serving Off-the-shelf {MoE}-based Large Language Models with Tunable Memory Budget},
  author    = {Kong, Rui and Li, Yuanchun and Feng, Qingtian and Wang, Weijun and Ye, Xiaozhou and Ouyang, Ye and Kong, Linghe and Liu, Yunxin},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL '24)},
  year      = {2024},
  note      = {arXiv:2308.15030},
  doi       = {10.48550/arXiv.2308.15030}
}

@inproceedings{cao1995integrated,
  title     = {A Study of Integrated Prefetching and Caching Strategies},
  author    = {Cao, Pei and Felten, Edward W. and Karlin, Anna R. and Li, Kai},
  booktitle = {Proceedings of the 1995 ACM SIGMETRICS Joint International Conference on Measurement and Modeling of Computer Systems},
  pages     = {188--197},
  year      = {1995},
  doi       = {10.1145/223587.223608}
}

@article{cao1996implementation,
  title   = {Implementation and Performance of Integrated Application-Controlled File Caching, Prefetching, and Disk Scheduling},
  author  = {Cao, Pei and Felten, Edward W. and Karlin, Anna R. and Li, Kai},
  journal = {ACM Transactions on Computer Systems},
  volume  = {14},
  number  = {4},
  pages   = {311--343},
  year    = {1996},
  doi     = {10.1145/235543.235544}
}

@inproceedings{patterson1995informed,
  title     = {Informed Prefetching and Caching},
  author    = {Patterson, R. Hugo and Gibson, Garth A. and Ginting, Eka and Stodolsky, Daniel and Zelenka, Jim},
  booktitle = {Proceedings of the Fifteenth ACM Symposium on Operating Systems Principles (SOSP '95)},
  pages     = {79--95},
  year      = {1995},
  doi       = {10.1145/224056.224064}
}

@inproceedings{wu2011pacman,
  title     = {{PACMan}: Prefetch-Aware Cache Management for High Performance Caching},
  author    = {Wu, Carole-Jean and Jaleel, Aamer and Martonosi, Margaret and Steely Jr., Simon C. and Emer, Joel},
  booktitle = {Proceedings of the 44th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO-44)},
  pages     = {442--453},
  year      = {2011},
  doi       = {10.1145/2155620.2155672}
}

@article{seshadri2015icp,
  title   = {Mitigating Prefetcher-Caused Pollution Using Informed Caching Policies for Prefetched Blocks},
  author  = {Seshadri, Vivek and Yedkar, Samihan and Xin, Hongyi and Mutlu, Onur and Gibbons, Phillip B. and Kozuch, Michael A. and Mowry, Todd C.},
  journal = {ACM Transactions on Architecture and Code Optimization},
  volume  = {11},
  number  = {4},
  pages   = {51:1--51:22},
  year    = {2015},
  doi     = {10.1145/2677956}
}

@inproceedings{jain2018rethinking,
  title     = {Rethinking Belady's Algorithm to Accommodate Prefetching},
  author    = {Jain, Akanksha and Lin, Calvin},
  booktitle = {Proceedings of the 45th Annual International Symposium on Computer Architecture (ISCA '18)},
  year      = {2018},
  doi       = {10.1109/ISCA.2018.00056}
}
```

## Verification notes

- All 13 arXiv IDs (12 seed + AdapMoE's 2408.10284, found via arXiv
  search; the seed list gave AdapMoE no ID) were resolved through
  `export.arxiv.org/api/query` on 2026-11-02; titles and author lists above
  come from those records. **No seed ID was wrong.** The flagged
  "SpecMD arXiv:2602.03921" resolves (v1, 2026-02-03; Hoang, Jaiswal,
  Samragh, Cho). arXiv:2502.05370's v1 title is "fMoE: ..." and the current
  v2 is the "Taming Latency-Memory Trade-Off ..." / FineMoE title — same
  paper, both names recorded.
- System mechanisms and numbers were extracted from the arXiv HTML full
  texts (`arxiv.org/html/<id>`) of all 13 papers; theory papers from the
  authors'/institutional PDFs: Cao '95 (homes.cs.washington.edu/~karlin),
  Cao '96 rules quoted via cseweb.ucsd.edu copy, Patterson '95
  (cs.columbia.edu), PACMan (istc-cc.cmu.edu), ICP TACO '15
  (istc-cc.cmu.edu), Jain & Lin ISCA '18 draft (cs.utexas.edu/~lin). The
  four Cao rules, PACMan-M/H/HM, Demand-MIN and Flex-MIN definitions, and
  the TIP prefetch-horizon equation are quoted/transcribed from those
  texts.
- Venue claims: ISCA'24 (Pre-gated), EuroSys'25 (ProMoE), MLSys'24 (SiDA),
  ACL'24 (SwapMoE) follow the round-1 survey's verified records; DAC'26
  (ExpertFlow) is in the arXiv record's comment field; ICCAD'24 (AdapMoE)
  carries ACM DOI 10.1145/3676536.3676741 (verified via the authors'
  repository citation). HOBBIT, Klotski, fMoE/FineMoE, MoE-Lightning and
  SpecMD are cited by arXiv ID only (no venue claimed).
- Numbers *not* extractable (math images in arXiv HTML) and therefore not
  stated: Mixtral-Offloading's cache sizes and speculative fetch count,
  SiDA's Table 5 hash-hit percentages (the >90% figure is attributed to
  Klotski's citation of SiDA, not to SiDA's own table). Everything else
  numeric above was read directly from the source text.
