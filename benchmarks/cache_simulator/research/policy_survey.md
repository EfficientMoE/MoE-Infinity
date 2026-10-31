# MoE / offloading expert-cache policy survey

Literature survey of expert **cache / prefetch / eviction** strategies for
Mixture-of-Experts (MoE) inference under GPU-memory offloading, mapped onto the
offline simulator in `benchmarks/cache_simulator/`. This is Task A of the
cache-policy research plan (`../auto_research_policies.md`,
`../prompts/01_research_baselines.md`); it produces a survey, a shortlist of
implementable baselines, and seed rows for `EXPERIMENTS.md`. **No simulator code
is added here** — that is Task B.

## How this maps onto the simulator

The simulator's cache key is `(layer_idx, expert_id)`; one entry holds one
expert's weights. Policies plug in two ways:

- **Belady / OPT bound** — `belady_cache_simulator.belady_simulate(accesses,
  capacity)` and `cache_strategy.ExpertCache` evict the resident whose next use
  is farthest in the future. This is the **"Ideal" upper bound**; no on-line
  policy can beat it, so every baseline must satisfy
  `hit_rate(policy) <= hit_rate(Belady)` on every trace.
- **On-line fetch engines** — `fetch_engines.FetchEngineBase` owns an
  `expert_states` matrix `[n_layer, n_expert]`; on a miss, `fetch()` calls
  `_prefetch()` (which experts to bring in + an exemption list) and
  `_apply_replacement_policy(n_to_evict, exemption_list)` (which residents to
  drop). `global_update(routes)` feeds per-step routing counts to frequency /
  trace policies. Existing engines: `StreamFetch` (FIFO + strided look-ahead),
  `StreamFetchLRU` (LRU + strided look-ahead), `LFUFetch` (LFU), `LFUKFetch`
  (windowed LFU).

Traces are SGLang `per_token` captures of
`Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` (~94 layers, 128 experts/layer,
top-8); see `../collect_traces_sglang.md`.

### What the simulator can and cannot model from routing traces alone

The trace records **which experts were activated per layer per token** — i.e.
the ground-truth routing decisions, after the fact. It does **not** record gate
logits, hidden states, or expert weights. This bounds what a faithful on-line
policy can use versus what only an oracle can use:

- **Modelable on-line**: recency (LRU), frequency (LFU / windowed LFU /
  precision-weighted LHU counts), FIFO, score accumulation over observed
  routes, global per-expert activation counts (model-tracing), and any
  stride / positional prefetch that reads only `(layer, expert)` indices.
- **Oracle-only (needs a predictor the trace cannot supply on-line)**: any
  policy whose *prefetch* depends on the gate output or hidden state of a
  *future* layer (Pre-gated MoE, ProMoE, HOBBIT gating-input similarity, SiDA
  hash predictor, Mixtral's speculative next-layer gate). The simulator can
  realize these only as a **trace-oracle upper bound**: let the predictor be
  "perfect" by reading the next layer's activated set straight from the trace,
  one (or `h`) layers ahead. That measures the *ceiling* of the prefetch idea,
  not the deployed predictor's accuracy — flag this wherever it applies.
- **Not modelable at all**: mixed-precision tiers, expert *substitution* /
  approximation, latency / bandwidth, and anything that changes *which* experts
  are activated (approximation changes the trace). We record these as
  out-of-scope assumptions per system.

## Taxonomy

### Eviction / admission axis

| Class | Rule | Trace-modelable? |
| --- | --- | --- |
| **LRU** | evict least-recently-used resident | yes (recency) |
| **LFU** | evict least-frequently-used resident (lifetime counts) | yes |
| **LFU-K / windowed** | LFU over a sliding window of the last `k` steps | yes |
| **LHU** (precision-aware LFU) | LFU where a hit on a low-precision copy counts less than a full-precision hit; evict lowest weighted-use | partial (weights need precision tiers; degrades to LFU without them) |
| **Score-based** | accumulate an importance score per expert (e.g. gate weight / activation magnitude); evict lowest score | partial (gate weights absent; degrades to counting / routed-frequency score) |
| **Model-tracing** | keep the globally hottest experts from an offline/online activation profile (global counts) | yes |
| **FIFO** | evict oldest-admitted resident | yes |
| **Belady / OPT** | evict the resident whose next use is farthest ahead | oracle only (bound) |
| Classics: ARC / 2Q / S3-FIFO / LRU-K / GDSF | recency+frequency hybrids | yes (optional) |

### Prefetch / prediction axis

| Class | Rule | Trace-modelable? |
| --- | --- | --- |
| **None (on-demand)** | fetch only the expert that just missed | yes |
| **Single-layer-ahead speculative** | from layer `l`'s hidden state / gate, predict layer `l+1`'s experts and prefetch | predictor absent → oracle (read `l+1` from trace) |
| **Multi-layer-ahead** | predict `l+1..l+h` via gating-input similarity across layers | predictor absent → oracle (read `l+1..l+h` from trace) |
| **Exact pre-gating** | relocate the gate so layer `l+1`'s activated set is known *before* layer `l` finishes; prefetch exactly that set | predictor absent → oracle (read `l+1` from trace) |
| **Stride-based speculative** | prefetch the next `stride` experts by index order | yes (positional) |
| **Activation-matrix / trace-based** | prefetch the experts most likely to co-activate, from an offline expert-activation matrix (EAM) | yes (build matrix from the trace prefix) |
| **Learned predictor (model / hash)** | a trained predictor / hash maps inputs → next experts, prefetched ahead | predictor absent → oracle (read trace) |
| **Prefetch-all** | stage every expert of the next layer(s) regardless of routing | yes (positional, capacity-bounded) |

### Granularity & precision

Expert-level (one `(layer, expert)` entry per cache slot) is the simulator's
native granularity and the norm (Mixtral-Offloading, MoE-Infinity, ProMoE,
Pre-gated, SiDA, ExpertFlow). Tensor / channel granularity (EdgeMoE,
Brainstorm's "Cell") and precision tiers (HOBBIT, Specter, FloE) are finer than
the trace supports and are flagged per system.

---

## Per-system entries

Each entry gives the **exact** admission / eviction / prefetch rule, the
granularity, the reported metric, and the taxonomy mapping. Identifiers are
verified against arXiv / the publisher. Where a venue is stated it is given; a
few systems are cited by their verified arXiv identifier only and flagged as
such. BibTeX is collected at the end.

### Mixtral-Offloading — Eliseev & Mazur, 2023 (arXiv:2312.17238)

- **Setting**: run Mixtral-8x7B on consumer GPUs / Colab with mixed
  quantization; experts offloaded to host RAM.
- **Eviction**: **LRU** over the GPU expert cache, applied *per MoE layer* — a
  fixed number of experts per layer are kept and the least-recently-used ones
  are evicted (the open-source `dvmazur/mixtral-offloading` `ExpertCache` keeps
  `k` experts/layer with LRU eviction groups).
- **Prefetch**: **single-layer-ahead speculative**. While layer `l` computes,
  guess layer `l+1`'s experts from the *current* hidden state (the router of
  `l+1` is applied speculatively to `l`'s output) and prefetch the guessed
  experts; on a correct guess the load is off the critical path.
- **Admission**: on-demand for the missed expert + speculatively prefetched
  guesses.
- **Granularity**: expert; experts stored in mixed quantization (HQQ).
- **Metric**: tokens/s and feasibility on 16 GB / free-tier hardware.
- **Taxonomy**: eviction = LRU (per-layer groups); prefetch =
  single-layer-ahead speculative; granularity = expert + precision (static
  quant, not a cache tier).
- **Simulator note**: LRU eviction is directly modelable. The speculative
  prefetch needs the `l+1` gate; realize it as a **one-layer-ahead trace
  oracle** (perfect guess) to upper-bound the idea. Per-layer quota (fixed
  experts/layer) can be modeled as a partitioned capacity if Task B wants to
  reproduce the group structure.

### MoE-Infinity — Xue et al., 2024 (arXiv:2401.14361) — this repo's lineage

- **Setting**: MoE inference on personal machines with a **sparsity-aware
  expert cache**; the simulator here is ported from this study's `simulation`
  code, and its hit-rate tables report **LRU / model-tracing / ours / Ideal
  (= Belady)**.
- **Prefetch**: **activation-matrix / trace-based**. MoE-Infinity builds an
  **Expert Activation Matrix (EAM)** that records, per request/sequence, how
  experts co-activate across layers; at run time it matches the in-flight
  request against EAM rows to **prefetch the experts most likely to be used
  next**, exploiting request-level sparsity and temporal locality.
- **Eviction / admission**: **activation-aware, LFU-style** — keep experts with
  high (recent) activation; admit prefetched + on-demand experts; evict the
  least-activated. Compared against LRU and a global **model-tracing** baseline
  (keep the globally hottest experts) and the Belady "Ideal".
- **Granularity**: expert.
- **Metric**: per-token decode latency and cache hit rate vs LRU /
  model-tracing / Ideal.
- **Taxonomy**: eviction = score-based / LFU-style (activation-aware); prefetch
  = activation-matrix (trace-based); granularity = expert.
- **Simulator note**: fully modelable from the trace. The EAM is exactly a
  statistic over the routing trace, so the prefetch can be built from a trace
  *prefix* (causal) for a faithful on-line policy, or from the whole trace for
  an oracle variant. The LFU-style cache is already approximated by `LFUFetch`
  / `LFUKFetch`; model-tracing is a global-count keep-hottest policy to add.

### DeepSpeed-Inference — Aminabadi et al., 2022 (arXiv:2207.00032; SC'22)

- **Setting**: ZeRO-Inference-style layer-wise parameter offloading to CPU /
  NVMe for transformers (dense + MoE), staging layers back to GPU on demand.
- **Eviction**: **LRU over layer-sized blocks**, no expert awareness — a layer
  (or parameter group) is fetched whole and the least-recently-used block is
  evicted.
- **Prefetch**: next-layer **prefetch-all** (double-buffer the upcoming layer)
  — positional, not routing-aware.
- **Granularity**: layer / parameter group (coarser than one expert).
- **Metric**: throughput / latency at scale.
- **Taxonomy**: eviction = LRU (layer granularity); prefetch = prefetch-all
  (positional); the **expert-agnostic lower-bound baseline**.
- **Simulator note**: modelable as LRU with a coarse (whole-layer) admission
  unit; serves as the floor against which expert-aware policies are compared.

### SE-MoE — Shen et al., 2022 (arXiv:2205.10034, v1 title; renamed "MoESys" in v3)

- **Setting**: distributed MoE training + single-node scalable inference when
  the model exceeds GPU memory.
- **Policy (inference)**: forms CPU + GPU memory into a **ring of sections** and
  executes across sections **round-robin**, i.e. a **prefetch-all / streaming**
  schedule that stages the whole model through GPU in order rather than caching
  hot experts. No routing-aware eviction.
- **Granularity**: parameter section (coarse).
- **Metric**: training/inference throughput vs DeepSpeed.
- **Taxonomy**: prefetch = prefetch-all (ring / streaming); eviction =
  FIFO-streaming; the **"stage everything in order"** baseline.
- **Simulator note**: the ring schedule maps to the existing `StreamFetch`
  (FIFO + strided look-ahead) family; use as a streaming lower bound.
- **Identifier note**: arXiv:2205.10034 was submitted as *SE-MoE* (v1, 2022)
  and retitled *MoESys* in v3 (2024); both names denote the same work. We cite
  the SE-MoE v1 title as in the seed list.

### Brainstorm — Cui et al., OSDI'23

- **Setting**: a compiler/runtime for dynamic NNs (incl. MoE) that makes
  sub-tensor dynamism **traceable** via `Cell` (data granularity) + `Router`
  (dispatch), then specializes execution to the observed runtime distribution.
- **Policy relevance**: not an expert cache per se, but the **ancestor of
  model-tracing** — it profiles the global activation distribution at the right
  granularity and uses it to pre-plan placement / loading of hot paths.
- **Granularity**: `Cell` (sub-tensor); finer than one expert.
- **Metric**: up to 11.7x speedup (3.29x avg) / 42% less memory on dynamic NNs.
- **Taxonomy**: eviction / placement = model-tracing (global profile);
  prefetch = profile-guided.
- **Simulator note**: the profile = global per-expert counts over the trace;
  realize as the **model-tracing keep-hottest** baseline. Sub-tensor
  granularity is out of scope.

### Pre-gated MoE — Hwang et al., 2023/2024 (arXiv:2308.12066; ISCA'24)

- **Setting**: algorithm-system co-design that **relocates the gate** so layer
  `l+1`'s router runs using layer `l`'s activations, making the next layer's
  expert set known one layer early.
- **Prefetch**: **exact pre-gating, single-layer-ahead** — prefetch *exactly*
  the experts the (relocated) gate selects for `l+1`, with no speculation
  waste. Trains the model with the pre-gating function so accuracy holds.
- **Eviction / admission**: admit the pre-gated set for `l+1`; only the current
  + next layer's experts need to be resident, so the cache is small and
  eviction is essentially "drop the layer you just finished".
- **Granularity**: expert.
- **Metric**: latency / memory vs on-demand offloading.
- **Taxonomy**: prefetch = exact pre-gating (single-layer-ahead); eviction =
  layer-scoped (drop finished layer).
- **Simulator note**: requires a *model change* (relocated gate), so it is not
  a drop-in on an unmodified trace; realize its *ceiling* as the
  **one-layer-ahead exact-prefetch oracle** (prefetch next layer's true
  activated set from the trace). Flag that the deployed accuracy depends on
  retraining, which the trace cannot capture.

### ProMoE — Song et al., 2024/2025 (arXiv:2410.22134; EuroSys'25)

- **Setting**: proactive caching for MoE serving on memory-constrained / edge
  GPUs; targets the latency cost of *reactive* cache misses.
- **Prefetch**: **learned predictor + stride-based proactive prefetch**. A
  lightweight predictor consumes intermediate results to **predict subsequent
  expert usage several layers ahead**, and a sliding-window / stride schedule
  issues prefetches ahead of the compute front so loads leave the critical
  path. Separates prediction from prefetch so mispredictions degrade gracefully.
- **Eviction / admission**: cache experts; admit predicted + on-demand; evict to
  make room for the proactive set (recency/frequency-style).
- **Granularity**: expert.
- **Metric**: 2.20x avg (up to 3.21x) prefill and 2.07x avg (up to 5.02x)
  decode speedup vs existing offloading.
- **Taxonomy**: prefetch = learned predictor + stride (multi-layer-ahead);
  eviction = recency/frequency.
- **Simulator note**: the predictor is not in the trace; realize as a
  **multi-layer-ahead trace oracle** (perfect prediction `h` layers ahead) to
  upper-bound, and separately a pure **stride** baseline (already `StreamFetch`
  with `stride>1`) for the no-predictor lower bound.

### HOBBIT — Tang et al., 2024 (arXiv:2411.01433)

- **Setting**: mixed-precision expert offloading for fast MoE inference; trade
  a little accuracy for much less load traffic.
- **Eviction**: **LHU (precision-aware LFU)** — maintain per-expert use counts
  but weight a hit on a **low-precision** resident copy differently from a
  full-precision hit; evict the lowest weighted-use expert. Low-precision
  copies act as a cheap cache tier below full precision.
- **Prefetch**: **multi-layer-ahead via gating-input similarity** — because
  consecutive layers' gating inputs are similar, predict experts several layers
  ahead from the current gating input's similarity to known patterns and
  prefetch them (at reduced precision if full precision would not fit in time).
- **Admission**: dynamic-precision — admit a low-precision copy when a
  full-precision load would stall, upgrade on reuse.
- **Granularity**: expert + **precision tiers** (full / low precision).
- **Metric**: speedup vs full-precision offloading at bounded accuracy loss.
- **Taxonomy**: eviction = LHU; prefetch = multi-layer-ahead (gating-input
  similarity); granularity = expert + precision.
- **Simulator note**: multi-layer-ahead prefetch → **`h`-layer trace oracle**.
  LHU degrades to LFU without precision tiers (the trace has no precision), so
  model a **single-tier LFU** as the faithful approximation and record that the
  precision weighting / mixed-precision admission is out of scope.

### SiDA-MoE — Du et al., 2023/2024 (arXiv:2310.18859; MLSys'24)

- **Setting**: sparsity-inspired, **data-aware** serving — exploit that for a
  given input only a small, *predictable* subset of experts fires.
- **Prefetch / admission**: a **learned hash predictor**, trained offline,
  maps the input (token/feature hash) to the experts it will activate across
  layers; SiDA **prefetches exactly those experts** and keeps only them
  resident (dynamically loads/offloads per input).
- **Eviction**: offload experts the predictor says won't fire for the current
  input (data-aware, not recency/frequency).
- **Granularity**: expert.
- **Metric**: throughput / memory with near-lossless accuracy from high
  predictor recall.
- **Taxonomy**: prefetch = learned predictor (hash); eviction = data-aware
  (predicted-inactive).
- **Simulator note**: the hash predictor is external; its *ceiling* is the
  **per-input trace oracle** — keep exactly the experts the trace says this
  input activates (equivalent to a perfect data-aware predictor). Flag that
  real recall < 1, so the oracle over-states SiDA.

### FineMoE — Yu et al., 2025 (arXiv:2502.05370)

- **Setting**: "Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via
  Fine-Grained Expert Offloading" — offload at **sub-expert (fine-grained)**
  units to tune the latency/memory trade-off, and **benchmarks LRU / LFU /
  ProMoE** as baselines.
- **Policy**: fine-grained (partial-expert) admission + a locality-aware
  cache; its *comparison table* is the useful artifact here (LRU / LFU /
  ProMoE on offloaded MoE).
- **Granularity**: sub-expert (finer than the simulator's one-entry = one
  expert).
- **Taxonomy**: eviction = LRU / LFU (baselines) + fine-grained locality;
  prefetch = per its ProMoE-style predictor.
- **Simulator note**: use chiefly as a **source of baseline definitions**
  (LRU / LFU / ProMoE) and to cross-check hit-rate trends. Sub-expert
  granularity is out of scope for the one-expert-per-entry cache.

### Supporting / comparison systems (classified, not all shortlisted)

- **EdgeMoE — Yi et al., 2023 (arXiv:2308.14352)**: expert offloading on mobile;
  holds non-expert weights resident and caches hot experts with **channel-wise
  bit-width adaptation** (precision granularity) + preloading. Taxonomy:
  eviction = frequency/score at channel granularity; prefetch = preload hot
  experts. Precision/channel granularity out of scope.
- **Fiddler — Kamahori et al., 2024 (arXiv:2402.07033; ICLR'25)**: CPU-GPU
  orchestration that **runs cold experts on the CPU** instead of moving weights,
  minimizing data movement. Taxonomy: not a cache eviction policy — a
  compute-placement decision; keep hot experts on GPU, compute the rest on CPU.
  Placement is out of scope (the simulator models residency, not compute site).
- **SwapMoE — Kong et al., 2023/2024 (arXiv:2308.15030; ACL'24)**: keeps a small
  dynamic set of "important experts" resident under a tunable memory budget and
  swaps the rest. Taxonomy: eviction = score-based (importance) + model-tracing;
  prefetch = on-demand swap. Modelable as a score/keep-hottest policy.
- **MoE-Lightning — Cao et al., 2024 (arXiv:2411.11217)**: throughput-oriented
  CPU-GPU pipelining (CGOPipe) + a roofline-guided schedule; batches and
  overlaps rather than caching hot experts. Taxonomy: prefetch = pipelined
  prefetch-all; eviction = schedule-driven. Pipeline/overlap out of scope.
- **ExpertFlow — He et al., 2024 (arXiv:2410.17954; DAC'26,
  DOI:10.1145/3770743.3804292)**: **predictive expert caching + token
  scheduling** — a routing-path predictor prefetches experts and reorders tokens
  to raise cache reuse. Taxonomy: prefetch = learned predictor; eviction =
  predicted-reuse. Predictor → trace oracle; token reordering out of scope.
- **Klotski — Fang et al., 2025 (arXiv:2502.06888)**: expert-aware **multi-batch
  pipeline** that reorders expert computation across batches to hide loading.
  Taxonomy: schedule-level; residency choices expert-aware. Pipeline out of
  scope; expert-hotness reuse is the cacheable part.
- **FloE — Zhou et al., 2025 (arXiv:2505.05950; ICML'25)**: on-the-fly MoE on
  memory-constrained GPU with **compressed/approximated experts** loaded lazily.
  Taxonomy: precision-tier admission + on-demand. Compression changes the trace
  → out of scope except as an on-demand baseline.
- **Lina — Li et al., 2022/2023 (arXiv:2210.17223; ATC'23)**: prioritizes hot
  experts in all-to-all and schedules resources by **measured expert
  popularity**. Taxonomy: model-tracing (popularity). Distributed scheduling
  mostly out of scope; popularity = global counts.
- **DeepSpeed-MoE — Rajbhandari et al., 2022 (arXiv:2201.05596; ICML'22)**:
  MoE inference/training system (PR-MoE, expert/tensor/data parallel serving).
  Taxonomy: parallel placement, not a cache; context/baseline only.

### Comparison studies mined for definitions

- **SpecMD — Hoang et al., 2026 (arXiv:2602.03921)**: "A Comprehensive Study On
  Speculative Expert Prefetching." Use for crisp definitions of the
  **speculative prefetch cascade** (predict `l+1..l+h` and fetch in a cascade,
  stopping on a resident hit) and how prediction depth trades accuracy against
  lead time. Grounds the **fetch-cascade** prefetch variant and the
  multi-layer-ahead oracle depth sweep.
  *Identifier note*: arXiv:2602.03921 (Feb 2026); verified via the arXiv API.
- **Specter — Wang et al., 2025 (arXiv:2511.14102)**: "Fast MoE Inference via
  Quantization-Enabled Multi-Step Speculative Offloading." Use for
  **expert substitution** (serve a *quantized/approximate* expert when the exact
  one is not resident, instead of stalling) and **score-based eviction** tied to
  multi-step speculation. Substitution changes outputs → out of scope for the
  trace, but its *eviction score* definition is reusable.

---

## Shortlist: implementable baselines for Task B

Six-plus baselines spanning **both axes**. Each gives an implementation-ready
rule against `FetchEngineBase` (or the `CachePolicy.on_access` ABC) and a
predictor note. The eviction-only baselines pair with on-demand (no) prefetch;
the prefetch baselines pair with LRU eviction unless stated, so the two axes are
covered independently and in combination.

### Eviction axis

**B1. LRU + on-demand** *(floor; = DeepSpeed-Inference expert-level, Mixtral
eviction).* Admit only the missed expert. On a full cache, evict the resident
with the oldest last-access timestamp. No prefetch. **Predictor: none.** Already
present as `StreamFetchLRU` with `stride=1`; expose as the canonical LRU.

**B2. LFU (lifetime) + on-demand** *(= MoE-Infinity / SwapMoE frequency core).*
Keep a lifetime activation count per `(layer, expert)` via `global_update`.
Admit the missed expert; evict the lowest-count resident (ties: oldest).
**Predictor: none.** Present as `LFUFetch`; keep as a baseline.

**B3. Windowed LFU (LFU-K) + on-demand.** As B2 but counts decay to the last
`k` routing steps (sliding window), capturing phase changes. Evict the lowest
windowed-count resident. **Predictor: none.** Present as `LFUKFetch`; sweep `k`.

**B4. Model-tracing "keep-hottest" (global counts) + on-demand** *(=
Brainstorm / Lina / MoE-Infinity's model-tracing baseline).* Precompute (or
accumulate online from the trace prefix) a **global per-expert activation
count** across all layers/requests; pin the top-`capacity` hottest experts and
only stream the remainder on demand, evicting any non-pinned resident first,
then the globally coldest. **Predictor: none** (a static global statistic). New
engine for Task B. Offline-global vs online-prefix are two variants; the
offline-global one is a mild oracle (uses the whole trace's counts) and should
be labelled as such.

**B5. Score-based / LHU-degraded (precision-free) + on-demand** *(= HOBBIT LHU
without precision tiers; SwapMoE importance).* Maintain a decaying importance
**score** per expert incremented on each activation (optionally weighted by how
many of the token's top-k slots it occupied in that layer, which *is* in the
trace); evict the lowest score. **Predictor: none.** New engine. Flag that true
LHU weights full- vs low-precision hits, which the trace cannot represent, so
this is the single-precision degradation.

### Prefetch axis (paired with LRU eviction unless noted)

**B6. Stride / FIFO-streaming prefetch** *(= SE-MoE ring, ProMoE's
no-predictor stride, DeepSpeed prefetch-all).* On a miss, fetch the missed
expert plus the next `stride` experts by `(layer, expert)` index order;
exempt them from eviction this step; evict by FIFO (or LRU). **Predictor:
none** (positional). Present as `StreamFetch` (FIFO) / `StreamFetchLRU` with
`stride>1`; sweep `stride`.

**B7. Activation-matrix (EAM) prefetch** *(= MoE-Infinity, the faithful
trace-based policy — no external predictor needed).* From the trace **prefix**
seen so far, build a co-activation statistic: for each expert, the experts most
likely to activate in the next layer(s). On an access, prefetch the top-`m`
co-activating experts for the upcoming layer; exempt them; evict by LRU/LFU.
**Predictor: built from the trace itself** (causal → faithful on-line; whole
trace → oracle variant). New engine; this is the headline MoE-Infinity-style
baseline.

**B8. One-layer-ahead exact-prefetch oracle** *(= Pre-gated MoE ceiling,
Mixtral speculative ceiling).* On entering layer `l`, read layer `l+1`'s
**true** activated expert set from the trace and prefetch exactly it; exempt;
evict by LRU. **Predictor: oracle** (perfect single-layer-ahead). Labels the
ceiling of exact pre-gating / perfect speculation; the deployed systems differ
only by predictor accuracy (Pre-gated: retrained gate; Mixtral: speculative
guess). New engine.

**B9. Multi-layer-ahead (`h`) prefetch oracle + fetch cascade** *(= ProMoE /
HOBBIT / SpecMD ceiling).* As B8 but read layers `l+1..l+h` ahead and prefetch
their true activated sets in a cascade, stopping early when the cache is full;
sweep depth `h`. **Predictor: oracle** (perfect `h`-ahead). Upper-bounds
multi-layer speculative prefetch and shows the hit-rate vs lead-time trade-off
SpecMD studies. New engine.

**B10. Data-aware per-input oracle** *(= SiDA / ExpertFlow ceiling).* For each
request, keep exactly the experts that request activates across all layers
(read from the trace), evicting everything else at request boundaries.
**Predictor: oracle** (perfect data-aware predictor; real recall < 1). Bounds
hash/learned-predictor caching. New engine; optional.

**Coverage check.** Eviction axis: B1 LRU, B2 LFU, B3 windowed LFU, B4
model-tracing, B5 score/LHU-degraded — five distinct eviction rules. Prefetch
axis: B6 stride/streaming (no predictor), B7 EAM trace-based (self-predicted),
B8 single-layer oracle, B9 multi-layer `h` oracle + cascade, B10 per-input
oracle — five distinct prefetch rules. Belady from `belady_simulate` is the
ceiling for all. This is ≥6 across both axes, each precise enough to implement
without further reading.

---

## BibTeX

```bibtex
@article{eliseev2023mixtraloffloading,
  title   = {Fast Inference of Mixture-of-Experts Language Models with Offloading},
  author  = {Eliseev, Artyom and Mazur, Denis},
  journal = {arXiv preprint arXiv:2312.17238},
  year    = {2023},
  note    = {Technical report; code: dvmazur/mixtral-offloading},
  doi     = {10.48550/arXiv.2312.17238}
}

@article{xue2024moeinfinity,
  title   = {{MoE-Infinity}: Efficient {MoE} Inference on Personal Machines with Sparsity-Aware Expert Cache},
  author  = {Xue, Leyang and Fu, Yao and Lu, Zhan and Mai, Luo and Marina, Mahesh},
  journal = {arXiv preprint arXiv:2401.14361},
  year    = {2024},
  doi     = {10.48550/arXiv.2401.14361}
}

@inproceedings{aminabadi2022deepspeedinference,
  title     = {{DeepSpeed-Inference}: Enabling Efficient Inference of Transformer Models at Unprecedented Scale},
  author    = {Aminabadi, Reza Yazdani and Rajbhandari, Samyam and Zhang, Minjia and Awan, Ammar Ahmad and Li, Conglong and Li, Du and Zheng, Elton and Rasley, Jeff and Smith, Shaden and Ruwase, Olatunji and He, Yuxiong},
  booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '22)},
  year      = {2022},
  doi       = {10.48550/arXiv.2207.00032}
}

@article{shen2022semoe,
  title   = {{SE-MoE}: A Scalable and Efficient Mixture-of-Experts Distributed Training and Inference System},
  author  = {Shen, Liang and Wu, Zhihua and Gong, WeiBao and Hao, Hongxiang and Bai, Yangfan and Wu, HuaChao and Wu, Xinxuan and Xiong, Haoyi and Yu, Dianhai and Ma, Yanjun},
  journal = {arXiv preprint arXiv:2205.10034},
  year    = {2022},
  note    = {Retitled ``MoESys'' in v3 (2024)},
  doi     = {10.48550/arXiv.2205.10034}
}

@inproceedings{cui2023brainstorm,
  title     = {Optimizing Dynamic Neural Networks with {Brainstorm}},
  author    = {Cui, Weihao and Han, Zhenhua and Ouyang, Lingji and Wang, Yichuan and Zheng, Ningxin and Ma, Lingxiao and Yang, Yuqing and Yang, Fan and Xue, Jilong and Qiu, Lili and Zhou, Lidong and Chen, Quan and Tan, Haisheng and Guo, Minyi},
  booktitle = {17th USENIX Symposium on Operating Systems Design and Implementation (OSDI '23)},
  pages     = {797--815},
  year      = {2023},
  isbn      = {978-1-939133-34-2},
  publisher = {USENIX Association},
  url       = {https://www.usenix.org/conference/osdi23/presentation/cui}
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

@inproceedings{du2024sidamoe,
  title     = {{SiDA}: Sparsity-Inspired Data-Aware Serving for Efficient and Scalable Large Mixture-of-Experts Models},
  author    = {Du, Zhixu and Li, Shiyu and Wu, Yuhao and Jiang, Xiangyu and Sun, Jingwei and Zheng, Qilin and Wu, Yongkai and Li, Ang and Li, Hai and Chen, Yiran},
  booktitle = {Proceedings of Machine Learning and Systems (MLSys '24)},
  year      = {2024},
  note      = {arXiv:2310.18859},
  doi       = {10.48550/arXiv.2310.18859}
}

@article{yu2025finemoe,
  title   = {Taming Latency-Memory Trade-Off in {MoE}-Based {LLM} Serving via Fine-Grained Expert Offloading},
  author  = {Yu, Hanfei and Cui, Xingqi and Zhang, Hong and Wang, Hao and Wang, Hao},
  journal = {arXiv preprint arXiv:2502.05370},
  year    = {2025},
  doi     = {10.48550/arXiv.2502.05370}
}

@article{yi2023edgemoe,
  title   = {{EdgeMoE}: Empowering Sparse Large Language Models on Mobile Devices},
  author  = {Yi, Rongjie and Guo, Liwei and Wei, Shiyun and Zhou, Ao and Wang, Shangguang and Xu, Mengwei},
  journal = {arXiv preprint arXiv:2308.14352},
  year    = {2023},
  doi     = {10.48550/arXiv.2308.14352}
}

@inproceedings{kamahori2024fiddler,
  title     = {{Fiddler}: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models},
  author    = {Kamahori, Keisuke and Tang, Tian and Gu, Yile and Zhu, Kan and Kasikci, Baris},
  booktitle = {International Conference on Learning Representations (ICLR '25)},
  year      = {2025},
  note      = {arXiv:2402.07033},
  doi       = {10.48550/arXiv.2402.07033}
}

@inproceedings{kong2024swapmoe,
  title     = {{SwapMoE}: Serving Off-the-shelf {MoE}-based Large Language Models with Tunable Memory Budget},
  author    = {Kong, Rui and Li, Yuanchun and Feng, Qingtian and Wang, Weijun and Ye, Xiaozhou and Ouyang, Ye and Kong, Linghe and Liu, Yunxin},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL '24)},
  year      = {2024},
  note      = {arXiv:2308.15030},
  doi       = {10.48550/arXiv.2308.15030}
}

@article{cao2024moelightning,
  title   = {{MoE-Lightning}: High-Throughput {MoE} Inference on Memory-constrained GPUs},
  author  = {Cao, Shiyi and Liu, Shu and Griggs, Tyler and Schafhalter, Peter and Liu, Xiaoxuan and Sheng, Ying and Gonzalez, Joseph E. and Zaharia, Matei and Stoica, Ion},
  journal = {arXiv preprint arXiv:2411.11217},
  year    = {2024},
  doi     = {10.48550/arXiv.2411.11217}
}

@inproceedings{he2024expertflow,
  title     = {{ExpertFlow}: Efficient Mixture-of-Experts Inference via Predictive Expert Caching and Token Scheduling},
  author    = {He, Xin and Zhang, Shunkang and Tang, Kaijie and Shi, Shaohuai and Wang, Yuxin and Zeng, Zihao},
  booktitle = {Proceedings of the 63rd ACM/IEEE Design Automation Conference (DAC '26)},
  year      = {2026},
  note      = {arXiv:2410.17954},
  doi       = {10.1145/3770743.3804292}
}

@article{fang2025klotski,
  title   = {{Klotski}: Efficient Mixture-of-Expert Inference via Expert-Aware Multi-Batch Pipeline},
  author  = {Fang, Zhiyuan and Huang, Yuegui and Hong, Zicong and Lyu, Yufeng and Chen, Wuhui and Yu, Yue and Yu, Fan and Zheng, Zibin},
  journal = {arXiv preprint arXiv:2502.06888},
  year    = {2025},
  doi     = {10.48550/arXiv.2502.06888}
}

@inproceedings{zhou2025floe,
  title     = {{FloE}: On-the-Fly {MoE} Inference on Memory-constrained {GPU}},
  author    = {Zhou, Yuxin and Li, Zheng and Zhang, Jun and Wang, Jue},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML '25)},
  year      = {2025},
  note      = {arXiv:2505.05950},
  doi       = {10.48550/arXiv.2505.05950}
}

@inproceedings{li2023lina,
  title     = {Accelerating Distributed {MoE} Training and Inference with {Lina}},
  author    = {Li, Jiamin and Jiang, Yimin and Zhu, Yibo and Wang, Cong and Xu, Hong},
  booktitle = {2023 USENIX Annual Technical Conference (USENIX ATC '23)},
  year      = {2023},
  note      = {arXiv:2210.17223},
  doi       = {10.48550/arXiv.2210.17223}
}

@inproceedings{rajbhandari2022deepspeedmoe,
  title     = {{DeepSpeed-MoE}: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale},
  author    = {Rajbhandari, Samyam and Li, Conglong and Yao, Zhewei and Zhang, Minjia and Aminabadi, Reza Yazdani and Awan, Ammar Ahmad and Rasley, Jeff and He, Yuxiong},
  booktitle = {Proceedings of the 39th International Conference on Machine Learning (ICML '22)},
  year      = {2022},
  note      = {arXiv:2201.05596},
  doi       = {10.48550/arXiv.2201.05596}
}

@article{hoang2026specmd,
  title   = {{SpecMD}: A Comprehensive Study On Speculative Expert Prefetching},
  author  = {Hoang, Duc and Jaiswal, Ajay and Samragh, Mohammad and Cho, Minsik},
  journal = {arXiv preprint arXiv:2602.03921},
  year    = {2026},
  doi     = {10.48550/arXiv.2602.03921}
}

@article{wang2025specter,
  title   = {{Specter}: Fast {MoE} Inference via Quantization-Enabled Multi-Step Speculative Offloading},
  author  = {Wang, Wenfeng and Hou, Xiaofeng and Xia, Xinfeng and Wang, Xinkai},
  journal = {arXiv preprint arXiv:2511.14102},
  year    = {2025},
  doi     = {10.48550/arXiv.2511.14102}
}
```

### Verification notes

- arXiv identifiers for Mixtral-Offloading, MoE-Infinity, DeepSpeed-Inference,
  SE-MoE/MoESys, Pre-gated, ProMoE, HOBBIT, SiDA, FineMoE, EdgeMoE, Fiddler,
  SwapMoE, MoE-Lightning, ExpertFlow, Klotski, FloE, Lina, DeepSpeed-MoE,
  SpecMD, and Specter were resolved against the arXiv API; titles/first authors
  match the text above. Brainstorm's title, authors, pages, and BibTeX are from
  the USENIX OSDI'23 proceedings page.
- **SE-MoE (2205.10034)**: the seed list labelled this "SE-MoE, Shen et al.,
  2022"; the arXiv record's **v1** title is *SE-MoE* and **v3** renames it
  *MoESys*. Both are the same paper; we keep the SE-MoE title per the seed.
- Author lists beyond the first few and some venue fields (ISCA'24, EuroSys'25,
  MLSys'24, ICLR'25, ACL'24, ICML'25, ATC'23, DAC'26) are taken from the
  published records; where a paper is cited by arXiv only, the BibTeX carries
  the arXiv id as the authoritative identifier and `note`/`booktitle` records
  the venue. No numbers, algorithms, or titles were invented; mechanisms the
  trace cannot model are flagged per entry.
