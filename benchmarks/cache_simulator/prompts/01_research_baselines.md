# Sub-agent prompt A — research MoE/offloading cache policies into baselines

Copy into a fresh research agent. It mines the literature for MoE expert
cache/prefetch/eviction strategies and turns them into a survey + an
implementable baseline shortlist for the cache simulator. It does **not** write
simulator code (that is prompt B).

See `../auto_research_policies.md` for the overall plan and shared guardrails.

## CONTEXT

The offline expert-cache simulator in `benchmarks/cache_simulator/` (repo
`EfficientMoE/MoE-Infinity`, PR #288) already has Belady-optimal (the "Ideal"
upper bound) plus LRU / LFU / LFU-K / FIFO-streaming policies (`fetch_engines.py`,
`cache_strategy.py`). Traces come from SGLang `per_token` captures of
`Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` (≈94 layers, 128 experts/layer, top-8);
see `../collect_traces_sglang.md`.

Use the available skills: `arxiv-search`, `arxiv-analyze`, `citation-convert`,
`research`, and `dispatching-parallel-agents` (fan out independent paper reads).
**Cite primary sources (arXiv / proceedings). Never fabricate titles, authors,
venues, numbers, or algorithms** — verify each id before using it.

## TASK

Produce `benchmarks/cache_simulator/research/policy_survey.md`: one entry per
system, with a BibTeX citation (via `citation-convert`), the caching strategy,
and its mapping to the taxonomy below. Then a **shortlist of ≥6 implementable
baselines** spanning both the eviction and prefetch axes, each with a precise
decision rule prompt B can implement directly.

### Taxonomy (classify every paper on these axes)

- **Eviction / admission**: LRU, LFU, LFU-K / windowed, **LHU** (precision-aware
  LFU), **score-based** (accumulated activation / importance), **model-tracing**
  (global expert counts), Belady/OPT (bound). Optional classics: ARC, 2Q,
  S3-FIFO, LRU-K, GDSF.
- **Prefetch / prediction**: none (on-demand); **single-layer-ahead speculative**
  (predict next layer from current hidden states / gate); **multi-layer-ahead**
  (gating-input similarity); **stride-based** speculative; **activation-matrix /
  trace-based**; **learned predictor** (separate model / hash).
- **Granularity**: expert / layer / tensor; precision tiers.
- **Setup & metric**: reported hit-rate / latency / memory, model family, cache
  budget units.

For each paper, extract the **exact** rule: what is admitted; what is evicted on
a miss when the cache is full; what/when is prefetched; at what granularity. Flag
any assumption the simulator cannot model from routing traces alone.

### Seed bibliography (VERIFY ids + EXPAND — do not trust blindly)

- **Mixtral-Offloading** — Eliseev & Mazur, 2023 (arXiv:2312.17238). LRU +
  single-layer-ahead speculative prefetch. Open source `dvmazur/mixtral-offloading`
  (`ExpertCache`, layer eviction groups).
- **MoE-Infinity** — Xue et al., 2024 (arXiv:2401.14361). Activation-aware (EAM)
  prefetch + caching; LFU-style; reports LRU / model-tracing / ours / Ideal
  (=Belady) tables. This repo's lineage.
- **DeepSpeed-Inference** — Aminabadi et al., SC'22. Layer-wise offload, LRU, no
  expert awareness (lower-bound baseline).
- **BrainStorm** — Cui et al., OSDI'23. Model-tracing caching (global counts).
- **SE-MoE** — Shen et al., 2022 (arXiv:2205.10034). "prefetch-all" baseline.
- **Pre-gated MoE** — Hwang et al., ISCA'24 (arXiv:2308.12066). Predictive
  pre-gating → prefetch only activated experts one layer ahead.
- **ProMoE** — Song et al., 2024 (arXiv:2410.22134). Stride-based proactive
  prefetch + learned predictor.
- **HOBBIT** — Tang et al., 2024 (arXiv:2411.01433). Mixed-precision offloading;
  **LHU** eviction; multi-layer-ahead prefetch via gating-input similarity.
- **SiDA-MoE** — Du et al., MLSys'24. Data-aware hash predictor.
- **FineMoE** — 2025 (arXiv:2502.05370). Fine-grained offload; benchmarks
  LRU / LFU / ProMoE.
- Also classify: **EdgeMoE** (arXiv:2308.14352), **Fiddler** (arXiv:2402.07033),
  **SwapMoE** (ACL'24), **MoE-Lightning** (arXiv:2411.11217), **ExpertFlow**
  (arXiv:2410.17954), **Klotski**, **FloE** (ICML'25), **Lina** (ATC'23),
  **DeepSpeed-MoE** (ICML'22). Mine the comparison studies **SpecMD**
  (arXiv:2602.03921) and **Specter** (arXiv:2511.14102) for "fetch cascade",
  "expert substitution", and score-based eviction definitions.

## DELIVERABLES

1. `research/policy_survey.md`: taxonomy table + per-paper rules + BibTeX.
2. A shortlist section naming ≥6 baselines, each with a one-paragraph,
   implementation-ready rule (admission / eviction / prefetch / granularity) and
   a note on whether it needs a predictor (and how to realize it as a
   trace-oracle upper bound).
3. Append your findings as rows to `research/EXPERIMENTS.md` (policy, source,
   taxonomy class) so prompt B can pick them up.

## ACCEPTANCE

- Every cited paper has a verified identifier and a BibTeX entry.
- The shortlist spans both eviction and prefetch axes and is precise enough that
  prompt B needs no further literature reading to implement it.
- No fabricated content; assumptions the simulator cannot model are flagged.

Commit the survey (markdown) on the shared feature branch per repo conventions
(Angular commit `-s`, CHANGELOG untouched for a docs-only research artifact
unless you add a new guide). Dispatch independent paper reads to parallel
sub-agents; synthesize into one survey.
