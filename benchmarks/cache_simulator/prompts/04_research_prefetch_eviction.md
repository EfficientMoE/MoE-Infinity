# Sub-agent prompt D - survey expert prefetchers and prefetch-aware eviction

Copy into a fresh research agent. It surveys how MoE offloading systems
prefetch experts and how classic caching theory couples prefetching with
replacement, and turns both into an implementable shortlist for prompt E. It
does **not** write simulator code.

See `../auto_research_prefetch_eviction.md` for the plan and guardrails.

## CONTEXT

The simulator (`benchmarks/cache_simulator/`) replays real batch-1 decode
routing of Qwen3-235B-A22B-FP8 (94 MoE layers x 128 experts, top-8) and
Qwen3-30B-A3B (48 x 128, top-8). Round 1 results
(`research/decode_policy_study.md`): LCNU is the best online evictor; perfect
knowledge of the next 8 layers' routing adds at most 0.7 points of demand hit
rate, so routing prediction should feed prefetch. The round-1 survey
(`research/policy_survey.md`) already covers MoE-Infinity, Mixtral-Offloading,
Pre-gated MoE, ProMoE, HOBBIT, SiDA-MoE, ExpertFlow and others from the
eviction angle; reuse it, do not duplicate it.

Use the available skills: `arxiv-search`, `arxiv-analyze`, `citation-convert`,
`research`, `dispatching-parallel-agents`. Verify every identifier.

## TASK

Write `benchmarks/cache_simulator/research/prefetch_survey.md`:

1. **MoE expert prefetchers.** For each system (seed list below; expand it):
   the predictor input (gate of the next layer applied to the current hidden
   state, hidden-state similarity, activation matrix / EAM, request-level
   history, learned model, hash), how many layers ahead, the prefetch budget,
   where prefetched experts are admitted (same cache, staging buffer,
   lower precision), whether eviction knows about prefetches, the reported
   prediction accuracy (recall@k) and the metric used. Flag which predictors
   can be rebuilt **online from routing traces alone** (no hidden states) and
   which need model internals (then only an oracle bound is possible).
2. **Prefetching + caching theory.** Primary sources on integrated
   prefetching and caching and prefetch-aware replacement, e.g. Cao, Felten,
   Karlin, Li, "A Study of Integrated Prefetching and Caching Strategies"
   (SIGMETRICS 1995: optimal prefetching, optimal replacement, do no harm,
   first opportunity); Patterson et al., "Informed Prefetching and Caching"
   (SOSP 1995); Wu et al., "PACMan: Prefetch-Aware Cache Management for High
   Performance Caching" (MICRO 2011); plus anything newer you verify. Extract
   the exact admission / replacement rules and what they assume.
3. **Shortlist for prompt E**: at least 4 online prefetchers implementable
   from routing traces (with the precise rule, inputs, horizon, budget) and at
   least 3 admission modes for prefetched experts (same-queue, probation,
   separate buffer, do-no-harm using a next-use estimate), each with a rule
   precise enough to implement directly.

Seed list (verify, then expand): MoE-Infinity (arXiv:2401.14361), ProMoE
(arXiv:2410.22134), HOBBIT (arXiv:2411.01433), Pre-gated MoE
(arXiv:2308.12066), Mixtral-Offloading (arXiv:2312.17238), ExpertFlow
(arXiv:2410.17954), SiDA-MoE (arXiv:2310.18859), AdapMoE, SwapMoE
(arXiv:2308.15030), Klotski (arXiv:2502.06888), FineMoE / fMoE
(arXiv:2502.05370), SpecMD (arXiv:2602.03921), MoE-Lightning
(arXiv:2411.11217).

## DELIVERABLES

- `research/prefetch_survey.md` with BibTeX and the shortlist.
- A short summary for prompt E/F in the PR description.

## ACCEPTANCE

- Every citation verified (arXiv API / proceedings); no invented numbers.
- Shortlist rules are implementable without model internals, or explicitly
  marked oracle-only.
