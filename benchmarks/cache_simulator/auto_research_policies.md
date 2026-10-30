# Autonomous caching-policy research + simulator acceleration (overview)

Orchestration hub for improving the expert-cache simulator in
`benchmarks/cache_simulator/` (repo `EfficientMoE/MoE-Infinity`, PR #288) on two
fronts:

1. mine MoE / offloading papers for cache / prefetch / eviction strategies and
   turn them into runnable **baselines** benchmarked against the Belady-optimal
   bound, and
2. **accelerate** the simulator with parallelism / GPU.

Dispatch the three self-contained sub-agent prompts below (each restates the
context it needs). They can run as parallel sub-agents — mind the dependency
graph — but all code lands through **one reviewed branch / PR**.

## Prompts

| Prompt | Task |
| --- | --- |
| [`prompts/01_research_baselines.md`](./prompts/01_research_baselines.md) | **A** — literature → `research/policy_survey.md` + baseline shortlist (verified citations, strategy taxonomy). No simulator code. |
| [`prompts/02_implement_baselines.md`](./prompts/02_implement_baselines.md) | **B** — implement the shortlist against the policy interface; unit-test; benchmark vs Belady; MRC tables/plots. |
| [`prompts/03_accelerate_simulator.md`](./prompts/03_accelerate_simulator.md) | **C** — profile → vectorize → one-pass MRC → parallel → optional GPU, gated bit-for-bit against the pure-Python oracle. |

## Dependency graph & dispatch

- **A → B**: B consumes A's shortlist, but can start on the already-known
  policies (LRU / LFU / model-tracing / score-based / single-layer speculative)
  while A finishes the long tail.
- **C ∥ A, B**: C optimizes the existing simulator independently and can start
  immediately. C's one-pass MRC engine is also what B's capacity sweep should
  call, so B and C must agree on the shared contract below.
- Use `dispatching-parallel-agents` to fan out independent paper reads (A) and
  independent policy implementations (B); land everything on one branch.

## Shared contracts (so B and C don't collide)

- **Policy interface**: extend `FetchEngineBase` (prefetch + eviction) or add a
  small `CachePolicy` ABC (`on_access(key) -> hit`); one shared harness
  (`research/run_policies.py`) runs any policy over an access sequence →
  `(hits, misses, hit_rate)`.
- **MRC engine** (owned by C): `miss_ratio_curve(accesses, capacities) ->
  {capacity: hit_rate}`, which B's sweep calls. It must match the reference
  simulator exactly.
- **Experiment log**: append-only `benchmarks/cache_simulator/research/EXPERIMENTS.md`
  (one row per policy × model × capacity: hit rate, source, commit).

## Shared guardrails (all three agents)

- Cite **primary sources** (arXiv / proceedings); never fabricate titles,
  authors, numbers, or algorithms; attach the command + environment for every
  number you report.
- Keep the pure-Python `belady_simulate` / `ExpertCache` as the **correctness
  oracle**; accelerated or approximate paths gate bit-for-bit (or within a
  stated error bound) against it and the existing fuzz / brute-force-OPT tests.
- Repo conventions: feature branch, Angular commits with `-s`, PR template, link
  issues, `CHANGELOG.md` `Unreleased`, pre-commit (ruff / ruff-format /
  codespell), tests under `tests/python/unit/`, torch imported lazily. **Ask
  before adding heavy dependencies.**
- Git-ignore trace data and large figures; commit code, specs, the survey, and
  small result tables / plot sources.

## Overall acceptance

1. `research/policy_survey.md` covers the seed list (verified) + additions, each
   mapped to the taxonomy with BibTeX (A).
2. ≥6 baselines implemented, unit-tested, each satisfying
   `hit_rate(policy) <= hit_rate(Belady)` on every trace (B).
3. `run_policies.py` reproduces a policy × capacity hit-rate table + MRC plots on
   the Qwen3-235B traces, with Belady as the ceiling (B).
4. Accelerated simulator matches the reference bit-for-bit on the fuzz / OPT
   oracle, with a documented speedup (and error bound for any approximation) (C).
5. CHANGELOG updated; tests green in CI.

Related runbook: [`collect_traces_sglang.md`](./collect_traces_sglang.md)
(capture the traces these agents consume).
