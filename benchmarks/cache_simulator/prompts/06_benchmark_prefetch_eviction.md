# Sub-agent prompt F - real-trace prefetch x eviction sweep + recommendations

Copy into a fresh analysis agent after prompt E's harness exists. It runs the
combination sweep on the real traces, answers the round-2 research questions
and proposes the engine knobs.

See `../auto_research_prefetch_eviction.md` for the plan, contracts and
guardrails.

## CONTEXT

- Harness and timing model from prompt E; prefetcher shortlist from prompt D
  (`research/prefetch_survey.md`).
- Baseline evictor results and protocol: `research/decode_policy_study.md`,
  `research/recommendations.md`, `research/lcnu_ablation.py`.
- Traces (git-ignored): `/mnt/raid0nvme0/leyang/moeinf-traces/belady_qwen3_235b/*.pt`
  (100 requests), `belady_qwen3_30b/*.pt` (150), merged
  `belady_<model>_stream/stream.pt`.

## TASK

1. **Tune** each prefetcher's horizon / budget and each admission mode's
   parameters on the first half of the Qwen3-30B-A3B requests; report on the
   second half and on Qwen3-235B without retuning.
2. **Sweep** evictor {LRU, LFU, LIRS, S3-FIFO, W-TinyLFU, per-layer LRU, LCNU}
   x prefetcher {none, each online prefetcher, oracle next-1 and next-h}
   x admission {same_queue, probation, buffer, do_no_harm} x capacity (from
   below one token's working set to half the key space), on decode-only and
   on the prefill-union stream. Metrics: demand hit rate, prefetch coverage,
   wasted prefetches, transfers per token, timing-model stall per token.
3. **Answer** the round-2 questions:
   - separability: is `best(pair)` close to `best evictor + best prefetcher`
     chosen independently? Quantify the interaction (e.g. two-way ANOVA-style
     decomposition of stall over evictor x prefetcher, or regret of the
     independent choice vs the joint optimum at each capacity);
   - the best pair and admission mode per cache-size regime;
   - when prefetching hurts (bandwidth wasted, useful experts evicted) and
     whether do-no-harm admission removes that;
   - whether prefetch changes which evictor is best.
4. **Recommend** engine knobs: separate `expert_eviction_policy` and
   `expert_prefetch_policy` (or not), the admission rule, defaults per regime,
   and what the engine needs at runtime for each (signals, state, cost).

## DELIVERABLES

- `research/prefetch_eviction_study.md`: method, tables (compact per regime
  plus full CSVs), the answers, limitations, reproduce commands.
- Update `research/recommendations.md` with the prefetch + admission advice.
- Rows appended to `research/EXPERIMENTS.md`.

## ACCEPTANCE

- Every number comes from a committed command on the real traces; tuning and
  reporting splits kept separate; demand hit rate <= Belady everywhere.
- A clear yes/no on separating the two policies, with the evidence, and a
  recommended default pair + admission mode per cache-size regime.
