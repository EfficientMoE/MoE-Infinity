# PD-DFlash Task 6 Step 5 — engine round-state is BLOCKED on a missing single-round speculator seam

> Status: **BLOCKED (honest gate).** Steps 1–4 of Task 6 are complete and merged
> here (from #156 + #157). Step 5 (engine DRAFT→VERIFY→DRAFT driving) cannot be
> implemented within Task 6's authorized files without a per-round control seam
> on `DFlashSpeculator` that does not exist and that Task 6 does not authorize
> creating. This note is the "exact seam needed" report the plan/task requires
> instead of hacking one in.

Plan: `docs/superpowers/plans/2026-08-14-pd-dflash-serving-scheduler.md`, Task 6
Step 5 (lines 527–529) and the File-Structure/anchor notes (lines 30, 86).

## What is already done (merged base of this branch)

This branch is `origin/feat/pd-dflash-phaseA-runner` (#157, `expert_nbytes`
instrumentation) with `origin/feat/pd-dflash-serving-scheduler` (#156, the
DRAFT/VERIFY scheduler) merged in. Together they already deliver:

- **Task 4** — `SequenceStatus.DRAFT` / `VERIFY` states and transitions
  (`moe_infinity/serving/sequence.py`).
- **Task 5** — the pure 2-D deficit rule `admit_verify_demands(...)`
  (`moe_infinity/serving/scheduler.py:50`).
- **Task 6 Step 3** — `SchedulerOutput.{draft_seq_ids, verify_seq_ids,
  num_verify_tokens, num_verify_expert_bytes}` (`moe_infinity/serving/batch.py`).
- **Task 6 Step 4** — `Scheduler.set_verify_demand` / `clear_verify_demand` +
  `_apply_verify_scheduling` (called at the end of `schedule()`), plus the
  engine-constructor wiring of `verify_*_budget` / `verify_*_deficit_cap`
  (`moe_infinity/serving/engine.py:135-147`, added by #156 commit `3cacafe`).
- **Task 6 Step 1 (scheduler-level)** — the "three speculative sequences"
  admission scenario is already pinned as
  `test_scheduler_admits_verify_demands_by_tokens_and_bytes`
  (`tests/python/serving/test_dflash_deficit_scheduler.py:231`).
- **#157** — `ExpertPrefetcher.expert_nbytes_map[(layer,expert)]` holds the real
  registration-time FP4 payload bytes; `RouteAheadStats` records byte-accurate
  coverage/waste.

The ONLY unfinished piece is **Step 5: the engine must drive per-round
DRAFT→VERIFY→DRAFT and register each pending verify's EXACT token/byte demand so
the scheduler decides when VERIFY runs.**

## Why Step 5 is blocked (root cause, not a symptom)

Step 5 requires the engine, **before** running a verify, to register that
verify's exact demand: `set_verify_demand(seq_id, tokens=B,
expert_bytes=Σ expert_nbytes[routed union], in_flight=…)`, then run VERIFY only
for the sequences the scheduler admits (`SchedulerOutput.verify_seq_ids`).

The serving engine's speculative path exposes **only** the whole-request loop:

- `ContinuousBatchingEngine._step_speculative` (`engine.py:326`) calls
  `speculator.generate(...)` once — the entire DFlash DRAFT→VERIFY→rollback loop
  for all rounds — then `update_after_step(..., committed_counts=…)`.
- `SpeculativeGenerator` Protocol (`engine.py:30`) has just `generate()`.
- `DFlashSpeculator` (`moe_infinity/spec_decode/dflash.py:471`) runs every round
  inside `_generate_single` (`dflash.py:841`). The per-round state
  (`target_kv`, `draft_kv`, `context_feature`, `anchor`, `start`, `step_trace`)
  is **local** to that method. `_run_drafter` (711) and `_verify_target_block`
  (661) are private and tightly coupled to those locals.
- Route-ahead is **internal to the verify forward**: `_verify_target_block`
  opens `route_ahead_context(...)` and `DistributedExpertExecutor.dispatch_local`
  pins/prefetches the layer's ACTUAL routed expert union during the verify
  (`_route_ahead_ctx.py`; `docs/serving.md:187`). The exact `expert_nbytes` sum
  is therefore known only **while/after** the verify runs, surfaced afterward in
  `RouteAheadStats` — never before it.

Consequences:

1. There is no public per-round API (`grep` across `moe_infinity/` for
   `draft_round|verify_round|draft_block|verify_block` → none).
2. The EXACT byte demand of a *pending* verify cannot be obtained without first
   running that verify's routing — which is exactly what admission is supposed
   to gate. `generate()` is atomic over all rounds, so the scheduler cannot
   interpose between draft and verify.

## Why there is no honest non-seam workaround

- **Post-hoc `RouteAheadStats`** gives bytes only *after* the verify already ran
  → useless for admission ("let Scheduler decide when VERIFY runs").
- **Fabricating** bytes from expert *count* × average is explicitly forbidden by
  the plan (Step 4: "exact summed FP4 payload, not expert count") and by the
  task ("do NOT fabricate a byte estimate").
- **Re-implementing** the draft/verify/rollback loop inside `engine.py` (driving
  the private `_run_drafter` / `_verify_target_block` / snapshot / rollback)
  would be a *second* draft/verify + routing path — forbidden by Step 5 ("Do not
  add another router or prefetch path") and the design's single-router rule.

## The exact seam required (NOT authorized by Task 6's file list)

A public, per-sequence, single-round control surface on `DFlashSpeculator`
(`moe_infinity/spec_decode/dflash.py`) that externalizes the `_generate_single`
loop state, plus a read-only route projection. Sketch:

```python
class SpecSession:                      # externalized per-sequence round state
    target_kv; draft_kv; context_feature; anchor; start; emitted; step_trace

def begin_session(prompt_ids, sampling_params, stop_ids) -> SpecSession:
    # prefill/anchor forward (_forward_target(..., logits_to_keep=1)); seed state

def draft_round(session) -> DraftResult:
    # one _run_drafter pass -> block; PROJECT the pending verify's routed expert
    # union via the SAME gate/top-k path the verify will use (NO expert exec,
    # NO second router — this is the design §10 "BM1" gate+topk_softmax
    # projection on the B-token block). Returns:
    #   tokens=block_size,
    #   expert_union: set[(layer_id, expert_id)],
    #   expert_bytes = sum(prefetcher.expert_nbytes_map[(l,e)] for (l,e) in union)

def verify_round(session) -> VerifyResult:
    # one _verify_target_block under route_ahead_context (existing prefetch seam),
    # accept rule, commit/rollback, append step_trace. Returns:
    #   accepted_token_ids, accept, committed_count, finished
```

Engine loop (in the authorized `engine.py`) then becomes, per active
speculative sequence: `draft_round` → `scheduler.set_verify_demand(seq_id, B,
expert_bytes, in_flight)` → `schedule()` → for admitted `verify_seq_ids`:
`verify_round` → `update_after_step(committed_counts=…)` →
`clear_verify_demand(seq_id)` → status back to DRAFT (or FINISHED); unadmitted
sequences stay DRAFT and their deficit carries.

The route projection in `draft_round` also touches
`moe_infinity/distributed/expert_executor.py` (a read-only "route-only" mode of
the existing dispatch, returning the routed expert ids without executing them).

### Authorization gap

Task 6's File list (plan lines 491–498) is exactly: `serving/batch.py`,
`serving/scheduler.py`, `serving/engine.py`, and their tests + the benchmark. It
does **not** include `spec_decode/dflash.py` or `distributed/expert_executor.py`.
Step 5's wording ("consume the already-built DFlash route-ahead result", "Do not
add another router or prefetch path") describes *consuming* an existing artifact,
not creating a per-round draft/verify control API. The plan's own anchor (line
30) scopes Task 6 to "replaces only that coarse serving admission with
round-state scheduling." So the seam is (a) genuinely required and (b) outside
Task 6's authorized scope.

## Recommendation

Add a follow-up task (or amend Task 6's File list) that authorizes the
`dflash.py` per-round `SpecSession` seam + the read-only route projection in
`expert_executor.py`, gated behind the same BM1 gate+top-k projection primitive
the design already specifies. Only then can `engine.py` Step 5 register EXACT
`expert_nbytes` demand and let the 2-D scheduler govern VERIFY under concurrency
without a second router or a fabricated byte estimate.
