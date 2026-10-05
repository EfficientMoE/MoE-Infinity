# Tail-policy evaluation: measure p99 + fidelity for every candidate (#234/#257)

Goal: decide between the six tail-targeting policies (menu in
`2026-10-02-tail-targeting-drop-policies.md`) **with data**, not by picking
blind. Produce one (p99 impact, fidelity impact) point per policy per knob
setting, pick the Pareto winner, then implement only that in the fused C++
path.

Core idea: **implementing six policies in the C++ route worker just to compare
them is wasteful.** Fidelity is measurable *exactly* offline by replaying real
routing weights through the real expert compute; p99 is *predictable* offline
from a fetch-cost model over the real residency trace. So evaluate all six in
a Python simulator first; only the shortlist touches C++ and a GPU.

## Phase 0 — Trace capture (one-time, per model)

Instrument a single baseline decode run (OLMoE + Qwen3-30B, ratio 0.05, the
frozen 24-prompt × 128-token set) to dump, per (layer, decode-step, token):
- routed expert ids + post-softmax routing weights (pre-drop),
- residency snapshot at dispatch (the `resident_on_gpu` bitmap),
- per-expert fetch size in bytes, and
- measured per-expert H2D fetch latency where available (from the dispatcher
  timing samples) + the realized per-layer wait time.

Hook point: the fused route worker already has all of this at
`ApplyFusedExpertDrop` time; add a trace-dump mode behind
`MOE_EXPERT_DROP_TRACE=/path` that appends a compact record (no behavior
change). One off-policy run per model produces the ground-truth substrate.
Store as `traces/{olmoe,qwen3_30b}_r005.npz`.

## Phase 1 — Fetch-cost model + calibration (the credibility gate)

Build a pure function `predict_layer_latency(kept_nonresident_experts, trace)`
that estimates a decode step's critical-path time from the residency trace and
per-expert fetch costs (overlapped fetches ⇒ latency ≈ max single required
fetch + queue term; fit the queue term to the trace).

**Calibrate, then validate before trusting it:** tune the model so its
predicted p50/p90/p99 reproduce the **already-measured** numbers for the two
policies we have real data for — baseline `off` and the shipped tail-drop
`fused` — on both OLMoE and Qwen3-30B (we have: OLMoE off/fused, Qwen off
114.2/95/88 and fused budgets). Acceptance: predicted p99 within ±10% of
measured for those known points. If it can't hit that, the offline p99
numbers are not trustworthy and we fall back to implementing each policy for
real (expensive path). This calibration step is non-negotiable.

## Phase 2 — Offline policy simulator (all six, cheap)

Pure-Python `simulate(policy, params, trace)` implementing the selection logic
of each candidate (A deadline-drop, B resident-redirect, C head-budget hybrid,
D adaptive-budget, E partial-wait) operating on the captured traces. For each
policy and each knob value it emits, per decode step:
- the modified kept-expert set + renormalized weights,
- predicted-p99 via the Phase-1 model,
- drop/redirect counts.

Sweep each policy's knob (A/E: `L_us`; C: `head_budget`; D: adaptive schedule
slope; B: similarity-k) across a grid. Reuse the existing selector parity
corpus discipline: C/D must match the current `select_expert_drops` exactly
when their extra levers are disabled (regression guard).

## Phase 3 — Fidelity replay (all six, exact)

Fidelity needs the real expert outputs, not a model. Two options, pick the
cheaper that fits:
- (a) **Weight-replay:** during Phase-0 capture also cache each routed expert's
  output tensor per (layer, step). Then fidelity of any policy = recombine the
  cached outputs with the policy's renormalized weights (and, for B, the
  redirected target's cached output) and compare to the baseline combine. Pure
  arithmetic, no GPU model pass per policy. Preferred.
- (b) **Live replay:** re-run the model forcing each policy's mask (GPU). Exact
  but N× model runs; fallback if caching all expert outputs is too large.

Metrics per policy/knob (vs the no-drop baseline on the frozen set):
- **Primary fidelity:** greedy next-token agreement rate.
- **Secondary:** mean + p99 KL of final-logit softmax; per-layer hidden-state
  cosine.
- **Tertiary (if time):** NLL on a small held-out continuation set.

For E (partial-wait) fidelity depends on *which* experts were stragglers —
drive it from the measured fetch latencies in the trace, not mass.

## Phase 4 — Pareto decision

Plot every (policy, knob) as a point in (predicted-p99-on/off,
greedy-agreement) space; overlay the Pareto frontier. Report two slices:
- at a **fidelity floor** (e.g. ≥99% greedy agreement): which policy gives the
  lowest p99?
- at the **p99 target** (≤0.8×off): which policy keeps the highest fidelity?

Winner = the policy on the frontier that meets p99≤0.8 at acceptable fidelity,
tie-broken by implementation complexity (C/D < A < B/E) from the menu doc.

## Phase 5 — Confirm the winner on real hardware

Implement only the winning policy (1, maybe 2) in the fused C++ route worker
(TDD against its Python simulator as the reference), then run the real two-model
A/B gate (OLMoE + Qwen3-30B, ratio 0.05, budgets, 24×128). Acceptance: measured
p99 within the ±10% band of the simulator's prediction for that policy (closes
the loop on Phase 1), and the fidelity metric matches the Phase-3 replay.

## Deliverables / branch

- `benchmarks/expert_drop/trace_capture.py`, `policy_sim.py`, `fidelity_replay.py`,
  `pareto_report.py` + captured traces, on a new branch
  `feat/expert-drop-tail-eval` off `feat/expert-drop-dispatcher-fused`.
- A results table (this doc, appended) with the Pareto frontier and the pick.
- Only then: the Phase-5 implementation plan for the chosen policy.

## Effort

Phase 0-1 ~1 day (capture + calibrate — the gating risk is calibration).
Phase 2 ~1 day (six selectors, pure Python). Phase 3 ~1 day (replay + metrics;
(a) if output caching fits, else (b) is longer). Phase 4 ~half day. Phase 5
per chosen policy, 1-2 days. The offline phases need no GPU except the one-time
capture and the final confirmation.

## Risk

The whole offline approach hinges on the Phase-1 cost model reproducing known
p99 within ±10%. If MoE-Infinity's overlap/queue behavior is too irregular to
model from the trace, abandon offline p99 prediction and instead measure p99
by implementing C and D only (cheapest two) for real, keeping the offline
*fidelity* replay (which does not depend on the cost model) for all six. State
this fork explicitly at the Phase-1 gate.
