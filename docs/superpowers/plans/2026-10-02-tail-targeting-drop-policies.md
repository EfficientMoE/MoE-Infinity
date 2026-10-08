# Tail-targeting drop policies — decision menu (#234 / #257)

Status: **decision required before implementation.** Pick one candidate (or
an explicit combination) from the menu below; each has a different
latency/accuracy/complexity profile. No code until a choice is made.

## Problem restatement

Fused drop-on-miss (shipped, correct, overhead-free) trims p50/p90 by ~13%
but leaves p99 ≈ 0.94×off on Qwen3-30B / 0.82×off on OLMoE. Root cause
(detail on #257): the mass-budget policy only drops the **low-mass tail** of
a token's routing distribution; the p99 driver is a token whose **high-mass**
expert missed cache, and that expert is never dropped. Moving p99 requires a
policy willing to act on a high-mass non-resident expert on the critical path
— an explicit fidelity/latency trade.

## Shared evaluation axes

For every candidate, Task-0 evaluation reports on the same harness
(`benchmarks/expert_drop/ab_latency_olmoe.py`, OLMoE + Qwen3-30B, ratio 0.05,
budgets as applicable):
- **Latency:** p50/p90/p99 on/off.
- **Fidelity:** greedy-token agreement vs the no-drop run on the frozen
  24-prompt set, plus mean KL of the post-renorm routing weights (cheap proxy
  for output perturbation); NLL on a small held-out set if time permits.
- **Complexity:** new runtime state, whether it stays inside the fused C++
  route worker, and whether it needs a fetch-cost model or similarity map.
- **Determinism:** does the decision stay data-independent on the GPU (no new
  host sync), preserving the zero-overhead property.

## Candidates

### A. Deadline-aware high-mass drop (`drop_on_deadline`)
Keep the tail-budget drop, and additionally drop the **most-expensive-to-fetch
non-resident expert** (even if high-mass) when the token's projected
critical-path fetch exceeds a latency budget `L_us`, down to `min_k`,
renormalizing. Needs a per-expert fetch-cost estimate (bytes / measured H2D
rate; the dispatcher already tracks residency and sizes).
- **Targets p99 directly.** Strongest tail lever.
- **Fidelity risk: high** — drops exactly the heavy experts on the hardest
  tokens. The KL/agreement metric will move most here.
- **Complexity: medium** — fetch-cost model + a budget knob; fits the route
  worker (residency + sizes already there).

### B. Nearest-resident mass redirect (`reroute_to_resident`)
Instead of dropping a high-mass miss, add its routing weight to the nearest
**resident** expert (by router-logit rank, or a precomputed expert-similarity
map). Mass is conserved; no fetch.
- **Targets p99** while preserving total routed mass → best expected fidelity
  of the tail-acting options.
- **Fidelity risk: medium**, quality depends entirely on similarity quality;
  logit-rank proxy is cheap but crude, a learned/offline similarity map is
  better but is new released-artifact state.
- **Complexity: high** — needs a similarity source and a conservation-correct
  combine; largest C++ change.

### C. Head-budget hybrid (`min_k_head_budget`)
Current tail drop, plus allow dropping **one** high-mass expert iff it is the
*sole* non-resident on the critical path AND its mass ≤ a small separate
`head_budget` (e.g. 0.15). Bounded, surgical.
- **Targets the specific p99 cause** (one cold heavy expert) with a hard cap
  on how much mass can be sacrificed.
- **Fidelity risk: low–medium**, explicitly bounded by `head_budget`.
- **Complexity: low** — one extra branch in the existing selector; no new
  state, stays in the fused worker, keeps the zero-sync property.

### D. Residency-adaptive budget (`adaptive_budget`)
Make the effective `mass_budget` a function of this token's **miss count**:
when many routed experts miss (the p99 tokens), raise the budget so mid-mass
experts also drop; when few miss, keep it tight. One monotone schedule.
- **Targets p99** indirectly by dropping more only when the token is already
  slow; spends fidelity only where latency is worst.
- **Fidelity risk: medium**, self-limiting (tight budget on easy tokens).
- **Complexity: low** — budget becomes a function of the miss count the
  selector already computes; no new state.

### E. Deadline partial-wait (`compute_on_arrived`) — non-drop
Change the execution model: issue all fetches, but at deadline `L_us` compute
the layer with **whatever experts have arrived**, treating stragglers as
dropped post-hoc (renormalize over arrived). Purest tail control; independent
of mass.
- **Targets p99 most directly** (hard latency cap per layer).
- **Fidelity risk: variable** — depends on how often the heavy expert is a
  straggler; unbounded in the worst case unless combined with `min_k`.
- **Complexity: highest** — touches the dispatcher wait/combine path and
  completion-event handling, not just selection; biggest risk to the shipped
  correctness guarantees.

## Comparison

| candidate | p99 lever | fidelity risk | complexity | new state | stays zero-sync |
|---|---|---|---|---|---|
| A deadline high-mass | strong | high | medium | fetch-cost model | yes |
| B resident redirect | strong | medium | high | similarity map | yes |
| C head-budget hybrid | medium | low–med (capped) | **low** | none | **yes** |
| D adaptive budget | medium | medium | **low** | none | **yes** |
| E partial-wait | **strongest** | variable | high | none | no (wait-path) |

## Recommendation

Prototype **C (head-budget hybrid)** and **D (adaptive budget)** first: both
are low-complexity, add no released artifacts, stay inside the fused worker
with the zero-overhead property intact, and bound the fidelity cost. Gate them
on the shared axes; if neither moves p99 enough at acceptable fidelity,
escalate to **A** (needs a fetch-cost model) or **B** (needs a similarity
map). Hold **E** unless a hard per-layer latency cap becomes a product
requirement, since it reopens the dispatcher correctness surface.

## Decision gate

Choose one of: {C, D, C+D, A, B, E}. On selection, write the per-task
implementation plan (TDD against the Python reference first, then the C++
fused path, then the two-model gate) as a follow-up to
`2026-09-30-dispatcher-fused-expert-drop.md`.
