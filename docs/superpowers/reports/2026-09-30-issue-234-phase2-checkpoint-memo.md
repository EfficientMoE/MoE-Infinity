# Issue #234 Phase 2 — decision-checkpoint memo (drop-on-miss)

Prepared 2026-09-30. Decision owner: maintainer. The Phase-2 spec
(`docs/superpowers/plans/2026-09-30-issue-234-phase2-expert-drop.md`) blocks
merge of `feat/expert-drop-on-miss` until this checkpoint is accepted or
replaced. Full raw data: `.sisyphus/evidence/gpu-followups-2026-09-30.md`.

## What is ready

- Branch `feat/expert-drop-on-miss` @ `a913084`, stacked on
  `feat/uniform-fp8-residency` @ `3d54d15` (conflict pre-resolved; backup at
  `backup/expert-drop-on-miss-pre-stack`). 116 unit tests green; every task
  passed spec + quality review; Oracle whole-branch verdict READY (code-wise).
- Default `off` verified byte-identical to main (object-identity pinned).
- Real `resident_on_gpu` binding verified on a compiled sm_120 extension.

## QA-gate evidence (OLMoE-1B-7B-0924-Instruct, RTX PRO 6000 Blackwell,
`device_memory_ratio=0.05`, 24 prompts x 128 greedy tokens)

Gate from the spec: decode `p99_on <= 0.8 * p99_off`.

| Path | Ratio | Budget | p99 off (ms) | p99 on (ms) | Ratio on/off | Gate |
|---|---:|---:|---:|---:|---:|:---:|
| HF (`use_native_engine=False`) | 0.05 | 0.05 | 40.56 | 40.88 | 1.008 | FAIL |
| HF | 0.05 | 0.10 | 40.56 | 41.98 | 1.035 | FAIL |
| HF | 0.05 | 0.20 | 40.56 | 38.51 | 0.949 | FAIL |
| Native | 0.05 | 0.10 | 40.47 | 43.64 | 1.078 | FAIL |
| Native | 0.05 | 0.20 | 40.47 | 45.11 | 1.115 | FAIL |
| Native | 0.10 | 0.10 | 36.62 | 45.64 | 1.246 | FAIL |

Supporting observations:

- Drops fired heavily (up to 93k experts dropped per arm; residency_unknown=0;
  per-layer histograms recorded), so the mechanism works as designed — the
  latency benefit does not materialize on this model/hardware.
- Overhead isolation (probe + flatness path, zero drops via floor=0.0):
  **+2.72 ms mean ITL (+12%)**, +0.44 ms p99 on the HF path. The per-token
  Python `_apply_expert_drop` cost consumes the head-room the gate targets.
- Accuracy (recorded, not gated): teacher-forced NLL 2.95 -> 6.92 / 7.52 /
  10.08 at budgets 0.05 / 0.10 / 0.20 (2.3x–3.4x worse).

## Options

1. **Reject `on_miss` as-is; keep the merged-ready code parked.** Cost: none
   (default off is inert). The branch stays open pending rework.
2. **Accept with rework precondition** (recommended): keep the config surface
   and residency probe, move the drop decision off the per-token Python path
   (vectorized batch decision or C++-side rewrite at `Enqueue`), then re-run
   the gate. The +2.7 ms overhead alone explains most of the gate miss on this
   small-expert model.
3. **Re-scope the gate**: OLMoE experts are small (~7 MB); fetch stalls are
   cheap relative to overhead. Re-run on a large-expert model (e.g.
   Mixtral-8x7B, DeepSeek-V2-Lite) where a miss costs tens of ms before
   deciding. Evidence harness is reusable as-is.
4. **Accept as-is** — not supported by the evidence (gate fails on both paths,
   accuracy cost is heavy).

## Non-blocking follow-ups surfaced during this work

- Drafted ticket A (dup speculative prefetch) and ticket B (dormant
  precision-policy block) below — file on request.

---

## Drafted ticket A — duplicate speculative_prefetch per dispatch

**Title:** route-ahead: `speculative_prefetch` may issue twice per dispatch
(overlap merge artifact, same class as the corrected `correct_prefetch` dup)

**Body:** `1f30bdb` (overlap-aware prefetch) added
`generation = trigger_speculative_prefetch(...)` at
`moe_infinity/distributed/expert_executor.py:546` in front of the pre-existing
`1da3fb8` call (now :556/:562, phase-annotated by `f439ad7`). With overlap
policy off, the miss-prediction prefetch issues twice; with it on, the second
call's generation is discarded. `fix/dflash-stale-route-ahead-tests` fixed the
identical pattern in `correct_prefetch` (`909b2c5`) but left this one because
its tests (`test_route_ahead_wire.py:299`, `test_route_ahead_metrics.py:365`)
now pin `call_count == 2` as the observed contract. Decide intended semantics,
then either dedupe (and relax the two assertions) or document the dual trigger
as intentional.

## Drafted ticket B — dormant precision-policy block references undefined names

**Title:** executor: precision-policy block in `dispatch_local` references
`expert_count` (undefined in scope) and `expert_list` (`None` at that point)

**Body:** `moe_infinity/distributed/expert_executor.py` (~:531–572, base
`09fd6da`): the `if self.precision_policy is not None:` block builds
observations from `expert_count[expert_id]` — a name never defined in
`dispatch_local` — and iterates `expert_list`, which is `None` there. Dormant
because `precision_policy` defaults to `None` and the empty
`RELEASED_ADAPTIVE_ENTRIES` allowlist keeps it unset in production; it will
raise `TypeError`/`NameError` the first time an adaptive-precision policy is
actually wired. Found during PR-A final review (byte-identical on main;
untouched by the feature branches).
