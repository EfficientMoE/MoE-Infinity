# Per-token tail classification: what do slow tokens actually fetch? (#234)

Answers the discriminating-log request on issue #234: for tokens at
p50 / p90 / p99, classify every fetched non-resident expert as
**must-keep** (policy-kept by mass), **droppable-but-kept** (min_k clamp,
flatness bypass, or guard), or **actually dropped**, and record in-layer
fetch depth — to decide between (1) re-scoping the gate to p50/p90,
(2) a tail-targeting drop policy, or (3) post-selection fetch priority.

## Setup

Per-token fetch records from the earlier Qwen3-30B run were **not kept**
(the #293/#294 traces are SGLang expert-ID streams without residency or
ITL), so this is the one instrumented rerun: `trace_capture.py` with
`MOE_EXPERT_DROP_TRACE` + raw per-prompt ITLs, joined offline by
`classify_tail_tokens.py`.

- Qwen3-30B-A3B bf16 (128 experts / top-8, 48 MoE layers), frozen
  24×128 greedy set, `device_memory_ratio 0.05`, native GPU-only routing,
  1× RTX PRO 6000 Blackwell (GPU idle, host shared with a 4-GPU training
  co-tenant — p99 magnitudes are noise-limited as documented in
  `methodology_fixes.md`; the *composition* analysis below uses within-run
  percentile bands and is insensitive to that).
- Arms: `off` (counterfactual replay at budget 0.20) and fused
  `on_miss` mass-0.20 (`min_k 1`, flatness floor 0.5); 330,668 experts
  dropped on the fused arm — same scale as the issue-thread run.
- 146,304 decode records per arm (24 prompts × 127 forwards × 48 layers),
  0 residency-unknown records; selector replay bit-checked against
  `select_expert_drops` on 2,000 records per arm (100% match).
- Alignment: decode forward *f* ↔ `itl[f-1]`; forward 0 per prompt excluded
  (unmeasured by the streamer timer).

## Result: fetch composition is FLAT across latency bands

Fused on_miss mass-0.20 arm (per-token means; `mk%` = must-keep share of
fetched experts, `depth` = max concurrent non-resident fetches in one
layer dispatch):

| band | tokens | itl ms | dropped | must-keep | minK | flat | mk% | mk mass-rank | depth | MB/tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p0–50 | 1512 | 83.5 | 100.8 | 184.9 | 0.00 | 17.1 | 91.5% | 3.43 | 6.94 | 1907 |
| p50–90 | 1209 | 94.7 | 98.5 | 195.9 | 0.00 | 21.1 | 90.3% | 3.42 | 7.11 | 2048 |
| p90–99 | 272 | 137.5 | 99.2 | 187.5 | 0.00 | 19.2 | 90.7% | 3.44 | 6.89 | 1951 |
| p99–100 | 31 | 296.0 | 96.5 | 195.1 | 0.00 | 19.3 | 91.0% | 3.45 | 7.10 | 2023 |

Off arm (counterfactual at the same budget) shows the same flatness:
would-drop ≈ 97–100/token and mk% 89.8–92.4% in every band.

A p99 token fetches the **same workload** as a p50 token — same expert
count, same must-keep share, same mass-rank profile (3.43 vs 3.45), same
in-layer fetch depth (6.9–7.1), same bytes (±6%) — and takes 3.4× longer.

## Verdict against the three candidate explanations

- **(a) "p99 tokens fetch must-keep high-mass experts" — holds, but
  degenerately for every token.** At `dmr 0.05` the cache holds ~5% of
  experts, so ~80% of the 384 routed experts/token miss; after the budget
  drops its ~100, the ~190 surviving fetches are must-keep by mass. This
  is true in all four bands, so it explains why dropping shifts the whole
  distribution (p50) rather than specifically the tail.
- **(b) "p99 tokens fetch droppable-but-kept experts" — ruled out.**
  Droppable-but-kept is 9.0% of tail fetches, entirely flatness-bypass
  rows (near-uniform routing weights — the rows where dropping is most
  fidelity-dangerous); `min_k` never binds (0.00 everywhere; the mass
  budget binds first on every row). A tail-targeting policy has almost no
  mass left to act on at this operating point: the mass policy is already
  saturated.
- **(c) "p99 tokens are congestion-queued cheap fetches" — ruled out in
  its discriminating form.** In-layer fetch depth at the tail (7.10) is
  identical to the body (6.94), so post-selection fetch priority cannot
  single out tail tokens — they are not distinguishable by their fetch
  sets at all.

## Conclusion

Tail inflation at this operating point is **service-time variance on an
identical fetch workload** (host RAM/PCIe contention, allocator and
scheduling jitter), not fetch composition. No drop-selection policy —
current or tail-targeting — can reach `p99 ≤ 0.8×off` here, because the
p99 token's selection problem is the same as the p50 token's. This backs
**option (1): re-scope the Phase-2 gate to p50/p90** (where the mechanism
provably acts: bytes/token −25% dropped mass → p50 −6% in this contended
run, −13% in the issue-thread run), and drops option (2) for this regime.
A fetch-priority lever (3) remains plausible only as a general throughput
optimization, not as a tail-specific fix.

## Addendum: per-prompt structure of the tail (external cross-validation)

Independent cross-validation on PR #295 reproduced every committed number
(pooled percentiles, band counts/means, findings-table cells, drop total)
from the committed JSONs alone, and then decomposed the same 3,024 ITLs by
prompt. We adopted that decomposition into the classifier
(`per_prompt`, `tail_concentration`, `--compare` paired mode) and the
regenerated reports confirm it bit-for-bit:

- **The p99 band is prompt-concentrated.** 80.6% of the 31 tail tokens
  come from 3 of 24 prompts in *both* arms (off: prompts 5/4/1; mass20:
  22/16/10); effective prompt count 3.2–3.3 vs 24 for a uniform spread.
- **The carriers are run-local, not prompt-intrinsic.** Top-3 tail
  carriers overlap 0/3 between arms; paired per-prompt mean-ITL deltas:
  median −3.4%, mean −2.0%, 14/24 prompts improved on the drop arm.
- **Qualifier on the pooled numbers.** The composition analysis above
  (within-run bands) is unaffected, but pooled *cross-arm tail* deltas
  (e.g. the +31% pooled p99 on the fused arm) cannot be attributed to the
  policy at n=24 prompts: which prompts land in contended windows
  dominates the band. Cross-arm tail claims must cite the paired
  per-prompt deltas and carrier overlap (now in `paired_compare` of the
  mass20 report), or come from repeated runs.

This does not change the verdict — the gate re-scope to p50/p90 was argued
from fetch-composition flatness, which is per-token and within-run — but it
strengthens it: the tail is not only compositionally identical to the body,
its carriers are also non-reproducible across runs on this host.
`trace_capture.py` now records `run_start_unix`, per-prompt spans, and
per-token wall-clock timestamps so future runs can localize contended
windows directly instead of inferring them from clustering.

## Reproduction

```bash
# capture (one process per arm; store built on first run)
python benchmarks/expert_drop/trace_capture.py \
  --checkpoint Qwen/Qwen3-30B-A3B --offload-dir <store> \
  --policy on_miss --mass-budget 0.20 --device-memory-ratio 0.05 \
  --trace-out mass20.trace --npz-out mass20.npz --latency-out mass20.json

# classify (off arm first, then the drop arm paired against it)
python benchmarks/expert_drop/classify_tail_tokens.py \
  --trace off.trace --latency off.json --mass-budget 0.20 \
  --output off_report.json
python benchmarks/expert_drop/classify_tail_tokens.py \
  --trace mass20.trace --latency mass20.json \
  --compare off_report.json --output mass20_report.json
```

Reports: `results/tailclass_qwen_mass20_report.json`,
`results/tailclass_qwen_off_report.json`; latency JSONs alongside. Raw
traces (122 MB each) under
`/mnt/raid0nvme0/leyang/moe-bench-results/tailclass/` (git-ignored).
