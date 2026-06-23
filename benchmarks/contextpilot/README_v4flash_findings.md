# ContextPilot on DeepSeek-V4-Flash — Findings

Measurement of ContextPilot's benefit on DeepSeek-V4-Flash (expert-offload path, mp2
in-memory sharding, RTX PRO 6000 / SM120). **Conclusion: ContextPilot offers no benefit on
V4-Flash and in fact regresses it.**

## Artifacts

| File | Purpose |
|---|---|
| `v4flash_ab.py` | In-process A/B harness (CP-on vs CP-off) over the 4 standard workloads |
| `v4flash_ab_report.py` | Reducer: results JSON → markdown table + GO/NO-GO |
| `gen_longctx_workload.py` | Generator for long-context shared-prefix fixtures |
| `fixtures/longctx_shared_prefix_32k.json` | 6 reqs, ~22.7K shared prefix, ~1K suffix |
| `fixtures/longctx_tinysuffix_32k.json` | 4 reqs, ~22.7K shared prefix, ~63-tok suffix |
| `results/v4flash_ab_20260622_183111.{json,md}` | Phase-B A/B results |
| `results/v4_reuse_32k_20260623_100600.json` | 3-mode prefix-reuse benchmark (1K suffix) |
| `results/v4_be_20260623_110000.json` | Break-even run (63-tok suffix) |

Related loader: `moe_infinity/models/deepseek_v4/inmem_shard_loader.py`
(`load_sharded_v4_flash` — single mp1 checkpoint → in-memory N-way TP shards).
Standalone validation/bench scripts live under the v4flash docker image tree
(`v4_snapshot_oracle_test.py`, `v4_prefix_reuse_bench.py`).

## Result 1 — Phase-B (prompt reorder + dedup)

ContextPilot's reorder wraps prompts in a RAG template
(`"Answer the question based on the provided documents <documents>..."`), which **adds**
tokens. On chat-style workloads the dedup savings never materialize, so TTFT regresses:

| workload | TTFT Δ | prompt-tok Δ |
|---|---:|---:|
| shared_prefix_rag | −13.7% | −29.2% |
| multi_turn_conversation | −19.5% | −24.8% |
| batch_with_overlap | −31.7% | −37.0% |
| no_overlap_baseline | −49.0% | −90.4% |

(Negative = CP-on worse.) Verdict: **NO-GO.**

## Result 2 — Cross-request prefix reuse (the "fair test")

V4-Flash has no contiguous paged KV (128-tok sliding window + learned-compressed older KV),
so RadixAttention-style sharing is unsound. The only viable mechanism — an exact-prefix
internal-state snapshot — was built and **proven correct** (restore 7 ms; faithful within the
model's intrinsic non-determinism). But:

| mode (32K prefix, ~1K suffix) | TTFT |
|---|---:|
| cold (chunked prefill) | 19.1 s |
| snapshot_reuse | **231.5 s (12× slower)** |

**Root cause:** after restoring the prefix, the suffix can only be replayed through V4's
single-token **decode** branch (~200×/token slower than chunked prefill). The avoided-prefill
saving is swamped.

**Break-even ≈ 80 suffix tokens** (confirmed: at 63-tok suffix, reuse is 1.18× faster raw).
Long context makes reuse *worse*, not better.

## Net

ContextPilot regresses V4-Flash three ways: (1) Phase-B RAG wrapper inflates prompts,
(2) prefix-reuse is architecturally defeated by V4's attention, (3) V4's own CSA/HCA already
minimizes the KV cost CP targets. A fix would require forking the official `model.py` to add a
chunked-continuation-prefill path (`start_pos>0, seqlen>1`) — Medium–Large effort, worth a
spike only if prefix reuse becomes a core goal.

## Notes

- V4-Flash forward is **non-deterministic** at the logit level (~6–9 abs diff run-to-run from
  expert-offload accumulation order) while argmax-stable; correctness gates use token-sequence
  equivalence + a noise-normalized logit envelope, not bit-equality.
- All runs: mp2 in-memory shard from `model0-mp1.safetensors`, `max_seq_len=40000`,
  `max_resident_experts=8`, temp=0.
