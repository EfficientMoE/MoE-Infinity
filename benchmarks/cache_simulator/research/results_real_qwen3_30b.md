# Real expert-cache trace results — Qwen3-30B-A3B (235B-family proxy)

Part of #289. Real-trace capture on the **actual Qwen3-MoE routing family**
(128 experts, top-8) that the 235B target belongs to, while the 235B itself
stays GPU-blocked. Every number below comes from an actual SGLang 0.5.20
`per_token` expert-routing capture replayed through the merged simulator; none
are synthetic. Companion: `capture_report.md` (addendum) for environment,
TP2-on-Blackwell status, dual-rank dedup, and the current 235B status.

## Headline

- **`Qwen/Qwen3-30B-A3B` captured on 2 free Blackwell GPUs via `--tp 2`.**
  TP2 initialized cleanly on sm_120 (NCCL 2.29.7); the `per_token` recorder
  dumped valid `.pt` traces; the full capture -> convert -> Belady -> policy
  sweep ran end to end on real data.
- **Same routing geometry as the 235B target** (128 experts, top-8, all-MoE
  layers), so these are the most faithful real numbers available for the 235B
  cache-key behaviour short of running the 235B itself.
- **On real Qwen3-MoE routing, temporal locality dominates.** LRU
  substantially beats frequency policies (LRU 0.4565 vs LFU 0.2021 at
  cap=256), the opposite of the frequency-skew assumption; there is a large
  Belady-vs-LRU gap at small cache (0.70 vs 0.46 at ~4% of the key space),
  i.e. real headroom for a smarter policy.
- **235B remains GPU-pending** — only 2 of 6 GPUs free at both mandated checks
  (start + post-30B); `--tp 4` needs >=4. Not a software blocker.

## Model geometry (real capture) vs the 235B target

Verified from the HF `config.json`:

| field | Qwen3-30B-A3B (captured) | Qwen3-235B target |
| --- | --- | --- |
| architecture | `Qwen3MoeForCausalLM` | `Qwen3MoeForCausalLM` |
| routed experts | **128** | **128** |
| experts per token (top_k) | **8** | **8** |
| recorder buffer top_k | 8 (no `-1` padding) | 8 |
| decoder layers | 48 (all MoE, `mlp_only_layers=[]`) ⇒ `ignore_layers=0` | ~94 (all MoE) ⇒ `ignore_layers=0` |
| cache-key space | 48 x 128 = **6144** | 94 x 128 = 12032 |
| weights | BF16 (~60 GB) | FP8 block-wise (~235 GB) |

Routing geometry is identical; only layer count and weight precision differ.
FP8 quantizes expert *weights*, not the gate, so the 235B's `topk_ids` come
from the same Qwen3-MoE routing — this proxy is faithful for cache analysis.

Validated from the dumps: `topk_ids_of_layer` is `[48, seq_len, 8]` int32,
expert ids in `[0, 127]` with **no `-1` padding** (top-8 fills the 8-slot
recorder buffer exactly), `last_physical_to_logical_map` is `[48, 128]` and
**identity** (`ep_num_redundant_experts=0`), so physical == logical and the
converter remap is a verified no-op. Record 0 is prefill (`forward_mode=1`),
records 1..64 are decode (`forward_mode=2`), one token each.

## Dataset

- **150 MMLU** questions: `cais/mmlu`, config `all`, split `test`, first 150
  in fixed dataset order (deterministic, no shuffle), each rendered as a
  4-choice multiple-choice prompt (prompt builder in `drive_capture_mmlu.py`).
- Decoding: greedy (`temperature=0`), fixed `max_new_tokens=64`, batch size 1
  per request (one dump per prompt).
- `--tp 2` makes both TP ranks dump the (replicated, non-DP) routing, so the
  150 prompts produced 300 raw `.pt`; de-duplicated by routing-content hash to
  150 (each content appeared in exactly 2 files) before merging.
- Merged access stream: **7,686,912** decode accesses; **5,796** distinct
  `(layer, expert)` keys observed = **94.3%** of the 6144 key space.

## Policy x capacity hit rate (REAL)

Demand hit rate (fraction of decode expert accesses served from the resident
cache). `belady` is the Belady-optimal ceiling ("Ideal"). Capacities in
`(layer, expert)` entries; full key space 6144. Capture commit `963aab0`.
Merged stream from the 150 deduped per-request dumps; `ignore_layers=0`.

| Policy | cap=256 | cap=512 | cap=1024 | cap=2048 | cap=3072 | cap=4096 | cap=6144 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Belady (Ideal) | 0.7028 | 0.7898 | 0.8705 | 0.9462 | 0.9810 | 0.9961 | 0.9992 |
| LRU | 0.4565 | 0.6617 | 0.7631 | 0.8696 | 0.9378 | 0.9891 | 0.9992 |
| LFU | 0.2021 | 0.3266 | 0.5069 | 0.7650 | 0.9188 | 0.9824 | 0.9992 |
| LFU-K | 0.4565 | 0.6617 | 0.7631 | 0.8696 | 0.9378 | 0.9891 | 0.9992 |
| Model-tracing | 0.2083 | 0.3387 | 0.5181 | 0.7750 | 0.9253 | 0.9846 | 0.9992 |
| Score-based | 0.4565 | 0.6617 | 0.7631 | 0.8696 | 0.9378 | 0.9891 | 0.9992 |
| LHU | 0.4485 | 0.6505 | 0.7535 | 0.8663 | 0.9412 | 0.9897 | 0.9992 |
| 1-layer-ahead* | 0.4565 | 0.6614 | 0.7622 | 0.8684 | 0.9363 | 0.9888 | 0.9992 |
| Stride* | 0.4565 | 0.6616 | 0.7629 | 0.8695 | 0.9378 | 0.9891 | 0.9992 |
| Multi-layer* | 0.4559 | 0.6613 | 0.7615 | 0.8677 | 0.9353 | 0.9886 | 0.9992 |
| Activation-matrix* | 0.4565 | 0.4567 | 0.7052 | 0.8422 | 0.9277 | 0.9866 | 0.9992 |

`*` = trace-oracle prefetch policy; the reported **demand** hit rate is bounded
by Belady, and the prefetch value shows up in `effective_hit_rate`
(see `results_real_qwen3_30b.csv`).

### Task C miss-ratio curve (Belady), independent engine cross-check

The vectorized `opt_mrc` MRC engine reproduces the Belady row above
**bit-for-bit** (the sweep asserts `|belady_simulate - opt_mrc| < 1e-9` at
every capacity). Belady miss-ratio curve (`1 - hit_rate`):

| capacity | 256 | 512 | 1024 | 2048 | 3072 | 4096 | 6144 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Belady hit rate | 0.7028 | 0.7898 | 0.8705 | 0.9462 | 0.9810 | 0.9961 | 0.9992 |
| Belady miss ratio | 0.2972 | 0.2102 | 0.1295 | 0.0538 | 0.0190 | 0.0039 | 0.0008 |

Hit rate is strictly in `(0, 1)` and monotonically non-decreasing in capacity.

## Observations (real vs synthetic, and vs the small smoke model)

- **Real Qwen3-MoE routing is far more cacheable** than both the synthetic
  Zipf model and the earlier 60-expert smoke model: Belady already hits 0.70
  at cap=256 (~4% of the 6144 keys), rising to 0.95 by cap=2048 (33%). The
  working set is effectively ~5,796 keys and nearly everything is resident by
  cap=4096.
- **Temporal locality beats frequency on real data.** LRU (and the LRU-based
  LFU-K / Score / oracle-prefetch variants, which share its demand behaviour)
  reach 0.4565 at cap=256, more than double LFU (0.2021) and Model-tracing
  (0.2083). The synthetic generator's Zipf skew favoured LFU/Model-tracing;
  real MMLU-driven Qwen3-MoE routing favours recency. This is a concrete,
  real-trace correction to the synthetic ordering in `results_synthetic.md`.
- **Largest Belady headroom is at small cache.** The Belady-vs-LRU gap is
  ~0.25 at cap=256 and closes to ~0.01 by cap=3072 — so a smarter policy only
  pays off when the GPU can hold well under ~half the experts.
- **Belady-ceiling invariant holds on real data** at every capacity (asserted
  at runtime in `run_policies_real.py`).

## Reproduce

```bash
# 1. Launch SGLang 0.5.20, Qwen3-30B-A3B, TP2 on two free GPUs:
export HF_HOME=/mnt/raid0nvme0/leyang/hf_cache
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=<raw_dir>
export CUDA_VISIBLE_DEVICES=4,5
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-30B-A3B --trust-remote-code \
    --tp-size 2 --dp-size 1 \
    --expert-distribution-recorder-mode per_token --disable-cuda-graph \
    --chunked-prefill-size 8192 --mem-fraction-static 0.85 \
    --host 127.0.0.1 --port 30000

# 2. Drive start/generate/stop/dump per MMLU prompt (batch size 1):
python drive_capture_mmlu.py --port 30000 --max-new-tokens 64 --num-prompts 150

# 3. De-dup the 300 TP0/TP1 dumps to 150 by routing content, then convert
#    (identity remap; merge into one access stream):
python -m benchmarks.cache_simulator.convert_sglang_traces \
    --inputs "<dedup_dir>/*.pt" --to belady --out <belady_dir>/stream.pt --merge

# 4. Belady acceptance sweep:
for C in 256 512 1024 2048 3072 4096 6144; do
  python -m benchmarks.cache_simulator.belady_cache_simulator \
      --files "<belady_dir>/stream.pt" --capacity $C --ignore-layers 0
done

# 5. Full policy x capacity sweep + this table / CSV / EXPERIMENTS rows:
python -m benchmarks.cache_simulator.research.run_policies_real \
    --files "<belady_dir>/stream.pt" \
    --model qwen3-30b-a3b --dataset mmlu150-greedy64 \
    --ignore-layers 0 --capacities 256 512 1024 2048 3072 4096 6144 \
    --tag qwen3_30b --out benchmarks/cache_simulator/research
```

## Trace location (git-ignored; on disk only)

Traces are excluded by `benchmarks/cache_simulator/.gitignore` (`*.pt`). On
this box:

```
/mnt/raid0nvme0/leyang/moeinf-traces/
  raw_qwen3_30b/              # 300 raw SGLang per_token dumps (TP0+TP1, 2 per prompt)
  raw_qwen3_30b_dedup/        # 150 deduped (1 per prompt, by routing content)
  belady_qwen3_30b/           # 150 per-request converted Belady .pt
  belady_qwen3_30b_stream/stream.pt   # merged access stream used for the table above
  qwen3_30b_dedup_filelist.json       # dedup manifest
  drive_capture_mmlu.py       # MMLU capture driver (prompts + start/stop/dump loop)
```

SGLang venv: `/mnt/raid0nvme0/leyang/sglang-venv` (sglang 0.5.20, torch
2.13.0+cu130, flashinfer 0.6.18). Simulator venv:
`/mnt/raid0nvme0/leyang/MoE-Infinity/.venv` (python 3.12.3, numpy 2.4.4,
torch 2.11.0+cu130).
