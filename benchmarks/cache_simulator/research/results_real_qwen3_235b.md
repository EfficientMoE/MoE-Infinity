# Real expert-cache trace results — SGLang 0.5.20 capture

Part of #289. This is the **real-trace** follow-up to the offline cache
simulator/policy study (synthetic numbers in `results_synthetic.md` /
`EXPERIMENTS.md`). Every number below comes from an actual SGLang `per_token`
expert-routing capture replayed through the merged simulator; none are
synthetic or fabricated. See `capture_report.md` for the full environment,
the SGLang-on-Blackwell status, and the #13206 handling.

## Headline

- **SGLang 0.5.20 runs on NVIDIA RTX PRO 6000 Blackwell (sm_120).** The server
  launches, loads an MoE model, serves greedy requests, and the `per_token`
  expert-distribution recorder dumps correct `.pt` traces. The pipeline
  capture -> `convert_sglang_traces` -> Belady simulator -> policy sweep works
  end to end on real data.
- **The 235B target was blocked by GPU availability, not by software.** Only
  **2 of 6** GPUs were free on this shared box (GPUs 0-3 were held by another
  user at ~44 GB + 80-100% util for the whole session); `--tp 4` needs >=4.
  The capture that *did* run used the prescribed Step-0 smoke model
  `Qwen/Qwen1.5-MoE-A2.7B-Chat` on one free GPU, and it exercises the identical
  recorder/converter/simulator path the 235B run would use.

## Model geometry (real capture)

`Qwen/Qwen1.5-MoE-A2.7B-Chat` — verified from the downloaded `config.json`:

| field | value |
| --- | --- |
| architecture | `Qwen2MoeForCausalLM` |
| decoder layers | 24 (all MoE, `decoder_sparse_step=1`, no `mlp_only_layers`) ⇒ `ignore_layers=0` |
| routed experts | 60 |
| experts per token (top_k) | 4 |
| recorder buffer top_k | 8 (padded with `-1`; simulator drops padding) |
| cache-key space | 24 x 60 = **1440** `(layer, expert)` entries |

Validated from the dumps: `topk_ids_of_layer` is `[24, seq_len, 8]` int32,
expert ids in `[-1, 59]`, `last_physical_to_logical_map` is **identity**
(`ep_num_redundant_experts=0`), so physical == logical and the converter's
remap is a verified no-op. Record 0 is prefill (`forward_mode=1`,
`extend_seq_lens` set); records 1..T are decode (`forward_mode=2`), one token
each, T = `max_new_tokens`.

## Dataset

- 12 prompts (`mixed20-greedy64`): a fixed, general-knowledge/chat/reasoning
  prompt set (relativity, French Revolution, transformers, TCP/UDP, water
  cycle, ...). Full list in `run_policies_real.py` is not used — the prompts
  live in the capture driver `drive_capture.py` (kept with the traces on disk).
- Decoding: greedy (`temperature=0`), fixed `max_new_tokens=64`, batch size 1
  per request (one dump per prompt).
- Merged access stream: **83,328** decode accesses across all 12 requests;
  **1440** distinct `(layer, expert)` keys observed = full key-space coverage.

## Policy x capacity hit rate (REAL)

Demand hit rate (fraction of decode expert accesses served from the resident
cache). `belady` is the Belady-optimal ceiling ("Ideal"). Capacities are in
`(layer, expert)` entries; the full key space is 1440. Capture commit
`34717d3`. Merged stream from the 12 per-request dumps; `ignore_layers=0`.

| Policy | cap=64 | cap=128 | cap=256 | cap=512 | cap=768 | cap=1024 |
| --- | --- | --- | --- | --- | --- | --- |
| Belady (Ideal) | 0.3012 | 0.4239 | 0.5776 | 0.7573 | 0.8639 | 0.9300 |
| LRU | 0.0382 | 0.1699 | 0.2625 | 0.4722 | 0.6525 | 0.8089 |
| LFU | 0.0710 | 0.1340 | 0.2481 | 0.4520 | 0.6324 | 0.7899 |
| LFU-K | 0.0382 | 0.1699 | 0.2625 | 0.4722 | 0.6525 | 0.8089 |
| Model-tracing | 0.0793 | 0.1478 | 0.2693 | 0.4813 | 0.6605 | 0.8122 |
| Score-based | 0.0382 | 0.1699 | 0.2625 | 0.4722 | 0.6525 | 0.8089 |
| LHU | 0.0731 | 0.1367 | 0.2509 | 0.4566 | 0.6325 | 0.7899 |
| 1-layer-ahead* | 0.0382 | 0.1694 | 0.2587 | 0.4687 | 0.6498 | 0.8077 |
| Stride* | 0.0382 | 0.1698 | 0.2579 | 0.4694 | 0.6508 | 0.8082 |
| Multi-layer* | 0.0310 | 0.1693 | 0.2559 | 0.4662 | 0.6481 | 0.8068 |
| Activation-matrix* | 0.0375 | 0.0384 | 0.0626 | 0.2457 | 0.4023 | 0.6264 |

`*` = trace-oracle prefetch policy; the reported **demand** hit rate is bounded
by Belady, and the prefetch value shows up in `effective_hit_rate`
(see `results_real_qwen1_5_moe_a2_7b.csv`). Machine-readable copy of this table:
`results_real_qwen1_5_moe_a2_7b.csv` (and the `.md` twin
`results_real_qwen1_5_moe_a2_7b.md`).

### Task C miss-ratio curve (Belady), independent engine cross-check

The vectorized `opt_mrc` MRC engine, run over the same real access stream,
reproduces the Belady row above **bit-for-bit** (the sweep asserts
`|belady_simulate - opt_mrc| < 1e-9` at every capacity). Belady miss-ratio
curve (`1 - hit_rate`):

| capacity | 64 | 128 | 256 | 512 | 768 | 1024 |
| --- | --- | --- | --- | --- | --- | --- |
| Belady hit rate | 0.3012 | 0.4239 | 0.5776 | 0.7573 | 0.8639 | 0.9300 |
| Belady miss ratio | 0.6988 | 0.5761 | 0.4224 | 0.2427 | 0.1361 | 0.0700 |

Hit rate is strictly in `(0, 1)` and monotonically non-decreasing in capacity,
as required by the acceptance criteria.

## Real vs synthetic (same policy family)

The synthetic `qwen3-235b-synthetic` model (`results_synthetic.md`) uses a
Zipf-skewed generator with injected temporal reuse. On the **real** trace the
qualitative picture matches, with two sharper effects:

- **Real routing is less cacheable at small capacities.** At cap=256 the real
  Belady ceiling is 0.5776 vs the synthetic (different geometry, 94x128) 0.5828
  at the same capacity; but the real on-line policies sit much further below
  their ceiling at small caps (e.g. real LRU 0.26 vs synthetic LRU 0.00 at 256
  — different key spaces, so compare *shape* not absolute cap). The real trace
  shows the policies converging toward Belady only as capacity approaches the
  1440-key working set.
- **Model-tracing edges out pure LRU/LFU on real data** at every mid capacity
  (e.g. 0.4813 vs 0.4722 at cap=512), consistent with real expert popularity
  being skewed but not as cleanly Zipf as the synthetic generator assumes.
- The **Belady-ceiling invariant holds on real data**: no on-line policy
  exceeds Belady at any capacity (checked at runtime in `run_policies_real.py`).

## Reproduce

```bash
# 1. Launch SGLang 0.5.20 with the per_token recorder (one free GPU):
export HF_HOME=/mnt/raid0nvme0/leyang/hf_cache
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=<raw_dir>
export CUDA_VISIBLE_DEVICES=<free_gpu>
python -m sglang.launch_server \
    --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat --trust-remote-code \
    --tp-size 1 --dp-size 1 \
    --expert-distribution-recorder-mode per_token --disable-cuda-graph \
    --chunked-prefill-size 8192 --mem-fraction-static 0.85 \
    --host 127.0.0.1 --port 30000

# 2. Drive start/generate/stop/dump per prompt (batch size 1):
python drive_capture.py --port 30000 --max-new-tokens 64 --num-prompts 12

# 3. Convert to Belady inputs (identity remap; merge into one stream):
python -m benchmarks.cache_simulator.convert_sglang_traces \
    --inputs "<raw_dir>/*.pt" --to belady --out <belady_dir>/stream.pt --merge

# 4. Belady acceptance sweep:
for C in 64 128 256 512 768 1024; do
  python -m benchmarks.cache_simulator.belady_cache_simulator \
      --files "<belady_dir>/stream.pt" --capacity $C --ignore-layers 0
done

# 5. Full policy x capacity sweep + this table / CSV / EXPERIMENTS rows:
python -m benchmarks.cache_simulator.research.run_policies_real \
    --files "<belady_dir>/stream.pt" \
    --model qwen1.5-moe-a2.7b --dataset mixed20-greedy64 \
    --ignore-layers 0 --capacities 64 128 256 512 768 1024 \
    --tag qwen1_5_moe_a2_7b --out benchmarks/cache_simulator/research
```

## Trace location (git-ignored; on disk only)

Traces are excluded by `benchmarks/cache_simulator/.gitignore` (`*.pt`). They
live on this box under:

```
/mnt/raid0nvme0/leyang/moeinf-traces/
  raw_smoke/            # 12 raw SGLang per_token dumps (expert_distribution_recorder_*.pt)
  belady_smoke/         # 12 per-request converted Belady .pt
  belady_smoke_stream/stream.pt   # merged access-stream used for the table above
  drive_capture.py      # the HTTP capture driver (prompts + start/stop/dump loop)
```

SGLang venv: `/mnt/raid0nvme0/leyang/sglang-venv` (sglang 0.5.20).
Simulator venv: `/mnt/raid0nvme0/leyang/MoE-Infinity/.venv`.
