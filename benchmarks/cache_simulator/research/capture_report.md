# Capture report — SGLang 0.5.20 expert-routing traces on Blackwell

Part of #289. What worked, what was blocked, and exactly how the real
expert-routing traces were captured and validated. Companion to
`results_real_qwen3_235b.md` (the hit-rate tables).

## TL;DR

- **SGLang 0.5.20 installs and runs on NVIDIA RTX PRO 6000 Blackwell (sm_120).**
  No install failure, no kernel/launch failure for the MoE smoke model.
- **The `per_token` expert-distribution recorder works** and dumps correct
  `.pt` traces. The #13206 AssertionError **did not fire** on 0.5.20's server
  dump path (it already calls `dump_record()` in file mode).
- **235B capture was blocked by GPU availability** (2 of 6 GPUs free; TP4 needs
  >=4), not by software. We completed the full Step-0 pipeline on the
  prescribed smoke model instead, which exercises the identical path.

## Hardware (verified with `nvidia-smi`)

- Box: **6x NVIDIA RTX PRO 6000 Blackwell**, 96 GB each, compute capability
  **sm_120**, driver 590.48.01, CUDA 13.1.
- GPUs **0-3**: busy the entire session — another user's `pard-hybrid` python
  process, ~43-44 GB used and 80-100% util on each. Not usable.
- GPUs **4-5**: free (idle, ~0 MiB). These are the **Max-Q Workstation
  Edition** variant (300 W cap) vs the Server Edition (600 W) on 0-3; both
  expose the full 96 GB.
- Capture ran on **GPU 4** (`CUDA_VISIBLE_DEVICES=4`), one GPU, TP1.

=> Only **2 free GPUs**. The 235B plan (`--tp 4 --dp 1`, FP8 weights ~235 GB)
needs at least 4 and realistically 4x96 GB; it could not be launched. This is
the single blocker for Step 1.

## Software / environment

- SGLang venv (separate, to avoid polluting the repo venv):
  `python3.12 -m venv /mnt/raid0nvme0/leyang/sglang-venv`,
  `pip install "sglang[all]==0.5.20"`. Installed clean.
  - Pulled `torch 2.13.0+cu130`, `flashinfer_python 0.6.18`,
    `sglang-kernel 0.4.7`, `triton 3.7.1`, `transformers 5.12.1`.
- HF cache on the big disk: `HF_HOME=/mnt/raid0nvme0/leyang/hf_cache`
  (`/mnt/raid0nvme0` had ~900 GB free). No `HF_TOKEN` was needed —
  `Qwen/Qwen1.5-MoE-A2.7B-Chat` is ungated and downloaded anonymously (an HF
  warning about unauthenticated/rate-limited requests appears but is harmless).
- Convert + simulate + benchmark ran in the **repo venv**
  `/mnt/raid0nvme0/leyang/MoE-Infinity/.venv`
  (`python 3.12.3, numpy 2.4.4, torch 2.11.0+cu130`), which has the merged
  simulator, policies, `run_policies`, `mrc`, and `convert_sglang_traces`.

## SGLang-on-Blackwell status (detail)

The smoke server launched successfully end to end:

- `Attention backend not specified. Use flashinfer backend by default.`
- `Load weight end ... type=Qwen2MoeForCausalLM`, KV cache allocated,
  `Uvicorn running on http://127.0.0.1:30000`, `The server is fired up`.
- Greedy generation produced normal text for all 12 prompts
  (gen throughput ~77 tok/s on the Max-Q card).
- One **non-fatal** note worth recording for the 235B run: SGLang logged
  `Using default MoE kernel config. Performance might be sub-optimal! Config
  file not found at .../E=60,N=1408,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Max-Q_Workstation_Edition.json`.
  SGLang falls back to a generic Triton fused-MoE config on this exact SKU.
  This affects MoE GEMM **performance only**, not routing / `topk_ids`, so the
  captured traces are valid. A tuned config could be generated via
  `benchmark/kernels/fused_moe_triton` if throughput mattered.

Because weight *quantization* (FP8 on 235B) does not change the gate's
`topk_ids`, and because routing is what the cache simulator consumes, the
Blackwell FP8/kernel situation does not threaten trace validity for the 235B
target either — only the ability to fit/launch it on 4 GPUs does.

## #13206 (per_token dump) handling

The known issue: `_DetailAccumulator.dump(output_mode)` hard-asserts
`output_mode == "file"`; it fires if the dump is invoked in object mode.

On **0.5.20 the server path is already safe**: inspected
`sglang/srt/managers/scheduler.py` — `DUMP_RECORD` calls
`get_global_expert_distribution_recorder().dump_record()` with no args, and
`_ExpertDistributionRecorderReal.dump_record(output_mode="file")` defaults to
file mode. So `GET/POST /dump_expert_distribution_record` writes the `.pt` to
`$SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR` without ever hitting object mode.

**Workaround used: none needed.** We confirmed empirically — 12
`start -> generate -> stop -> dump` cycles each produced a `.pt` on disk
(`Write expert distribution to .../expert_distribution_recorder_*.pt` in the
server log), zero AssertionErrors. We drive the server over HTTP
(`drive_capture.py`) rather than the offline `sglang.Engine`, because in 0.5.20
`Engine` exposes neither the `expert_distribution_recorder_mode` constructor
kwarg nor `*_expert_distribution_record` methods — the recorder is a
server/`ServerArgs` feature.

## Physical vs logical expert ids

- Launched with defaults: `enable_eplb=False`, `ep_size=1`,
  `ep_num_redundant_experts=0`. => no redundant experts.
- The dumped `last_physical_to_logical_map` is shape `[24, 60]` and **exactly
  identity** (verified `torch.equal(map, arange)`), so physical == logical.
- All observed expert ids are in `[-1, 59]` (`-1` = unused top_k slot, since the
  recorder buffers 8 slots but this model is top-4). Max id 59 = 60 experts.
- The converter's `remap_to_logical` therefore runs as a verified no-op; it
  would transparently handle a non-identity map on the 235B EP path.

## Pipeline validation (acceptance criteria)

1. **Schema loads**: `records` = 1 prefill (`[24, 11, 8]`) + 64 decode
   (`[24, 1, 8]`), int32, ids in `[-1, 59]`. OK.
2. **Belady sane + monotonic**: merged stream, `ignore_layers=0`:
   cap 64/128/256/512/768/1024 -> 0.3012 / 0.4239 / 0.5776 / 0.7573 / 0.8639 /
   0.9300. Strictly in (0,1) and non-decreasing. OK.
3. **Belady-ceiling invariant**: every on-line policy <= Belady at every
   capacity (asserted in `run_policies_real.py`). OK.
4. **Independent MRC cross-check**: vectorized `opt_mrc` reproduces the Belady
   row bit-for-bit (`<1e-9`). OK.
5. **Coverage**: 1440 distinct `(layer, expert)` keys = full 24x60 space;
   64 decode steps per request = `max_new_tokens`. OK.
6. **Determinism**: greedy decode is deterministic; a direct re-capture check
   was inconclusive only because SGLang's radix/prefix cache served a repeated
   prompt from cache (the "re-run" record 0 was a 1-token continuation, not the
   11-token prefill), which is a caching artifact, not routing nondeterminism.
   For a strict determinism test one would disable the radix cache
   (`--disable-radix-cache`) or vary the prompt; not required for the policy
   results.

## What was skipped and why

- **235B capture (Step 1).** Blocked: <4 free GPUs. Not attempted rather than
  risk OOM/instability on a shared box. Everything needed to run it — launch
  flags, converter, driver, simulator, report scaffolding — is in place; it is
  a matter of GPU availability, not code.
- **`.pkl` / `cache_strategy.py` export.** The Belady `.pt` path fully covers
  the deliverable (policy x capacity table + MRC). The converter's `--to pkl`
  path is available for the batch-size-1 dumps if a `cache_strategy.py` preset
  is wanted later.
- **Per-SKU tuned MoE Triton config.** Not needed for routing traces.

## Exact commands

See the "Reproduce" block in `results_real_qwen3_235b.md`. Capture used
`CUDA_VISIBLE_DEVICES=4`, `--tp-size 1`, `--expert-distribution-recorder-mode
per_token`, `--disable-cuda-graph`, `--chunked-prefill-size 8192`,
`--mem-fraction-static 0.85`, greedy (`temperature=0`, `max_new_tokens=64`),
12 prompts, batch size 1 per dump.

---

# Addendum — 2026-10-31 run: Qwen3-30B-A3B (235B-family proxy)

Follow-up capture after PR #293 merged. Goal: real traces on the **actual
235B routing family** while the 235B itself stays GPU-blocked.

## Why Qwen3-30B-A3B

`Qwen/Qwen3-30B-A3B` is the closest available proxy for
`Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`:

| field | Qwen3-30B-A3B (captured) | Qwen3-235B target |
| --- | --- | --- |
| architecture | `Qwen3MoeForCausalLM` | `Qwen3MoeForCausalLM` |
| routed experts | **128** | **128** |
| experts per token | **top-8** | **top-8** |
| MoE layers | 48 (all MoE) | ~94 (all MoE) |
| `ignore_layers` | 0 | 0 |
| weights | BF16 (~60 GB) | FP8 block-wise (~235 GB) |

The routing geometry (128 experts, top-8, all-MoE) is **identical** to the
235B; only layer count and weight precision differ. Since FP8 quantizes expert
*weights* not the gate, the 235B's `topk_ids` would come from the same
Qwen3-MoE routing — so this is a faithful proxy for the cache-key behaviour,
with the only caveat that the 235B has ~2x the layers (larger absolute
key-space 94x128=12032 vs 48x128=6144).

## Hardware / config (verified)

- GPUs **4 and 5** free (97 GB each, 0% util); GPUs 0-3 pinned by another user
  (~44 GB used, 53 GB free each, 80-94% util) the whole run — Max-Q
  Workstation Edition (300 W) on 4/5, Server Edition on 0-3; sm_120, driver
  590.48.01.
- Launched `--tp-size 2 --dp-size 1` on `CUDA_VISIBLE_DEVICES=4,5`, SGLang
  0.5.20, `--expert-distribution-recorder-mode per_token --disable-cuda-graph
  --chunked-prefill-size 8192 --mem-fraction-static 0.85`.
- **TP2 came up cleanly on sm_120** (NCCL 2.29.7, both TP0/TP1 ranks), ~35
  tok/s greedy decode. First multi-GPU SGLang run validated on these cards.

## Dataset

- **150 MMLU** questions (`cais/mmlu`, config `all`, split `test`), first 150
  in fixed dataset order (deterministic — no shuffle), each rendered as a
  4-choice MC prompt (see `drive_capture_mmlu.py`).
- Greedy (`temperature=0`), `max_new_tokens=64`, batch size 1 per dump.
- Merged decode access stream: **7,686,912** accesses, **5,796** distinct
  `(layer, expert)` keys out of the 6144 key space (94.3% coverage).

## TP2 dual-rank dump handling (new vs the TP1 smoke run)

With `--tp 2`, **both** TP ranks call `dump_record()` on each
`/dump_expert_distribution_record`, so each prompt produced **2** `.pt` files
(300 files for 150 prompts). Because attention is non-DP (Config A), the MoE
gate is replicated and the two ranks' dumps carry **identical routing** — I
verified `torch.equal` on the topk of same-prompt file pairs. I de-duplicated
by routing-content hash to exactly **150** files (each of 150 contents appeared
in exactly 2 files) before merging, so no prompt is double-counted. Dedup
filelist: `/mnt/raid0nvme0/leyang/moeinf-traces/qwen3_30b_dedup_filelist.json`.

## Physical vs logical (verified)

`last_physical_to_logical_map` is `[48, 128]` and **exactly identity**
(`ep_num_redundant_experts=0`, `enable_eplb=False`), expert ids observed in
`[0, 127]` with **no `-1` padding** (top-8 exactly fills the recorder's
8-slot buffer, unlike the 60-expert smoke model). Converter remap is a verified
no-op.

## Validation (acceptance criteria)

1. **Schema**: 65 records = 1 prefill `[48, 60, 8]` + 64 decode `[48, 1, 8]`,
   int32, ids in `[0, 127]`. OK.
2. **Belady sane + monotonic** (merged stream, `ignore_layers=0`):
   cap 256/512/1024/2048/3072/4096/6144 -> 0.7028 / 0.7898 / 0.8705 / 0.9462 /
   0.9810 / 0.9961 / 0.9992. In (0,1), non-decreasing. OK.
3. **Belady-ceiling invariant**: enforced at runtime in `run_policies_real.py`.
4. **MRC cross-check**: `opt_mrc` reproduces the Belady row bit-for-bit.
5. **Coverage**: 5,796 / 6,144 keys (94.3%); 64 decode steps/request.

## 235B status (this run): GPU-PENDING

Re-checked `nvidia-smi` at the **start** of this run and **again after** the
30B capture (the two mandated checks). Both times GPUs 0-3 had only ~53 GB free
each at 80-94% util; only GPUs 4-5 were free. That is **2 usable GPUs**, below
the >=4 (each >=90 GB free) needed for `Qwen3-235B-A22B-Instruct-2507-FP8`
`--tp 4 --dp 1`. The 235B capture was therefore **not run** and remains
**GPU-pending** — purely a shared-box availability issue, not a software or
pipeline limitation. The exact 235B launch recipe is in
`collect_traces_sglang.md`; the 30B proxy here already validates the full
128-expert/top-8 Qwen3-MoE routing path end to end, including multi-GPU TP.

## Exact commands (30B run)

```bash
export HF_HOME=/mnt/raid0nvme0/leyang/hf_cache
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=<raw_dir>
export CUDA_VISIBLE_DEVICES=4,5
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-30B-A3B --trust-remote-code \
    --tp-size 2 --dp-size 1 \
    --expert-distribution-recorder-mode per_token --disable-cuda-graph \
    --chunked-prefill-size 8192 --mem-fraction-static 0.85 \
    --host 127.0.0.1 --port 30000
python drive_capture_mmlu.py --port 30000 --max-new-tokens 64 --num-prompts 150
# de-dup 300 -> 150 by routing content, then:
python -m benchmarks.cache_simulator.convert_sglang_traces \
    --inputs "<dedup_dir>/*.pt" --to belady --out <belady_dir>/stream.pt --merge
python -m benchmarks.cache_simulator.research.run_policies_real \
    --files "<belady_dir>/stream.pt" --model qwen3-30b-a3b \
    --dataset mmlu150-greedy64 --ignore-layers 0 \
    --capacities 256 512 1024 2048 3072 4096 6144 \
    --tag qwen3_30b --out benchmarks/cache_simulator/research
```
