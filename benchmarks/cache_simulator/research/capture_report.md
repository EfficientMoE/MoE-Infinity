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
