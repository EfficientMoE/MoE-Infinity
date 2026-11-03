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

---

# Addendum — 2026-11-01 run: Qwen3-235B-A22B-Instruct-2507-FP8 (THE TARGET)

The 235B window finally opened (all 6 GPUs free). This is the real capture of
the #289 target model, no longer a proxy.

## GPU availability — unblocked

At the start of this run `nvidia-smi` showed **all 6** RTX PRO 6000 Blackwell
GPUs free (~0 MiB, 0% util), 767 GB free on `/mnt/raid0nvme0`. Used GPUs
**0,1,2,3** (`CUDA_VISIBLE_DEVICES=0,1,2,3`) for `--tp 4`. (`num_key_value_heads
= 4`, so TP4 divides the KV heads exactly; TP6 would not.)

## Launch / download

- `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` is **ungated** — the config and
  weights downloaded anonymously (no `HF_TOKEN`); the usual HF
  unauthenticated-rate-limit warning is harmless. **No auth blocker.**
- `--tp-size 4 --dp-size 1 --expert-distribution-recorder-mode per_token
  --disable-cuda-graph --chunked-prefill-size 8192 --mem-fraction-static 0.85`,
  `SGLANG_ENABLE_JIT_DEEPGEMM=1`, SGLang 0.5.20.
- **TP4 came up cleanly on sm_120**: all 4 ranks init torch distributed,
  "Detected fp8 checkpoint", weights streamed to ~85 GB/GPU incl. KV cache,
  "The server is fired up", healthy. Greedy decode ~12 tok/s (no tuned
  per-SKU MoE config; routing unaffected). This is the first 4-GPU FP8 run
  validated on these cards.

## Geometry / id verification (from config.json + dumps)

| field | value |
| --- | --- |
| architecture | `Qwen3MoeForCausalLM` |
| decoder layers | 94 (all MoE, `mlp_only_layers=[]`) ⇒ `ignore_layers=0` |
| routed experts | 128 |
| experts per token | 8 (fills the recorder's 8-slot buffer; no `-1` padding) |
| FP8 | e4m3 block-wise (128x128); `mlp.gate` NOT quantized ⇒ routing is exact |
| cache-key space | 94 x 128 = 12032 |

Dumps: `topk_ids_of_layer` is `[94, seq_len, 8]` int32, expert ids in
`[0, 127]`, `last_physical_to_logical_map` is `[94, 128]` and **identity**
(`ep_num_redundant_experts=0`) ⇒ physical == logical, converter remap is a
verified no-op. Record 0 prefill (`forward_mode=1`), records 1..64 decode.

## Dataset

- **100 MMLU** questions (`cais/mmlu` all/test, first 100 in fixed order),
  4-choice MC prompts, greedy (`temperature=0`), `max_new_tokens=64`, batch
  size 1 per dump.
- `--tp 4` → all 4 ranks dump the (replicated, non-DP) routing: 100 prompts
  produced 397 raw `.pt` (97 prompts x4 + 3 prompts x3, a late-flush race on
  the final prompt dropped 3 rank copies but every prompt kept >=3 identical
  copies). De-duplicated by routing-content hash to **100** (verified all 100
  have 65 records each).
- Converted + merged into one access-stream `.pt`.

## Decode-only vs merged-stream pitfall (important)

A per-request dump has `records = [prefill, decode_1..decode_T]`. Merging 100
requests gives `records` with **one prefill pass per request** (100 prefill +
6400 decode = 6500 records here). The naive replay
(`extract_access_sequence(include_prefill=False)`) drops only the **global**
`records[0]`, so 99 prefill passes stay in the stream — ~5.19M of the ~10.0M
accesses. Those inflate the "decode" hit rate (e.g. cap=512 reads **0.6987**
with the contamination vs the correct **0.4898**). **Do not use the merged
stream's naive replay for decode numbers.** The correct decode-only stream
keeps every `seq_len==1` record and drops all prefill passes:
**4,812,800** accesses, **10,551 / 12,032** distinct keys (87.7%).

## Validation (acceptance criteria)

1. **Schema**: 65 records = 1 prefill `[94, 60, 8]` + 64 decode `[94, 1, 8]`,
   int32, ids in `[0, 127]`. OK.
2. **Belady sane + monotonic** (decode-only, `ignore_layers=0`):
   cap 256/512/1024/2048/4096/8192 -> 0.3067 / 0.4898 / 0.6581 / 0.8082 /
   0.9320 / 0.9947. In (0,1), non-decreasing. OK. (Reproduced bit-for-bit by a
   per-record `seq_len==1` decode filter and by the vectorized `opt_mrc`.)
3. **Belady-ceiling invariant**: enforced by the decode-only policy study.
4. **Coverage**: 10,551 / 12,032 keys (87.7%); 64 decode steps/request.

## 235B status: CAPTURED (full policy study on `feat/cache-policy-lcnu`)

The 235B target is now captured end to end on real hardware — the #289 "235B
GPU-pending" status from PRs #293/#294 is resolved. This document records the
**capture provenance** and the verified decode-only Belady ceiling. The full
decode-only **policy x capacity study** on this capture (many eviction policies
+ a new LCNU policy, Belady asserted) lives on branch `feat/cache-policy-lcnu`
(`research/results_real_qwen3_235b_decode.md`, `research/decode_policy_study.md`),
along with the per-record decode-detection fix to `run_policies_real.py` /
`mrc.py` that makes merged-stream replays decode-correct. The contaminated
merged-stream policy table was deliberately **not** published.

## Exact commands (235B run)

```bash
export HF_HOME=/mnt/raid0nvme0/leyang/hf_cache
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=<raw_dir>
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 --trust-remote-code \
    --tp-size 4 --dp-size 1 \
    --expert-distribution-recorder-mode per_token --disable-cuda-graph \
    --chunked-prefill-size 8192 --mem-fraction-static 0.85 \
    --host 127.0.0.1 --port 30000
python drive_capture_mmlu.py --port 30000 --max-new-tokens 64 --num-prompts 100
# de-dup 397 -> 100 by routing content, then convert + sweep:
python -m benchmarks.cache_simulator.convert_sglang_traces \
    --inputs "<dedup_dir>/*.pt" --to belady --out <belady_dir>/stream.pt --merge
python -m benchmarks.cache_simulator.research.run_policies_real \
    --files "<belady_dir>/stream.pt" --model qwen3-235b \
    --dataset mmlu100-greedy64 --ignore-layers 0 \
    --capacities 512 1024 2048 4096 6144 8192 12032 \
    --tag qwen3_235b --out benchmarks/cache_simulator/research
```
||||||| parent of fb72ad5 (docs(benchmarks): granite3.1-moe-3b real-trace results — third model, third capture stack)
---

# Addendum — granite3.1-moe-3b capture on Apple Silicon (llama.cpp), third capture stack

Companion to `results_real_granite3_1_moe_3b_{decode,union}.{md,csv}`. Extends
the real-trace series (#293 qwen3-235b, #294 qwen3-30b-a3b + qwen1.5-moe-a2.7b,
all SGLang `per_token` captures) with a **small-E model captured by a different
stack**: an observation-only llama.cpp eval-callback tracer on CPU decode
(Apple M4, 24 GB, no GPU), running ollama's granite3.1-moe:3b GGUF
(sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b).

## Capture

- Tracer: llama.cpp `examples/exp646-trace/trace.cpp` eval callback on
  `ffn_moe_topk-*` tensors; writes CSV rows
  `rowid,prompt_id,n_tokens,layer_name,e1..e8`. Observation-only: no model
  behavior change (same binary on/off verified byte-identical output).
- 8 hand-crafted short prompts (PROMPTS.txt sha256
  43703bef14f0ce4f6b290c23280fdac7b99137f9bd640498b765ef9c0572bc75),
  greedy n_predict=512; model determinism pre-verified (two identical runs).
- Trace: 89,828 rows; decode stream = 2,493 forward passes x 32 layers x 8
  experts = 638,208 accesses; 1,262 of 1,280 (layer, expert) keys touched.

## Conversion + validation

- Converted to the SGLang `per_token` record format (one record per decode
  step, `topk_ids_of_layer[32,1,8]`, forward_mode=2; one extend record per
  prompt prefill; identity physical->logical map).
- Reconciliation gates (all exact): per-prompt decode-step counts equal the
  sealed per-prompt token counts; total 2,493 steps / 638,208 accesses;
  simulator-side parse independently reports 32 layers / 1,262 keys /
  638,208 accesses.

## Known capture artifacts (documented, not repaired)

- **Layer-31 prefill gap**: the callback never reports `ffn_moe_topk-31` while
  a batch has n_tokens>1 (split prefill graph); each prefill contributes 31
  full layers plus ONE lone layer-31 row at n_tokens=1 (reclassified into the
  prefill record; other layer-31 prefill slots use the -1 "unused" sentinel
  both `extract_access_sequence` and `union_prefill_sequence` skip). The union
  arm therefore undercounts layer-31 first-use fetches by at most 256 of
  ~647k accesses. Decode captures all 32 layers.

## Headline findings (see results files for full tables)

- **Zero-hit sub-band replicates**: LRU = 0.0000 at every capacity <= 128
  (= layers x top_k / 2); sharp cliff to 0.3503 at 256 = the per-step working
  set (layers x top_k). The *inclusive* zero band up to and including
  layers x top_k does NOT hold for this small-E model — consistent with the
  prior series itself (qwen3-30b: LRU 0.3053 at its 384 boundary; a2.7b:
  no zero band at all).
- **LCNU parameter transfer holds cross-model AND cross-capture-stack**: with
  the qwen-tuned defaults and no re-tuning, lcnu is the best online policy at
  7 of 8 capacities (second at 1024 by 0.0024).
- **Belady floors**: 0.6458 at 256 slots (>= 0.4242 static top-8 floor) and
  0.8547 at 512 (>= 0.6785 top-16 floor).

---

# Addendum — qwen3-30b-a3b capture on x86_64 (llama.cpp tracer), fourth capture
stack, first big-E (E=128) llama.cpp arm

## Capture

- Model: Qwen3-30B-A3B (GGUF v3, sha256-58574f2e…, 18,556,685,856 B, digest
  verified pre-run), MoE config 48 layers / 128 experts / top_k 8 (k/E = 6.25%,
  vs granite-3b 20% and qwen3-235b 3.1%).
- Stack: llama.cpp @ cd8427d0 with the expert-routing tracer (src sha
  5c9d38bb…) — same tracer as the granite3.1-moe-3b addendum, now on
  Linux/x86_64 (WSL2) instead of Apple Silicon. Greedy decoding pinned
  (--temp 0 --top-k 1 --top-p 1.0 --repeat-penalty 1.0 -c 4096), batch 1,
  -n 512.
- Dataset: handcrafted8-greedy512 (same 8 prompts as the granite arm; sha of
  PROMPTS 43703bef…). Per-prompt decode steps 512×7 + 126 (p3 lawful greedy
  EOS at 126 tokens — disclosed, not repaired); prompt lens 20–44.
- Host: single x86_64 host end-to-end (capture AND conversion substrate). The
  exp647 cross-ISA check showed expert traces are NOT byte-reproducible across
  M4 (arm64) and x86_64 under pinned greedy (81.3% byte-diff, first order-flip
  at rowid 21): near-tie router logits flip across ISAs. Rule adopted: this
  research line's traces are captured, converted and consumed on ONE host;
  cross-host trace reuse is invalid methodology.
- Determinism (same host): two back-to-back decode runs of prompt 0 produced
  byte-identical traces (sha 0a5c6f2d… ×2, 26,128 rows each).

## Conversion + validation

- Converter: parametrized copy of the granite converter (identical logic,
  NL=48/NE=128). 191,389-row trace -> 3,718 records (3,710 decode
  [48,1,8] forward_mode=2 + 8 prefill [48,L,8] forward_mode=1), identity
  physical->logical map.
- Reconciliation gates (all exact, matching the sealed capture verdict):
  3,710 clean decode forwards (= (178,088 − 8 lone rows)/48);
  1,424,640 decode accesses (= 3,710 × 48 × 8); records 3,718;
  distinct decode keys 5,948 ≤ 6,144 (= 48×128; 96.8% of the layer×expert key
  space touched across only 8 prompts — the big-E regime the granite arm
  couldn't probe).

## Known capture artifacts (documented, not repaired)

- **Layer-47 prefill gap** (same class as granite's layer-31): each prefill
  contributes 47 full layers plus ONE lone layer-47 row at n_tokens=1
  (reclassified into the prefill record; remaining layer-47 prefill slots use
  the -1 sentinel that `extract_access_sequence`/`union_prefill_sequence`
  skip). Union arm undercounts layer-47 first-use fetches by at most 8×128 of
  ~1.44M accesses. Decode captures all 48 layers.
- **p3 early EOS**: 126 decode steps (greedy-converged), consistent with the
  sealed exp647 verdict (legitimate EOS, not capture failure).

## Capacity grid + run disclosure

Both arms use the series' canonical qwen3-30b grid from #294 (capacities =
multiples of E: 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096),
chosen because the cliff hypothesis (H1) is stated at 3×E = 384. A first
decode sweep was accidentally launched on the harness default grid
(64…2048, no 384) — a run-configuration error, discovered before any
analysis; those outputs were snapshotted (sha256 in the provenance record)
and discarded, EXPERIMENTS.md reverted, and both arms re-run end-to-end on
the canonical grid. No hypothesis was edited after seeing data (PREREG
frozen 2026-11-03T01:26:31Z, sha 416bbe06…); all numbers below come from
the re-run.

## Headline findings

Frozen hypotheses (PREREG sha 416bbe06…, frozen 2026-11-03T01:26:31Z, before any
sweep): verdict **PASS 3/4**.

- **H1 CONFIRMED — the zero-band cliff scales with E.** LRU hit rate is
  exactly 0.0000 at every capacity ≤ 256 (= 2×E) and jumps to 0.3043 at 384
  (= 3×E) — inside the preregistered [0.25, 0.40] window. Granite (E=40,
  k/E=20%) zero band ended at 128 with cliff 0.3503@256; qwen3-30b (E=128,
  k/E=6.25%) zeroes out twice as far relative to E, and the cliff lands at
  3×E. Pattern across three models: the useless-cache band is set by the
  active working set, which tracks E at fixed k.
- **H3 CONFIRMED — belady floor at the cliff.** belady @384 = 0.6012
  (window [0.55, 0.75]): even the offline optimum only reaches ~60% at the
  cliff capacity, i.e. at 3×E roughly 40% of decode accesses are
  first-touch-of-run and unrecoverable by any cache policy. Same shape as
  granite (0.6458@256) — a policy-independent floor for sizing "useful"
  expert caches.
- **H4 CONFIRMED — prefill barely widens the key set.** union vs decode
  distinct keys: +34 (window [0, 64]); union LRU first-nonzero capacity
  384 = decode's cliff. Prompt prefill adds almost no long-lived experts
  beyond what decode touches: for sizing purposes the decode trace is the
  signal; caching prefill-derived keys buys nothing at small capacities.
- **H2 REFUTED at large capacities — LCNU is a small/medium-budget policy.**
  LCNU is the best online policy at every capacity ≤ 1024 (= 8×E): +0.2083
  over LRU at cap 128 (0.2083 vs 0.0000!), +0.11 at the 384 cliff, and
  still +0.03 at 1024. But it does NOT dominate everywhere: at ≥ 1536 LRU
  recency wins (2048: LRU 0.8873 vs LCNU 0.8648) and s3fifo takes
  best-online at 1536/2048/3072 (e.g. 0.8083 vs LRU 0.7877 at 1536).
  This replicates exp650's granite result (LCNU best online 7/8 caps, not
  universal) on a second architecture and a 3.2× larger expert count, and
  sharpens it into a regime rule: **LCNU's layer-cyclic utility pays below
  ~8×E; above that, plain recency (LRU/s3fifo) is as good or better.**

Full per-capacity table (hit rates, decode arm):

| cap | LRU | LCNU | belady | best online |
|----:|----:|----:|----:|:--|
| 128 | 0.0000 | 0.2083 | 0.3067 | lcnu |
| 256 | 0.0000 | 0.3273 | 0.4977 | lcnu |
| 384 | 0.3043 | 0.4148 | 0.6012 | lcnu |
| 512 | 0.4266 | 0.4853 | 0.6722 | lcnu |
| 768 | 0.5486 | 0.5905 | 0.7682 | lcnu |
| 1024 | 0.6413 | 0.6685 | 0.8327 | lcnu |
| 1536 | 0.7877 | 0.7871 | 0.9124 | s3fifo 0.8083 |
| 2048 | 0.8873 | 0.8648 | 0.9532 | s3fifo 0.8956 |
| 3072 | 0.9679 | 0.9445 | 0.9856 | s3fifo 0.9685 |
| 4096 | 0.9894 | 0.9780 | 0.9935 | score/lru 0.9894 |

Verdict artifact: `exp651-verdict.json` (frozen analyzer output). Analyzer
note: the preregistered H2 online-filter followed the CSV's
`is_oracle_upper_bound` flag, which does not flag `belady` itself; `belady`
(the offline optimum) was therefore excluded from the online set explicitly —
filter repair only, hypothesis text unchanged.

---

## Addendum 2 — crossover localization (fine capacity sweep, exp652)

Pre-registered before the sweep (sha 1c0be3bf…, frozen 2026-11-03T03:35:29Z) on
the same frozen trace (no recapture): decode arm at capacities 7E..14E step E
(896..1792). Verdict PASS 3/4; sharpens the regime rule to an exact boundary:

- LCNU is strictly best online at 896, 1024, 1152 — its **last crown is 9E**.
- s3fifo takes over at exactly **1280 = 10E** (0.7420 vs LCNU 0.7328) and holds
  every capacity through 1792 = 14E.
- LCNU keeps beating plain LRU until **1536 = 12E** (H4 refuted: flip at 12E,
  not 10-11E) — losing the crown and losing to LRU are different events,
  two E-steps apart.
- s3fifo >= LRU at every fine capacity (e.g. 0.6026 vs 0.5990 at 896): the
  deployable rule collapses to **LCNU below 10E, s3fifo at/above 10E** —
  s3fifo never loses to LRU in the swept range.

| cap (=E×) | LRU | LCNU | s3fifo | belady | best online |
|----:|----:|----:|----:|----:|:--|
| 896 (7E) | 0.5990 | 0.6318 | 0.6026 | 0.8033 | lcnu |
| 1024 (8E) | 0.6413 | 0.6685 | 0.6547 | 0.8327 | lcnu |
| 1152 (9E) | 0.6808 | 0.7018 | 0.7005 | 0.8577 | lcnu |
| 1280 (10E) | 0.7195 | 0.7328 | **0.7420** | 0.8788 | s3fifo |
| 1408 (11E) | 0.7544 | 0.7614 | **0.7773** | 0.8970 | s3fifo |
| 1536 (12E) | 0.7878 | 0.7871 | **0.8083** | 0.9124 | s3fifo |
| 1664 (13E) | 0.8172 | 0.8100 | **0.8354** | 0.9253 | s3fifo |
| 1792 (14E) | 0.8449 | 0.8301 | **0.8581** | 0.9362 | s3fifo |

---

## Addendum 3 — cross-model replication of the crossover rule (exp653): PARTIAL, rule corrected

Pre-registered transfer test (sha f21c10af…, frozen 2026-11-03T04:14:59Z) of
exp652's 10E boundary to three more models — two captured on a different stack
(SGLang) and two on different datasets (mixed20-greedy64, mmlu100-greedy64).
Verdict PARTIAL: the transfer half holds everywhere, the flip half is refuted
on all three models. This corrects addendum 2's deployable rule.

- **Holds (all models):** LCNU strictly best online at 7E..9E (granite fine
  sweep: +3.0 to +10.7pp over s3fifo at 280..360); lcnu > s3fifo at <= 8E on
  a2.7b and 235b.
- **Refuted:** no s3fifo takeover at 10E..14E on granite — LCNU stays strictly
  best at every fine capacity (400=10E: 0.6291 vs 0.5399, +8.9pp) — and lcnu
  > s3fifo through the full tested range on a2.7b (<= 17.1E) and 235b
  (<= 16E; 12E: 0.5840 vs 0.5128).
- **Where granite actually flips:** only at 1024 = 25E (0.9732 vs 0.9708,
  +0.24pp) with everything else saturated at 32E (0.998) — a near-saturation
  flip, unlike qwen3-30b's mid-curve flip at 10E (+0.92pp at 1280, growing to
  +2.8pp by 14E).

Corrected guidance: **LCNU remains the best online default across all four
models and all three capture stacks**; the s3fifo takeover at 10E is a
qwen3-30b-specific property, not a law. What makes that model's trace flip
mid-curve (layer count? absolute working set? capture length?) is an open
question — flagged as the next experiment in this series.

Granite fine sweep (this addendum, decode arm, same frozen trace as #340):
20 policies x 8 capacities, results below; EXPERIMENTS.md rows appended.

| cap (=E×) | LRU | LCNU | s3fifo | belady | best online |
|----:|----:|----:|----:|----:|:--|
| 280 (7E) | 0.4677 | 0.4982 | 0.4049 | 0.6739 | lcnu |
| 320 (8E) | 0.4691 | 0.5449 | 0.4475 | 0.7154 | lcnu |
| 360 (9E) | 0.4826 | 0.5883 | 0.4914 | 0.7515 | lcnu |
| 400 (10E) | 0.5244 | **0.6291** | 0.5399 | 0.7832 | lcnu |
| 440 (11E) | 0.5659 | **0.6672** | 0.5895 | 0.8114 | lcnu |
| 480 (12E) | 0.6113 | **0.7024** | 0.6393 | 0.8367 | lcnu |
| 520 (13E) | 0.6431 | **0.7352** | 0.6856 | 0.8589 | lcnu |
| 560 (14E) | 0.6848 | **0.7654** | 0.7265 | 0.8785 | lcnu |

---

## Addendum 4 — the 10E flip is qwen3-30b-structural, not capture-length (exp654): REFUTES-LENGTH

Addendum 3 left the flip's root cause open, with per-prompt decode length
(512 vs 64 tokens) the leading confound: every non-flipping model was captured
at greedy64. Pre-registered test (sha 887b827d…, frozen 2026-11-03T04:28:07Z):
truncate the frozen 30b trace to greedy64-equivalent (8 prompts x 64 decode
steps; frozen source 08732f27 untouched; derived artifact 3998785e) and re-run
the identical 7E..14E fine grid.

- **H2 control holds:** LCNU strictly best at 896/1024/1152 (7E..9E).
- **H1 refuted:** s3fifo STILL strictly best at 1408 (11E, 0.7523 vs 0.7482)
  and 1536 (12E, 0.7821 vs 0.7731). The flip does not vanish — it shifts one
  E-step later and its margin shrinks (+0.4pp at 11E vs +0.9pp at 10E on the
  full trace).
- **Pre-committed conclusion (H3):** decode length is NOT the driver. Length
  modulates the flip's magnitude; existence is structural to the qwen3-30b
  expert-access stream. Combined with addendum 3: the mid-curve s3fifo
  takeover is a model-structural property observed only in qwen3-30b-a3b so
  far (k/E 6.25% with 48 layers; 235b shares k/E but has 94 layers and does
  not flip through 64E).

Remaining structural candidates (next in series): layer count relative to
expert count (interleave depth per layer), absolute per-layer working set,
and expert-reuse skew. Data: results_real_qwen3-30b-a3b-g64.{csv,md},
EXPERIMENTS.md rows appended.

| cap (=Ex) | LRU | LCNU | s3fifo | belady | best online |
|----:|----:|----:|----:|----:|:--|
| 896 (7E) | 0.6029 | **0.6257** | 0.5870 | 0.7963 | lcnu |
| 1024 (8E) | 0.6456 | **0.6608** | 0.6365 | 0.8234 | lcnu |
| 1152 (9E) | 0.6796 | **0.6910** | 0.6779 | 0.8466 | lcnu |
| 1280 (10E) | 0.7123 | **0.7203** | 0.7171 | 0.8671 | lcnu |
| 1408 (11E) | 0.7422 | 0.7482 | **0.7523** | 0.8848 | s3fifo |
| 1536 (12E) | 0.7721 | 0.7731 | **0.7821** | 0.8990 | s3fifo |

---

## Addendum 5 — mechanism probe: demand imbalance separates the flipper (exp655): SUPPORTED

Addendum 4 excluded capture length. Remaining candidates: spatial demand
imbalance (expert concentration) and temporal hot-set rotation. Pre-registered
test (sha 2eb17537…, frozen 2026-11-03T04:37:39Z) on the two local frozen
traces, granite size-matched by step subsampling (per-expert sample depth
equalized; seeds 654321-23, sign required on all three):

- **S1 spatial (Gini of per-(prompt,layer) expert counts):** qwen3-30b 0.706
  vs granite-matched 0.469 — the flipper's demand is far more concentrated.
- **S2 temporal (total-variation distance between per-layer first/second-half
  decode distributions):** qwen3-30b 0.203 vs granite-matched 0.148 — the
  flipper's hot set rotates more within a single generation.
- **S3 footprint (distinct experts touched per layer, observational):**
  granite 37.7/40 (94%, near-uniform) vs qwen3-30b 85.2/128 (66.5%,
  concentrated).

Verdict **SUPPORTED** (P1 and P2 hold on all three matched seeds). Mechanism,
consistent with addenda 2-4: LCNU's layer-fair allocation is optimal while
demand is near-uniform across experts (granite: LCNU best through 19E, flip
only near saturation); once demand concentrates on a subset of experts AND
the hot set rotates within a generation (qwen3-30b), global recency
(s3fifo) exploits the concentration and takes over at ~10E, with the margin
modulated by decode length (addendum 4). Caveats: two models, single traces;
a2.7b/235b streams not measured locally (their non-flip is consistent with
this story but unverified). Deployable synthesis stands: LCNU default;
expect a recency takeover only for concentrated, rotating expert demand.

---

## Addendum 6 — within-model prompt-subset test: NOT-SEPARATED (exp656)

exp655's cross-model imbalance gap invites a causal reading; this
pre-registered test (sha 6691daf8…, frozen 2026-11-03T04:40:02Z) checks it
within one model. The frozen 30b trace's 8 prompts split by per-prompt Gini:
HIGH = {2,4,6,7} (mean 0.742), LOW = {0,1,3,5} (mean 0.669); identical fine
grid re-run on each 4-prompt subset.

- H1 held: HIGH flips — s3fifo strictly best at 1280..1792 (10E..14E).
- H2 refuted: LOW ALSO flips — s3fifo edges lcnu at 1152 (0.6567 vs 0.6560),
  then wins decisively from 1280 (0.6975 vs 0.6861) through 1792
  (0.8147 vs 0.7853).
- H3 refuted (control): the 7E..9E lcnu sweep breaks on LOW at 9E.

Verdict NOT-SEPARATED, per the pre-committed interpretation: per-prompt
demand imbalance does NOT predict the flip within qwen3-30b. Across
exp652-656 the flip boundary is stable at ~10E for every stream variant
(full 8-prompt, greedy64-truncated, HIGH subset, LOW subset) — the takeover
is a property of the router's expert-load structure itself, inherited by
every prompt mix, not of capture length (addendum 4) or prompt-level demand
concentration (this addendum). The cross-model Gini gap (addendum 5) remains
descriptively valid but is not the causal driver. Remaining question narrows
to model-level router properties (load-distribution shape at E=128, k=8,
48 layers — 235b shares E and k at 94 layers and does not flip through 64E).
Data: results_real_qwen3-30b-a3b-{hi,lo}.{csv,md}, EXPERIMENTS.md rows
appended.

---

## Addendum 7 — union-arm fine sweep: the 10E boundary is accounting-invariant (exp657): PASS

exp652's 10E boundary was measured on the decode-only stream. Deployments
that keep prefill-activated experts resident see the union stream, where
exp651's coarse grid hinted a different late regime (LRU over LCNU by 12E).
Pre-registered fine union sweep (sha 904d87e9…, frozen 2026-11-03T05:14:22Z),
same frozen trace, grid 896..1792:

- **H1 holds:** LCNU strictly best at 896/1024/1152 (7E..9E).
- **H2 holds:** best online at 1536/1664/1792 is s3fifo, not LCNU.
- **H3 holds:** first non-LCNU capacity is exactly **1280 = 10E** — identical
  to the decode arm. s3fifo 0.7342 vs LCNU 0.7296 at the boundary, gap
  widening with capacity.

The takeover boundary is invariant to prefill/decode accounting, reinforcing
addendum 6's conclusion that it is a router-structure property, not an
artifact of how the stream is counted — and making the operational guidance
clean: the same ~10E crossover applies whether or not prefill experts are
cached. Data: results_real_qwen3-30b-a3b-fine-u.{csv,md}, EXPERIMENTS.md rows
appended.
