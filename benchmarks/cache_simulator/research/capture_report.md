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

---

## Addendum 8 — two-oracle decomposition: allocation structure does NOT explain the flip; static popularity is a weak predictor (exp659): NOT-STRUCTURAL

Pre-registered (sha 280ed822..., frozen in exp659 ARTIFACT.sha256) hindsight test:
at each capacity, compare a PER-LAYER oracle (top C/L keys per layer — the
ideal layer-fair allocation LCNU approximates) against a GLOBAL oracle
(top C keys overall — what global-recency policies approximate). If the
flip were an allocation-structure phenomenon, per-layer should dominate on
granite and lose on 30b near 10E.

**Result: NOT-STRUCTURAL — H2 refuted.** The global oracle beats the
per-layer oracle at EVERY capacity on BOTH traces (30b 896..1792:
+0.15 to +0.24pp; granite 280..560: +0.19 to +0.42pp). Allocation
structure alone never favors layer-fair placement, even on the model where
LCNU wins online. Per the pre-commit, no post-hoc rescue: the mechanism
question stays open at the allocation level.

**The contrast the decomposition exposes — static popularity vs online
tracking, per model (decode arm):**
- qwen3-30b @1280 (=10E): per-layer 0.5482, global 0.5497 vs online LRU
  0.7195, LCNU 0.7328, s3fifo 0.7420, belady 0.8788. EVERY online policy
  beats the static-popularity ceiling by 17+ pp.
- granite @400 (=10E): per-layer 0.5799, global 0.5819 vs online LRU
  0.5244, s3fifo 0.5399, LCNU 0.6291, belady 0.7832. Static popularity
  BEATS online LRU and s3fifo by 4-6pp; only LCNU clears it (+4.7pp).

On 30b, aggregate popularity knowledge is worth little because demand
rotates too fast to plan statically — all the hit rate lives in temporal
tracking. On granite, popularity is far more predictive (rotation slower,
cf. addendum 5 TVD 0.148 vs 0.203): naive recency (LRU/s3fifo) actually
UNDERPERFORMS the static optimum, and LCNU's edge is precisely that it
exploits static structure while remaining online. This reframes the flip:
what separates 30b from the non-flippers is rotation rate, not static
skew — and it gives the static-vs-online gap (17pp vs 4.7pp) as a direct,
model-comparable rotation-rate proxy.

---

## Addendum 9 — hot-set rotation half-life: direct measurement of the flip driver (exp660): SUPPORTED

exp659 inferred rotation rate from static-vs-online gaps; this
pre-registered analysis (sha 97c77dfd..., frozen in exp660 ARTIFACT.sha256)
measures it directly. Decode streams windowed at W=64 records; hot set per
window = top-N keys (N = each model's 10E capacity: 1280 / 400);
J(d) = mean Jaccard(H_w, H_{w+d}).

- qwen3-30b: J(1) = 0.5473, decay to 0.27 by d=8; **half-life 1.60 windows
  = 102 decode steps**.
- granite: J(1) = 0.6339, decay to 0.45 by d=8; **half-life 4.59 windows =
  294 decode steps**.
- All three pre-registered hypotheses hold: 30b rotates faster (H1),
  half-life 2.9x shorter (H2), and rotation ordering matches the
  static-vs-online gaps from addendum 8 (18.3pp vs 4.7pp, H3).

**Mechanism, closed loop (addenda 2-9):** the s3fifo takeover at 10E is
qwen3-30b-specific because its hot set at the flip capacity rotates ~3x
faster than granite's — cheap global recency tracks fast rotation better
than layer-fair accounting (LCNU), while under slow rotation LCNU's
exploitation of persistent per-layer structure wins. Not E-scaling
(addendum 3), not decode length (4), not prompt mix (6), not accounting
mode (7), not static allocation structure (8): temporal rotation rate is
the discriminating variable, now measured rather than inferred.

Operational rule for practitioners: estimate your router's hot-set
half-life from a short probe (the windowed-Jaccard method above needs only
a few hundred decode steps, no labels); if it is short relative to
capacity-refill time, prefer global-recency eviction (s3fifo/LRU) at
>=10E; if long, LCNU-style layer-fair policies dominate at every capacity
we tested.

---

## Addendum 10 — prefix-probe validation: short probes rank correctly but misestimate magnitude (exp661): FAIL

Addendum 9's practitioner rule assumed a few hundred decode steps suffice
to estimate half-life. Pre-registered prefix test (sha ef9fdbe2..., frozen
in exp661 ARTIFACT.sha256): recompute the rotation probe from prefixes
P=256/512 of the frozen decode streams.

- H1 holds: ordering survives at P=512 — half-life(30b) 1.165 windows <
  half-life(granite) 2.376.
- H3 fails: granite prefix-512 half-life 2.376 vs full-stream 4.59 — a 48%
  underestimate, outside the pre-registered ±35% bound (30b: 1.165 vs
  1.60, -27%, within bound). H2 fails on the strict reading (granite P256
  censored at exactly 3.0 windows with J(3)=0.5038).
- Verdict FAIL per the pre-commit; no post-hoc rescue.

**Substance:** short probes get the ORDER right but not the MAGNITUDE.
Granite's rotation is nonstationary — early decode steps rotate ~2x faster
than the stream average (the probe sees mostly early steps, so it
overestimates rotation speed). Two practical corrections to the
addendum-9 rule: (1) a >=512-step probe is sufficient to CHOOSE between
fast/slow rotation regimes (ordering is robust), but (2) half-life
MAGNITUDES — and any threshold calibrated on them — need the full serving
window, and rotation speed itself drifts over generation. Also visible:
30b's prefix J(d) is U-shaped (min at d~3, partial recovery by d~6),
consistent with prompt-cycle periodicity rather than pure noise decay.

## Addendum 11 — exp662: RotHead (rotation-adaptive eviction) — VERDICT: FAIL

Prereg `477fadd4` frozen before the sweep. RotHead (`research/policies_rothead.py`)
estimates adjacent-window set-Jaccard J(1) online (2048-access windows,
re-evaluated every 4096 accesses, threshold 0.59 = midpoint of exp660's two
full-stream J(1) values) and governs each segment with S3FIFO when J(1) is
low (fast rotation) else LCNU. Both inner policies are simulated in parallel;
hits are attributed per segment to the chosen inner (residency-migration cost
at switches not modeled; switch counts reported). Decoded-arm canonical grid,
frozen traces 30b `08732f27` / granite `7f8e4303`; baselines from the frozen
decode CSVs. Smoke test at 30b/1280 disclosed in the prereg. All 15 re-simulated
rothead rates reproduced the sweep CSVs to the last stored digit.

**All three preregistered hypotheses refuted — verdict FAIL, no post-hoc rescue.**

Qwen3-30B-A3B (hit rates %; Δ = rothead − max(lcnu, s3fifo)):

| cap | LCNU | S3FIFO | RotHead | Δ |
|---|---|---|---|---|
| 128 | 20.83 | 0.00 | 0.19 | -20.65 |
| 256 | 32.73 | 6.89 | 7.08 | -25.65 |
| 384 | 41.48 | 35.98 | 36.02 | -5.46 |
| 512 | 48.53 | 42.27 | 42.33 | -6.20 |
| 768 | 59.05 | 54.24 | 54.28 | -4.76 |
| 1024 | 66.85 | 65.47 | 65.47 | -1.38 |
| 1536 | 78.71 | 80.83 | 80.82 | -0.02 |
| 2048 | 86.48 | 89.56 | 89.55 | -0.01 |
| 3072 | 94.45 | 96.85 | 96.85 | -0.00 |
| 4096 | 97.80 | 98.89 | 98.89 | -0.00 |

Granite3.1-MoE-3B (Δ = rothead − LCNU; frozen baseline grid overlaps at 5 caps):

| cap | LCNU | S3FIFO | RotHead | Δ |
|---|---|---|---|---|
| 128 | 28.38 | 0.00 | 23.53 | -4.84 |
| 256 | 46.82 | 37.89 | 45.27 | -1.55 |
| 512 | 72.87 | 67.65 | 71.87 | -1.00 |
| 768 | 88.64 | 87.93 | 88.45 | -0.20 |
| 1024 | 97.08 | 97.32 | 97.10 | 0.01 |

- **H1 refuted**: worst 30b deficit -25.65pp at cap 256 — the
  probe locks onto S3FIFO at *every* capacity (5 switches, capacity-independent),
  which is near-optimal at ≥10E (deficit ≤0.02pp at 1536–4096) but catastrophic
  below it (S3FIFO is the *worst* baseline at low caps; at 128 it scores 0.00%).
- **H2 refuted**: granite deficits up to -4.84pp (cap 128).
- **H3 refuted**: granite oscillates — 48 switches (J(1) straddles the fixed
  threshold; exp660's point estimates are full-stream averages, windowed J(1)
  varies widely). 30b stays stable at 5.

**Mechanism conclusion (post-hoc, clearly labeled):** rotation rate is a
*stream* property and discriminates *models* (exp659/660), but the LCNU→S3FIFO
flip is *capacity*-dependent within a fixed stream — governed by hot-set-vs-
capacity fit, not rotation alone. A capacity-blind rotation probe therefore
cannot arbitrate the choice; any future adaptive design must feed capacity
(or hot-set fit) into the arbitration. Positive observation (post-hoc): in the
≥10E regime the probe picks correctly and RotHead trails the best baseline by
≤0.02pp while beating LCNU by up to 3.07pp (cap 2048) — the probe
works exactly where its threshold was measured.

Artifacts: `results_real_qwen3-30b-a3b-rothead.csv`,
`results_real_granite3.1-moe-3b-rothead.csv`, `policies_rothead.py`
(+ `rothead` registration in `policies.py` / `run_policies_real.py`);
prereg + verdict JSON + artifact hashes in `exp662-moei-rothead/`. EXPERIMENTS.md
rows carry `647ed87+dirty` provenance (run preceded this commit; policy sha
frozen in ARTIFACT.sha256 before the run).

## Addendum 12 — exp663 HotFit (capacity-aware hot-set arbitration): FAIL

**Prereg** `edc3d916` (frozen before sweep; smoke disclosed). Policy: `research/policies_hotfit.py` —
online probe computes N_hot = smallest hottest-key set covering >= 80% of a 2048-access window,
re-evaluated every 4096 accesses; arbitration `capacity >= 1.0 x N_hot -> S3FIFO else LCNU`.
COVERAGE=0.80, ALPHA=1.0, window/segment sizes are round a-priori values, not tuned.

### Execution integrity note (disclosed)
The first sweep was **invalid and fully excised**: `hotfit` had been registered in the policy
factory and CLI but omitted from `GEOMETRY_POLICIES`, so the sweep harness built the wrapper with
default geometry (num_layers=1, prior=0.05) instead of trace geometry (num_layers=48/32,
prior=k/E). The wrapper's deterministic re-simulation gate (sweep CSV vs independent resim with
explicit kwargs) flagged rate mismatches (e.g. 30b@cap128 CSV 0.183462 vs resim 0.208345);
root cause confirmed in-process (`geometry_kwargs('hotfit', stats) == {}`), the one-line set
membership fix applied, corrupted CSV/MD files and their 42 EXPERIMENTS.md rows removed, and both
sweeps re-run. All numbers below are from the corrected runs; determinism_ok=True at 5e-7.

### Verdict: **FAIL** (H1=False, H2=False, H3=False)

Per-capacity hit rates (decode, real traces; baselines from frozen `*_decode.csv`):

| cap | 30b LCNU | 30b S3FIFO | 30b HotFit | sw | granite LCNU | granite S3FIFO | granite HotFit | sw |
|---|---|---|---|---|---|---|---|---|
| 128 | 0.2083 | 0.0000 | **0.2083** | 0 | 0.2838 | 0.0000 | **0.2838** | 0 |
| 256 | 0.3273 | 0.0689 | **0.3273** | 0 | 0.4682 | 0.3789 | **0.4682** | 0 |
| 384 | 0.4148 | 0.3598 | **0.4148** | 0 | 0.7287 | 0.6765 | **0.6779** | 7 |
| 512 | 0.4853 | 0.4227 | **0.4829** | 26 | 0.8864 | 0.8793 | **0.8794** | 1 |
| 768 | 0.5905 | 0.5424 | **0.5591** | 131 | 0.9708 | 0.9732 | **0.9732** | 1 |

- **H1 (30b near-dominance, <=0.5pp of best baseline at every cap): FAIL.** Worst deficit vs best
  baseline: cap 768, -3.14pp below LCNU (HotFit entered S3FIFO at cap 1024 where LCNU still
  leads; the exp652 LCNU->S3FIFO crossover is at ~1280).
- **H2 (granite no-loss vs LCNU, <=0.5pp): FAIL.** Worst deficit cap 512, -5.08pp (early
  S3FIFO entry at 768 + boundary oscillation at 512).
- **H3 (stability <=6 switches): FAIL.** Max switches: 30b 131,
  granite 7 — N_hot hovers near capacity in the transition band and
  the arbitration flips per segment.
- Both extremes are exact: caps <=384 (30b) / <=256 (granite) reproduce LCNU bit-for-bit
  (0 switches); caps >=1536 (30b) / 1024 (granite) reproduce S3FIFO (1 switch).

### Mechanism conclusion
Capacity-aware hot-set arbitration is a strict improvement over exp662's capacity-blind probe
(worst-case error shrinks from -25.65pp to -3.14pp on 30b, and both regimes are now identified
exactly), but the specific criterion "capacity >= 80%-coverage hot set" **overshoots the
crossover**: the hot set at 80% coverage fits well before the measured 10E boundary, so the
policy enters S3FIFO one-to-two grid steps early and pays the LCNU advantage in that band. The
transition band additionally suffers probe-noise oscillation (131 switches at 30b cap 768).
A correct online criterion must sit between hot-set-fit (too early) and full working set
(exp659 ceilings) — i.e., the boundary is governed by the *effective* reuse structure (exp660
half-life), not by static coverage quantiles.

## Addendum 13 — exp664 FlipFit (crossover-statistic discrimination): FAIL

**Prereg** `b1ca30cc` (frozen 2026-11-03T07:18:54Z before
any computation). Pure measurement over frozen traces + frozen baseline CSVs — no new policy
simulation, no sweep, hence no EXPERIMENTS.md rows (disclosed deviation). Statistics computed
online-style: sliding window W=8192, re-evaluated every 4096, first 3 evaluations skipped,
median over the rest; arbitration model `capacity >= 1.0 x S -> S3FIFO else LCNU`.
Determinism gate: two independent runs produced byte-identical stats/verdict JSONs.

### Verdict: **FAIL** (H1=False, H2=False, H3=False)

Observed winner-flip bands (frozen CSVs, strict margin 5e-7) and per-statistic medians /
predicted first-S3FIFO caps:

| model | flip band (last LCNU win, first S3FIFO win] | S1 median | S2 median | S3 median | S4 median |
|---|---|---|---| ---|---| ---|---| ---|---|
| 30b | (1152, 1280] | 1543 -> cap 1664 (miss) | 1128 -> cap 1152 (miss) | 1430 -> cap 1536 (miss) | 1728 -> cap 1792 (miss) |
| granite | (768, 1024] | 951 -> cap 1024 (HIT) | 846 -> cap 1024 (HIT) | 701 -> cap 768 (miss) | 820 -> cap 1024 (HIT) |

- **H1 FAIL**: primary cov90 on 30b — median 1430 predicts cap
  1536, one grid step ABOVE the band end
  (1280). (Coverage 90% overshoots on 30b, the same
  direction exp663 found for 80%, but now the crossover is bracketed rather than beaten.)
- **H2 FAIL**: primary cov90 on granite — median 701 predicts cap
  768, one grid step BELOW the band start
  (768). Coverage 90% UNDERSHOOTS on granite — the
  opposite direction from 30b.
- **H3 FAIL**: secondaries hit granite (S1, S2, S4)
  but ALL miss 30b (medians 1543/1128/1728
  vs band (1152, 1280]) — S2 lands one step
  early, S1/S4 one step late.
- Brackets: 30b crossover 1280 lies between core>=3
  (1128) and cov90 (1430);
  granite crossover lies ABOVE cov95 (820) — the
  coverage-quantile family overshoots on one model and undershoots on the other, so NO
  shared quantile at theta=1.0 can pin both.

### Mechanism conclusion
Static windowed coverage/frequency quantiles — the natural capacity-aware arbitration
statistics — are refuted as a FAMILY: exp663 (80%, overshoot 30b) + exp664 (90% overshoots
30b, undershoots granite; repeat/core sets bracket but do not pin) show the crossover is not
a fixed coverage quantile of the access stream. The measured crossovers are 0.215x distinct
keys (30b) vs ~0.7x (granite) — the deciding structure is the temporal reuse geometry
(exp660 half-lives 102 vs 294 steps) interacting with policy internals (S3FIFO's
queue-ratio/admission dynamics), not the instantaneous hot-set shape. Direction for any
next algorithmic attempt: arbitration must be temporal (reuse-distance/decay-aware), or
abandon single-statistic arbitration for direct per-policy online scoring (e.g., shadow
mini-caches), which sidesteps the crossover-prediction problem entirely.

## Addendum 14 — exp665 ShadowFit (cumulative-rate shadow arbitration): FAIL

**Prereg** `66e7547e` frozen before the sweep (smoke at 3 caps disclosed in it). Policy
`research/policies_shadowfit.py`; five-point harness registration verified in-process
BEFORE freeze (`geometry_kwargs("shadowfit", stats) == {"num_layers": 48, "prior": 0.0625}`).
Arbitration: both LCNU and S3FIFO inners maintained in parallel at full capacity; cumulative
hit-rate scoreboard; challenger takes over only when its cumulative rate leads the incumbent
by >= 1.0pp (re-evaluated every 4096 accesses; initial LCNU). Parallel-maintenance (2x
metadata) with migration cost NOT modeled, per the exp662/663 disclosed convention.

### Verdict: **FAIL** (H1=False, H2=True, H3=True)
Frozen rules: SUPPORTED iff H1&H2&H3; PARTIAL iff H1&H2; else FAIL. H1 (30b: shadowfit >=
best(lcnu,s3fifo) − 0.005 at every canonical cap) FAILS at a single cap: 0.982309
vs best 0.988872 (+0.656pp at cap 4096).
H2 (granite, 5 overlap caps) passes with worst deficit +0.235pp
(cap 1024). H3 stability passes:
max 1 switch (30b) / 0 (granite) vs bound 6.

| model | cap | shadowfit | lcnu | s3fifo | best | deficit (pp) | switches |
|---|---|---|---|---|---|---|---|
| 30b | 128 | 0.208345 | 0.208345 | 0.000000 | lcnu | +0.000 | 0 |
| 30b | 256 | 0.327289 | 0.327289 | 0.068852 | lcnu | +0.000 | 0 |
| 30b | 384 | 0.414760 | 0.414760 | 0.359777 | lcnu | +0.000 | 0 |
| 30b | 512 | 0.485328 | 0.485328 | 0.422742 | lcnu | +0.000 | 0 |
| 30b | 768 | 0.590480 | 0.590480 | 0.542383 | lcnu | +0.000 | 0 |
| 30b | 1024 | 0.668520 | 0.668520 | 0.654659 | lcnu | +0.000 | 0 |
| 30b | 1536 | 0.805380 | 0.787092 | 0.808347 | s3fifo | +0.297 | 1 |
| 30b | 2048 | 0.893690 | 0.864790 | 0.895616 | s3fifo | +0.193 | 1 |
| 30b | 3072 | 0.965522 | 0.944494 | 0.968484 | s3fifo | +0.296 | 1 |
| 30b | 4096 | 0.982309 | 0.977971 | 0.988872 | s3fifo | +0.656 | 1 |
| granite | 128 | 0.283762 | 0.283762 | 0.000000 | lcnu | +0.000 | 0 |
| granite | 256 | 0.468164 | 0.468164 | 0.378923 | lcnu | +0.000 | 0 |
| granite | 512 | 0.728697 | 0.728697 | 0.676469 | lcnu | +0.000 | 0 |
| granite | 768 | 0.886415 | 0.886415 | 0.879340 | lcnu | +0.000 | 0 |
| granite | 1024 | 0.970839 | 0.970839 | 0.973187 | s3fifo | +0.235 | 0 |

### Determinism and the accounting incident (fully disclosed)
- The resim determinism gate (CSV vs independent in-process rerun, 5e-7) fired on the FIRST
  sweep: at every switched cap the harness CSV equalled the pure-S3FIFO standalone rate
  exactly. Root cause: the harness reads the policy's `hit_rate` property post-run, and the
  property initially reported the FINAL-choice inner's cumulative — erasing the pre-switch
  LCNU segment — while `on_access` return values (used by the analyzer rerun and the
  pre-freeze smoke) always attributed correctly. Fix: properties now report hits credited
  under the choice AT ACCESS TIME; post-fix smoke reproduced the pre-freeze smoke values
  bit-identically (128: 0.208345 / 1536: 0.805380 / 2048: 0.893690), proving the fix
  changed only harness-visible accounting, not behavior. Corrupted CSV/MD pairs and
  EXPERIMENTS.md rows were regenerated by clean re-sweeps; a botched manual row-dedupe was
  recovered by restoring EXPERIMENTS.md from HEAD and letting the harness re-append.
- Corrected analyzer: determinism_ok TRUE at all 15 caps; verdict computed from corrected
  CSVs. The initial SUPPORTED readout (0.0pp deficits at switched caps) was an artifact of
  the accounting bug and is retracted here; the corrected verdict is FAIL.

### Mechanism conclusion
Cumulative-rate shadow arbitration is near-dominant at 14/15 caps (worst +0.297pp)
and structurally stable (<=1 switch, no oscillation — the anti-correlation argument holds:
scoreboard gap large where wrongness is costly). It fails exactly where the sustained gap
between policies (S3FIFO−LCNU @4096 = +0.0109pp)
barely exceeds the fixed margin (1.0pp): the cumulative lead crosses the margin only late
in the stream, leaving a long LCNU prefix (+0.656pp deficit). A FIXED
margin cannot separate "gap >> margin" from "gap ~= margin" regimes. Next direction (not
run here): replace the fixed margin with a scaled one (e.g., switch when the lead exceeds a
fraction of the running gap, or use a discounted/decayed scoreboard that tracks late-stream
gaps) — the 4096 cap is the discriminating test case. Overall series position: exp661–665
now refute both predictive statistics AND fixed-margin direct scoring; the crossover regime
needs either gap-scaled arbitration or acceptance of per-cap tuning.

## Addendum 15 — exp666 DecayFit: decayed-rate shadow arbitration — VERDICT: FAIL (H2 oscillation; H1 passes with first cap-level WIN)

Follow-up to Addendum 14 (exp665): the cumulative scoreboard's history-drag was
replaced by an exponentially decayed scoreboard (half-life 32768 accesses,
margin 0.5pp, re-arbitration every 4096, initial LCNU; prereg sha 40f2f8dd
frozen before sweep, 3-cap smoke disclosed in prereg).

**Rules (frozen):** H1 near-dominance (>= best standalone − 0.5pp, both
traces); H2 stability (<= 6 switches per cap); H3 credit-accounting (credited
at access time; CSV must diverge from either inner's standalone cumulative at
switched caps).

**Verdict: FAIL.** H1 PASS (30b worst +0.155pp, granite worst +0.112pp);
H2 FAIL at 30b cap 1024 (8 switches); H3 PASS (determinism + accounting clean
at all 15 caps).

| Trace | Cap | DecayFit | Best standalone | Deficit | Switches | Note |
|---|---|---|---|---|---|---|
| qwen30b | 128 | 0.208345 | lcnu 0.208345 | +0.000 | 0 | — exact LCNU anchor
| qwen30b | 256 | 0.327289 | lcnu 0.327289 | +0.000 | 0 | — exact LCNU anchor
| qwen30b | 384 | 0.414760 | lcnu 0.414760 | +0.000 | 0 | — exact LCNU anchor
| qwen30b | 512 | 0.485328 | lcnu 0.485328 | +0.000 | 0 | — exact LCNU anchor
| qwen30b | 768 | 0.590480 | lcnu 0.590480 | +0.000 | 0 | — exact LCNU anchor
| qwen30b | 1024 | 0.666973 | lcnu 0.668520 | +0.155 | 8 | — OSCILLATION (H2 breach)
| qwen30b | 1536 | 0.809308 | s3fifo 0.808347 | -0.096 | 1 | — BEATS both standalones
| qwen30b | 2048 | 0.895268 | s3fifo 0.895616 | +0.035 | 3 |
| qwen30b | 3072 | 0.968435 | s3fifo 0.968484 | +0.005 | 1 |
| qwen30b | 4096 | 0.988625 | s3fifo 0.988872 | +0.025 | 1 |
| granite | 128 | 0.283762 | lcnu 0.283762 | +0.000 | 0 | — exact LCNU anchor
| granite | 256 | 0.468164 | lcnu 0.468164 | +0.000 | 0 | — exact LCNU anchor
| granite | 512 | 0.728697 | lcnu 0.728697 | +0.000 | 0 | — exact LCNU anchor
| granite | 768 | 0.886415 | lcnu 0.886415 | +0.000 | 0 | — exact LCNU anchor
| granite | 1024 | 0.972067 | s3fifo 0.973187 | +0.112 | 1 |

**Mechanism findings**

- The decayed scoreboard FIXES exp665's 4096 failure: deficit +0.025pp
  (was +0.656pp); the switch tracks the current gap instead of dragging
  all-history averages.
- FIRST CAP-LEVEL WIN of the series: at 1536 DecayFit (0.809308) beats both
  standalones (s3fifo 0.808347, lcnu lower) by 0.096pp — running the LCNU
  prefix before switching to S3FIFO is worth more than either fixed policy.
- The preregistered oscillation risk realized at exactly the flagged cap:
  1024 (LCNU overall leader) exhibits local rate inversions near the 0.5pp
  margin → 8 switches, +0.155pp. The margin that fixes 4096 (needs to be
  < 1.09pp sustained gap) is small enough to be crossed repeatedly at 1024.
- Fixed-margin arbitration is now refuted in BOTH directions: too large →
  late switch (exp665 @4096); too small → oscillation (exp666 @1024). The
  margin band that satisfies both caps simultaneously is narrow or empty;
  a margin schedule (or hysteresis with per-switch cost accounting) is the
  remaining untested variant.

**Artifacts:** results_real_{qwen3-30b-a3b,granite3.1-moe-3b}-decayfit.{csv,md},
policies_decayfit.py, analyze_exp666_decayfit.py, exp666-verdict.json
(programmatic from verdict JSON; worst deficits 0.155/0.112pp).

## Addendum 16 — exp667 CostFit: self-limiting switch-cost arbitration — VERDICT: SUPPORTED (first fully-passing arbitration design of the series)

Completes the fixed-margin refutation pair (exp665: margin too large → late
switch @4096; exp666: margin too small → oscillation @1024) with the remaining
untested variant: replace the fixed margin with a SELF-LIMITING barrier —
required_lead = MARGIN + GAMMA × (decayed recent-switch count). A first switch
from cold start faces only the base margin (regime shifts stay cheap); each
flip raises its own barrier (1 recent switch → 0.75pp, 2 → ~1.0pp, 4 →
~1.5pp), damping thrash. Prereg sha 462b9cdb frozen before sweep, 3-cap smoke
disclosed in prereg.

**Rules (frozen):** H1 near-dominance (>= best standalone − 0.5pp, both
traces); H2 stability (<= 6 switches per cap); H3 credit-accounting (credited
at access time; divergence from either inner's standalone cumulative at
switched caps; resim determinism).

**Verdict: SUPPORTED** — H1 PASS (30b worst +0.113pp, granite worst +0.112pp);
H2 PASS (30b max 4 switches — exp666 DecayFit had 8 at cap 1024; granite max 1);
H3 PASS (determinism + accounting clean at all 15 caps; analyzer rerun
byte-identical).

| Trace | Cap | CostFit | Best standalone | Deficit (pp) | Switches | Note |
|---|---|---|---|---|---|---|
| qwen30b | 128 | 0.208345 | lcnu 0.208345 | +0.000 | 0 | — exact LCNU anchor
| qwen30b | 256 | 0.327289 | lcnu 0.327289 | +0.000 | 0 | — exact LCNU anchor
| qwen30b | 384 | 0.414760 | lcnu 0.414760 | +0.000 | 0 | — exact LCNU anchor
| qwen30b | 512 | 0.485328 | lcnu 0.485328 | +0.000 | 0 | — exact LCNU anchor
| qwen30b | 768 | 0.590480 | lcnu 0.590480 | +0.000 | 0 | — exact LCNU anchor
| qwen30b | 1024 | 0.667390 | lcnu 0.668520 | +0.113 | 4 |
| qwen30b | 1536 | 0.809308 | s3fifo 0.808347 | -0.096 | 1 | — BEATS both standalones
| qwen30b | 2048 | 0.894752 | s3fifo 0.895616 | +0.086 | 3 |
| qwen30b | 3072 | 0.968435 | s3fifo 0.968484 | +0.005 | 1 |
| qwen30b | 4096 | 0.988625 | s3fifo 0.988872 | +0.025 | 1 |
| granite | 128 | 0.283762 | lcnu 0.283762 | +0.000 | 0 | — exact LCNU anchor
| granite | 256 | 0.468164 | lcnu 0.468164 | +0.000 | 0 | — exact LCNU anchor
| granite | 512 | 0.728697 | lcnu 0.728697 | +0.000 | 0 | — exact LCNU anchor
| granite | 768 | 0.886415 | lcnu 0.886415 | +0.000 | 0 | — exact LCNU anchor
| granite | 1024 | 0.972067 | s3fifo 0.973187 | +0.112 | 1 |

**Mechanism findings**

- The self-limiting barrier damps the exp666 oscillation exactly as designed:
  cap 1024 goes 8 → 4 switches while IMPROVING its rate
  (0.667390 vs DecayFit 0.666973; deficit +0.113pp).
- Cold-start switching is preserved: 4096 still takes exactly 1 switch at
  +0.025pp (barrier at first switch = base margin only).
- The cap-level WIN at 1536 is preserved (−0.096pp vs both standalones) —
  running the LCNU prefix before switching to S3FIFO beats either fixed policy.
- 9 of 15 caps reproduce the best standalone EXACTLY (zero-switch caps are
  bit-identical to the anchor policy): CostFit never pays for its arbitration
  when arbitration is not needed.
- Series conclusion: prediction-free shadow arbitration becomes PRACTICAL with
  decayed scoreboards + self-limiting switch costs: near-dominant everywhere
  (worst +0.113pp), ≤4 switches, 2× metadata overhead unchanged.

**Artifacts:** results_real_{qwen3-30b-a3b,granite3.1-moe-3b}-costfit.{csv,md},
policies_costfit.py, analyze_exp667_costfit.py, exp667-verdict.json
(programmatic from verdict JSON).

## Addendum 17 — exp668: CostFit robustness is a region, not a point (VERDICT: ROBUST, 9/9)

exp667 (Addendum 16) passed at one parameter point. exp668 sweeps the two
control parameters 4x each way around it — MARGIN {0.0025, 0.005, 0.010} x
GAMMA {0.00125, 0.0025, 0.005}, HALF_LIFE fixed 32768 — over both traces at
all canonical caps, with the frozen pass rule: >=7/9 cells passing
H1 (near-dominance 0.5pp, both traces), H2 (<=6 switches), H3 (loop==property
+ zero-switch caps equal baseline CSVs exactly).

**Result: ROBUST — 9/9 cells pass.**
Two full grid runs byte-identical (determinism asserted in runner).
Center cell (MARGIN 0.005, GAMMA 0.0025) reproduces the exp667 published
CSV rates exactly at all 15 caps (cross-artifact anchor).

Cell summary (worst deficit pp / max switches):

| MARGIN\GAMMA | 0.00125 | 0.0025 | 0.005 |
|---|---|---|---|
| 0.0025 | PASS: 0.115pp/sw6, 0.065pp/sw2 | PASS: 0.072pp/sw4, 0.070pp/sw1 | PASS: 0.086pp/sw3, 0.070pp/sw1 |
| 0.005 | PASS: 0.096pp/sw4, 0.112pp/sw1 | PASS: 0.113pp/sw4, 0.112pp/sw1 | PASS: 0.116pp/sw3, 0.112pp/sw1 |
| 0.01 | PASS: 0.332pp/sw2, 0.220pp/sw1 | PASS: 0.332pp/sw2, 0.220pp/sw1 | PASS: 0.332pp/sw2, 0.220pp/sw1 |

Failure-mode boundary map: worst deficit 0.332pp at MARGIN 0.01
(qwen30b cap 4096) — the exp665 late-switch direction; max switches 6
at MARGIN 0.0025/GAMMA 0.00125 (qwen30b cap 1024)
— the exp666 oscillation direction, exactly at the H2 boundary. The pass
region is bounded by both refuted failure modes on opposite edges and the
entire tested 4x4 region stays inside: CostFit needs no per-deployment
parameter tuning across this range.

Artifacts: exp668-grid.json (per-cell per-cap rates, switches, H-flags);
prereg frozen 81079e60 before the grid ran; instance-attribute cell override
(sealed exp667 policy source unedited, sha 4bfff5bd unchanged).

## Addendum 18 — exp670: CostFit generalizes out-of-family (VERDICT: GENERALIZES)

All prior CostFit evidence (Addenda 14–17) rests on two real traces from one
capture family. exp670 tests the sealed center config (MARGIN 0.005, GAMMA
0.0025, HALF_LIFE 32768) on 11 synthetic streams it has never seen: Zipf
skew s {0.8, 1.1, 1.4} × temporal locality λ {0.0, 0.3, 0.6} (E=128, L=48,
638,208 accesses; recency-weighted replay H=2048, R=4096) plus two E=32
topology variants. Caps {128…4096}; lcnu/s3fifo/costfit evaluated in the
same harness path; informative-cell rule (inner gap ≥ 0.1pp) and verdict
bars frozen in prereg 606e9507 before any stream was generated.

**Result: GENERALIZES** — H1 True, H2 True, H3 True;
59 informative cells across 9/9
primary streams (all informative); zero violations; full double-run
determinism (stream shas and all rates identical across both runs).

Worst deficit at any informative cell: **0.176pp** (bar 0.5pp).
Max switches anywhere: **2** (bar 6). No near-boundary cells
(no cell with deficit > 0.3pp or switches ≥ 5). The design is more stable
on the synthetic family than on the real traces (real center: 4 switches
@30b/1024; here max 2).

Cells where arbitration BEATS both standalones (4): s0.8/λ0.0/E128@1024 (+0.002pp); s1.1/λ0.0/E128@1024 (+0.015pp); s1.1/λ0.3/E128@512 (+0.002pp); s1.4/λ0.3/E128@1024 (+0.005pp).

Uninformative cells (7): s1.4/λ0.3/E128@4096; s1.1/λ0.3/E32@1024; s1.1/λ0.3/E32@2048; s1.1/λ0.3/E32@4096; s1.4/λ0.6/E32@1024; s1.4/λ0.6/E32@2048; s1.4/λ0.6/E32@4096 — all in the
cache-covers-working-set regime (inner gap < 0.1pp; arbitration inert by
construction), concentrated in E=32 topology at high caps.

Interpretation: the self-limiting barrier mechanism is not an artifact of
the capture family — it tracks the better inner within 0.18pp and never
oscillates across a 3×3 structure grid and a 4× topology shift. Combined
with exp668's parameter-region robustness, the design now has both
parameter and structure generalization evidence.

Artifacts: exp670-synth.json (per-stream per-cap rates/switches, stream
shas, violation map); generator + evaluation frozen in run_synth.py
(sha in ARTIFACT.sha256); prereg 606e9507 with pre-freeze smoke disclosed.

## Addendum 19: exp673 — CostFit3: the barrier generalizes to 3 candidates (toxic-rejection regime)

**Question.** exp667/668/670 established the self-limiting barrier for TWO
candidates (LCNU vs S3FIFO): switch only when the decayed lead exceeds
MARGIN + GAMMA x decayed switch mass. Does the mechanism survive a THIRD
candidate — rothead, itself a probe-adaptive LCNU<->S3FIFO switcher on an
8x faster probe timescale (PROBE_WINDOW 2048 vs the barrier's
HALF_LIFE 32768)?

**Design.** Three pre-freeze iterations tried to construct a stream family
where rothead DESERVES adoption; all three failed (disclosed in the prereg):
lambda-flips left the inners tied (<=0.8pp); on churn-flips LCNU dominated
and rothead was toxic by up to -8.3pp. What survived is the regime the
barrier exists for: a third candidate that is persistently harmful. exp673
therefore freezes the honest question — does the N-way generalization
(challenger = highest decayed-rate non-incumbent, name-order tie-break;
barrier and switch-mass accounting unchanged) correctly REJECT the toxic
candidate while staying within the verified no-harm envelope?

**Verdict: MECHANISM-GENERALIZES-3WAY.** Informative cells: 27;
violations: 0 (H1 no-harm True, H2 toxic-rejection
True, H3 accounting True); synthetic double-run determinism:
true (bit-identical JSON).

Arms: both real traces (30b: 10 caps; granite: 5 caps; sealed harness,
costfit3 fresh, all references parsed from sealed CSVs) plus two frozen
synthetic flip families (N=262,144; 16 blocks of 16,384; caps 128-4096;
lcnu/s3fifo/rothead/costfit/costfit3 all evaluated fresh).

**Risk map (worst 5 informative cells by cf3 deficit vs best standalone):**

| cell | cf3 deficit (pp) | rothead deficit (pp) | cf3 switches |
| --- | --- | --- | --- |
| flipB @ 128 | +0.245 | +2.202 | 2 (lcnu:1+rothead:1+s3fifo:0) |
| flipA @ 2048 | +0.135 | +1.261 | 2 (lcnu:1+rothead:1+s3fifo:0) |
| flipA @ 128 | +0.083 | +0.082 | 1 (lcnu:0+rothead:0+s3fifo:1) |
| flipB @ 2048 | +0.072 | +3.158 | 2 (lcnu:1+rothead:1+s3fifo:0) |
| flipA @ 256 | +0.069 | +0.069 | 1 (lcnu:0+rothead:0+s3fifo:1) |

**Toxic-rejection: 8 cells have rothead trailing best2 by >1pp;
in every one, costfit3 finished within 0.5pp of the sealed 2-way costfit
(H2) — wrong excursions onto rothead occurred (visible as rothead switch-ins
that revert) and self-corrected. The barrier never adopted a candidate the
evidence did not support, on any family, real or synthetic.**

**Mechanism note.** The N-way generalization changed exactly one thing —
which challengers the scoreboard may consider — and the barrier's function
(rate evidence must exceed the accumulated switch-mass penalty before
control transfers) transferred unchanged. This is the third independent
stress family (parameters exp668, stream structure exp670, candidate set
exp673) in which the self-limiting property held with zero violations.

Cost model, parameters (exp667 center), registration points, prereg sha,
and the three disclosed failed design iterations: `exp673-moei-costfit3/PREREG.md`
(frozen pre-run, `f7378dd3`). Data: `research/exp673-costfit3.json`.


## Addendum 20 (exp669)

**Verdict: IMPROVED** (frozen rule: IMPROVED if H1) — re-eval determinism gate PASS (best config rebuilt post-run on both traces; all recorded cells byte-identical).

## Addendum 21 (exp674)

**Verdict: SAFETY-BROKEN** — the BO point (HL=160975, 4.9x center) degrades the mean hit rate vs center (-0.1234pp) and violates the no-harm barrier on the synthetic grid (see ADDENDUM21.md same-day correction). The center calibration stands; the BO objective overfit the worst case to two real traces.

## Addendum 22 (exp676 drift): DRIFT-SAFE — center bounds harm through mid-stream regime change (worst trans 0.781pp < 1.0 bar, post 0.264pp, max 1 switch/window, 16/16 cells, det + accounting clean). See ADDENDUM22.md.
## Addendum 23 (exp677 W-sweep): W-SWEEP-SAFE — drift bounded at fast transitions

W {1024..8192} locality-drop sweep: 12/12 cells inside bars; only nonzero
transition deficit is W=4096/cap=4096 = 0.439pp. Monotonicity REFUTED —
exposure peaks at W=RESEGMENT alignment (mid-transition arbitration on stale
mass), faster blockwise shifts fully absorbed. Blockwise drift now bounded
both sides (slow 0.781pp exp676 / fast 0.439pp exp677). Residual axis:
sub-block intra-window shifts.
## Addendum 24 (exp678 cost model): COST-H1 — hit-rate proxy breaks under bursty reload

Time metric with 1+kappa*u_e+phi*burst miss cost, 24 cells: 1 H1 violation
(skew 1.3, cap 2048, kappa=0, phi=4 — 1.057% time deficit at 0.079pp hit
deficit). Mechanism: costfit's 502 extra misses CLUSTER; burst penalty
amplifies clustered misses. Clustering, not expert heterogeneity, is the
failure mode. Hit-rate proxy is blind to miss temporal structure — all
prior rankings carry this caveat. Follow-up: anti-clustering candidate +
(phi,cap) boundary map.
## Addendum 25 (exp680 stagger): STAG-SAFE-NO-GAIN — cadence is not the lever

RESEGMENT 2048 vs 4096 on exp677-exact streams: identical deficits in 6/6
cells incl. the 0.439pp peak (where the barrier never fires — cadence
irrelevant by construction). Center reproduced exp677 exactly (replication
gate). Alignment exposure flows through inner-policy transition response,
not switch timing. det=False incident disclosed: module-level seed spawn
advanced between passes; fixed, rerun det=True, numbers reproduced; no
publication on det=False. Pivot: sub-block flash-crowd probe.
## Addendum 27 (exp681 crowd): CROWD-H1 — third failure mode, admission lag

Cold-crowd bursts (d 256-4096, ranks [64,128) uniform) at midpoint of
stationary streams: H1 violated 8/8 cells, worst 10.86pp (d4096/cap2048);
costfit burst hit collapses to 0.016-0.188 while best standalone adapts.
Cap-dependent split: cap 2048 harm in-burst; cap 512 harm moves to recovery
(worst 6.98pp — crowd admissions evicted the hot set). Aligned control
still loses (sw=1, no help): failure is admission speed of repeating cold
keys, not arbitration blindness per se. Third failure mode joins alignment
(exp677) and miss clustering (exp678); all point at the admission lever.
Det byte-identical; serialization-crash amendment disclosed in prereg.
## Addendum 28 (exp682 door): DOOR-HARM on stationary — correct null-space result

Doorkeeper admission filter on exp678's exact testbed (replication 24/24
exact, det True): harm everywhere on stationary streams — H2 16/16 hit
losses >0.5pp, H3 21 combos slower, H1 only 4/16; lcnu+door also loses
(policy-independent tax). Protection value is zero without crowd structure;
the double-miss tax on one-touch reuse is constant. Sharpens exp683: door
on exp681's flash-crowd streams, where one-touch crowd admissions are the
proven poison (10.86pp). If it still harms there, admission filtering is
dead for this family and the fix must live in inner admission.
## Addendum 29 (exp683 doorcrowd): DOOR-CROWD-HARM — admission filtering closed at wrapper level

Door on exp681's exact crowd streams (anchor exact, det True): burst
deficit GROWS everywhere (worst 14.8pp vs center 10.9pp at d4096/cap2048),
H2 safety 11/16 violated — but recovery flips to ADVANTAGE where crowd
admissions were the poison (d4096/cap512: -1.0pp vs center's +6.98pp).
Mechanistic synthesis across 681/682/683: the lever is admission SPEED
with post-admission demotion. exp684: probation subclass (admit first
sight, evict never-rehit first).

## exp684 — ring-less probation (PROB-HARM, ledger #208)
ProbationLCNU (admit first sight, evict oldest never-rehit first, no partition) on exp681 exact crowd streams: burst deficit -3.6..-8.7pp in ALL 8 cells, loop -2.2pp at cap2048, center_prob rec -2.2..-5.4pp cap2048; h3 4/8; det+repl exact. Unbounded probation = the door failure moved inside the cache; capacity partition required -> exp685. ADDENDUM30 (0b54a645), freeze fe1d97f.

## exp685 — bounded nursery ring (IN_BURST_FIXED_ONLY, ledger #209)
RingProbationLCNU (FIFO nursery drains ONLY when full; citizens by LCNU scoring; rehit promotes) on exp681 exact streams, dual-anchored replication (center==exp681, door4096==exp683, 24 cells byte-exact): in-burst deficit <= center ALL cells (0.0 at cap2048 — the door 12.7-14.8pp catastrophe erased); ring25 rec >= lcnu-0.5 ALL; single failed gate = composite rec_def +6.982pp == center at d4096/cap512, byte-equal to center => ring inner INVISIBLE to arbitration at cap512 (not overflow). Standalone ring10 = new Pareto point (b0.0288/r0.6853 vs door b0.0205/r0.6934 vs lcnu b0.0884/r0.6780); at cap2048 ring = pure churn-waste. exp686a: lazy promotion + arbitration visibility + stride-adaptive sizing. ADDENDUM31 (3bc4e1a1), freeze 63d4e09.

## exp689 — ring visibility arms (VIS_ONLY, ledger #212)
center_ring10: rec beats lcnu ALL 4 cap512 cells (-0.71/-1.10/-1.37/-0.73pp) with sealed barrier, pays burst +5-6pp -> ladder fails; center_zb REFUTES barrier hypothesis (zero-barrier still collapses rec at home == center); center_lazy dead dose composite-level; center_ghost dominated. Convergent frontier with exp686a/exp687: burst wants LCNU inner, recovery wants RING, s3fifo never best in either critical window. exp690: 2-way lcnu+ring composite + double-window confirmation. ADDENDUM33, freeze c05c4d5, det+repl exact.

## exp690 — 2-way + confirmation arms (REC_ONLY, ledger #213)
ALL 3 T-arms (twoway_lr/twoway_cf/fit3_cf) converge to the standalone ring10 endpoint at home (burst 0.0288, rec 0.6853; sw=1-2, zero thrash): removing s3fifo does NOT keep lcnu through burst — ring decayed rate dominates post-loop so burst always inherits the ring (burst_def 5.96pp home, worst 9.23pp). Double-window confirmation is a no-op (single early switch). Recov (-0.73pp) + safety PASS. Triple repl exact (exp681/685/686a); det. Conclusion: rate-scored menus cannot time the handoff; burst edge belongs to the LCNU inner itself. exp691 registered: miss-signal adoption freeze. ADDENDUM34; freeze fd8b92d -> repair c50a310.

## exp691 — storm-revert + adoption-freeze arms (REC_ONLY, ledger #214)
Storm detector fires once per stream, only AT/AFTER the burst (d256: invisible — burst < window); revert saves neither burst (==ring 0.0288) nor recovery (barrier ~0.75pp > ring advantage +0.73pp -> rec collapses to lcnu 0.6780); freeze arms == twoway control byte-identical. All T-arms recov+safety PASS, burst_def FAIL (5.96 home/9.23 worst). FAMILY CLOSED after 5 experiments: burst < arbitration window means no rate signal can act in time; burst-vs-rec is a capacity frontier. exp692 registered: SecondSightRing revisit-only annex, no timing needed. ADDENDUM35; freeze a1df70b; det; triple repl exact.

## exp693 — SecondSightRing revisit-only annex (BOTH_PASS, ledger #215)
First BOTH_PASS in the composite program: all 3 annex fracs (5/10/15%) pass burst+recov+stretch+safety. Burst byte-preserved (burst_def 0.000 exactly at all cap512 cells; ssr burst 0.0884 == lcnu); cap2048 burst_def NEGATIVE (up to -0.78pp — fencing revisits purges victim ranking, ssr beats plain lcnu); rec_def ~0 (null capture: burst-vs-rec at cap512 is a capacity frontier, confirmed end-to-end across timing/judging/hysteresis/sparing). Repair disclosed: batch annex exclusion (equivalence-proven, 25x). det; repl exp681+exp685 exact. ADDENDUM36; freezes 9c4d4dc + 9fbf133; git rebuilt as standalone clone after parent-repo deletion. exp694 registered: capacity-scaling axis.

## exp694 — capacity-scaling map of burst/recovery frontier (DUAL_STABLE, ledger #216)
Caps 512-8192 x d{1024,4096} x lcnu/center/ring10/ssr_10. Ring dual advantage PEAKS at cap1024 (+8.86pp burst, +2.34pp rec — rec edge triples vs cap512) and cap2048 (+9.23/+1.44); dissolves cap4096 (+0.88/+0.20); total convergence at cap8192 (all arms 0.9985-1.0). lcnu absolute burst penalty GROWS with capacity (0.0884->0.4099). CostFit == lcnu byte-equal at cap>=1024 (arbitration no-op). ssr never beats lcnu at burst (any cap). Frontier is a MID-CAP phenomenon: worst 1.5-4x working set, moot at saturation. det True; repl exp681+exp685 exact; ADDENDUM37; freeze f3b4795. exp695 registered: lcnu victim-ranking pathology under flood.

## exp694 erratum — sign convention: burst_adv positive = ring SACRIFICES burst (deployment-hint paragraph in the exp694 entry above misread it as a gain; corrected in ADDENDUM37 erratum). Table/verdict unchanged (bars were pre-registered with the correct sign).

## exp695 — flood decomposition (REFUTED — mechanism rewritten, ledger #217)
Both hypotheses dead: burst window is 99.9% CARRY-INS (a=6, b=4, c=4086 of 4096) — bursts RE-ACCESS the calm-seeded crowd, not novelty. lcnu burst hit == its carry-in hits (359->4086 by cap: more crowd keys resident when burst lands); ring probation holds HOME keys so carry-ins miss (118->4086, converges at saturation). MECHANISM REWRITTEN: burst-vs-recovery = slot competition between two re-access populations; decisive admissions happen in the CALM SEEDING PHASE, not the burst. det True; repl exp694 exact; consistency PASS; ADDENDUM38; freeze ad3b362. exp696 registered: admission-time population throttling.

## exp696 — sring composition (TRADE_UNCHANGED — conflict pinned, ledger #218)
SringLCNU (revisit annex + churn ring on LCNU) == ssr_10 to 4dp at every cap: the revisit-miss routing rule (annex vs ring) fully determines behavior. First-sight-only quarantine delivers ZERO rec gain (0.7666==lcnu vs ring 0.7898) — ring ENTIRE +2.34pp rec win comes from quarantining re-entry revisits, the same population ssr protects for burst. Mechanisms are opposite answers to one binary question on one population; composition impossible; the frontier IS the routing rule. Middle point (earn-then-immune) registered as exp697. det True; REPL exact vs exp694 both depths; ADDENDUM39; freeze 0c2e7fb.

## exp697 — earn-then-immune (HARM by gate artifact; interior point found, ledger #219)
Churn-collapse prediction REFUTED: ETI = strict interior point. d4096 cap1024 +1.35pp burst/-0.68pp rec vs ring; cap2048 +2.59/-0.51; cap4096 +0.37/+0.02. Exchange ~0.3pp rec per pp burst (5-10x flatter than lcnu->ring). Mechanism exact: recovery carried by class-c carry-in hits (+55/+106 = the deltas to the 4th decimal). HARM gate anchored to lcnu burst (registration error, self-caught, disclosed). det True; REPL exact; ADDENDUM41 (40 taken by exp682 lane); freeze c8bb520. exp698 registered: ETI fraction sweep.

## exp698 — ETI fraction sweep (FLAT_FRONTIER; beyond-pole points; ledger #220)
9-variant sweep traces a smooth interior family (annex drives burst, ring gates rec). No DOMINANT_POINT (honest negative). Flat exchange confirmed: eti_0510 0.45/0.17pp/pp, eti_1010 0.50/0.20 at cap1024/2048. BEYOND-POLE: cap4096 eti_2020 0.4285/0.9517 beats ssr pole +1.57pp burst at +0.03pp rec. Runner verdict STEEPENS = unit bug in flat check (dr/db>=-0.01 vs registered >=-1.0), self-caught, sealed JSON not edited, corrected per binding spec. det True; anchors==exp694; eti_1010==exp697 (cross-exp determinism); ADDENDUM42; freeze 71ef7db. exp699: beyond-pole existence test.

## exp699 — beyond-pole existence test (NEITHER; regime-specific; ledger #221)
Beyond-pole REAL but NARROW: Grid A cap4096/d4096 6 variants beat ssr (opt eti_4020 0.4292/0.9517, annex plateaus 20%+, ring axis inert past 20%); FAILS Grid B (cap2048 any depth; d256/d1024 all converge). FLAT_GENERALIZES FALSE: eti_0510==ring10 exactly at cap4096 all depths; flat exchange only d4096 caps 1024-2048. MODEL: burst depth x capacity jointly set arbitration dominance; carry-in mechanism needs deep bursts + capacity headroom. Run 1 exit 3 = impossible cross-depth totals gate (disclosed, aborted pre-adjudication, no selection); gate v2 freeze c0c7dfe. det True; anchors==exp694; eti_2020==exp698 exact; ADDENDUM43. exp700: phase-boundary sweep.

## exp700 — phase-boundary sweep (STRUCTURE_UNEXPLAINED; ridge found; ledger #223)
Spread matrix d{256..4096} x cap{1024,2048,4096}: RIDGE at d~2xcap (max 11.91pp @d4096/cap2048 = the ssr-vs-ring trade band); cap1024 col saturates ~8pp; cap4096 col low with rising tail 2.73pp = beyond-pole. Registered plane-fit bar honest-failed (R2=0.5804<0.8 — wrong model class, not absence of structure). MONOTONE_BOUNDARY passed. det True; anchors==exp694+exp699 exact; ADDENDUM44; freeze d9f52af. exp701: pairwise decomposition + off-grid ridge probes.

## exp701 — ridge decomposition (RIDGE_CREST by 0.012pp; crest off-grid ratio 1.6)
PAIR_DOMINANT PASS: ring-vs-ssr carries 13/15 cells (87%). RIDGE_CREST pass at tolerance edge (12.402 <= 12.414); DISCLOSED: surface max is OFF-GRID at (d4096,cap2560)=12.40pp -> ridge tracks d/cap ~1.6, NOT the 2.0 read from exp700's grid; cap-profile collapses 12.40/10.01/0.17 at 2560/3072/5120; d3072 row rises through cap2560 (peak drifts right with d); crest cell itself = ring-vs-eti (0.2148 vs 0.0957), not the ssr trade. Run1 d5120 generator crash (BURSTS ceiling 4096) disclosed + rescope v2 freeze d77a4ab; run2 gates all green (REPL==exp700 exact; det True; 1433s); post-JSON cosmetic print KeyError, no data impact, sealed JSON untouched.

## exp702 — real-trace carry-in (mechanism REAL, policy lesson INVERTS)
H1 REAL_CARRYIN PASS: c-share 99.24% on burst windows (synthetic 99.9% — re-access dominance is real). H3 PASS at 20.65pp/14.26pp (cap256/512) but INVERTED: LCNU best on real traffic (0.4682 whole @256), ring10 worst (0.2385), eti −20.6pp, ssr near-neutral — synthetic frontier (ring buys burst) does NOT transfer. Root cause: real novelty is FRONT-LOADED (burst windows = ramp 1–23; windows 24–154 have 17 first-sights total; all later session switches zero-novelty warm reshuffles; no mid-stream cold crowds exist). H2 UNEVALUABLE (median novelty 0.0 -> empty strict-below base set; disclosed). ANCHOR: lcnu@256 = 0.468164 byte-equal to exp650 repo simulator (band was ±0.5pp). det True; structure exact; 60s; freeze 062b413. Consequence: burst-arbitration policy line refuted for this traffic class; next = second model family (qwen-30b) replication + ramp-regime study.

## exp703 — qwen-30b replication (carry-in = 2-family law; inversion sharper)
PRIMARY REPLICATED: c-share 94.8% (hi) / 93.1% (lo) on burst windows (granite 99.2%). S2 effectively confirmed: LCNU top or tied everywhere except a 0.0003 ssr hair on hi@1024 burst; no arm beats lcnu whole-stream in any cell. Inversion DEEPENS: ring10 whole @1024 −31.3pp (hi) / −34.1pp (lo) vs granite's −20.7pp; eti −7.5/−10.1pp; ssr neutral. S1 SPLIT (lo w90=64 pass, hi w90=98 fail by 2): front-loading NOT universal — qwen SESSION SWITCHES carry novelty bursts ({48,144} hi, {48} lo burst∩starts), granite's warm-switch claim revised to model-specific. S3: spread 18.7-19.6pp @1024 (1/6 key space) collapses to 0.03-2.1pp @2048 (1/3) — capacity is the divergence knob. Anchors BYTE-EQUAL again (0.704076/0.624469 == repo). det True; 518s; freeze c599dab. Next: exp704 real-trace capacity fine-map (spread-collapse curves, 3 traces) -> cost-integration cache-sizing curves.

## exp704 — real-trace capacity fine-map (policy premium curves)
CAPACITY_MAP: B1 decay PASS 3/3 (granite 28.5->38.2 hump@192->0.05@896; hi 24.9->0.03; lo 21.0->2.1). B2 CROSS1: granite 768 (60% key space), hi 1792 (29%), lo NOT_REACHED by 2048 (2.1pp @33%) — diverse-footprint traffic stays policy-sensitive longest. B3 strict FAIL on saturation ties only (granite@896 +0.0045pp float hair w/ lcnu better whole; hi@2048 exact tie) — in divergence band ring never wins 27/27 cells. B4 anchors 13/13, deltas <=4.3e-07 = CSV print precision (2 families, 13 capacity points). Low-cap hump: spread peaks at 192 (38.2pp) just above starvation. Engineering artifact: policy premium curve per trace — below CROSS1 policy worth up to 38.2pp, above it capacity alone decides; CROSS1 tracks steady-state working set. det 3/3; 2376s; freeze 29e8d18. Next: consolidation draft exp681-704 (needs user go-ahead for external form).

## exp705 — cost-integration byte-denominated cache-sizing curves (analysis tier; C2 honest negative; ledger #228)
Freeze 1a8514a -> D1 loader-shape crash repaired pre-measurement (ARTIFACT-r2) -> FROZEN-track honest ANCHOR-FAIL exit 5 preserved verbatim (D2: prereg misread sealed spread as whole-hit; exp704 spread_pp is BURST max-min, run_capacitymap.py:138-141; recompute from sealed policy_grid reproduces it EXACTLY 0.0 at all 27 cells) -> ARTIFACT-r3 -> repaired track COST_CURVES exit 0, bars unchanged. C1 PASS: CROSS1 granite 768 slots = 3.375 GiB (<=4.0), qwen-hi 1792 = 15.75 GiB (>=15.0), lo NOT_REACHED -> lower bound 18.0 GiB. C2 FAIL (negative kept): WS20 1228/3396/4662 (det + independent 2nd impl exact); CROSS1/WS20 = 0.625/0.528 -> best-vs-worst convergence needs only ~53-63% of the final-20% working set resident; NOT full-WS tracking, NOT fixed keyspace fraction (60.0% vs 29.2%). C3 PASS (cliff >=10x both reached traces). C4 PASS: byte consts exact (b_e 4,718,592 / 9,437,184 B), N 638208/786432/638208, APT 256/384. Premium pricing: granite 344.5 MB/token best-vs-worst miss-traffic delta @0.5625 GiB cache -> 4.55 MB/token at CROSS1 (3.375 GiB). Follow-up controller re-adjudicated every gate from primary sources with an independently written loader: exact match, verdict COST_CURVES stands. Analysis tier — NOT 1001-countable. Next: exp706 lo cap-grid extension (overlap-anchored 1792-3328) to locate qwen-lo CROSS1; 53-63%-of-WS20 law predicts 2470-2936.

## exp706 — lo CROSS1 located (law 3-for-3; ledger #229)
LOCROSS_MAP: qwen-lo CROSS1 = 2816 slots = 24.75 GiB (grid 1792-3328, overlap-anchored vs sealed exp704). L3 exact: spread@1792/2048 4.211/2.116 AND lcnu whole byte-equal to sealed exp704 — cross-process determinism holds a 3rd time. L2 PASS: CROSS1/WS20 = 0.604 in [0.50,0.66] -> 3-trace law 0.625 (granite) / 0.528 (hi) / 0.604 (lo): best-vs-worst convergence needs only ~53-63% of the final-20% working set resident; deployable sizing rule cap ~ 0.6 x WS20 (5.27 GiB per 1000 WS20 keys 30B-class, 2.64 GiB 3B-class). Completes exp705's C1 hardware mapping 3/3. Spread decays 4.211 -> 0.055pp; policy premium below CROSS1 up to 4.2pp on this extension; 3072-3328 arms indistinguishable. Continuity 8/9 strict + 1 saturation tie @3328 (ring never wins). L4 det True; exit 0; 1036s; freeze commit a75c7de (sha frozen pre-launch); ADDENDUM50. Follow-up controller re-adjudicated every gate programmatically from primary sources (exp704 + exp705 sealed JSONs parsed at runtime): exact match, verdict stands. Next: exp707 cost-layer completion (lo GiB sizing table + consolidated 3-trace rule; analysis tier over sealed artifacts).

## exp707 — cost-layer completion: 3-trace deployable sizing rule (analysis tier; ledger #230)
SIZRULE_COMPLETE: S1 anchors exact (granite/hi byte-equal exp705; lo@1792/2048 recomputed-from-706 == exp705 == exp704 — cross-artifact determinism 3rd witness), S2 law band {0.528, 0.625, 0.604} all in [0.50,0.66] width 0.098 <= 0.10, S3 lo CROSS1 2816 slots = 24.75 GiB exact (C1 3/3), S4 savings identity 1e-9, S5 det. Deployable rule: cap ~ 0.6 x WS20 (30B-class 5.27 GiB / 3B-class 2.64 GiB per 1000 WS20 keys); lo GiB table 768-3328 complete (rows 2176-3328 = frozen formula over sealed exp706 grid). Analysis tier — NOT 1001-countable; cost layer COMPLETE. Follow-up controller re-adjudicated 39/39 checks programmatically from primary sources (runner sha 7f1fbb71 intact, never re-executed); exit 0; 4s; freeze commit 36f665a; ADDENDUM51. Next: beyond-law flat check lo 3584-4662 (0.77-1.0x WS20) — does convergence persist to the full working set?

## exp708 — beyond-law flat check (BEYOND_FLAT; ledger #231)
Maps lo caps 3328->4662 (0.71->1.00 x WS20, ending exactly at the working set): s_max over beyond-law caps = 0.021pp (bar 1.0), no first-rise; WS20 cell spread 0.015pp. "Capacity alone decides above CROSS1" now measured through cap == WS20 — no re-divergence near full residency; exp707 rule band (0.528-0.625) unchanged, validity extended upward. B1 anchors byte-equal sealed exp706 @3328 (4th cross-run determinism witness); B4 det; exit 0; 318s; freeze 7145f31; ADDENDUM52. Follow-up controller re-adjudicated 18/18 checks from primary sources. Next: family-3 trace band-generality probe (third model family, existing frozen traces).

## exp709 — beyond-law ceiling probe hi + granite (CEILING_FLAT_BOTH; ledger #232)
Maps hi caps 2048->3396 (0.60->1.00 x WS20) and granite 1024->1228 (0.83->1.00 x WS20), each ending exactly at the working set: s_max hi 0.198pp @2304, granite 0.125pp @1088 (bar 1.0), no first-rise; WS20 cells 0.037 / 0.000pp. With exp708 (lo 0.021pp): "capacity alone decides above CROSS1" holds through cap == WS20 on 3/3 traces, 2 model families — exp707 rule (cap ~0.6 x WS20) has no upper validity bound found. C1 anchors byte-equal sealed exp704 (all 4 arms; burst lists identical; 5th/6th determinism witness); C4 det; exit 0; 344s; freeze 0ac3207; ADDENDUM53. Disclosed: run 1 died mid-grid with no measured values emitted; one bounded relaunch under the lane standing rule (runner sha intact, RELAUNCH.lock). Follow-up controller re-adjudicated 28/28 checks from primary sources. Next: exp710 HL-region probe (frontier).
