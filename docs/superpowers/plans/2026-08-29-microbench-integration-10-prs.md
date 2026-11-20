# Microbenchmark-Driven Integration of PRs #176–#185 — Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to run this plan wave-by-wave. Steps use checkbox (`- [ ]`) syntax for tracking. This is a **measurement + decision** plan, not a code-writing plan: each task RUNS an existing per-branch benchmark harness and records a verdict. Do **not** modify feature code on the plan branches while benchmarking.

**Goal:** Produce a defensible KEEP / ROLLBACK / NEEDS-DATA verdict for each of the 10 optimization PRs (#176–#185) and a recommended integration order + composition matrix, by running each PR's already-shipped benchmark harness with real models on this 6-GPU host, in parallel.

**Architecture:** Each PR lives on its own `plan/*` branch and already ships (a) an opt-in feature, (b) a benchmark harness that emits a schema'd JSON with a built-in verdict, and (c) recorded verification that is **blocked only on real-model CUDA evidence** (models were absent / downloads disabled / single-GPU degenerate concurrency). This host removes those blockers (6× 96 GB Blackwell, 1.5 TB RAM, 756 GB shm, downloads enabled). We create one **git worktree + isolated venv per branch**, build its native extension in-tree, pin each benchmark to a dedicated GPU, run standalone A/B (Wave 1), then run the interaction stacks that matter for integration (Wave 2), then synthesize verdicts + merge order (Wave 3).

**Tech Stack:** MoE-Infinity (`moe_infinity`), PyTorch 2.12+cu130, CUDA 13.1, native C++/CUDA ext (CMake/Ninja + setuptools), FlashInfer (to be installed for sm_120), Nsight Systems (`nsys`), HuggingFace models (DeepSeek-V2-Lite-Chat, Qwen3, gpt-oss).

---

## 0. Assumptions & Decisions Needed (confirm before Wave 1)

These defaults are used throughout. Correct any that are wrong; each materially changes scope.

1. **Deliverable = decisions, not merges.** This plan runs benchmarks and produces verdicts + a recommended integration order. It does **not** merge/rebase any PR. (If you also want me to execute the merges after verdicts, that is a separate follow-up.)
2. **"Determine the integration" = both** (a) per-PR KEEP/ROLLBACK/NEEDS-DATA, and (b) a composition matrix + recommended merge order across the interacting/mutually-exclusive PRs. Wave 3 produces both.
3. **Canonical models** (verify exact HF ids before Wave 0):
   - `deepseek-ai/DeepSeek-V2-Lite-Chat` — expert-offload / KV-swap / memory PRs (MLA, host-only offload).
   - `Qwen/Qwen3-30B-A3B` — Qwen3 paged-attention serving PRs (#178, #181) + INT8 perplexity gate (#183).
   - `Qwen/Qwen3-8B` (dense GQA, resident) — decode-CUDA-graph model mode (#182); DeepSeek-V2-Lite is MLA and only exercises #182's capability-fallback path.
   - `openai/gpt-oss-20b` (MXFP4) — #176 native dispatch parity and a candidate low-bit model for #184.
4. **Isolation model:** one git worktree per branch under `$WT`, each with its own venv (`--system-site-packages`) and in-tree native build, so 6 A/B processes run concurrently without cross-contaminating compiled `.so`s. (Alternative: sequential single-checkout with rebuilds — ~6× slower, no parallelism. Not recommended given 6 idle GPUs.)
5. **Statistical rigor:** each latency A/B uses ≥30 measured rounds after ≥3 warmup rounds, median-of-medians across 3 repetitions; a PR is KEEP only if its verdict holds on all 3 repetitions.
6. **#179 and #184 PR bodies say "plan-only," but their branch diffs contain full implementation + benchmarks + native code.** We treat them as implemented and Wave 0 gates each on a successful build + smoke import. If a branch is genuinely plan-only (no runnable feature), it auto-returns NEEDS-DATA.

**Open questions for you:**
- Q1: Confirm the three model ids in (3) — especially the exact Qwen3 id your Qwen3 paged-attention path supports, and the #182 resident-GQA target.
- Q2: Is any regression budget non-default? (Plan uses each harness's built-in gate, e.g. #176 p99≤+5%/p50≤+2%; #183 PPL≤+1% rel.)
- Q3: Should Wave 2 composition include the full stacks listed in §7, or only the two hard mutual-exclusions?

---

## 1. Environment Reality (captured 2026-08-29)

```
GPUs:      6× NVIDIA RTX PRO 6000 Blackwell (sm_120), 97 887 MiB each, idle
CPU/RAM:   128 cores, 1511 GiB total (~1153 GiB free)
shm:       /dev/shm = 756 GiB   (use for host-only offload)
disk:      /mnt/raid0nvme0 = 4.8 TiB free
torch:     2.12.0+cu130, cuda_avail=True, device_count=6
toolchain: nsys 2025.1.3 ✓ ; cmake 3.28.3 ; ninja 1.13 ; nvcc 13.1 ✓
MISSING:   flashinfer (import fails) ; ncu ; compute-sanitizer
HF:        HF_HOME=/mnt/raid0nvme0/public/huggingface (hub/ empty) ; HF_HUB_OFFLINE unset → downloads enabled
repo:      /mnt/raid0nvme0/leyang/MoE-Infinity ; moe_infinity 0.0.1 importable
```

Consequences:
- The recorded PR blockers ("model not in cache / downloads disabled", "shared cache not writable", "single-GPU concurrency-1 degenerate TPOT") are **all resolved** here.
- **FlashInfer must be installed for sm_120** (Wave 0, Task 0.3). It gates #178, #181, #183 (baseline arm), and #182 model-mode. #182 **fixture** mode and all expert-path PRs (#176/#177/#179/#180/#184/#185) do **not** need it.
- `compute-sanitizer`/`ncu` absent → #183's sanitizer memcheck and any ncu step remain **blocked** (documented, not worked around).

---

## 1b. Wave 0 execution status (2026-08-29 — COMPLETE)

Wave 0 has been run. State on disk:
- **Models downloaded** (complete, 0 incomplete) to writable `HF_HOME=/mnt/raid0nvme0/leyang/hf`: DeepSeek-V2-Lite-Chat (30 G), Qwen3-30B-A3B (57 G), Qwen3-8B (16 G), gpt-oss-20b (39 G). (The shared `/mnt/raid0nvme0/public/huggingface` cache is READ-ONLY for this user.)
- **FlashInfer 0.6.18** installed in the base env; sees device cc `(12,0)` = sm_120. **CUTLASS** headers at `~/cutlass`. **nsys 2025.1.3** present. `ncu`/`compute-sanitizer` absent (→ #183 sanitizer step stays blocked).
- **10 worktrees** at `$WT=/mnt/raid0nvme0/leyang/moe-wt/<branch>`, each with fresh in-tree **sm_120** native `.so`s (`_store`, `_marlin`, `_paged_attn`, `_kv_cache`, `_engine`, `_v4_fp4`) built with `MOE_ENABLE_SM120=1 MOE_ENABLE_SM90=0 CUTLASS_DIR=~/cutlass`.
- **Isolation solved without venvs.** anaconda's `sitecustomize.py` injects the base repo onto `sys.path`, shadowing worktrees. Launcher `$WT/_runwt.py` pins `MOE_WT_ROOT` first + purges base `moe_infinity`, run with the base interpreter (torch is in the `~/.local` user site — do NOT set `PYTHONNOUSERSITE`). Also `_smoke.py` present.
- **Smoke gate: 10/10** — every worktree loads its own branch code (discriminating markers verified: #176 `_validate_gpu_routing_overlap`, #183 `runtime.kv_cache_format`, #182 `kernel.paged_kv_write`) + native `_store`. #178/#179 OpenAI servers launch in isolation via `-m` (chunked-prefill flags confirmed).

**Known Wave-1 follow-ups (not blockers for setup):**
- **WikiText-2 loader** fails under `datasets`+`huggingface_hub 1.16.1` (`Invalid HF URI 'wikitext'`). Affects **only** #183's PPL *release gate*; the #183 A/B storage/quality matrix still runs. Fix at Wave-1 start (pin `datasets`, or load via explicit `wikitext/wikitext-2-raw-v1` parquet path).
- The 10 `.venv/` dirs created during build are unused at runtime (wrapper mechanism replaced them).

---

## 2. The 10 PRs at a glance

| PR | Branch (`origin/plan/…`) | Feature | Primary harness | Model | Built-in gate | Needs FlashInfer | Needs native build |
|----|----|----|----|----|----|:--:|:--:|
| #176 | gpu-only-expert-routing | GPU-only expert routing (native active-expert discovery, stream-ordered completion) | `benchmarks/serving/latency.py` + `expert_io_microbench/run_decision_profile.py` + `nsys_parser.py` | DeepSeek-V2-Lite | `gpu_routing_verdict`: ROLLBACK if tpot_p99>base×1.05 or tpot_p50>base×1.02; route_failures=0; fallback=0; Nsight stream/device sync=0 | No | **Yes** |
| #177 | adaptive-expert-kv-memory | Bounded adaptive expert↔KV memory reallocation | `benchmarks/serving/memory.py` + `adaptive_memory_trace.py` | DeepSeek-V2-Lite | fixed vs adaptive: no TTFT/TPOT regression + controller fires + safety gates hold | No | **Yes** (`test_sparse_cache_resize`) |
| #178 | chunked-prefill-scheduling | Bounded decode-first chunked prefill | `benchmarks/serving/chunked_prefill_latency.py` | Qwen3-30B-A3B | paired disabled/enabled: decode TPOT/TTFT tails improve under mixed load, no correctness change | **Yes** | No |
| #179 | phase-specific-expert-policy | Prefill/decode-split expert admission over one ExpertResidencyManager | `benchmarks/serving/phase_specific_expert_policy.py` | DeepSeek-V2-Lite | policy off vs on: token-ID parity + no latency regression | No | **Yes** (`expert_residency`) |
| #180 | overlap-aware-expert-prefetch | Overlap-budgeted speculative expert prefetch | `benchmarks/expert_io_microbench/bench_overlap_prefetch.py` (off\|observe\|enforce) | DeepSeek-V2-Lite | `build_report`: enforce reduces expert bubble / improves decode TPOT p50; waste below cap; route unchanged | No | **Yes** (`prefetch_op`) |
| #181 | prefix-kv-reuse | Longest-prefix KV reuse on Qwen3 paged path | `benchmarks/serving/prefix_cache_benchmark.py` (disabled\|cold\|warm) | Qwen3-30B-A3B | **correctness:** cold==warm token+logit digests (mismatch→exit 2); **perf:** warm TTFT p50 ≪ cold | **Yes** | No |
| #182 | decode-cuda-graphs | Decode-only CUDA graph capture/replay (resident GQA) | `benchmarks/serving/decode_cuda_graph.py` (`--mode fixture`/`model`) | none (fixture) / Qwen3-8B (model) | fixture: KV checksums equal, 0 capture failures, replay coverage; ratio eager_p50/replay_p50 (>1.0 = win) | model-mode only | **Yes** (paged-attn kernel) |
| #183 | kv-cache-quantization | Opt-in INT8-sym paged KV format | `benchmarks/serving/kv_cache_quantization.py` | Qwen3-30B-A3B + WikiText-2 | storage ratio ≈0.508 (≈2× KV); PPL(int8) − PPL(native) ≤ +1% rel; no silent fallback | **Yes** (baseline arm) | **Yes** (INT8 kernel) |
| #184 | adaptive-mixed-precision-experts | Adaptive mixed-precision routed-expert cache | `benchmarks/adaptive_precision/bench_{policy,transfer,e2e}.py` + `report.py` | gpt-oss-20b (allowlist) | `release_gate`: quality attestation passes + peak/transfer bytes ↓ vs canonical + TPOT not regressed | No | **Yes** (marlin/fp8) |
| #185 | async-hierarchical-kv | Async GPU↔pinned-host KV swap | `benchmarks/serving/kv_offload_benchmark.py` (`--kv-swap-mode async`/`sync`) | DeepSeek-V2-Lite | async vs sync: failures=0, no ticket leaks, hidden swap latency (E2E/decode stall ↓) at KV pressure | No | **Yes** (coordinator) |

---

## 3. Dependency & Mutual-Exclusion Matrix (drives integration)

**Hard mutual-exclusions (enforced in-code — Wave 2 verifies the guard fires, does NOT co-benchmark for perf):**
- **#176 ⊥ #180** — `gpu_only_expert_routing=true` is rejected with any active `overlap_prefetch_policy` (ArcherConfig raises before engine construction). These are **competing** expert-path optimizations; pick the winner by data.
- **#178 ⊥ #181** — chunked prefill rejects simultaneous prefix caching at startup (`ValueError`). **Competing** serving-path features that share `LayeredPagedKVStore`/`PagedBatchLengths`; pick winner or ship both behind the guard.

**Hard dependency (must compose):**
- **#184 → #179** — adaptive mixed-precision must rebase onto #179's single `ExpertResidencyManager` ("no second residency authority"). #184 cannot be benchmarked in isolation of #179's substrate.

**Shared-substrate interactions (compatible; Wave 2 checks no negative interaction + additive benefit):**
- **KV stack:** #183 (INT8 store) + #185 (async swap of that store) + #177 (adaptive KV/expert memory) — all touch KV storage/transfer; async swap must correctly slice INT8 scales.
- **Serving stack:** #181 (prefix reuse warms KV) + #182 (decode graphs) ; #178 (chunked prefill) + #182.
- **Expert stack:** #179 (phase policy) + {#176 or #180}.

---

## 4. Parallelization Strategy (6 GPUs)

- **One benchmark process per GPU** via `CUDA_VISIBLE_DEVICES=<n>`. Latency A/B numbers are only valid on an uncontended GPU — never co-locate two latency runs on one GPU.
- **Wave 1 = two rounds** (10 PRs across 6 GPUs):
  - **Round A (6 concurrent):** #176→GPU0, #177→GPU1, #179→GPU2, #180→GPU3, #185→GPU4, #182-fixture→GPU5
  - **Round B:** #178 uses **two** servers (baseline GPU0:8001 + candidate GPU5:8002) + a client process; #181→GPU1, #183→GPU2, #184→GPU3. (#178 and #179 are client/server harnesses — see Tasks 1.178/1.179.)
- Each worktree is fully isolated (own venv + in-tree `.so`), so Round-A processes cannot interfere at the Python/native level.
- Model memory fits comfortably: DeepSeek-V2-Lite ≈ 16 GB bf16; Qwen3-30B-A3B ≈ 60 GB bf16; both < 96 GB with room for KV + graphs.

Wave-launcher pattern (used in every wave). **Isolation is via `$WT/_runwt.py` + the base interpreter — NOT venvs.** anaconda's `sitecustomize.py` injects the base repo onto `sys.path`, so `_runwt.py` pins the worktree ahead of it and purges the base `moe_infinity`. torch lives in the user site (`~/.local`), so do **not** set `PYTHONNOUSERSITE`.
```bash
# run_pr.sh <gpu> <branch> <script.py | -m module> [args...]   → logs to $RESULTS/<branch>.log
gpu=$1; b=$2; shift 2
CUDA_VISIBLE_DEVICES=$gpu MOE_WT_ROOT="$WT/$b" env -u PYTHONPATH \
  /home/leyang/anaconda3/bin/python "$WT/_runwt.py" "$@" > "$RESULTS/$b.log" 2>&1 &
# in-process:      run_pr.sh 0 gpu-only-expert-routing "$WT/gpu-only-expert-routing/benchmarks/serving/latency.py" --model ... --offload-dir ...
# client/server:   run_pr.sh 0 chunked-prefill-scheduling -m moe_infinity.entrypoints.openai.api_server_v2 --port 8001 --model ... --offload-dir ...
```

---

## 5. Canonical parameters (source once per shell)

```bash
export REPO=/mnt/raid0nvme0/leyang/MoE-Infinity
export WT=/mnt/raid0nvme0/leyang/moe-wt                 # worktrees root
export OFFLOAD_SHM=/dev/shm/moe-offload                 # host-only offload (756 GiB shm)
export OFFLOAD_DISK=/mnt/raid0nvme0/leyang/moe-offload  # disk offload (4.8 TiB)
export RESULTS=/mnt/raid0nvme0/leyang/moe-bench-results
export HF_HOME=/mnt/raid0nvme0/leyang/hf                 # writable; the shared /mnt/raid0nvme0/public/huggingface cache is READ-ONLY for this user
export MODEL_DS=deepseek-ai/DeepSeek-V2-Lite-Chat
export MODEL_QWEN3=Qwen/Qwen3-30B-A3B                   # verify id (Q1)
export MODEL_QWEN3_DENSE=Qwen/Qwen3-8B                  # #182 resident GQA (verify)
export MODEL_GPTOSS=openai/gpt-oss-20b                  # #176 native dispatch, #184 low-bit
mkdir -p "$WT" "$RESULTS" "$OFFLOAD_SHM" "$OFFLOAD_DISK"
```

---

## 6. Wave 0 — Shared prerequisites (run once, mostly parallelizable)

### Task 0.1: Sanity + connectivity
- [ ] **Step 1: Confirm GPUs + downloads work.**
  Run: `nvidia-smi --query-gpu=index,memory.total --format=csv && huggingface-cli download $MODEL_DS --revision main config.json`
  Expected: 6 GPUs listed; `config.json` downloads (proves HF egress works).
- [ ] **Step 2: If egress is blocked,** set `HF_HUB_OFFLINE=1` and pre-stage models manually; otherwise continue.

### Task 0.2: Pre-download models (parallel, disk cache is shared across worktrees)
- [ ] Run each in background:
  ```bash
  for M in "$MODEL_DS" "$MODEL_QWEN3" "$MODEL_QWEN3_DENSE" "$MODEL_GPTOSS"; do
    huggingface-cli download "$M" > "$RESULTS/dl_${M//\//_}.log" 2>&1 &
  done; wait
  ```
  Expected: each model materializes under `$HF_HOME/hub`. (DeepSeek-V2-Lite ~31 GB, Qwen3-30B-A3B ~60 GB, Qwen3-8B ~16 GB, gpt-oss-20b ~40 GB → well within 4.8 TiB.)
- [ ] Download WikiText-2 for #183 PPL gate:
  `python -c "import datasets; datasets.load_dataset('wikitext','wikitext-2-raw-v1')"`

### Task 0.3: Install FlashInfer for sm_120 (gates #178/#181/#183/#182-model)
- [ ] **Step 1: Try wheel.** `pip install flashinfer-python` then `python -c "import flashinfer; print(flashinfer.__version__)"`.
  Expected: import succeeds.
- [ ] **Step 2: If the wheel lacks sm_120/CUDA-13 support,** build from source with `TORCH_CUDA_ARCH_LIST=12.0` (Blackwell) per FlashInfer docs, or rely on its JIT path. Record which path worked.
- [ ] **Step 3: Validate.** Run `python benchmarks/serving/validate_flashinfer.py` (from repo `dev`/main) on GPU0.
  Expected: import + correctness checks GO. If FlashInfer cannot be built for sm_120, mark #178/#181/#183-baseline/#182-model as **NEEDS-DATA (flashinfer_unavailable)** and proceed with the rest.

### Task 0.4: CUTLASS headers + one worktree + isolated venv + **sm_120** native build per branch (parallel)
- [ ] **Prereq (once): CUTLASS headers** (required by the from-source build):
  ```bash
  [ -d ~/cutlass ] || git clone --depth 1 https://github.com/NVIDIA/cutlass.git ~/cutlass
  export CUTLASS_DIR=~/cutlass
  ```
- [ ] For each `PR:BRANCH` in the §2 table — build with the **Blackwell sm_120** flags from the README. **Without `MOE_ENABLE_SM120=1` the native ext targets sm_80/sm_90 and will not run on these GPUs:**
  ```bash
  git -C "$REPO" worktree add "$WT/$BRANCH" "origin/plan/$BRANCH"
  python -m venv --system-site-packages "$WT/$BRANCH/.venv"
  "$WT/$BRANCH/.venv/bin/pip" install "setuptools>=78.1.1,<82" "setuptools-scm>=8" wheel ninja py-cpuinfo
  MOE_ENABLE_SM120=1 MOE_ENABLE_SM90=0 CUTLASS_DIR=~/cutlass \
    "$WT/$BRANCH/.venv/bin/pip" install --no-build-isolation -e "$WT/$BRANCH"   # in-tree sm_120 ext for THIS branch
  ```
  Build the 10 in parallel (independent; 128 cores handle it). Expected: each `pip install -e` exits 0 and produces the branch's compiled `sm_120` extension. Ensure a recent `libstdc++` (gcc 12+) is on the library path.
- [ ] **Step: Smoke-import each** (proves the branch's feature + native symbols load):
  ```bash
  PYTHONPATH="$WT/$BRANCH" "$WT/$BRANCH/.venv/bin/python" -c "import moe_infinity, moe_infinity.utils.config as c; print('$BRANCH ok')"
  ```
  Expected: `<branch> ok`. **Any branch that fails to build/import → verdict NEEDS-DATA (build_failed); exclude from Wave 1.** (This is the gate that catches genuinely plan-only #179/#184 if their code is incomplete.)

### Task 0.5: Per-branch offline test gate (fast confidence before spending GPU time)
- [ ] For each branch, run its CPU suite headline from the PR body (e.g. #178 `tests/python/serving`, #183 KV suite) with `-m "not gpu"`:
  `"$WT/$BRANCH/.venv/bin/python" -m pytest <branch-suite> -q -m "not gpu"`
  Expected: matches the PR-recorded pass counts. Divergence = environment problem to fix before benchmarking.

---

## 7. Wave 1 — Standalone per-PR microbenchmarks (verdict per PR)

Each task: pin GPU, run A/B, emit JSON to `$RESULTS`, apply the built-in gate 3× (median-of-medians), record KEEP/ROLLBACK/NEEDS-DATA. Commands assume `cd "$WT/$BRANCH"` with that branch's venv python.

### Task 1.176: GPU-only expert routing → GPU0
**Fixes the recorded degeneracy:** keep prefill batch tokens ≤ `kMaxTokens=256` so decode TPOT is non-degenerate. Use concurrency×prompt_len ≤ 256.
- [ ] **Baseline (off):**
  ```bash
  CUDA_VISIBLE_DEVICES=0 python benchmarks/serving/latency.py \
    --model $MODEL_DS --offload-dir $OFFLOAD_SHM \
    --concurrency 4 --num-rounds 30 --warmup-rounds 3 \
    --prompt-length 64 --max-new-tokens 32 \
    --gpu-only-expert-routing off \
    --output-json $RESULTS/176_latency_off.json
  ```
- [ ] **Candidate (on)**, feeding the baseline for the verdict:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python benchmarks/serving/latency.py \
    --model $MODEL_DS --offload-dir $OFFLOAD_SHM \
    --concurrency 4 --num-rounds 30 --warmup-rounds 3 \
    --prompt-length 64 --max-new-tokens 32 \
    --gpu-only-expert-routing on \
    --routing-baseline-json $RESULTS/176_latency_off.json \
    --output-json $RESULTS/176_latency_on.json
  ```
- [ ] **Nsight sync-gate** (decisive semantic evidence):
  ```bash
  CUDA_VISIBLE_DEVICES=0 nsys profile -o $RESULTS/176_on --force-overwrite true \
    python benchmarks/expert_io_microbench/run_decision_profile.py \
    --model $MODEL_DS --offload-dir $OFFLOAD_SHM --hardware-tag rtxpro6000-sm120 \
    --mode host-only --iters 3 --gpu-only-expert-routing on \
    --output-json $RESULTS/176_decision.json
  python benchmarks/expert_io_microbench/nsys_parser.py $RESULTS/176_on.nsys-rep \
    --output-json $RESULTS/176_nsys.json
  ```
- [ ] **Also sweep** concurrency 8 / prompt-length 32 (256 tokens) to confirm the verdict holds off-degenerate.
- [ ] **Gate → KEEP** iff: `176_latency_on.json.decision == KEEP` (tpot_p99 ≤ base×1.05 **and** tpot_p50 ≤ base×1.02), `route_failures==0`, `fallback_count==0`, and `176_nsys.json` shows `stream_synchronize_count==0 && device_synchronize_count==0`. Else ROLLBACK with the emitted reasons.

### Task 1.177: Adaptive expert/KV memory → GPU1
- [ ] **Deterministic trace sim** (fast correctness of controller):
  `CUDA_VISIBLE_DEVICES=1 python benchmarks/serving/adaptive_memory_trace.py --output-json $RESULTS/177_trace.json`
- [ ] **Fixed arm** (static ratios; `--adaptive-memory` omitted = off):
  ```bash
  CUDA_VISIBLE_DEVICES=1 python benchmarks/serving/memory.py \
    --model $MODEL_DS --offload-dir $OFFLOAD_SHM \
    --batch-size 8 --prompt-length 128 --max-new-tokens 32 \
    --device-memory-ratio 0.75 --kv-cache-ratio 0.25 \
    --warmup-runs 3 --repetitions 3 \
    --output-json $RESULTS/177_fixed.json
  ```
- [ ] **Adaptive arm** (same, plus the bare `--adaptive-memory` flag):
  ```bash
  CUDA_VISIBLE_DEVICES=1 python benchmarks/serving/memory.py \
    --model $MODEL_DS --offload-dir $OFFLOAD_SHM \
    --batch-size 8 --prompt-length 128 --max-new-tokens 32 \
    --device-memory-ratio 0.75 --kv-cache-ratio 0.25 --adaptive-memory \
    --warmup-runs 3 --repetitions 3 \
    --baseline-json $RESULTS/177_fixed.json \
    --output-json $RESULTS/177_adaptive.json
  ```
- [ ] **Gate → KEEP** iff adaptive shows: (a) `controller_reallocations > 0` (feature actually fired), (b) all safety gates (hard budget / min capacity / reserve) never violated, (c) TTFT p50 and TPOT p50 within +2% of fixed (opt-in must not regress the static case), and (d) demonstrable memory headroom gain (peak_gpu_memory_mb or admitted-tokens improvement). Else ROLLBACK / observe-only.
  > Flags verified against the branch's `memory.py` argparse: `--device-memory-ratio`, `--kv-cache-ratio`, bare `--adaptive-memory` (presence = on), `--warmup-runs`, `--repetitions`, `--baseline-json`.

### Task 1.178: Chunked prefill → baseline server GPU0:8001 + candidate server GPU5:8002 — **requires FlashInfer + Qwen3**
> `chunked_prefill_latency.py` is an **HTTP client** comparing two already-running servers (`--baseline-url`/`--candidate-url`), not an in-process harness. Launch both servers first (each on its own GPU), then run the client.
- [ ] **Launch baseline server** (chunked prefill OFF):
  ```bash
  CUDA_VISIBLE_DEVICES=0 python -m moe_infinity.entrypoints.openai.api_server_v2 \
    --host 127.0.0.1 --port 8001 --model $MODEL_QWEN3 --offload-dir $OFFLOAD_DISK &
  ```
- [ ] **Launch candidate server** (chunked prefill ON):
  ```bash
  CUDA_VISIBLE_DEVICES=5 python -m moe_infinity.entrypoints.openai.api_server_v2 \
    --host 127.0.0.1 --port 8002 --model $MODEL_QWEN3 --offload-dir $OFFLOAD_DISK \
    --enable-chunked-prefill --prefill-starvation-threshold-steps 8 &
  ```
  (Confirm in Task 0.5 whether the server also exposes `--prefill-chunk-size`; if so add `--prefill-chunk-size 256` to the candidate.)
- [ ] **Run the paired client** once both servers answer `/health`:
  ```bash
  python benchmarks/serving/chunked_prefill_latency.py \
    --baseline-url http://127.0.0.1:8001/v1 --candidate-url http://127.0.0.1:8002/v1 \
    --tokenizer $MODEL_QWEN3 --model-label qwen3-30b-a3b --gpu-label rtxpro6000-sm120 \
    --short-requests 64 --long-requests 16 \
    --short-prompt-tokens 128 --long-prompt-tokens 8192 \
    --max-tokens 128 --rounds 5 \
    --output-json $RESULTS/178_ab.json
  ```
- [ ] **Gate → KEEP** iff the candidate reduces **decode-row TPOT/TTFT tails** on the short (decode-heavy) requests while long prefills still advance, with no correctness change or throughput collapse. Neutral/regression → ROLLBACK (keep opt-in). NEEDS-DATA if FlashInfer unavailable.

### Task 1.179: Phase-specific expert policy → server GPU2:8003 (client `--policy off|on`)
> `phase_specific_expert_policy.py` is an **HTTP client** (`--server-url`, `--policy off|on`). The policy is a server-side config, so run the client against a policy-OFF server, then a policy-ON server, and let `--compare-json` validate parity + deltas.
- [ ] **OFF arm:** launch the server without the phase policy, then:
  ```bash
  CUDA_VISIBLE_DEVICES=2 python -m moe_infinity.entrypoints.openai.api_server_v2 \
    --host 127.0.0.1 --port 8003 --model $MODEL_DS --offload-dir $OFFLOAD_SHM &
  python benchmarks/serving/phase_specific_expert_policy.py \
    --server-url http://127.0.0.1:8003/v1 --model $MODEL_DS --policy off \
    --prompt-lengths 128 2048 --output-lengths 16 256 --concurrency 1 8 \
    --warmup 1 --repeats 5 --output-json $RESULTS/179_off.json
  ```
- [ ] **ON arm:** restart the server with the phase policy enabled (exact flag/env confirmed in Task 0.5 from `docs/serving.md`/`docs/configuration.md`, e.g. `--phase-specific-expert-policy`), then:
  ```bash
  python benchmarks/serving/phase_specific_expert_policy.py \
    --server-url http://127.0.0.1:8003/v1 --model $MODEL_DS --policy on \
    --prompt-lengths 128 2048 --output-lengths 16 256 --concurrency 1 8 \
    --warmup 1 --repeats 5 --compare-json $RESULTS/179_off.json \
    --output-json $RESULTS/179_on.json
  ```
- [ ] **Gate → KEEP** iff the `--compare-json` parity check passes (identical token IDs OFF vs ON) **and** signed TPOT/TTFT deltas within ±2%. Parity break → ROLLBACK (blocking). A KEEP here unblocks Task 1.184 (this PR is its **substrate**).

### Task 1.180: Overlap-aware expert prefetch → GPU3
- [ ] **Paired off | observe | enforce:**
  ```bash
  CUDA_VISIBLE_DEVICES=3 python benchmarks/expert_io_microbench/bench_overlap_prefetch.py \
    --model $MODEL_DS --offload-dir $OFFLOAD_SHM \
    --iters 30 --warmup-iters 3 --max-new-tokens 32 \
    --output-json $RESULTS/180_report.json
  CUDA_VISIBLE_DEVICES=3 python benchmarks/expert_io_microbench/run_decision_profile.py \
    --model $MODEL_DS --offload-dir $OFFLOAD_SHM --hardware-tag rtxpro6000-sm120 \
    --mode host-only --overlap-prefetch-policy enforce --iters 3 \
    --output-json $RESULTS/180_decision.json
  ```
- [ ] **Gate → KEEP (enforce)** iff `build_report` shows enforce reduces expert-transfer bubble / improves decode TPOT p50 vs off, `waste_bytes` below the report's cap, and exact-route correction shows no coverage loss. If observe helps but enforce regresses → **KEEP-as-observe**. Else ROLLBACK.

### Task 1.181: Prefix KV reuse → GPU1 (Round B) — **requires FlashInfer + Qwen3**
- [ ] **disabled | cold | warm** shared-prefix workload (flags verified against the branch argparse; the harness default model is already `Qwen/Qwen3-30B-A3B`). Do a `--dry-run` schema check first, then the real run:
  ```bash
  CUDA_VISIBLE_DEVICES=1 python benchmarks/serving/prefix_cache_benchmark.py \
    --model $MODEL_QWEN3 --offload-dir $OFFLOAD_DISK \
    --shared-prefix-tokens 1024 --suffix-tokens 128 \
    --output-json $RESULTS/181_ab.json
  ```
- [ ] **Gate → KEEP** iff: (correctness, blocking) cold and warm produce **equal token + logit digests** (harness aborts exit 2 on mismatch — a nonzero exit is an automatic ROLLBACK), **and** (perf) warm TTFT p50 is materially below cold TTFT p50 on the shared-prefix workload. NEEDS-DATA if FlashInfer unavailable.

### Task 1.182: Decode CUDA graphs → GPU5 (fixture) + GPU3 (Round B, model)
- [ ] **Fixture qualification (no model, decisive):**
  ```bash
  CUDA_VISIBLE_DEVICES=5 python benchmarks/serving/decode_cuda_graph.py \
    --mode fixture --batch-sizes 1 2 4 8 --context-sizes 128 512 2048 \
    --warmup-iters 5 --measure-iters 100 --profile-launches \
    --output-json $RESULTS/182_fixture.json
  ```
- [ ] **Model capability (dense GQA Qwen3):**
  ```bash
  CUDA_VISIBLE_DEVICES=3 python benchmarks/serving/decode_cuda_graph.py \
    --mode model --model $MODEL_QWEN3_DENSE --offload-dir $OFFLOAD_DISK \
    --batch-sizes 1 --context-sizes 128 --warmup-iters 1 --measure-iters 20 \
    --output-json $RESULTS/182_model.json
  ```
- [ ] **Gate → KEEP** iff fixture shows: per-layer KV checksums equal (correctness), 0 capture failures, high replay coverage, and observed ratio `eager_p50_us/replay_p50_us > 1.0` at small batch/context (launch-overhead win). No speedup threshold is claimed by the harness, so KEEP here = "correctness-clean + net-positive at target shapes; ship opt-in." Model mode need only prove the capability gate + eager fallback.

### Task 1.183: INT8 KV-cache quantization → GPU2 (Round B) — **requires FlashInfer + Qwen3 + WikiText-2**
- [ ] **A/B storage/quality matrix** (flags verified against the branch argparse):
  ```bash
  CUDA_VISIBLE_DEVICES=2 python benchmarks/serving/kv_cache_quantization.py \
    --model $MODEL_QWEN3 --offload-dir $OFFLOAD_DISK \
    --formats native int8_sym --context-lengths 2048 8192 32768 \
    --batch-sizes 1 4 --decode-tokens 128 --warmups 2 --repeats 5 \
    --strict-fallback \
    --output-json $RESULTS/183_ab.json
  ```
- [ ] **Perplexity release gate** (env-gated in the branch):
  `MOE_KV_QUANT_PPL=1 CUDA_VISIBLE_DEVICES=2 python -m pytest tests/python/e2e/test_kv_cache_quantized_perplexity.py -q`
- [ ] **Gate → KEEP** iff: `effective_kv_cache_format==int8_sym` (no silent native fallback under `--strict-fallback`), storage ratio ≈ 0.508 (≈2× KV capacity) realized, and WikiText-2 `PPL(int8) − PPL(native) ≤ +1%` relative. MLA models correctly falling back (`mla_not_validated`) is expected, not a failure. Record that `compute-sanitizer` memcheck is **blocked (tool absent)**.

### Task 1.184: Adaptive mixed-precision experts → GPU3 (Round B) — **depends on #179**
- [ ] **Precondition:** Task 1.179 KEEP. Rebase this worktree onto #179's ExpertResidencyManager (or benchmark on a combined #179+#184 worktree — see Task 2.d). If #179 is ROLLBACK/NEEDS-DATA, this PR auto-returns **NEEDS-DATA (substrate_missing)**.
- [ ] **Verified caveat:** the three `benchmarks/adaptive_precision/bench_*.py` modules expose **no standalone CLI** (no `__main__`/argparse); only `report.py` is a CLI (`report.py <input.json> --output <out.json>`). The arms are produced by the branch's benchmark driver/tests, then reduced:
  ```bash
  # 1) In Task 0.5, identify the arm driver: grep how bench_e2e is invoked
  #    (tests/python/unit/test_adaptive_precision_benchmark_schema.py references it).
  # 2) Run that driver on GPU3 to emit $RESULTS/184_e2e.json (arms: canonical|static-low|adaptive).
  # 3) Reduce to a release-gate report:
  python benchmarks/adaptive_precision/report.py $RESULTS/184_e2e.json --output $RESULTS/184_report.json
  ```
- [ ] **Gate → KEEP** iff `184_report.json.release_gate == PASS`: quality attestation passes for every admitted variant (fingerprint-qualified allowlist), adaptive reduces `peak_accounted_bytes`/H2D vs canonical, and TPOT p99 not regressed beyond the report's threshold.
- [ ] **Realistic outcome:** given (a) no standalone arm CLI, (b) hard dependency on #179's ExpertResidencyManager, and (c) a required released mixed-precision allowlist/manifest for `$MODEL_GPTOSS`, #184 most likely returns **NEEDS-DATA** in Wave 1 unless the driver + allowlist are supplied.

### Task 1.185: Async hierarchical KV swap → GPU4
- [ ] **async vs sync at KV pressure** (per PR #185 runbook):
  ```bash
  for MODE in async sync; do
  CUDA_VISIBLE_DEVICES=4 python benchmarks/serving/kv_offload_benchmark.py \
    --model $MODEL_DS --offload-dir $OFFLOAD_DISK \
    --num-requests 64 --prompt-length 2048 --max-new-tokens 128 \
    --warmup-requests 8 --trials 5 --host-memory-mib 2048 --max-inflight-mib 1024 \
    --kv-swap-mode $MODE --output-json $RESULTS/185_$MODE.json ; done
  ```
- [ ] **Gate → KEEP** iff async shows: `failures==0`, no leaked tickets (all retired), and hidden swap latency vs sync — i.e. E2E/decode-stall or effective decode TPOT improves (or is neutral) while sustaining higher KV pressure. Observed per-swap p99 may rise (includes polling) as long as end-to-end improves. Regression → ROLLBACK.

**Wave 1 exit criterion:** every PR has a JSON verdict artifact in `$RESULTS/` and a KEEP / ROLLBACK / KEEP-as-observe / NEEDS-DATA label recorded in the Wave 3 table.

---

## 8. Wave 2 — Interaction & composition microbenchmarks

Only run stacks whose components were KEEP in Wave 1. Build a **combined worktree per stack** (cherry-pick/merge the relevant branches into a scratch integration branch) so features coexist; rebuild its venv/ext.

### Task 2.a: Verify mutual-exclusion guards (correctness, fast)
- [ ] **#176+#180:** construct `ArcherConfig(gpu_only_expert_routing=True, overlap_prefetch_policy="enforce")` → expect raise before engine construction.
- [ ] **#178+#181:** start engine with both `--enable-chunked-prefill` and prefix cache enabled → expect startup `ValueError`.
- [ ] **Gate:** both guards fire. If a guard does NOT fire, that is a blocking integration bug (report immediately).

### Task 2.b: KV stack #183 + #185 (+ #177) → GPU0
- [ ] Combined worktree with INT8 store + async swap (+ adaptive memory). Run the #185 async/sync A/B **with `--kv-cache-format int8_sym`**.
- [ ] **Gate:** async swap correctly transfers INT8 payload **and** its FP16 scales (byte accounting matches; no corruption → digest/PPL unchanged vs Wave-1 #183), failures=0. Confirms the two KV features compose. Report combined memory headroom (INT8 2× capacity × async overlap).

### Task 2.c: Serving stack #181 + #182, and #178 + #182 → GPU1 / GPU2
- [ ] **#181+#182:** prefix reuse warms KV, decode graphs replay decode — run prefix warm workload with decode CUDA graphs enabled; gate on cold==warm digests **and** decode replay checksums equal (both correctness gates hold together).
- [ ] **#178+#182:** chunked prefill + decode graphs — gate on chunked-prefill decode TPOT improvement persisting with graph replay, checksums equal.
- [ ] Since **#178 ⊥ #181**, these two serving stacks are alternatives; the winner of Task 1.178 vs 1.181 decides which serving stack to integrate.

### Task 2.d: Expert stack #179 + {#176 or #180} and #179 + #184 → GPU3 / GPU4
- [ ] **#179 + winner(#176,#180):** phase policy composed with the surviving expert-path optimization; gate on token-ID parity + no TPOT regression vs the standalone winner.
- [ ] **#179 + #184:** the dependency stack; this is the real environment for Task 1.184's `release_gate`. Gate as in 1.184.

**Wave 2 exit criterion:** for every compatible stack, a "composes cleanly + additive-or-neutral" or "conflicts (reason)" record; for every exclusion, "guard fires."

---

## 9. Wave 3 — Integration decision synthesis

### Task 3.1: Aggregate verdicts
- [ ] Collect all `$RESULTS/*.json`, extract each harness's `decision`/`release_gate`/digest/checksum/sync-count fields, and emit one table:
  ```bash
  python - <<'PY' > $RESULTS/VERDICTS.md
  import json, glob, os
  # load every *_*.json, map PR→{verdict, key_metrics, gate_reasons}; render markdown table
  PY
  ```
- [ ] Produce `docs/superpowers/reports/2026-08-29-integration-verdicts.md` with columns: PR | verdict | headline metric (Δp50/Δp99/PPL/ratio/bytes) | gate reasons | blockers.

### Task 3.2: Recommended integration order + composition matrix
- [ ] Resolve the two exclusions by data: **#176 vs #180** (expert path) and **#178 vs #181** (serving path) — integrate the KEEP winner; keep the loser behind its opt-in flag or drop.
- [ ] Emit the **composition matrix** (which KEEP features ship together) and a **merge order** honoring dependencies, e.g. (data-dependent draft):
  1. #179 (substrate) → 2. #182-fixture + #176-or-#180 (independent, high-confidence) → 3. KV family #183 + #185 (+ #177) → 4. serving winner (#178 xor #181) + #182-model → 5. #184 (last; on #179).
- [ ] For each ROLLBACK/NEEDS-DATA, record the specific missing evidence (e.g. `flashinfer_unavailable`, `substrate_missing`, `ppl_regression`) so it is actionable.

### Task 3.3: Present to user
- [ ] Deliver `VERDICTS.md` + merge-order + composition matrix + per-PR blockers. Stop for go/no-go on merges.

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| FlashInfer won't build for sm_120/CUDA-13 | Try wheel → source (`TORCH_CUDA_ARCH_LIST=12.0`) → JIT; if all fail, #178/#181/#183-baseline/#182-model = NEEDS-DATA(flashinfer). Rest proceed. |
| 10 in-tree native builds collide | One venv + worktree per branch; never share compiled `.so`. Verified by per-branch smoke import (Task 0.4). |
| Latency numbers polluted by GPU sharing | One benchmark per GPU via `CUDA_VISIBLE_DEVICES`; two-round Wave 1. |
| #176 TPOT degenerate again | Keep prefill batch tokens ≤ `kMaxTokens=256` (concurrency×prompt_len ≤ 256); sweep 4×64 and 8×32. |
| #179/#184 actually plan-only | Wave 0 build+import gate auto-flags NEEDS-DATA(build_failed) before GPU spend. |
| `compute-sanitizer`/`ncu` absent | #183 sanitizer + any ncu step recorded as blocked(tool-absent); not worked around. |
| Model id / support mismatch (Qwen3) | Q1 confirmation + Task 0.2 download validates before Wave 1. |
| Harness flag drift vs PR body | Task 0.5 runs `--help` per harness; adjust exact flags before the run (only source of allowed inline flexibility). |

## 11. Self-Review Checklist (run before executing)
- [ ] Every PR #176–#185 has a Wave-1 task, a model, a GPU, and an explicit gate. ✅ (10/10)
- [ ] Both mutual-exclusions and the one dependency have Wave-2 coverage. ✅
- [ ] No benchmark shares a GPU with another latency run. ✅
- [ ] FlashInfer-dependent PRs are marked and degrade to NEEDS-DATA, not silent failure. ✅
- [ ] Metric names match harness output (`tpot_p50_ms`, `tpot_p99_ms`, `effective_kv_cache_format`, `release_gate`, `eager_p50_us/replay_p50_us`). ✅
- [ ] Verdict labels are consistent: KEEP / KEEP-as-observe / ROLLBACK / NEEDS-DATA. ✅
