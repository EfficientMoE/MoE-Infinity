# Hy3-FP8 (`hy_v3`) performance benchmarks

First performance characterization of the Hy3-FP8 per-tensor FP8 expert
offloading path (issue #283): a dequant/kernel microbenchmark
(`bench_hy3_kernel.py`, no checkpoint needed) and a real-checkpoint
end-to-end TTFT/decode benchmark (`bench_hy3_e2e.py`). Follows the rules
in [docs/benchmarking.md](../../docs/benchmarking.md): warmup outside
measurement, TTFT separated from decode throughput, fixed config,
commit + command recorded.

## Reproducibility

| Field | Value |
| --- | --- |
| Commit | `4504955ec83efca71234566f6194c3dcbccfafbf` (branch `feat/hy3-fp8-support`; benchmark scripts added in the commit that carries this README) |
| Benchmark workflow | `benchmarks/hy3/bench_hy3_kernel.py`, `benchmarks/hy3/bench_hy3_e2e.py` |
| Model / checkpoint | `tencent/Hy3-FP8` snapshot `ecc1d8e194e093f33177f2f0ef7ce8f397b2d68b` (~299 GB, FP8 e4m3 experts, 80 layers / 79 sparse, 192 routed experts/layer, top-8, expert ≈ 18.87 MB FP8 across gate/up `[1536,4096]` + down `[4096,1536]`) |
| GPUs, CPU, RAM, storage | 6x NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB each; kernel bench on one idle GPU, e2e pinned to another idle GPU via `CUDA_VISIBLE_DEVICES`), 2x AMD EPYC 9355 (128 hw threads), 1.5 TB RAM, RAID0 NVMe (`/mnt/raid0nvme0`) |
| CUDA, PyTorch, Transformers | CUDA 13.0 (driver 590.48.01), PyTorch 2.11.0+cu130, Transformers 5.12.1 |
| Residency / offload / quantization | `device_memory_ratio=0.75`, `offload_path=/mnt/raid0nvme0/leyang/hy3-offload-bench` (persistent, ~290 GB), experts FP8 via `glm_fp8_block128` dequant-on-copy with uniform 128x128 scale grids expanded from per-tensor scalars; non-expert stack (~9 GB BF16) resident; HF eager attention (FlashAttention unavailable) |
| Batch / concurrency | batch 1, single request |
| Prompt / output lengths | 3 distinct short prompts (12-16 tokens), `max_new_tokens=64` |
| Warmup / iterations | kernel: warmup 20, iters 100; e2e: one 8-token warmup generate, then per prompt one `max_new_tokens=1` (TTFT) + one full greedy generate |
| Baseline | none (first hy_v3 measurement) |
| Metrics captured | p50/p90 µs + effective GB/s (kernel); TTFT, E2E, decode tok/s, expert cache hit rate, `expert_h2d_bytes_total` per token (e2e) |
| Notes | greedy (`do_sample=False`); e2e uses the v4flash_ab convention `decode_tok_s = (N-1)/(e2e - ttft)`; raw JSON in `results/` |

Commands (run from the repo root):

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python benchmarks/hy3/bench_hy3_kernel.py \
    --out benchmarks/hy3/results/hy3_kernel.json

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/mnt/raid0nvme0/leyang/MoE-Infinity-hy3-fp8:/mnt/raid0nvme0/leyang/moe-store-hy3 \
.venv/bin/python benchmarks/hy3/bench_hy3_e2e.py \
    --out benchmarks/hy3/results/hy3_e2e.json
```

The worktree must precede the editable install on `PYTHONPATH` so the
hy_v3-aware Python loads (a run that resolves the wrong tree builds the
store with `0 sparse` topology and fails in the dense fallback). A valid
load prints `TOPO: 79 stages, 79 sparse`.

## Kernel microbenchmark results

GPU: RTX PRO 6000 Blackwell Server Edition; warmup 20, iters 100;
effective GB/s computed over the FP8 input bytes so dequant and H2D
share a denominator.

| Tensor | Op | FP8 MB | p50 µs | p90 µs | eff GB/s (p50) |
| --- | --- | ---: | ---: | ---: | ---: |
| gate/up [1536,4096] | native fp8_dequant_blockwise (uniform grid) | 6.00 | 19.6 | 20.3 | 298.7 |
| gate/up [1536,4096] | torch (w.float() * s).to(bf16) | 6.00 | 39.8 | 41.8 | 147.2 |
| gate/up [1536,4096] | torch w.to(bf16) * s_bf16 | 6.00 | 29.3 | 30.2 | 199.9 |
| gate/up [1536,4096] | H2D pinned copy (same FP8 bytes) | 6.00 | 111.8 | 112.2 | 52.4 |
| down [4096,1536] | native fp8_dequant_blockwise (uniform grid) | 6.00 | 20.0 | 20.2 | 293.0 |
| down [4096,1536] | torch (w.float() * s).to(bf16) | 6.00 | 39.7 | 41.9 | 147.4 |
| down [4096,1536] | torch w.to(bf16) * s_bf16 | 6.00 | 29.4 | 29.5 | 199.2 |
| down [4096,1536] | H2D pinned copy (same FP8 bytes) | 6.00 | 111.9 | 112.8 | 52.4 |
| expert bundle (18.0 MB) | native dequant x3 (one expert) | 18.00 | 53.0 | 53.1 | 331.5 |
| expert bundle (18.0 MB) | H2D pinned copy x3 (one expert) | 18.00 | 335.6 | 336.4 | 52.4 |

| Tensor | dequant p50 / H2D p50 |
| --- | ---: |
| gate/up [1536,4096] | 0.175 |
| down [4096,1536] | 0.179 |
| expert bundle (18.0 MB) | 0.158 |

## End-to-end results (real checkpoint)

Cold store build + model load: 564 s (store reused on later runs).
Warmup generate (8 tokens): 6.8 s.

| Prompt | prompt tokens | TTFT (s) | E2E 64 tok (s) | decode tok/s | hit rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 2.69 | 38.40 | 1.764 | 0.663 |
| 2 | 12 | 3.26 | 34.07 | 2.045 | 0.632 |
| 3 | 13 | 4.22 | 28.58 | 2.586 | 0.626 |

| Metric | p50 |
| --- | ---: |
| TTFT (s) | 3.26 |
| E2E 64 tokens (s) | 34.07 |
| Decode tok/s | 2.045 |
| Expert H2D MB/token | 4539.4 |
| Expert cache hit rate | 0.626 |

Raw JSON: `results/hy3_kernel.json`, `results/hy3_e2e.json`.

## Analysis

### (a) Dequant kernel overhead vs H2D transfer

The native uniform-grid blockwise dequant takes **0.16-0.18x** the time
of the pinned H2D copy it follows (53 µs vs 336 µs per 18.87 MB expert).
It also already beats both pure-torch per-tensor upper bounds (19.6 µs
vs 29.3/39.8 µs per tensor), so a dedicated per-tensor scalar kernel
could not even halve the dequant cost that remains. In the e2e run,
~236 missed experts/token x 53 µs ≈ 12.5 ms of dequant per ~489 ms
decode step, i.e. **~2.5% of token time**. **No dedicated per-tensor
dequant kernel is warranted**; the uniform-grid expansion path is
effectively free next to PCIe.

### (b) What bounds decode

Decode p50 is 2.045 tok/s (~489 ms/token). The counters are internally
consistent: 79 sparse layers x top-8 = 632 expert activations/token;
with hit rate 0.626, misses ≈ 236/token x 18.87 MB = ~4.45 GB, matching
the measured 4539 MB/token (all-miss worst case would be ~11.9 GB/token,
so the cache absorbs ~62%). At the microbenchmark's raw pinned-H2D rate
(52.4 GB/s) those bytes would take only ~87 ms/token; the pipeline
instead sustains ~9.3 GB/s effective, so **decode is bound by the expert
fetch pipeline (host store reads, 1 MB pinned-chunk staging, per-expert
orchestration/sync), not by raw PCIe bandwidth and not by the dequant
kernel**. TTFT (3.26 s p50 at 12-16 prompt tokens) is dominated by the
same cold-cache expert fetches during prefill.

### (c) Improvement recommendations

1. **No kernel improvement needed** (supporting ratio: dequant/H2D =
   0.16; ~2.5% of decode token time). Close the "dedicated per-tensor
   dequant kernel" question as not worth it.
2. **Expert fetch pipeline (largest win).** Effective transfer rate is
   ~9.3 GB/s vs 52.4 GB/s raw pinned H2D on the same machine. Batching
   per-expert fetches into larger contiguous staged transfers (or a
   bigger pinned pool than 128 x 1 MB chunks) has an upper-bound win of
   ~5.6x on decode (489 -> ~87+ ms/token) if transfers reach the raw
   rate; even reaching half of it would roughly double decode tok/s.
3. **Hit rate / residency.** 0.626 hit rate with `device_memory_ratio`
   0.75 (~72 GB cache ≈ 25% of the 15,168-expert working set). Raising
   the ratio on a 96 GB card or improving prefetch accuracy reduces the
   4.5 GB/token miss traffic proportionally: each +0.1 hit rate removes
   ~1.2 GB/token, worth ~25% of the current decode step at the observed
   effective rate. Multi-GPU expert caching would scale this further.
