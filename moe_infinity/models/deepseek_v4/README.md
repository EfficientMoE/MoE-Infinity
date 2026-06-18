# DeepSeek-V4-Flash expert offloading for MoE-Infinity

Runs `deepseek-ai/DeepSeek-V4-Flash` with routed experts streamed from host
memory, so the model fits where its ~132 GB of FP4 experts otherwise would not.

## Why offloading is required

V4-Flash routed experts are FP4 (E2M1) and total ~132 GB across 43 layers ×
256 experts × 3 projections. A single 95 GB GPU cannot hold them. This module
keeps experts in pinned host RAM and streams only the experts a layer actually
routes to onto the GPU, with an LRU GPU cache and async copy-stream prefetch.

## Architecture

- The checkpoint is DeepSeek-native (not HuggingFace) and uses FP4 routed
  experts + FP8 shared experts + FP8 attention weights. `DeepseekV4ForCausalLM`
  in transformers cannot load it; the official `inference/model.py` can.
- Integration reuses the official model's validated MLA + DeepSeek Sparse
  Attention + hyper-connections, and replaces only routed-expert computation
  with `OfficialExpertHostStore` (host->GPU streaming) via
  `patch_moe_with_offload`. Shared experts stay resident.
- Routed-expert math uses the official `fp4_gemm` kernel (tilelang).

## Components

- `OfficialExpertHostStore` — per-expert host store; `prefetch(layer, ids)`
  (async copy stream) and `get(layer, id)` (LRU GPU cache, FP4-aware copy).
- `patch_moe_with_offload(model, store, official_module)` — replaces each
  `MoE.forward` to stream + run routed experts (respects TP per-rank expert
  range and all-reduce), keeping the official gate and shared experts.
- `load_offloaded_v4_flash(official_module, ckpt_path, config_path, device,
  shard_file, max_resident_experts)` — full loader: null-routed-expert
  construction, offload patch, non-expert weight load. Returns `(model, store)`.

Pure-PyTorch building blocks (used by unit tests / host venv):
`DeepSeekV4ExpertTensorIndexer`, `fp4_expert_forward`, `dequant_fp4_e2m1`,
`fp8_shared_expert_forward`, `dequant_fp8_blockwise`, routing helpers
(`topk_route`, `hash_route`, sqrtsoftplus), `SyncDeepSeekV4MoEBlock`,
`DeepSeekV4PythonExpertExecutor`, `HostOffloadBundleProvider`.

## Usage (full model, multi-GPU, in the v4flash docker image)

```python
import model as M                       # official inference/model.py
from generate import generate
from moe_infinity.models.deepseek_v4 import load_offloaded_v4_flash

model, store = load_offloaded_v4_flash(
    M, ckpt_path, "config.json", device, shard_file,
    max_resident_experts=64,
)
out = generate(model, [prompt_ids], max_new_tokens=128, eos_id=1, temperature=0.0)
```

`max_resident_experts` must be >= the max number of distinct experts routed in
any single layer per step (use 64 for full per-rank coverage; lower trades GPU
memory for more host->device traffic).

## Checkpoint prep

Convert HF weights to the official mp-sharded format once:

```bash
python convert.py --hf-ckpt-path <HF_CKPT> --save-path <OUT> \
  --n-experts 256 --model-parallel 4
```

The official sparse-attention kernel is tuned for TP-sharded head counts; run
with `--model-parallel 4` (16 heads/rank). mp1 (64 heads) exceeds the kernel's
shared-memory limit on a single GPU.

## Tests

Host venv (pure-torch building blocks + offload store round-trip):

```bash
DSV4_FLASH_CKPT=<HF_SNAPSHOT> pytest tests/python/v4/ -q
```

End-to-end golden parity (docker image with tilelang, 4 GPUs):

```bash
torchrun --nproc-per-node 4 tests/python/v4/e2e_mp4_offload.py
```

Verified: prefill argmax and 15-token greedy decode match the official golden
for the smoke prompts; ~5-6 GB/GPU resident with experts offloaded.

## Native C++ FP4 expert execution (single native path)

By default routed experts run via the official tilelang `fp4_gemm`. A native
CUDA path is also available and selected with `use_native=True`:

```python
model, store = load_offloaded_v4_flash(M, ckpt, cfg, dev, shard,
                                       max_resident_experts=8, use_native=True)
```

It uses `moe_infinity._v4_fp4`:
- `v4_fp4_dequant.cu` — FP4 E2M1 (packed uint8) + ue8m0 block-32 scale ->
  BF16 dequant CUDA kernel. Bit-exact vs the reference `dequant_fp4_e2m1`.
- `v4_expert_forward` — dequant w1/w2/w3 then BF16 SwiGLU GEMM (libtorch).
  Bit-exact vs the Python `fp4_expert_forward`.

Build (SM120 / Blackwell, e.g. RTX PRO 6000) inside the v4flash image:

```bash
MOE_ENABLE_SM120=1 CUTLASS_DIR=<cutlass> pip install -e .
```

### Why not a CUTLASS block-scaled FP4 MMA?

The intended single-kernel route was a CUTLASS 4.x SM120 block-scaled GEMM
(`mx_float8_t x mx_float4_t`). The stock nvfp4 example (`79a`) compiles and runs
on SM120 with `-gencode arch=compute_120a,code=sm_120a`, but the
`mx_float4_t` (e4m3 x e2m1) MMA atom hits a compile bug in
`cute/atom/mma_traits_sm120.hpp` (`fp4_shift_B`) in the available CUTLASS 4.2.
The dequant + BF16 GEMM path avoids that bug, keeps host->GPU traffic quantized
(FP4 bytes cross PCIe; dequant happens on-GPU), and is numerically exact, so it
is the shipping native path. A fused FP4 MMA can replace it once the CUTLASS
atom is fixed, behind the same `use_native` seam.

## End-to-end validation & performance

Hardware: 4x RTX PRO 6000 Blackwell (SM120), mp4 tensor parallel, docker
v4flash image. Prompt: "identity" (8-token prompt, 64 decode tokens).

### Correctness
- Native FP4 expert forward is **bit-exact** to the reference dequant path
  (`fp4_expert_forward`): per-expert max abs error = 0.0 on real weights.
- Full model, greedy decode vs the official sampled golden (temperature 0.6):
  - `identity`: 64/64 tokens match the golden prefix.
  - `tiny-math`: exact ("4").
  - Creative/long prompts (`haiku`) diverge as expected — the golden was
    sampled (temp 0.6), not greedy, so exact match is not meaningful there.
- Native vs tilelang `fp4_gemm` diverge slowly over decode steps; the native
  path is the more accurate one (bit-exact to reference), since tilelang adds
  FP8 activation-quant rounding.

### Throughput (decode, 64 tokens, mp4)

| Configuration                         | decode tok/s | ms/tok | peak GPU/rank |
|---------------------------------------|-------------:|-------:|--------------:|
| Resident (no offload, baseline)       |         9.01 |  111.0 |      42.98 GB |
| Offload native, cache=4               |         4.75 |  210.3 |       5.58 GB |
| Offload native, cache=16              |         5.16 |  193.8 |       5.71 GB |
| Offload native, cache=64              |         5.37 |  186.1 |       6.34 GB |
| Offload tilelang, cache=8             |         5.07 |  197.4 |       ~5.3 GB |

Observations:
- Offloading cuts resident GPU memory ~**6.8x** (43 GB -> 5.6-6.3 GB/rank) at
  a ~1.7-1.9x decode-latency cost — the expected memory/throughput trade.
- Larger GPU expert cache improves throughput (4.75 -> 5.37 tok/s from cache
  4 -> 64) for modest extra memory; cache 16 is a good default.
- Native vs tilelang decode throughput is comparable (5.16-5.37 vs 5.07 tok/s);
  native additionally is numerically exact to the reference.
- TTFT (prefill) ~0.45-0.75 s; warm host expert cache removes first-token
  streaming stalls.
