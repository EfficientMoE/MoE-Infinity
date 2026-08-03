# MoE-Infinity

MoE-Infinity is a cost-effective, fast, and easy-to-use library for Mixture-of-Experts (MoE) inference.

## Overview

MoE-Infinity runs large Mixture-of-Experts models on memory-constrained GPUs by **offloading expert weights to host memory (and SSD)** and fetching them just in time. An activation-aware cache keeps hot experts resident on the GPU, while activation tracing and prefetching hide most of the transfer cost. On top of the offloading runtime, MoE-Infinity ships a HuggingFace-compatible `MoE` class and an async, OpenAI-compatible serving engine with continuous batching, paged KV cache, and streaming.

This open-sourced version has been redesigned to be HuggingFace-friendly and differs from the version reported in the [paper](https://arxiv.org/abs/2401.14361), which prioritizes extreme performance. **Single-server multi-GPU inference is supported** — expert parameters are distributed round-robin across all visible GPUs, with per-GPU caching, peer-to-peer transfers, and dedicated I/O threads. Multi-node distributed inference (across separate machines) is not yet supported.

## Features

Key benefits include:

- **Cost-effective.** Expert offloading to host memory/SSD lets memory-constrained GPUs serve MoE models that would otherwise not fit. DeepSeek-V4-Flash additionally offloads **FP4-quantized** experts for a large memory reduction.
- **Fast.** Activation-aware expert caching, prefetching, and tracing minimize offloading overhead; fused CUDA kernels, CUDA graph capture, Marlin INT4 GEMM, and FP4/MXFP4 expert paths accelerate the hot path.
- **HuggingFace-native.** Drop-in `MoE` class with the familiar `from_pretrained` / `generate` workflow, compatible with standard HuggingFace checkpoints.
- **Production serving.** OpenAI-compatible HTTP server with continuous batching, paged KV cache, request scheduling with preemption, prefix caching, streaming (SSE), runtime hot reload, watchdog/health monitoring, and crash-recovery logging.
- **Acceleration-aware.** Automatically integrates with [FlashAttention](https://github.com/Dao-AILab/flash-attention) and [FlashInfer](https://flashinfer.ai/) when installed, with graceful fallback to built-in kernels.
- **Multi-GPU.** Single-server multi-GPU with round-robin expert distribution, per-GPU caching, and an in-memory N-way tensor-parallel shard loader.

## Contents
- [Key Features](#key-features)
- [Supported Models](#supported-models)
- [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Install from conda environment](#install-from-conda-environment)
    - [Install from PyPI](#install-from-pypi)
    - [Install from Source](#install-from-source)
    - [Enable FlashAttention (Optional)](#enable-flashattention-optional)
    - [Enable FlashInfer (Optional)](#enable-flashinfer-optional)
- [Usage and Examples](#usage-and-examples)
    - [Sample Code of Huggingface LLM Inference](#sample-code-of-huggingface-llm-inference)
    - [Running Inference](#running-inference)
    - [Benchmarking](#benchmarking)
    - [OpenAI-Compatible Server (Continuous Batching)](#openai-compatible-server-continuous-batching)
- [ContextPilot Integration (Optional)](#contextpilot-integration-optional)
- [Architecture](#architecture)
- [Release Plan](#release-plan)
- [Contributing and Security](#contributing-and-security)
- [Citation](#citation)

## Key Features

- **Expert offloading** with activation-aware caching, prefetching, and tracing for memory-constrained GPUs.
- **FP4/MXFP4 expert quantization** with host offloading (DeepSeek-V4-Flash native FP4 path, plus Triton fallback).
- **KV cache offloading** with paged attention for long-context serving.
- **Continuous batching** serving engine with request scheduling, preemption, swapping, and prefix caching.
- **Fused CUDA/Triton kernels** — fused QKV projection, fused Gate+Up+SiLU FFN, fused decode-phase paged attention, and Marlin INT4 (W4A16) GEMM.
- **CUDA graph capture** for decode batches to cut per-step launch overhead.
- **OpenAI-compatible API** with streaming chat/completions, runtime hot reload, and live config management.
- **Serving stability** hardening with watchdogs, health monitoring, and crash-recovery (incremental result writer).
- **Memory and prefetch coordination** to balance the GPU budget between experts and KV cache and improve throughput.

## Supported Models

MoE-Infinity supports HuggingFace MoE checkpoints registered in [`moe_infinity/common/constants.py`](./moe_infinity/common/constants.py):

| Model | Example checkpoints |
|---|---|
| [DeepSeek-V2 / V3](https://huggingface.co/collections/deepseek-ai/deepseek-v2-669a1c8b8f2dbc203fbd7746) | `deepseek-ai/DeepSeek-V2-Lite-Chat`, `deepseek-ai/DeepSeek-V3` |
| DeepSeek-V4-Flash (FP4 expert offloading) | requires a `transformers` build shipping `DeepseekV4ForCausalLM` |
| [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) | `mistralai/Mixtral-8x7B-Instruct-v0.1`, `Mixtral-8x22B` |
| [Qwen3-MoE](https://huggingface.co/Qwen/Qwen3-30B-A3B) | `Qwen/Qwen3-30B-A3B` |
| [Qwen3.5-MoE](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | `Qwen/Qwen3.5-35B-A3B` (text-only; see note) |
| [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2-FP8) | `zai-org/GLM-5.2-FP8` (requires `transformers` >= 5.12) |
| [GPT-OSS](https://huggingface.co/models?search=gpt-oss) | `openai/gpt-oss-*` |
| [DBRX](https://huggingface.co/models?search=dbrx) | `databricks/dbrx-instruct` |
| [Jamba](https://huggingface.co/models?search=jamba) | `ai21labs/Jamba-*` |
| [OLMoE](https://huggingface.co/models?search=olmoe) | `allenai/OLMoE-*` |
| [Meta NLLB-MoE](https://huggingface.co/facebook/nllb-moe-54b) | `facebook/nllb-moe-54b` |

> DeepSeek-V4-Flash is only registered when your installed `transformers` provides `DeepseekV4ForCausalLM`; otherwise it is skipped automatically.

> Qwen3.5-MoE (`Qwen3_5MoeForConditionalGeneration`, requires `transformers` >= 5.12) is a vision-language checkpoint served **text-only**: its 256 routed experts are offloaded while the small text backbone — token embeddings, the hybrid linear (GatedDeltaNet) / full attention layers, shared expert, and `lm_head` — stays resident on GPU. The v5 packed expert tensors are expanded to per-expert on load. Vision and MTP weights are present but unused for text generation.

> GLM-5.2 (`GlmMoeDsaForCausalLM`, `model_type="glm_moe_dsa"`) requires `transformers` >= 5.12 and is registered only when that class is importable (otherwise skipped automatically). Its 256 routed FP8 experts (block-scale e4m3, dequantized to BF16 on load) are offloaded; the 3 dense layers, shared expert, MLA attention, DSA indexer, and MTP layer stay resident. Sparse attention uses `attn_implementation="eager"`.

## Installation

We recommend installing MoE-Infinity in a virtual environment. To install MoE-Infinity, you can either install it from PyPI or build it from source.

### Prerequisites

- Python 3.10+ (3.12 recommended). Some required dependencies (e.g. `sglang-kernel`) publish wheels for Python ≥ 3.10 only, so Python 3.8/3.9 will fail to install.
- A CUDA-capable GPU. The from-source build targets compute capabilities `sm_80`/`sm_90` by default; for Blackwell (`sm_120`, e.g. RTX PRO 6000 / RTX 50-series) build with `MOE_ENABLE_SM120=1` (see [Install from Source](#install-from-source)).
- When building from source, a CUDA toolkit whose major version matches your installed PyTorch build (PyTorch enforces this at compile time).
- Recommended: isolated virtual environment (conda or venv).

### Install from conda environment

```bash
conda create -n moe-infinity python=3.12
conda activate moe-infinity
# install from either PyPI or Source will trigger requirements.txt automatically
```

### Install from PyPI

> **Note:** Official PyPI wheels are not published yet — the current `moe-infinity` entry on PyPI is a placeholder that does **not** contain the runtime (importing `MoE` from it will fail). Until the official release, please [install from source](#install-from-source).

```bash
# (available once official wheels are published) stable release
pip install moe-infinity

# (available once official wheels are published) nightly / pre-release build
pip install --pre moe-infinity
```

### Install from Source

Building the CUDA/C++ extensions needs a few system packages, the CUTLASS headers, and PyTorch installed **before** `pip install -e .`:

```bash
# 1. System build dependencies (Debian/Ubuntu; use your distro's equivalents otherwise)
sudo apt-get update && sudo apt-get install -y build-essential cmake ninja-build git uuid-dev

# 2. Build tools + PyTorch. Match PyTorch's CUDA build to your CUDA toolkit
#    (pick the index URL for your CUDA version from https://pytorch.org).
pip install "setuptools>=78.1.1,<82" wheel ninja py-cpuinfo
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 3. CUTLASS headers (header-only; no separate build required)
git clone --depth 1 https://github.com/NVIDIA/cutlass.git ~/cutlass
export CUTLASS_DIR=~/cutlass

# 4. Build and install MoE-Infinity
git clone https://github.com/EfficientMoE/MoE-Infinity.git
cd MoE-Infinity
pip install --no-build-isolation -e .

# 5. Ensure a recent libstdc++ is available for the compiled extensions
conda install -c conda-forge libstdcxx-ng=12 # with conda; otherwise install libstdc++ (gcc 12+) via your package manager
```

**Building for Blackwell / SM120 GPUs** (RTX PRO 6000, RTX 50-series): the default build targets `sm_80`+`sm_90`. Enable the `sm_120` path (and the native FP4 kernel) explicitly:

```bash
MOE_ENABLE_SM120=1 MOE_ENABLE_SM90=0 CUTLASS_DIR=~/cutlass pip install --no-build-isolation -e .
```

### Enable FlashAttention (Optional)

FlashAttention is **not** installed by default. Install it (>=2.5.2) for faster inference:
```bash
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn
# or, equivalently, via the optional extra:
pip install -e '.[flash_attn]'
```
Post-installation, MoE-Infinity will automatically integrate with FlashAttention to enhance performance.

### Enable FlashInfer (Optional)

Install [FlashInfer](https://flashinfer.ai/) for optimized paged attention kernels during prefill and decode. FlashInfer provides significant speedups for paged KV cache attention compared to standard PyTorch SDPA.

```bash
# Install the FlashInfer Python package (JIT-compiles kernels to match your Torch/CUDA):
pip install flashinfer-python
# or, equivalently, via the optional extra:
pip install -e '.[flashinfer]'
```

Check the [FlashInfer installation guide](https://docs.flashinfer.ai/installation.html) for prebuilt-wheel options matching specific CUDA/PyTorch versions.

Post-installation, MoE-Infinity will automatically detect and use FlashInfer for faster paged attention in both prefill and decode phases. When FlashInfer is not installed, MoE-Infinity gracefully falls back to its built-in attention kernels with no behavior change.

## Usage and Examples

We provide a simple API for diverse setups, including single GPU and multiple GPUs. The following examples show how to use MoE-Infinity to run generation on a Huggingface LLM model.

### Important Note

- The `offload_path` must be unique for each MoE model. Reusing the same `offload_path` for different MoE models will result in unexpected behavior.


### Sample Code of Huggingface LLM Inference

```python
import torch
import os
from transformers import AutoTokenizer
from moe_infinity import MoE

user_home = os.path.expanduser('~')

checkpoint = "deepseek-ai/DeepSeek-V2-Lite-Chat"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

config = {
    "offload_path": os.path.join(user_home, "moe-infinity"),
    "device_memory_ratio": 0.75, # 75% of the device memory is used for caching, change the value according to your device memory size on OOM
}

model = MoE(checkpoint, config)

input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda:0")

output_ids = model.generate(input_ids)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(output_text)
```

### Running Inference

Run on a single GPU:
```bash
CUDA_VISIBLE_DEVICES=0 python script.py
```

Run on multiple GPUs (expert parameters are automatically distributed across all visible devices):
```bash
CUDA_VISIBLE_DEVICES=0,1 python script.py
```

We provide ready-to-run examples under [`examples/`](./examples). The scripts download the checkpoint, run inference on the input, and print the output.

```bash
# Minimal single-prompt example (DeepSeek-V2-Lite-Chat)
CUDA_VISIBLE_DEVICES=0 python examples/deepseek_v2_chat_example.py --offload_dir <your local path on SSD>

# Streaming benchmark example over GSM8K (TTFT + decode timing)
CUDA_VISIBLE_DEVICES=0 python examples/interface_example.py --model_name_or_path "deepseek-ai/DeepSeek-V2-Lite-Chat" --offload_dir <your local path on SSD>
```

**Suggested hardware (DeepSeek-V2-Lite-Chat):** a single GPU with **>= 16 GB VRAM** (e.g. RTX 4090 / A5000 / A100) plus a fast local SSD for `--offload_dir`. Lower `--device_memory_ratio` (default `0.75`) if you hit OOM on smaller GPUs.

### DeepSeek-V4-Flash (FP4 Expert Offloading)

DeepSeek-V4-Flash has two usage paths depending on your checkpoint.

**Path A — HuggingFace-native `MoE` API.** If your installed `transformers` ships `DeepseekV4ForCausalLM`, V4-Flash works through the same drop-in `MoE` class as above:

```python
from moe_infinity import MoE

model = MoE("deepseek-ai/DeepSeek-V4-Flash", {
    "offload_path": "/ssd/moe-infinity/deepseek-v4-flash",
    "device_memory_ratio": 0.75,
})
```

**Path B — Official FP4 offload loader.** The native V4-Flash checkpoint (FP4 routed experts + FP8 shared/attention weights) cannot be loaded by HuggingFace; it is driven through the official `inference/model.py` and MoE-Infinity's `load_offloaded_v4_flash` adapter, which streams the ~132 GB of FP4 experts from pinned host RAM (resident GPU memory drops to ~5-6 GB/rank):

```bash
torchrun --nproc-per-node 4 examples/deepseek_v4_flash_example.py \
    --ckpt-path <MP_SHARDED_CKPT> --config-path <MP_SHARDED_CKPT>/config.json \
    --max-resident-experts 16
```

**Suggested hardware / environment (Path B):** **4x GPUs** (tensor-parallel mp4; mp1 exceeds the sparse-attention kernel's shared-memory limit), **>= ~140 GB pinned host RAM**, and the `v4flash` docker image (tilelang `fp4_gemm`; on Blackwell/SM120 the native `moe_infinity._v4_fp4` CUDA path is auto-selected and is 1.5–3.2x faster). The checkpoint must first be converted to the official mp-sharded format. See [`moe_infinity/models/deepseek_v4/README.md`](./moe_infinity/models/deepseek_v4/README.md) for checkpoint conversion, kernel selection, and validation details.

### GLM-5.2 (FP8 Expert Offloading)

GLM-5.2 (`zai-org/GLM-5.2-FP8`) runs through the drop-in `MoE` class. Its FP8 block-scaled routed experts are dequantized to BF16 on load and offloaded to host memory:

```python
from moe_infinity import MoE

model = MoE("zai-org/GLM-5.2-FP8", {
    "offload_path": "/ssd/moe-infinity/glm-5.2",
    "device_memory_ratio": 0.5,
})
```

> **Memory note:** dequant-on-load expands the ~753 GB of FP8 experts to ~1.5 TB in host RAM. On hosts with limited RAM, prefer the fp8-in-store path (keeps experts FP8 in the store) once available. Requires `transformers` >= 5.12.

### Benchmarking

For correct throughput and latency measurement, it is critical to separate **prefill time (TTFT)** from **decode throughput**. Including prefill in your throughput calculation will produce misleadingly low numbers.

We provide a `StopWatch` utility and ready-to-use benchmark scripts. See the **[Benchmarking Guide](docs/benchmarking.md)** for:

- How to correctly measure decode throughput vs TTFT
- Common measurement pitfalls and how to avoid them
- Ready-to-use benchmark scripts (`benchmarks/serving/`)
- Fair comparison methodology with llama.cpp, vLLM, and other frameworks
- Tuning `device_memory_ratio` for optimal performance

Quick example using the benchmark scripts:
```bash
# Single-request baseline (TTFT + per-token latency + peak memory)
python benchmarks/serving/baseline_performance.py \
    --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --offload-dir /path/to/offload/dir

# Throughput sweep across batch sizes
python benchmarks/serving/throughput.py \
    --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --offload-dir /path/to/offload/dir \
    --batch-sizes 1 2 4 8 16

# Latency percentiles (TTFT + ITL at p50/p90/p99)
python benchmarks/serving/latency.py \
    --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --offload-dir /path/to/offload/dir \
    --concurrency 1 2 4 8
```

### OpenAI-Compatible Server (Continuous Batching)

MoE-Infinity includes a continuous batching serving engine with an OpenAI-compatible API. The server supports concurrent requests, streaming, request scheduling with preemption, and paged KV cache management.

Start the server:
```bash
python -m moe_infinity.entrypoints.openai.api_server_v2 \
    --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --offload-dir ./offload_dir \
    --device-memory-ratio 0.5 \
    --kv-cache-ratio 0.15 \
    --max-batch-size 8
```

| Flag | Default | Description |
|---|---|---|
| `--device-memory-ratio` | 0.75 | Fraction of GPU memory for expert caching. Lower this if you hit OOM (0.5 is a safe starting point for 24GB GPUs). |
| `--kv-cache-ratio` | 0.25 | Fraction of remaining GPU memory for paged KV cache blocks. |
| `--max-batch-size` | 32 | Maximum number of concurrent sequences in a batch. |
| `--enable-prefix-caching` | off | Enable prefix caching for shared prompt prefixes. |

You can also start the server programmatically from Python:
```python
from moe_infinity import MoE

model = MoE("deepseek-ai/DeepSeek-V2-Lite-Chat", {
    "offload_path": "./offload_dir/deepseek-v2-lite",
    "device_memory_ratio": 0.5,
})
model.serve(host="0.0.0.0", port=8000, offload_dir="./offload_dir")
```

Query via `/v1/completions`:
```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "prompt": "Hello, my name is",
        "max_tokens": 32
    }'
```

Query via `/v1/chat/completions` with streaming:
```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a joke"}
        ],
        "max_tokens": 128,
        "stream": true
    }'
```

Supported request fields: `model`, `prompt`/`messages`, `max_tokens`, `temperature`, `top_p`, `stop`, `stream`.

The server returns `finish_reason: "stop"` when the model emits an EOS token or hits a stop sequence, and `finish_reason: "length"` when `max_tokens` is reached.

The server also exposes operational endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/models` | GET | List the loaded model(s). |
| `/health` | GET | Liveness/readiness state (`STARTING`, `HEALTHY`, `UNHEALTHY`). |
| `/metrics` | GET | Prometheus-format serving metrics. |
| `/admin/stats` | GET | Engine statistics (queue depth, batch sizes, cache hit rates). |
| `/v1/config` | GET / POST | Inspect and update runtime configuration. |
| `/v1/reload` | POST | Hot-reload server Python modules without restarting the process. |

You can also use the `openai` Python package:
```bash
pip install openai
python tests/python/integration/test_oai_completions.py
python tests/python/integration/test_oai_chat_completions.py
```

## ContextPilot Integration (Optional)

ContextPilot is an optional overlap-aware prompt optimization layer for shared-prefix and multi-turn workloads. You can enable it inside the OpenAI-compatible server before tokenization, or extend it into KV allocation and scheduling for deeper reuse gains.

Phase B quick start, in-process middleware:

```bash
python -m moe_infinity.entrypoints.openai.api_server_v2 \
    --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --offload-dir ./offload_dir \
    --enable-contextpilot
```

Set `CONTEXTPILOT_ENABLED=0` to force-disable ContextPilot at runtime, even if the CLI flag is enabled.

Measured baseline on single A5000 (24 GB) with DeepSeek-V2-Lite-Chat (expert offloading):

| Workload | TTFT p50 | E2E p50 | Prefill tok/s |
|---|---:|---:|---:|
| Shared-prefix RAG | 3.70s | 5.29s | 25.4 |
| Multi-turn conversation | 3.82s | 5.46s | 26.3 |
| Batch with overlap | 2.21s | 2.69s | 35.5 |
| No-overlap baseline | 3.40s | 4.86s | 4.2 |

Projected Phase B/C improvements (based on [ContextPilot benchmarks on vLLM/SGLang](https://github.com/EfficientContext/ContextPilot)):

| Phase | Integration mode | Expected TTFT reduction | Expected token savings |
|---|---|---:|---:|
| Phase B | In-process middleware | 15–25% | 20–30% |
| Phase C | Deep scheduler integration | 20–30% | 25–35% |

Actual improvements depend on context overlap ratio. Run `python benchmarks/contextpilot/compare_phases.py` for detailed dry-run projections, or run Phase B against a live server for real measurements.

See [docs/contextpilot/README.md](docs/contextpilot/README.md) for setup details, CLI flags, environment variables, admin endpoints, and troubleshooting.

## Architecture

For a contributor-oriented map of the codebase — the two execution paths (synchronous `engine/` vs async `serving/`), module layout, request lifecycle, and the public API surface — see **[ARCHITECTURE.md](./ARCHITECTURE.md)**.

## Release Plan

Recent releases and near-term roadmap:

* ✅ Expert offloading runtime with activation-aware caching, prefetching, and tracing.
* ✅ FP4/MXFP4 expert quantization with host offloading (DeepSeek-V4-Flash native FP4 path).
* ✅ Continuous batching server with paged KV cache, streaming, preemptive scheduling, and prefix caching.
* ✅ Fused CUDA/Triton kernels (QKV, Gate+Up+SiLU FFN, decode attention), Marlin INT4 GEMM, and CUDA graph capture.
* ✅ Serving stability: watchdog/health monitoring, runtime hot reload, live config, and crash-recovery (incremental writer).
* 🚧 Improving vLLM runtime interoperability.
* 🚧 Expert parallelism and multi-node distributed MoE inference.
* 🚧 OpenAI-compatible Batch API (`/v1/batches`).
* More (We welcome contributors to join us!).

## Contributing and Security

- See [CONTRIBUTING.md](./CONTRIBUTING.md) for development workflow, coding standards, and tests.
- See [SECURITY.md](./SECURITY.md) for vulnerability reporting and support policy.

## Citation

If you use MoE-Infinity for your research, please cite our [paper](https://arxiv.org/abs/2401.14361):
```bibtex
@misc{moe-infinity,
  author       = {Leyang Xue and
                  Yao Fu and
                  Zhan Lu and
                  Chuanhao Sun and
                  Luo Mai and
                  Mahesh Marina},
  title        = {MoE{-}Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache},
  archivePrefix= {arXiv},
  eprint       = {2401.14361},
  year         = {2024}
}
```
