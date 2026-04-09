# MoE-Infinity

MoE-Infinity is a cost-effective, fast, and easy-to-use library for Mixture-of-Experts (MoE) inference.

MoE-Infinity is cost-effective yet fast:

- Offloading MoE's experts to host memory, allowing memory-constrained GPUs to serve MoE models.
- Minimizing the expert offloading overheads through several novel techniques: expert activation tracing, activation-aware expert prefetching, and activation-aware expert caching.
- Supporting LLM acceleration techniques (such as [FlashAttention](https://github.com/Dao-AILab/flash-attention)).
- Supporting multi-GPU environments with numerous OS-level performance optimizations.
- Achieving SOTA latency performance when serving MoEs in a resource-constrained GPU environment (in comparison with [vLLM](https://github.com/vllm-project/vllm), HuggingFace [Accelerate](https://github.com/huggingface/accelerate), [DeepSpeed](https://github.com/microsoft/DeepSpeed), [Mixtral-Offloading](https://github.com/dvmazur/mixtral-offloading), and [Ollama/LLama.cpp](https://github.com/ollama/ollama)).

MoE-Infinity is easy-to-use:

- HuggingFace model compatible, and HuggingFace programmer friendly.
- Supporting all available MoE checkpoints (including [DeepSeek-V2/V3](https://huggingface.co/collections/deepseek-ai/deepseek-v2-669a1c8b8f2dbc203fbd7746), [Meta NLLB-MoE](https://huggingface.co/facebook/nllb-moe-54b), [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1), [Qwen3-MoE](https://huggingface.co/Qwen/Qwen3-30B-A3B), [DBRX](https://huggingface.co/models?search=dbrx), [Jamba](https://huggingface.co/models?search=jamba), and [OLMoE](https://huggingface.co/models?search=olmoe)).

Note that: The open-sourced MoE-Infinity has been redesigned for making it HuggingFace-users friendly. This version is different from the version reported in the paper, which takes extreme performance as the top priority. Single-server multi-GPU inference is supported: expert parameters are distributed round-robin across all visible GPUs, with per-GPU caching, peer-to-peer transfers, and dedicated I/O threads. Multi-node distributed inference (across separate machines) is not yet supported in this open-sourced version.

## Contents
- [Key Features](#key-features)
- [Performance](#performance)
- [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Install from conda environment](#install-from-conda-environment)
    - [Install from PyPI](#install-from-pypi)
    - [Install from Source](#install-from-source)
    - [Enable FlashAttention (Optional)](#enable-flashattention-optional)
- [Usage and Examples](#usage-and-examples)
    - [Sample Code of Huggingface LLM Inference](#sample-code-of-huggingface-llm-inference)
    - [Running Inference](#running-inference)
    - [Benchmarking](#benchmarking)
- [Release Plan](#release-plan)
- [Contributing and Security](#contributing-and-security)
- [Citation](#citation)

## Key Features

- Expert offloading with activation-aware caching and prefetching for memory-constrained GPUs.
- KV cache offloading with paged attention support for long-context serving.
- Continuous batching serving engine with request scheduling, preemption, and prefix caching.
- Streaming responses for OpenAI-compatible chat completion APIs.
- Serving stability hardening with watchdogs and health monitoring.
- Memory coordination and expert prefetch coordination to improve throughput and utilization.

## Performance

Single GPU A5000 (24GB Memory), per-token-latency (seconds) for generation with a mixed dataset that includes [LongBench](https://huggingface.co/datasets/THUDM/LongBench), [GSM8K](https://huggingface.co/datasets/openai/gsm8k),  [FLAN](https://huggingface.co/datasets/Muennighoff/flan), [BIG-Bench](https://huggingface.co/datasets/bigbench) and [MMLU](https://huggingface.co/datasets/lukaemon/mmlu) datasets.
Lower per-token-latency is preferable.

|  | DeepSeek-V2-Lite-Chat | Mixtral-8x7b | Qwen3-30B-A3B | gpt-oss-20b |
| :---: | :---: | :---: | :---: | :---: |
| <ins>MoE-Infinity</ins> (FP16) | <ins>*0.100*</ins> | <ins>*0.735*</ins> | <ins>*0.150*</ins> | <ins>*0.555*</ins> |
| vLLM v0.18.1 | 0.011 | X | X | 0.007 |
| llama.cpp b8640 (Q4_K_M) | 0.006 | X | 0.007 | X |

> **—** = Not yet measured. Run [`benchmarks/comparison/run_all.sh`](benchmarks/comparison/run_all.sh) to populate.
> **X** = Model cannot run on this framework with a single 24GB GPU.
> Precision: MoE-Infinity uses FP16 with expert offloading (full quality, no quantization loss). vLLM uses FP8 for DeepSeek-V2-Lite (fell back from FP16 OOM) and native MXFP4 for gpt-oss-20b; Mixtral-8x7b and Qwen3-30B-A3B OOM at FP8. llama.cpp uses Q4_K_M GGUF quantization; Mixtral-8x7b exceeds 24GB at Q4_K_M; no GGUF is available for gpt-oss-20b.
> MoE-Infinity's expert offloading enables serving models that exceed GPU memory at full FP16 precision. Other frameworks require the full model to fit in VRAM (with quantization), limiting which models they can serve on a single 24GB GPU.
> See [Benchmark Reproduction Guide](docs/benchmark_reproduction.md) to reproduce these numbers.

<details>
<summary>Legacy comparison (Accelerate, DeepSpeed, Mixtral Offloading, Ollama, vLLM v0.8.5)</summary>

|  | NLLB-MoE-54B | Mixtral-8x7b | DeepSeek-V2-Lite-Chat | Qwen3-30B-A3B |
| :---: | :---: | :---: | :---: | :---: |
| <ins>MoE-Infinity</ins> | <ins>*0.119*</ins> | <ins>*0.735*</ins> | <ins>*0.100*</ins> | <ins>*0.150*</ins> |
| Accelerate | 3.071 | 6.633 | 1.743 | — |
| DeepSpeed (0.16.2) | 8.381 | 2.486 | 0.737 | 7.857 |
| Mixtral Offloading | X | 1.752 | X | X |
| Ollama | X | 0.903 | 1.250 | — |
| vLLM (v0.8.5) | X | 2.137 | 0.149 | 0.205 |

</details>

## Installation

We recommend installing MoE-Infinity in a virtual environment. To install MoE-Infinity, you can either install it from PyPI or build it from source.

### Prerequisites

- Python 3.8+
- CUDA-capable environment for GPU inference
- Recommended: isolated virtual environment

### Install from conda environment

```bash
conda create -n moe-infinity python=3.9
conda activate moe-infinity
# install from either PyPI or Source will trigger requirements.txt automatically
```

### Install from PyPI

```bash
# install stable release
pip install moe-infinity

# install nightly release (latest development build from main branch, published to PyPI as pre-release dev versions)
pip install --pre moe-infinity
```

### Install from Source

```bash
git clone https://github.com/EfficientMoE/MoE-Infinity.git
cd MoE-Infinity
pip install -e .
conda install -c conda-forge libstdcxx-ng=12 # assume using conda, otherwise install libstdcxx-ng=12 using your package manager or gcc=12
```

### Enable FlashAttention (Optional)

Install FlashAttention (>=2.5.2) for faster inference with the following command.
```bash
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn
```
Post-installation, MoE-Infinity will automatically integrate with FlashAttention to enhance performance.

### Enable FlashInfer (Optional)

Install [FlashInfer](https://flashinfer.ai/) for optimized paged attention kernels during prefill and decode. FlashInfer provides significant speedups for paged KV cache attention compared to standard PyTorch SDPA.

```bash
# For CUDA 12.4 + PyTorch 2.5:
pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.5/

# For CUDA 12.1 + PyTorch 2.4:
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
```

Check the [FlashInfer installation guide](https://docs.flashinfer.ai/installation.html) for other CUDA/PyTorch version combinations.

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

We provide a simple example to run inference on a Huggingface LLM model. The script will download the model checkpoint and run inference on the specified input text. The output will be printed to the console.

```bash
CUDA_VISIBLE_DEVICES=0 python examples/interface_example.py --model_name_or_path "deepseek-ai/DeepSeek-V2-Lite-Chat" --offload_dir <your local path on SSD>
```

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

## Release Plan

Recent releases and near-term roadmap:

* ✅ PyTorch runtime now includes KV cache offloading, paged attention kernels, continuous batching serving, streaming support, preemptive request scheduling with prefix caching, watchdog-based health hardening, memory coordination, and expert prefetch coordination.
* 🚧 We continue improving vLLM runtime interoperability.
* 🚧 Supporting expert parallelism for distributed MoE inference.
* More (We welcome contributors to join us!).

## Contributing and Security

- See [CONTRIBUTING.md](./CONTRIBUTING.md) for development workflow, coding standards, and tests.
- See [SECURITY.md](./SECURITY.md) for vulnerability reporting and support policy.

## Citation

If you use MoE-Inifity for your research, please cite our [paper](https://arxiv.org/abs/2401.14361):
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
