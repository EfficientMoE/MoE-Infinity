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
- Supporting all available MoE checkpoints (including [DeepSeek-V2/V3](https://huggingface.co/collections/deepseek-ai/deepseek-v2-669a1c8b8f2dbc203fbd7746), [Meta NLLB-MoE](https://huggingface.co/facebook/nllb-moe-54b), [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1), [Qwen3-MoE](https://huggingface.co/Qwen/Qwen3-30B-A3B), [Arctic](https://huggingface.co/Snowflake/snowflake-arctic-instruct), [DBRX](https://huggingface.co/models?search=dbrx), [Grok](https://huggingface.co/models?search=grok-1), [Jamba](https://huggingface.co/models?search=jamba), and [OLMoE](https://huggingface.co/models?search=olmoe)).

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

|  | NLLB-MoE-54B | Mixtral-8x7b | DeepSeek-V2-Lite-Chat | Qwen3-30B-A3B |
| :---: | :---: | :---: | :---: | :---: |
| <ins>MoE-Infinity</ins> | <ins>*0.119*</ins> | <ins>*0.735*</ins> | <ins>*0.100*</ins> | <ins>*0.150*</ins> |
| Accelerate | 3.071 | 6.633 |  1.743  | |
|DeepSpeed (0.16.2) | 8.381 | 2.486 | 0.737 | 7.857 |
|Mixtral Offloading| X | 1.752 | X |X|
|Ollama | X | 0.903 | 1.250 ||
|vLLM (v0.8.5)| X | 2.137 | 0.149 | 0.205 |



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

### OpenAI-Compatible Server

Start the OpenAI-compatible server locally
```bash
python -m moe_infinity.entrypoints.openai.api_server --model deepseek-ai/DeepSeek-V2-Lite-Chat --offload-dir ./offload_dir
```

Query the model via `/v1/completions`. (We currently only support the required fields, i.e., "model" and "prompt").
```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "prompt": "Hello, my name is"
    }'
```
You can also use `openai` python package to query the model.
```bash
pip install openai
python tests/python/integration/test_oai_completions.py
```

Query the model via `/v1/chat/completions`. (We currently only support the required fields, i.e., "model" and "messages").
```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a joke"}
        ]
    }'
```
You can also use `openai` python package to query the model.
```bash
pip install openai
python tests/python/integration/test_oai_chat_completions.py
```

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
