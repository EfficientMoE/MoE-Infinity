# Benchmark Comparison Reproduction on NVIDIA A5000

This guide shows how to reproduce the single-GPU benchmark comparison on an NVIDIA A5000 with 24GB VRAM. It focuses on setup and execution. For metric definitions, timing methodology, and fairness notes, see the [Benchmarking Guide](benchmarking.md).

## Prerequisites

- Hardware: NVIDIA A5000 (24GB VRAM), PCIe 4.0 recommended for faster expert offloading
- Software: Docker with NVIDIA Container Toolkit, CUDA 12.x driver (check with `nvidia-smi`), about 200GB of disk space for model downloads
- Network: HuggingFace access for model downloads, set `HF_TOKEN` if you need gated models

## Quick Start

```bash
# Clone and build
git clone https://github.com/EfficientMoE/MoE-Infinity.git
cd MoE-Infinity

# Run all benchmarks (takes 2-3 hours)
bash benchmarks/comparison/run_all.sh \
    --offload-dir /path/to/ssd/offload \
    --results-dir benchmarks/comparison/results/

# View comparison table
cat benchmarks/comparison/results/comparison_table.md
```

Expected runtime: about 2 to 3 hours, covering 4 models, 3 frameworks, warmup, 20 measured runs, and a 60 second cooldown between frameworks.

## Per-Framework Setup

### MoE-Infinity

- Docker image: `moe-infinity-bench`, built from `docker/Dockerfile`
- Key flags:
  - `--offload-dir`: use an SSD if possible, NVMe is preferred
  - `--device-memory-ratio`: defaults to `0.75`
- Build command:

```bash
docker build -t moe-infinity-bench -f docker/Dockerfile .
```

- Run standalone:

```bash
docker run --gpus '"device=0"' --rm -v /path/to/offload:/offload -v $(pwd)/results:/results moe-infinity-bench python3 benchmarks/comparison/run_moe_infinity.py --offload-dir /offload --output-dir /results
```

### vLLM v0.18.1

- Docker image: `bench-vllm`, based on `vllm/vllm-openai:v0.18.1`
- Precision: FP8 quantization for large MoE models like Mixtral and Qwen3-30B, FP16 attempted first for DeepSeek-V2-Lite
- Build command:

```bash
docker build -t bench-vllm -f benchmarks/comparison/Dockerfile.vllm .
```

- Run standalone:

```bash
docker run --gpus '"device=0"' --rm -v $(pwd)/results:/results bench-vllm python3 /workspace/run_vllm.py --output-dir /results
```

### llama.cpp b8640

- Docker image: `bench-llamacpp`, based on `ghcr.io/ggml-org/llama.cpp:server-cuda`
- Precision: Q4_K_M GGUF quantization. Models are downloaded automatically from HuggingFace.
- GPU offloading: `-ngl 99`, which offloads as many layers as fit in 24GB
- Build command:

```bash
docker build -t bench-llamacpp -f benchmarks/comparison/Dockerfile.llamacpp .
```

- Run standalone:

```bash
docker run --gpus '"device=0"' --rm -v $(pwd)/results:/results bench-llamacpp python3 /workspace/run_llamacpp.py --output-dir /results
```

## Understanding Results

- **Per-token-latency**: decode phase only, it excludes prefill and TTFT. Lower is better. See the [Benchmarking Guide](benchmarking.md) for the exact definition.
- **`X`**: the model cannot run on that framework with a 24GB GPU, usually because of OOM or unsupported architecture
- **`—`**: not measured yet, run `run_all.sh` to populate the missing result
- **Precision differences**: MoE-Infinity runs FP16 with expert offloading, which preserves full model quality. vLLM uses FP8 quantization to fit larger MoE models into memory. llama.cpp uses Q4_K_M GGUF quantization. Those are real deployment trade-offs, MoE-Infinity keeps full precision while the other two reduce precision to fit.
- **Phase B update**: after the benchmark finishes, update the README by re-running `python3 benchmarks/comparison/aggregate_results.py --results-dir benchmarks/comparison/results/ --format markdown`, then copy the generated table into `README.md`

## Troubleshooting

- **CUDA OOM in MoE-Infinity**: lower `--device-memory-ratio`, start with `0.5`. Make sure `--offload-dir` is on a fast SSD.
- **CUDA OOM in llama.cpp**: reduce `--ngl`, try `24` or fewer layers
- **Docker GPU access fails**: verify NVIDIA Container Toolkit with `docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi`
- **Slow model downloads**: pre-download to `~/.cache/huggingface`, containers mount that directory automatically
- **Thermal throttling**: keep GPU temperature below 85°C. The benchmark runner already inserts a 60 second cooldown between frameworks. Check live telemetry with `nvidia-smi dmon`.
- **HuggingFace authentication**: set `HF_TOKEN=hf_xxx` before you start the run

## Extending the Benchmark

**Add a new model**:

1. Add it to `MODEL_CONFIGS` in `benchmarks/comparison/common.py`
2. Add the GGUF mapping to `GGUF_MODELS` in `benchmarks/comparison/run_llamacpp.py`
3. Update the `MODELS` list in `benchmarks/comparison/aggregate_results.py`

**Add a new framework**:

1. Create `benchmarks/comparison/run_FRAMEWORK.py`, following the pattern used by the current runners
2. Create `benchmarks/comparison/Dockerfile.FRAMEWORK`, wrapping the official image
3. Add the framework step to `benchmarks/comparison/run_all.sh`
4. Add the framework name to `FRAMEWORKS` in `benchmarks/comparison/aggregate_results.py`
