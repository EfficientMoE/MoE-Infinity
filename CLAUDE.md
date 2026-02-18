# CLAUDE.md - MoE-Infinity Development Guide

Call me Bessus in every conversation.

## Project Overview

MoE-Infinity is a cost-effective, fast, and easy-to-use library for Mixture-of-Experts (MoE) inference. It enables memory-constrained GPUs to serve large MoE models by offloading experts to host memory, with novel techniques including:
- Expert activation tracing
- Activation-aware expert prefetching
- Activation-aware expert caching

Note: The open-sourced MoE-Infinity prioritizes HuggingFace usability over
extreme performance, and distributed inference is currently not supported.

The project is a hybrid Python + C++ (CUDA) codebase with a Python package (`moe_infinity/`) and C++ core (`core/`).

## Key Commands

### Installation & Building

```bash
# (Recommended) Create and activate a virtual environment
conda create -n moe-infinity python=3.9
conda activate moe-infinity

# Install stable release from PyPI
pip install moe-infinity

# Install nightly release from TestPyPI
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ moe-infinity

# Install from source
git clone https://github.com/EfficientMoE/MoE-Infinity.git
cd MoE-Infinity
pip install -e .
conda install -c conda-forge libstdcxx-ng=12

# Build with custom ops (requires PyTorch)
BUILD_OPS=1 MAX_JOBS=$(nproc) pip install -e .

# (Optional) Enable FlashAttention for faster inference
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn
```

### Running Examples

```bash
# Basic inference example
CUDA_VISIBLE_DEVICES=0 python examples/interface_example.py --model_name_or_path "deepseek-ai/DeepSeek-V2-Lite-Chat" --offload_dir <path>

# Start OpenAI-compatible server
python -m moe_infinity.entrypoints.openai.api_server --model deepseek-ai/DeepSeek-V2-Lite-Chat --offload-dir ./offload_dir

# Query /v1/completions (required fields only)
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "prompt": "Hello, my name is"
    }'

# Query /v1/chat/completions (required fields only)
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a joke"}
        ]
    }'

# Run OpenAI client smoke tests
pip install openai
python tests/python/integration/test_oai_completions.py
python tests/python/integration/test_oai_chat_completions.py
```

### Development

```bash
# Run pre-commit hooks
pre-commit install
pre-commit run --all-files

# Lint and format (via pre-commit)
ruff check --fix .
ruff format .
clang-format --style=file --i core/**/*.cpp core/**/*.h
```

## Architecture

### Package Structure

```
moe_infinity/
├── __init__.py          # Main package entry, exports MoE class
├── common/              # Constants
├── distributed/         # Device map, expert executor, prefetcher
├── entrypoints/         # API server (OpenAI-compatible), big_modeling
├── kernel/              # Router kernels
├── memory/              # Expert cache/entry/predictor/prefetcher/tracer
├── models/              # Arctic, DeepSeek v2/v3, Grok, Mixtral, Qwen, NLLB
│   └── modeling_*       # HF-compatible model implementations
├── ops/                 # Custom CUDA operators + build helpers
│   ├── core/            # Packaged C++ core sources
│   ├── op_builder/      # Legacy build helpers (deprecated)
│   └── prefetch/        # Legacy prefetch ops (deprecated)
├── runtime/             # Compile, hooks, model offload, state dict
└── utils/               # Arguments, checkpoints, config, HF config
```

### Core C++ Components (`core/`)

- `aio/` - Async I/O handles and thread pools
- `base/` - Threading, logging, file utilities
- `common/` - Status, types, context
- `engine/` - Event loop (libevent-based)
- `kernel/` - CUDA kernels (MLP, topk, softmax; symlink to `extensions/kernel/`)
- `memory/` - Device/host caching allocators, memory pool, KV cache
- `model/` - Model topology
- `parallel/` - Expert dispatcher, expert module
- `prefetch/` - Task scheduler, prefetch handles
- `python/` - Pybind/torch bindings
- `utils/` - CUDA utilities, lock-free queues, logger

### Build System

- Uses `setuptools` with `torch.utils.cpp_extension.CUDAExtension`
- Custom ops built with Ninja
- Requires CUDA toolkit (sm_80+)

## Code Style

### Python

- Follows `ruff` configuration (see `pyproject.toml`)
- Line length: 80 characters
- Uses `isort` for imports

### C++

- Follows `.clang-format` configuration
- Uses Google-style naming where applicable

### Pre-commit Hooks

Required before committing:
- `ruff` - Linting (auto-fixes)
- `ruff-format` - Formatting
- `clang-format` - C++ formatting
- `codespell` - Spell checking

## Important Conventions

### Offload Path

- The `offload_path` config must be **unique for each MoE model**
- Reusing the same `offload_path` for different models causes unexpected behavior

### Device Memory

- Default `device_memory_ratio: 0.75` uses 75% of GPU memory for caching
- Adjust based on your GPU memory to avoid OOM errors

### Model Compatibility

- HuggingFace compatible - supports `AutoModel` loading
- Supports: DeepSeek-V2/V3, Switch Transformers, NLLB-MoE, Mixtral, Qwen,
  Arctic, Grok

## Common Issues

### Installation

- **No torch**: Pre-compiled ops disabled, install PyTorch first
- **Build failures**: Ensure CUDA toolkit installed, try `conda install -c conda-forge libstdcxx-ng=12`

### Runtime

- **OOM errors**: Lower `device_memory_ratio` or use larger GPU
- **Slow inference**: Enable FlashAttention (`FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn`)

## Testing

### Test Structure

```
tests/
├── cpp/
│   └── unittest/
│       ├── queues/
│       │   ├── CMakeLists.txt
│       │   ├── test_lockfree_queue.cpp   # LockFreeQueue correctness & stress
│       │   └── test_threadsafe_queue.cpp # ThreadSafeQueue correctness & stress
│       └── utils/
│           ├── CMakeLists.txt
│           ├── test_lfu_cache.cpp         # LFUCache eviction & frequency logic
│           └── test_simple_object_pool.cpp # SimpleObjectPool reuse & concurrency
├── cuda/
│   ├── CMakeLists.txt
│   ├── test_fused_mlp.cu
│   ├── test_fused_mlp_cutlass.cu  # BF16 CUTLASS vs Torch-native MLP benchmark
│   ├── test_topk_softmax.cu
│   └── ...
└── python/
    ├── benchmark/
    │   └── test_fused_glu_cutlass.py
    └── integration/
        ├── test_oai_chat_completions.py
        └── test_oai_completions.py

tests/docker/                              # I/O integration tests (pybind11 + pytest)
├── conftest.py                            # pytest fixtures (workspace_tmpdir)
├── run_tests.py                           # Orchestrates Tier 1 / Tier 2 runs
├── test_io_integration.py                 # pytest test classes
└── test_io_module.cpp                     # C++ pybind11 test harness
```

### Building & Running C++ Unit Tests

```bash
# Build and run queue tests
cd tests/cpp/unittest/queues
cmake -B build && cmake --build build -j$(nproc)
ctest --test-dir build -V

# Build and run utils tests (LFUCache, SimpleObjectPool)
cd tests/cpp/unittest/utils
cmake -B build && cmake --build build -j$(nproc)
ctest --test-dir build -V
```

### Building & Running CUDA Benchmark Tests

```bash
# Build a specific CUDA benchmark (requires CUTLASS at ~/cutlass and a GPU)
cd tests/cuda
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test_fused_mlp_cutlass -j$(nproc)
./build/test_fused_mlp_cutlass

# Build all CUDA tests
cmake --build build -j$(nproc)
```

> **CMakeLists.txt pattern:** tests that need `${KERNEL_SRC}` (i.e. link
> `extensions/kernel/*.cu`) must be added to `TORCH_SRC_LIST` **and** listed
> in the `IF(SRC_NAME STREQUAL "..." OR ...)` guard in the `FOREACH` loop.

### Docker Build & Integration Tests

```bash
# Build the Docker image (runs Tier 1 I/O tests at build time)
DOCKER_BUILDKIT=1 docker build -t moe-infinity-test -f docker/Dockerfile .

# Run full test suite (Tier 2 requires a GPU)
docker run --gpus all moe-infinity-test

# Interactive shell inside the container
docker run --gpus all -it moe-infinity-test bash

# Run only Tier 1 tests (no GPU needed)
docker run moe-infinity-test python tests/docker/run_tests.py
```

The Docker image is based on `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`.
Build steps inside the image:
1. Build with `pip install -e .` (builds `moe_infinity._store` and `moe_infinity._engine` extensions)
2. Run `python -c "import moe_infinity._store"` to verify the extension loads
3. Build `test_io_module.so` via CMake (see `extensions/kernels/test_io/CMakeLists.txt`)
4. Run `tests/docker/run_tests.py` — Tier 1 tests (threading, file I/O, no CUDA) at image build time; Tier 2 (pinned memory, tensor roundtrip) at `docker run` time with `--gpus all`

## Release Process

See `RELEASE.md` for the full release checklist. In short:
1. Update the version in `setup.py`.
2. Commit the change.
3. Tag the release (e.g., `v1.0.0`) and push the tag to GitHub.
4. The GitHub Actions workflow publishes to PyPI.

## Task Management (Task Master MCP)

This project can use [Task Master](https://github.com/eyaltoledano/claude-task-master) for AI-powered task management.

### Installation

Add the MCP server to your configuration at `~/.cursor/mcp.json` (Cursor) or `~/.codeium/windsurf/mcp_config.json` (Windsurf):

```json
{
  "mcpServers": {
    "task-master-ai": {
      "command": "npx",
      "args": ["-y", "task-master-ai"],
      "env": {
        "TASK_MASTER_TOOLS": "standard",
        "ANTHROPIC_API_KEY": "YOUR_ANTHROPIC_API_KEY"
      }
    }
  }
}
```

For Claude Code CLI:
```bash
claude mcp add taskmaster-ai -- npx -y task-master-ai
```

### Required API Keys

At least one API key is required (add to your `.env` or MCP config):
- `ANTHROPIC_API_KEY` - Claude API
- `OPENAI_API_KEY` - OpenAI
- `GOOGLE_API_KEY` - Google Gemini
- Or use Claude Code (`claude-code/sonnet`) - no API key required

### Tool Modes

Configure how many tools load (~token usage):
- `core` (~5,000 tokens) - Essential: get_tasks, next_task, get_task, set_task_status, update_subtask, parse_prd, expand_task
- `standard` (~10,000 tokens) - Core + project management tools
- `all` (~21,000 tokens) - All 36 tools (default)

### Usage

After installation, initialize Task Master in the project:
```
Initialize taskmaster-ai in my project
```

Then use commands like:
- `Parse my PRD at .taskmaster/docs/prd.txt`
- `What's the next task I should work on?`
- `Can you help me implement task 3?`
- `Can you show me tasks 1, 3, and 5?`

## Citation

If you use MoE-Infinity for research, cite:
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
