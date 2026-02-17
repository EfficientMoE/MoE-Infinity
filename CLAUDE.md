# CLAUDE.md - MoE-Infinity Development Guide

## Project Overview

MoE-Infinity is a cost-effective, fast, and easy-to-use library for Mixture-of-Experts (MoE) inference. It enables memory-constrained GPUs to serve large MoE models by offloading experts to host memory, with novel techniques including:
- Expert activation tracing
- Activation-aware expert prefetching
- Activation-aware expert caching

The project is a hybrid Python + C++ (CUDA) codebase with a Python package (`moe_infinity/`) and C++ core (`core/`).

## Key Commands

### Installation & Building

```bash
# Install from PyPI
pip install moe-infinity

# Install from source
git clone https://github.com/EfficientMoE/MoE-Infinity.git
cd MoE-Infinity
pip install -e .
conda install -c conda-forge libstdcxx-ng=12

# Build with custom ops (requires PyTorch)
BUILD_OPS=1 pip install -e .
```

### Running Examples

```bash
# Basic inference example
CUDA_VISIBLE_DEVICES=0 python examples/interface_example.py --model_name_or_path "deepseek-ai/DeepSeek-V2-Lite-Chat" --offload_dir <path>

# Start OpenAI-compatible server
python -m moe_infinity.entrypoints.openai.api_server --model deepseek-ai/DeepSeek-V2-Lite-Chat --offload-dir ./offload_dir
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
├── entrypoints/         # API server (OpenAI-compatible)
├── memory/              # Expert cache, predictor, tracer, priority scores
├── models/              # Model implementations (DeepSeek, Mixtral, etc.)
├── ops/                 # Custom CUDA operators
│   ├── core/           # Core operations
│   └── prefetch/       # Prefetch operations
├── runtime/            # Hooks, model offload, state dict
└── utils/              # Arguments, checkpoints, config, HF config
```

### Core C++ Components (`core/`)

- `base/` - Threading, logging, file utilities
- `engine/` - Event loop (libevent-based)
- `memory/` - Device/host caching allocators, memory pool, KV cache
- `model/` - Model topology
- `parallel/` - Expert dispatcher, expert module
- `prefetch/` - Task scheduler, prefetch handles
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
- Supports: DeepSeek-V2, Switch Transformers, NLLB-MoE, Mixtral

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
└── python/
    ├── benchmark/
    │   └── test_fused_glu_cutlass.py
    └── integration/
        ├── test_oai_chat_completions.py
        └── test_oai_completions.py

docker/tests/                              # I/O integration tests (pybind11 + pytest)
├── build_test_module.py                   # Builds test_io_module.so
├── conftest.py                            # pytest fixtures (workspace_tmpdir)
├── run_tests.py                           # Orchestrates Tier 1 / Tier 2 runs
├── test_io_integration.py                 # pytest test classes
└── test_io_module.cpp                     # C++ pybind11 test harness
```

### Building & Running C++ Unit Tests

```bash
# Build and run queue tests
cd tests/cpp/unittest/queues
cmake -B build && cmake --build build
ctest --test-dir build -V

# Build and run utils tests (LFUCache, SimpleObjectPool)
cd tests/cpp/unittest/utils
cmake -B build && cmake --build build
ctest --test-dir build -V
```

### Docker Build & Integration Tests

```bash
# Build the Docker image (runs Tier 1 I/O tests at build time)
docker build -t moe-infinity-test -f docker/Dockerfile .

# Run full test suite (Tier 2 requires a GPU)
docker run --gpus all moe-infinity-test

# Interactive shell inside the container
docker run --gpus all -it moe-infinity-test bash

# Run only Tier 1 tests (no GPU needed)
docker run moe-infinity-test python docker/tests/run_tests.py
```

The Docker image is based on `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`.
Build steps inside the image:
1. Build `prefetch_op.so` via `docker/build_prefetch.py` (avoids the pre-existing `fused_glu_cuda.cu` build error)
2. Run `docker/verify_build.py` — smoke test the shared library
3. Build `test_io_module.so` via `docker/tests/build_test_module.py`
4. Run `docker/tests/run_tests.py` — Tier 1 tests (threading, file I/O, no CUDA) at image build time; Tier 2 (pinned memory, tensor roundtrip) at `docker run` time with `--gpus all`

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
  author = {Leyang Xue and Yao Fu and Zhan Lu and Luo Mai and Mahesh Marina},
  title = {MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache},
  archivePrefix = {arXiv},
  eprint = {2401.14361},
  year = {2024}
}
```
