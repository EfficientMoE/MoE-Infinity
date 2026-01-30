# CLAUDE.md - AI Assistant Guidelines for MoE-Infinity

## Project Overview

**MoE-Infinity** is a cost-effective, fast, and easy-to-use library for Mixture-of-Experts (MoE) inference on resource-constrained GPUs. It enables running large MoE models on personal machines by offloading experts to host memory with novel optimization techniques.

**Repository**: https://github.com/EfficientMoE/MoE-Infinity
**License**: Apache License 2.0

### Key Features
- Expert offloading from GPU to host memory
- Activation-aware expert prefetching and caching
- FlashAttention integration support
- Multi-GPU environment support
- HuggingFace-compatible API
- OpenAI-compatible REST API server

---

## Directory Structure

```
MoE-Infinity/
├── moe_infinity/                 # Main Python package
│   ├── __init__.py              # Exports: MoE, OffloadEngine
│   ├── entrypoints/             # User-facing APIs
│   │   ├── big_modeling.py      # MoE class - main inference interface
│   │   └── openai/              # OpenAI-compatible server
│   │       ├── api_server.py    # FastAPI server implementation
│   │       └── protocol.py      # OpenAI protocol definitions
│   ├── runtime/                 # Core inference engine
│   │   ├── model_offload.py     # OffloadEngine class
│   │   ├── hooks.py             # Forward/backward hooks
│   │   └── state_dict.py        # State management
│   ├── models/                  # Model-specific implementations
│   │   ├── deepseek.py          # DeepseekMoEBlock
│   │   ├── mixtral.py           # SyncMixtralSparseMoeBlock
│   │   ├── nllb_moe.py          # SyncNllbMoeSparseMLP
│   │   ├── switch_transformers.py
│   │   ├── arctic.py            # ArcticConfig, SyncArcticMoeBlock
│   │   ├── grok.py              # SyncGrokMoeBlock
│   │   ├── modeling_deepseek/   # Full DeepSeek V2 implementation
│   │   ├── modeling_deepseek_v3/# Full DeepSeek V3 implementation
│   │   ├── modeling_arctic/     # Arctic model implementation
│   │   └── modeling_grok/       # Grok model implementation
│   ├── memory/                  # Expert memory management
│   │   ├── expert_cache.py      # Sparsity-aware expert caching
│   │   ├── expert_tracer.py     # Activation tracing
│   │   ├── expert_prefetcher.py # Expert prefetching logic
│   │   └── expert_predictor.py  # Expert prediction
│   ├── ops/                     # Custom CUDA operations
│   │   └── prefetch/            # Precompiled prefetch ops
│   ├── distributed/             # Multi-GPU support
│   │   ├── devicemap_manager.py # GPU device mapping
│   │   └── expert_executor.py   # Distributed expert execution
│   ├── common/                  # Shared constants
│   │   └── constants.py         # MODEL_MAPPING_NAMES, MODEL_MAPPING_TYPES
│   └── utils/                   # Utilities
│       ├── config.py            # ArcherConfig dataclass
│       └── checkpoints.py       # Checkpoint utilities
├── core/                        # C++ core implementation
│   ├── aio/                     # Async I/O threading
│   ├── base/                    # Base utilities (logging, threading)
│   ├── memory/                  # Memory allocators
│   ├── parallel/                # Expert dispatching
│   ├── prefetch/                # Prefetching engine
│   ├── python/                  # Python bindings
│   └── utils/                   # C++ utilities
├── op_builder/                  # C++ extension builder (DeepSpeed-based)
│   ├── all_ops.py               # Auto-discover ops builders
│   ├── builder.py               # OpBuilder base class
│   └── prefetch.py              # Prefetch operation builder
├── examples/                    # Example scripts
├── tests/                       # Test files
├── .github/workflows/           # CI/CD pipelines
└── pyproject.toml               # Build system config
```

---

## Supported Models

| Architecture | Example Models | Mapping Type |
|---|---|---|
| Switch Transformers | `google/switch-large-128` | 0 |
| NLLB-MoE | `facebook/nllb-moe-54b` | 2 |
| Mixtral | `mistralai/Mixtral-8x7B-*` | 4 |
| Grok | `xai-org/Grok-1` | 4 |
| Arctic | `Snowflake/arctic` | 4 |
| DeepSeek V2 | `deepseek-ai/DeepSeek-V2-*` | 5 |
| DeepSeek V3 | `deepseek-ai/DeepSeek-V3` | 5 |

Model types are defined in `moe_infinity/common/constants.py`.

---

## Development Setup

### Prerequisites
- Python 3.8-3.11
- CUDA 12.1+ (for GPU support)
- PyTorch 2.1.1+

### Installation for Development
```bash
# Create conda environment
conda create -n moe-infinity python=3.9
conda activate moe-infinity

# Install in development mode
pip install -e .

# Install C++ stdlib (if using conda)
conda install -c conda-forge libstdcxx-ng=12

# Optional: Enable FlashAttention
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn
```

### Install Linting Tools
```bash
pip install -r requirements-lint.txt
```

---

## Build Commands

### Build Python Package (without C++ ops)
```bash
pip install -e .
```

### Build with C++ Extensions
```bash
BUILD_OPS=1 python -m build
```

### Build Wheel for Distribution
```bash
pip install build
BUILD_OPS=1 python -m build
```

---

## Code Quality

### Pre-commit Hooks
The project uses pre-commit hooks. Install and run:
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

### Formatting Tools
- **Python**: ruff (linting + formatting), isort (imports)
- **C++**: clang-format v18.1.4

### Run Formatters Manually
```bash
# Python - lint and fix
ruff check --fix .
ruff format .

# C++ formatting
clang-format -i core/**/*.{h,cc,cpp}
```

### Linting Configuration
- Line length: 80 characters
- Python style: Enforced by ruff (pyproject.toml)
- C++ style: Google style with modifications (.clang-format)
- Import sorting: Enforced by ruff with isort rules

---

## Testing

### OpenAI API Tests
Requires a running server:
```bash
# Terminal 1: Start server
python -m moe_infinity.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --offload-dir ./offload_dir

# Terminal 2: Run tests
python tests/test_oai_completions.py
python tests/test_oai_chat_completions.py
```

### C++ Unit Tests
Located in `tests/queues/`:
```bash
cd tests/queues
mkdir build && cd build
cmake ..
make
./test_lockfree_queue
./test_threadsafe_queue
```

---

## Running the Project

### Basic Inference
```python
from moe_infinity import MoE
from transformers import AutoTokenizer

checkpoint = "deepseek-ai/DeepSeek-V2-Lite-Chat"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

config = {
    "offload_path": "/path/to/offload/dir",  # Must be unique per model
    "device_memory_ratio": 0.75,
}

model = MoE(checkpoint, config)
input_ids = tokenizer("Hello", return_tensors="pt").input_ids.to("cuda:0")
output = model.generate(input_ids)
```

### Multi-GPU Inference
```bash
CUDA_VISIBLE_DEVICES=0,1 python script.py
```

### OpenAI-Compatible Server
```bash
python -m moe_infinity.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --offload-dir ./offload_dir
```

Query endpoints:
- `/v1/completions` - Text completions
- `/v1/chat/completions` - Chat completions

---

## Configuration Reference

The `ArcherConfig` dataclass (`moe_infinity/utils/config.py`) accepts:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `offload_path` | str | Required | Path for parameter storage (must be unique per model) |
| `device_memory_ratio` | float | 0.9 | GPU memory utilization (0-1) |
| `host_memory_ratio` | float | 0.9 | Host memory utilization (0-1) |
| `prefetch` | bool | False | Enable expert prefetching |
| `trace_capacity` | int | 1000 | Activation history size |
| `trace_path` | str | None | Optional trace file location |
| `num_threads` | int | 8 | Executor threads per GPU |

Configuration can be passed as a Python dict, JSON file path, or `ArcherConfig` object.

---

## CI/CD Pipelines

### Workflows (`.github/workflows/`)

1. **build-test.yml** - Build validation on PRs
   - Triggers on PRs to main/dev (skips doc/example/test changes)
   - Builds wheel with `BUILD_OPS=1`
   - Environment: Ubuntu 20.04 + CUDA 12.1.1

2. **pre-commit-format.yml** - Code formatting checks
   - Runs pre-commit hooks on all files
   - Triggers on all PRs and daily schedule

3. **publish-test.yml** - Test publishing to TestPyPI
   - Manual trigger only
   - Builds wheels for Python 3.8-3.11

4. **publish.yml** - Production release
   - Triggers on `v*` tags (e.g., `v1.0.0`)
   - Publishes to PyPI and GitHub Package Registry

---

## Release Process

### Automated Release (Recommended)
1. Update version in `setup.py`
2. Commit: `git commit -m "Update version for X.Y.Z release"`
3. Tag and push:
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```
4. CI/CD automatically builds and publishes

### Manual Release
```bash
pip install build twine
BUILD_OPS=1 python -m build
twine upload dist/*
```

---

## Code Architecture Patterns

### Key Patterns
1. **Hook-based Integration**: PyTorch forward/backward hooks intercept expert execution
2. **Expert Offloading**: GPU -> Host memory -> GPU with prediction-driven prefetching
3. **Async I/O**: Dedicated C++ thread pools for non-blocking I/O
4. **Model Abstraction**: Pluggable model blocks for different MoE architectures

### Public API
```python
from moe_infinity import MoE, OffloadEngine
```

- `MoE`: High-level inference interface (HuggingFace compatible)
- `OffloadEngine`: Low-level offloading engine

---

## Important Conventions

### File Headers
All Python files should include:
```python
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team
```

### Offload Path
The `offload_path` configuration **must be unique for each MoE model**. Reusing the same path for different models will cause unexpected behavior.

### Version Management
- Version is defined via `MOEINF_VERSION` environment variable
- Falls back to `0.0.1` if not set
- Located in `setup.py`

### Dependencies
- Core deps: `requirements.txt`
- Lint deps: `requirements-lint.txt`
- Key constraints:
  - `transformers>=4.37.1, <4.47`
  - `torch>=2.1.1`
  - `pydantic==1.10.12`

---

## Common Development Tasks

### Adding a New Model
1. Create model implementation in `moe_infinity/models/`
2. Add to `MODEL_MAPPING_NAMES` and `MODEL_MAPPING_TYPES` in `moe_infinity/common/constants.py`
3. Implement the corresponding MoE block class

### Modifying C++ Core
1. Edit files in `core/` directory
2. Rebuild with `BUILD_OPS=1 pip install -e .`
3. Run C++ tests in `tests/queues/`

### Adding OpenAI Endpoints
1. Modify `moe_infinity/entrypoints/openai/api_server.py`
2. Update protocol definitions in `protocol.py`
3. Add tests in `tests/`

---

## Troubleshooting

### Out of Memory (OOM)
- Reduce `device_memory_ratio` in config (e.g., 0.5-0.75)
- Ensure only necessary GPUs are visible via `CUDA_VISIBLE_DEVICES`

### Build Failures
- Ensure CUDA toolkit is installed and matches PyTorch version
- Install `ninja` for faster C++ compilation
- On conda, install `libstdcxx-ng=12`

### Model Loading Issues
- Verify the model is in `MODEL_MAPPING_NAMES`
- Check `offload_path` is writable and has sufficient space
- Ensure `offload_path` is unique for the model
