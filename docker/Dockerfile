# MoE-Infinity Build & Test Dockerfile
#
# Reproduces the full build of the refactored core/aio/ I/O layer with:
#   - Phase 0: Bug fixes (atomic bools, condition variables, round-robin)
#   - Phase 1: Multi-threaded parallel reads
#   - Phase 2: Pre-allocated pinned memory pool
#   - Phase 3: Pipelined disk-to-GPU transfer
#   - Phase 4: Partitioned storage files
#
# Usage:
#   docker build -t moe-infinity-build -f docker/Dockerfile .
#   docker run --gpus all moe-infinity-build
#
# To enter an interactive shell:
#   docker run --gpus all -it moe-infinity-build bash

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

LABEL maintainer="EfficientMoE Team"
LABEL description="MoE-Infinity build environment for refactored AIO layer"

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/MoE-Infinity

# Copy source code
COPY . .

# Install Python build dependencies first (torch is already in the base image)
RUN pip install --no-cache-dir \
    setuptools==75.3.2 \
    wheel \
    ninja \
    py-cpuinfo

# ---------------------------------------------------------------
# Build the prefetch extension (core AIO + memory + model + prefetch)
#
# This is a targeted build of the prefetch_op shared library which
# contains all our refactored code. We use a helper script instead
# of the full `BUILD_OPS=1 pip install -e .` because setup.py
# unconditionally compiles fused_glu_cuda.cu and expert_gemm.cu
# which have pre-existing issues on the xly/code-clean branch.
# ---------------------------------------------------------------
RUN python docker/build_prefetch.py

# Verify the shared library was produced
RUN ls -lh moe_infinity/ops/prefetch/prefetch_op*.so

# ---------------------------------------------------------------
# Run build verification tests
# ---------------------------------------------------------------
RUN python docker/verify_build.py

# Default command: print build info
CMD ["python", "-c", "\
import torch, sys, os;\
print('=== MoE-Infinity Build Info ===');\
print(f'Python:  {sys.version}');\
print(f'PyTorch: {torch.__version__}');\
print(f'CUDA:    {torch.version.cuda}');\
print(f'GPU:     {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}');\
so = [f for f in os.listdir('moe_infinity/ops/prefetch') if f.endswith('.so')];\
print(f'Prefetch extension: {so}');\
print('Build: OK');\
"]
