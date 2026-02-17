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
#   docker build -t moe-infinity-test -f docker/Dockerfile .
#   docker run --gpus all moe-infinity-test
#
# To enter an interactive shell:
#   docker run --gpus all -it moe-infinity-test bash

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

LABEL maintainer="EfficientMoE Team"
LABEL description="MoE-Infinity build environment for refactored AIO layer"

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ccache \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/*

# Enable ccache for C++/CUDA builds
ENV CCACHE_DIR=/ccache
ENV CCACHE_MAXSIZE=10G
ENV PATH=/usr/lib/ccache:${PATH}

WORKDIR /workspace/MoE-Infinity

# Create ccache directory for compiler caching
RUN mkdir -p ${CCACHE_DIR}

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

# ---------------------------------------------------------------
# Build C++ test extension (test_io_module)
# ---------------------------------------------------------------
RUN python docker/tests/build_test_module.py

# Verify the test module was produced
RUN ls -lh docker/tests/test_io_module*.so

# Install test dependencies
RUN pip install --no-cache-dir pytest

# ---------------------------------------------------------------
# Run Tier 1 tests (build-time, no CUDA)
# Threading, file I/O, and configuration tests that don't need a GPU.
# ---------------------------------------------------------------
RUN python docker/tests/run_tests.py

# ---------------------------------------------------------------
# Default: run full test suite (Tier 2 needs --gpus all)
# ---------------------------------------------------------------
CMD ["python", "docker/tests/run_tests.py"]
