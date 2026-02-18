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
    cmake \
    ninja-build \
    git \
    uuid-dev \
    && rm -rf /var/lib/apt/lists/*

# Enable ccache for C++/CUDA builds
ENV CCACHE_DIR=/ccache
ENV CCACHE_MAXSIZE=10G
ENV PATH=/usr/lib/ccache:${PATH}

# Create ccache directory for compiler caching (early for better layer reuse)
RUN mkdir -p ${CCACHE_DIR}

# ---------------------------------------------------------------
# Install CUTLASS (default to a 3.x release; override with build args)
# ---------------------------------------------------------------
ARG CUTLASS_REPO=https://github.com/NVIDIA/cutlass.git
ARG CUTLASS_VERSION=v3.9.2
ARG CUTLASS_NVCC_ARCHS=80
ENV CUTLASS_DIR=/root/cutlass

RUN git clone --branch ${CUTLASS_VERSION} --depth 1 ${CUTLASS_REPO} ${CUTLASS_DIR}
RUN --mount=type=cache,target=/ccache \
    cmake -B ${CUTLASS_DIR}/build -S ${CUTLASS_DIR} \
    -DCUTLASS_ENABLE_LIBRARY=ON \
    -DCUTLASS_NVCC_ARCHS=${CUTLASS_NVCC_ARCHS} \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    && cmake --build ${CUTLASS_DIR}/build --target cutlass_library -j$(nproc)

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
# Build the torch extensions (inplace, no package install)
RUN --mount=type=cache,target=/ccache \
    CUTLASS_DIR=${CUTLASS_DIR} python setup.py build_ext --inplace

# Copy .so files to the expected location
RUN cp build/lib.linux-x86_64-cpython-311/moe_infinity/_*.so moe_infinity/ 2>/dev/null || true

# Verify the shared libraries were produced
RUN ls -lh moe_infinity/*.so

# ---------------------------------------------------------------
# Run build verification tests
# ---------------------------------------------------------------
RUN python docker/verify_build.py

# ---------------------------------------------------------------
# Build C++ test extension (test_io_module)
# ---------------------------------------------------------------
RUN cmake -B build/cmake_main -S . \
    -DCUTLASS_DIR=${CUTLASS_DIR} \
    -DCMAKE_CUDA_ARCHITECTURES=${CUTLASS_NVCC_ARCHS} \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/workspace/MoE-Infinity \
    -GNinja
RUN --mount=type=cache,target=/ccache \
    cmake --build build/cmake_main --target test_io_module -j$(nproc)
RUN cmake --install build/cmake_main --component TestRuntime

# Verify the test module was produced
RUN ls -lh tests/docker/test_io_module*.so

# Install test dependencies
RUN pip install --no-cache-dir pytest

# ---------------------------------------------------------------
# Run Tier 1 tests (build-time, no CUDA)
# Threading, file I/O, and configuration tests that don't need a GPU.
# ---------------------------------------------------------------
RUN python tests/docker/run_tests.py

# ---------------------------------------------------------------
# Default: run full test suite (Tier 2 needs --gpus all)
# ---------------------------------------------------------------
CMD ["python", "tests/docker/run_tests.py"]
