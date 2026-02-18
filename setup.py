# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import io
import os
import sys

from setuptools import find_packages, setup

torch_available = True
try:
    import torch  # noqa: F401
except ImportError:
    torch_available = False
    print(
        "[WARNING] Unable to import torch, pre-compiling ops will be disabled. "
        "Please visit https://pytorch.org/ to see how to properly install torch on your system."
    )

ROOT_DIR = os.path.dirname(__file__)

sys.path.insert(0, ROOT_DIR)

from torch.utils import cpp_extension

RED_START = "\033[31m"
RED_END = "\033[0m"
ERROR = f"{RED_START} [ERROR] {RED_END}"


def fetch_requirements(path):
    with open(path, "r") as fd:
        return [r.strip() for r in fd.readlines()]


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def abort(msg):
    print(f"{ERROR} {msg}")
    assert False, msg


def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        return io.open(get_path("README.md"), "r", encoding="utf-8").read()
    else:
        return ""


install_requires = fetch_requirements("requirements.txt")

# Get CUTLASS_DIR from environment or default to ~/cutlass
CUTLASS_DIR = os.path.expanduser(os.environ.get("CUTLASS_DIR", "~/cutlass"))

# Common include paths
COMMON_INCLUDE_PATHS = [
    get_path("core"),
    get_path("extensions"),
    os.path.join(CUTLASS_DIR, "include"),
    os.path.join(CUTLASS_DIR, "tools/util/include"),
]

# Common compile args
COMMON_NVCC_ARGS = [
    "-O3",
    "--use_fast_math",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
]

COMMON_CXX_ARGS = [
    "-O3",
    "-Wall",
    "-Wno-reorder",
    "-fPIC",
    "-fopenmp",
]

# _store extension: IO/checkpoint and prefetch functionality
# Includes AIO, prefetch handle, tensor index, memory pools, model topology
_STORE_SOURCES = [
    # utils
    "core/utils/logger.cpp",
    "core/utils/cuda_utils.cpp",
    # model
    "core/model/model_topology.cpp",
    "core/model/moe.cpp",
    # prefetch
    "core/prefetch/archer_prefetch_handle.cpp",
    "core/prefetch/task_scheduler.cpp",
    "core/prefetch/task_thread.cpp",
    # memory
    "core/memory/caching_allocator.cpp",
    "core/memory/memory_pool.cpp",
    "core/memory/pinned_memory_pool.cpp",
    "core/memory/stream_pool.cpp",
    "core/memory/host_caching_allocator.cpp",
    "core/memory/device_caching_allocator.cpp",
    # parallel
    "core/parallel/expert_dispatcher.cpp",
    "core/parallel/expert_module.cpp",
    # aio
    "core/aio/archer_aio_thread.cpp",
    "core/aio/archer_prio_aio_handle.cpp",
    "core/aio/archer_aio_utils.cpp",
    "core/aio/archer_aio_threadpool.cpp",
    "core/aio/archer_tensor_handle.cpp",
    "core/aio/archer_tensor_index.cpp",
    # base
    "core/base/thread.cc",
    "core/base/exception.cc",
    "core/base/date.cc",
    "core/base/process_info.cc",
    "core/base/logging.cc",
    "core/base/log_file.cc",
    "core/base/timestamp.cc",
    "core/base/file_util.cc",
    "core/base/countdown_latch.cc",
    "core/base/timezone.cc",
    "core/base/log_stream.cc",
    "core/base/thread_pool.cc",
    # CUDA kernels for store
    "core/model/fused_mlp.cu",
    "extensions/kernel/activation_kernels.cu",
    "extensions/kernel/topk_softmax_kernels.cu",
    # Python binding
    "core/python/py_archer_prefetch.cpp",
]

_STORE_EXTRA_LINK_ARGS = [
    "-luuid",
    "-lcublas",
    "-lcudart",
    "-lcuda",
    "-lpthread",
]

# _engine extension: compute kernels (fused_glu + expert_gemm)
_ENGINE_SOURCES = [
    "core/python/fused_glu_cuda.cu",
]

# Note: _engine needs CUTLASS for fused_glu_cuda.cu

ext_modules = []

if torch_available:
    _cuda_arch_flags = ["-gencode=arch=compute_80,code=sm_80"]
    if os.environ.get("MOE_ENABLE_SM90", "1") == "1":
        _cuda_arch_flags.append("-gencode=arch=compute_90,code=sm_90")

    # _store extension: IO and prefetch
    ext_modules.append(
        cpp_extension.CUDAExtension(
            name="moe_infinity._store",
            sources=_STORE_SOURCES,
            include_dirs=COMMON_INCLUDE_PATHS,
            extra_compile_args={
                "cxx": COMMON_CXX_ARGS,
                "nvcc": COMMON_NVCC_ARGS + _cuda_arch_flags,
            },
            extra_link_args=_STORE_EXTRA_LINK_ARGS,
        )
    )

    # _engine extension: compute kernels (needs CUTLASS)
    ext_modules.append(
        cpp_extension.CUDAExtension(
            name="moe_infinity._engine",
            sources=_ENGINE_SOURCES,
            include_dirs=COMMON_INCLUDE_PATHS,
            extra_compile_args={
                "nvcc": COMMON_NVCC_ARGS
                + _cuda_arch_flags
                + ["-DBF16_AVAILABLE"],
            },
        )
    )

cmdclass = {
    "build_ext": cpp_extension.BuildExtension.with_options(use_ninja=True)
}

print(f"find_packages: {find_packages()}")

# install all files in the package, rather than just the egg
setup(
    name="moe_infinity",
    version=os.getenv("MOEINF_VERSION", "0.0.1"),
    packages=find_packages(exclude=["extensions", "extensions.*"]),
    include_package_data=True,
    install_requires=install_requires,
    author="EfficientMoE Team",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/EfficientMoE/MoE-Infinity",
    project_urls={"Homepage": "https://github.com/EfficientMoE/MoE-Infinity"},
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="Apache License 2.0",
    python_requires=">=3.8",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
