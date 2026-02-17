#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# op_builder/async_io.py
#
# Part of the DeepSpeed Project, under the Apache-2.0 License.
# See https://github.com/microsoft/DeepSpeed/blob/master/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0

# MoE-Infinity: replaced AsyncIOBuilder with PrefetchBuilder

import glob
import os

from .builder import CUDAOpBuilder, OpBuilder


class PrefetchBuilder(CUDAOpBuilder):
    BUILD_VAR = "MOE_BUILD_PREFETCH"
    NAME = "prefetch"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f"moe_infinity.ops.prefetch.{self.NAME}_op"

    def sources(self):
        return [
            "core/utils/logger.cpp",
            "core/utils/cuda_utils.cpp",
            "core/model/model_topology.cpp",
            "core/model/fused_mlp.cu",
            "core/model/moe.cpp",
            "core/kernel/activation_kernels.cu",
            "core/kernel/topk_softmax_kernels.cu",
            "core/prefetch/archer_prefetch_handle.cpp",
            "core/prefetch/task_scheduler.cpp",
            "core/prefetch/task_thread.cpp",
            "core/memory/caching_allocator.cpp",
            "core/memory/memory_pool.cpp",
            "core/memory/pinned_memory_pool.cpp",
            "core/memory/stream_pool.cpp",
            "core/memory/host_caching_allocator.cpp",
            "core/memory/device_caching_allocator.cpp",
            "core/parallel/expert_dispatcher.cpp",
            "core/parallel/expert_module.cpp",
            "core/aio/archer_aio_thread.cpp",
            "core/aio/archer_prio_aio_handle.cpp",
            "core/aio/archer_aio_utils.cpp",
            "core/aio/archer_aio_threadpool.cpp",
            "core/aio/archer_tensor_handle.cpp",
            "core/aio/archer_tensor_index.cpp",
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
            "core/python/py_archer_prefetch.cpp",
        ]

    def cutlass_dir(self):
        CUTLASS_DIR = os.path.expanduser("~") + "/cutlass"
        if not os.path.exists(CUTLASS_DIR):
            raise FileNotFoundError(
                f"Cutlass directory not found: {CUTLASS_DIR}"
            )
        else:
            print(f"Using Cutlass directory: {CUTLASS_DIR}")
        return CUTLASS_DIR

    def include_paths(self):
        CUTLASS_DIR = self.cutlass_dir()

        return [
            "core",
            f"{CUTLASS_DIR}/include",
            f"{CUTLASS_DIR}/tools/util/include",
        ]

    def cxx_args(self):
        # -O0 for improved debugging, since performance is bound by I/O
        CPU_ARCH = self.cpu_arch()
        SIMD_WIDTH = self.simd_width()
        CUTLASS_DIR = self.cutlass_dir()

        return [
            "-g",
            "-Wall",
            "-O3",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-Wno-reorder",
            CPU_ARCH,
            "-fopenmp",
            SIMD_WIDTH,
            "-lpthread",
            "-L/usr/local/cuda/lib64",
            f"-L{CUTLASS_DIR}/build/tools/library",
            "-lcutlass",
        ]

    def extra_ldflags(self):
        return [
            "-luuid",
            "-lcublas",
            "-lcudart",
            "-lcuda",
        ]

    def is_compatible(self, verbose=True):
        return super().is_compatible(verbose)
