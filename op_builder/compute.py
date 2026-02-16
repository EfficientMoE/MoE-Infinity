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

from .builder import OpBuilder, CUDAOpBuilder


class ComputeBuilder(CUDAOpBuilder):
    BUILD_VAR = "MOE_BUILD_COMPUTE"
    NAME = "compute"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f"moe_infinity.ops.prefetch.{self.NAME}_op"

    def sources(self):
        return [
            "core/python/fused_glu_cuda.cu",
        ]

    def include_paths(self):
        return ["core"]

    def cxx_args(self):
        # -O0 for improved debugging, since performance is bound by I/O
        CPU_ARCH = self.cpu_arch()
        SIMD_WIDTH = self.simd_width()

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
            "-I/usr/local/cuda/include",
            "-L/usr/local/cuda/lib64",
            "-lcuda",
            "-lcudart",
            "-lcublas",
            "-lpthread",
        ]

    def extra_ldflags(self):
        return []

    def is_compatible(self, verbose=True):
        return super().is_compatible(verbose)
