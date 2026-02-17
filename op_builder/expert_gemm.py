#!/usr/bin/env python3
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# MoE-Infinity: Expert GEMM Op Builder

import os

from .builder import CUDAOpBuilder


class ExpertGemmBuilder(CUDAOpBuilder):
    BUILD_OP = os.environ.get("BUILD_EXPERT_GEMM", "1") == "1"

    def __init__(self):
        super().__init__("expert_gemm")

    def absolute_name(self):
        return "moe_infinity.ops.expert_gemm"

    def sources(self):
        return ["core/python/expert_gemm.cu"]

    def include_paths(self):
        return []

    def libraries_args(self):
        return []

    def nvcc_args(self):
        args = super().nvcc_args()

        # Enable BF16
        args.append("-DBF16_AVAILABLE")

        # Ampere-specific optimizations
        args.append("--use_fast_math")

        # Disable specific warnings
        args.extend(["-Wno-reorder", "-Wno-deprecated-declarations"])

        return args

    def is_compatible(self):
        # Check for CUDA 11.0+ (required for BF16 tensor cores)
        cuda_major, _ = self.installed_cuda_version()
        if cuda_major < 11:
            self.warning(
                "CUDA 11.0+ required for BF16 tensor cores. Disabling expert_gemm."
            )
            return False

        return True

    def cxx_args(self):
        args = super().cxx_args()
        args.append("-std=c++17")
        return args
