#!/usr/bin/env python3
"""
Build script for the MoE-Infinity prefetch extension.

This builds just the prefetch_op shared library, which contains all
the refactored core/aio/ code (multi-threaded I/O, pinned memory pool,
pipelined transfers, partitioned storage).

This is equivalent to the prefetch portion of `BUILD_OPS=1 pip install -e .`
but avoids building the fused_glu_cuda and expert_gemm extensions which
have pre-existing issues on the development branch.
"""

import sys

sys.path.insert(0, ".")

from op_builder.prefetch import PrefetchBuilder
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension

builder = PrefetchBuilder()
ext = builder.builder()

print(f"Building prefetch extension with {len(ext.sources)} source files:")
for src in ext.sources:
    print(f"  {src}")
print()

setup(
    name="prefetch_build",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    script_args=["build_ext", "--inplace"],
)

print("\nPrefetch extension built successfully.")
