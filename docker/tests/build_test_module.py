#!/usr/bin/env python3
"""
Build script for the C++ test extension (test_io_module).

Reuses PrefetchBuilder to get the same source files, include paths, and
compiler flags as the main prefetch_op extension, but swaps the pybind11
entry point from py_archer_prefetch.cpp to test_io_module.cpp.
"""

import sys

sys.path.insert(0, ".")

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension

from op_builder.prefetch import PrefetchBuilder

builder = PrefetchBuilder()
ext = builder.builder()

# Replace the pybind11 entry point with our test module
ext.sources = [s for s in ext.sources if "py_archer_prefetch" not in s]
ext.sources.append("docker/tests/test_io_module.cpp")

# Set the module name so --inplace puts it in docker/tests/
ext.name = "docker.tests.test_io_module"

print(f"Building test_io_module with {len(ext.sources)} source files:")
for src in ext.sources:
    print(f"  {src}")
print()

setup(
    name="test_io_build",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    script_args=["build_ext", "--inplace"],
)

print("\ntest_io_module built successfully.")
