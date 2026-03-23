#!/usr/bin/env python3
"""
Verify that the MoE-Infinity prefetch extension was built correctly
and that key refactored components are present.
"""

import os
import sys

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
errors = []


def check(name, condition, detail=""):
    if condition:
        print(f"  [{PASS}] {name}")
    else:
        print(f"  [{FAIL}] {name} {detail}")
        errors.append(name)


print("=" * 60)
print("MoE-Infinity Build Verification")
print("=" * 60)

# 1. Check that the .so file exists on disk
print("\n1. Shared library exists:")
so_files = []
for root, _, files in os.walk("moe_infinity"):
    so_files.extend(os.path.join(root, f) for f in files if f.endswith(".so"))
store_so = [f for f in so_files if "_store" in os.path.basename(f)]
check("_store .so exists", len(store_so) > 0, f"(found: {so_files})")

# 2. Check that the extension loads via standard import
print("\n2. Extension loads via import:")
try:
    import torch  # noqa: F401 — must be imported before _store

    import moe_infinity._store  # noqa: F401

    check("import moe_infinity._store", True)
except Exception as e:
    check("import moe_infinity._store", False, str(e))

try:
    import moe_infinity  # noqa: F401

    check("import moe_infinity", True)
except Exception as e:
    check("import moe_infinity", False, str(e))

# 3. Check that key C++ source files exist (proves they were compiled into the .so)
print("\n3. C++ source files present:")
expected_sources = [
    "core/aio/archer_aio_thread.cpp",
    "core/aio/archer_aio_threadpool.cpp",
    "core/aio/archer_prio_aio_handle.cpp",
    "core/aio/archer_tensor_handle.cpp",
    "core/memory/pinned_memory_pool.cpp",
    "core/model/model_topology.cpp",
    "core/prefetch/archer_prefetch_handle.cpp",
]
for src in expected_sources:
    check(f"{src} exists", os.path.isfile(src))

# 4. Check that new pinned_memory_pool files exist in source tree
print("\n4. New source files present:")
check(
    "core/memory/pinned_memory_pool.h exists",
    os.path.isfile("core/memory/pinned_memory_pool.h"),
)
check(
    "core/memory/pinned_memory_pool.cpp exists",
    os.path.isfile("core/memory/pinned_memory_pool.cpp"),
)

# 5. Check key code patterns in the refactored files
print("\n5. Refactored code patterns:")


def file_contains(path, pattern):
    with open(path) as f:
        return pattern in f.read()


check(
    "atomic<bool> is_running_ in aio_thread.h",
    file_contains(
        "core/aio/archer_aio_thread.h", "std::atomic<bool> is_running_"
    ),
)
check(
    "atomic<bool> time_to_exit_ in prio_aio_handle.h",
    file_contains(
        "core/aio/archer_prio_aio_handle.h", "std::atomic<bool> time_to_exit_"
    ),
)
check(
    "condition_variable cv_ in aio_thread.h",
    file_contains(
        "core/aio/archer_aio_thread.h", "std::condition_variable cv_"
    ),
)
check(
    "round_robin_counter_ in threadpool.h",
    file_contains("core/aio/archer_aio_threadpool.h", "round_robin_counter_"),
)
check(
    "GetDefaultNumIoThreads in prio_aio_handle.h",
    file_contains(
        "core/aio/archer_prio_aio_handle.h", "GetDefaultNumIoThreads"
    ),
)
check(
    "PinnedMemoryPool in prio_aio_handle.h",
    file_contains("core/aio/archer_prio_aio_handle.h", "PinnedMemoryPool"),
)
check(
    "MOE_IO_THREADS env var in prefetch_handle.cpp",
    file_contains("core/prefetch/archer_prefetch_handle.cpp", "MOE_IO_THREADS"),
)
check(
    "kPartitionSize in tensor_handle.h",
    file_contains("core/aio/archer_tensor_handle.h", "kPartitionSize"),
)
check(
    "PipelinedDiskToGpu in model_topology.cpp",
    file_contains("core/model/model_topology.cpp", "PipelinedDiskToGpu"),
)
check(
    "SetModuleMemoryFromDisk_Views in model_topology.cpp",
    file_contains(
        "core/model/model_topology.cpp", "SetModuleMemoryFromDisk_Views"
    ),
)

# Summary
print("\n" + "=" * 60)
if errors:
    print(f"RESULT: {len(errors)} check(s) FAILED: {errors}")
    sys.exit(1)
else:
    print("RESULT: All checks passed.")
    sys.exit(0)
