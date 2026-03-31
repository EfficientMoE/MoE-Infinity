"""Root conftest: stub CUDA-only deps so CPU-only CI can collect all tests."""

import sys
from unittest.mock import MagicMock

import pytest

_OPTIONAL_CUDA_MODULES = [
    "nvtx",
    "flash_attn",
    "sglang_kernel",
    "moe_infinity._store",
    "moe_infinity._engine",
    "moe_infinity._kv_cache",
    "moe_infinity._paged_attn",
]


def _stub_if_missing(name: str) -> None:
    if name in sys.modules:
        return
    try:
        __import__(name)
    except (ImportError, OSError):
        sys.modules[name] = MagicMock()


for _mod in _OPTIONAL_CUDA_MODULES:
    _stub_if_missing(_mod)


def pytest_collection_modifyitems(config, items):
    try:
        import torch

        if torch.cuda.is_available():
            return
    except ImportError:
        pass

    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "gpu" in item.keywords or "multi_gpu" in item.keywords:
            item.add_marker(skip_gpu)
