"""Pytest configuration for I/O integration tests."""

import os
import shutil
import tempfile

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: tests requiring CUDA GPU")


def pytest_collection_modifyitems(config, items):
    """Auto-skip cuda-marked tests when no GPU is available."""
    try:
        import torch

        if torch.cuda.is_available():
            return
    except ImportError:
        pass

    skip_cuda = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip_cuda)


@pytest.fixture
def workspace_tmpdir():
    """Provide a temp directory on a real filesystem (not tmpfs).

    O_DIRECT requires a filesystem that supports direct I/O.
    /tmp may be tmpfs in some Docker configurations, so we create
    temp dirs under the workspace instead.
    """
    # Prefer workspace dir (Docker), fall back to script dir, then /tmp
    for candidate in ["/workspace/MoE-Infinity", os.getcwd()]:
        if os.path.isdir(candidate):
            tmpdir = tempfile.mkdtemp(prefix="moe_test_", dir=candidate)
            yield tmpdir
            shutil.rmtree(tmpdir, ignore_errors=True)
            return
    # Fallback to system default
    tmpdir = tempfile.mkdtemp(prefix="moe_test_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)
