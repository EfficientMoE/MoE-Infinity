"""Unit-test conftest: stub optional/GPU dependencies before collection.

Prevents ImportError when moe_infinity submodules are imported in
environments where the optional 'nvtx' profiling library or compiled
CUDA extensions (_store, _engine) are not installed.
"""

import sys
from unittest.mock import MagicMock


def _stub_if_missing(name: str) -> None:
    if name in sys.modules:
        return
    try:
        __import__(name)
    except (ImportError, OSError):
        sys.modules[name] = MagicMock()


_stub_if_missing("nvtx")
