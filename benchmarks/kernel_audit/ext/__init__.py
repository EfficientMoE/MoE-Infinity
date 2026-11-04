"""Loader for the audit shim that links against moe_infinity/_store*.so."""

from __future__ import annotations

import os
import sysconfig
import tempfile
from functools import lru_cache
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_PKG = _REPO / "moe_infinity"


@lru_cache(maxsize=1)
def load():
    from torch.utils.cpp_extension import load as _load

    import moe_infinity._store  # noqa: F401  (pulls in torch/cuda deps first)

    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    store = _PKG / f"_store{suffix}"
    if not store.exists():
        raise FileNotFoundError(store)
    build_dir = Path(
        os.environ.get(
            "KAUDIT_BUILD_DIR",
            str(Path(tempfile.gettempdir()) / "moe_infinity_kernel_audit"),
        )
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    return _load(
        name="kaudit_ext",
        sources=[str(_HERE / "audit_ext.cpp")],
        extra_cflags=["-O2", "-std=c++17"],
        extra_ldflags=[
            f"-L{_PKG}",
            f"-l:{store.name}",
            f"-Wl,-rpath,{_PKG}",
        ],
        with_cuda=True,
        build_directory=str(build_dir),
        verbose=False,
    )
