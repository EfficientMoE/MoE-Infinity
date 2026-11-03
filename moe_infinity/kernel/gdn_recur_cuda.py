"""JIT-built CUDA recurrence for the L2-blocked GDN path (#304).

Compiled on first use with ``torch.utils.cpp_extension.load`` for the
current device's architecture; ``available()`` is False if nvcc is missing
or the build fails, and callers then use the Triton recurrence.
"""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
_SRC = Path(__file__).with_name("csrc") / "gdn_recur.cu"

_CPP = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
void gdn_recur_launch(const void*, const void*, const void*, const void*,
                      const void*, const float*, void*, float*, int, int, int,
                      int, int, int, cudaStream_t);
void recur(torch::Tensor qg, torch::Tensor kd, torch::Tensor w, torch::Tensor u,
           torch::Tensor S, torch::Tensor g, torch::Tensor o, torch::Tensor h,
           int64_t t_base, int64_t nb) {
  int B = qg.size(0), TB = qg.size(1), H = qg.size(2), T = o.size(1);
  gdn_recur_launch(qg.data_ptr(), kd.data_ptr(), w.data_ptr(), u.data_ptr(),
                   S.data_ptr(), g.data_ptr<float>(), o.data_ptr(),
                   h.data_ptr<float>(), B, H, T, (int)t_base, TB, (int)nb,
                   c10::cuda::getCurrentCUDAStream());
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("recur", &recur); }
"""


@functools.lru_cache(maxsize=None)
def _module():
    from torch.utils.cpp_extension import load

    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}"
    build = Path(
        os.environ.get(
            "MOE_LINEAR_ATTN_BUILD_DIR",
            Path.home() / ".cache" / "moe_infinity" / f"gdn_recur_sm{arch}",
        )
    )
    build.mkdir(parents=True, exist_ok=True)
    cpp = build / "gdn_recur_binding.cpp"
    if not cpp.exists() or cpp.read_text() != _CPP:
        cpp.write_text(_CPP)
    return load(
        name=f"moe_gdn_recur_sm{arch}",
        sources=[str(cpp), str(_SRC)],
        extra_cuda_cflags=[
            "-O3",
            f"-gencode=arch=compute_{arch},code=sm_{arch}",
            "--use_fast_math",
        ],
        build_directory=str(build),
        verbose=False,
    )


def available() -> bool:
    try:
        _module()
        return True
    except Exception as exc:  # noqa: BLE001 - nvcc missing / build failure
        logger.warning("CUDA GDN recurrence unavailable: %s", exc)
        return False


def recur(qg, kd, w, u, S, g, o, h, t_base: int, nb: int) -> None:
    _module().recur(qg, kd, w, u, S, g, o, h, t_base, nb)
