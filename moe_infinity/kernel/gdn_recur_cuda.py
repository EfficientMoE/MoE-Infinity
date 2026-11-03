"""JIT-built CUDA recurrence for the L2-blocked GDN path (#304).

Compiled on first use with ``torch.utils.cpp_extension.load`` for the
current device's architecture; ``available()`` is False if nvcc is missing
or the build fails, and callers then use the Triton recurrence.

Hardware: RTX_PRO_6000. 92.5 KiB of dynamic shared memory per CTA (fits the
99 KiB limit), one wave of 128 CTAs on 188 SMs and the persisting-L2 window
were sized for the RTX PRO 6000 Blackwell; ``available()`` is False on any
other device (generic mma.sync / ldmatrix / movmatrix ISA, sm_80+).
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
// Persisting L2 window on [ptr, ptr + bytes) for the current stream
// (bytes == 0 clears it).
void l2_window(torch::Tensor buf, int64_t bytes) {
  cudaStream_t s = c10::cuda::getCurrentCUDAStream();
  cudaStreamAttrValue a = {};
  if (bytes > 0) {
    int dev; cudaGetDevice(&dev);
    int maxp = 0, maxw = 0;
    cudaDeviceGetAttribute(&maxp, cudaDevAttrMaxPersistingL2CacheSize, dev);
    cudaDeviceGetAttribute(&maxw, cudaDevAttrMaxAccessPolicyWindowSize, dev);
    size_t lim = std::min<size_t>(bytes, maxp);
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, lim);
    a.accessPolicyWindow.base_ptr = buf.data_ptr();
    a.accessPolicyWindow.num_bytes = std::min<size_t>(bytes, maxw);
    a.accessPolicyWindow.hitRatio = std::min(1.0f, (float)lim / (float)a.accessPolicyWindow.num_bytes);
    a.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    a.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  } else {
    a.accessPolicyWindow.num_bytes = 0;
    cudaCtxResetPersistingL2Cache();
  }
  cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &a);
}
std::vector<int64_t> l2_limits() {
  int dev; cudaGetDevice(&dev);
  int maxp = 0, maxw = 0;
  cudaDeviceGetAttribute(&maxp, cudaDevAttrMaxPersistingL2CacheSize, dev);
  cudaDeviceGetAttribute(&maxw, cudaDevAttrMaxAccessPolicyWindowSize, dev);
  return {maxp, maxw};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("recur", &recur);
  m.def("l2_window", &l2_window);
  m.def("l2_limits", &l2_limits);
}
"""


@functools.lru_cache(maxsize=None)
def _module():
    from torch.utils.cpp_extension import load

    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}"
    build = Path(
        os.environ.get(
            "MOE_LINEAR_ATTN_BUILD_DIR",
            Path.home()
            / ".cache"
            / "moe_infinity"
            / f"gdn_recur_pro6000_sm{arch}",
        )
    )
    build.mkdir(parents=True, exist_ok=True)
    cpp = build / "gdn_recur_binding.cpp"
    if not cpp.exists() or cpp.read_text() != _CPP:
        cpp.write_text(_CPP)
    return load(
        name=f"moe_gdn_recur_pro6000_sm{arch}",
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
    from moe_infinity.utils.hw_profile import RTX_PRO_6000, supports

    if not supports(RTX_PRO_6000):
        return False
    try:
        _module()
        return True
    except Exception as exc:  # noqa: BLE001 - nvcc missing / build failure
        logger.warning("CUDA GDN recurrence unavailable: %s", exc)
        return False


def recur(qg, kd, w, u, S, g, o, h, t_base: int, nb: int) -> None:
    _module().recur(qg, kd, w, u, S, g, o, h, t_base, nb)


def l2_window(buf, nbytes: int) -> None:
    _module().l2_window(buf, nbytes)


def l2_limits() -> tuple[int, int]:
    return tuple(_module().l2_limits())
