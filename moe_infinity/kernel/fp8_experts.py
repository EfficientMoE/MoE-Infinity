"""Backend choice for FP8 (e4m3, 128x128 block-scaled) routed experts.

Hardware-gated (docs/hardware-gated-kernels.md). The native FP8 expert
kernels (``extensions/kernel/fp8_moe``) were written and tuned for the RTX PRO
6000 Blackwell (sm_120, 188 SMs, 128 MiB L2):

* CUTLASS sm120 blockwise FP8 GEMM -> ``SM120``
* tensor-core decode GEMV and the tile / stream-K choices for 188 SMs
  -> ``RTX_PRO_6000``

Both gates must pass. On every other device the previous path runs: the
dispatcher de-quantises the FP8 weights to BF16 per call and runs the BF16
CUTLASS GEMM. The C++ side (``MoEMLP::TryBindFp8Sm120Pro6000``) checks the
same gates per module, so the native kernels never run on other hardware even
if this decision is bypassed.
"""

from __future__ import annotations

import torch

from moe_infinity.utils.hw_profile import RTX_PRO_6000, SM120, supports

FP8_EXPERTS_SM120_PRO6000 = "fp8_sm120_pro6000"
FP8_EXPERTS_DEQUANT_BF16 = "fp8_dequant_bf16"


def fp8_expert_backend(device=None) -> str:
    """Backend for FP8 routed experts on ``device`` (default: current)."""
    if supports(SM120, device) and supports(RTX_PRO_6000, device):
        return FP8_EXPERTS_SM120_PRO6000
    return FP8_EXPERTS_DEQUANT_BF16


def fp8_sm120_pro6000_allowed() -> bool:
    """True if any visible GPU may run the sm120 / RTX PRO 6000 kernels.

    The dispatcher's exec modules re-check the gate for their own device, so
    on a mixed machine only the matching GPUs use them.
    """
    if not torch.cuda.is_available():
        return False
    return any(
        fp8_expert_backend(i) == FP8_EXPERTS_SM120_PRO6000
        for i in range(torch.cuda.device_count())
    )
