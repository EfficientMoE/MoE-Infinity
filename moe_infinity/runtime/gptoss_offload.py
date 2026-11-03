"""Kernel choice for offloaded GPT-OSS experts (native MoEMLP, expert type 6).

The fused MXFP4 expert kernels (``extensions/kernel/v4_fp4/
mxfp4_gptoss_moe.cu``, ``gptoss_expert.cpp``) were tuned on the RTX PRO 6000
Blackwell: tile / grid tables for 188 SMs, 99 KiB shared-memory stage budgets
and an 80 MB L2 persisting window. They run only where
``supports(RTX_PRO_6000)`` holds; every other device keeps the previous path
(de-quantise MXFP4 to BF16, then matmul). Each native module also re-checks
its own device (``moe_infinity::hw::Supports``).
"""

from __future__ import annotations

import torch

from moe_infinity.utils.hw_profile import RTX_PRO_6000, supports


def gptoss_mxfp4_pro6000_enabled() -> bool:
    """True if any visible device may run the RTX PRO 6000 MXFP4 kernels."""
    if not torch.cuda.is_available():
        return False
    return any(
        supports(RTX_PRO_6000, i) for i in range(torch.cuda.device_count())
    )
