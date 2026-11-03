"""Dispatch flag for the grouped decode MoE kernels (#306).

RTX PRO 6000 only: the kernels' split-K / row-split layouts and launch table
(extensions/kernel/grouped_moe_decode_pro6000.cu) were tuned on the RTX PRO
6000 Blackwell Max-Q. Every other device keeps per-expert execution. The
native dispatcher applies the same gate itself (moe_infinity::hw::Supports);
this mirrors it for Python-side setup and tests.
"""

from __future__ import annotations

import os

from moe_infinity.utils.hw_profile import RTX_PRO_6000, supports


def grouped_decode_pro6000_enabled(device=None) -> bool:
    """True if the grouped decode path should run on ``device``."""
    if os.environ.get("MOE_GROUPED_DECODE", "1") == "0":
        return False
    return supports(RTX_PRO_6000, device)
