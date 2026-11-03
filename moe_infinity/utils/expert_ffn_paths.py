"""Which expert-FFN prefill path runs on a device (#307 / #308).

The RTX PRO 6000 paths were tuned on that card (188 SMs, 128 MiB L2, 99 KiB
shared memory per block) and are gated with ``RTX_PRO_6000``
(docs/hardware-gated-kernels.md):

* ``cutlass_pro6000``: ``fused_moe_ffn_into`` as a gate/up dual GEMM + down
  GEMM with the tuned tile / split-K table, cost model, L2 persisting window
  and L2-fit rasterisation. Elsewhere ``cutlass_generic`` runs: the previous
  three CUTLASS GEMMs with fixed 64x64 tiles.
* ``grouped_pro6000``: ``grouped_moe_prefill`` as routing + grouped gate/up +
  grouped down (incl. the bf16 per-token combine). Elsewhere ``per_expert``
  runs: one ``fused_moe_ffn_into`` per expert and a weighted fp32 index_add.

The C++ dispatch makes the same decision (``fused_moe_ffn_pro6000_enabled`` /
``grouped_moe_prefill_pro6000_enabled`` in ``moe_infinity._store``, through
``extensions/kernel/hw_profile.h``); these helpers let Python callers and
tests see it without a GPU.
"""

from __future__ import annotations

from moe_infinity.utils.hw_profile import RTX_PRO_6000, supports

CUTLASS_PRO6000 = "cutlass_pro6000"
CUTLASS_GENERIC = "cutlass_generic"
GROUPED_PRO6000 = "grouped_pro6000"
PER_EXPERT = "per_expert"


def expert_ffn_path(device=None) -> str:
    """Path of ``fused_moe_ffn_into`` on ``device``."""
    return (
        CUTLASS_PRO6000 if supports(RTX_PRO_6000, device) else CUTLASS_GENERIC
    )


def grouped_prefill_path(device=None) -> str:
    """Path of ``grouped_moe_prefill`` on ``device``."""
    return GROUPED_PRO6000 if supports(RTX_PRO_6000, device) else PER_EXPERT
