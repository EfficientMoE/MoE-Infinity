"""Hardware gates for kernels written or tuned for one GPU.

Two kinds of specialisation are gated (see docs/hardware-gated-kernels.md):

* ``SM120``: the kernel uses instructions that only exist on compute
  capability 12.0 (sm_120a), e.g. block-scaled ``mma.sync ... kind::mxf8f6f4``,
  ``cvt.rn.f16x2.e2m1x2`` or CUTLASS sm120 collectives.
* ``RTX_PRO_6000``: tile tables, cost models, grid sizes or L2 windows were
  tuned on the RTX PRO 6000 Blackwell: compute capability 12.0, 188 SMs,
  128 MiB L2, 99 KiB shared memory per block. The Workstation, Max-Q and
  Server editions match; the RTX 5090 (170 SMs, 96 MiB L2) does not.
* ``RTX_PRO_6000_SERVER`` / ``RTX_PRO_6000_MAXQ``: edition-specific sub-gates
  for the 600 W Server Edition (335.5 TF BF16 / 656.9 TF FP8, 1530.5 GB/s DRAM)
  and the 300 W Max-Q Workstation Edition (268.8 TF BF16 / 530.3 TF FP8,
  1641.1 GB/s DRAM). Whenever Server edits do not regress Max-Q, both editions
  are fused under ``RTX_PRO_6000``.

A gated path runs only when ``supports(gate, device)`` is true; every other
device keeps the previous implementation. ``MOE_INFINITY_UNVALIDATED_HW=1``
lifts the ``RTX_PRO_6000*`` tuning gates, for re-tuning on another card; it
never lifts ``SM120``. ``MOE_INFINITY_PRO6000_EDITION=server|maxq`` overrides
edition detection on a matching RTX PRO 6000 device. The C++ twin is
``extensions/kernel/hw_profile.h``.

Call ``supports`` once when a module or op is set up and keep the result; it
is cheap but not free on a per-token path.
"""

from __future__ import annotations

import functools
import os

import torch

SM120 = "sm120"
RTX_PRO_6000 = "rtx_pro_6000"
RTX_PRO_6000_SERVER = "rtx_pro_6000_server"
RTX_PRO_6000_MAXQ = "rtx_pro_6000_maxq"

_ALL_GATES = (
    SM120,
    RTX_PRO_6000,
    RTX_PRO_6000_SERVER,
    RTX_PRO_6000_MAXQ,
)

_SM120_CAPABILITY = (12, 0)
_PRO6000_SMS = 188
_PRO6000_L2_BYTES = 128 * 1024 * 1024
_PRO6000_SMEM_PER_BLOCK = 99 * 1024


def _detect_pro6000_edition(p) -> str | None:
    env = os.environ.get("MOE_INFINITY_PRO6000_EDITION", "").strip().lower()
    if env == "server":
        return RTX_PRO_6000_SERVER
    if env in ("maxq", "max_q", "max-q"):
        return RTX_PRO_6000_MAXQ
    name = getattr(p, "name", "") or ""
    if "Max-Q" in name or "MaxQ" in name:
        return RTX_PRO_6000_MAXQ
    if "Server" in name:
        return RTX_PRO_6000_SERVER
    mem_khz = getattr(p, "memory_clock_rate", 0) or 0
    if 0 < mem_khz <= 13_000_000:
        return RTX_PRO_6000_SERVER
    if mem_khz >= 13_500_000:
        return RTX_PRO_6000_MAXQ
    return None


@functools.lru_cache(maxsize=None)
def _gates(index: int) -> frozenset:
    p = torch.cuda.get_device_properties(index)
    gates = set()
    if (p.major, p.minor) == _SM120_CAPABILITY:
        gates.add(SM120)
        if (
            p.multi_processor_count == _PRO6000_SMS
            and p.L2_cache_size == _PRO6000_L2_BYTES
            and p.shared_memory_per_block_optin == _PRO6000_SMEM_PER_BLOCK
        ):
            gates.add(RTX_PRO_6000)
            edition = _detect_pro6000_edition(p)
            if edition is not None:
                gates.add(edition)
    return frozenset(gates)


def _index(device) -> int:
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device
    d = torch.device(device)
    if d.type != "cuda":
        return -1
    return torch.cuda.current_device() if d.index is None else d.index


def supports(gate: str, device=None) -> bool:
    """True if ``device`` (default: the current CUDA device) passes ``gate``."""
    if gate not in _ALL_GATES:
        raise ValueError(f"unknown hardware gate {gate!r}")
    if not torch.cuda.is_available():
        return False
    index = _index(device)
    if index < 0:
        return False
    if (
        gate != SM120
        and os.environ.get("MOE_INFINITY_UNVALIDATED_HW") == "1"
    ):
        return True
    return gate in _gates(index)


def pro6000_edition(device=None) -> str | None:
    """Return ``'server'``, ``'maxq'``, or ``None`` for ``device``."""
    if not torch.cuda.is_available():
        return None
    index = _index(device)
    if index < 0:
        return None
    gs = _gates(index)
    if RTX_PRO_6000_SERVER in gs:
        return "server"
    if RTX_PRO_6000_MAXQ in gs:
        return "maxq"
    return None
