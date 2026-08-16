"""Additive NVTX phase markers for the BM4 expert-H2D / compute overlap trace.

``benchmarks/dflash/parse_overlap.py`` (design §10 BM4 / §7 hide inequality)
unions three NVTX push/pop ranges to decide which offloaded-expert host->device
bytes are *hidden* behind DFlash compute: ``dflash_draft`` (the drafter pass),
``route_ahead_router`` (the target router/gate that runs ahead of the expert
fetch), and ``target_verify`` (the width-B verify forward). The parser reads
``nsys stats --report nvtx_pushpop_trace``; only push/pop ranges land there
(default-domain start/end ranges are dropped by nsys 2025.1.3), so these markers
MUST be emitted with ``torch.cuda.nvtx.range_push``/``range_pop``.

The markers are read-only: they never change routing, compute, or emitted
tokens. They are a no-op when NVTX/CUDA is unavailable and free when no profiler
is attached, so leaving them in the hot path costs nothing outside a trace.

Leaf module (no ``moe_infinity`` imports) so it can be imported from both
``spec_decode.dflash`` and, lazily, ``models.gpt_oss`` without a cycle.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, Optional, Tuple

BM4_COMPUTE_PHASES: Tuple[str, str, str] = (
    "dflash_draft",
    "route_ahead_router",
    "target_verify",
)

_range_push: Optional[Callable[[str], object]]
_range_pop: Optional[Callable[[], object]]

try:
    import torch

    _range_push = torch.cuda.nvtx.range_push
    _range_pop = torch.cuda.nvtx.range_pop
except Exception:  # pragma: no cover - torch is a hard dependency of the pkg
    _range_push = None
    _range_pop = None


@contextmanager
def nvtx_phase(name: str) -> Iterator[None]:
    """Emit a balanced NVTX push/pop range ``name`` around the wrapped phase."""
    push = _range_push
    pop = _range_pop
    pushed = False
    if push is not None:
        try:
            push(name)
            pushed = True
        except Exception:
            pushed = False
    try:
        yield
    finally:
        if pushed and pop is not None:
            try:
                pop()
            except Exception:
                pass


__all__ = ["BM4_COMPUTE_PHASES", "nvtx_phase"]
