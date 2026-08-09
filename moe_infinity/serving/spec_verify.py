from __future__ import annotations

from dataclasses import dataclass

import torch

from moe_infinity.serving.kv_cache import PagedKVCache
from moe_infinity.serving.spec_state import SpecDecodeState
from moe_infinity.spec_decode._dflash_ops import (
    acceptance_length,
    committed_tokens,
)


@dataclass
class VerifyResult:
    emitted_tokens: list[int]
    next_anchor: int
    accept: int
    cache_committed: int
    cached_len: int


def apply_verify_step(
    *,
    kv_cache: PagedKVCache,
    seq_id: int,
    state: SpecDecodeState,
    block: torch.Tensor,
    posterior: torch.Tensor,
    block_size: int,
) -> VerifyResult:
    """Commit one serving-path DFlash verify step, rolling back rejected KV.

    Precondition: the verify forward has already appended KV for all
    ``block_size`` block tokens to ``kv_cache[seq_id]``. This keeps only the
    committed prefix (``accept + 1`` tokens: anchor + accepted drafts) via
    ``truncate_tokens`` and returns the ``accept + 1`` newly emitted tokens
    (``[d_1..d_accept, bonus]``) plus the bonus, which is emitted-but-not-cached
    and becomes the next step's anchor. Mirrors the sync ``_generate_single``
    protocol (dflash.py) so serving output is token-identical.
    """
    accept = acceptance_length(block, posterior)
    committed = committed_tokens(block, posterior, accept)
    cache_committed = accept + 1
    accounting = state.record_verify(
        block_len=block_size, committed=cache_committed
    )
    kv_cache.truncate_tokens(seq_id, accounting.truncate_target)
    emitted = [int(t) for t in committed.emitted[0].tolist()]
    next_anchor = int(committed.bonus[0, 0].item())
    return VerifyResult(
        emitted_tokens=emitted,
        next_anchor=next_anchor,
        accept=accept,
        cache_committed=cache_committed,
        cached_len=accounting.cached_len,
    )


__all__ = ["VerifyResult", "apply_verify_step"]
