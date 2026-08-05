"""Pure, CPU-friendly tensor ops for the DFlash accept rule and block build.

No model loading, no KV cache, no state machine -- just the deterministic
argmax-domain math from RFC 1.2 (batch==1 in v1). The native draft->verify->
rollback loop (Task 6) composes these; the emitted-vs-cached split lives in
``Committed`` because conflating the two is the plan's #1 losslessness risk.
"""

from __future__ import annotations

from typing import NamedTuple, Union

import torch

AnchorLike = Union[int, torch.Tensor]


class Committed(NamedTuple):
    emitted: torch.Tensor  # [B, accept+1] accepted drafts ++ bonus; appended to output
    block_prefix: torch.Tensor  # [B, accept+1] anchor ++ accepted drafts; KV kept (start += accept+1)
    bonus: torch.Tensor  # [B, 1] posterior[:, accept]; emitted-but-NOT-cached, next anchor


def build_block(anchor: AnchorLike, mask_token_id: int, block_size: int) -> torch.Tensor:
    """Return ``[anchor, MASK x (block_size - 1)]`` as an int64 ``[B, block_size]``."""
    if not torch.is_tensor(anchor):
        anchor = torch.tensor(anchor)
    anchor = anchor.to(torch.long).reshape(-1, 1)
    masks = torch.full(
        (anchor.shape[0], block_size - 1),
        int(mask_token_id),
        dtype=anchor.dtype,
        device=anchor.device,
    )
    return torch.cat([anchor, masks], dim=1)


def acceptance_length(candidates: torch.Tensor, target_predict: torch.Tensor) -> int:
    """Number of leading draft tokens the target agrees with (RFC 1.2).

    ``accept = cumprod(candidates[:, 1:] == target_predict[:, :-1]).sum()`` --
    ``cumprod`` zeroes out everything past the first mismatch, so the sum is the
    count of accepted tokens in ``[0, block_size - 1]``.
    """
    matches = (candidates[:, 1:] == target_predict[:, :-1]).long()
    return int(matches.cumprod(dim=1).sum().item())


def committed_tokens(
    block: torch.Tensor, posterior: torch.Tensor, accept: int
) -> Committed:
    """Split one verify step into emitted / cached / bonus tensors.

    ``emitted`` = the ``accept`` accepted drafts followed by the bonus token
    ``posterior[:, accept]`` (the anchor at ``block[:, 0]`` is already in the
    running sequence, so it is not re-emitted). ``block_prefix`` = the KV-retained
    slice ``block[:, :accept+1]`` (advances ``start`` by ``accept+1``); the bonus
    is deliberately excluded from it -- emitted-but-not-cached.
    """
    accept = int(accept)
    accepted_drafts = block[:, 1 : accept + 1]
    bonus = posterior[:, accept : accept + 1]
    return Committed(
        emitted=torch.cat([accepted_drafts, bonus], dim=1),
        block_prefix=block[:, : accept + 1],
        bonus=bonus,
    )


__all__ = ["Committed", "build_block", "acceptance_length", "committed_tokens"]
