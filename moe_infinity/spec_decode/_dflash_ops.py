"""Pure, CPU-friendly tensor ops for the DFlash accept rule and block build.

No model loading, no KV cache, no state machine -- just the deterministic
argmax-domain math from RFC 1.2. The native draft->verify->rollback loop
composes these; the emitted-vs-cached split lives in ``Committed`` because
conflating the two is the plan's #1 losslessness risk.

Batching (Track C): ``build_block`` / ``acceptance_length`` /
``committed_tokens`` keep their original single-effective-accept behaviour
(byte-identical); the ``*_batched``/``*_ragged`` variants generalize them to a
leading batch dim with PER-SEQUENCE accept lengths and ragged commits. A batch
row whose drafts share no uniform accept length cannot be represented in one
dense ``Committed`` tensor, so the ragged variant returns one ``Committed``
per row.
"""

from __future__ import annotations

from typing import List, NamedTuple, Sequence, Union

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


def build_block_with_prefixes(
    prefixes: Sequence[Sequence[int]], mask_token_id: int, block_size: int
) -> torch.Tensor:
    """Ragged batched block build: one row per prefix, right-filled with MASK.

    Row ``b`` is ``prefixes[b] ++ [MASK x (block_size - len(prefixes[b]))]``.
    A prefix holds tokens ALREADY known for that row (the re-fed
    emitted-but-uncached run of the batched loop -- at minimum the anchor);
    the drafter only fills the MASK slots afterwards. ``build_block`` is the
    special case where every prefix is a single anchor. Prefixes may be empty
    (an all-MASK row: a finished sequence kept in the dense batch whose row is
    never read) and may fill the whole block (no room for new drafts).
    """
    rows: List[List[int]] = []
    for prefix in prefixes:
        row = [int(t) for t in prefix]
        if len(row) > int(block_size):
            raise ValueError(
                f"block prefix length {len(row)} exceeds block_size {block_size}"
            )
        row = row + [int(mask_token_id)] * (int(block_size) - len(row))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long)


def acceptance_lengths(
    candidates: torch.Tensor, target_predict: torch.Tensor
) -> List[int]:
    """Batched accept rule: per-row leading-match counts (RFC 1.2).

    Same ``cumprod`` math as ``acceptance_length`` but reduced per row, so a
    ``[batch, block]`` block yields one accept count in ``[0, block_size - 1]``
    per sequence instead of a single batch-wide sum.
    """
    matches = (candidates[:, 1:] == target_predict[:, :-1]).long()
    return [int(x) for x in matches.cumprod(dim=1).sum(dim=1).tolist()]


def committed_tokens_ragged(
    block: torch.Tensor, posterior: torch.Tensor, accepts: Sequence[int]
) -> List[Committed]:
    """Per-row ``committed_tokens`` for ragged per-sequence accept lengths.

    Row ``b`` is split with its own ``accepts[b]``; the returned list is
    ragged (row ``b``'s ``emitted`` has ``accepts[b] + 1`` tokens), which a
    single dense ``Committed`` cannot express.
    """
    if len(accepts) != int(block.shape[0]):
        raise ValueError(
            f"accepts has {len(accepts)} rows but block has batch {block.shape[0]}"
        )
    return [
        committed_tokens(block[b : b + 1], posterior[b : b + 1], int(accepts[b]))
        for b in range(int(block.shape[0]))
    ]


__all__ = [
    "Committed",
    "acceptance_length",
    "acceptance_lengths",
    "build_block",
    "build_block_with_prefixes",
    "committed_tokens",
    "committed_tokens_ragged",
]
