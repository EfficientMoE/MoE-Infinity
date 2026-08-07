"""Pure, CPU-friendly helpers for DFlash route-ahead expert prefetch (Track A1).

No model loading, no prefetcher state, no CUDA -- just set math over router
outputs. The route-ahead design (``.sisyphus/plans/dflash-deferred-tracks-plan.md``,
Track A A0) prefetches the ACTUAL routed union of experts -- the same union
``ExpertExecutor.dispatch_local`` derives from ``router_mask``
(``distributed/expert_executor.py:101-111``) -- instead of the legacy
``mean(0).topk(2)`` pool. These helpers compute that union from either the
boolean mask or the raw logits, plus the coverage/waste metrics used to score
predicted-vs-actual prefetch sets. A2/A3 consume them; nothing here mutates
routing or model outputs.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch

MaskLike = Union[torch.Tensor, np.ndarray, Sequence[Sequence[int]]]
LogitsLike = Union[torch.Tensor, np.ndarray, Sequence[Sequence[float]]]
IdsLike = Union[torch.Tensor, np.ndarray, Sequence[int]]


def _as_2d_tensor(x: Union[torch.Tensor, np.ndarray, Sequence[object]], name: str) -> torch.Tensor:
    """Coerce ``x`` to a 2-D torch tensor without changing device/dtype semantics."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif not torch.is_tensor(x):
        x = torch.tensor(x)
    if x.dim() != 2:
        raise ValueError(f"{name} must be 2-D [num_tokens, num_experts]; got shape {tuple(x.shape)}")
    return x


def _to_id_set(ids: IdsLike) -> set[int]:
    """Coerce a tensor/ndarray/sequence of expert ids to a python ``set[int]``."""
    if torch.is_tensor(ids):
        items = ids.flatten().tolist()
    elif isinstance(ids, np.ndarray):
        items = ids.flatten().tolist()
    else:
        items = list(ids)
    return {int(i) for i in items}


def union_experts_from_mask(router_mask: MaskLike) -> list[int]:
    """Sorted union of expert indices routed by ANY token in ``router_mask``.

    ``router_mask`` is the per-token routing mask ``[num_tokens, num_experts]``
    (bool or 0/1 int) built by the MoE block, e.g. ``models/gpt_oss.py:141-147``.
    This matches the executor's derivation exactly:
    ``expert_list = arange(num_experts)[mask.sum(0) > 0]``
    (``distributed/expert_executor.py:101-111``). Returns ``[]`` when no token
    routes anywhere (or ``num_tokens == 0``).
    """
    mask = _as_2d_tensor(router_mask, "router_mask").to(torch.bool)
    routed = mask.any(dim=0).nonzero().flatten().tolist()
    return sorted(int(i) for i in routed)


def union_experts_from_logits(router_logits: LogitsLike, top_k: int) -> list[int]:
    """Sorted union of per-token ``top_k`` experts from ``[num_tokens, num_experts]`` logits.

    Applies ``torch.topk`` row-wise and unions the selected indices. Softmax is
    monotonic, so top-k on raw logits equals top-k on the routing probabilities
    computed in the MoE block (``models/gpt_oss.py:132-135``) -- hence this
    matches ``union_experts_from_mask`` whenever the mask was built with the
    same ``top_k``. Ties follow ``torch.topk`` semantics. Logits are cast to
    float32 first so bf16/fp16 inputs behave deterministically on CPU.
    """
    logits = _as_2d_tensor(router_logits, "router_logits").to(torch.float32)
    num_tokens, num_experts = logits.shape
    top_k = int(top_k)
    if not 1 <= top_k <= num_experts:
        raise ValueError(f"top_k must be in [1, {num_experts}]; got {top_k}")
    if num_tokens == 0:
        return []
    selected = torch.topk(logits, k=top_k, dim=1).indices
    return sorted({int(i) for i in selected.flatten().tolist()})


def prefetch_coverage(predicted_ids: IdsLike, actual_ids: IdsLike) -> float:
    """Fraction of actually-routed experts that the prediction covered.

    ``|predicted ∩ actual| / |actual|``; returns ``1.0`` when ``actual`` is
    empty (nothing to cover -- nothing was wasted either). This is the
    per-layer term of the plan's ``coverage = Σ|P_l ∩ A_l| / Σ|A_l|`` metric
    (Track A A0 item 4); callers aggregate across layers.
    """
    predicted = _to_id_set(predicted_ids)
    actual = _to_id_set(actual_ids)
    if not actual:
        return 1.0
    return len(predicted & actual) / len(actual)


def rejected_expert_ids(full_union_ids: IdsLike, kept_union_ids: IdsLike) -> list[int]:
    r"""Sorted set-difference ``full \ kept`` -- the wasted prefetch set.

    ``full_union_ids`` is the union prefetched over the whole speculative block;
    ``kept_union_ids`` is the union over only the tokens that survived
    verification (the kept prefix). The difference was fetched but never used:
    the per-layer ``rejected_waste`` id set of Track A A0 item 4.
    """
    return sorted(_to_id_set(full_union_ids) - _to_id_set(kept_union_ids))


__all__ = [
    "union_experts_from_mask",
    "union_experts_from_logits",
    "prefetch_coverage",
    "rejected_expert_ids",
]
