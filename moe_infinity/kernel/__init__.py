from __future__ import annotations

import torch

from .paged_attention_ops import paged_attention_fwd
from .sglang_adapter import sglang_topk_softmax as topk_softmax


def launch_fused_softmax_topk_nobias(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    top_k: int,
    normalize_topk: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    from .router import launch_fused_softmax_topk_nobias as _impl

    return _impl(hidden_states, weight, top_k, normalize_topk)


__all__ = [
    "topk_softmax",
    "launch_fused_softmax_topk_nobias",
    "paged_attention_fwd",
]
