# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def sqrtsoftplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x).sqrt()


def topk_route(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    top_k: int,
    routed_scaling_factor: float = 1.0,
    norm_topk_prob: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    logits = F.linear(flat.float(), gate_weight.float())
    scores = sqrtsoftplus(logits)
    select = scores if bias is None else scores + bias.float().unsqueeze(0)
    indices = torch.topk(select, top_k, dim=-1, sorted=False).indices
    weights = scores.gather(1, indices)
    if norm_topk_prob:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return weights * routed_scaling_factor, indices


def hash_route(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    gate_weight: torch.Tensor,
    tid2eid: torch.Tensor,
    routed_scaling_factor: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    logits = F.linear(flat.float(), gate_weight.float())
    scores = sqrtsoftplus(logits)
    indices = tid2eid[input_ids.reshape(-1)].long()
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return weights * routed_scaling_factor, indices


def indices_weights_to_masks(
    indices: torch.Tensor,
    weights: torch.Tensor,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens = indices.shape[0]
    router_mask = torch.zeros(
        num_tokens, num_experts, dtype=torch.bool, device=indices.device
    )
    router_mask.scatter_(1, indices, True)
    routing_weights_mask = torch.zeros(
        num_tokens, num_experts, dtype=weights.dtype, device=weights.device
    )
    routing_weights_mask.scatter_add_(
        1, indices, weights.to(routing_weights_mask.dtype)
    )
    return router_mask, routing_weights_mask
