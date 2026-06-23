# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import torch

from .fp4_expert import fp4_expert_forward

BundleProvider = Callable[[int, int], Sequence[torch.Tensor]]


class DeepSeekV4PythonExpertExecutor:
    def __init__(
        self,
        bundle_provider: BundleProvider,
        swiglu_limit: float = 0.0,
    ):
        self.bundle_provider = bundle_provider
        self.swiglu_limit = swiglu_limit

    def active_experts(self, router_mask: torch.Tensor) -> List[int]:
        counts = router_mask.view(-1, router_mask.shape[-1]).sum(dim=0)
        return torch.nonzero(counts, as_tuple=False).flatten().tolist()

    def execute(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        router_mask: torch.Tensor,
        routing_weight: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        final = torch.zeros(
            num_tokens,
            hidden_dim,
            dtype=torch.float32,
            device=hidden_states.device,
        )

        for expert_id in self.active_experts(router_mask):
            token_mask = router_mask[:, expert_id]
            token_indices = torch.nonzero(token_mask, as_tuple=False).flatten()
            if token_indices.numel() == 0:
                continue

            expert_input = hidden_states.index_select(0, token_indices)
            w1, s1, w2, s2, w3, s3 = self.bundle_provider(layer_id, expert_id)

            expert_out = fp4_expert_forward(
                expert_input,
                w1,
                s1,
                w2,
                s2,
                w3,
                s3,
                swiglu_limit=self.swiglu_limit,
            )

            weights = routing_weight[token_indices, expert_id].unsqueeze(1)
            final.index_add_(
                0, token_indices, expert_out.float() * weights.float()
            )

        return final.to(hidden_states.dtype)


def make_indexer_bundle_provider(
    indexer,
    device: torch.device,
    cache: bool = True,
) -> BundleProvider:
    _cache: dict[Tuple[int, int], Sequence[torch.Tensor]] = {}

    def provider(layer_id: int, expert_id: int) -> Sequence[torch.Tensor]:
        key = (layer_id, expert_id)
        if cache and key in _cache:
            return _cache[key]
        bundle = indexer.bundle(layer_id, expert_id)
        tensors = [t.to(device) for t in indexer.load_bundle_tensors(bundle)]
        if cache:
            _cache[key] = tensors
        return tensors

    return provider
