# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.mixtral.modeling_mixtral import (
    MixtralBlockSparseTop2MLP,
)

from moe_infinity.utils import ArcherConfig


class SyncMixtralSparseMoeBlock(nn.Module):
    archer_config: ArcherConfig = None
    layer_id: int = None
    is_gptq: bool = False

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList(
            [MixtralBlockSparseTop2MLP(config) for _ in range(self.num_experts)]
        )

        self.expert_executor = None
        self.archer_tracer = None
        self.archer_engine = None
        self.expert_tensor_ids: Dict[int, int] = None

    def _forward_gptq(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros_like(hidden_states)

        expert_mask = F.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = self.experts[expert_idx](current_state)
            current_hidden_states *= routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )

        final_hidden_states = final_hidden_states.view(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        num_tokens = batch_size * sequence_length
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states_flat)

        if getattr(self, "is_gptq", False):
            final_hidden_states = self._forward_gptq(
                hidden_states, router_logits
            )
            return final_hidden_states, router_logits

        hidden_states = hidden_states_flat

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        router_mask = F.one_hot(selected_experts, num_classes=self.num_experts)
        routing_weights_mask = (
            routing_weights[:, :, None] * router_mask
        ).permute(0, 2, 1)
        router_mask = router_mask.permute(0, 2, 1)
        router_mask = torch.logical_or(
            router_mask[:, :, 0], router_mask[:, :, 1]
        )
        routing_weights_mask = torch.sum(routing_weights_mask, dim=-1)

        expert_index = selected_experts.reshape(
            batch_size, sequence_length, self.top_k
        )

        self.expert_executor.dispatch_local(
            self.layer_id,
            hidden_states,
            router_mask,
            routing_weights_mask,
            router_logits=router_logits,
        )
        final_hidden_states = self.expert_executor.wait_dispatch_local()

        final_hidden_states = final_hidden_states.view(
            batch_size, sequence_length, hidden_dim
        ).to(hidden_states.dtype)
        return final_hidden_states, router_logits
