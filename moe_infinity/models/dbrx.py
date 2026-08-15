# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from typing import Dict

import torch
import torch.nn as nn
from transformers.models.dbrx.modeling_dbrx import DbrxExperts, DbrxRouter


class SyncDbrxFFNBlock(nn.Module):
    layer_id: int = None

    def __init__(self, config):
        super().__init__()
        ffn_config = config.ffn_config

        self.router = DbrxRouter(
            hidden_size=config.d_model,
            moe_num_experts=ffn_config.moe_num_experts,
            moe_top_k=ffn_config.moe_top_k,
            moe_jitter_eps=ffn_config.moe_jitter_eps,
            moe_normalize_expert_weights=ffn_config.moe_normalize_expert_weights,
        )

        self.experts = DbrxExperts(
            hidden_size=config.d_model,
            ffn_hidden_size=ffn_config.ffn_hidden_size,
            moe_num_experts=ffn_config.moe_num_experts,
            ffn_act_fn=ffn_config.ffn_act_fn,
        )

        self.num_experts = ffn_config.moe_num_experts
        self.top_k = ffn_config.moe_top_k
        self.hidden_size = config.d_model

        self.archer_tracer = None
        self.archer_engine = None
        self.expert_tensor_ids: Dict[int, int] = None

    def forward(self, x: torch.Tensor) -> tuple:
        batch_size, sequence_length, hidden_dim = x.shape

        weights, top_weights, top_experts = self.router(x)

        # Build router_mask and routing_weights_mask for dispatch_local
        # top_experts shape: (batch*seq, top_k); weights shape: (batch*seq, num_experts)
        hidden_states_flat = x.view(-1, hidden_dim)
        B = hidden_states_flat.shape[0]

        router_mask = torch.zeros(
            B, self.num_experts, dtype=torch.bool, device=top_experts.device
        )
        router_mask.scatter_(1, top_experts.view(B, -1), True)

        routing_weights_mask = torch.zeros(
            B,
            self.num_experts,
            dtype=top_weights.dtype,
            device=top_weights.device,
        )
        routing_weights_mask.scatter_add_(
            1, top_experts.view(B, -1), top_weights.view(B, -1)
        )

        self.expert_executor.dispatch_local(
            self.layer_id,
            hidden_states_flat,
            router_mask,
            routing_weights_mask,
            router_logits=weights,
        )
        final_hidden_states = self.expert_executor.wait_dispatch_local()

        final_hidden_states = final_hidden_states.view(
            batch_size, sequence_length, hidden_dim
        ).to(x.dtype)
        return final_hidden_states, weights
