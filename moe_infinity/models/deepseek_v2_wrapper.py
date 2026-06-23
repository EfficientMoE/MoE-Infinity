# pyright: reportMissingImports=false

from typing import Optional

import nvtx
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2MLP,
)

from moe_infinity.models.deepseek import DeepseekMoEGate


class SyncDeepseekV2MoEBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_expert = config.n_routed_experts

        self.experts = nn.ModuleList(
            [
                DeepseekV2MLP(
                    config, intermediate_size=config.moe_intermediate_size
                )
                for i in range(config.n_routed_experts)
            ]
        )

        self.gate = DeepseekMoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = (
                config.moe_intermediate_size * config.n_shared_experts
            )
            self.shared_experts = DeepseekV2MLP(
                config=config, intermediate_size=intermediate_size
            )

        self.archer_tracer = None
        self.archer_engine = None
        self.expert_tensor_ids: Optional[dict[int, int]] = None

    @nvtx.annotate("DeepSeekPrepare", color="blue")
    def __prepare_expert_route(self, hidden_states):
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )
        B, E = routing_weights.shape[0], self.num_expert
        router_mask = torch.zeros(
            B, E, dtype=torch.bool, device=selected_experts.device
        )

        router_mask.scatter_(1, selected_experts, True)

        routing_weights_mask = torch.zeros(
            B, E, dtype=routing_weights.dtype, device=routing_weights.device
        )
        routing_weights_mask.scatter_add_(1, selected_experts, routing_weights)

        return router_mask, routing_weights_mask, router_logits

    @nvtx.annotate(message="DeepseekMoEBlock", color="blue")
    def forward(self, hidden_states):
        identity = hidden_states
        routing_mask, routing_weight, router_logits = (
            self.__prepare_expert_route(hidden_states)
        )
        batch_size, sequence_length, hidden_dim = identity.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        self.expert_executor.dispatch_local(
            self.layer_id,
            hidden_states,
            routing_mask,
            routing_weight,
            router_logits=router_logits,
        )
        final_hidden_states = self.expert_executor.wait_dispatch_local()

        final_hidden_states = final_hidden_states.view(
            batch_size, sequence_length, hidden_dim
        ).to(hidden_states.dtype)
        if self.config.n_shared_experts is not None:
            final_hidden_states = final_hidden_states + self.shared_experts(
                identity
            )
        return final_hidden_states
