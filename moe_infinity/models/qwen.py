import nvtx
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP


class Qwen3MoEBlock(nn.Module):
    layer_id = None
    lib = None

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.gate = nn.Linear(
            config.hidden_size, config.num_experts, bias=False
        )
        self.experts = nn.ModuleList(
            [
                Qwen3MoeMLP(
                    config, intermediate_size=config.moe_intermediate_size
                )
                for _ in range(self.num_experts)
            ]
        )

    @nvtx.annotate("Qwen3Prepare", color="blue")
    def __prepare_expert_route(self, hidden_states):
        router_logits = self.gate(hidden_states)

        if self.lib is not None:
            topk_indices, router_mask, routing_weights_mask = (
                self.lib.topk_softmax(router_logits)
            )
            return router_logits, router_mask, routing_weights_mask

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        B, E = routing_weights.shape[0], self.num_experts
        router_mask = torch.zeros(
            B, E, dtype=torch.bool, device=selected_experts.device
        )
        router_mask.scatter_(1, selected_experts, True)

        routing_weights_mask = torch.zeros(
            B, E, dtype=routing_weights.dtype, device=routing_weights.device
        )
        routing_weights_mask.scatter_add_(1, selected_experts, routing_weights)

        return router_logits, router_mask, routing_weights_mask

    @nvtx.annotate("Qwen3MoEBlock", color="blue")
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits, router_mask, routing_weights_mask = (
            self.__prepare_expert_route(hidden_states)
        )

        self.expert_executor.dispatch_local(
            self.layer_id, hidden_states, router_mask, routing_weights_mask
        )
        final_hidden_states = self.expert_executor.wait_dispatch_local()

        final_hidden_states = final_hidden_states.view(
            batch_size, sequence_length, hidden_dim
        ).to(hidden_states.dtype)

        return final_hidden_states, router_logits
