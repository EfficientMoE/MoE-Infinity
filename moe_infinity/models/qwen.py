from typing import Optional, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP


class _ExpertExecutor(Protocol):
    def dispatch_local(
        self,
        layer_id: Optional[int],
        hidden_states: torch.Tensor,
        router_mask: torch.Tensor,
        router_weights: torch.Tensor,
        router_logits: Optional[torch.Tensor] = None,
        prefetcher: object = None,
    ) -> None: ...

    def wait_dispatch_local(self) -> torch.Tensor: ...


class Qwen3MoEBlock(nn.Module):
    layer_id: Optional[int] = None

    def __init__(self, config: Qwen3MoeConfig):
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

        self.expert_executor: Optional[_ExpertExecutor] = None
        self.lib: Optional[object] = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        router_mask = F.one_hot(selected_experts, num_classes=self.num_experts)
        routing_weights_mask = (
            routing_weights[:, :, None] * router_mask
        ).permute(0, 2, 1)
        router_mask = router_mask.permute(0, 2, 1)
        router_mask = torch.any(router_mask, dim=-1)
        routing_weights_mask = torch.sum(routing_weights_mask, dim=-1)

        executor = self.expert_executor
        if executor is None:
            raise RuntimeError("Qwen3MoEBlock requires an expert executor")
        executor.dispatch_local(
            self.layer_id,
            hidden_states,
            router_mask,
            routing_weights_mask,
            router_logits=router_logits,
        )
        final_hidden_states = executor.wait_dispatch_local()

        final_hidden_states = final_hidden_states.view(
            batch_size, sequence_length, hidden_dim
        ).to(hidden_states.dtype)

        return final_hidden_states
