from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_infinity.utils import ArcherConfig


class _PackedExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_experts = config.num_local_experts
        hidden = config.hidden_size
        intermediate = config.intermediate_size

        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, hidden, 2 * intermediate)
        )
        self.gate_up_proj_bias = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, intermediate, hidden)
        )
        self.down_proj_bias = nn.Parameter(torch.empty(num_experts, hidden))


class SyncGptOssMLP(nn.Module):
    archer_config: Optional[ArcherConfig] = None
    layer_id: Optional[int] = None

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.alpha = 1.702
        self.swiglu_limit = 7.0

        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=True)
        self.experts = _PackedExperts(config)

        self.expert_executor = None
        self.archer_tracer = None
        self.archer_engine = None
        self.expert_tensor_ids: Optional[Dict[int, int]] = None

    def _swiglu(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        gate = gate.clamp(max=self.swiglu_limit)
        up = up.clamp(-self.swiglu_limit, self.swiglu_limit)
        return (up + 1) * (gate * torch.sigmoid(gate * self.alpha))

    def _expert_forward(
        self, hidden_states: torch.Tensor, expert_idx: int
    ) -> torch.Tensor:
        device = hidden_states.device
        gate_up_w = self.experts.gate_up_proj[expert_idx].to(device)
        gate_up_b = self.experts.gate_up_proj_bias[expert_idx].to(device)
        down_w = self.experts.down_proj[expert_idx].to(device)
        down_b = self.experts.down_proj_bias[expert_idx].to(device)

        gate_up = F.linear(hidden_states, gate_up_w.t(), gate_up_b)
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        activated = self._swiglu(gate, up)
        return F.linear(activated, down_w.t(), down_b)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        num_tokens = batch_size * sequence_length
        hidden_flat = hidden_states.view(-1, hidden_dim)

        router_logits = self.router(hidden_flat)

        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights = routing_weights / routing_weights.sum(
            dim=-1, keepdim=True
        )
        routing_weights = routing_weights.to(hidden_states.dtype)

        router_mask = torch.zeros(
            num_tokens,
            self.num_experts,
            dtype=torch.bool,
            device=hidden_states.device,
        )
        router_mask.scatter_(1, selected_experts, True)

        routing_weights_mask = torch.zeros(
            num_tokens,
            self.num_experts,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        routing_weights_mask.scatter_(1, selected_experts, routing_weights)

        if self.expert_executor is not None:
            self.expert_executor.dispatch_local(
                self.layer_id,
                hidden_flat,
                router_mask,
                routing_weights_mask,
                router_logits=router_logits,
            )
            final_hidden = self.expert_executor.wait_dispatch_local()
        else:
            final_hidden = torch.zeros_like(hidden_flat)
            for expert_idx in range(self.num_experts):
                token_mask = router_mask[:, expert_idx]
                if not token_mask.any():
                    continue
                expert_input = hidden_flat[token_mask]
                expert_output = self._expert_forward(expert_input, expert_idx)
                weight = routing_weights_mask[token_mask, expert_idx].unsqueeze(
                    -1
                )
                final_hidden[token_mask] += weight * expert_output

        final_hidden = final_hidden.view(
            batch_size, sequence_length, hidden_dim
        )
        final_hidden = final_hidden.to(hidden_states.dtype)
        return final_hidden, router_logits
