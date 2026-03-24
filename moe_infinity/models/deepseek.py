from typing import Dict, Optional

import nvtx
import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_infinity.kernel.router import launch_fused_softmax_topk_nobias


class DeepseekMoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_routed_experts = config.n_routed_experts
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )

    def forward(self, hidden_states):
        """
        Forward pass for the MoE gate.
        :param hidden_states: Input tensor of shape (batch_size, sequence_length, hidden_size).
        :return: Gating logits of shape (batch_size, sequence_length, n_routed_experts).
        """
        # Compute the gating logits
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32),
            self.weight.type(torch.float32),
            None,
        )
        return logits


class DeepseekMoEBlock(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_expert = config.n_routed_experts

        if self.config.model_type == "deepseek_v2":
            from .modeling_deepseek_v2 import DeepseekV2MLP, MoEGate

            self.mlp_cls = DeepseekV2MLP
            self.gate_cls = MoEGate
        if self.config.model_type == "deepseek_v3":
            from .modeling_deepseek_v3 import DeepseekV3MLP, MoEGate

            self.mlp_cls = DeepseekV3MLP
            self.gate_cls = MoEGate

        self.experts = nn.ModuleList(
            [
                self.mlp_cls(
                    config, intermediate_size=config.moe_intermediate_size
                )
                for i in range(config.n_routed_experts)
            ]
        )

        self.gate = self.gate_cls(config)
        if config.n_shared_experts is not None:
            intermediate_size = (
                config.moe_intermediate_size * config.n_shared_experts
            )
            self.shared_experts = self.mlp_cls(
                config=config, intermediate_size=intermediate_size
            )

        self.archer_tracer = None
        self.archer_engine = None
        self.expert_tensor_ids: Optional[Dict[int, int]] = None

    @nvtx.annotate("DeepSeekPrepare", color="blue")
    def __prepare_expert_route(self, hidden_states):
        gate_output = self.gate(hidden_states)

        # Native MoEGate returns tuple: V2=(topk_idx, topk_weight, aux_loss),
        # V3=(topk_idx, topk_weight). Legacy DeepseekMoEGate returns raw logits.
        if isinstance(gate_output, tuple):
            if len(gate_output) == 3:
                selected_experts, routing_weights, _ = gate_output
            elif len(gate_output) == 2:
                selected_experts, routing_weights = gate_output
            else:
                raise ValueError(
                    f"Unsupported gate output with {len(gate_output)} elements"
                )
            routing_weights = routing_weights.to(torch.float32)
        else:
            # Fallback for legacy gates that return raw logits
            router_logits = gate_output
            routing_weights = F.softmax(
                router_logits, dim=1, dtype=torch.float32
            )
            routing_weights, selected_experts = torch.topk(
                routing_weights, self.num_experts_per_tok, dim=-1
            )

        # Convert (topk_idx, topk_weight) -> (router_mask, routing_weights_mask)
        # for expert_executor.dispatch_local()
        B, E = selected_experts.shape[0], self.num_expert
        router_mask = torch.zeros(
            B, E, dtype=torch.bool, device=selected_experts.device
        )
        router_mask.scatter_(1, selected_experts, True)

        routing_weights_mask = torch.zeros(
            B, E, dtype=torch.float32, device=routing_weights.device
        )
        routing_weights_mask.scatter_(1, selected_experts, routing_weights)

        return router_mask, routing_weights_mask

    @nvtx.annotate(message="DeepseekMoEBlock", color="blue")
    def forward(self, hidden_states):
        identity = hidden_states
        routing_mask, routing_weight = self.__prepare_expert_route(
            hidden_states
        )
        batch_size, sequence_length, hidden_dim = identity.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        self.expert_executor.dispatch_local(
            self.layer_id, hidden_states, routing_mask, routing_weight
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
