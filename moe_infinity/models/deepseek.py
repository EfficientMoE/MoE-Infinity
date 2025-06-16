from typing import Dict

import nvtx
import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # self.gate = self.gate_cls(config)
        self.gate = DeepseekMoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = (
                config.moe_intermediate_size * config.n_shared_experts
            )
            self.shared_experts = self.mlp_cls(
                config=config, intermediate_size=intermediate_size
            )

        self.archer_tracer = None
        self.archer_engine = None
        self.expert_tensor_ids: Dict[int, int] = None

    @nvtx.annotate("DeepSeekPrepare", color="blue")
    def __prepare_expert_route(self, hidden_states):
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)  # dtype float32

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )
        # if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
        #     routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        # routing_weights = routing_weights.to(hidden_states.dtype)

        # print(f"hidden_states shape: {hidden_states.shape}")
        # print(f"routing_weights shape: {routing_weights.shape}")

        # Compute sparse mask via scatter
        B, E = routing_weights.shape[0], self.num_expert
        router_mask = torch.zeros(
            B, E, dtype=torch.bool, device=selected_experts.device
        )

        # print("selected_experts", selected_experts.shape)
        # print("routing_weights", routing_weights.shape)
        # print("router_mask", router_mask.shape)
        # print("router_logits", router_logits.shape)
        router_mask.scatter_(1, selected_experts, True)

        routing_weights_mask = torch.zeros(
            B, E, dtype=routing_weights.dtype, device=routing_weights.device
        )
        routing_weights_mask.scatter_add_(1, selected_experts, routing_weights)

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
