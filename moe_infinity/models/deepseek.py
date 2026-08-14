from typing import Dict, Optional

import nvtx
import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_infinity.kernel import topk_softmax as kernel_topk_softmax


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
            import transformers.models.deepseek_v2.modeling_deepseek_v2 as _dsv2
            from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
                DeepseekV2MLP,
            )

            self.mlp_cls = DeepseekV2MLP
            # DeepseekV2MoEGate was removed in newer Transformers; fall back to
            # the local DeepseekMoEGate which has the same raw-logits interface.
            _gate_cls = getattr(_dsv2, "DeepseekV2MoEGate", None)
            self.gate_cls = (
                _gate_cls if _gate_cls is not None else DeepseekMoEGate
            )
        if self.config.model_type == "deepseek_v3":
            from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
                DeepseekV3MLP,
            )

            self.mlp_cls = DeepseekV3MLP
            # V3 upstream has no standalone gate; use V2 gate when available,
            # otherwise fall back to the local DeepseekMoEGate.
            import transformers.models.deepseek_v2.modeling_deepseek_v2 as _dsv2

            _gate_cls = getattr(_dsv2, "DeepseekV2MoEGate", None)
            self.gate_cls = (
                _gate_cls if _gate_cls is not None else DeepseekMoEGate
            )

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
            router_logits = None
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
            router_mask, routing_weights_mask = kernel_topk_softmax(
                gate_output,
                self.num_experts_per_tok,
                self.num_expert,
                renormalize=True,
            )
            return router_mask, routing_weights_mask, gate_output

        B, E = selected_experts.shape[0], self.num_expert
        router_mask = torch.zeros(
            B, E, dtype=torch.bool, device=selected_experts.device
        )
        router_mask.scatter_(1, selected_experts, True)

        routing_weights_mask = torch.zeros(
            B, E, dtype=torch.float32, device=routing_weights.device
        )
        routing_weights_mask.scatter_(1, selected_experts, routing_weights)

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
