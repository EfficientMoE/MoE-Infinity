from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaMLP,
        GlmMoeDsaMoE,
        GlmMoeDsaTopkRouter,
    )

    _GLM_AVAILABLE = True
except ImportError:
    _GLM_AVAILABLE = False
    GlmMoeDsaMoE = GlmMoeDsaTopkRouter = GlmMoeDsaMLP = None


class SyncGlmMoeDsaMoEBlock(nn.Module):
    archer_config = None
    layer_id: Optional[int] = None
    expert_executor = None
    expert_prefetcher = None
    expert_tracer = None
    expert_predictor = None
    archer_engine = None
    lib = None
    expert_tensor_map = None

    def __init__(self, config):
        super().__init__()
        if not _GLM_AVAILABLE:
            raise ImportError(
                "transformers >= 5.12 is required for GLM-MoE-DSA support"
            )
        self.config = config
        self.num_experts = config.n_routed_experts
        self.n_routed_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor

        self.gate = GlmMoeDsaTopkRouter(config)
        self.shared_experts = GlmMoeDsaMLP(
            config=config,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
        )
        self._hf_route_tokens = GlmMoeDsaMoE.route_tokens_to_experts

    def _route(self, hidden_flat: torch.Tensor):
        dev = hidden_flat.device
        if self.gate.e_score_correction_bias.device != dev:
            self.gate.e_score_correction_bias = self.gate.e_score_correction_bias.to(dev)
        router_logits = self.gate(hidden_flat)
        return self._hf_route_tokens(self, router_logits)

    def _local_experts(self, hidden_flat, router_mask, routing_weights_mask):
        return torch.zeros_like(hidden_flat)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq, hid = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hid)
        N = hidden_flat.shape[0]

        topk_idx, topk_weights = self._route(hidden_flat)

        router_mask = torch.zeros(
            N, self.num_experts, dtype=torch.bool, device=hidden_flat.device
        )
        router_mask.scatter_(1, topk_idx, True)

        routing_weights_mask = torch.zeros(
            N,
            self.num_experts,
            dtype=topk_weights.dtype,
            device=hidden_flat.device,
        )
        routing_weights_mask.scatter_(1, topk_idx, topk_weights)

        if self.expert_executor is not None:
            self.expert_executor.dispatch_local(
                self.layer_id,
                hidden_flat,
                router_mask,
                routing_weights_mask,
                router_logits=None,
            )
            expert_output = self.expert_executor.wait_dispatch_local()
        else:
            expert_output = self._local_experts(
                hidden_flat, router_mask, routing_weights_mask
            )

        shared_output = self.shared_experts(hidden_flat)
        result = expert_output.view(-1, hid) + shared_output
        return result.view(bsz, seq, hid).to(hidden_states.dtype)
