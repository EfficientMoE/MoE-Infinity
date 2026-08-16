from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaMLP,
        GlmMoeDsaTopkRouter,
    )

    _GLM_AVAILABLE = True
except ImportError:
    _GLM_AVAILABLE = False
    GlmMoeDsaTopkRouter = GlmMoeDsaMLP = None


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
        self.experts = nn.ModuleList(
            [
                GlmMoeDsaMLP(
                    config=config,
                    intermediate_size=config.moe_intermediate_size,
                )
                for _ in range(config.n_routed_experts)
            ]
        )
        self.shared_experts = GlmMoeDsaMLP(
            config=config,
            intermediate_size=config.moe_intermediate_size
            * config.n_shared_experts,
        )

    def _route(self, hidden_flat: torch.Tensor):
        dev = hidden_flat.device
        if self.gate.e_score_correction_bias.device != dev:
            self.gate.e_score_correction_bias = (
                self.gate.e_score_correction_bias.to(dev)
            )
        routed = self.gate(hidden_flat)
        if isinstance(routed, tuple):
            # transformers >= 5.15 moved routing into GlmMoeDsaTopkRouter, which
            # returns (router_logits, topk_weights, topk_indices).
            _, topk_weights, topk_indices = routed
            return topk_indices, topk_weights
        return self._route_tokens_to_experts(routed)

    def _route_tokens_to_experts(self, router_logits: torch.Tensor):
        # Mirror of transformers GlmMoeDsaMoE.route_tokens_to_experts (grouped
        # sigmoid top-k). Inlined so offload routing stays correct on builds
        # that no longer expose that method (see #144); sigmoid gating, not
        # softmax.
        router_logits = router_logits.sigmoid()
        scores = router_logits + self.gate.e_score_correction_bias
        group_scores = (
            scores.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(
            group_scores, k=self.topk_group, dim=-1, sorted=False
        )[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
        topk_indices = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

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
