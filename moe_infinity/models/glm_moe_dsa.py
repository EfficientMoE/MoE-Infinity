# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

# pyright: reportMissingImports=false

from typing import Optional

import nvtx
import torch
import torch.nn as nn


class SyncGlmMoeDsaMoEBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
            GlmMoeDsaMLP,
            GlmMoeDsaTopkRouter,
        )

        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_expert = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok

        self.experts = nn.ModuleList(
            [
                GlmMoeDsaMLP(
                    config, intermediate_size=config.moe_intermediate_size
                )
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = GlmMoeDsaTopkRouter(config)
        if config.n_shared_experts is not None:
            self.shared_experts = GlmMoeDsaMLP(
                config=config,
                intermediate_size=config.moe_intermediate_size
                * config.n_shared_experts,
            )

        self.archer_tracer = None
        self.archer_engine = None
        self.expert_tensor_ids: Optional[dict[int, int]] = None

    # Verbatim port of transformers GlmMoeDsaMoE.route_tokens_to_experts
    # (sigmoid + noaux_tc group top-k + norm + routed_scaling_factor). Must be
    # kept in sync with the installed modeling_glm_moe_dsa.py.
    def route_tokens_to_experts(self, router_logits):
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.gate.e_score_correction_bias.to(
            scores.device
        )
        group_scores = (
            scores_for_choice.view(
                -1, self.n_group, self.num_expert // self.n_group
            )
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
            .expand(-1, self.n_group, self.num_expert // self.n_group)
            .reshape(-1, self.num_expert)
        )
        scores_for_choice = scores_for_choice.masked_fill(
            ~score_mask.bool(), float("-inf")
        )
        topk_indices = torch.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False
        )[1]
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    @nvtx.annotate("GlmMoeDsaPrepare", color="blue")
    def _prepare_expert_route(self, hidden_states):
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        topk_weights = topk_weights.to(torch.float32)

        num_tokens = topk_indices.shape[0]
        router_mask = torch.zeros(
            num_tokens,
            self.num_expert,
            dtype=torch.bool,
            device=topk_indices.device,
        )
        router_mask.scatter_(1, topk_indices, True)
        routing_weights_mask = torch.zeros(
            num_tokens,
            self.num_expert,
            dtype=torch.float32,
            device=topk_weights.device,
        )
        routing_weights_mask.scatter_(1, topk_indices, topk_weights)
        return router_mask, routing_weights_mask, router_logits

    @nvtx.annotate(message="GlmMoeDsaMoEBlock", color="blue")
    def forward(self, hidden_states):
        identity = hidden_states
        routing_mask, routing_weight, router_logits = (
            self._prepare_expert_route(hidden_states)
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
