# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .routing import (
    hash_route,
    indices_weights_to_masks,
    topk_route,
)


class SyncDeepSeekV4MoEBlock(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_expert = config.n_routed_experts
        self.hidden_size = config.hidden_size
        self.routed_scaling_factor = getattr(
            config, "routed_scaling_factor", 1.0
        )
        self.norm_topk_prob = getattr(config, "norm_topk_prob", True)
        self.num_hash_layers = getattr(config, "num_hash_layers", 0)
        self.is_hash = layer_idx < self.num_hash_layers

        self.gate_weight = nn.Parameter(
            torch.empty(self.num_expert, self.hidden_size), requires_grad=False
        )
        if self.is_hash:
            self.register_parameter("gate_bias", None)
            self.register_buffer(
                "tid2eid",
                torch.zeros(
                    config.vocab_size,
                    self.num_experts_per_tok,
                    dtype=torch.long,
                ),
                persistent=True,
            )
        else:
            self.gate_bias = nn.Parameter(
                torch.zeros(self.num_expert), requires_grad=False
            )
            self.tid2eid = None

        self.archer_tracer = None
        self.archer_engine = None
        self.expert_executor = None
        self.expert_tensor_ids: Optional[dict] = None
        self.layer_id = None

    def compute_routing(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.is_hash:
            if input_ids is None:
                raise ValueError("hash routing requires input_ids")
            weights, indices = hash_route(
                hidden_states,
                input_ids,
                self.gate_weight,
                self.tid2eid,
                self.routed_scaling_factor,
            )
        else:
            weights, indices = topk_route(
                hidden_states,
                self.gate_weight,
                self.gate_bias,
                self.num_experts_per_tok,
                self.routed_scaling_factor,
                self.norm_topk_prob,
            )
        return weights, indices

    def compute_routing_masks(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weights, indices = self.compute_routing(hidden_states, input_ids)
        return indices_weights_to_masks(indices, weights, self.num_expert)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.expert_executor is None:
            raise RuntimeError(
                "expert_executor not attached; SyncDeepSeekV4MoEBlock requires "
                "OffloadEngine wiring for forward"
            )
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat = hidden_states.view(-1, hidden_dim)
        router_mask, routing_weight = self.compute_routing_masks(
            hidden_states, input_ids
        )

        self.expert_executor.dispatch_local(
            self.layer_id,
            flat,
            router_mask,
            routing_weight,
        )
        final_hidden_states = self.expert_executor.wait_dispatch_local()
        return final_hidden_states.view(
            batch_size, sequence_length, hidden_dim
        ).to(hidden_states.dtype)
