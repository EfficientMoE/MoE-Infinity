# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import NllbMoeConfig
from transformers.models.nllb_moe.modeling_nllb_moe import (
    NllbMoeDenseActDense,
    NllbMoeTop2Router,
)

from moe_infinity.utils import ArcherConfig

GPU_IDX_COUNTER = 0


class SyncNllbMoeSparseMLP(nn.Module):
    archer_config: ArcherConfig = None
    layer_id: int = None

    def __init__(
        self,
        config: NllbMoeConfig,
        ffn_dim: int,
        expert_class: nn.Module = NllbMoeDenseActDense,
    ):
        super().__init__()
        self.router = NllbMoeTop2Router(config)
        self.moe_token_dropout = config.moe_token_dropout
        self.token_dropout = nn.Dropout(self.moe_token_dropout)

        self.num_experts = config.num_experts

        self.experts = nn.ModuleDict()
        for idx in range(self.num_experts):
            self.experts[f"expert_{idx}"] = expert_class(config, ffn_dim)

        self.archer_tracer = None
        self.archer_engine = None
        self.expert_tensor_ids: Dict[int, int] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        top_1_mask, router_probs = self.router(hidden_states, padding_mask)
        combining_weights = router_probs.reshape(
            (batch_size, sequence_length, self.num_experts)
        )
        router_mask = combining_weights.bool()

        top_1_expert_index = torch.argmax(top_1_mask, dim=-1)

        self.expert_executor.dispatch_local(
            self.layer_id,
            hidden_states,
            router_mask,
            combining_weights,
            router_logits=None,
        )
        next_states = self.expert_executor.wait_dispatch_local()

        next_states[next_states == 0] = hidden_states[next_states == 0]
        hidden_states = next_states.to(hidden_states.dtype)

        return hidden_states, (
            router_probs.to("cuda:0", non_blocking=True),
            top_1_expert_index.to("cuda:0", non_blocking=True),
        )
