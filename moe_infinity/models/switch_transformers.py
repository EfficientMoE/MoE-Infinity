# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from typing import Dict

import torch
import torch.nn as nn
from transformers import SwitchTransformersConfig
from transformers.activations import ACT2FN
from transformers.models.switch_transformers.modeling_switch_transformers import (
    SwitchTransformersDenseActDense,
    SwitchTransformersTop1Router,
)

from ..memory import ExpertPredictor
from ..utils import ArcherConfig

GPU_IDX_COUNTER = 0


class SwitchTransformersDenseGatedActDense(nn.Module):
    def __init__(self, config: SwitchTransformersConfig):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class SyncSwitchTransformersSparseMLP(nn.Module):
    r"""
    Implementation of the Switch Transformers Sparse MLP module.
    """

    archer_config: ArcherConfig = None
    layer_id: int = None

    def __init__(
        self,
        config: SwitchTransformersConfig,
        expert_class: nn.Module = SwitchTransformersDenseActDense,
    ):
        super().__init__()
        # Step 1: Get the correct router according to its class
        self.router = SwitchTransformersTop1Router(config)

        if config.model_type == "switch_transformers" and config.d_ff == 10240:
            expert_class = SwitchTransformersDenseGatedActDense

        # Step 2: Get the experts
        self.experts = nn.ModuleDict()
        for idx in range(config.num_experts):
            self.experts[f"expert_{idx}"] = expert_class(config)

        # self.archer_engine = None
        self.expert_tensor_ids: Dict[int, int] = None
        # self.expert_dispatcher = None

        self.expert_executor = None
        self.expert_prefetcher = None
        self.expert_predictor: ExpertPredictor = None

    def forward(self, hidden_states):
        # Step 1: Get the router_mask from the router as well as the probabilities
        router_mask, router_probs, router_logits = self.router(hidden_states)
        expert_index = torch.argmax(router_mask, dim=-1)

        # The routers introduced might not always map all the tokens, to a router, which means that some hidden states
        # can be unchanged from one layer to another. That is why the hidden states are cloned before updating only the selected ones.
        next_states = hidden_states.clone()

        # n_tokens = hidden_states.shape[1] * hidden_states.shape[0]
        batch_size = hidden_states.shape[0]
        expert_index = expert_index.reshape(batch_size, -1)
        # for i in range(batch_size):
        #     seq_id = self.seq_id_list[i]
        #     expert_matrix = self.expert_predictor.predict(
        #         seq_id, expert_index[i], self.layer_id
        #     )
        #     self.expert_prefetcher.prefetch_experts(
        #         self.layer_id, expert_matrix
        #     )

        results = self.expert_executor.dispatch_local(
            hidden_states, router_mask, self.layer_id
        )

        for output, _, idx, _ in results:
            token_indices = router_mask[:, :, idx].bool()
            next_states[token_indices] = output.to(next_states.device)

        # for expert_id, expert in self.experts.items():
        #     idx = int(expert_id.split("_")[-1])
        #     token_indices = router_mask[:, :, idx].bool()
        #     if token_indices.any():
        #         next_states[token_indices] = expert(hidden_states[token_indices]).to(next_states.device)

        hidden_states = router_probs * next_states
        return hidden_states, (
            router_logits.to("cuda:0", non_blocking=True),
            expert_index.to("cuda:0", non_blocking=True),
        )
