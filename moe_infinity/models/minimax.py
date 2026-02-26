# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

# MoE block replacement for MiniMax-M2.5 (model_type=minimax_m2).
#
# Architecture facts (62 MoE layers, 256 total experts, 8 active/token):
#   - Sigmoid routing (not softmax)
#   - Gate: nn.Linear(hidden_size, num_local_experts, bias=False)
#   - Expert MLP class name and exact param paths MUST be confirmed by
#     inspecting the downloaded modeling_minimax_m2.py source.
#   - Expected param path: model.layers.{i}.feed_forward.experts.{j}.*
#     (see moe_infinity/utils/hf_config.py parse_expert_id)

import nvtx
import torch
import torch.nn as nn

import moe_infinity._store as prefetch_lib  # noqa: F401 (imported for type hints)


class MiniMaxMoEBlock(nn.Module):
    """
    MoE block for MiniMax-M2.5.

    This class replaces the trust_remote_code MoE block (class name TBD) at
    model-load time.  The actual MLP sub-module class and attribute names are
    read from the *already-imported* modeling_minimax_m2 module so that the
    experts are constructed identically to the original architecture.

    TODO: After downloading MiniMaxAI/MiniMax-M2.5 run::

        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained("MiniMaxAI/MiniMax-M2.5",
                                          trust_remote_code=True)
        import sys, pprint
        # find the model module
        mod = [v for k, v in sys.modules.items()
               if "modeling_minimax" in k][0]
        pprint.pprint([n for n in dir(mod)])  # find MoE + MLP class names

    Then update the ``__init__`` below accordingly.
    """

    def __init__(self, config):
        super().__init__()
        # MoE hyper-params (confirmed from public MiniMax-M2.5 config):
        #   num_local_experts = 256, num_experts_per_tok = 8
        self.num_experts = config.num_local_experts  # 256
        self.top_k = config.num_experts_per_tok  # 8

        # Gate projection: hidden_size → num_experts
        self.gate = nn.Linear(
            config.hidden_size, config.num_local_experts, bias=False
        )

        # Expert MLP modules.
        # TODO: Replace nn.Identity with the actual MLP class once the model
        # code is confirmed.  Example:
        #   from <model_module> import MiniMaxM2MLP
        #   self.experts = nn.ModuleList([
        #       MiniMaxM2MLP(config) for _ in range(self.num_experts)
        #   ])
        self.experts = nn.ModuleList(
            [nn.Identity() for _ in range(self.num_experts)]
        )

        # Set by model_offload.py after model creation:
        self.expert_executor = None
        self.lib = None
        self.layer_id = None

    @nvtx.annotate("MiniMaxPrepare", color="green")
    def __prepare_expert_route(self, hidden_states):
        router_logits = self.gate(hidden_states)
        router_mask, routing_weights_mask = self.lib.topk_softmax(
            router_logits
        )
        return router_logits, router_mask, routing_weights_mask

    @nvtx.annotate("MiniMaxMoEBlock", color="green")
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits, router_mask, routing_weights_mask = (
            self.__prepare_expert_route(hidden_states)
        )

        self.expert_executor.dispatch_local(
            self.layer_id, hidden_states, router_mask, routing_weights_mask
        )
        final_hidden_states = self.expert_executor.wait_dispatch_local()

        final_hidden_states = final_hidden_states.view(
            batch_size, sequence_length, hidden_dim
        ).to(hidden_states.dtype)

        return final_hidden_states, router_logits
