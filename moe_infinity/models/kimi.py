# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

# MoE block replacement for Kimi-VL-A3B-Thinking (model_type=kimi_vl).
#
# Architecture facts (27 language-decoder layers, DeepSeek-V3-like):
#   - n_shared_experts = 2, n_routed_experts = 64, num_experts_per_tok = 6
#   - Sigmoid routing (similar to DeepSeek-V3)
#   - MoE params live in config.text_config (VL model)
#   - Expert MLP class name and exact param paths MUST be confirmed by
#     inspecting the downloaded Kimi-VL modeling file.
#   - Expected param path:
#     language_model.model.layers.{i}.mlp.experts.{j}.*
#     (see moe_infinity/utils/hf_config.py parse_expert_id)

import nvtx
import torch
import torch.nn as nn
import torch.nn.functional as F

import moe_infinity._store as prefetch_lib  # noqa: F401 (imported for type hints)


class KimiMoEBlock(nn.Module):
    """
    MoE block for moonshotai/Kimi-VL-A3B-Thinking.

    This class replaces the trust_remote_code MoE block (class name TBD) at
    model-load time.  It follows the DeepSeek-V3 pattern for shared + routed
    experts.

    TODO: After downloading moonshotai/Kimi-VL-A3B-Thinking run::

        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained("moonshotai/Kimi-VL-A3B-Thinking",
                                          trust_remote_code=True)
        import sys, pprint
        mod = [v for k, v in sys.modules.items()
               if "modeling_kimi" in k.lower() or "kimi" in k.lower()][0]
        pprint.pprint([n for n in dir(mod)])  # find MoE + MLP class names

    Then update the ``__init__`` below with the correct MLP class and
    verify the ``model_offload.py`` ``_patch_trust_remote_code_moe`` call
    uses the right class name.
    """

    def __init__(self, config):
        super().__init__()
        # Kimi-VL MoE params are nested under config.text_config
        text_cfg = getattr(config, "text_config", config)

        self.num_experts_per_tok = text_cfg.num_experts_per_tok  # 6
        self.n_routed_experts = text_cfg.n_routed_experts  # 64
        self.n_shared_experts = getattr(text_cfg, "n_shared_experts", None)

        # Gate projection: hidden_size → n_routed_experts
        self.gate = nn.Linear(
            text_cfg.hidden_size, text_cfg.n_routed_experts, bias=False
        )

        # Routed expert MLP modules.
        # TODO: Replace nn.Identity with the actual MLP class once confirmed.
        self.experts = nn.ModuleList(
            [nn.Identity() for _ in range(self.n_routed_experts)]
        )

        # Shared experts (always active, like DeepSeek-V3).
        # TODO: Replace nn.Identity with the actual shared-expert MLP class.
        if self.n_shared_experts is not None:
            self.shared_experts = nn.Identity()

        # Set by model_offload.py after model creation:
        self.expert_executor = None
        self.lib = None
        self.layer_id = None

    @nvtx.annotate("KimiPrepare", color="purple")
    def __prepare_expert_route(self, hidden_states):
        router_logits = self.gate(hidden_states.view(-1, hidden_states.shape[-1]))

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )

        B, E = routing_weights.shape[0], self.n_routed_experts
        router_mask = torch.zeros(
            B, E, dtype=torch.bool, device=selected_experts.device
        )
        router_mask.scatter_(1, selected_experts, True)

        routing_weights_mask = torch.zeros(
            B, E, dtype=routing_weights.dtype, device=routing_weights.device
        )
        routing_weights_mask.scatter_add_(1, selected_experts, routing_weights)

        return router_mask, routing_weights_mask

    @nvtx.annotate("KimiMoEBlock", color="purple")
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
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

        if self.n_shared_experts is not None:
            final_hidden_states = final_hidden_states + self.shared_experts(
                identity
            )

        return final_hidden_states
