import nvtx
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP


class Qwen3MoEBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # gating
        self.gate = nn.Linear(
            config.hidden_size, config.num_experts, bias=False
        )
        self.experts = nn.ModuleList(
            [
                Qwen3MoeMLP(
                    config, intermediate_size=config.moe_intermediate_size
                )
                for _ in range(self.num_experts)
            ]
        )

    @nvtx.annotate("Qwen3MoEBlock", color="blue")
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # print(f"hidden_states shape: {hidden_states.shape}")
        # print(f"routing_weights shape: {routing_weights.shape}")

        router_mask = F.one_hot(selected_experts, num_classes=self.num_experts)
        routing_weights_mask = (
            routing_weights[:, :, None] * router_mask
        ).permute(0, 2, 1)
        routing_weights_mask = torch.sum(routing_weights_mask, dim=-1)
        router_mask = router_mask.permute(0, 2, 1)
        # print(f"router_mask shape: {router_mask.shape}")
        # print(f"routing_weights_mask shape: {routing_weights_mask.shape}")

        # use logical or to merge last dimension
        for i in range(self.top_k):
            router_mask[:, :, 0] = torch.logical_or(
                router_mask[:, :, 0], router_mask[:, :, i]
            )
        router_mask = router_mask[:, :, 0]

        results = self.expert_executor.dispatch_local(
            hidden_states, router_mask, self.layer_id
        )
        for output, _, idx, _ in results:
            token_indices = router_mask[:, idx].bool()
            final_hidden_states[token_indices, :] += (
                output.to(routing_weights_mask.device)
                * routing_weights_mask[token_indices, idx][:, None]
            )

        final_hidden_states = final_hidden_states.view(
            batch_size, sequence_length, hidden_dim
        )

        return final_hidden_states

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = (
                expert_layer(current_state) * routing_weights[top_x, idx, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states, router_logits
