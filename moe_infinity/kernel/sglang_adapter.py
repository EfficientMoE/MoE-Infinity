from typing import Optional, Tuple

import torch


def sglang_topk_softmax(
    gating_output: torch.Tensor,
    topk: int,
    num_experts: int,
    renormalize: bool = True,
    moe_softcapping: float = 0.0,
    correction_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from sgl_kernel import topk_softmax

    num_tokens = gating_output.shape[0]

    topk_weights = torch.empty(
        (num_tokens, topk), dtype=torch.float32, device=gating_output.device
    )
    topk_ids = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device=gating_output.device
    )

    topk_softmax(
        topk_weights,
        topk_ids,
        gating_output,
        renormalize,
        moe_softcapping,
        correction_bias,
    )

    router_mask = torch.zeros(
        (num_tokens, num_experts),
        dtype=torch.bool,
        device=gating_output.device,
    )
    routing_weights_mask = torch.zeros(
        (num_tokens, num_experts),
        dtype=gating_output.dtype,
        device=gating_output.device,
    )

    topk_ids_long = topk_ids.long()
    router_mask.scatter_(1, topk_ids_long, True)
    routing_weights_mask.scatter_(
        1, topk_ids_long, topk_weights.to(gating_output.dtype)
    )

    return router_mask, routing_weights_mask
