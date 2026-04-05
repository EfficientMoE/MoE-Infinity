# pyright: reportAny=false

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Protocol, runtime_checkable

import torch
import torch.nn.functional as F


@runtime_checkable
class _ExpertExecutorLike(Protocol):
    def dispatch_local(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        router_mask: torch.Tensor,
        router_weights: torch.Tensor,
        router_logits: Optional[torch.Tensor] = None,
    ) -> None: ...

    def wait_dispatch_local(self) -> torch.Tensor: ...


class BatchedExpertDispatch:
    @staticmethod
    def dispatch(
        expert_executor: _ExpertExecutorLike,
        layer_id: int,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        token_offsets: Sequence[int],
    ) -> torch.Tensor:
        if hidden_states.ndim != 2:
            raise ValueError(
                "hidden_states must be rank-2 [total_tokens, hidden_dim]"
            )
        if router_logits.ndim != 2:
            raise ValueError(
                "router_logits must be rank-2 [total_tokens, num_experts]"
            )

        total_tokens, _ = hidden_states.shape
        if router_logits.shape[0] != total_tokens:
            raise ValueError(
                "hidden_states and router_logits must share total_tokens"
            )

        BatchedExpertDispatch._validate_token_offsets(
            token_offsets=token_offsets,
            total_tokens=total_tokens,
        )

        num_experts = router_logits.shape[1]
        if not 1 <= top_k <= num_experts:
            raise ValueError(
                f"top_k must be in [1, {num_experts}], got {top_k}"
            )

        router_mask, router_weights = BatchedExpertDispatch._build_routing(
            router_logits=router_logits,
            top_k=top_k,
            out_dtype=hidden_states.dtype,
        )

        expert_executor.dispatch_local(
            layer_id,
            hidden_states,
            router_mask,
            router_weights,
            router_logits=router_logits,
        )
        output = expert_executor.wait_dispatch_local()

        if output.shape != hidden_states.shape:
            raise ValueError(
                f"expert output shape must match hidden_states shape; got {tuple(output.shape)} vs {tuple(hidden_states.shape)}"
            )
        return output

    @staticmethod
    def split_output(
        output: torch.Tensor,
        token_offsets: Sequence[int],
        seq_lengths: Sequence[int],
    ) -> list[torch.Tensor]:
        if output.ndim != 2:
            raise ValueError("output must be rank-2 [total_tokens, hidden_dim]")

        total_tokens = output.shape[0]
        BatchedExpertDispatch._validate_split_inputs(
            token_offsets=token_offsets,
            seq_lengths=seq_lengths,
            total_tokens=total_tokens,
        )

        split_tensors: list[torch.Tensor] = []
        for seq_idx, seq_len in enumerate(seq_lengths):
            start = token_offsets[seq_idx]
            end = token_offsets[seq_idx + 1]
            if end - start != seq_len:
                raise ValueError(
                    "token_offsets must match seq_lengths at every sequence"
                )
            split_tensors.append(output[start:end])
        return split_tensors

    @staticmethod
    def _build_routing(
        router_logits: torch.Tensor,
        top_k: int,
        out_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        topk_weights, selected_experts = torch.topk(
            routing_weights,
            top_k,
            dim=-1,
        )
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(out_dtype)

        total_tokens, num_experts = router_logits.shape
        router_mask = torch.zeros(
            (total_tokens, num_experts),
            dtype=torch.bool,
            device=router_logits.device,
        )
        _ = router_mask.scatter_(1, selected_experts, True)

        router_weights_mask = torch.zeros(
            (total_tokens, num_experts),
            dtype=out_dtype,
            device=router_logits.device,
        )
        _ = router_weights_mask.scatter_add_(1, selected_experts, topk_weights)

        return router_mask, router_weights_mask

    @staticmethod
    def _validate_token_offsets(
        token_offsets: Sequence[int],
        total_tokens: int,
    ) -> None:
        if len(token_offsets) == 0:
            raise ValueError("token_offsets must contain at least one element")
        if token_offsets[0] != 0:
            raise ValueError("token_offsets must start at 0")
        if token_offsets[-1] != total_tokens:
            raise ValueError(
                f"token_offsets must end at total_tokens ({total_tokens})"
            )
        for idx in range(len(token_offsets) - 1):
            if token_offsets[idx + 1] < token_offsets[idx]:
                raise ValueError(
                    "token_offsets must be monotonically non-decreasing"
                )

    @staticmethod
    def _validate_split_inputs(
        token_offsets: Sequence[int],
        seq_lengths: Sequence[int],
        total_tokens: int,
    ) -> None:
        BatchedExpertDispatch._validate_token_offsets(
            token_offsets=token_offsets,
            total_tokens=total_tokens,
        )
        if len(token_offsets) != len(seq_lengths) + 1:
            raise ValueError(
                "token_offsets must have len(seq_lengths) + 1 elements"
            )


ExpertBatch = BatchedExpertDispatch

__all__ = ["BatchedExpertDispatch", "ExpertBatch"]
