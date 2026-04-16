from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch


class _SamplingParamsLike(Protocol):
    temperature: float
    top_k: int
    top_p: float
    logprobs: int


@dataclass
class SamplerOutput:
    token_ids: torch.Tensor
    token_logprobs: list[float | None] | None = None
    top_logprobs: list[dict[int, float] | None] | None = None


@dataclass
class Sampler:
    def sample(
        self,
        logits: torch.Tensor,
        sampling_params: Sequence[_SamplingParamsLike],
    ) -> SamplerOutput:
        if logits.dim() != 2:
            raise ValueError("logits must have shape [num_seqs, vocab_size]")
        if logits.size(0) != len(sampling_params):
            raise ValueError("sampling_params must match batch size")
        if logits.size(1) <= 0:
            raise ValueError("vocab_size must be positive")

        sampled = torch.empty(
            logits.size(0),
            dtype=torch.long,
            device=logits.device,
        )
        requested_logprobs = any(
            params.logprobs > 0 for params in sampling_params
        )
        token_logprobs: list[float | None] | None = None
        top_logprobs: list[dict[int, float] | None] | None = None
        if requested_logprobs:
            token_logprobs = [None for _ in range(logits.size(0))]
            top_logprobs = [None for _ in range(logits.size(0))]

        for idx, params in enumerate(sampling_params):
            row = logits[idx]
            temperature = params.temperature
            filtered = row

            if temperature == 0:
                sampled[idx] = torch.argmax(row).to(dtype=torch.long)
            elif temperature < 0:
                raise ValueError("temperature must be non-negative")

            else:
                filtered = row / temperature
                filtered = self._apply_top_k(filtered, params.top_k)
                filtered = self._apply_top_p(filtered, params.top_p)

                probs = torch.softmax(filtered, dim=-1)
                sampled[idx] = torch.multinomial(probs, num_samples=1).squeeze(
                    0
                )

            if not requested_logprobs or params.logprobs <= 0:
                continue

            logprobs = torch.log_softmax(filtered, dim=-1)
            sampled_token_id = int(sampled[idx].item())
            if token_logprobs is not None:
                token_logprobs[idx] = float(logprobs[sampled_token_id].item())

            if top_logprobs is not None:
                k = min(int(params.logprobs), logprobs.numel())
                top_values, top_indices = torch.topk(logprobs, k=k)
                top_logprobs[idx] = {
                    int(token_id.item()): float(token_logprob.item())
                    for token_id, token_logprob in zip(top_indices, top_values)
                }

        return SamplerOutput(
            token_ids=sampled,
            token_logprobs=token_logprobs,
            top_logprobs=top_logprobs,
        )

    @staticmethod
    def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        if top_k <= 0 or top_k >= logits.numel():
            return logits

        topk_indices = torch.topk(logits, k=top_k).indices
        filtered = torch.full_like(logits, float("-inf"))
        filtered[topk_indices] = logits[topk_indices]
        return filtered

    @staticmethod
    def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        if top_p >= 1.0:
            return logits
        if top_p <= 0:
            keep_index = torch.argmax(logits)
            filtered = torch.full_like(logits, float("-inf"))
            filtered[keep_index] = logits[keep_index]
            return filtered

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        remove_mask = cumulative_probs > top_p
        remove_mask[0] = False

        filtered_sorted = sorted_logits.masked_fill(remove_mask, float("-inf"))
        filtered = torch.full_like(logits, float("-inf"))
        filtered[sorted_indices] = filtered_sorted
        return filtered


__all__ = ["Sampler", "SamplerOutput"]
