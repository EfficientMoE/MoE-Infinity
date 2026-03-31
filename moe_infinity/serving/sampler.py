from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch


class _SamplingParamsLike(Protocol):
    temperature: float
    top_k: int
    top_p: float


@dataclass
class Sampler:
    def sample(
        self,
        logits: torch.Tensor,
        sampling_params: Sequence[_SamplingParamsLike],
    ) -> torch.Tensor:
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

        for idx, params in enumerate(sampling_params):
            row = logits[idx]
            temperature = params.temperature

            if temperature == 0:
                sampled[idx] = torch.argmax(row).to(dtype=torch.long)
                continue
            if temperature < 0:
                raise ValueError("temperature must be non-negative")

            filtered = row / temperature
            filtered = self._apply_top_k(filtered, params.top_k)
            filtered = self._apply_top_p(filtered, params.top_p)

            probs = torch.softmax(filtered, dim=-1)
            sampled[idx] = torch.multinomial(probs, num_samples=1).squeeze(0)

        return sampled

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


__all__ = ["Sampler"]
