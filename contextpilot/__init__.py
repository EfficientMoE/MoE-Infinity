from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class ContextPilot:
    use_gpu: bool = False

    def optimize(
        self, contexts: Iterable[str], query: str
    ) -> list[dict[str, str]]:
        normalized_contexts = [str(context) for context in contexts or []]
        ordered_contexts = self._order_contexts(normalized_contexts)

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": "ContextPilot message reconstruction (CPU-only).",
            }
        ]

        for context in ordered_contexts:
            messages.append({"role": "user", "content": context})

        if query:
            messages.append({"role": "user", "content": str(query)})

        return messages

    @staticmethod
    def _order_contexts(contexts: list[str]) -> list[str]:
        if len(contexts) <= 1:
            return list(contexts)

        ranked = sorted(
            enumerate(contexts),
            key=lambda item: (-ContextPilot._prefix_score(item[1]), item[0]),
        )
        return [context for _, context in ranked]

    @staticmethod
    def _prefix_score(text: str) -> int:
        prefix = 0
        for char in text:
            if char.isspace():
                break
            prefix += 1
        return prefix
