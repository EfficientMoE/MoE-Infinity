from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class WorkloadCase:
    input_ids: tuple[int, ...]


def deterministic_workload(
    seed: int,
    cases: int,
    min_tokens: int,
    max_tokens: int,
    vocab_size: int = 32000,
):
    if (
        cases < 0
        or min_tokens <= 0
        or max_tokens < min_tokens
        or vocab_size <= 1
    ):
        raise ValueError("invalid deterministic workload bounds")
    rng = random.Random(seed)
    return tuple(
        WorkloadCase(
            tuple(
                rng.randrange(1, vocab_size)
                for _ in range(rng.randint(min_tokens, max_tokens))
            )
        )
        for _ in range(cases)
    )
