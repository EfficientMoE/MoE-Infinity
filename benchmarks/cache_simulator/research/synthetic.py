"""Synthetic expert-access trace generation for profiling and benchmarking.

No real Qwen3-235B traces ship with the repo, so the accelerated paths are
profiled and timed against *synthetic* access sequences. Sequences are
reproducible from a seed and modelled on the recorder shape
(``94`` layers x ``128`` experts, ``topk=8`` per layer per decode token).

Two generators are provided:

* :func:`synthetic_accesses` -- a flat ``(layer, expert)`` list, with a tunable
  skew so experts are reused with locality (a Zipf-like popularity) rather than
  uniformly, which is closer to real MoE routing.
* :func:`synthetic_records` -- recorder-shaped ``{"topk_ids_of_layer": ...}``
  records to exercise the trace-parsing path.

Every number produced from these traces must be labelled synthetic.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

Key = Tuple[int, int]


def synthetic_accesses(
    num_accesses: int,
    *,
    num_layers: int = 94,
    num_experts: int = 128,
    ignore_layers: int = 3,
    topk: int = 8,
    skew: float = 1.1,
    seed: int = 0,
) -> List[Key]:
    """Return a reproducible flat list of ``(layer, expert)`` accesses.

    Accesses are grouped as ``topk`` consecutive experts per visited layer, and
    layers are visited in ascending order repeatedly (decode-step structure).
    Expert popularity follows a Zipf-like distribution (``skew`` > 1 => more
    locality) so the cache simulator sees realistic reuse rather than a uniform
    stream.
    """

    rng = np.random.default_rng(seed)
    active_layers = np.arange(ignore_layers, num_layers, dtype=np.int64)
    if active_layers.size == 0 or num_accesses <= 0:
        return []

    # Per-layer expert popularity weights (shared across layers for simplicity).
    ranks = np.arange(1, num_experts + 1, dtype=np.float64)
    weights = 1.0 / np.power(ranks, skew)
    weights /= weights.sum()

    accesses: List[Key] = []
    produced = 0
    per_step = active_layers.size * topk
    # Precompute in blocks to keep memory bounded for very large traces.
    while produced < num_accesses:
        remaining = num_accesses - produced
        steps = max(1, min(4096, (remaining + per_step - 1) // per_step))
        # [steps, active_layers, topk] expert draws.
        experts = rng.choice(
            num_experts, size=(steps, active_layers.size, topk), p=weights
        ).astype(np.int64)
        layers = np.broadcast_to(active_layers[None, :, None], experts.shape)
        pairs = np.stack([layers.reshape(-1), experts.reshape(-1)], axis=1)
        take = min(remaining, pairs.shape[0])
        block = pairs[:take]
        accesses.extend(map(tuple, block.tolist()))
        produced += take
    return accesses


def synthetic_records(
    num_steps: int,
    *,
    num_layers: int = 94,
    num_experts: int = 128,
    topk: int = 8,
    seq_len: int = 1,
    skew: float = 1.1,
    seed: int = 0,
) -> List[dict]:
    """Return recorder-shaped records (``topk_ids_of_layer`` arrays).

    Each record is one decode step with a ``[num_layers, seq_len, topk]`` array,
    matching the shape :func:`extract_access_sequence` consumes.
    """

    rng = np.random.default_rng(seed)
    ranks = np.arange(1, num_experts + 1, dtype=np.float64)
    weights = 1.0 / np.power(ranks, skew)
    weights /= weights.sum()

    records: List[dict] = []
    for _ in range(num_steps):
        arr = rng.choice(
            num_experts, size=(num_layers, seq_len, topk), p=weights
        ).astype(np.int64)
        records.append({"topk_ids_of_layer": arr})
    return records
