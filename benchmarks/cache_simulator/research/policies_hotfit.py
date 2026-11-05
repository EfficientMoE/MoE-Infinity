"""Hot-set-fit adaptive eviction (exp663): capacity-aware LCNU <-> S3FIFO.

exp662 refuted capacity-blind rotation probing (RotHead): the probe locked
onto S3FIFO at every capacity — near-optimal at >=10E, catastrophic below.
Its mechanism conclusion: arbitration must feed capacity (hot-set fit) in.

HotFit implements that: over a PROBE_WINDOW-access window, re-evaluated every
RESEGMENT accesses, compute the window frequency distribution and N_hot =
the smallest set of hottest keys covering >= COVERAGE of the window's
accesses. Arbitrate: capacity >= ALPHA * N_hot -> S3FIFO (recency-friendly;
hot set fits), else -> LCNU (layer-fair allocation; hot set overflows).

Hyperparameters COVERAGE=0.80 and ALPHA=1.0 are round-number a-priori
defaults, not tuned (frozen in the exp663 prereg before the sweep).

Same wrapper structure as exp662 RotHead (parallel inner simulation, per-
segment hit attribution, no residency-migration cost at switches, switch
counts reported).
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Tuple

from benchmarks.cache_simulator.policies import (
    CachePolicy,
    LayerCycleNextUsePolicy,
    S3FIFOPolicy,
)

Key = Tuple[int, int]

PROBE_WINDOW = 2048  # accesses per frequency window
RESEGMENT = 4096  # re-evaluate the probe every N accesses
COVERAGE = 0.80  # fraction of window accesses the hot set must cover
ALPHA = 1.0  # capacity multiplier on N_hot


class HotFitPolicy(CachePolicy):
    """Capacity-aware hot-set-fit switch between LCNU and S3FIFO."""

    uses_oracle = False

    def _select_victim(self, protect):
        # Never called: on_access is fully overridden and the wrapper keeps
        # no resident set of its own. Delegate defensively to the active
        # inner policy if something ever routes here.
        inner = self.lcnu if self.choice == "lcnu" else self.s3fifo
        return inner._select_victim(protect)

    def __init__(self, capacity: int, num_layers: int = 1, prior: float = 0.05):
        super().__init__(capacity)
        self.lcnu = LayerCycleNextUsePolicy(
            capacity, num_layers=num_layers, prior=prior
        )
        self.s3fifo = S3FIFOPolicy(capacity)
        self.buffer: deque = deque(maxlen=PROBE_WINDOW)
        self.choice = "lcnu"
        self.switches = 0
        self._since_probe = 0
        self.last_n_hot = None

    def _probe_choice(self) -> str:
        if len(self.buffer) < PROBE_WINDOW:
            return self.choice
        counts = sorted(Counter(self.buffer).values(), reverse=True)
        need = COVERAGE * PROBE_WINDOW
        cum = 0
        n_hot = len(counts)
        for i, c in enumerate(counts, start=1):
            cum += c
            if cum >= need:
                n_hot = i
                break
        self.last_n_hot = n_hot
        return "s3fifo" if self.capacity >= ALPHA * n_hot else "lcnu"

    def on_access(self, key: Key) -> bool:
        lc_h0, s3_h0 = self.lcnu.hits, self.s3fifo.hits
        self.lcnu.on_access(key)
        self.s3fifo.on_access(key)
        self.buffer.append(key)
        self._since_probe += 1
        if self._since_probe >= RESEGMENT:
            self._since_probe = 0
            new = self._probe_choice()
            if new != self.choice:
                self.switches += 1
                self.choice = new
        inner = self.lcnu if self.choice == "lcnu" else self.s3fifo
        base = lc_h0 if self.choice == "lcnu" else s3_h0
        hit = inner.hits > base
        if hit:
            self.hits += 1
        else:
            self.misses += 1
        self._index += 1
        return hit
