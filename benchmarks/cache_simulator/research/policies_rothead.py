"""Rotation-adaptive eviction (exp662): online probe switching LCNU <-> S3FIFO.

Mechanism (exp659/660): under fast hot-set rotation, global recency (S3FIFO)
tracks better; under slow rotation, layer-fair LCNU dominates. This policy
estimates rotation speed ONLINE from the live access stream — adjacent-window
set-Jaccard over a recent-access buffer, re-evaluated every RESEGMENT
accesses — and attributes hits per segment to the probe-chosen inner policy.

Modeling assumptions (disclosed in exp662 PREREG):
- Both inner policies are simulated fully in parallel at the same capacity;
  the adaptive wrapper attributes each access's hit/miss to the currently
  chosen inner. Residency-migration cost at switches is NOT modeled; switch
  counts are reported so readers can judge (few switches => small cost).
- Probe: J(1) = Jaccard(second-half set, first-half set) of the last
  2*PROBE_WINDOW accesses. Threshold THRESH_J1 = midpoint of the two
  full-stream J(1) values measured in exp660 (0.547 fast / 0.634 slow).
  Initial segment runs LCNU (the cross-model default per exp653).
"""

from __future__ import annotations

from collections import deque
from typing import Tuple

from benchmarks.cache_simulator.policies import (
    CachePolicy,
    LayerCycleNextUsePolicy,
    S3FIFOPolicy,
)

Key = Tuple[int, int]

PROBE_WINDOW = 2048  # accesses per probe window
RESEGMENT = 4096  # re-evaluate the probe every N accesses
THRESH_J1 = 0.59  # exp660 J(1) midpoint: below => fast rotation


class RotHeadPolicy(CachePolicy):
    """Online rotation-adaptive switch between LCNU and S3FIFO."""

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
        self.buffer: deque = deque(maxlen=2 * PROBE_WINDOW)
        self.choice = "lcnu"
        self.switches = 0
        self._since_probe = 0

    def _probe_choice(self) -> str:
        if len(self.buffer) < 2 * PROBE_WINDOW:
            return self.choice
        half = PROBE_WINDOW
        first = set(list(self.buffer)[:half])
        second = set(list(self.buffer)[half:])
        if not first and not second:
            return self.choice
        j1 = len(first & second) / len(first | second)
        return "s3fifo" if j1 < THRESH_J1 else "lcnu"

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
