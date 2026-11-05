"""FOLLOW: a realizable single-resident-set arbiter between LCNU and S3-FIFO.

``costfit`` (exp667) runs LCNU and S3-FIFO side by side at full capacity and
credits each access to whichever inner is currently chosen. That is an
optimistic bound: on a switch the real expert set would have to become the
other inner's set instantly (2x residency, migration not modeled).

FOLLOW keeps ONE real resident set ``R`` of ``capacity`` expert slots and uses
the CostFit control law, unmodified, purely as metadata: both inners become
ghosts (keys only, no expert weights). On a real miss with ``R`` full, FOLLOW
evicts a resident key that the current leader ghost does NOT hold
(``D = R - leader``). ``R`` converges to the leader's set lazily through
demand misses only: no extra fetches, no bulk migration at a switch.

``D`` order (deterministic): FIFO of "became out-of-leader"; a real hit on a
key in ``D`` moves it to the end. Victim = first key of ``D``. If ``D`` is
empty while ``R`` is full, the leader holds every resident key; with LCNU and
S3-FIFO as the inners (both admit every miss) this does not occur, but the
policy falls back to the LRU key of ``R`` and counts it in ``fallbacks``.

Engine cost: one LCNU and one S3-FIFO metadata update per access plus O(1)
ordered-set operations; ghost state is keys only.

Results on three full real decode traces (granite3.1-moe-3b, qwen3-30b-a3b
hi/lo, 27 capacity cells): within 0.5pp of the best of the 13 shipped
policies in 27/27 cells (LCNU default: 17/27), never more than 0.10pp below
LCNU, and within 0.02pp of the optimistic ``costfit`` bound. See
``results_follow_real.md``.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Tuple

from .policies_costfit import CostFitPolicy

Key = Tuple[int, int]


class FollowPolicy:
    """LCNU/S3-FIFO arbiter with one real resident set (see module docstring)."""

    uses_oracle = False

    def __init__(
        self, capacity: int, num_layers: int = 1, prior: float = 1.0 / 16
    ):
        self.capacity = capacity
        self.cf = CostFitPolicy(capacity, num_layers=num_layers, prior=prior)
        self._real: "OrderedDict[Key, None]" = OrderedDict()  # LRU order
        self._out: "OrderedDict[Key, None]" = OrderedDict()  # R - leader
        self.hits = 0
        self.misses = 0
        self.fallbacks = 0
        self.d_mass = 0  # sum of |D| over accesses (convergence-lag diagnostic)
        for name, pol in self.cf._inners.items():
            pol._evict_callback = self._make_evict_callback(name)  # type: ignore[attr-defined]

    def _make_evict_callback(self, name: str):
        def callback(key: Key) -> None:
            # The leader ghost dropped ``key``: if it is really resident it
            # becomes an eviction candidate (end of the FIFO).
            if name == self.cf.choice and key in self._real:
                self._out[key] = None
                self._out.move_to_end(key)

        return callback

    # -- CachePolicy surface ------------------------------------------------ #
    @property
    def resident(self):
        return self._real.keys()

    @property
    def keys(self) -> set:
        return set(self._real)

    @property
    def switches(self) -> int:
        return self.cf.switches

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    @property
    def effective_hit_rate(self) -> float:
        # Demand-only (no prefetch), so every hit is a real resident hit.
        return self.hit_rate

    def bind_trace(self, accesses) -> None:  # online policy: nothing to bind
        return None

    def on_access(self, key: Key) -> bool:
        before = self.cf.choice
        self.cf.on_access(key)  # ghosts + unmodified CostFit control law
        if self.capacity <= 0:
            # Match every other policy in the registry: nothing can ever be
            # resident, so count the miss and keep the ghosts consistent.
            self.misses += 1
            return False
        leader = self.cf._inners[self.cf.choice].resident
        real = self._real
        if self.cf.choice != before:  # switch: rebuild D against new leader
            self._out = OrderedDict((r, None) for r in real if r not in leader)
        out = self._out
        if key in real:
            self.hits += 1
            real.move_to_end(key)
            hit = True
        else:
            self.misses += 1
            if len(real) >= self.capacity:
                if out:
                    victim = next(iter(out))
                    del out[victim]
                else:
                    victim = next(iter(real))
                    self.fallbacks += 1
                del real[victim]
            real[key] = None
            hit = False
        if key in leader:
            out.pop(key, None)
        else:
            out[key] = None
            out.move_to_end(key)
        self.d_mass += len(out)
        return hit

    def check_invariants(self) -> None:
        """Assert ``|R| <= C`` and ``D == R - leader`` (O(C); for tests)."""
        leader = self.cf._inners[self.cf.choice].resident
        assert len(self._real) <= self.capacity
        assert set(self._out) == {r for r in self._real if r not in leader}
