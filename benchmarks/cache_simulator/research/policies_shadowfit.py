"""ShadowFit: direct per-policy online scoring via parallel inner policies.

exp661 (rank probe), exp662 (rotation J(1)) and exp663/664 (hot-set coverage
quantiles) each failed to *predict* the LCNU->S3FIFO crossover. ShadowFit
sidesteps prediction: both inner policies are maintained in parallel at the
full cache capacity, every access is scored against both, and the wrapper
attributes demand hits to the currently chosen inner. The choice follows the
CUMULATIVE hit-rate leader with a hysteresis margin M: the challenger takes
over only once its cumulative rate exceeds the incumbent's by >= M.

A-priori parameters (not tuned): MARGIN = 0.01 (1.0pp), RESEGMENT = 4096
accesses between scoreboard evaluations, initial choice LCNU (exp662
convention). Cumulative (not windowed) scoring suppresses the burst-driven
oscillation that produced exp662's 48-switch granite run.

Cost-model disclosure (same convention as exp662/663 wrappers): both inners
stay resident, so the wrapper models a 2x metadata footprint and does NOT
model residency-migration cost at switch time.
"""

from collections import deque
from typing import Dict, Tuple

from ..policies import Key, build_policy


class ShadowFitPolicy:
    """Follow the cumulative-rate winner between LCNU and S3FIFO inners."""

    MARGIN = 0.01
    RESEGMENT = 4096
    uses_oracle = False

    def __init__(
        self, capacity: int, num_layers: int = 1, prior: float = 1.0 / 16
    ):
        self.capacity = capacity
        self.choice = "lcnu"
        self.switches = 0
        self._since = 0
        self._inners = {
            "lcnu": build_policy(
                "lcnu", capacity, num_layers=num_layers, prior=prior
            ),
            "s3fifo": build_policy("s3fifo", capacity),
        }
        self._cum = {"lcnu": [0, 0], "s3fifo": [0, 0]}  # [hits, total]
        self._credited = [0, 0]  # hits credited under the choice AT ACCESS TIME

    # -- CachePolicy surface ------------------------------------------------ #
    @property
    def resident(self):
        return self._inners[self.choice].resident

    @property
    def keys(self) -> set:
        return self._inners[self.choice].keys

    def bind_trace(self, accesses) -> None:  # pragma: no cover - unused online
        for p in self._inners.values():
            if hasattr(p, "bind_trace"):
                p.bind_trace(accesses)

    def _select_victim(self):
        return self._inners[self.choice]._select_victim()

    # -- core --------------------------------------------------------------- #
    def on_access(self, key: Key) -> bool:
        hit = False
        for name, pol in self._inners.items():
            h = pol.on_access(key)
            self._cum[name][0] += 1 if h else 0
            self._cum[name][1] += 1
            if name == self.choice:
                hit = h
                self._credited[0] += 1 if h else 0
                self._credited[1] += 1
        self._since += 1
        if self._since >= self.RESEGMENT:
            self._since = 0
            self._re_arbitrate()
        return hit

    def _re_arbitrate(self) -> None:
        rates = self._scoreboard()
        inc, chg = self.choice, "s3fifo" if self.choice == "lcnu" else "lcnu"
        if rates[chg] - rates[inc] >= self.MARGIN:
            self.choice = chg
            self.switches += 1

    def _scoreboard(self):
        out = {}
        for name, (h, t) in self._cum.items():
            out[name] = h / t if t else 0.0
        return out

    @property
    def hits(self) -> int:
        return self._credited[0]

    @property
    def misses(self) -> int:
        return self._credited[1] - self._credited[0]

    @property
    def hit_rate(self) -> float:
        h, t = self._credited
        return h / t if t else 0.0

    @property
    def effective_hit_rate(self) -> float:
        h, t = self._credited
        return h / t if t else 0.0
