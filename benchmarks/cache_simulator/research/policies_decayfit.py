"""DecayFit: decayed-rate shadow arbitration between LCNU and S3FIFO.

exp665 (ShadowFit, cumulative scoreboard, M=1.0pp) failed at a single cap: 4096,
where the sustained S3FIFO-LCNU gap (1.09pp) barely exceeds the fixed margin, so
the CUMULATIVE lead crosses the margin only late in the stream and the long LCNU
prefix costs 0.656pp. Diagnosis: a cumulative average drags its own history; the
arbitration signal should track the CURRENT gap, not the all-time average.

DecayFit replaces the cumulative scoreboard with exponentially decayed hit-mass
and access-mass per inner (decay factor lambda = 0.5 ** (1/HALF_LIFE) per
access). The scoreboard ratio estimates the *recent* hit rate of each inner;
the challenger takes over once its decayed rate leads the incumbent's by
MARGIN. Hits are credited under the choice AT ACCESS TIME (the exp665
accounting lesson); `hit_rate`/`effective_hit_rate` report the credited stream
rate, never a final-choice inner's standalone rate.

A-priori parameters (not tuned): HALF_LIFE = 32768 accesses (2^15, ~2.3% of the
30b stream), MARGIN = 0.005 (half of exp665's; leaves ~2x headroom over the
1.09pp gap that broke exp665), RESEGMENT = 4096, initial choice LCNU.

Cost-model disclosure (exp662/663/665 convention): both inners stay resident
(2x metadata footprint); residency-migration cost at switch NOT modeled.
"""

from typing import Dict

from ..policies import Key, build_policy


class DecayFitPolicy:
    """Follow the decayed-rate winner between LCNU and S3FIFO inners."""

    HALF_LIFE = 32768
    MARGIN = 0.005
    RESEGMENT = 4096
    uses_oracle = False

    def __init__(
        self, capacity: int, num_layers: int = 1, prior: float = 1.0 / 16
    ):
        self.capacity = capacity
        self.choice = "lcnu"
        self.switches = 0
        self._since = 0
        self._decay = 0.5 ** (1.0 / self.HALF_LIFE)
        self._inners = {
            "lcnu": build_policy(
                "lcnu", capacity, num_layers=num_layers, prior=prior
            ),
            "s3fifo": build_policy("s3fifo", capacity),
        }
        # decayed [hit_mass, access_mass] per inner
        self._mass = {"lcnu": [0.0, 0.0], "s3fifo": [0.0, 0.0]}
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

    def _select_victim(self, protect):
        return self._inners[self.choice]._select_victim(protect)

    # -- core --------------------------------------------------------------- #
    def on_access(self, key: Key) -> bool:
        hit = False
        d = self._decay
        for name, pol in self._inners.items():
            h = pol.on_access(key)
            m = self._mass[name]
            m[0] = m[0] * d + (1.0 if h else 0.0)
            m[1] = m[1] * d + 1.0
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
        inc = self.choice
        chg = "s3fifo" if inc == "lcnu" else "lcnu"
        ri = self._rate(inc)
        rc = self._rate(chg)
        if ri is not None and rc is not None and rc - ri >= self.MARGIN:
            self.choice = chg
            self.switches += 1

    def _rate(self, name: str):
        hm, am = self._mass[name]
        return hm / am if am > 0 else None

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
