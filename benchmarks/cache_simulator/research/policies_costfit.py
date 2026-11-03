"""CostFit: decayed-rate shadow arbitration with self-limiting switch cost.

exp665 (cumulative scoreboard, 1.0pp) failed at 4096: margin too large ->
late switch. exp666 (decayed scoreboard, 0.5pp) failed at 1024: margin too
small -> oscillation (8 switches). Fixed margins are refuted in both
directions. CostFit replaces the FIXED margin with a SELF-LIMITING barrier:

    required_lead = MARGIN + GAMMA * (decayed recent-switch count)

The switch count is a decayed mass (same per-access decay as the scoreboard):
each switch adds 1; the mass decays with the scoreboard half-life. A genuine
regime shift (first switch from cold start, mass ~0) faces only the base
margin — so 4096's sustained 1.09pp gap still switches quickly. Repeated
flipping (1024's local inversions) raises its own barrier: 1 recent switch
-> 0.75pp, 2 -> ~1.0pp, 4 -> ~1.5pp, damping thrash while keeping one-time
adaptation free.

Scoreboard and accounting are exactly exp666's (decayed hit/access mass per
inner, half-life 32768; hits credited under the choice AT ACCESS TIME;
`hit_rate`/`effective_hit_rate` report the credited stream rate).

A-priori parameters (none tuned per-cap): HALF_LIFE = 32768, MARGIN = 0.005,
GAMMA = 0.0025 (half the margin per effective recent switch), RESEGMENT =
4096, initial choice LCNU.

Cost-model disclosure (series convention): both inners resident (2x metadata
footprint); residency-migration cost at switch NOT modeled (the barrier is a
control-law surcharge, not a measured migration cost).
"""

from typing import Dict

from ..policies import Key, build_policy


class CostFitPolicy:
    """Decayed-rate shadow arbitration with self-limiting switch barrier."""

    HALF_LIFE = 32768
    MARGIN = 0.005
    GAMMA = 0.0025
    RESEGMENT = 4096
    uses_oracle = False

    def __init__(self, capacity: int, num_layers: int = 1, prior: float = 1.0 / 16):
        self.capacity = capacity
        self.choice = "lcnu"
        self.switches = 0
        self._since = 0
        self._decay = 0.5 ** (1.0 / self.HALF_LIFE)
        self._inners = {
            "lcnu": build_policy("lcnu", capacity, num_layers=num_layers, prior=prior),
            "s3fifo": build_policy("s3fifo", capacity),
        }
        # decayed [hit_mass, access_mass] per inner; decayed switch mass
        self._mass = {"lcnu": [0.0, 0.0], "s3fifo": [0.0, 0.0]}
        self._sw_mass = 0.0
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
        barrier = self.MARGIN + self.GAMMA * self._sw_mass
        if ri is not None and rc is not None and rc - ri >= barrier:
            self.choice = chg
            self.switches += 1
            self._sw_mass = self._sw_mass * self._decay + 1.0

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
