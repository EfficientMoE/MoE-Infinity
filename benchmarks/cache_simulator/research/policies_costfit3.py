"""CostFit3: N-way (3-candidate) generalization of the CostFit barrier.

exp667/668/670 established the self-limiting barrier for TWO candidates
(lcnu vs s3fifo): required_lead = MARGIN + GAMMA * decayed_switch_mass.
exp673 asks whether the same mechanism ARBITRATES THREE candidates —
lcnu, s3fifo, and rothead (itself an adaptive lcnu<->s3fifo switcher with
a 8x faster probe, PROBE_WINDOW 2048 / RESEGMENT 4096) — where the third
candidate can deserve adoption on streams whose rotation regime flips at
timescales between the two probes.

Mechanism: identical to sealed policies_costfit.py (sha 4bfff5bd, unedited)
except _re_arbitrate generalizes to N candidates: the challenger is the
highest decayed-rate NON-INCUMBENT (deterministic name-order tie-break);
the barrier and switch-mass accounting are unchanged.

Cost-model disclosure (series convention): all three inners resident
(3x metadata footprint: lcnu + s3fifo + rothead's own lcnu+s3fifo pair);
rothead's INTERNAL switches are not counted as outer switches; residency-
migration cost at outer switches NOT modeled (control-law surcharge only).

A-priori parameters (identical to exp667 center; none tuned per-cap):
HALF_LIFE = 32768, MARGIN = 0.005, GAMMA = 0.0025, RESEGMENT = 4096,
initial choice LCNU.
"""

from typing import Dict

from ..policies import Key, build_policy


class CostFit3Policy:
    """3-candidate decayed-rate arbitration with self-limiting barrier."""

    HALF_LIFE = 32768
    MARGIN = 0.005
    GAMMA = 0.0025
    RESEGMENT = 4096
    uses_oracle = False

    def __init__(self, capacity: int, num_layers: int = 1, prior: float = 1.0 / 16):
        self.capacity = capacity
        self.choice = "lcnu"
        self.switches = 0
        self.choice_switches = {"lcnu": 0, "s3fifo": 0, "rothead": 0}
        self._since = 0
        self._decay = 0.5 ** (1.0 / self.HALF_LIFE)
        self._inners = {
            "lcnu": build_policy("lcnu", capacity, num_layers=num_layers, prior=prior),
            "s3fifo": build_policy("s3fifo", capacity),
            "rothead": build_policy("rothead", capacity, num_layers=num_layers, prior=prior),
        }
        # decayed [hit_mass, access_mass] per inner; decayed switch mass
        self._mass = {n: [0.0, 0.0] for n in self._inners}
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
        ri = self._rate(inc)
        # challenger: highest decayed-rate non-incumbent, name-order tie-break
        chg, rc = None, None
        for name in sorted(self._inners):
            if name == inc:
                continue
            r = self._rate(name)
            if r is not None and (rc is None or r > rc):
                chg, rc = name, r
        barrier = self.MARGIN + self.GAMMA * self._sw_mass
        if ri is not None and chg is not None and rc is not None and rc - ri >= barrier:
            self.choice = chg
            self.switches += 1
            self.choice_switches[chg] += 1
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
