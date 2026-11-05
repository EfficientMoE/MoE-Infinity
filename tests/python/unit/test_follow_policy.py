"""Independent behavioural tests for the FOLLOW arbiter (``follow``).

Subject: ``benchmarks/cache_simulator/research/policies_follow.py``.

Claims under test (module docstring):
  * one real resident set ``R`` of ``capacity`` keys; CostFit (LCNU and
    S3-FIFO ghosts) runs unmodified as metadata only;
  * on a real miss with ``R`` full the victim is the first key of
    ``D = R - leader`` in "became out-of-leader" FIFO order (a real hit on a
    ``D`` key moves it to the end); if ``D`` is empty the LRU key of ``R`` is
    evicted and counted in ``fallbacks``.

Written independently of the implementation. Section 5 re-derives the
bookkeeping naively (recompute ``R - leader`` every access, order candidates
by explicit timestamps) instead of reusing FOLLOW's ordered dicts.

Semantics pinned here (derived from the code, not the docstring):
  * with the shipped inners every accessed key is resident in the leader
    ghost after the access (both admit every miss), so the "real hit on a
    ``D`` key moves it to the end" branch only runs with a non-admitting
    leader; section 6 exercises it with an LCNU ``bypass="always"`` leader;
  * at a switch ``D`` is rebuilt in ``R``'s LRU order (ties of the
    simultaneous "became out-of-leader" event go to the least recent key).
"""

from __future__ import annotations

import functools
import itertools
import random
from typing import Callable, Dict, List, Optional, Set, Tuple

import pytest

from benchmarks.cache_simulator.policies import (
    GEOMETRY_POLICIES,
    ORACLE_POLICIES,
    POLICY_FACTORIES,
    LayerCycleNextUsePolicy,
    build_policy,
    simulate,
)
from benchmarks.cache_simulator.research.policies_costfit import CostFitPolicy
from benchmarks.cache_simulator.research.policies_follow import FollowPolicy

Key = Tuple[int, int]
Tune = Optional[Callable[[CostFitPolicy], None]]


# =========================================================================== #
# Trace generators (seeded, deterministic)
# =========================================================================== #
def moe_trace(
    seed: int,
    layers: int,
    experts: int,
    topk: int,
    tokens: int,
    shift_at: int,
    zipf_a: float,
) -> List[Key]:
    """Layer-cyclic decode trace: zipf-skewed top-k routing per layer visit.

    At token ``shift_at`` the expert popularity ranking is re-permuted (a
    distribution shift), which is what makes CostFit change its leader.
    """
    rng = random.Random(seed)
    perms = []
    for _ in range(2):
        perm = list(range(experts))
        rng.shuffle(perm)
        perms.append(perm)
    cum = list(
        itertools.accumulate(1.0 / (i + 1) ** zipf_a for i in range(experts))
    )
    out: List[Key] = []
    for t in range(tokens):
        perm = perms[t >= shift_at]
        for layer in range(layers):
            picks: List[int] = []
            while len(picks) < topk:
                e = rng.choices(perm, cum_weights=cum)[0]
                if e not in picks:
                    picks.append(e)
            out.extend((layer, e) for e in sorted(picks))
    return out


def small_trace(seed: int, layers: int, experts: int, topk: int, n: int):
    """Short uniform layer-cyclic trace (for the brute-force reference)."""
    rng = random.Random(seed)
    out: List[Key] = []
    while len(out) < n:
        for layer in range(layers):
            for e in sorted(rng.sample(range(experts), topk)):
                out.append((layer, e))
    return out[:n]


# (layers, experts, topk, zipf_a, capacity, shift_at, seed); every config
# makes CostFit switch at its default (shipped) parameters.
SWITCH_CONFIGS = [
    (4, 32, 2, 1.0, 24, 100, 0),
    (4, 32, 2, 1.2, 20, 300, 1),
    (4, 16, 2, 1.5, 16, 300, 0),
    (8, 16, 2, 1.2, 32, 300, 0),
]
# Same generator, shipped parameters, CostFit never leaves LCNU.
NO_SWITCH_CONFIG = (8, 16, 2, 1.2, 32, 100, 1)


def _cfg_trace(cfg) -> Tuple[List[Key], int, int, float]:
    layers, experts, topk, a, cap, shift, seed = cfg
    tokens = 9000 // (layers * topk) + 1  # > 2 * RESEGMENT accesses
    trace = moe_trace(seed, layers, experts, topk, tokens, shift, a)
    return trace, layers, cap, topk / experts


def thrash(cf: CostFitPolicy) -> None:
    """Make CostFit re-arbitrate often and switch on ties (many switches)."""
    cf.RESEGMENT = 16
    cf.MARGIN = 0.0
    cf.GAMMA = 0.0
    cf._decay = 0.5 ** (1.0 / 64)


def never_switch(cf: CostFitPolicy) -> None:
    cf.MARGIN = float("inf")


def make_follow(cap: int, layers: int, prior: float, tune: Tune = None):
    pol = FollowPolicy(cap, num_layers=layers, prior=prior)
    if tune is not None:
        tune(pol.cf)
    return pol


def leader_set(pol: FollowPolicy) -> Set[Key]:
    return set(pol.cf._inners[pol.cf.choice].resident)


def run_follow(pol: FollowPolicy, trace: List[Key], check: bool = True):
    """Replay with per-access checks; return (hit list, victim list)."""
    hits: List[bool] = []
    victims: List[Optional[Key]] = []
    for key in trace:
        before = set(pol.resident)
        hits.append(pol.on_access(key))
        after = set(pol.resident)
        gone = before - after
        assert len(gone) <= 1
        victims.append(next(iter(gone)) if gone else None)
        if check:
            pol.check_invariants()
            assert len(after) <= pol.capacity
            assert key in after
    return hits, victims


# =========================================================================== #
# 1. simulate() contract
# =========================================================================== #
def test_simulate_empty_trace():
    assert simulate(build_policy("follow", 4, num_layers=2), []) == (
        0,
        0,
        0.0,
    )


@pytest.mark.parametrize("cap", [1, 2, 5, 12])
def test_simulate_contract(cap):
    trace = small_trace(7, layers=4, experts=8, topk=2, n=400)
    pol = build_policy("follow", cap, num_layers=4, prior=2 / 8)
    hits, misses, rate = simulate(pol, trace)
    assert hits + misses == len(trace)
    assert (pol.hits, pol.misses) == (hits, misses)
    assert rate == pytest.approx(hits / len(trace))
    assert pol.hit_rate == pytest.approx(rate)
    assert len(pol.resident) <= cap
    assert pol.keys == set(pol.resident)
    assert misses >= len(set(trace))  # compulsory misses at least
    assert pol.fallbacks == 0


def test_capacity_zero_all_miss():
    # Fixed contract: capacity 0 returns all-miss (0, n, 0.0), same as every
    # other registered policy (lcnu, s3fifo, costfit, lru) -- no StopIteration.
    pol = build_policy("follow", 0, num_layers=2)
    assert isinstance(pol, FollowPolicy)
    assert simulate(pol, [(0, 1), (1, 2), (0, 1)]) == (0, 3, 0.0)
    assert pol.misses == 3
    assert len(pol._real) == 0


# =========================================================================== #
# 2./3. Invariants on MoE-like traces with real CostFit switches
# =========================================================================== #
@pytest.mark.parametrize("cfg", SWITCH_CONFIGS, ids=str)
def test_invariants_and_no_fallback_with_natural_switches(cfg):
    trace, layers, cap, prior = _cfg_trace(cfg)
    pol = make_follow(cap, layers, prior)
    for key in trace:
        pol.on_access(key)
        pol.check_invariants()
        # Both shipped inners admit every miss, so the accessed key is in
        # the leader afterwards and therefore never left in D.
        assert key in leader_set(pol)
        assert key not in pol._out
    assert pol.switches >= 1, "trace must exercise a CostFit switch"
    assert pol.cf.choice == "s3fifo"
    assert pol.fallbacks == 0
    assert pol.hits + pol.misses == len(trace)


def test_invariants_with_switches_in_both_directions():
    trace = small_trace(3, layers=4, experts=8, topk=2, n=3000)
    pol = make_follow(6, 4, 2 / 8, tune=thrash)
    choices = set()
    for key in trace:
        pol.on_access(key)
        pol.check_invariants()
        choices.add(pol.cf.choice)
    assert pol.switches >= 4
    assert choices == {"lcnu", "s3fifo"}
    assert pol.fallbacks == 0


def test_check_invariants_detects_corruption():
    trace, layers, cap, prior = _cfg_trace(SWITCH_CONFIGS[0])
    pol = make_follow(cap, layers, prior)
    for key in trace[:5000]:
        pol.on_access(key)
    pol.check_invariants()
    pol._out[(99, 99)] = None  # key not in R
    with pytest.raises(AssertionError):
        pol.check_invariants()
    del pol._out[(99, 99)]
    in_leader = next(k for k in pol._real if k in leader_set(pol))
    pol._out[in_leader] = None  # key the leader holds
    with pytest.raises(AssertionError):
        pol.check_invariants()


def test_r_converges_to_leader_after_switch():
    trace, layers, cap, prior = _cfg_trace(SWITCH_CONFIGS[0])
    pol = make_follow(cap, layers, prior)
    converged_after_switch = False
    for key in trace:
        sw = pol.switches
        pol.on_access(key)
        if pol.switches != sw:
            converged_after_switch = False
            assert pol._out, "fresh switch should leave R != leader"
        elif pol.switches and not pol._out:
            converged_after_switch = True
            assert set(pol.resident) == leader_set(pol)
    assert converged_after_switch


# =========================================================================== #
# 4. No switch => FOLLOW is exactly standalone LCNU
# =========================================================================== #
# Proof sketch: R == L (LCNU's set) holds initially. LCNU (bypass "never")
# admits every miss and evicts exactly one key iff full; FOLLOW hits iff L
# hits, and on a full miss D == {LCNU's victim}. So R == L after every
# access and the hit/victim sequences coincide.
def _assert_follow_equals_lcnu(trace, layers, cap, prior, tune):
    pol = make_follow(cap, layers, prior, tune=tune)
    lcnu = build_policy("lcnu", cap, num_layers=layers, prior=prior)
    for key in trace:
        assert pol.on_access(key) == lcnu.on_access(key)
        assert set(pol.resident) == set(lcnu.resident)
        assert not pol._out
    assert pol.switches == 0
    assert pol.cf.choice == "lcnu"
    assert (pol.hits, pol.misses) == (lcnu.hits, lcnu.misses)
    return pol


def test_equals_lcnu_when_costfit_never_switches_naturally():
    trace, layers, cap, prior = _cfg_trace(NO_SWITCH_CONFIG)
    assert len(trace) > 2 * CostFitPolicy.RESEGMENT
    _assert_follow_equals_lcnu(trace, layers, cap, prior, tune=None)


def test_equals_lcnu_when_switching_is_disabled():
    trace, layers, cap, prior = _cfg_trace(SWITCH_CONFIGS[0])
    _assert_follow_equals_lcnu(trace, layers, cap, prior, tune=never_switch)
    # Non-vacuous: the same trace switches at shipped parameters, and then
    # FOLLOW's hit sequence departs from LCNU's.
    natural = make_follow(cap, layers, prior)
    lcnu = build_policy("lcnu", cap, num_layers=layers, prior=prior)
    nat_hits = [natural.on_access(k) for k in trace]
    lcnu_hits = [lcnu.on_access(k) for k in trace]
    assert natural.switches >= 1
    assert nat_hits != lcnu_hits


# =========================================================================== #
# 5. Brute-force reference (naive R - leader, explicit timestamps)
# =========================================================================== #
class ReferenceFollow:
    """Naive re-derivation of FOLLOW's victim choice.

    Every access recomputes ``D = R - leader`` from scratch. Each ``D`` key
    carries a stamp ``(access index, phase, tiebreak)``; the victim is the
    minimum stamp:

      * phase 0 -- leader switched at this access: every ``D`` key is
        (re)stamped, tiebreak = last real use (LRU first);
      * phase 1 -- the leader ghost evicted the key at this access,
        tiebreak = position in the leader's eviction sequence (observed
        through ``_evict_callback``);
      * phase 2 -- the accessed key itself is resident but out-of-leader
        (moved to the end).
    """

    def __init__(self, cap, layers, prior, tune: Tune = None, patch=None):
        self.cap = cap
        self.cf = CostFitPolicy(cap, num_layers=layers, prior=prior)
        if tune is not None:
            tune(self.cf)
        if patch is not None:
            patch(self.cf)
        self._log: List[Tuple[str, Key]] = []
        for name, inner in self.cf._inners.items():
            inner._evict_callback = functools.partial(self._record, name)
        self.R: Set[Key] = set()
        self.last_use: Dict[Key, int] = {}
        self.stamp: Dict[Key, tuple] = {}
        self.t = 0
        self.hits = 0
        self.fallbacks = 0
        self.d_mass = 0

    def _record(self, name: str, key: Key) -> None:
        self._log.append((name, key))

    def access(self, key: Key) -> Tuple[bool, Optional[Key]]:
        self.t += 1
        t = self.t
        before = self.cf.choice
        self._log = []
        self.cf.on_access(key)
        choice = self.cf.choice
        leader = set(self.cf._inners[choice].resident)
        out_now = {r for r in self.R if r not in leader}
        if choice != before:
            self.stamp = {r: (t, 0, self.last_use[r]) for r in out_now}
        else:
            evicted = [k for name, k in self._log if name == choice]
            for r in out_now - set(self.stamp):
                self.stamp[r] = (t, 1, evicted.index(r))
            for r in set(self.stamp) - out_now:
                assert r == key  # only the accessed key can rejoin
        hit = key in self.R
        victim = None
        if hit:
            self.hits += 1
        else:
            if len(self.R) >= self.cap:
                cands = [r for r in self.stamp if r in out_now]
                if cands:
                    victim = min(cands, key=self.stamp.__getitem__)
                else:
                    victim = min(self.R, key=self.last_use.__getitem__)
                    self.fallbacks += 1
                self.R.remove(victim)
                self.stamp.pop(victim, None)
                del self.last_use[victim]
            self.R.add(key)
        self.last_use[key] = t
        if key in leader:
            self.stamp.pop(key, None)
        else:
            self.stamp[key] = (t, 2, 0)
        assert set(self.stamp) == {r for r in self.R if r not in leader}
        self.d_mass += len(self.stamp)
        return hit, victim


def _compare_with_reference(trace, cap, layers, prior, tune=None, patch=None):
    pol = make_follow(cap, layers, prior, tune=tune)
    if patch is not None:
        patch(pol.cf)
        for name, inner in pol.cf._inners.items():
            inner._evict_callback = pol._make_evict_callback(name)
    ref = ReferenceFollow(cap, layers, prior, tune=tune, patch=patch)
    for i, key in enumerate(trace):
        ref_hit, ref_victim = ref.access(key)
        before = set(pol.resident)
        hit = pol.on_access(key)
        gone = before - set(pol.resident)
        victim = next(iter(gone)) if gone else None
        assert (hit, victim) == (ref_hit, ref_victim), f"access {i}: {key}"
        assert set(pol.resident) == ref.R
        assert pol.cf.choice == ref.cf.choice
    assert pol.hits == ref.hits
    assert pol.fallbacks == ref.fallbacks
    assert pol.d_mass == ref.d_mass
    assert pol.switches == ref.cf.switches
    return pol


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("cap", [2, 3, 4, 5, 6, 8])
def test_matches_reference_with_frequent_switches(seed, cap):
    trace = small_trace(seed, layers=3, experts=6, topk=2, n=600)
    pol = _compare_with_reference(trace, cap, 3, 2 / 6, tune=thrash)
    if cap >= 4:  # LCNU dominates at caps 2-3 on these traces
        assert pol.switches >= 3


def test_matches_reference_on_natural_switch_trace():
    trace, layers, cap, prior = _cfg_trace(SWITCH_CONFIGS[0])
    pol = _compare_with_reference(trace, cap, layers, prior)
    assert pol.switches >= 1


# =========================================================================== #
# 6. Non-admitting leader: LRU fallback and "hit on D moves to end"
# =========================================================================== #
def _bypass_lcnu(cap, layers, prior):
    def patch(cf: CostFitPolicy) -> None:
        cf._inners["lcnu"] = LayerCycleNextUsePolicy(
            cap, num_layers=layers, prior=prior, bypass="always"
        )

    return patch


@pytest.mark.parametrize("seed", range(4))
def test_fallback_and_d_hit_paths_match_reference(seed):
    cap, layers, prior = 4, 3, 2 / 6
    trace = small_trace(seed, layers=layers, experts=6, topk=2, n=500)
    patch = _bypass_lcnu(cap, layers, prior)
    pol = _compare_with_reference(
        trace, cap, layers, prior, tune=never_switch, patch=patch
    )
    assert pol.fallbacks > 0
    assert len(pol.resident) <= cap
    pol.check_invariants()


class EmptyGhost:
    """Stub leader that never holds anything (admits no miss)."""

    _evict_callback = None

    def __init__(self) -> None:
        self.resident: Dict[Key, None] = {}

    def on_access(self, key: Key) -> bool:
        return False


@pytest.mark.parametrize("cap", [1, 3, 6])
def test_empty_leader_degenerates_to_exact_lru(cap):
    # Leader holds nothing => D == R always, every miss appends to D and every
    # hit must move the key to the end of D: FOLLOW must then be exactly LRU
    # (this is the only way the "hit on D moves to end" rule is observable
    # with a full resident set).
    trace = moe_trace(5, 1, 6, 1, tokens=600, shift_at=300, zipf_a=1.5)
    pol = make_follow(cap, 1, 1 / 6, tune=never_switch)
    pol.cf._inners["lcnu"] = EmptyGhost()
    lru = build_policy("lru", cap)
    for key in trace:
        assert pol.on_access(key) == lru.on_access(key)
        assert list(pol._out) == list(pol.resident)
        assert set(pol.resident) == set(lru.resident)
        pol.check_invariants()
    assert pol.fallbacks == 0
    assert pol.hits == lru.hits > 0


# =========================================================================== #
# 7. Determinism and registry wiring
# =========================================================================== #
def test_deterministic_replay():
    trace, layers, cap, prior = _cfg_trace(SWITCH_CONFIGS[1])
    runs = []
    for _ in range(2):
        pol = make_follow(cap, layers, prior)
        hits, victims = run_follow(pol, trace, check=False)
        runs.append(
            (hits, victims, list(pol.resident), pol.d_mass, pol.switches)
        )
    assert runs[0] == runs[1]
    assert runs[0][4] >= 1


def test_registry_wiring():
    assert "follow" in POLICY_FACTORIES
    assert "follow" in GEOMETRY_POLICIES
    assert "follow" not in ORACLE_POLICIES
    pol = build_policy("follow", 10, num_layers=6, prior=0.125)
    assert isinstance(pol, FollowPolicy)
    assert pol.capacity == 10
    assert pol.uses_oracle is False
    lcnu = pol.cf._inners["lcnu"]
    assert isinstance(lcnu, LayerCycleNextUsePolicy)
    assert (lcnu.num_layers, lcnu.prior, lcnu.capacity) == (6, 0.125, 10)
    assert pol.cf._inners["s3fifo"].capacity == 10
    assert pol.cf.choice == "lcnu"
    assert (pol.switches, pol.fallbacks, pol.d_mass) == (0, 0, 0)
    assert pol.bind_trace([(0, 1)]) is None
