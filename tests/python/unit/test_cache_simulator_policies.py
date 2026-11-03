"""Behavioral tests for the CachePolicy baselines.

Two kinds of checks:

* **Deterministic hand-computed cases** per policy pin down the exact
  eviction / prefetch decision on tiny traces.
* **Invariants over fuzzed traces** that must hold for *every* policy and
  capacity: ``hits + misses == len(accesses)``, the resident set never exceeds
  capacity, and the demand hit rate never exceeds Belady on the same trace
  (``hit_rate(policy) <= hit_rate(Belady)``). Belady is the demand-paging
  ceiling; prefetch policies may only exceed it on the separately reported
  ``effective_hit_rate`` (prefetch-coverage) upper bound.
"""

import random

import pytest

from benchmarks.cache_simulator.belady_cache_simulator import belady_simulate
from benchmarks.cache_simulator.policies import (
    GEOMETRY_POLICIES,
    ORACLE_POLICIES,
    POLICY_FACTORIES,
    ActivationMatrixPolicy,
    ARCPolicy,
    LayerCycleNextUsePolicy,
    LayerLRUPolicy,
    LFUKPolicy,
    LFUPolicy,
    LHUPolicy,
    LIRSPolicy,
    LRUKPolicy,
    LRUPolicy,
    ModelTracingPolicy,
    MultiLayerAheadPolicy,
    S3FIFOPolicy,
    ScorePolicy,
    SIEVEPolicy,
    SingleLayerAheadPolicy,
    StridePrefetchPolicy,
    TwoQPolicy,
    WTinyLFUPolicy,
    build_policy,
    layer_segments,
    multi_layer_oracle,
    next_layer_oracle,
    simulate,
)

# Convenience single-layer keys (layer 0).
A, B, C, D, E = (0, 0), (0, 1), (0, 2), (0, 3), (0, 4)


def random_trace(rng, length, layers=3, experts=6):
    return [
        (rng.randrange(layers), rng.randrange(experts)) for _ in range(length)
    ]


def layered_trace(rng, steps, layers=4, experts=8, topk=2):
    """Layer-major access trace resembling decode steps (L layers, top-k)."""
    accesses = []
    for _ in range(steps):
        for layer in range(layers):
            picks = rng.sample(range(experts), topk)
            for expert in picks:
                accesses.append((layer, expert))
    return accesses


# --------------------------------------------------------------------------- #
# Harness basics
# --------------------------------------------------------------------------- #
def test_simulate_empty_trace():
    assert simulate(LRUPolicy(4), []) == (0, 0, 0.0)


def test_capacity_zero_all_miss():
    accesses = [A, A, B, A]
    hits, misses, rate = simulate(LRUPolicy(0), accesses)
    assert (hits, misses, rate) == (0, 4, 0.0)


def test_conservation_and_capacity_invariants():
    rng = random.Random(1)
    accesses = random_trace(rng, 60)
    for name in POLICY_FACTORIES:
        for cap in (1, 2, 5, 12):
            policy = build_policy(name, cap)
            hits, misses, _ = simulate(policy, accesses)
            assert hits + misses == len(accesses), name
            assert len(policy.resident) <= cap, name


# --------------------------------------------------------------------------- #
# LRU hand-computed
# --------------------------------------------------------------------------- #
def test_lru_evicts_least_recently_used():
    # cap 2. Accesses: A B A C -> on C, B is LRU (A was touched again), evict B.
    policy = LRUPolicy(2)
    for key in (A, B, A, C):
        policy.on_access(key)
    assert A in policy.resident
    assert C in policy.resident
    assert B not in policy.resident
    # hits: only the 2nd A. misses: A,B,C = 3.
    assert policy.hits == 1
    assert policy.misses == 3


def test_lru_consecutive_repeat_hits():
    hits, misses, _ = simulate(LRUPolicy(1), [A, A, A])
    assert (hits, misses) == (2, 1)


# --------------------------------------------------------------------------- #
# LFU hand-computed
# --------------------------------------------------------------------------- #
def test_lfu_evicts_least_frequent():
    # Build frequencies A:3, B:2, then C arrives at cap 2 -> evict B (freq 1<2).
    policy = LFUPolicy(2)
    for key in (A, A, A, B, B):
        policy.on_access(key)
    # resident {A,B}; now C: B freq 2, A freq 3 -> evict B.
    policy.on_access(C)
    assert A in policy.resident
    assert C in policy.resident
    assert B not in policy.resident


def test_lfu_tie_breaks_by_lru():
    policy = LFUPolicy(2)
    # A and B each freq 1; A older. C arrives -> evict A (LRU tie-break).
    for key in (A, B):
        policy.on_access(key)
    policy.on_access(C)
    assert A not in policy.resident
    assert B in policy.resident


# --------------------------------------------------------------------------- #
# LFU-K window
# --------------------------------------------------------------------------- #
def test_lfuk_window_forgets_old_accesses():
    policy = LFUKPolicy(3, k=2)
    # window keeps only last 2 accesses' contribution.
    for key in (A, A, B):  # after this window={A(pos1),B}; A count within win=1
        policy.on_access(key)
    # Older A (pos0) has expired from the window.
    assert policy.freq[A] == 1
    assert policy.freq[B] == 1


def test_lfuk_rejects_bad_k():
    with pytest.raises(ValueError):
        LFUKPolicy(2, k=0)


# --------------------------------------------------------------------------- #
# Model-tracing: global counts from an offline pass
# --------------------------------------------------------------------------- #
def test_model_tracing_evicts_globally_least_popular():
    # Global counts over the whole trace: A appears 3x, B 2x, C 1x.
    accesses = [A, B, C, A, B, A]
    policy = ModelTracingPolicy(2)
    policy.bind_trace(accesses)
    assert policy.global_count[A] == 3
    assert policy.global_count[B] == 2
    assert policy.global_count[C] == 1
    # Drive the first three: A,B resident, then C at cap 2 evicts the globally
    # least popular resident among {A,B} = B.
    for key in accesses[:3]:
        policy.on_access(key)
    assert A in policy.resident
    assert C in policy.resident
    assert B not in policy.resident


# --------------------------------------------------------------------------- #
# Score-based eviction
# --------------------------------------------------------------------------- #
def test_score_decay_prefers_recent():
    policy = ScorePolicy(2, decay=0.5)
    # A accessed early then idles; B accessed repeatedly recently.
    for key in (A, B, B):
        policy.on_access(key)
    # scores: A decayed twice, B boosted twice -> B higher. C evicts A.
    policy.on_access(C)
    assert A not in policy.resident
    assert B in policy.resident
    assert C in policy.resident


def test_score_decay_one_is_lfu_like():
    policy = ScorePolicy(2, decay=1.0)
    for key in (A, A, B):
        policy.on_access(key)
    policy.on_access(C)  # B freq 1 < A freq 2 -> evict B
    assert A in policy.resident
    assert B not in policy.resident


def test_score_rejects_bad_decay():
    with pytest.raises(ValueError):
        ScorePolicy(2, decay=0.0)
    with pytest.raises(ValueError):
        ScorePolicy(2, decay=1.5)


# --------------------------------------------------------------------------- #
# LHU (least hot unit)
# --------------------------------------------------------------------------- #
def test_lhu_evicts_coldest():
    policy = LHUPolicy(2, alpha=0.5)
    for key in (A, A, B):  # A hot (two hits), B cold
        policy.on_access(key)
    policy.on_access(C)  # evict coldest resident = B
    assert A in policy.resident
    assert B not in policy.resident
    assert C in policy.resident


def test_lhu_rejects_bad_alpha():
    with pytest.raises(ValueError):
        LHUPolicy(2, alpha=-0.1)
    with pytest.raises(ValueError):
        LHUPolicy(2, alpha=1.1)


# --------------------------------------------------------------------------- #
# Lookahead oracle helpers
# --------------------------------------------------------------------------- #
def test_layer_segments_splits_on_layer_change():
    accesses = [(0, 1), (0, 2), (1, 3), (1, 4), (0, 1)]
    assert layer_segments(accesses) == [(0, 0, 2), (1, 2, 4), (0, 4, 5)]


def test_next_layer_oracle_points_to_following_layer():
    accesses = [(0, 1), (0, 2), (1, 3), (1, 4)]
    oracle = next_layer_oracle(accesses)
    # positions in layer-0 segment see layer-1 experts.
    assert oracle[0] == [(1, 3), (1, 4)]
    assert oracle[1] == [(1, 3), (1, 4)]
    # last layer has no next.
    assert oracle[2] == []
    assert oracle[3] == []


def test_multi_layer_oracle_unions_horizon():
    accesses = [(0, 1), (1, 2), (2, 3), (3, 4)]
    oracle = multi_layer_oracle(accesses, horizon=2)
    assert oracle[0] == [(1, 2), (2, 3)]
    assert oracle[1] == [(2, 3), (3, 4)]
    assert oracle[3] == []


# --------------------------------------------------------------------------- #
# Prefetch policies: oracle coverage
# --------------------------------------------------------------------------- #
def test_single_layer_ahead_covers_next_layer():
    # Two layers, enough capacity to hold both layers' experts.
    accesses = [(0, 0), (0, 1), (1, 5), (1, 6), (0, 0), (0, 1), (1, 5), (1, 6)]
    policy = SingleLayerAheadPolicy(8)
    simulate(policy, accesses)
    # With perfect next-layer prefetch every layer-1 first touch is covered.
    assert policy.prefetch_covered > 0
    assert policy.effective_hit_rate >= policy.hit_rate


def test_stride_prefetch_rejects_bad_stride():
    with pytest.raises(ValueError):
        StridePrefetchPolicy(4, stride=0)


def test_multi_layer_ahead_rejects_bad_horizon():
    with pytest.raises(ValueError):
        MultiLayerAheadPolicy(4, horizon=0)


def test_activation_matrix_learns_coactivation():
    # A in layer 0 is always followed by expert (1, 7) in layer 1.
    accesses = [(0, 0), (1, 7)] * 6
    policy = ActivationMatrixPolicy(4, top_m=1)
    simulate(policy, accesses)
    # After the first transition the matrix knows (0,0) -> (1,7).
    assert policy._matrix[(0, 0)][(1, 7)] > 0
    assert policy.prefetch_covered > 0


def test_activation_matrix_rejects_bad_top_m():
    with pytest.raises(ValueError):
        ActivationMatrixPolicy(4, top_m=0)


# --------------------------------------------------------------------------- #
# Core invariant: demand hit rate <= Belady on every trace
# --------------------------------------------------------------------------- #
def test_hit_rate_never_exceeds_belady_fuzz():
    rng = random.Random(20260301)
    for _ in range(80):
        length = rng.randrange(1, 60)
        accesses = random_trace(rng, length)
        for cap in (1, 2, 4, 8):
            _, _, belady_rate = belady_simulate(accesses, cap)
            for name in POLICY_FACTORIES:
                policy = build_policy(name, cap)
                hits, misses, rate = simulate(policy, accesses)
                assert hits + misses == length, (name, cap)
                assert len(policy.resident) <= cap, (name, cap)
                assert rate <= belady_rate + 1e-9, (
                    name,
                    cap,
                    rate,
                    belady_rate,
                )


def test_hit_rate_never_exceeds_belady_layered_traces():
    rng = random.Random(777)
    for _ in range(20):
        accesses = layered_trace(rng, steps=rng.randrange(2, 8))
        for cap in (2, 4, 8, 16):
            _, _, belady_rate = belady_simulate(accesses, cap)
            for name in POLICY_FACTORIES:
                policy = build_policy(name, cap)
                _, _, rate = simulate(policy, accesses)
                assert rate <= belady_rate + 1e-9, (name, cap)


def test_oracle_registry_matches_uses_oracle_flag():
    for name, factory in POLICY_FACTORIES.items():
        policy = factory(4)
        assert policy.uses_oracle == (name in ORACLE_POLICIES), name


def test_build_policy_unknown_name():
    with pytest.raises(KeyError):
        build_policy("does_not_exist", 4)


def test_costfit3_n_way_arbitration_accounting():
    # 3-candidate arbitration: credited-rate property, deterministic, and the
    # switch bookkeeping must be internally consistent (total == per-choice).
    rng = random.Random(7)
    accesses = bursty_decode_trace(rng, 400, 6, 16, 2)
    cap = 6 * 2 - 2
    pol = build_policy("costfit3", cap, num_layers=6, prior=2 / 16)
    hits = sum(1 for k in accesses if pol.on_access(k))
    assert pol.hit_rate == pytest.approx(hits / len(accesses))
    assert pol.effective_hit_rate == pol.hit_rate  # credited at access time
    assert pol.switches == sum(pol.choice_switches.values())
    assert pol.uses_oracle is False
    # deterministic: a second run must agree bit-for-bit
    pol2 = build_policy("costfit3", cap, num_layers=6, prior=2 / 16)
    hits2 = sum(1 for k in accesses if pol2.on_access(k))
    assert (hits2, pol2.switches, pol2.choice_switches) == (
        hits,
        pol.switches,
        pol.choice_switches,
    )
    # no-regression vs the sealed 2-way arbiter on the same stream
    pol_c2 = build_policy("costfit", cap, num_layers=6, prior=2 / 16)
    hits_c2 = sum(1 for k in accesses if pol_c2.on_access(k))
    assert hits >= hits_c2 - len(accesses) * 0.001  # within 0.1% of stream length


def test_full_capacity_only_compulsory_misses():
    accesses = [A, B, C, A, B, C, A]
    distinct = len({*accesses})
    names = (
        "lru",
        "lfu",
        "model_tracing",
        "score",
        "lhu",
        "arc",
        "2q",
        "lru_k",
        "lirs",
        "s3fifo",
        "sieve",
        "w_tinylfu",
        "layer_lru",
        "lcnu",
    )
    for name in names:
        policy = build_policy(name, distinct)
        _, misses, _ = simulate(policy, accesses)
        assert misses == distinct, name


# --------------------------------------------------------------------------- #
# Classic recency/frequency hybrids: hand-computed cases
# --------------------------------------------------------------------------- #
def test_arc_ghost_hit_adapts_target():
    policy = ARCPolicy(2)
    for key in (A, B, A, C):
        policy.on_access(key)
    # A was seen twice (T2); the miss on C evicts T1's LRU (B) into ghost B1.
    assert set(policy.resident) == {A, C}
    assert B in policy.b1
    policy.on_access(B)  # ghost hit in B1 grows the recency target p
    assert policy.p == 1.0
    assert set(policy.resident) == {B, C}
    assert A in policy.b2


def test_two_q_remembered_key_enters_am():
    policy = TwoQPolicy(4)  # kin = 1, kout = 2
    for key in (A, B, C, D, E):
        policy.on_access(key)
    assert A in policy.a1out and A not in policy.resident
    policy.on_access(A)  # remembered in A1out -> admitted to the LRU Am
    assert A in policy.am
    assert B in policy.a1out
    assert set(policy.resident) == {A, C, D, E}


def test_lru_k_prefers_keys_with_k_references():
    # LRU would evict A (B is more recent); LRU-2 evicts B, which has only one
    # reference and therefore an infinite backward 2-distance.
    policy = LRUKPolicy(2)
    for key in (A, A, B, C):
        policy.on_access(key)
    assert set(policy.resident) == {A, C}


def test_lirs_promotes_hir_with_small_irr():
    policy = LIRSPolicy(3)  # 2 LIR slots + 1 resident-HIR slot
    for key in (A, B, C, D, C):
        policy.on_access(key)
    # C was re-referenced while still in the stack: it becomes LIR and the
    # oldest LIR key (A) is demoted to the resident-HIR queue.
    assert policy.lir == {B, C}
    assert set(policy.resident) == {A, B, C}
    assert list(policy.queue) == [A]


def test_s3fifo_promotion_and_ghost_admission():
    policy = S3FIFOPolicy(4)  # small queue 1 slot, main 3
    for key in (A, A, A, B, C, D, E):
        policy.on_access(key)
    # A was hit twice in S -> moved to M; B (never hit) -> ghost.
    assert A in policy.main
    assert B in policy.ghost and B not in policy.resident
    policy.on_access(B)  # ghost hit -> straight into M
    assert B in policy.main
    assert C in policy.ghost
    assert set(policy.resident) == {A, B, D, E}


def test_sieve_skips_visited_and_keeps_hand():
    policy = SIEVEPolicy(3)
    for key in (A, B, C, A, D):
        policy.on_access(key)
    # Hand starts at the oldest (A, visited -> cleared), evicts B.
    assert set(policy.resident) == {A, C, D}
    policy.on_access(E)  # hand resumes at C (unvisited) -> evict C
    assert set(policy.resident) == {A, D, E}


def test_w_tinylfu_rejects_cold_candidates():
    policy = WTinyLFUPolicy(4, window=0.25)  # window 1, main 3
    for key in (A, A, A, B, C, D, E, (0, 5)):
        policy.on_access(key)
    # D and E left the window with frequency 1 < A's 3 and were rejected.
    assert set(policy.resident) == {A, B, C, (0, 5)}
    assert A in policy.probation


def test_layer_lru_evicts_within_the_layer():
    p0, p1, q0, q1, q2 = (0, 0), (0, 1), (1, 0), (1, 1), (1, 2)
    policy = LayerLRUPolicy(4, num_layers=2)  # 2 slots per layer
    for key in (p0, q0, q1, p1, q2):
        policy.on_access(key)
    # Global LRU would evict p0; the per-layer quota evicts layer 1's LRU.
    assert set(policy.resident) == {p0, p1, q1, q2}


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TwoQPolicy(4, kin=0.0),
        lambda: LRUKPolicy(4, k=0),
        lambda: LIRSPolicy(4, hir_frac=1.0),
        lambda: S3FIFOPolicy(4, small=0.0),
        lambda: WTinyLFUPolicy(4, window=0.0),
        lambda: LayerLRUPolicy(4, num_layers=0),
        lambda: LayerCycleNextUsePolicy(4, alpha=0.0),
        lambda: LayerCycleNextUsePolicy(4, beta=1.5),
        lambda: LayerCycleNextUsePolicy(4, prior=0.0),
        lambda: LayerCycleNextUsePolicy(4, num_layers=0),
    ],
)
def test_new_policies_reject_bad_parameters(factory):
    with pytest.raises(ValueError):
        factory()


# --------------------------------------------------------------------------- #
# LCNU (layer-cycle next-use)
# --------------------------------------------------------------------------- #
def cyclic_decode_trace(rng, steps, layers, experts, topk):
    """Decode-like trace: every step visits layers 0..L-1 in order."""
    accesses = []
    for _ in range(steps):
        for layer in range(layers):
            picks = rng.sample(range(experts), topk)
            accesses.extend((layer, e) for e in picks)
    return accesses


class _BruteForceLCNU(LayerCycleNextUsePolicy):
    """Reference victim search: score every resident from scratch."""

    def _victim(self):
        cur, cycle = self._cur, self.cycle
        best = None
        for layer, keys in self._by_layer.items():
            for key in keys:
                if not self.cycle_aware:
                    d = 0
                elif layer == cur:
                    d = cycle if key in self._used_now else 0
                else:
                    d = (layer - cur) % cycle
                p = self.routing_probability(key)
                score = d + cycle * (1.0 / p - 1.0)
                if (
                    best is None
                    or score > best[0]
                    or (score == best[0] and key < best[1])
                ):
                    best = (score, key, layer)
        return best[2], best[1]


@pytest.mark.parametrize("cycle_aware", [True, False])
def test_lcnu_cached_victims_match_brute_force(cycle_aware):
    rng = random.Random(4242)
    for trial in range(40):
        layers = rng.randrange(1, 6)
        experts = rng.randrange(2, 9)
        topk = rng.randrange(1, experts + 1)
        if trial % 2:
            accesses = cyclic_decode_trace(
                rng, rng.randrange(2, 12), layers, experts, topk
            )
        else:  # arbitrary layer order and in-visit repeats
            accesses = random_trace(rng, rng.randrange(1, 120), layers, experts)
        num_layers = None if trial % 3 == 0 else layers
        for cap in (1, 2, 3, 5, 8, 13):
            kwargs = {"num_layers": num_layers, "cycle_aware": cycle_aware}
            fast = LayerCycleNextUsePolicy(cap, **kwargs)
            ref = _BruteForceLCNU(cap, **kwargs)
            for key in accesses:
                assert fast.on_access(key) == ref.on_access(key)
                assert set(fast.resident) == set(ref.resident)


def test_lcnu_routing_probability_estimator():
    policy = LayerCycleNextUsePolicy(4, alpha=0.5, beta=1.0, prior=0.25)
    policy.on_access((0, 0))  # first visit of layer 0: ema = 0.5*0.25 + 0.5
    assert policy.routing_probability((0, 0)) == pytest.approx(0.625)
    policy.on_access((0, 0))  # same visit: at most one routing event
    assert policy.routing_probability((0, 0)) == pytest.approx(0.625)
    policy.on_access((1, 0))
    policy.on_access((0, 1))  # layer 0 runs again without expert 0
    assert policy.routing_probability((0, 0)) == pytest.approx(0.3125)

    freq_only = LayerCycleNextUsePolicy(4, beta=0.0, prior=0.25)
    for key in ((0, 0), (1, 0), (0, 1), (1, 0)):
        freq_only.on_access(key)
    # Routed on 1 of 2 visits since first seen, smoothed: (1 + 1) / (2 + 4).
    assert freq_only.routing_probability((0, 0)) == pytest.approx(2.0 / 6.0)


def test_lcnu_keeps_the_layer_that_runs_next():
    # Three layers, cap 2, identical routing history for (0, 0) and (1, 0).
    # On the miss in layer 2, layer 0 runs next while layer 1 just ran, so
    # LCNU drops (1, 0); LRU would drop (0, 0), the one needed soonest.
    policy = LayerCycleNextUsePolicy(2, num_layers=3)
    for key in ((0, 0), (1, 0), (2, 0)):
        policy.on_access(key)
    assert set(policy.resident) == {(0, 0), (2, 0)}
    lru = LRUPolicy(2)
    for key in ((0, 0), (1, 0), (2, 0)):
        lru.on_access(key)
    assert set(lru.resident) == {(1, 0), (2, 0)}


def test_lcnu_prefers_frequently_routed_experts():
    # Same layer distance, different routing history: the expert that was
    # routed on every visit outlives the one routed once.
    policy = LayerCycleNextUsePolicy(2, num_layers=2)
    for key in ((0, 0), (1, 0), (0, 0), (1, 1)):
        policy.on_access(key)
    # Miss on (1, 1): (1, 0) was not routed this visit (score uses d = 0 but a
    # decayed estimate), (0, 0) runs next and was routed on both visits.
    assert set(policy.resident) == {(0, 0), (1, 1)}


def test_lcnu_does_not_thrash_on_cyclic_scan():
    # A fixed set of 2 experts per layer over 4 layers = 8 keys per step.
    # With 6 slots LRU never hits (the cyclic scan is longer than the cache),
    # while LCNU keeps a stable subset resident.
    accesses = [
        (layer, e) for _ in range(30) for layer in range(4) for e in (0, 1)
    ]
    _, _, lru_rate = simulate(LRUPolicy(6), accesses)
    _, _, lcnu_rate = simulate(LayerCycleNextUsePolicy(6), accesses)
    _, _, belady_rate = belady_simulate(accesses, 6)
    assert lru_rate == 0.0
    assert lcnu_rate > 0.6
    assert lcnu_rate <= belady_rate + 1e-9


def bursty_decode_trace(rng, steps, layers, experts, topk, keep=0.6):
    """Cyclic decode trace with skewed popularity and token-to-token reuse.

    Each step keeps each of a layer's previous experts with probability
    ``keep`` and refills from a Zipf-like popularity distribution, the two
    effects measured on real Qwen3-MoE routing.
    """
    weights = [1.0 / (rank + 1) for rank in range(experts)]
    current = [rng.sample(range(experts), topk) for _ in range(layers)]
    accesses = []
    for _ in range(steps):
        for layer in range(layers):
            picks = [e for e in current[layer] if rng.random() < keep]
            pool = [e for e in range(experts) if e not in picks]
            while len(picks) < topk:
                expert = rng.choices(pool, [weights[e] for e in pool])[0]
                picks.append(expert)
                pool.remove(expert)
            current[layer] = picks
            accesses.extend((layer, e) for e in picks)
    return accesses


def test_lcnu_beats_lru_and_lfu_below_one_step_working_set():
    rng = random.Random(0)
    layers, experts, topk = 6, 16, 2
    accesses = bursty_decode_trace(rng, 300, layers, experts, topk)
    cap = layers * topk - 2  # smaller than one decode step's working set
    rates = {}
    for name in ("lru", "lfu", "lcnu"):
        kwargs = {"num_layers": layers} if name in GEOMETRY_POLICIES else {}
        _, _, rates[name] = simulate(
            build_policy(name, cap, **kwargs), accesses
        )
    _, _, belady_rate = belady_simulate(accesses, cap)
    assert rates["lru"] == 0.0
    assert rates["lcnu"] > rates["lfu"]
    assert rates["lcnu"] <= belady_rate + 1e-9
