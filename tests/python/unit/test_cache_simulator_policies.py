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
    ORACLE_POLICIES,
    POLICY_FACTORIES,
    ActivationMatrixPolicy,
    LFUKPolicy,
    LFUPolicy,
    LHUPolicy,
    LRUPolicy,
    ModelTracingPolicy,
    MultiLayerAheadPolicy,
    ScorePolicy,
    SingleLayerAheadPolicy,
    StridePrefetchPolicy,
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


def test_full_capacity_only_compulsory_misses():
    accesses = [A, B, C, A, B, C, A]
    distinct = len({*accesses})
    for name in ("lru", "lfu", "model_tracing", "score", "lhu"):
        policy = build_policy(name, distinct)
        _, misses, _ = simulate(policy, accesses)
        assert misses == distinct, name
