"""Behavioral tests for the prefetch/replacement fetch engines.

The central invariant for every engine is that, starting full, each fetch of a
non-resident expert keeps the resident set exactly at ``mem_capacity`` while
making the requested expert resident. On top of that we pin down the
policy-specific eviction/prefetch decisions (FIFO order, LRU timestamps, LFU
frequency, LFU-K sliding window) and the constructor's input validation.
"""

import itertools
import random

import numpy as np
import pytest

import benchmarks.cache_simulator.fetch_engines as fe
from benchmarks.cache_simulator.fetch_engines import (
    FetchEngineBase,
    LFUFetch,
    LFUKFetch,
    StreamFetch,
    StreamFetchLRU,
)


def full_states(n_layer, n_expert, capacity):
    """Fresh [n_layer, n_expert] state grid with ``capacity`` resident, in
    row-major order."""
    states = np.zeros((n_layer, n_expert), dtype=int)
    remaining = capacity
    for layer in range(n_layer):
        for expert in range(n_expert):
            if remaining <= 0:
                break
            states[layer, expert] = 1
            remaining -= 1
    assert int(states.sum()) == capacity
    return states


def routes_from(n_layer, n_expert, active):
    grid = np.zeros((n_layer, n_expert), dtype=int)
    for layer, expert in active:
        grid[layer, expert] = 1
    return grid


def make_engine(cache_cls, n_layer=4, n_expert=4, capacity=6, stride=2):
    states = full_states(n_layer, n_expert, capacity)
    if cache_cls is LFUKFetch:
        return cache_cls(n_layer, n_expert, capacity, states, stride, k=3)
    return cache_cls(n_layer, n_expert, capacity, states, stride)


ALL_ENGINES = (StreamFetch, StreamFetchLRU, LFUFetch, LFUKFetch)


# --------------------------------------------------------------------------- #
# Core invariant: resident count stays at capacity; requested becomes resident
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cache_cls", ALL_ENGINES)
def test_fetch_keeps_capacity_and_admits_request(cache_cls):
    engine = make_engine(cache_cls)
    rng = random.Random(99)
    for _ in range(200):
        layer = rng.randrange(4)
        expert = rng.randrange(4)
        if not engine.access(layer, expert):
            engine.fetch(layer, expert)
            assert engine.access(layer, expert)  # now resident
        assert int(np.sum(engine.expert_states)) == engine.mem_capacity


# --------------------------------------------------------------------------- #
# Constructor input validation
# --------------------------------------------------------------------------- #
def test_base_rejects_shape_mismatch():
    with pytest.raises(AssertionError):
        FetchEngineBase(3, 3, 4, np.ones((2, 2), dtype=int))


def test_base_rejects_resident_count_not_matching_capacity():
    with pytest.raises(AssertionError):
        StreamFetch(2, 3, 3, np.ones((2, 3), dtype=int), stride=1)  # sum=6!=3


@pytest.mark.parametrize("stride", [0, 3, 5])
def test_stream_fetch_rejects_out_of_range_stride(stride):
    states = full_states(2, 3, 3)
    with pytest.raises(AssertionError):
        StreamFetch(2, 3, 3, states, stride=stride)  # need 1 <= stride < 3


# --------------------------------------------------------------------------- #
# StreamFetch: FIFO eviction order
# --------------------------------------------------------------------------- #
def test_stream_fetch_evicts_in_fifo_order():
    engine = StreamFetch(3, 3, 4, full_states(3, 3, 4), stride=1)
    # Resident, oldest first: (0,0) (0,1) (0,2) (1,0)
    assert not engine.access(2, 2)
    engine.fetch(2, 2)
    assert not engine.access(0, 0)  # oldest evicted
    assert engine.access(2, 2)
    for resident in [(0, 1), (0, 2), (1, 0)]:
        assert engine.access(*resident)

    engine.fetch(2, 1)
    assert not engine.access(0, 1)  # next oldest evicted
    assert engine.access(2, 1)


# --------------------------------------------------------------------------- #
# StreamFetchLRU: evicts the least-recently fetched (deterministic clock)
# --------------------------------------------------------------------------- #
def test_stream_fetch_lru_evicts_smallest_timestamp(monkeypatch):
    counter = itertools.count(1)
    monkeypatch.setattr(fe.time, "time_ns", lambda: next(counter))

    engine = StreamFetchLRU(3, 3, 4, full_states(3, 3, 4), stride=1)
    # Timestamps assigned in construction order: (0,0)<(0,1)<(0,2)<(1,0).
    engine.fetch(2, 2)
    assert not engine.access(0, 0)  # smallest timestamp evicted
    engine.fetch(2, 1)
    assert not engine.access(0, 1)


# --------------------------------------------------------------------------- #
# LFUFetch: evict least-frequent, prefetch most-frequent
# --------------------------------------------------------------------------- #
def test_lfu_evicts_least_frequently_used():
    engine = LFUFetch(2, 3, 3, full_states(2, 3, 3), stride=1)
    # Frequencies: (0,0)=3 > (0,1)=2 > (0,2)=1
    for active in (
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (0, 1)],
        [(0, 0)],
    ):
        engine.global_update(routes_from(2, 3, active))

    assert not engine.access(1, 0)
    engine.fetch(1, 0)
    assert not engine.access(0, 2)  # least frequent evicted
    assert engine.access(1, 0)
    assert engine.access(0, 0)
    assert engine.access(0, 1)


def test_lfu_prefetches_most_frequent_non_resident():
    engine = LFUFetch(2, 3, 3, full_states(2, 3, 3), stride=1)
    # Make (1,1) the most frequent expert, but it is not resident.
    for _ in range(5):
        engine.global_update(routes_from(2, 3, [(1, 1)]))
    engine.global_update(routes_from(2, 3, [(0, 0)]))

    engine.fetch(1, 0)
    assert engine.access(1, 1)  # prefetched because most frequent
    assert engine.access(1, 0)
    assert not engine.access(0, 1)  # evicted (count 0)
    assert not engine.access(0, 2)  # evicted (count 0)


# --------------------------------------------------------------------------- #
# LFUKFetch: counts reflect a sliding window of the last k requests
# --------------------------------------------------------------------------- #
def test_lfuk_window_decrements_expired_requests():
    engine = LFUKFetch(2, 3, 3, full_states(2, 3, 3), stride=1, k=2)
    engine.global_update(routes_from(2, 3, [(0, 0), (0, 1)]))  # r1
    engine.global_update(routes_from(2, 3, [(0, 0)]))  # r2
    engine.global_update(routes_from(2, 3, [(1, 1)]))  # r3, evicts r1

    # Window is {r2, r3}: (0,0) seen once (r2), (1,1) once (r3), (0,1) expired.
    assert engine.expert_count[(0, 0)] == 1
    assert engine.expert_count[(0, 1)] == 0
    assert engine.expert_count[(1, 1)] == 1
    assert all(count >= 0 for count in engine.expert_count.values())


# --------------------------------------------------------------------------- #
# Streaming engines treat global_update as a no-op
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cache_cls", [StreamFetch, StreamFetchLRU])
def test_streaming_global_update_is_noop(cache_cls):
    engine = cache_cls(2, 3, 3, full_states(2, 3, 3), stride=1)
    before = engine.expert_states.copy()
    engine.global_update(routes_from(2, 3, [(0, 0), (1, 2)]))
    assert np.array_equal(engine.expert_states, before)
