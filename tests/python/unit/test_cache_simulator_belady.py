"""Behavioral tests for the Belady optimal expert-cache simulator.

Correctness here means *optimality*: the simulator must achieve the minimum
possible number of misses for an access sequence and capacity. We verify that
against an independent brute-force offline optimum, and cross-check the three
Belady implementations (``belady_simulate`` plus ``ExpertCache`` /
``HeapExpertCache``) against it over fuzzed traces.
"""

import contextlib
import io
import random
from collections import OrderedDict

import numpy as np
import pytest

from benchmarks.cache_simulator.belady_cache_simulator import (
    belady_simulate,
    build_future_positions,
    extract_access_sequence,
)
from benchmarks.cache_simulator.cache_strategy import (
    ExpertCache,
    ExpertIndex,
    HeapExpertCache,
)

Key = tuple


# --------------------------------------------------------------------------- #
# Independent references (ground truth)
# --------------------------------------------------------------------------- #
def opt_min_misses(accesses, capacity):
    """Exhaustive demand-paging optimum: minimum misses over all evictions."""
    assert capacity >= 1
    n = len(accesses)
    memo = {}

    def rec(i, cache):
        if i == n:
            return 0
        state = (i, cache)
        cached = memo.get(state)
        if cached is not None:
            return cached
        key = accesses[i]
        if key in cache:
            res = rec(i + 1, cache)
        elif len(cache) < capacity:
            res = 1 + rec(i + 1, cache | {key})
        else:
            best = None
            for victim in cache:
                r = 1 + rec(i + 1, (cache - {victim}) | {key})
                if best is None or r < best:
                    best = r
            res = best
        memo[state] = res
        return res

    return rec(0, frozenset())


def lru_misses(accesses, capacity):
    """Reference LRU replacement; OPT must never do worse than this."""
    cache: OrderedDict = OrderedDict()
    misses = 0
    for key in accesses:
        if key in cache:
            cache.move_to_end(key)
            continue
        misses += 1
        if len(cache) >= capacity:
            cache.popitem(last=False)
        cache[key] = True
    return misses


def random_trace(rng, length, layers=2, experts=3):
    return [
        (rng.randrange(layers), rng.randrange(experts)) for _ in range(length)
    ]


def run_strategy_cache(cache_cls, accesses, capacity):
    """Drive an ExpertCache/HeapExpertCache over ``accesses`` (quietly)."""
    sequence = [ExpertIndex(expert, layer) for (layer, expert) in accesses]
    cache = cache_cls(capacity)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        cache.set_access_sequence(list(sequence))
        for item in sequence:
            cache.access(item, value=1)
    return cache


A, B, C, D = (0, 0), (0, 1), (0, 2), (0, 3)


# --------------------------------------------------------------------------- #
# Hand-computed known values
# --------------------------------------------------------------------------- #
def test_hand_computed_cases():
    assert belady_simulate([A, B, C, B, A, C], 2)[:2] == (2, 4)
    assert belady_simulate([A, B, C, B, A, C], 3)[:2] == (3, 3)
    assert belady_simulate([A, A, B, B, A], 1)[:2] == (2, 3)
    assert belady_simulate([A, B, A, B], 2)[:2] == (2, 2)
    assert belady_simulate([A, A, A, A], 1)[:2] == (3, 1)


def test_cold_start_first_touch_always_misses():
    hits, misses, _ = belady_simulate([A, B, C], 5)
    assert (hits, misses) == (0, 3)  # all compulsory misses


def test_capacity_covers_working_set_only_compulsory_misses():
    accesses = [A, B, C, A, B, C, A]
    distinct = len(set(accesses))
    _, misses, _ = belady_simulate(accesses, distinct)
    assert misses == distinct


def test_capacity_one_only_consecutive_repeats_hit():
    hits, misses, _ = belady_simulate([A, A, B, A, A], 1)
    # Hits: the 2nd A, then the 4th->5th run contributes one hit on the 5th.
    assert hits == 2
    assert misses == 3


def test_edge_cases():
    assert belady_simulate([], 4) == (0, 0, 0.0)
    assert belady_simulate([A, B, A], 0) == (0, 3, 0.0)
    assert belady_simulate([A, B, A], -1) == (0, 3, 0.0)
    assert belady_simulate([A], 1) == (0, 1, 0.0)


def test_conservation_and_hit_rate_bounds():
    accesses = [A, B, C, A, D, B, A, C, D, B]
    hits, misses, rate = belady_simulate(accesses, 2)
    assert hits + misses == len(accesses)
    assert 0.0 <= rate <= 1.0
    assert rate == pytest.approx(hits / len(accesses))


def test_determinism():
    accesses = [A, B, C, A, D, B, A, C, D, B]
    assert belady_simulate(accesses, 3) == belady_simulate(accesses, 3)


# --------------------------------------------------------------------------- #
# Optimality against brute force + cross-implementation agreement (fuzz)
# --------------------------------------------------------------------------- #
def test_matches_brute_force_optimum_and_cross_impl():
    rng = random.Random(20260101)
    for _ in range(120):
        length = rng.randrange(1, 15)
        capacity = rng.randrange(1, 7)
        accesses = random_trace(rng, length)

        optimum = opt_min_misses(accesses, capacity)
        b_hits, b_misses, _ = belady_simulate(accesses, capacity)

        assert b_misses == optimum, (accesses, capacity)
        assert b_hits + b_misses == length
        assert b_misses <= lru_misses(accesses, capacity)
        if capacity >= len(set(accesses)):
            assert b_misses == len(set(accesses))

        for cache_cls in (ExpertCache, HeapExpertCache):
            cache = run_strategy_cache(cache_cls, accesses, capacity)
            assert cache.misses == optimum, (cache_cls, accesses, capacity)
            assert cache.hits + cache.misses == length


def test_no_belady_anomaly_misses_non_increasing_in_capacity():
    rng = random.Random(7)
    for _ in range(25):
        accesses = random_trace(rng, rng.randrange(6, 16))
        previous = None
        for capacity in range(1, 8):
            misses = belady_simulate(accesses, capacity)[1]
            if previous is not None:
                assert misses <= previous
            previous = misses


def test_lazy_heap_handles_stale_entries():
    # Repeated hits re-push heap entries; eviction must still pick the true
    # farthest-future victim despite stale entries lingering in the heap.
    accesses = [A, A, A, B, B, C, A, D, A, C]
    for capacity in (1, 2, 3):
        assert belady_simulate(accesses, capacity)[1] == opt_min_misses(
            accesses, capacity
        )


# --------------------------------------------------------------------------- #
# build_future_positions
# --------------------------------------------------------------------------- #
def test_build_future_positions_lists_ascending_occurrences():
    accesses = [A, B, A, C, A]
    future = build_future_positions(accesses)
    assert list(future[A]) == [0, 2, 4]
    assert list(future[B]) == [1]
    assert list(future[C]) == [3]


def test_build_future_positions_absent_key_is_empty():
    future = build_future_positions([A, A])
    assert list(future[B]) == []  # defaultdict yields an empty deque


# --------------------------------------------------------------------------- #
# extract_access_sequence
# --------------------------------------------------------------------------- #
def test_extract_skips_layers_and_negative_padding():
    decode = np.array([[[0, 1]], [[2, -1]], [[3, 3]]])  # [L=3, T=1, K=2]
    records = [{"topk_ids_of_layer": decode}, {"topk_ids_of_layer": decode}]
    accesses = extract_access_sequence(
        records, ignore_layers=1, include_prefill=False
    )
    assert accesses == [(1, 2), (2, 3), (2, 3)]


def test_extract_include_prefill_toggle():
    rec = {"topk_ids_of_layer": np.array([[[5]], [[6]]])}  # [L=2, T=1, K=1]
    only_decode = extract_access_sequence(
        [rec, rec], ignore_layers=0, include_prefill=False
    )
    with_prefill = extract_access_sequence(
        [rec, rec], ignore_layers=0, include_prefill=True
    )
    assert only_decode == [(0, 5), (1, 6)]
    assert with_prefill == [(0, 5), (1, 6), (0, 5), (1, 6)]


def test_extract_multi_token_flatten_order():
    # [L=2, T=2, K=2]; reshape flattens token-major within a layer.
    tensor = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    accesses = extract_access_sequence(
        [{"topk_ids_of_layer": tensor}],
        ignore_layers=0,
        include_prefill=True,
    )
    assert accesses == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (1, 7),
    ]


def test_extract_ignore_all_layers_and_empty():
    rec = {"topk_ids_of_layer": np.array([[[1]], [[2]]])}
    assert (
        extract_access_sequence([rec], ignore_layers=5, include_prefill=True)
        == []
    )
    assert (
        extract_access_sequence([], ignore_layers=0, include_prefill=True) == []
    )


def test_extract_works_on_torch_tensors():
    torch = pytest.importorskip("torch")
    tensor = torch.tensor([[[0, 1]], [[2, -1]], [[3, 3]]])
    accesses = extract_access_sequence(
        [{"topk_ids_of_layer": tensor}],
        ignore_layers=1,
        include_prefill=True,
    )
    assert accesses == [(1, 2), (2, 3), (2, 3)]
