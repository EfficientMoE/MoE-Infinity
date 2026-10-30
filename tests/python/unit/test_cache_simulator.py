"""Unit tests for the offline MoE expert-cache simulator.

These exercise the pure-Python simulation core only: no GPU, no torch, and no
trace files on disk.
"""

import numpy as np

from benchmarks.cache_simulator.belady_cache_simulator import (
    belady_simulate,
    extract_access_sequence,
)
from benchmarks.cache_simulator.cache_strategy import (
    ExpertCache,
    ExpertIndex,
    HeapExpertCache,
)
from benchmarks.cache_simulator.fetch_engines import StreamFetch

A, B, C = (0, 0), (0, 1), (0, 2)


def test_belady_matches_hand_computed_optimal():
    # Capacity 2 over A B C B A C: Belady keeps the entry whose next use is
    # nearest, yielding exactly two hits (the second B and the second C).
    accesses = [A, B, C, B, A, C]
    hits, misses, hit_rate = belady_simulate(accesses, capacity=2)
    assert hits == 2
    assert misses == 4
    assert abs(hit_rate - 2 / 6) < 1e-9


def test_belady_capacity_covers_working_set():
    # Capacity >= number of distinct keys: only the first touch of each misses.
    accesses = [A, B, C, B, A, C]
    hits, misses, _ = belady_simulate(accesses, capacity=3)
    assert hits == 3
    assert misses == 3


def test_belady_edge_cases():
    assert belady_simulate([], capacity=4) == (0, 0, 0.0)
    assert belady_simulate([A, B, A], capacity=0) == (0, 3, 0.0)


def test_extract_access_sequence_skips_layers_and_padding():
    # Shape [num_layers=3, seq_len=1, topk=2]; -1 marks an unused slot.
    decode = np.array([[[0, 1]], [[2, -1]], [[3, 3]]])
    records = [{"topk_ids_of_layer": decode}, {"topk_ids_of_layer": decode}]

    accesses = extract_access_sequence(
        records, ignore_layers=1, include_prefill=False
    )
    assert accesses == [(1, 2), (2, 3), (2, 3)]

    # Ignoring every layer yields no accesses.
    assert (
        extract_access_sequence(records, ignore_layers=3, include_prefill=False)
        == []
    )


def test_expert_index_is_hashable_and_comparable():
    assert ExpertIndex(0, 0) == ExpertIndex(0, 0)
    assert hash(ExpertIndex(0, 0)) == hash(ExpertIndex(0, 0))
    assert len({ExpertIndex(0, 0), ExpertIndex(0, 0), ExpertIndex(1, 0)}) == 2


def _run_cache(cache_cls, sequence, cache_size):
    cache = cache_cls(cache_size)
    cache.set_access_sequence(list(sequence))
    for key in sequence:
        cache.access(key, value=1)
    return cache


def test_cache_classes_count_hits_and_misses():
    seq = [ExpertIndex(0, 0), ExpertIndex(0, 1), ExpertIndex(0, 0)]
    for cache_cls in (HeapExpertCache, ExpertCache):
        cache = _run_cache(cache_cls, seq, cache_size=2)
        assert cache.hits == 1
        assert cache.misses == 2
        assert abs(cache.get_hit_rate() - 1 / 3) < 1e-9


def test_cache_warm_working_set_is_all_hits():
    key = ExpertIndex(2, 5)
    for cache_cls in (HeapExpertCache, ExpertCache):
        cache = _run_cache(cache_cls, [key, key, key], cache_size=4)
        assert cache.hits == 2
        assert cache.misses == 1


def test_stream_fetch_respects_capacity_and_fetches_requested():
    states = np.array([[1, 1], [0, 0]])
    engine = StreamFetch(
        n_layer=2, n_expert=2, mem_capacity=2, expert_states=states, stride=1
    )
    assert not engine.access(1, 0)  # not resident yet

    engine.fetch(1, 0)

    assert int(np.sum(engine.expert_states)) == 2  # capacity preserved
    assert engine.access(1, 0)  # requested expert now resident
