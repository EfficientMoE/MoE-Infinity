"""Behavioral tests for cache_strategy data structures.

Covers the cache key (``ExpertIndex``), the probe/insert and decode accounting
semantics of ``ExpertCache`` / ``HeapExpertCache``, and the ``AccessSequence``
trace parser. Belady optimality itself is cross-checked in
``test_cache_simulator_belady.py``.
"""

import contextlib
import io

import numpy as np
import pytest

from benchmarks.cache_simulator.cache_strategy import (
    AccessSequence,
    ExpertCache,
    ExpertIndex,
    HeapExpertCache,
)

CACHES = (ExpertCache, HeapExpertCache)


def _quiet_set(cache, sequence):
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        cache.set_access_sequence(list(sequence))


# --------------------------------------------------------------------------- #
# ExpertIndex
# --------------------------------------------------------------------------- #
def test_expert_index_equality_and_hashing():
    assert ExpertIndex(0, 0) == ExpertIndex(0, 0)
    assert ExpertIndex(0, 0) != ExpertIndex(0, 1)
    assert hash(ExpertIndex(2, 3)) == hash(ExpertIndex(2, 3))
    assert len({ExpertIndex(0, 0), ExpertIndex(0, 0), ExpertIndex(1, 0)}) == 2


def test_expert_index_ordering_is_expert_then_layer():
    assert ExpertIndex(0, 9) < ExpertIndex(1, 0)  # expert dominates
    assert ExpertIndex(1, 0) < ExpertIndex(1, 2)  # tie on expert -> layer
    ordered = sorted([ExpertIndex(2, 0), ExpertIndex(0, 0), ExpertIndex(1, 0)])
    assert ordered == [ExpertIndex(0, 0), ExpertIndex(1, 0), ExpertIndex(2, 0)]


def test_expert_index_equality_across_numpy_ints():
    # Parsed traces produce numpy integer ids; keys must still match.
    assert ExpertIndex(np.int64(3), np.int64(1)) == ExpertIndex(3, 1)
    assert hash(ExpertIndex(np.int64(3), np.int64(1))) == hash(
        ExpertIndex(3, 1)
    )


# --------------------------------------------------------------------------- #
# ExpertCache / HeapExpertCache semantics
# --------------------------------------------------------------------------- #
def test_probe_without_value_never_inserts():
    for cache_cls in CACHES:
        cache = cache_cls(2)
        _quiet_set(cache, [ExpertIndex(0, 0), ExpertIndex(0, 0)])
        assert cache.access(ExpertIndex(0, 0)) is None  # miss, no insert
        assert cache.misses == 1
        assert cache.hits == 0
        assert len(cache.cache) == 0
        assert cache.access(ExpertIndex(0, 0)) is None  # still a miss
        assert cache.misses == 2


def test_hit_returns_stored_value():
    for cache_cls in CACHES:
        cache = cache_cls(2)
        _quiet_set(cache, [ExpertIndex(0, 0), ExpertIndex(0, 0)])
        assert cache.access(ExpertIndex(0, 0), value=42) is None  # insert
        assert cache.access(ExpertIndex(0, 0), value=42) == 42  # hit


def test_capacity_is_never_exceeded():
    sequence = [ExpertIndex(0, e) for e in [0, 1, 2, 3, 0, 1, 2, 3, 0]]
    for cache_cls in CACHES:
        cache = cache_cls(2)
        _quiet_set(cache, sequence)
        for item in sequence:
            cache.access(item, value=1)
            assert len(cache.cache) <= 2


def test_decode_hit_miss_accounting():
    for cache_cls in CACHES:
        cache = cache_cls(4)
        seq = [ExpertIndex(0, 0), ExpertIndex(0, 0), ExpertIndex(0, 1)]
        _quiet_set(cache, seq)
        cache.access(ExpertIndex(0, 0), value=1, decode=False)  # prefill miss
        cache.access(ExpertIndex(0, 0), value=1, decode=True)  # decode hit
        cache.access(ExpertIndex(0, 1), value=1, decode=True)  # decode miss
        assert (cache.hits, cache.misses) == (1, 2)
        assert (cache.decode_hits, cache.decode_misses) == (1, 1)
        assert cache.get_decode_hit_rate() == 0.5
        assert cache.get_hit_rate() == pytest.approx(1 / 3)


def test_hit_rates_zero_without_accesses():
    for cache_cls in CACHES:
        cache = cache_cls(2)
        assert cache.get_hit_rate() == 0.0
        assert cache.get_decode_hit_rate() == 0.0


def test_evicts_farthest_future_entry():
    # cap=2 over A B C B A C: inserting C must evict A (next use 4) and keep
    # B (next use 3), the Belady choice.
    a, b, c = ExpertIndex(0, 0), ExpertIndex(1, 0), ExpertIndex(2, 0)
    full = [a, b, c, b, a, c]
    for cache_cls in CACHES:
        cache = cache_cls(2)
        _quiet_set(cache, full)
        for item in full[:3]:  # A, B, C
            cache.access(item, value=1)
        assert set(cache.cache.keys()) == {b, c}


# --------------------------------------------------------------------------- #
# AccessSequence parser
# --------------------------------------------------------------------------- #
def _simple_traces():
    prefilling = {0: np.array([[0, 0, 1], [2, 2, 2]])}  # [L=2, T=3]
    decoding = {0: [np.array([[1, 3], [0, 0]])]}  # one iter, [L=2, K=2]
    return prefilling, decoding


def test_access_sequence_prefill_dedup_and_decode_order():
    prefilling, decoding = _simple_traces()
    seq = AccessSequence(prefilling, decoding)
    request = seq.requests[0]
    assert request.prefillAccessSequence == [
        ExpertIndex(0, 0),
        ExpertIndex(1, 0),
        ExpertIndex(2, 1),
    ]
    assert request.decodeAccessSequence == [
        ExpertIndex(1, 0),
        ExpertIndex(3, 0),
        ExpertIndex(0, 1),
        ExpertIndex(0, 1),
    ]


def test_access_sequence_filters_negative_padding():
    # -1 marks an unused top-k slot; it must not leak in as a (layer, -1)
    # cache key, which would occupy capacity and skew hit/miss accounting.
    prefilling = {0: np.array([[0, -1, 1], [-1, -1, 2]])}  # [L=2, T=3]
    decoding = {0: [np.array([[1, -1], [-1, -1]])]}  # one iter, [L=2, K=2]
    request = AccessSequence(prefilling, decoding).requests[0]
    assert request.prefillAccessSequence == [
        ExpertIndex(0, 0),
        ExpertIndex(1, 0),
        ExpertIndex(2, 1),
    ]
    assert request.decodeAccessSequence == [ExpertIndex(1, 0)]
    assert all(e.expert_idx >= 0 for e in request.prefillAccessSequence)
    assert all(e.expert_idx >= 0 for e in request.decodeAccessSequence)


def test_access_sequence_2d_and_3d_decode_are_equivalent():
    prefilling, decoding_2d = _simple_traces()
    decoding_3d = {0: [np.array([[[1, 3]], [[0, 0]]])]}  # [L=2, 1, K=2]
    seq_2d = AccessSequence(prefilling, decoding_2d)
    seq_3d = AccessSequence(prefilling, decoding_3d)
    assert (
        seq_2d.requests[0].decodeAccessSequence
        == seq_3d.requests[0].decodeAccessSequence
    )


def test_access_sequence_rejects_bad_decode_shape():
    prefilling, _ = _simple_traces()
    bad = {0: [np.array([1, 2, 3])]}  # 1-D is unsupported
    with pytest.raises(ValueError):
        AccessSequence(prefilling, bad)


def test_access_sequence_respects_sorted_order_and_counts():
    prefilling = {k: np.array([[k]]) for k in (2, 0, 1)}  # [L=1, T=1]
    decoding = {k: [np.array([[k]])] for k in (2, 0, 1)}  # [L=1, K=1]

    limited = AccessSequence(prefilling, decoding, sequence_num=2)
    assert limited.get_request_count() == 2

    full = AccessSequence(prefilling, decoding)
    # Iteration order follows sorted decode keys 0, 1, 2.
    first_experts = [
        req.prefillAccessSequence[0].expert_idx for req in full.requests
    ]
    assert first_experts == [0, 1, 2]

    one = full.get_access_sequence(count=1)
    assert one == [ExpertIndex(0, 0), ExpertIndex(0, 0)]


def test_access_sequence_get_next_request_exhausts():
    prefilling, decoding = _simple_traces()
    seq = AccessSequence(prefilling, decoding)
    assert seq.get_request_count() == 1
    assert seq.get_next_request() is not None
    assert seq.get_next_request() is None
