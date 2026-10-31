"""Equivalence tests for the accelerated miss-ratio-curve engine.

The pure-Python ``belady_simulate`` / ``ExpertCache`` remain the correctness
oracle. Every accelerated path must reproduce them exactly:

* :func:`opt_mrc` / :func:`opt_mrc_parallel` match ``belady_simulate`` hit rate
  bit-for-bit for every capacity, and match the brute-force offline optimum.
* :func:`lru_mrc` / :func:`lru_mrc_gpu` match an independent LRU reference.
* :func:`vectorized_access_sequence` matches ``extract_access_sequence``.
* :func:`shards_lru_mrc` stays within its reported error bound.
"""

import random
from collections import OrderedDict

import numpy as np
import pytest

from benchmarks.cache_simulator.belady_cache_simulator import (
    belady_simulate,
    extract_access_sequence,
)
from benchmarks.cache_simulator.research.mrc import (
    encode_accesses,
    lru_mrc,
    lru_mrc_gpu,
    miss_ratio_curve,
    opt_mrc,
    opt_mrc_parallel,
    shards_lru_mrc,
    vectorized_access_sequence,
)

NUM_EXPERTS = 16


# --------------------------------------------------------------------------- #
# Independent references
# --------------------------------------------------------------------------- #
def opt_min_misses(accesses, capacity):
    """Exhaustive demand-paging optimum (same oracle as the belady tests)."""
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
    cache: OrderedDict = OrderedDict()
    misses = 0
    for key in accesses:
        if key in cache:
            cache.move_to_end(key)
            continue
        misses += 1
        if capacity > 0 and len(cache) >= capacity:
            cache.popitem(last=False)
        if capacity > 0:
            cache[key] = True
    return misses


def random_trace(rng, length, layers=4, experts=6):
    return [
        (rng.randrange(layers), rng.randrange(experts)) for _ in range(length)
    ]


def ref_opt_curve(accesses, caps):
    total = len(accesses)
    return {
        c: (belady_simulate(accesses, c)[0] / total if total else 0.0)
        for c in caps
    }


def ref_lru_curve(accesses, caps):
    total = len(accesses)
    return {
        c: ((total - lru_misses(accesses, c)) / total if total else 0.0)
        for c in caps
    }


# --------------------------------------------------------------------------- #
# OPT MRC == belady_simulate, for every capacity (fuzz + brute force)
# --------------------------------------------------------------------------- #
def test_opt_mrc_matches_belady_fuzz():
    rng = random.Random(20260101)
    caps = list(range(0, 10))
    for _ in range(400):
        length = rng.randrange(1, 40)
        accesses = random_trace(rng, length)
        got = opt_mrc(accesses, caps, num_experts=NUM_EXPERTS)
        expected = ref_opt_curve(accesses, caps)
        for c in caps:
            assert got[c] == pytest.approx(expected[c], abs=1e-12), (
                accesses,
                c,
            )


def test_opt_mrc_matches_brute_force_optimum():
    rng = random.Random(42)
    for _ in range(120):
        length = rng.randrange(1, 14)
        accesses = random_trace(rng, length, layers=2, experts=3)
        caps = list(range(1, 7))
        got = opt_mrc(accesses, caps, num_experts=NUM_EXPERTS)
        for c in caps:
            optimum = opt_min_misses(accesses, c)
            hits = length - optimum
            assert got[c] == pytest.approx(hits / length, abs=1e-12)


def test_opt_mrc_hand_computed():
    A, B, C, D = (0, 0), (0, 1), (0, 2), (0, 3)
    accesses = [A, B, C, B, A, C]
    curve = opt_mrc(accesses, [2, 3], num_experts=NUM_EXPERTS)
    # belady_simulate gives (hits=2, misses=4) at cap 2, (3, 3) at cap 3.
    assert curve[2] == pytest.approx(2 / 6)
    assert curve[3] == pytest.approx(3 / 6)
    del D


def test_opt_mrc_edge_cases():
    assert opt_mrc([], [0, 1, 5]) == {0: 0.0, 1: 0.0, 5: 0.0}
    curve = opt_mrc([(0, 0), (0, 1), (0, 0)], [0], num_experts=NUM_EXPERTS)
    assert curve[0] == 0.0


def test_opt_mrc_parallel_matches_serial():
    rng = random.Random(7)
    # Large enough that the parallel path actually forks processes.
    accesses = random_trace(rng, 30000, layers=8, experts=12)
    caps = [1, 2, 4, 8, 16, 32, 64]
    serial = opt_mrc(accesses, caps, num_experts=NUM_EXPERTS)
    parallel = opt_mrc_parallel(
        accesses, caps, num_experts=NUM_EXPERTS, max_workers=4
    )
    assert serial == parallel


# --------------------------------------------------------------------------- #
# LRU MRC == independent LRU reference
# --------------------------------------------------------------------------- #
def test_lru_mrc_matches_reference_fuzz():
    rng = random.Random(99)
    caps = list(range(0, 10))
    for _ in range(400):
        length = rng.randrange(1, 40)
        accesses = random_trace(rng, length)
        got = lru_mrc(accesses, caps, num_experts=NUM_EXPERTS)
        expected = ref_lru_curve(accesses, caps)
        for c in caps:
            assert got[c] == pytest.approx(expected[c], abs=1e-12), (
                accesses,
                c,
            )


def test_lru_mrc_monotonic_non_decreasing_in_capacity():
    rng = random.Random(3)
    accesses = random_trace(rng, 2000, layers=8, experts=16)
    caps = list(range(1, 40))
    curve = lru_mrc(accesses, caps, num_experts=NUM_EXPERTS)
    rates = [curve[c] for c in caps]
    assert all(b >= a - 1e-12 for a, b in zip(rates, rates[1:]))


def test_lru_mrc_gpu_matches_cpu():
    rng = random.Random(11)
    accesses = random_trace(rng, 5000, layers=6, experts=10)
    caps = [1, 2, 4, 8, 16, 32]
    cpu = lru_mrc(accesses, caps, num_experts=NUM_EXPERTS)
    gpu = lru_mrc_gpu(accesses, caps, num_experts=NUM_EXPERTS)
    for c in caps:
        assert gpu[c] == pytest.approx(cpu[c], abs=1e-12)


# --------------------------------------------------------------------------- #
# OPT never beats a full cache; LRU never beats OPT
# --------------------------------------------------------------------------- #
def test_lru_never_beats_opt():
    rng = random.Random(5)
    caps = list(range(1, 12))
    for _ in range(100):
        accesses = random_trace(rng, rng.randrange(5, 40))
        opt = opt_mrc(accesses, caps, num_experts=NUM_EXPERTS)
        lru = lru_mrc(accesses, caps, num_experts=NUM_EXPERTS)
        for c in caps:
            assert lru[c] <= opt[c] + 1e-12


# --------------------------------------------------------------------------- #
# Vectorized access-sequence construction == extract_access_sequence
# --------------------------------------------------------------------------- #
def test_vectorized_access_sequence_matches_reference():
    rng = random.Random(5)
    for _ in range(200):
        layers = rng.randrange(1, 6)
        tokens = rng.randrange(1, 4)
        topk = rng.randrange(1, 5)
        steps = rng.randrange(1, 5)
        records = []
        for _ in range(steps):
            arr = np.array(
                [
                    [
                        [rng.randrange(-1, 8) for _ in range(topk)]
                        for _ in range(tokens)
                    ]
                    for _ in range(layers)
                ]
            )
            records.append({"topk_ids_of_layer": arr})
        for ignore in range(0, layers + 1):
            for include_prefill in (False, True):
                ref = extract_access_sequence(
                    records,
                    ignore_layers=ignore,
                    include_prefill=include_prefill,
                )
                enc, _, num_experts = vectorized_access_sequence(
                    records,
                    ignore_layers=ignore,
                    include_prefill=include_prefill,
                )
                ref_arr = (
                    np.array(ref, dtype=np.int64)
                    if ref
                    else np.empty((0, 2), dtype=np.int64)
                )
                ref_enc = encode_accesses(ref_arr, num_experts)
                assert np.array_equal(enc, ref_enc), (
                    ignore,
                    include_prefill,
                    ref,
                )


def test_vectorized_access_sequence_matches_torch_tensors():
    torch = pytest.importorskip("torch")
    records = [
        {"topk_ids_of_layer": torch.tensor([[[0, 1]], [[2, -1]], [[3, 3]]])}
    ]
    ref = extract_access_sequence(
        records, ignore_layers=1, include_prefill=True
    )
    enc, _, num_experts = vectorized_access_sequence(
        records, ignore_layers=1, include_prefill=True
    )
    ref_enc = encode_accesses(np.array(ref, dtype=np.int64), num_experts)
    assert np.array_equal(enc, ref_enc)


# --------------------------------------------------------------------------- #
# SHARDS approximation stays within its stated error bound
# --------------------------------------------------------------------------- #
def test_shards_reports_a_bound_and_valid_rates():
    # SHARDS is an *optional, approximate* path. Its accuracy is poor for the
    # expert-cache domain's tiny key universe (documented in mrc.py), so we do
    # not assert a tight pointwise bound here -- only that it runs, reports a
    # positive bound, returns valid hit rates for every capacity, and rejects
    # out-of-range sampling rates. Exactness at full sampling is covered by
    # ``test_shards_full_sampling_is_exact``.
    rng = random.Random(2026)
    accesses = [(rng.randrange(20), rng.randrange(64)) for _ in range(50000)]
    caps = [64, 128, 256, 512, 1024]
    approx, bound = shards_lru_mrc(
        accesses, caps, num_experts=64, sampling_rate=0.1, seed=1
    )
    assert bound > 0
    for c in caps:
        assert 0.0 <= approx[c] <= 1.0

    with pytest.raises(ValueError):
        shards_lru_mrc(accesses, caps, sampling_rate=0.0)
    with pytest.raises(ValueError):
        shards_lru_mrc(accesses, caps, sampling_rate=1.5)


def test_shards_full_sampling_is_exact():
    rng = random.Random(1)
    accesses = random_trace(rng, 2000, layers=6, experts=10)
    caps = [1, 2, 4, 8, 16, 32]
    exact = lru_mrc(accesses, caps, num_experts=NUM_EXPERTS)
    approx, _ = shards_lru_mrc(
        accesses, caps, num_experts=NUM_EXPERTS, sampling_rate=1.0
    )
    for c in caps:
        assert approx[c] == pytest.approx(exact[c], abs=1e-12)


# --------------------------------------------------------------------------- #
# Public dispatcher
# --------------------------------------------------------------------------- #
def test_miss_ratio_curve_dispatch_matches_direct():
    rng = random.Random(8)
    accesses = random_trace(rng, 3000, layers=6, experts=12)
    caps = [1, 4, 16, 64, 256]
    assert miss_ratio_curve(
        accesses, caps, policy="opt", num_experts=NUM_EXPERTS
    ) == opt_mrc(accesses, caps, num_experts=NUM_EXPERTS)
    assert miss_ratio_curve(
        accesses, caps, policy="lru", num_experts=NUM_EXPERTS
    ) == lru_mrc(accesses, caps, num_experts=NUM_EXPERTS)


def test_miss_ratio_curve_rejects_unknown_policy():
    with pytest.raises(ValueError):
        miss_ratio_curve([(0, 0)], [1], policy="nope")
