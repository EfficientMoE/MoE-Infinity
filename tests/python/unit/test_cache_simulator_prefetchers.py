"""Tests for composable prefetch x eviction policies and the timing model.

* ``PrefetchCachePolicy(evictor, None)`` (and with a prefetcher that never
  predicts) reproduces every evictor's hits exactly.
* Invariants for every evictor x prefetcher x admission mode x capacity:
  ``hits + misses == len(accesses)``, capacity (buffer included) never
  exceeded, demand hit rate <= Belady, and the prefetch accounting identity.
* Online prefetchers are causal: changing the future of a trace does not
  change what they predict before it.
* Timing-model hand cases.
"""

import random

import pytest

from benchmarks.cache_simulator.belady_cache_simulator import belady_simulate
from benchmarks.cache_simulator.policies import (
    BYPASS_POLICIES,
    GEOMETRY_POLICIES,
    LCNU_POLICIES,
    ORACLE_POLICIES,
    POLICY_FACTORIES,
    LayerCycleNextUsePolicy,
    LRUPolicy,
    build_policy,
    simulate,
)
from benchmarks.cache_simulator.prefetchers import (
    ADMISSION_MODES,
    ORACLE_PREFETCHERS,
    PREFETCHER_FACTORIES,
    CoActivationPrefetcher,
    KNNPrefetcher,
    LogisticPrefetcher,
    OraclePrefetcher,
    PrefetchCachePolicy,
    PrefetchPolicy,
    SameIdPrefetcher,
    build_composed,
    build_prefetcher,
)
from benchmarks.cache_simulator.research.timing_model import (
    LinkModel,
    qwen3_30b_a3b_link,
    qwen3_235b_a22b_fp8_link,
    replay,
    replay_policy,
    simulate_timing,
    stall_floor_s,
)

LAYERS, EXPERTS, TOPK = 4, 8, 2
SWEEP_EVICTORS = (
    "lru",
    "lfu",
    "lirs",
    "s3fifo",
    "w_tinylfu",
    "layer_lru",
    "lcnu",
)


def decode_trace(
    rng, steps, layers=LAYERS, experts=EXPERTS, topk=TOPK, keep=0.5
):
    """Cyclic decode trace with token-to-token reuse (like real routing)."""
    current = [rng.sample(range(experts), topk) for _ in range(layers)]
    accesses = []
    for _ in range(steps):
        for layer in range(layers):
            picks = [e for e in current[layer] if rng.random() < keep]
            pool = [e for e in range(experts) if e not in picks]
            rng.shuffle(pool)
            picks += pool[: topk - len(picks)]
            current[layer] = picks
            accesses.extend((layer, e) for e in picks)
    return accesses


def evictor(name, cap, layers=LAYERS):
    kwargs = {}
    if name in GEOMETRY_POLICIES:
        kwargs["num_layers"] = layers
    if name in LCNU_POLICIES:
        kwargs["prior"] = TOPK / EXPERTS
    return build_policy(name, cap, **kwargs)


def prefetcher(name, horizon=2, budget=TOPK):
    return build_prefetcher(
        name,
        horizon=horizon,
        budget=budget,
        num_layers=LAYERS,
        num_experts=EXPERTS,
    )


class _Silent(PrefetchPolicy):
    """Never predicts anything (exercises the composed demand path)."""

    def _predict(self, layer, d):
        return []


ON_DEMAND = sorted(
    name for name in POLICY_FACTORIES if name not in ORACLE_POLICIES | {"score"}
)


# --------------------------------------------------------------------------- #
# Bit-identical composition
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ON_DEMAND)
def test_no_prefetcher_is_bit_identical(name):
    rng = random.Random(1)
    for trial in range(4):
        accesses = decode_trace(rng, 40)
        for cap in (1, 3, 6, 11):
            ref = evictor(name, cap)
            wrapped = PrefetchCachePolicy(evictor(name, cap), None)
            for key in accesses:
                assert wrapped.on_access(key) == ref.on_access(key)
            assert (wrapped.hits, wrapped.misses) == (ref.hits, ref.misses)
            assert sum(wrapped.visit_fresh) == ref.misses


@pytest.mark.parametrize("name", SWEEP_EVICTORS)
@pytest.mark.parametrize(
    "admission", [m for m in ADMISSION_MODES if m != "buffer"]
)
def test_silent_prefetcher_reproduces_evictor(name, admission):
    rng = random.Random(2)
    accesses = decode_trace(rng, 60)
    for cap in (1, 3, 6, 11):
        ref = evictor(name, cap)
        simulate(ref, accesses)
        composed = PrefetchCachePolicy(
            evictor(name, cap),
            _Silent(num_layers=LAYERS),
            admission=admission,
            num_layers=LAYERS,
            prior=TOPK / EXPERTS,
        )
        simulate(composed, accesses)
        assert (composed.hits, composed.misses) == (ref.hits, ref.misses)
        assert composed.prefetch_issued == 0


# --------------------------------------------------------------------------- #
# Invariants for every combination
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ev_name", SWEEP_EVICTORS + ("lcnu_b",))
def test_invariants_every_combination(ev_name):
    rng = random.Random(sum(map(ord, ev_name)))
    accesses = decode_trace(rng, 50)
    bypass = ev_name in BYPASS_POLICIES
    for cap in (2, 5, 8, 14):
        belady_hits = belady_simulate(accesses, cap, bypass=bypass)[0]
        for pf_name in PREFETCHER_FACTORIES:
            for admission in ADMISSION_MODES:
                slots = max(1, cap // 4) if admission == "buffer" else 0
                pol = build_composed(
                    ev_name,
                    cap,
                    prefetcher(pf_name),
                    admission=admission,
                    buffer_slots=slots,
                    num_layers=LAYERS,
                    prior=TOPK / EXPERTS,
                )
                assert pol.capacity == cap
                hits, misses, _ = simulate(pol, accesses)  # asserts capacity
                assert hits + misses == len(accesses)
                assert hits <= belady_hits, (pf_name, admission, cap)
                live = len(pol.staged) + len(pol._spec)
                assert pol.prefetch_issued == (
                    pol.prefetch_covered + pol.prefetch_wasted + live
                )
                assert sum(pol.visit_fresh) + pol.prefetch_covered == misses
                assert pol.uses_oracle == (pf_name in ORACLE_PREFETCHERS)


def test_buffer_is_carved_out_of_the_capacity():
    pol = build_composed(
        "lru", 10, prefetcher("prev_token"), admission="buffer", buffer_slots=3
    )
    assert pol.evictor.capacity == 7
    assert pol.capacity == 10
    with pytest.raises(ValueError):
        build_composed(
            "lru",
            4,
            prefetcher("prev_token"),
            admission="buffer",
            buffer_slots=4,
        )


def test_oracle_prefetch_covers_cold_misses_with_a_large_cache():
    rng = random.Random(3)
    accesses = decode_trace(rng, 20, keep=0.0)
    cap = LAYERS * EXPERTS
    for admission in ("same_queue", "probation", "pin", "do_no_harm"):
        pol = build_composed(
            "lru",
            cap,
            prefetcher("oracle", horizon=1, budget=TOPK),
            admission=admission,
            num_layers=LAYERS,
            prior=TOPK / EXPERTS,
        )
        simulate(pol, accesses)
        belady_misses = len(accesses) - belady_simulate(accesses, cap)[0]
        # Every compulsory miss after the first visit is prefetched.
        assert sum(pol.visit_fresh) == TOPK
        assert pol.prefetch_covered == belady_misses - TOPK
        assert pol.prefetch_wasted == 0
        stats = pol.prediction_stats()
        assert stats[1] == (1.0, 1.0)


def test_do_no_harm_never_evicts_a_sooner_needed_expert_for_a_guess():
    # One slot per layer of headroom: the cyclic scan keeps the cache full.
    rng = random.Random(4)
    accesses = decode_trace(rng, 80, keep=0.9)
    cap = LAYERS * TOPK
    plain = build_composed(
        "lcnu", cap, None, num_layers=LAYERS, prior=TOPK / EXPERTS
    )
    simulate(plain, accesses)
    for name in ("coactivation", "topk_ema"):
        guarded = build_composed(
            "lcnu",
            cap,
            prefetcher(name, horizon=1),
            admission="do_no_harm",
            num_layers=LAYERS,
            prior=TOPK / EXPERTS,
        )
        simulate(guarded, accesses)
        # Admission is gated on LCNU's own next-use estimate, so the demand
        # hit rate cannot collapse the way naive same-queue admission can.
        assert guarded.hits >= 0.9 * plain.hits


# --------------------------------------------------------------------------- #
# Prefetchers
# --------------------------------------------------------------------------- #
def _predictions(pf, accesses):
    """Predictions after every visit, as the composed policy would ask."""
    out = []
    last = None
    for key in accesses:
        if last is not None and key[0] != last:
            out.append(pf.predict())
        pf.observe(key)
        last = key[0]
    return out


@pytest.mark.parametrize(
    "name", sorted(set(PREFETCHER_FACTORIES) - set(ORACLE_PREFETCHERS))
)
def test_online_prefetchers_are_causal(name):
    rng = random.Random(5)
    prefix = decode_trace(rng, 12)
    future_a = decode_trace(random.Random(6), 8)
    future_b = decode_trace(random.Random(7), 8)
    future_b.reverse()  # a different future, same length
    cut = len(prefix) // TOPK  # visits fully inside the prefix
    for horizon in (1, 3):
        pred_a = _predictions(prefetcher(name, horizon, 4), prefix + future_a)
        pred_b = _predictions(prefetcher(name, horizon, 4), prefix + future_b)
        assert pred_a[: cut - 1] == pred_b[: cut - 1]
        assert any(c for p in pred_a[: cut - 1] for _, _, c in p)
        assert pred_a != pred_b


def test_oracle_prefetcher_reads_the_future():
    rng = random.Random(8)
    accesses = decode_trace(rng, 6)
    pf = OraclePrefetcher(horizon=2, budget=TOPK, num_layers=LAYERS)
    pf.bind_trace(accesses)
    preds = _predictions(pf, accesses)
    for visit, pred in enumerate(preds[:-2]):
        for d, layer, cands in pred:
            start = (visit + d) * TOPK
            assert [k for k, _ in cands] == accesses[start : start + TOPK]
            assert layer == accesses[start][0]


def test_same_id_prefetcher_mirrors_the_engine():
    pf = SameIdPrefetcher(horizon=1, budget=2, num_layers=3)
    for key in ((0, 5), (0, 2), (0, 7)):
        pf.observe(key)
    [(d, layer, cands)] = pf.predict()
    assert (d, layer) == (1, 1)
    assert [k for k, _ in cands] == [(1, 5), (1, 2)]


def test_coactivation_learns_cross_layer_pairs():
    pf = CoActivationPrefetcher(
        horizon=1, budget=1, num_layers=2, num_experts=4
    )
    for _ in range(5):  # expert 1 at layer 0 is always followed by 3 at layer 1
        for key in ((0, 1), (1, 3)):
            pf.observe(key)
            pf.predict()
    pf.observe((0, 1))
    [(_, layer, cands)] = pf.predict()
    assert layer == 1 and cands[0][0] == (1, 3)
    assert cands[0][1] == pytest.approx(1.0)


def test_prefetcher_argument_validation():
    with pytest.raises(ValueError):
        build_prefetcher("prev_token", horizon=0)
    with pytest.raises(ValueError):
        build_prefetcher("topk_ema", budget=0)
    with pytest.raises(KeyError):
        build_prefetcher("nope")
    with pytest.raises(ValueError):
        PrefetchCachePolicy(LRUPolicy(4), None, admission="nope")
    with pytest.raises(ValueError):
        PrefetchCachePolicy(build_policy("single_layer_ahead", 4), None)


# --------------------------------------------------------------------------- #
# Timing model
# --------------------------------------------------------------------------- #
def _link(tx=1.0, compute=0.5):
    return LinkModel(
        bytes_per_expert=tx, link_bytes_per_s=1.0, layer_compute_s=compute
    )


def test_timing_without_prefetch_is_the_sum_of_miss_stalls():
    rng = random.Random(9)
    accesses = decode_trace(rng, 30)
    for cap in (2, 6, 12):
        link = _link(tx=2.0, compute=0.7)
        report = simulate_timing(evictor("lcnu", cap), accesses, link)
        misses = len(accesses) - simulate(evictor("lcnu", cap), accesses)[0]
        assert report.stall_s == pytest.approx(misses * 2.0)
        assert report.demand_transfers == misses
        assert report.prefetch_transfers == 0
        assert report.tokens == pytest.approx(30)


def test_infinite_bandwidth_prefetch_has_zero_stall():
    rng = random.Random(10)
    accesses = decode_trace(rng, 20, keep=0.0)
    pol = build_composed(
        "lru",
        LAYERS * EXPERTS,
        prefetcher("oracle", horizon=1),
        num_layers=LAYERS,
    )
    simulate(pol, accesses)
    link = LinkModel(1.0, float("inf"), 0.5)
    assert replay_policy(pol, link, num_layers=LAYERS).stall_s == 0.0
    # With ample compute every prefetch lands in time: only the first visit
    # (no prediction yet) stalls.
    slow = replay_policy(pol, _link(tx=1.0, compute=10.0), num_layers=LAYERS)
    assert slow.stall_s == pytest.approx(TOPK * 1.0)
    assert slow.covered_late == 0 and slow.prefetch_promoted == 0


def test_late_prefetch_stalls_for_the_remainder():
    # Visit 0: 1 fresh miss, prefetch 0 issued for visit 1; compute 0.25.
    # The prefetch starts after the miss (t=1) and ends at t=2; visit 1 starts
    # at t=1.25, so it waits 0.75 for the prefetch.
    report = replay([1, 0], [0], [1], [-1], _link(tx=1.0, compute=0.25))
    assert report.stall_s == pytest.approx(1.0 + 0.75)
    assert report.covered_late == 1


def test_demand_priority_overtakes_queued_prefetches():
    # Visit 0 issues 3 prefetches that are never used; visit 1 has a miss.
    log = ([0, 1], [0, 0, 0], [-1, -1, -1], [-1, -1, -1])
    link = _link(tx=1.0, compute=0.5)
    fifo = replay(*log, link, demand_priority=False)
    prio = replay(*log, link, demand_priority=True)
    # FIFO: the miss waits behind all three; priority: only behind the one
    # that started during visit 0's compute.
    assert fifo.stall_s == pytest.approx(3.0 + 1.0 - 0.5)
    assert prio.stall_s == pytest.approx(1.0 - 0.5 + 1.0)
    assert prio.wasted_transfers >= 1


def test_queued_prefetches_are_served_earliest_target_first():
    # Two prefetches issued at visit 0, for visits 3 and 1. Earliest-target
    # order lands the visit-1 expert half a transfer late; issue order makes
    # visit 1 fetch it on demand behind the visit-3 prefetch.
    log = ([1, 0, 0, 0], [0, 0], [3, 1], [-1, -1])
    link = _link(tx=1.0, compute=0.5)
    edf = replay(*log, link, pf_target=[3, 1])
    in_order = replay(*log, link)
    assert edf.stall_s == pytest.approx(1.0 + 0.5)
    assert in_order.stall_s == pytest.approx(1.0 + 1.5)
    assert edf.covered_late == 1 and edf.covered_on_time == 1
    assert in_order.prefetch_promoted == 1


def test_cancelled_prefetch_costs_nothing():
    # Two prefetches issued at visit 0 (busy link until t=3). The first starts
    # at t=3 and is wasted; the second is removed at visit 1 (gate t=3.1)
    # before it can start at t=4, so it is cancelled. FIFO: the miss of visit
    # 1 waits for the first prefetch only.
    report = replay(
        [3, 1],
        [0, 0],
        [-1, -1],
        [-1, 1],
        _link(tx=1.0, compute=0.1),
        demand_priority=False,
    )
    assert report.prefetch_cancelled == 1
    assert report.prefetch_transfers == 1
    assert report.wasted_transfers == 1
    assert report.stall_s == pytest.approx(3.0 + (5.0 - 3.1))


def test_promoted_prefetch_is_a_demand_transfer():
    # The first prefetch holds the link through visit 1's gate, so the second
    # is still queued when its expert is needed: fetched as a demand miss.
    report = replay(
        [2, 0], [0, 0], [-1, 1], [-1, -1], _link(tx=1.0, compute=0.1)
    )
    assert report.prefetch_promoted == 1
    assert report.demand_transfers == 3
    assert report.stall_s == pytest.approx(2.0 + (4.0 - 2.1))


def test_presets_and_stall_floor():
    small = qwen3_30b_a3b_link()
    big = qwen3_235b_a22b_fp8_link()
    assert small.bytes_per_expert == 9_437_184
    assert big.bytes_per_expert == 18_874_368
    assert small.expert_transfer_s == pytest.approx(9_437_184 / 57.1e9)
    assert small.layer_compute_s == pytest.approx(113_770_496 / 1.792e12)
    assert big.layer_compute_s == pytest.approx(223_346_688 / 1.792e12)
    assert qwen3_30b_a3b_link(compute_scale=2.0).layer_compute_s == (
        pytest.approx(2 * small.layer_compute_s)
    )
    assert stall_floor_s(10, 4, _link(tx=1.0, compute=0.5)) == 8.0
    assert stall_floor_s(1, 4, _link(tx=1.0, compute=0.5)) == 0.0


# --------------------------------------------------------------------------- #
# Round 3: more predictors, LCNU admission control and prefill handling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs",
    [
        {"depth": 2},
        {"depth": 3, "decay": 0.5},
        {"scoring": "bayes"},
        {"depth": 2, "scoring": "bayes"},
    ],
)
def test_coactivation_variants_are_causal(kwargs):
    prefix = decode_trace(random.Random(11), 12)
    future_a = decode_trace(random.Random(12), 8)
    future_b = list(reversed(decode_trace(random.Random(13), 8)))
    cut = len(prefix) // TOPK

    def make():
        return CoActivationPrefetcher(
            horizon=2,
            budget=4,
            num_layers=LAYERS,
            num_experts=EXPERTS,
            **kwargs,
        )

    pred_a = _predictions(make(), prefix + future_a)
    pred_b = _predictions(make(), prefix + future_b)
    assert pred_a[: cut - 1] == pred_b[: cut - 1]
    assert any(c for p in pred_a[: cut - 1] for _, _, c in p)


def _cross_layer_trace(steps):
    """Layer 1 always routes ``(e + 1) % 4`` after layer 0 routes ``e``."""
    rng = random.Random(14)
    out = []
    for _ in range(steps):
        e = rng.randrange(4)
        out += [(0, e), (1, (e + 1) % 4)]
    return out


@pytest.mark.parametrize(
    "make",
    [
        lambda: LogisticPrefetcher(
            horizon=1, budget=1, num_layers=2, num_experts=4, lr=0.5, top_k=1
        ),
        lambda: KNNPrefetcher(
            horizon=1, budget=1, num_layers=2, num_experts=4, context=1, top_k=1
        ),
        lambda: CoActivationPrefetcher(
            horizon=1, budget=1, num_layers=2, num_experts=4, scoring="bayes"
        ),
    ],
)
def test_new_predictors_learn_a_cross_layer_rule(make):
    pf = make()
    trace = _cross_layer_trace(200)
    preds = _predictions(pf, trace)
    # After a warm-up, the layer-1 prediction made after each layer-0 visit
    # is the rule's expert.
    right = total = 0
    for i, pred in enumerate(preds[100:], start=100):
        if trace[i][0] != 0:
            continue
        ((_, layer, cands),) = pred
        assert layer == 1
        total += 1
        right += bool(cands) and cands[0][0] == (1, (trace[i][1] + 1) % 4)
    assert right == total > 0


def test_min_score_filters_low_confidence_candidates():
    pf = CoActivationPrefetcher(
        horizon=1, budget=4, num_layers=2, num_experts=4, min_score=0.6
    )
    trace = _cross_layer_trace(50)
    for pred in _predictions(pf, trace)[10:]:
        for _, _, cands in pred:
            assert all(score >= 0.6 for _, score in cands)


def test_lcnu_bypass_keeps_the_sooner_needed_expert():
    # cap 1, one layer: A is routed on every visit, B once. With bypass the
    # one-off B is served without displacing A.
    trace = [(0, 0), (0, 1), (0, 0), (0, 0)]
    plain = LayerCycleNextUsePolicy(1, num_layers=1, prior=0.5)
    bypass = LayerCycleNextUsePolicy(
        1, num_layers=1, prior=0.5, bypass="always"
    )
    for key in [(0, 0)] * 6:  # make A frequent
        plain.on_access(key)
        bypass.on_access(key)
    for key in trace:
        plain.on_access(key)
        bypass.on_access(key)
    assert bypass.hits > plain.hits
    assert (0, 0) in bypass.resident


def _union_trace(rng, requests=4, steps=6):
    """Per request: one wide prefill visit per layer, then decode visits."""
    out = []
    for _ in range(requests):
        for layer in range(LAYERS):
            out += [(layer, e) for e in rng.sample(range(EXPERTS), 6)]
        out += decode_trace(rng, steps)
    return out


def test_lcnu_prefill_handling():
    rng = random.Random(15)
    trace = _union_trace(rng)
    decode_only = decode_trace(random.Random(16), 30)
    for cap in (2, 4, 8):
        # On decode-only streams the prefill variants are plain LCNU.
        ref = simulate(evictor("lcnu", cap), decode_only)[0]
        for name in ("lcnu_p",):
            pol = build_policy(
                name,
                cap,
                num_layers=LAYERS,
                prior=TOPK / EXPERTS,
                decode_width=TOPK,
            )
            assert simulate(pol, decode_only)[0] == ref
        # With prefill visits: bounded by Belady with bypass.
        bound = belady_simulate(trace, cap, bypass=True)[0]
        for name in ("lcnu_p", "lcnu_b"):
            pol = build_policy(
                name,
                cap,
                num_layers=LAYERS,
                prior=TOPK / EXPERTS,
                decode_width=TOPK,
            )
            hits, misses, _ = simulate(pol, trace)
            assert hits <= bound and hits + misses == len(trace)
    # prefill_weight=0: a prefill-only expert gets no routing evidence.
    pol = LayerCycleNextUsePolicy(
        64, num_layers=LAYERS, prior=0.25, decode_width=TOPK, prefill_weight=0.0
    )
    simulate(pol, trace)
    prefill_only = {k for k in trace[: LAYERS * 6]} - set(trace[LAYERS * 6 :])
    for key in prefill_only:
        assert pol._count[key] == 0


@pytest.mark.parametrize("name", ["knn", "coactivation", "hybrid", "logistic"])
def test_decode_width_ignores_prefill_visits(name):
    rng = random.Random(17)
    union = _union_trace(rng)
    decode = [k for i, k in enumerate(union) if not _in_wide_visit(union, i)]

    def make():
        return build_prefetcher(
            name,
            horizon=1,
            budget=4,
            num_layers=LAYERS,
            num_experts=EXPERTS,
            decode_width=TOPK,
        )

    with_prefill = [p for p in _predictions(make(), union) if p]
    decode_only = [p for p in _predictions(make(), decode) if p]
    # Prefill visits yield no predictions and teach nothing.
    assert with_prefill == decode_only[: len(with_prefill)]
    assert len(with_prefill) >= len(decode_only) - 1


def _in_wide_visit(trace, i):
    layer = trace[i][0]
    lo = i
    while lo > 0 and trace[lo - 1][0] == layer:
        lo -= 1
    hi = i
    while hi + 1 < len(trace) and trace[hi + 1][0] == layer:
        hi += 1
    return hi - lo + 1 > TOPK
