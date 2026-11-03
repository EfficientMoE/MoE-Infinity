"""Independent parity tests for SharedState CostFit / CostFit3 (exp675 Stage A).

Subject: ``benchmarks/cache_simulator/research/policies_shared_costfit.py``.
Claim under test: ``shared_costfit`` / ``shared_costfit3`` are BIT-IDENTICAL to
the sealed ``costfit`` / ``costfit3`` on any access stream (same per-access
hit sequence, hits/misses/rates, switches/choice_switches, eviction sequence,
terminal resident order, and full per-key inner state); the sharing is purely
structural (one ``Key -> record`` table instead of one dict per field).

Written independently of the implementation author. Nothing here uses
``census_shared_costfit.py`` (same author as the implementation).

Sections:
  1. SharedFieldView vs ``dict`` conformance (scripted edge cases + fuzz).
  2. Parity on adversarial synthetic streams x capacities {8, 64, 300}.
  3. Shared-table slot lifecycle invariant (see ``test_slot_lifecycle_*``).
  4. Registration (build_policy, GEOMETRY_POLICIES, POLICY_ORDER, geometry).
  5. Determinism / no cross-instance state.
"""

from __future__ import annotations

import functools
import random
from typing import Dict, List, Tuple

import pytest

from benchmarks.cache_simulator.policies import (
    GEOMETRY_POLICIES,
    LayerCycleNextUsePolicy,
    S3FIFOPolicy,
    build_policy,
)
from benchmarks.cache_simulator.research import run_policies_real
from benchmarks.cache_simulator.research.policies_costfit import CostFitPolicy
from benchmarks.cache_simulator.research.policies_costfit3 import CostFit3Policy
from benchmarks.cache_simulator.research.policies_rothead import RotHeadPolicy
from benchmarks.cache_simulator.research.policies_shared_costfit import (
    SharedCostFit3Policy,
    SharedCostFitPolicy,
    SharedFieldView,
    SharedKeyTable,
    SharedLCNU,
    SharedRotHead,
    SharedS3FIFO,
)

Key = Tuple[int, int]

NUM_LAYERS = 8
PRIOR = 1.0 / 16
CAPACITIES = (8, 64, 300)
PAIRS = (("costfit", "shared_costfit"), ("costfit3", "shared_costfit3"))
LCNU_FIELDS = ("_last", "_ema", "_count", "_first")

_MISSING = object()


# =========================================================================== #
# 1. SharedFieldView vs dict conformance
# =========================================================================== #
def _outcome(fn):
    """Run ``fn``; return ("ok", value) or ("raise", exc type, exc args)."""
    try:
        return ("ok", fn())
    except Exception as exc:  # noqa: BLE001 - we compare exception identity
        return ("raise", type(exc), exc.args)


def _check_mirror(table, views, dicts, universe):
    """Every observable of every view equals its mirror dict; table is tight."""
    for v, d in zip(views, dicts):
        for k in universe:
            assert (k in v) == (k in d), k
            assert v.get(k) == d.get(k), k
            assert v.get(k, _MISSING) == d.get(k, _MISSING), k
            assert _outcome(lambda: v[k]) == _outcome(lambda: d[k]), k
    union = set().union(*(d.keys() for d in dicts))
    # A record exists iff at least one slot of it is occupied (no leaks, no
    # premature drops).
    assert set(table.table) == union
    for k, rec in table.table.items():
        assert rec.live == sum(k in d for d in dicts), k
        assert rec.live >= 1, k


def _bump(x):
    return x + 1 if isinstance(x, (int, float)) else 0


def _new_table(n_views):
    t = SharedKeyTable()
    views = [SharedFieldView(t, f"role{i}", "f") for i in range(n_views)]
    return t, views


def test_view_missing_key_semantics_match_dict():
    t, (v,) = _new_table(1)
    d: Dict = {}
    k = (0, 1)
    assert _outcome(lambda: v[k]) == _outcome(lambda: d[k])
    assert _outcome(lambda: v[k])[1] is KeyError
    assert v.get(k) is None and d.get(k) is None
    assert v.get(k, "dflt") == "dflt"
    assert (k in v) is False
    # del of an absent key: KeyError, and it must not create a record.
    assert _outcome(lambda: v.__delitem__(k)) == _outcome(lambda: d.__delitem__(k))
    assert t.table == {}


def test_view_delete_only_occupied_slot_drops_record():
    t, (a, b, c) = _new_table(3)
    k = (3, 7)
    b[k] = 5
    assert set(t.table) == {k}
    assert k in b and k not in a and k not in c
    del b[k]
    assert k not in t.table, "record must be dropped when its last slot is unset"
    assert k not in b
    assert _outcome(lambda: b[k])[1] is KeyError
    assert b.get(k, -1) == -1
    # second delete: KeyError like dict
    with pytest.raises(KeyError):
        del b[k]
    assert t.table == {}


def test_view_delete_one_slot_keeps_others():
    t, (a, b, c) = _new_table(3)
    k = (1, 2)
    a[k] = 1.5
    b[k] = 0
    c[k] = None
    assert t.table[k].live == 3
    del b[k]
    assert k in t.table and t.table[k].live == 2
    assert a[k] == 1.5 and k in c and c[k] is None and c.get(k, "x") is None
    assert k not in b and b.get(k, "x") == "x"
    with pytest.raises(KeyError):
        b[k]
    del a[k]
    assert k in t.table and t.table[k].live == 1
    del c[k]
    assert k not in t.table


def test_view_overwrite_does_not_double_count_live():
    t, (a, b) = _new_table(2)
    k = (0, 0)
    a[k] = 0
    a[k] = 1
    a[k] += 1  # the S3-FIFO ``freq[key] += 1`` pattern
    a[k] = 0
    assert a[k] == 0 and t.table[k].live == 1
    del a[k]
    assert k not in t.table, "overwrite must not inflate the live count"
    # re-insert after delete works and starts fresh
    a[k] = 9
    assert a[k] == 9 and t.table[k].live == 1 and k not in b


def test_view_falsy_values_are_present():
    t, (a,) = _new_table(1)
    for i, val in enumerate((0, 0.0, False, None, "", ())):
        k = (i, i)
        a[k] = val
        assert k in a
        assert a[k] == val
        assert a.get(k, "dflt") == val


def test_alloc_after_records_exist_is_rejected():
    t, (a,) = _new_table(1)
    a[(0, 0)] = 1
    with pytest.raises(RuntimeError):
        SharedFieldView(t, "late", "f")


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_view_fuzz_against_dict(seed):
    rng = random.Random(1000 + seed)
    n_views = rng.choice((1, 2, 3, 5))
    t, views = _new_table(n_views)
    dicts: List[Dict] = [{} for _ in range(n_views)]
    # small universe => heavy cross-view collisions on the same record
    universe = [(l, e) for l in range(3) for e in range(4)] + [7, -1, "k"]
    values = [0, 1, -3, 2.5, 0.0, None, False, "s"]
    for step in range(6000):
        i = rng.randrange(n_views)
        v, d = views[i], dicts[i]
        k = rng.choice(universe)
        op = rng.random()
        if op < 0.30:
            val = rng.choice(values)
            assert _outcome(lambda: v.__setitem__(k, val)) == _outcome(
                lambda: d.__setitem__(k, val)
            )
        elif op < 0.45:
            # read-modify-write, only legal when present (as sealed code does)
            if k in d:
                d[k] = _bump(d[k])
                v[k] = _bump(v[k])
        elif op < 0.70:
            assert _outcome(lambda: v.__delitem__(k)) == _outcome(
                lambda: d.__delitem__(k)
            ), (step, k)
        elif op < 0.80:
            assert _outcome(lambda: v[k]) == _outcome(lambda: d[k]), (step, k)
        elif op < 0.90:
            dflt = rng.choice((None, -7, _MISSING))
            assert v.get(k, dflt) == d.get(k, dflt), (step, k)
        else:
            assert (k in v) == (k in d), (step, k)
        if step % 50 == 0:
            _check_mirror(t, views, dicts, universe)
    _check_mirror(t, views, dicts, universe)
    # drain everything: table must end empty
    for v, d in zip(views, dicts):
        for k in list(d):
            del v[k]
            del d[k]
    _check_mirror(t, views, dicts, universe)
    assert t.table == {}


# =========================================================================== #
# 2. Adversarial stream generators (all fixed-seed)
# =========================================================================== #
def gen_uniform_shrink_expand(n=20000, seed=11) -> List[Key]:
    """Uniform random keys; per-phase universe shrinks then expands."""
    rng = random.Random(seed)
    widths = (64, 24, 6, 1, 3, 32, 160, 2, 96, 256)
    per = n // len(widths)
    out: List[Key] = []
    for w in widths:
        for _ in range(per):
            out.append((rng.randrange(NUM_LAYERS), rng.randrange(w)))
    return out


def gen_bursty(n=20000, seed=12) -> List[Key]:
    """Long runs of a single key, then a jump to a random (often new) key.

    The expert universe (2400 per layer = 8x the largest tested capacity) is
    wide, and runs are short enough, that even the first 8000 accesses stream
    hundreds of DISTINCT one/two-touch keys through the S3-FIFO small queue,
    so the ghost FIFO (m_cap entries, up to 58 at cap 64) overflows and keys
    are pushed out of it (forgotten). A narrower universe or longer runs
    leaves the ghost queue never full.
    """
    rng = random.Random(seed)
    out: List[Key] = []
    while len(out) < n:
        key = (rng.randrange(NUM_LAYERS), rng.randrange(2400))
        run = rng.choice((1, 2, 3, 8, 40, 150)) if rng.random() < 0.5 else rng.randint(1, 10)
        out.extend([key] * run)
    return out[:n]


def gen_hot_rotation(k: int, n=20000, seed=13, experts=64, hot=6, topk=2) -> List[Key]:
    """Layer-cycled top-k routing from per-layer hot sets, rotated every k accesses."""
    rng = random.Random(seed * 7919 + k)
    hs = {l: rng.sample(range(experts), hot) for l in range(NUM_LAYERS)}
    out: List[Key] = []
    while len(out) < n:
        for l in range(NUM_LAYERS):
            pool = hs[l] if rng.random() < 0.85 else range(experts)
            for e in rng.sample(list(pool), topk):
                out.append((l, e))
                if len(out) % k == 0:
                    hs = {ll: rng.sample(range(experts), hot) for ll in range(NUM_LAYERS)}
    return out[:n]


def gen_single_layer(n=20000, seed=14) -> List[Key]:
    """Every access on layer 0 (LCNU current-layer path only)."""
    rng = random.Random(seed)
    hot = rng.sample(range(128), 12)
    out: List[Key] = []
    for i in range(n):
        if i % 3000 == 0:
            hot = rng.sample(range(128), 12)
        e = rng.choice(hot) if rng.random() < 0.7 else rng.randrange(128)
        out.append((0, e))
    return out


def gen_first_touch_flood(n=20000) -> List[Key]:
    """Monotone first-touch: every access is a never-seen key (all misses)."""
    return [(i % NUM_LAYERS, i // NUM_LAYERS) for i in range(n)]


def gen_regime_flip(n=64000, seed=0, seg=16384, rotate_p=1.0, experts=64, hot=6, topk=2) -> List[Key]:
    """Alternate stable / fast-rotating segments so the arbiters actually switch."""
    rng = random.Random(seed)
    hs = {l: rng.sample(range(experts), hot) for l in range(NUM_LAYERS)}
    out: List[Key] = []
    while len(out) < n:
        fast = (len(out) // seg) % 2 == 1
        for l in range(NUM_LAYERS):
            pool = hs[l] if rng.random() < 0.9 else range(experts)
            for e in rng.sample(list(pool), topk):
                out.append((l, e))
        if fast and rng.random() < rotate_p:
            l = rng.randrange(NUM_LAYERS)
            hs[l] = rng.sample(range(experts), hot)
    return out[:n]


STREAMS = {
    "uniform_shrink_expand": gen_uniform_shrink_expand,
    "bursty": gen_bursty,
    "rotation_k17": functools.partial(gen_hot_rotation, 17),
    "rotation_k97": functools.partial(gen_hot_rotation, 97),
    "rotation_k1024": functools.partial(gen_hot_rotation, 1024),
    "single_layer": gen_single_layer,
    "first_touch_flood": gen_first_touch_flood,
    "regime_flip": gen_regime_flip,
}


@functools.lru_cache(maxsize=None)
def stream(name: str) -> Tuple[Key, ...]:
    return tuple(STREAMS[name]())


# =========================================================================== #
# Instrumented runner
# =========================================================================== #
def _leaves(policy) -> List[Tuple[str, object]]:
    """(path, leaf CachePolicy) for every LCNU / S3-FIFO cache inside ``policy``.

    Structure is taken from the SEALED classes (costfit: lcnu, s3fifo;
    costfit3: lcnu, s3fifo, rothead{lcnu, s3fifo}), not from the shared module.
    """
    out = []
    for name, inner in policy._inners.items():
        if isinstance(inner, RotHeadPolicy):
            out.append((f"{name}.lcnu", inner.lcnu))
            out.append((f"{name}.s3fifo", inner.s3fifo))
        else:
            out.append((name, inner))
    return out


def _install_evict_recorders(policy, log: list, clock: list) -> None:
    """Wrap every leaf's base ``_evict`` hook; log (path, access_index, key)."""
    for path, leaf in _leaves(policy):
        orig = leaf._evict

        def rec(key, _orig=orig, _path=path):
            log.append((_path, clock[0], key))
            return _orig(key)

        leaf._evict = rec  # instance attribute shadows the method


def _leaf_state(leaf, universe) -> tuple:
    """Order structures + full per-key state through the public dict surface."""
    if isinstance(leaf, S3FIFOPolicy):
        fields = ("freq",)
        orders = (
            tuple(leaf.resident),
            tuple(leaf.small),
            tuple(leaf.main),
            tuple(leaf.ghost),
        )
    else:
        assert isinstance(leaf, LayerCycleNextUsePolicy)
        fields = LCNU_FIELDS
        orders = (
            tuple(leaf.resident),
            tuple(sorted((l, tuple(sorted(s))) for l, s in leaf._by_layer.items())),
            tuple(sorted(leaf._visits.items())),
            leaf._cur,
            tuple(sorted(leaf._used_now)),
        )
    per_key = []
    for f in fields:
        d = getattr(leaf, f)
        per_key.append((f, tuple((k, k in d, d[k] if k in d else None) for k in universe)))
    per_key = tuple(per_key)
    return (leaf.hits, leaf.misses, orders, per_key)


def run_case(name: str, cap: int, accesses) -> dict:
    pol = build_policy(name, cap, num_layers=NUM_LAYERS, prior=PRIOR)
    log: list = []
    clock = [0]
    _install_evict_recorders(pol, log, clock)
    hitbits = bytearray()
    for i, key in enumerate(accesses):
        clock[0] = i
        hitbits.append(1 if pol.on_access(key) else 0)
    universe = sorted(set(accesses))
    rothead = pol._inners.get("rothead")
    return {
        "policy": pol,
        "hitbits": bytes(hitbits),
        "hits": pol.hits,
        "misses": pol.misses,
        "hit_rate": pol.hit_rate,
        "effective_hit_rate": pol.effective_hit_rate,
        "switches": pol.switches,
        "choice_switches": getattr(pol, "choice_switches", None),
        "choice": pol.choice,
        "resident_order": tuple(pol.resident),
        "resident_sorted": tuple(sorted(pol.resident)),
        "evictions": tuple(log),
        "mass": {k: tuple(v) for k, v in pol._mass.items()},
        "sw_mass": pol._sw_mass,
        "rothead": None
        if rothead is None
        else (rothead.choice, rothead.switches, rothead.hits, rothead.misses, tuple(rothead.buffer)),
        "leaves": tuple((path, _leaf_state(leaf, universe)) for path, leaf in _leaves(pol)),
    }


def fingerprint(r: dict) -> tuple:
    return tuple((k, r[k]) for k in sorted(r) if k != "policy")


@functools.lru_cache(maxsize=None)
def cached_case(name: str, cap: int, stream_name: str) -> dict:
    return run_case(name, cap, stream(stream_name))


# =========================================================================== #
# 2. Parity matrix
# =========================================================================== #
@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: p[1])
@pytest.mark.parametrize("cap", CAPACITIES)
@pytest.mark.parametrize("stream_name", list(STREAMS))
def test_parity_on_adversarial_streams(stream_name, cap, pair):
    sealed_name, shared_name = pair
    a = cached_case(sealed_name, cap, stream_name)
    b = cached_case(shared_name, cap, stream_name)

    # The shared run must really be shared (else parity is vacuous).
    assert type(a["policy"]) is (CostFitPolicy if sealed_name == "costfit" else CostFit3Policy)
    assert type(b["policy"]) is (SharedCostFitPolicy if shared_name == "shared_costfit" else SharedCostFit3Policy)
    for _, leaf in _leaves(a["policy"]):
        assert type(leaf) in (LayerCycleNextUsePolicy, S3FIFOPolicy)
    for _, leaf in _leaves(b["policy"]):
        assert type(leaf) in (SharedLCNU, SharedS3FIFO)

    # Headline accounting (brief items), then everything else.
    assert a["hits"] == b["hits"]
    assert a["misses"] == b["misses"]
    assert a["hits"] + a["misses"] == len(stream(stream_name))
    assert a["hit_rate"] == b["hit_rate"]
    assert a["effective_hit_rate"] == b["effective_hit_rate"]
    assert a["switches"] == b["switches"]
    assert a["choice_switches"] == b["choice_switches"]
    if sealed_name == "costfit3":
        assert isinstance(b["choice_switches"], dict)
    assert a["choice"] == b["choice"]
    assert a["resident_sorted"] == b["resident_sorted"]
    assert a["resident_order"] == b["resident_order"]
    assert len(a["evictions"]) == len(b["evictions"])
    assert a["evictions"] == b["evictions"]
    assert a["hitbits"] == b["hitbits"]
    assert a["mass"] == b["mass"] and a["sw_mass"] == b["sw_mass"]
    assert a["rothead"] == b["rothead"]
    for (pa, sa), (pb, sb) in zip(a["leaves"], b["leaves"]):
        assert pa == pb
        assert sa == sb, f"inner state diverged in {pa}"
    assert fingerprint(a) == fingerprint(b)


def test_matrix_exercises_the_interesting_paths():
    """Guard against a vacuous parity matrix: the streams must actually hit
    evictions, outer switches (both directions), and every costfit3 choice."""
    flood = cached_case("costfit", 64, "first_touch_flood")
    assert flood["hits"] == 0 and flood["misses"] == len(stream("first_touch_flood"))
    assert all(k[0] == 0 for k in stream("single_layer"))
    for sname in STREAMS:
        for cap in CAPACITIES:
            r = cached_case("costfit3", cap, sname)
            if cap == 8:
                assert r["evictions"], (sname, cap)
    cf_switches = max(cached_case("costfit", c, s)["switches"] for s in STREAMS for c in CAPACITIES)
    assert cf_switches >= 2, "no costfit case switched back"
    cf3_adopted = {
        n
        for s in STREAMS
        for c in CAPACITIES
        for n, v in cached_case("costfit3", c, s)["choice_switches"].items()
        if v
    }
    rothead_internal = max(
        cached_case("costfit3", c, s)["rothead"][1] for s in STREAMS for c in CAPACITIES
    )
    assert "s3fifo" in cf3_adopted or "rothead" in cf3_adopted, cf3_adopted
    assert rothead_internal >= 1


def test_parity_costfit3_multi_switch_stream():
    """Dedicated stream where costfit3 adopts s3fifo AND rothead (seed found by
    sweeping the sealed policy only), at a capacity outside the matrix."""
    acc = gen_regime_flip(n=64000, seed=1, seg=12288, rotate_p=1.0)
    a = run_case("costfit3", 64, acc)
    b = run_case("shared_costfit3", 64, acc)
    assert a["switches"] >= 2
    assert sum(1 for v in a["choice_switches"].values() if v) >= 2
    assert fingerprint(a) == fingerprint(b)
    c = run_case("costfit", 64, acc)
    d = run_case("shared_costfit", 64, acc)
    assert c["switches"] >= 2
    assert fingerprint(c) == fingerprint(d)


@pytest.mark.parametrize("cap", [1, 2, 3, 10])
def test_parity_tiny_capacities(cap):
    """Degenerate S3-FIFO geometry (s_cap=1, m_cap in {0,1,2,9})."""
    acc = stream("bursty")[:6000] + stream("rotation_k17")[:6000]
    for sealed, shared in PAIRS:
        assert fingerprint(run_case(sealed, cap, acc)) == fingerprint(run_case(shared, cap, acc))


# =========================================================================== #
# 3. Shared-table slot lifecycle
# =========================================================================== #
# Invariant (derived from the sealed policies in policies.py):
#   * LCNU._record_access writes _last/_ema/_count/_first for EVERY accessed
#     key and never deletes them -> each LCNU field's key set == the set of
#     distinct keys accessed so far (permanent history).
#   * S3FIFO.freq is set to 0 on admission (_on_miss), bumped on hit, reset
#     to 0 (kept) on promotion small->main, and DELETED whenever the key is
#     evicted (from small into ghost, or from main). Ghost entries carry no
#     freq. -> freq key set == small U main == set(resident), always; a key in
#     ghost, or dropped out of ghost, owns no freq slot.
#   Therefore for the shared table:
#     (I1) every record has live == #occupied slots >= 1 (no empty records);
#     (I2) per LCNU role, occupied keys == distinct keys accessed;
#     (I3) per S3FIFO role, occupied keys == set(resident) == small U main,
#          disjoint from ghost, and equal (keys AND values) to the sealed freq;
#     (I4) table keys == union of occupied keys over all roles — for the
#          composites this equals the distinct-keys-accessed set (LCNU's
#          permanent history dominates); for a standalone SharedS3FIFO table
#          it equals set(resident) exactly (records of forgotten keys freed).
def _views_of(leaf):
    if isinstance(leaf, S3FIFOPolicy):
        return [("freq", leaf.freq)]
    return [(f, getattr(leaf, f)) for f in LCNU_FIELDS]


def _assert_table_invariant(shared_pol, sealed_pol, accesses):
    table = shared_pol.shared_table
    universe = set(accesses)
    sh_leaves = _leaves(shared_pol)
    se_leaves = dict(_leaves(sealed_pol))
    # one table for all leaves (structural claim)
    for _, leaf in sh_leaves:
        assert leaf.shared_table is table
    if "rothead" in shared_pol._inners:
        rh = shared_pol._inners["rothead"]
        assert type(rh) is SharedRotHead and rh.shared_table is table
    all_views = [(p, f, v) for p, leaf in sh_leaves for f, v in _views_of(leaf)]
    for _, _, v in all_views:
        assert isinstance(v, SharedFieldView)
    assert len(table.labels) == len(all_views) == len(set(table.labels))

    assert set(table.table) <= universe
    # (I1)
    for k, rec in table.table.items():
        occ = sum(k in v for _, _, v in all_views)
        assert rec.live == occ >= 1, k
    # (I2)/(I3) + exact equality to the sealed per-key dicts
    for path, leaf in sh_leaves:
        sealed_leaf = se_leaves[path]
        for f, v in _views_of(leaf):
            occupied = {k for k in universe if k in v}
            sealed_dict = getattr(sealed_leaf, f)
            assert type(sealed_dict) is dict
            assert occupied == set(sealed_dict), (path, f)
            assert {k: v[k] for k in occupied} == sealed_dict, (path, f)
            if isinstance(leaf, S3FIFOPolicy):
                assert occupied == set(leaf.resident) == set(leaf.small) | set(leaf.main)
                assert occupied.isdisjoint(leaf.ghost)
            else:
                assert occupied == universe
    # (I4)
    union = {k for _, _, v in all_views for k in universe if k in v}
    assert set(table.table) == union == universe


@pytest.mark.parametrize("stream_name", ["uniform_shrink_expand", "first_touch_flood", "rotation_k17"])
@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: p[1])
def test_slot_lifecycle_composite(stream_name, pair):
    sealed_name, shared_name = pair
    for cap in (8, 64):
        a = cached_case(sealed_name, cap, stream_name)
        b = cached_case(shared_name, cap, stream_name)
        _assert_table_invariant(b["policy"], a["policy"], stream(stream_name))


@pytest.mark.parametrize("stream_name", ["uniform_shrink_expand", "first_touch_flood", "bursty", "rotation_k97"])
@pytest.mark.parametrize("cap", [1, 8, 64])
def test_slot_lifecycle_standalone_s3fifo_frees_forgotten_keys(stream_name, cap):
    """Without LCNU history in the table, records must track residency exactly:
    keys sitting in ghost or dropped out of ghost own NO record. Checked at
    every step (prefix), against a sealed S3FIFO stepped in lockstep."""
    t = SharedKeyTable()
    sh = SharedS3FIFO(cap, table=t, role="solo")
    se = S3FIFOPolicy(cap)
    ever_ghosted = set()
    acc = stream(stream_name)[:8000]
    for i, key in enumerate(acc):
        assert sh.on_access(key) == se.on_access(key), i
        ever_ghosted.update(sh.ghost)
        if i % 97 == 0 or i == len(acc) - 1:
            assert set(t.table) == set(sh.resident) == set(se.freq), i
            assert tuple(sh.ghost) == tuple(se.ghost)
            for k in t.table:
                assert t.table[k].live == 1 and sh.freq[k] == se.freq[k]
            for k in sh.ghost:
                assert k not in sh.freq and k not in t.table
    # some keys must have cycled through ghost and been forgotten, else the
    # check above is weaker than intended
    if cap >= 8:
        forgotten = ever_ghosted - set(sh.ghost) - set(sh.resident)
        assert forgotten, "stream never pushed keys out of ghost"
        assert not (forgotten & set(t.table))


# =========================================================================== #
# 4. Registration
# =========================================================================== #
def test_registration_build_policy_classes():
    p = build_policy("shared_costfit", 16, num_layers=NUM_LAYERS, prior=PRIOR)
    q = build_policy("shared_costfit3", 16, num_layers=NUM_LAYERS, prior=PRIOR)
    assert type(p) is SharedCostFitPolicy and isinstance(p, CostFitPolicy)
    assert type(q) is SharedCostFit3Policy and isinstance(q, CostFit3Policy)
    assert list(p._inners) == ["lcnu", "s3fifo"]
    assert list(q._inners) == ["lcnu", "s3fifo", "rothead"]
    assert type(q._inners["rothead"]) is SharedRotHead
    # geometry forwarded identically to the sealed builds
    for pol in (p, q):
        lc = pol._inners["lcnu"]
        assert lc.num_layers == NUM_LAYERS and lc.prior == PRIOR and pol.capacity == 16
    assert q._inners["rothead"].lcnu.num_layers == NUM_LAYERS
    assert q._inners["rothead"].lcnu.prior == PRIOR


def test_registration_geometry_and_policy_order():
    assert {"shared_costfit", "shared_costfit3"} <= GEOMETRY_POLICIES
    order = run_policies_real.POLICY_ORDER
    assert order.count("shared_costfit") == 1 and order.count("shared_costfit3") == 1
    i3 = order.index("costfit3")
    assert order.index("shared_costfit") > i3
    assert order.index("shared_costfit3") > i3
    assert "shared_costfit" in run_policies_real.PRETTY
    assert "shared_costfit3" in run_policies_real.PRETTY
    # real-trace sweeps must hand the shared variants the same geometry
    stats = {"num_layers": 48, "max_expert_id": 127, "top_k": 8}
    gk = run_policies_real.geometry_kwargs
    assert gk("shared_costfit", stats) == gk("costfit", stats)
    assert gk("shared_costfit3", stats) == gk("costfit3", stats)
    assert "prior" in gk("shared_costfit3", stats)


# =========================================================================== #
# 5. Determinism / isolation
# =========================================================================== #
@pytest.mark.parametrize("name", ["costfit", "shared_costfit", "costfit3", "shared_costfit3"])
def test_determinism_same_case_twice(name):
    acc = stream("rotation_k97")
    r1 = fingerprint(run_case(name, 64, acc))
    r2 = fingerprint(run_case(name, 64, acc))
    assert r1 == r2
    assert r1 == fingerprint(cached_case(name, 64, "rotation_k97"))


def test_no_cross_instance_state_interleaved():
    """Two shared instances stepped interleaved on different streams must equal
    their solo runs (no class-level / module-level shared table leakage)."""
    s1 = stream("bursty")[:8000]
    s2 = stream("rotation_k17")[:8000]
    solo1 = run_case("shared_costfit3", 64, s1)
    solo2 = run_case("shared_costfit3", 64, s2)
    p1 = build_policy("shared_costfit3", 64, num_layers=NUM_LAYERS, prior=PRIOR)
    p2 = build_policy("shared_costfit3", 64, num_layers=NUM_LAYERS, prior=PRIOR)
    assert p1.shared_table is not p2.shared_table
    h1, h2 = bytearray(), bytearray()
    for a, b in zip(s1, s2):
        h1.append(p1.on_access(a))
        h2.append(p2.on_access(b))
    assert bytes(h1) == solo1["hitbits"] and bytes(h2) == solo2["hitbits"]
    assert tuple(p1.resident) == solo1["resident_order"]
    assert tuple(p2.resident) == solo2["resident_order"]
