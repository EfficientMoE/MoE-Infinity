"""Tests for the real-trace sweep driver (``research/run_policies_real.py``).

Covers the three prefill stream modes on merged multi-request dumps, the
geometry passed to geometry-aware policies, and that the parallel sweep
reproduces the serial one with every policy bounded by Belady.
"""

import random

import numpy as np
import pytest

from benchmarks.cache_simulator.research.run_policies_real import (
    geometry_kwargs,
    load_accesses,
    run_sweep,
    union_prefill_sequence,
)


def _records():
    prefill = np.array(
        [
            [[0, 1], [1, 2], [0, 1]],  # layer 0, 3 prompt tokens, top-2
            [[3, 3], [4, -1], [3, 5]],  # layer 1 (with padding)
        ]
    )
    decode_a = np.array([[[0, 2]], [[3, 4]]])
    decode_b = np.array([[[1, 2]], [[5, 3]]])
    return [
        {"topk_ids_of_layer": prefill, "forward_mode": 1},
        {"topk_ids_of_layer": decode_a, "forward_mode": 2},
        {"topk_ids_of_layer": decode_b, "forward_mode": 2},
    ]


def test_union_prefill_fetches_each_expert_once_per_layer():
    accesses = union_prefill_sequence(_records(), ignore_layers=0)
    prefill_part = [(0, 0), (0, 1), (0, 2), (1, 3), (1, 4), (1, 5)]
    decode_part = [
        (0, 0),
        (0, 2),
        (1, 3),
        (1, 4),
        (0, 1),
        (0, 2),
        (1, 5),
        (1, 3),
    ]
    assert accesses == prefill_part + decode_part


def test_union_prefill_respects_ignore_layers():
    accesses = union_prefill_sequence(_records(), ignore_layers=1)
    assert accesses == [(1, 3), (1, 4), (1, 5), (1, 3), (1, 4), (1, 5), (1, 3)]


def test_load_accesses_prefill_modes(tmp_path):
    torch = pytest.importorskip("torch")
    for i in range(2):  # two single-request dumps
        records = [
            {
                "topk_ids_of_layer": torch.tensor(rec["topk_ids_of_layer"]),
                "forward_mode": rec["forward_mode"],
            }
            for rec in _records()
        ]
        torch.save({"records": records}, str(tmp_path / f"req{i}.pt"))
    files = sorted(str(p) for p in tmp_path.glob("*.pt"))
    counts = {}
    for mode in ("skip", "union", "raw"):
        accesses, stats = load_accesses(files, ignore_layers=0, prefill=mode)
        counts[mode] = len(accesses)
        assert stats["num_layers"] == 2
        assert stats["top_k"] == 2
        assert stats["max_expert_id"] == 5
    assert counts == {"skip": 2 * 8, "union": 2 * (8 + 6), "raw": 2 * (8 + 11)}
    with pytest.raises(ValueError):
        load_accesses(files, ignore_layers=0, prefill="bogus")


def test_geometry_kwargs_only_for_geometry_policies():
    stats = {"num_layers": 48, "max_expert_id": 127, "top_k": 8}
    assert geometry_kwargs("lru", stats) == {}
    assert geometry_kwargs("layer_lru", stats) == {"num_layers": 48}
    assert geometry_kwargs("lcnu", stats) == {
        "num_layers": 48,
        "prior": pytest.approx(8 / 128),
    }


def test_parallel_sweep_matches_serial_and_respects_belady():
    rng = random.Random(3)
    layers, experts, topk = 4, 8, 2
    accesses = [
        (layer, e)
        for _ in range(40)
        for layer in range(layers)
        for e in rng.sample(range(experts), topk)
    ]
    stats = {
        "num_layers": layers,
        "max_expert_id": experts - 1,
        "top_k": topk,
    }
    policies = ["lru", "lfu", "arc", "lirs", "layer_lru", "lcnu"]
    serial = run_sweep(
        accesses, [3, 6, 12], "m", "d", stats, policies=policies, workers=1
    )
    parallel = run_sweep(
        accesses, [3, 6, 12], "m", "d", stats, policies=policies, workers=3
    )
    assert serial == parallel
    belady = {r.capacity: r.hit_rate for r in serial if r.policy == "belady"}
    for row in serial:
        assert row.hit_rate <= belady[row.capacity] + 1e-9
