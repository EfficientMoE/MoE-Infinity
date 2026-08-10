# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from moe_infinity.utils.topology import build_topology_specs

MASK32 = 0xFFFFFFFF


def _reference(topology):
    n = len(topology)
    out = []
    for layer_id, (name, groups) in enumerate(topology):
        is_sparse = len(groups) > 1
        corr = [
            (layer_id & MASK32) | ((e & MASK32) << 32)
            for e in range(len(groups))
        ]
        if layer_id == n - 1:
            corr = [(c & MASK32) | (MASK32 << 32) for c in corr]
        out.append((name, is_sparse, groups, corr))
    return out


def test_dense_single_stage_is_last_and_marked():
    specs = build_topology_specs([("embed", [[0, 1]])])
    _, is_sparse, _, corr = specs[0]
    assert is_sparse is False
    assert corr == [MASK32 << 32]


def test_sparse_flag_by_group_count():
    topo = [("a", [[0]]), ("b", [[1], [2], [3]]), ("c", [[4]])]
    specs = build_topology_specs(topo)
    assert [s[1] for s in specs] == [False, True, False]


def test_middle_sparse_corr_packing():
    topo = [("a", [[0]]), ("b", [[1], [2], [3]]), ("c", [[4]])]
    specs = build_topology_specs(topo)
    assert specs[1][3] == [1, 1 | (1 << 32), 1 | (2 << 32)]


def test_last_stage_high_bits_marker():
    topo = [("a", [[0]]), ("b", [[1], [2]])]
    specs = build_topology_specs(topo)
    assert all((c >> 32) == MASK32 for c in specs[-1][3])


def test_matches_reference_bitpacking():
    topo = [
        ("embed", [[0]]),
        ("l0", [[1], [2], [3]]),
        ("l1", [[4], [5], [6]]),
        ("head", [[7]]),
    ]
    assert build_topology_specs(topo) == _reference(topo)


def test_empty_topology():
    assert build_topology_specs([]) == []
