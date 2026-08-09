# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

_MASK32 = 0xFFFFFFFF


def build_topology_specs(topology):
    num_stages = len(topology)
    specs = []
    for stage_idx, (name, groups) in enumerate(topology):
        is_sparse = len(groups) > 1
        is_last_stage = stage_idx == num_stages - 1
        # corr_id mirrors C++ ArcherTopologyHandle::InitializeTopology: low 32
        # bits = stage index, high 32 bits = group index, except the last stage
        # whose high 32 bits are the 0xFFFFFFFF end-of-pipeline marker.
        corr_ids = [
            (stage_idx & _MASK32)
            | ((_MASK32 if is_last_stage else group_idx & _MASK32) << 32)
            for group_idx in range(len(groups))
        ]
        specs.append((name, is_sparse, groups, corr_ids))
    return specs
