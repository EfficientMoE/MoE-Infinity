# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import torch

import moe_infinity._store as store
from moe_infinity.utils.topology import build_topology_specs

MASK32 = 0xFFFFFFFF
MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"
NUM_HIDDEN_LAYERS = 27
FIRST_K_DENSE_REPLACE = 1
NUM_ROUTED_EXPERTS = 64


def _deepseek_v2_lite_topology():
    """Contract-level topology shaped from DeepSeek-V2-Lite config values."""
    next_tensor_id = 0
    topology = [("model.embed_tokens", [[next_tensor_id]])]
    next_tensor_id += 1

    for layer_id in range(NUM_HIDDEN_LAYERS):
        topology.append((f"model.layers.{layer_id}", [[next_tensor_id]]))
        next_tensor_id += 1
        if layer_id >= FIRST_K_DENSE_REPLACE:
            groups = [
                [next_tensor_id + expert_id]
                for expert_id in range(NUM_ROUTED_EXPERTS)
            ]
            topology.append((f"model.layers.{layer_id}.mlp.experts", groups))
            next_tensor_id += NUM_ROUTED_EXPERTS

    topology.append(("model.norm", [[next_tensor_id]]))
    next_tensor_id += 1
    topology.append(("lm_head", [[next_tensor_id]]))
    return topology, next_tensor_id + 1


def _snapshot(tmp_path, api_name):
    topology, tensor_count = _deepseek_v2_lite_topology()
    store_path = tmp_path / api_name
    store_path.mkdir()
    handle = store.prefetch_handle(str(store_path), 0.01)
    try:
        for tensor_id in range(tensor_count):
            tensor = torch.tensor([tensor_id], dtype=torch.float32)
            handle.offload(tensor, tensor_id)
        if api_name == "v1":
            handle.set_topology(topology)
        else:
            handle.set_topology_v2(build_topology_specs(topology))
        return [tuple(item) for item in handle.get_topology_snapshot()]
    finally:
        handle.clean_up_resources()


@torch.no_grad()
def test_deepseek_v2_lite_v1_v2_topology_metadata_and_placement_match(
    tmp_path,
):
    assert torch.cuda.device_count() == 6
    topology, _ = _deepseek_v2_lite_topology()

    legacy = _snapshot(tmp_path, "v1")
    enriched = _snapshot(tmp_path, "v2")

    assert enriched == legacy

    first_sparse_stage = 3
    first_sparse_node = sum(len(groups) for _, groups in topology[:3])
    assert legacy[first_sparse_node] == (first_sparse_stage, True, 3)
    assert legacy[first_sparse_node + 63] == (
        first_sparse_stage | (63 << 32),
        True,
        0,
    )

    last_stage = len(topology) - 1
    assert last_stage == 55
    assert legacy[-1][0] == last_stage | (MASK32 << 32)
    assert legacy[-1][1] is False
    assert legacy[-1][2] == 5

    first_six_sparse_devices = [
        item[2]
        for item in legacy[first_sparse_node : first_sparse_node + 6]
    ]
    assert first_six_sparse_devices == [3, 4, 5, 0, 1, 2]
