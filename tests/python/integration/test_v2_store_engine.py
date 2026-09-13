# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Model-agnosticism gate: the engine reads a moe-store v2 store built
from a synthetic provider with no transformers/HuggingFace involvement,
and every tensor round-trips byte-exact through the C++ read path.

The engine session runs in a subprocess: native teardown after
clean_up_resources can crash the interpreter (pre-existing engine quirk,
same family as the dispatcher-destruction teardown race), and isolating
it keeps the assertion phase - which must fully pass before the marker
prints - unaffected."""

import os
import re
import subprocess
import sys

import pytest
import torch
from moe_store.convert.planner import TensorSpec, dtype_token, plan_layout
from moe_store.convert.writer import write_store

NUM_LAYERS = 2
NUM_EXPERTS = 4

_SESSION_SCRIPT = r"""
import sys

import torch

from moe_store.index import read_index
from moe_infinity import _store as store_lib
from moe_infinity.utils.topology import build_topology_specs

store_dir = sys.argv[1]
state = torch.load(sys.argv[2], weights_only=True)
index = read_index(store_dir)

engine = store_lib.prefetch_handle(store_dir + "/", 0.5)
assert engine.is_tensor_index_initialized()

name_to_id = {m.name: m.tensor_id for _, m in index.iter_members()}
for name in state:
    assert engine.is_tensor_offloaded(name_to_id[name]), name
max_id = max(name_to_id.values())
assert not engine.is_tensor_offloaded(max_id + 1)

buffers = {}
for name, original in state.items():
    buffer = torch.zeros_like(original)
    engine.register(buffer, name_to_id[name])
    buffers[name] = buffer

topo = []
for group in index.groups:
    ids = [[m.tensor_id for m in group.members]]
    if group.is_expert:
        stage = f"model.layers.{group.layer_id}.mlp.experts"
        if topo and topo[-1][0] == stage:
            topo[-1][1].extend(ids)
        else:
            topo.append((stage, ids))
    else:
        topo.append((group.members[0].name.rsplit(".", 1)[0], ids))
engine.set_topology_v2(build_topology_specs(topo))

for group, member in index.iter_members():
    if group.is_expert:
        continue
    buffer = buffers[member.name]
    assert tuple(buffer.shape) == tuple(state[member.name].shape)
    assert torch.equal(buffer.cpu(), state[member.name]), member.name

request_id = 0
for group in index.groups:
    if not group.is_expert:
        continue
    engine.fetch_tensors(request_id, [m.tensor_id for m in group.members])
    for member in group.members:
        buffer = buffers[member.name]
        assert tuple(buffer.shape) == tuple(state[member.name].shape)
        assert torch.equal(buffer.cpu(), state[member.name]), member.name
    request_id += 1

print("V2_STORE_ENGINE_BYTE_EXACT_PASS", flush=True)
"""


def _synthetic_state() -> dict[str, torch.Tensor]:
    torch.manual_seed(1234)
    state = {"model.embed_tokens.weight": torch.randn(64, 32)}
    for layer in range(NUM_LAYERS):
        state[f"model.layers.{layer}.self_attn.q_proj.weight"] = torch.randn(
            32, 32
        )
        for expert in range(NUM_EXPERTS):
            for slot in ("gate_proj", "up_proj", "down_proj"):
                key = f"model.layers.{layer}.mlp.experts.{expert}.{slot}.weight"
                state[key] = torch.randn(16, 32, dtype=torch.bfloat16)
    state["lm_head.weight"] = torch.randn(64, 32)
    return state


def _expert_of(name):
    match = re.search(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.", name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


@pytest.mark.gpu
def test_engine_reads_synthetic_v2_store_byte_exact(tmp_path):
    state = _synthetic_state()
    specs = [
        TensorSpec(
            name,
            t.numel() * t.element_size(),
            dtype_token(t.dtype),
            tuple(t.shape),
        )
        for name, t in state.items()
    ]
    index = plan_layout(
        specs,
        _expert_of,
        model_type="synthetic",
        checkpoint_name="synthetic/mock-provider",
    )
    store_dir = tmp_path / "store"
    write_store(index, lambda name: state[name], store_dir)
    state_path = tmp_path / "state.pt"
    torch.save(state, state_path)
    script_path = tmp_path / "session.py"
    script_path.write_text(_SESSION_SCRIPT)

    env = dict(os.environ, MKL_THREADING_LAYER="GNU")
    result = subprocess.run(
        [sys.executable, str(script_path), str(store_dir), str(state_path)],
        capture_output=True,
        text=True,
        timeout=240,
        env=env,
    )
    assert (
        "V2_STORE_ENGINE_BYTE_EXACT_PASS" in result.stdout
    ), f"stdout={result.stdout[-2000:]}\nstderr={result.stderr[-2000:]}"
