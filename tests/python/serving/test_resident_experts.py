"""Resident experts through the grouped decode kernel and a pointer table."""

from __future__ import annotations

import types

import pytest
import torch

from moe_infinity.serving import resident_experts
from moe_infinity.serving.resident_experts import (
    ResidentExpertTable,
    install_grouped_resident_experts,
)
from tests.python.ops.conftest import requires_cuda


def _experts(num_experts=8, hidden=256, inter=128, device="cpu"):
    from transformers.models.qwen3_moe.configuration_qwen3_moe import (
        Qwen3MoeConfig,
    )
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeExperts,
    )

    config = Qwen3MoeConfig(
        hidden_size=hidden,
        moe_intermediate_size=inter,
        num_experts=num_experts,
        num_experts_per_tok=2,
    )
    config._experts_implementation = "batched_mm"
    torch.manual_seed(0)
    experts = Qwen3MoeExperts(config)
    with torch.no_grad():
        experts.gate_up_proj.normal_(0, 0.05)
        experts.down_proj.normal_(0, 0.05)
    return experts.to(device=device, dtype=torch.bfloat16)


def test_table_rows_point_at_gate_up_down_slices() -> None:
    experts = _experts()
    table = ResidentExpertTable(experts)
    gate_up, down = experts.gate_up_proj, experts.down_proj
    inter = down.shape[2]
    row = table.table[3].tolist()
    assert row == [
        gate_up[3, :inter].data_ptr(),
        gate_up[3, inter:].data_ptr(),
        down[3].data_ptr(),
        3,
    ]
    assert table.intermediate == inter


def test_update_rewrites_one_row_in_place() -> None:
    experts = _experts()
    table = ResidentExpertTable(experts)
    storage = table.table.data_ptr()
    before = table.table.clone()
    inter, hidden = table.intermediate, experts.gate_up_proj.shape[2]
    fresh = [
        torch.zeros(s, dtype=torch.bfloat16)
        for s in ((inter, hidden), (inter, hidden), (hidden, inter))
    ]
    table.update(5, *fresh)
    assert table.table.data_ptr() == storage
    assert table.table[5].tolist() == [t.data_ptr() for t in fresh] + [5]
    others = [row for row in range(8) if row != 5]
    assert torch.equal(table.table[others], before[others])


def test_install_is_a_no_op_without_the_kernel(monkeypatch) -> None:
    monkeypatch.setattr(resident_experts, "grouped_decode_op", lambda: None)
    model = torch.nn.Sequential(_experts())
    assert install_grouped_resident_experts(model) == []


def _op_or_skip():
    op = resident_experts.grouped_decode_op()
    if op is None:
        pytest.skip("moe_infinity._store built without grouped_moe_decode")
    return op


def _routing(tokens, num_experts=8, top_k=2, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(tokens, num_experts, generator=gen, device="cuda")
    weights, index = torch.softmax(logits, dim=-1).topk(top_k, dim=-1)
    x = torch.randn(tokens, 256, generator=gen, device="cuda")
    return x.bfloat16(), index, weights.bfloat16()


def _close(actual, expected):
    scale = expected.float().abs().max().clamp_min(1e-6)
    err = (actual.float() - expected.float()).abs().max()
    assert err <= 4 * 2**-8 * scale, f"max |diff| {err} vs scale {scale}"


@requires_cuda
@pytest.mark.parametrize("tokens", [1, 3, 16])
def test_grouped_forward_matches_hf_experts(tokens: int) -> None:
    _op_or_skip()
    experts = _experts(device="cuda")
    x, index, weights = _routing(tokens)
    with torch.no_grad():
        expected = experts(x, index, weights)
        model = torch.nn.Sequential(experts)
        assert len(install_grouped_resident_experts(model)) == 1
        actual = experts(x, index, weights)
    _close(actual, expected)


@requires_cuda
def test_residency_change_updates_graph_without_recapture() -> None:
    _op_or_skip()
    experts = _experts(device="cuda")
    reference = _experts(device="cuda")  # same seed, plain HF forward
    tables = install_grouped_resident_experts(torch.nn.Sequential(experts))
    table = tables[0]
    x, index, weights = _routing(2, seed=3)
    static = types.SimpleNamespace(out=None)

    def step():
        static.out = experts(x, index, weights)

    with torch.no_grad():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            step()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        graph.replay()
        before = static.out.clone()

        # "Re-fetch" one routed expert into new device memory.
        moved = int(index[0, 0])
        inter = table.intermediate
        gate = (experts.gate_up_proj[moved, :inter] * 2).contiguous()
        up = experts.gate_up_proj[moved, inter:].clone()
        down = experts.down_proj[moved].clone()
        # Only the table changes; the module's own parameters do not.
        table.update(moved, gate, up, down)
        graph.replay()
        reference.gate_up_proj[moved, :inter].copy_(gate)
        expected = reference(x, index, weights)

    assert not torch.equal(static.out, before)
    _close(static.out, expected)
