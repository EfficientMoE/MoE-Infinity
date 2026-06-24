import torch
import torch.nn.functional as F

from moe_infinity.runtime.model_offload import _remap_mixtral_v5_experts


def _v5_expert_forward(gate_up_e, down_e, x):
    gate, up = F.linear(x, gate_up_e).chunk(2, dim=-1)
    return F.linear(F.silu(gate) * up, down_e)


def _v4_expert_forward(w1, w2, w3, x):
    return F.linear(F.silu(F.linear(x, w1)) * F.linear(x, w3), w2)


def test_remap_expands_batched_keys_to_per_expert():
    num_experts, hidden, inter = 4, 8, 16
    prefix = "model.layers.0.block_sparse_moe.experts"
    state_dict = {
        f"{prefix}.gate_up_proj": torch.randn(num_experts, 2 * inter, hidden),
        f"{prefix}.down_proj": torch.randn(num_experts, hidden, inter),
        "model.layers.0.block_sparse_moe.gate.weight": torch.randn(
            num_experts, hidden
        ),
    }

    _remap_mixtral_v5_experts(state_dict)

    assert f"{prefix}.gate_up_proj" not in state_dict
    assert f"{prefix}.down_proj" not in state_dict
    for e in range(num_experts):
        assert state_dict[f"{prefix}.{e}.w1.weight"].shape == (inter, hidden)
        assert state_dict[f"{prefix}.{e}.w3.weight"].shape == (inter, hidden)
        assert state_dict[f"{prefix}.{e}.w2.weight"].shape == (hidden, inter)
    assert "model.layers.0.block_sparse_moe.gate.weight" in state_dict


def test_remap_is_numerically_equivalent_to_v5_forward():
    num_experts, hidden, inter = 3, 8, 16
    prefix = "model.layers.0.block_sparse_moe.experts"
    gate_up = torch.randn(num_experts, 2 * inter, hidden)
    down = torch.randn(num_experts, hidden, inter)
    x = torch.randn(5, hidden)

    state_dict = {
        f"{prefix}.gate_up_proj": gate_up.clone(),
        f"{prefix}.down_proj": down.clone(),
    }
    _remap_mixtral_v5_experts(state_dict)

    for e in range(num_experts):
        ref = _v5_expert_forward(gate_up[e], down[e], x)
        out = _v4_expert_forward(
            state_dict[f"{prefix}.{e}.w1.weight"],
            state_dict[f"{prefix}.{e}.w2.weight"],
            state_dict[f"{prefix}.{e}.w3.weight"],
            x,
        )
        assert torch.allclose(ref, out, atol=1e-6)


def test_remap_noop_on_v4_per_expert_keys():
    prefix = "model.layers.0.block_sparse_moe.experts"
    state_dict = {
        f"{prefix}.0.w1.weight": torch.randn(16, 8),
        f"{prefix}.0.w2.weight": torch.randn(8, 16),
        f"{prefix}.0.w3.weight": torch.randn(16, 8),
    }
    before = {k: v.clone() for k, v in state_dict.items()}

    _remap_mixtral_v5_experts(state_dict)

    assert set(state_dict.keys()) == set(before.keys())
    for k in before:
        assert torch.equal(state_dict[k], before[k])
