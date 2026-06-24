from types import SimpleNamespace
from typing import cast

import torch
import torch.nn.functional as F
from transformers import PretrainedConfig

from moe_infinity.runtime.model_offload import _remap_v5_batched_experts
from moe_infinity.utils.hf_config import parse_expert_id


def _cfg(arch, **kw):
    return cast(
        PretrainedConfig,
        cast(object, SimpleNamespace(architectures=[arch], **kw)),
    )


def _v5_expert_forward(gate_up_e, down_e, x):
    gate, up = F.linear(x, gate_up_e).chunk(2, dim=-1)
    return F.linear(F.silu(gate) * up, down_e)


def _v4_expert_forward(gate_w, up_w, down_w, x):
    return F.linear(F.silu(F.linear(x, gate_w)) * F.linear(x, up_w), down_w)


def test_mixtral_remap_normalizes_block_and_uses_w_names():
    n, hidden, inter = 4, 8, 16
    v5 = "model.layers.0.mlp.experts"
    out = "model.layers.0.block_sparse_moe"
    sd = {
        f"{v5}.gate_up_proj": torch.randn(n, 2 * inter, hidden),
        f"{v5}.down_proj": torch.randn(n, hidden, inter),
        "model.layers.0.mlp.gate.weight": torch.randn(n, hidden),
    }

    _remap_v5_batched_experts(
        sd, _cfg("MixtralForCausalLM", num_hidden_layers=1, num_local_experts=n)
    )

    assert f"{v5}.gate_up_proj" not in sd
    assert "model.layers.0.mlp.gate.weight" not in sd
    assert f"{out}.gate.weight" in sd
    for e in range(n):
        assert sd[f"{out}.experts.{e}.w1.weight"].shape == (inter, hidden)
        assert sd[f"{out}.experts.{e}.w3.weight"].shape == (inter, hidden)
        assert sd[f"{out}.experts.{e}.w2.weight"].shape == (hidden, inter)


def test_qwen3_remap_keeps_mlp_and_uses_proj_names():
    n, hidden, inter = 4, 8, 16
    experts = "model.layers.0.mlp.experts"
    sd = {
        f"{experts}.gate_up_proj": torch.randn(n, 2 * inter, hidden),
        f"{experts}.down_proj": torch.randn(n, hidden, inter),
    }

    _remap_v5_batched_experts(
        sd, _cfg("Qwen3MoeForCausalLM", num_hidden_layers=1, num_experts=n)
    )

    assert f"{experts}.gate_up_proj" not in sd
    for e in range(n):
        assert sd[f"{experts}.{e}.gate_proj.weight"].shape == (inter, hidden)
        assert sd[f"{experts}.{e}.up_proj.weight"].shape == (inter, hidden)
        assert sd[f"{experts}.{e}.down_proj.weight"].shape == (hidden, inter)


def test_mixtral_remap_numerically_equivalent():
    n, hidden, inter = 3, 8, 16
    v5 = "model.layers.0.mlp.experts"
    out = "model.layers.0.block_sparse_moe.experts"
    gate_up = torch.randn(n, 2 * inter, hidden)
    down = torch.randn(n, hidden, inter)
    x = torch.randn(5, hidden)
    sd = {
        f"{v5}.gate_up_proj": gate_up.clone(),
        f"{v5}.down_proj": down.clone(),
    }

    _remap_v5_batched_experts(
        sd, _cfg("MixtralForCausalLM", num_hidden_layers=1, num_local_experts=n)
    )

    for e in range(n):
        ref = _v5_expert_forward(gate_up[e], down[e], x)
        out_v = _v4_expert_forward(
            sd[f"{out}.{e}.w1.weight"],
            sd[f"{out}.{e}.w3.weight"],
            sd[f"{out}.{e}.w2.weight"],
            x,
        )
        assert torch.allclose(ref, out_v, atol=1e-6)


def test_qwen3_remap_numerically_equivalent():
    n, hidden, inter = 3, 8, 16
    experts = "model.layers.0.mlp.experts"
    gate_up = torch.randn(n, 2 * inter, hidden)
    down = torch.randn(n, hidden, inter)
    x = torch.randn(5, hidden)
    sd = {
        f"{experts}.gate_up_proj": gate_up.clone(),
        f"{experts}.down_proj": down.clone(),
    }

    _remap_v5_batched_experts(
        sd, _cfg("Qwen3MoeForCausalLM", num_hidden_layers=1, num_experts=n)
    )

    for e in range(n):
        ref = _v5_expert_forward(gate_up[e], down[e], x)
        out_v = _v4_expert_forward(
            sd[f"{experts}.{e}.gate_proj.weight"],
            sd[f"{experts}.{e}.up_proj.weight"],
            sd[f"{experts}.{e}.down_proj.weight"],
            x,
        )
        assert torch.allclose(ref, out_v, atol=1e-6)


def test_remapped_keys_match_parse_expert_id_mixtral():
    n, hidden, inter = 4, 8, 16
    v5 = "model.layers.1.mlp.experts"
    sd = {
        f"{v5}.gate_up_proj": torch.randn(n, 2 * inter, hidden),
        f"{v5}.down_proj": torch.randn(n, hidden, inter),
    }
    cfg = _cfg("MixtralForCausalLM", num_hidden_layers=2, num_local_experts=n)
    _remap_v5_batched_experts(sd, cfg)

    for e in range(n):
        key = f"model.layers.1.block_sparse_moe.experts.{e}.w1.weight"
        assert parse_expert_id(key, cfg) == (1, e)


def test_remapped_keys_match_parse_expert_id_qwen3():
    n, hidden, inter = 4, 8, 16
    experts = "model.layers.1.mlp.experts"
    sd = {
        f"{experts}.gate_up_proj": torch.randn(n, 2 * inter, hidden),
        f"{experts}.down_proj": torch.randn(n, hidden, inter),
    }
    cfg = _cfg("Qwen3MoeForCausalLM", num_hidden_layers=2, num_experts=n)
    _remap_v5_batched_experts(sd, cfg)

    for e in range(n):
        key = f"model.layers.1.mlp.experts.{e}.gate_proj.weight"
        assert parse_expert_id(key, cfg) == (1, e)


def test_remap_skips_gpt_oss():
    n, hidden, inter = 4, 8, 16
    experts = "model.layers.0.mlp.experts"
    sd = {
        f"{experts}.gate_up_proj": torch.randn(n, 2 * inter, hidden),
        f"{experts}.down_proj": torch.randn(n, hidden, inter),
    }
    before = set(sd.keys())

    _remap_v5_batched_experts(
        sd, _cfg("GptOssForCausalLM", num_hidden_layers=1, num_local_experts=n)
    )

    assert set(sd.keys()) == before


def test_remap_noop_on_v4_per_expert_keys():
    prefix = "model.layers.0.block_sparse_moe.experts"
    sd = {
        f"{prefix}.0.w1.weight": torch.randn(16, 8),
        f"{prefix}.0.w2.weight": torch.randn(8, 16),
        f"{prefix}.0.w3.weight": torch.randn(16, 8),
    }
    before = {k: v.clone() for k, v in sd.items()}

    _remap_v5_batched_experts(
        sd, _cfg("MixtralForCausalLM", num_hidden_layers=1, num_local_experts=1)
    )

    assert set(sd.keys()) == set(before.keys())
    for k in before:
        assert torch.equal(sd[k], before[k])
