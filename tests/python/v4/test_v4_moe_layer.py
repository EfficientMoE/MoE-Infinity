# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from types import SimpleNamespace

import pytest
import torch

from moe_infinity.models.deepseek_v4 import (
    DeepSeekV4PythonExpertExecutor,
    SyncDeepSeekV4MoEBlock,
    dequant_fp8_blockwise,
    fp8_shared_expert_forward,
    make_indexer_bundle_provider,
)
from moe_infinity.models.deepseek_v4.expert_bundle import TensorRef


def _v4_config():
    return SimpleNamespace(
        num_experts_per_tok=6,
        n_routed_experts=256,
        hidden_size=4096,
        vocab_size=129280,
        routed_scaling_factor=1.5,
        norm_topk_prob=True,
        num_hash_layers=3,
    )


def _load(indexer, key):
    shard = indexer._weight_map[key]
    meta = indexer._shard_header(shard)[key]
    dtype = indexer._ST_DTYPE_MAP[meta["dtype"]]
    return indexer.load_tensor(
        TensorRef(key, dtype, tuple(meta["shape"]), shard)
    )


def _load_shared(indexer, layer, device):
    out = []
    for proj in ("w1", "w2", "w3"):
        out.append(
            _load(
                indexer, f"layers.{layer}.ffn.shared_experts.{proj}.weight"
            ).to(device)
        )
        out.append(
            _load(
                indexer, f"layers.{layer}.ffn.shared_experts.{proj}.scale"
            ).to(device)
        )
    return out


def test_fp8_blockwise_dequant_matches_manual(indexer, device):
    if device == "cpu":
        pytest.skip("no free CUDA device")
    w = _load(indexer, "layers.5.ffn.shared_experts.w1.weight").to(device)
    s = _load(indexer, "layers.5.ffn.shared_experts.w1.scale").to(device)
    dw = dequant_fp8_blockwise(w, s, torch.bfloat16)
    bs = 128
    for i, j in [(0, 0), (1, 2), (15, 31)]:
        block = (
            w[i * bs : (i + 1) * bs, j * bs : (j + 1) * bs].float()
            * s[i, j].float()
        )
        assert torch.equal(
            block, dw[i * bs : (i + 1) * bs, j * bs : (j + 1) * bs].float()
        )


def test_full_moe_layer_matches_reference(indexer, device):
    if device == "cpu":
        pytest.skip("no free CUDA device")
    cfg = _v4_config()
    layer = 5
    dev = torch.device(device)

    block = SyncDeepSeekV4MoEBlock(cfg, layer_idx=layer).to(dev)
    block.gate_weight.data = (
        _load(indexer, f"layers.{layer}.ffn.gate.weight")
        .to(dev)
        .to(block.gate_weight.dtype)
    )
    block.gate_bias.data = (
        _load(indexer, f"layers.{layer}.ffn.gate.bias").to(dev).float()
    )

    provider = make_indexer_bundle_provider(indexer, dev)
    executor = DeepSeekV4PythonExpertExecutor(provider, swiglu_limit=10.0)
    shared = _load_shared(indexer, layer, dev)

    torch.manual_seed(0)
    x = torch.randn(8, cfg.hidden_size, dtype=torch.bfloat16, device=dev)

    router_mask, routing_weight = block.compute_routing_masks(x)
    routed = executor.execute(layer, x, router_mask, routing_weight)
    shared_out = fp8_shared_expert_forward(x, *shared, swiglu_limit=10.0)
    full = routed + shared_out.float()

    weights, indices = block.compute_routing(x)
    ref = torch.zeros_like(x, dtype=torch.float32)
    from moe_infinity.models.deepseek_v4 import fp4_expert_forward

    for e in range(cfg.n_routed_experts):
        pos = indices == e
        if not pos.any():
            continue
        tok = torch.nonzero(pos.any(dim=1), as_tuple=False).flatten()
        w1, s1, w2, s2, w3, s3 = provider(layer, e)
        ey = fp4_expert_forward(
            x[tok], w1, s1, w2, s2, w3, s3, swiglu_limit=10.0
        )
        col = (indices[tok] == e).float()
        ew = (weights[tok] * col).sum(dim=1, keepdim=True)
        ref[tok] += ey.float() * ew.float()
    ref += shared_out.float()

    assert full.shape == (8, cfg.hidden_size)
    assert torch.isfinite(full).all()
    assert torch.allclose(full, ref, rtol=1e-2, atol=1e-2)
