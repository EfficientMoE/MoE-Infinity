# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from moe_infinity.models.deepseek_v4 import (
    SyncDeepSeekV4MoEBlock,
    sqrtsoftplus,
)


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


def _load_gate(indexer, layer, name):
    from moe_infinity.models.deepseek_v4.expert_bundle import TensorRef

    key = f"layers.{layer}.ffn.gate.{name}"
    shard = indexer._weight_map[key]
    meta = indexer._shard_header(shard)[key]
    dtype = indexer._ST_DTYPE_MAP[meta["dtype"]]
    ref = TensorRef(
        key=key, dtype=dtype, shape=tuple(meta["shape"]), shard=shard
    )
    return indexer.load_tensor(ref)


def test_sqrtsoftplus_matches_definition():
    x = torch.randn(100)
    assert torch.allclose(sqrtsoftplus(x), F.softplus(x).sqrt())


def test_dense_layer_topk_matches_reference(indexer, device):
    cfg = _v4_config()
    block = SyncDeepSeekV4MoEBlock(cfg, layer_idx=5).to(device)
    gate_w = _load_gate(indexer, 5, "weight").to(device)
    gate_b = _load_gate(indexer, 5, "bias").to(device)
    block.gate_weight.data = gate_w.to(block.gate_weight.dtype)
    block.gate_bias.data = gate_b.float()

    torch.manual_seed(0)
    x = torch.randn(8, cfg.hidden_size, dtype=torch.bfloat16, device=device)
    weights, indices = block.compute_routing(x)

    scores = sqrtsoftplus(
        F.linear(x.reshape(-1, cfg.hidden_size).float(), gate_w.float())
    )
    ref_idx = torch.topk(
        scores + gate_b.float().unsqueeze(0),
        cfg.num_experts_per_tok,
        dim=-1,
        sorted=False,
    ).indices
    ref_w = scores.gather(1, ref_idx)
    ref_w = ref_w / (ref_w.sum(dim=-1, keepdim=True) + 1e-20)
    ref_w = ref_w * cfg.routed_scaling_factor

    assert torch.equal(indices.sort(dim=-1).values, ref_idx.sort(dim=-1).values)
    assert torch.allclose(
        weights.gather(1, indices.argsort(dim=-1)),
        ref_w.gather(1, ref_idx.argsort(dim=-1)),
        rtol=1e-4,
        atol=1e-4,
    )


def test_hash_layer_uses_tid2eid(indexer, device):
    cfg = _v4_config()
    block = SyncDeepSeekV4MoEBlock(cfg, layer_idx=0).to(device)
    assert block.is_hash
    gate_w = _load_gate(indexer, 0, "weight").to(device)
    tid2eid = _load_gate(indexer, 0, "tid2eid").to(device)
    block.gate_weight.data = gate_w.to(block.gate_weight.dtype)
    block.tid2eid.data = tid2eid.long()

    torch.manual_seed(1)
    input_ids = torch.randint(0, cfg.vocab_size, (8,), device=device)
    x = torch.randn(8, cfg.hidden_size, dtype=torch.bfloat16, device=device)
    weights, indices = block.compute_routing(x, input_ids=input_ids)

    expected_idx = tid2eid[input_ids].long()
    assert torch.equal(indices, expected_idx)
    assert indices.shape == (8, cfg.num_experts_per_tok)
    assert torch.allclose(
        weights.sum(dim=-1),
        torch.full((8,), cfg.routed_scaling_factor, device=device),
        rtol=1e-3,
        atol=1e-3,
    )


def test_routing_masks_match_indices(indexer, device):
    cfg = _v4_config()
    block = SyncDeepSeekV4MoEBlock(cfg, layer_idx=5).to(device)
    gate_w = _load_gate(indexer, 5, "weight").to(device)
    gate_b = _load_gate(indexer, 5, "bias").to(device)
    block.gate_weight.data = gate_w.to(block.gate_weight.dtype)
    block.gate_bias.data = gate_b.float()

    torch.manual_seed(2)
    x = torch.randn(4, cfg.hidden_size, dtype=torch.bfloat16, device=device)
    weights, indices = block.compute_routing(x)
    router_mask, routing_weight = block.compute_routing_masks(x)

    assert router_mask.shape == (4, cfg.n_routed_experts)
    assert router_mask.sum().item() == 4 * cfg.num_experts_per_tok
    for tok in range(4):
        for e in indices[tok].tolist():
            assert router_mask[tok, e]
