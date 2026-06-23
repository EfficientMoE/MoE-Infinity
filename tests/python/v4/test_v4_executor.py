# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from types import SimpleNamespace

import pytest
import torch

from moe_infinity.models.deepseek_v4 import (
    DeepSeekV4PythonExpertExecutor,
    SyncDeepSeekV4MoEBlock,
    fp4_expert_forward,
    make_indexer_bundle_provider,
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


def _reference_loop(
    provider, layer_id, hidden, router_mask, routing_weight, swiglu
):
    num_tokens, hidden_dim = hidden.shape
    out = torch.zeros(
        num_tokens, hidden_dim, dtype=torch.float32, device=hidden.device
    )
    num_experts = router_mask.shape[-1]
    for e in range(num_experts):
        mask = router_mask[:, e]
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        w1, s1, w2, s2, w3, s3 = provider(layer_id, e)
        for t in idx.tolist():
            xi = hidden[t : t + 1]
            yi = fp4_expert_forward(
                xi, w1, s1, w2, s2, w3, s3, swiglu_limit=swiglu
            )
            out[t] += yi.float().squeeze(0) * routing_weight[t, e].float()
    return out.to(hidden.dtype)


def test_executor_matches_per_token_reference(indexer, device):
    if device == "cpu":
        pytest.skip("no free CUDA device")
    cfg = _v4_config()
    layer = 5
    block = SyncDeepSeekV4MoEBlock(cfg, layer_idx=layer).to(device)
    block.gate_weight.data = (
        _load_gate(indexer, layer, "weight")
        .to(device)
        .to(block.gate_weight.dtype)
    )
    block.gate_bias.data = _load_gate(indexer, layer, "bias").to(device).float()

    torch.manual_seed(0)
    x = torch.randn(16, cfg.hidden_size, dtype=torch.bfloat16, device=device)
    router_mask, routing_weight = block.compute_routing_masks(x)

    provider = make_indexer_bundle_provider(indexer, torch.device(device))
    executor = DeepSeekV4PythonExpertExecutor(provider, swiglu_limit=10.0)

    out = executor.execute(layer, x, router_mask, routing_weight)
    ref = _reference_loop(provider, layer, x, router_mask, routing_weight, 10.0)

    assert out.shape == (16, cfg.hidden_size)
    assert torch.isfinite(out).all()
    assert torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_executor_active_experts_subset(indexer, device):
    if device == "cpu":
        pytest.skip("no free CUDA device")
    cfg = _v4_config()
    provider = make_indexer_bundle_provider(indexer, torch.device(device))
    executor = DeepSeekV4PythonExpertExecutor(provider, swiglu_limit=10.0)

    num_tokens = 4
    router_mask = torch.zeros(
        num_tokens, cfg.n_routed_experts, dtype=torch.bool, device=device
    )
    chosen = [3, 7, 42]
    router_mask[:, chosen] = True
    active = executor.active_experts(router_mask)
    assert active == chosen


def test_executor_grouped_faster_than_per_token(indexer, device):
    if device == "cpu":
        pytest.skip("no free CUDA device")
    cfg = _v4_config()
    layer = 5
    block = SyncDeepSeekV4MoEBlock(cfg, layer_idx=layer).to(device)
    block.gate_weight.data = (
        _load_gate(indexer, layer, "weight")
        .to(device)
        .to(block.gate_weight.dtype)
    )
    block.gate_bias.data = _load_gate(indexer, layer, "bias").to(device).float()

    torch.manual_seed(0)
    x = torch.randn(64, cfg.hidden_size, dtype=torch.bfloat16, device=device)
    router_mask, routing_weight = block.compute_routing_masks(x)
    provider = make_indexer_bundle_provider(indexer, torch.device(device))
    executor = DeepSeekV4PythonExpertExecutor(provider, swiglu_limit=10.0)

    for _ in range(2):
        executor.execute(layer, x, router_mask, routing_weight)
    torch.cuda.synchronize(device)

    import time

    t0 = time.perf_counter()
    for _ in range(5):
        executor.execute(layer, x, router_mask, routing_weight)
    torch.cuda.synchronize(device)
    grouped = (time.perf_counter() - t0) / 5

    t0 = time.perf_counter()
    _reference_loop(provider, layer, x, router_mask, routing_weight, 10.0)
    torch.cuda.synchronize(device)
    per_token = time.perf_counter() - t0

    assert grouped < per_token
