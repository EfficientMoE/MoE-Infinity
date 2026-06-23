# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from types import SimpleNamespace

import pytest
import torch

from moe_infinity.models.deepseek_v4 import (
    DeepSeekV4PythonExpertExecutor,
    HostOffloadBundleProvider,
    SyncDeepSeekV4MoEBlock,
    make_indexer_bundle_provider,
)
from moe_infinity.models.deepseek_v4.expert_bundle import TensorRef

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="host offload streaming requires CUDA"
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


def _load(indexer, key):
    shard = indexer._weight_map[key]
    meta = indexer._shard_header(shard)[key]
    dtype = indexer._ST_DTYPE_MAP[meta["dtype"]]
    return indexer.load_tensor(
        TensorRef(key, dtype, tuple(meta["shape"]), shard)
    )


def test_streamed_tensors_match_source_bytes(indexer, device):
    if device == "cpu":
        pytest.skip("no free CUDA device")
    provider = HostOffloadBundleProvider(
        indexer, torch.device(device), max_resident_experts=4
    )
    bundle = indexer.bundle(0, 5)
    source = indexer.load_bundle_tensors(bundle)
    streamed = provider(0, 5)
    for src, got in zip(source, streamed):
        assert got.device.type == "cuda"
        assert got.dtype == src.dtype
        assert torch.equal(got.cpu().view(torch.uint8), src.view(torch.uint8))


def test_residency_bounded_by_cache_limit(indexer, device):
    if device == "cpu":
        pytest.skip("no free CUDA device")
    provider = HostOffloadBundleProvider(
        indexer, torch.device(device), max_resident_experts=4
    )
    for e in range(10):
        provider(0, e)
    resident = provider.resident_experts()
    assert len(resident) == 4
    assert resident == [(0, e) for e in range(6, 10)]


def test_lru_reuse_keeps_hot_expert(indexer, device):
    if device == "cpu":
        pytest.skip("no free CUDA device")
    provider = HostOffloadBundleProvider(
        indexer, torch.device(device), max_resident_experts=3
    )
    provider(0, 0)
    provider(0, 1)
    provider(0, 2)
    provider(0, 0)
    provider(0, 3)
    assert provider.is_resident(0, 0)
    assert not provider.is_resident(0, 1)


def test_executor_with_host_offload_matches_in_memory(indexer, device):
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

    torch.manual_seed(0)
    x = torch.randn(16, cfg.hidden_size, dtype=torch.bfloat16, device=dev)
    router_mask, routing_weight = block.compute_routing_masks(x)

    mem_provider = make_indexer_bundle_provider(indexer, dev)
    host_provider = HostOffloadBundleProvider(
        indexer, dev, max_resident_experts=4
    )

    mem_exec = DeepSeekV4PythonExpertExecutor(mem_provider, swiglu_limit=10.0)
    host_exec = DeepSeekV4PythonExpertExecutor(host_provider, swiglu_limit=10.0)

    mem_out = mem_exec.execute(layer, x, router_mask, routing_weight)
    host_out = host_exec.execute(layer, x, router_mask, routing_weight)

    assert torch.equal(mem_out, host_out)
    assert len(host_provider.resident_experts()) <= 4
