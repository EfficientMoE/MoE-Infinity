# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="residency check requires CUDA"
)


def _bundle_nbytes(indexer, bundle):
    total = 0
    for ref in bundle.tensors:
        numel = 1
        for d in ref.shape:
            numel *= d
        total += numel * torch._utils._element_size(ref.dtype)
    return total


def test_streaming_subset_uses_per_expert_memory(indexer, device):
    if device == "cpu":
        pytest.skip("no free CUDA device for residency measurement")
    dev = torch.device(device)
    torch.cuda.synchronize(dev)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)

    bundles = indexer.bundles_for_layer(0)
    one_bundle_bytes = _bundle_nbytes(indexer, bundles[0])
    full_layer_bytes = one_bundle_bytes * len(bundles)

    selected = [0, 1, 2, 3]
    base = torch.cuda.memory_allocated(dev)
    resident = []
    for e in selected:
        tensors = [t.to(dev) for t in indexer.load_bundle_tensors(bundles[e])]
        resident.append(tensors)
    torch.cuda.synchronize(dev)
    used = torch.cuda.memory_allocated(dev) - base

    expected = len(selected) * one_bundle_bytes
    assert used <= 1.10 * expected, (
        f"streaming {len(selected)} experts used {used} bytes, "
        f">110% of expected {expected}"
    )
    assert used < 0.10 * full_layer_bytes, (
        f"streaming {len(selected)}/256 experts used {used} bytes, "
        f"not far below full-layer {full_layer_bytes}"
    )

    del resident
    torch.cuda.empty_cache()


def test_released_experts_free_memory(indexer, device):
    if device == "cpu":
        pytest.skip("no free CUDA device for residency measurement")
    dev = torch.device(device)
    torch.cuda.empty_cache()
    bundles = indexer.bundles_for_layer(0)

    base = torch.cuda.memory_allocated(dev)
    tensors = [t.to(dev) for t in indexer.load_bundle_tensors(bundles[0])]
    torch.cuda.synchronize(dev)
    after_load = torch.cuda.memory_allocated(dev)
    assert after_load > base

    del tensors
    torch.cuda.empty_cache()
    torch.cuda.synchronize(dev)
    after_free = torch.cuda.memory_allocated(dev)
    assert after_free <= base + 1024
