# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import os

import pytest
import torch
import torch.nn.functional as F

from moe_infinity.models.deepseek_v4 import (
    dequant_fp4_e2m1,
    fp4_expert_forward,
)

BATCHGEN_ROOT = os.environ.get(
    "BATCHGEN_ROOT", "/mnt/raid0nvme0/leyang/batchgen"
)


def _load_batchgen_dequant():
    import sys

    if BATCHGEN_ROOT not in sys.path:
        sys.path.insert(0, BATCHGEN_ROOT)
    try:
        from batchgen_kernels.common.v4_fp4_dequant import (
            dequant_fp4_e2m1 as bg,
        )

        return bg
    except Exception:
        return None


def test_dequant_bit_matches_batchgen(indexer):
    bg = _load_batchgen_dequant()
    if bg is None:
        pytest.skip("batchgen reference dequant not importable")
    bundle = indexer.bundle(0, 0)
    w1, s1, _, _, w3, s3 = indexer.load_bundle_tensors(bundle)
    for w, s in ((w1, s1), (w3, s3)):
        mine = dequant_fp4_e2m1(w, s, torch.bfloat16)
        ref = bg(w, s, torch.bfloat16)
        assert torch.equal(mine, ref)


def test_expert_forward_matches_reference_dequant_path(indexer, device):
    bundle = indexer.bundle(0, 0)
    w1, s1, w2, s2, w3, s3 = [
        t.to(device) for t in indexer.load_bundle_tensors(bundle)
    ]
    torch.manual_seed(0)
    x = torch.randn(8, indexer.hidden_size, dtype=torch.bfloat16, device=device)

    out = fp4_expert_forward(
        x, w1, s1, w2, s2, w3, s3, swiglu_limit=indexer.swiglu_limit
    )

    dw1 = dequant_fp4_e2m1(w1, s1, torch.bfloat16)
    dw2 = dequant_fp4_e2m1(w2, s2, torch.bfloat16)
    dw3 = dequant_fp4_e2m1(w3, s3, torch.bfloat16)
    gate = torch.clamp(F.linear(x, dw1).float(), max=indexer.swiglu_limit)
    up = torch.clamp(
        F.linear(x, dw3).float(),
        min=-indexer.swiglu_limit,
        max=indexer.swiglu_limit,
    )
    activated = F.silu(gate) * up
    ref = F.linear(activated.to(torch.bfloat16), dw2)

    assert out.shape == (8, indexer.hidden_size)
    assert torch.isfinite(out).all()
    assert torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_routing_weight_scales_output(indexer, device):
    bundle = indexer.bundle(0, 0)
    w1, s1, w2, s2, w3, s3 = [
        t.to(device) for t in indexer.load_bundle_tensors(bundle)
    ]
    torch.manual_seed(1)
    x = torch.randn(4, indexer.hidden_size, dtype=torch.bfloat16, device=device)

    base = fp4_expert_forward(
        x, w1, s1, w2, s2, w3, s3, swiglu_limit=indexer.swiglu_limit
    )
    rw = torch.full((4, 1), 0.5, dtype=torch.float32, device=device)
    scaled = fp4_expert_forward(
        x,
        w1,
        s1,
        w2,
        s2,
        w3,
        s3,
        swiglu_limit=indexer.swiglu_limit,
        routing_weight=rw,
    )
    assert torch.allclose(
        scaled.float(), base.float() * 0.5, rtol=1e-2, atol=1e-2
    )
