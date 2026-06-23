# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import torch

EXPECTED_LAYER0 = {
    "w1.weight": (torch.int8, (2048, 2048)),
    "w1.scale": (torch.float8_e8m0fnu, (2048, 128)),
    "w2.weight": (torch.int8, (4096, 1024)),
    "w2.scale": (torch.float8_e8m0fnu, (4096, 64)),
    "w3.weight": (torch.int8, (2048, 2048)),
    "w3.scale": (torch.float8_e8m0fnu, (2048, 128)),
}


def test_layer0_has_256_bundles(indexer):
    bundles = indexer.bundles_for_layer(0)
    assert len(bundles) == indexer.n_routed_experts == 256


def test_bundle_has_six_tensors_with_expected_shapes(indexer):
    bundle = indexer.bundle(0, 0)
    assert bundle.num_tensors == 6
    for (proj, part), (exp_dtype, exp_shape) in zip(
        [
            ("w1", "weight"),
            ("w1", "scale"),
            ("w2", "weight"),
            ("w2", "scale"),
            ("w3", "weight"),
            ("w3", "scale"),
        ],
        [
            EXPECTED_LAYER0[f"{p}.{q}"]
            for p, q in [
                ("w1", "weight"),
                ("w1", "scale"),
                ("w2", "weight"),
                ("w2", "scale"),
                ("w3", "weight"),
                ("w3", "scale"),
            ]
        ],
    ):
        ref = bundle.part(proj, part)
        assert (
            ref.dtype == exp_dtype
        ), f"{proj}.{part}: {ref.dtype} != {exp_dtype}"
        assert (
            ref.shape == exp_shape
        ), f"{proj}.{part}: {ref.shape} != {exp_shape}"


def test_bundle_keys_follow_native_contract(indexer):
    bundle = indexer.bundle(3, 42)
    assert bundle.part("w1", "weight").key == (
        "layers.3.ffn.experts.42.w1.weight"
    )
    assert bundle.part("w2", "scale").key == (
        "layers.3.ffn.experts.42.w2.scale"
    )


def test_loaded_tensors_preserve_dtype(indexer):
    bundle = indexer.bundle(0, 0)
    tensors = indexer.load_bundle_tensors(bundle)
    assert len(tensors) == 6
    for ref, t in zip(bundle.tensors, tensors):
        assert t.dtype == ref.dtype
        assert tuple(t.shape) == ref.shape
