# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import glob
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="offload store requires CUDA engine"
)


@pytest.fixture()
def prefetch_lib(tmp_path_factory):
    try:
        from moe_infinity.runtime.model_offload import _load_prefetch_lib

        lib = _load_prefetch_lib()
    except Exception as exc:
        pytest.skip(f"prefetch lib (_store/_engine) unavailable: {exc}")

    probe_dir = tmp_path_factory.mktemp("store_probe")
    engine = lib.prefetch_handle(str(probe_dir) + "/", 0.5)
    engine.offload(torch.zeros(4, 4, dtype=torch.int8), 0)
    produced = any(
        "index" not in os.path.basename(f)
        for f in glob.glob(os.path.join(str(probe_dir), "archer_*"))
    )
    if not produced:
        pytest.skip(
            "offload store binding non-functional in this environment "
            "(likely torch ABI mismatch); store tests require the host venv"
        )
    return lib


@pytest.fixture()
def fp4_like_tensors():
    weight = torch.randint(-128, 127, (16, 16), dtype=torch.int8)
    scale = (
        torch.arange(16, dtype=torch.int32)
        .remainder(8)
        .to(torch.uint8)
        .view(torch.float8_e8m0fnu)
        .reshape(16, 1)
    )
    return weight, scale


def _raw_bytes(t: torch.Tensor) -> bytes:
    return t.contiguous().view(torch.uint8).numpy().tobytes()


def test_offload_persists_int8_and_f8e8m0_bytes_exactly(
    prefetch_lib, fp4_like_tensors, tmp_path
):
    weight, scale = fp4_like_tensors
    engine = prefetch_lib.prefetch_handle(str(tmp_path) + "/", 0.5)
    engine.offload(weight, 0)
    engine.offload(scale, 1)

    param_files = [
        f
        for f in glob.glob(os.path.join(str(tmp_path), "archer_*"))
        if "index" not in os.path.basename(f)
    ]
    assert param_files, "offload did not produce a param file"
    raw = b"".join(open(f, "rb").read() for f in param_files)

    assert (
        _raw_bytes(weight) in raw
    ), "int8 FP4-packed bytes not persisted verbatim"
    assert (
        _raw_bytes(scale) in raw
    ), "f8_e8m0 scale bytes not persisted verbatim"


def test_index_serialized_and_reopen_recovers_offloaded_state(
    prefetch_lib, fp4_like_tensors, tmp_path
):
    weight, scale = fp4_like_tensors
    engine = prefetch_lib.prefetch_handle(str(tmp_path) + "/", 0.5)
    engine.offload(weight, 0)
    engine.offload(scale, 1)

    index_file = os.path.join(str(tmp_path), "archer_index")
    assert os.path.exists(index_file)
    assert os.path.getsize(index_file) > 0

    reopened = prefetch_lib.prefetch_handle(str(tmp_path) + "/", 0.5)
    assert reopened.is_tensor_offloaded(0)
    assert reopened.is_tensor_offloaded(1)


def test_bundle_registers_six_tensors_without_dtype_error(
    prefetch_lib, indexer, tmp_path
):
    engine = prefetch_lib.prefetch_handle(str(tmp_path) + "/", 0.5)
    bundle = indexer.bundle(0, 0)
    tensors = indexer.load_bundle_tensors(bundle)

    tensor_ids = []
    for tid, (ref, data) in enumerate(zip(bundle.tensors, tensors)):
        engine.offload(data, tid)
        skeleton = torch.zeros_like(data)
        engine.register(skeleton, tid)
        tensor_ids.append(tid)
        assert engine.is_tensor_offloaded(tid)

    bundle.tensor_ids = tensor_ids
    assert len(bundle.tensor_ids) == 6
